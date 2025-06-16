import os
import time
import threading
import logging
import numpy as np
import cv2  # OpenCV
import torch
import mediapipe as mp
from collections import deque
from torch import nn
import requests

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# === 로깅 설정 ===
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
DEBUG_MODE = False  # 디버깅이 필요하면 True로 설정

# === FastAPI 앱 및 CORS 설정 ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# === 템플릿 및 정적 파일 설정 ===
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
static_path = os.path.join(BASE_DIR, "static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=static_path), name="static")

##############################################
# 데이터 전처리 및 키포인트 처리 함수들
##############################################
def check_static_hand_shape(kp_seq):
    return True

def normalize_and_pad(kp_seq, sequence_length):
    if kp_seq.shape[0] == 0:
        kp_seq = np.zeros((sequence_length, 42, 2), dtype=np.float32)
    normalized = []
    for frame in kp_seq:
        left_wrist = frame[0]
        right_wrist = frame[21]
        frame_norm = frame.copy()
        frame_norm[:21] -= left_wrist
        frame_norm[21:] -= right_wrist
        normalized.append(frame_norm)
    normalized = np.array(normalized)
    logger.info(f"normalize_and_pad: shape {normalized.shape}")
    if normalized.shape[0] < sequence_length:
        pad = np.repeat(normalized[-1][np.newaxis, :], sequence_length - normalized.shape[0], axis=0)
        normalized = np.concatenate([normalized, pad], axis=0)
    else:
        idxs = np.linspace(0, normalized.shape[0]-1, sequence_length).astype(int)
        normalized = normalized[idxs]
    return normalized

def filter_static_hand(sequence, movement_threshold=0.01):
    diff = np.diff(sequence, axis=0)
    left_diff = np.linalg.norm(diff[:,0:21,:], axis=-1)
    right_diff = np.linalg.norm(diff[:,21:42,:], axis=-1)
    left_mean = np.mean(left_diff)
    right_mean = np.mean(right_diff)
    logger.info(f"filter_static_hand: left_mean={left_mean:.4f}, right_mean={right_mean:.4f}")
    if left_mean < movement_threshold and right_mean < movement_threshold:
        return None
    if left_mean < right_mean:
        sequence[:,0:21,:] = 0
    else:
        sequence[:,21:42,:] = 0
    return sequence

def compute_movement(kp_seq):
    diff = np.diff(kp_seq, axis=0)
    movement = np.linalg.norm(diff, axis=-1)
    return np.mean(movement)

def smooth_keypoints_ema(kp_seq, alpha=0.95):
    T, _, _ = kp_seq.shape
    smoothed = np.copy(kp_seq)
    for t in range(1, T):
        smoothed[t] = alpha * kp_seq[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

def process_frame(frame, results, last_left, last_right, no_hand_printed, fill_missing=True):
    if results.multi_hand_landmarks is None:
        return None, last_left, last_right, True
    combined = np.zeros((42, 2), dtype=np.float32)
    left_pts, right_pts = None, None
    for i in range(len(results.multi_hand_landmarks)):
        label = results.multi_handedness[i].classification[0].label
        landmarks = results.multi_hand_landmarks[i].landmark
        if len(landmarks) < 21:
            if label == "Left":
                pts = last_left if (fill_missing and last_left is not None) else np.zeros((21,2), dtype=np.float32)
            elif label == "Right":
                pts = last_right if (fill_missing and last_right is not None) else np.zeros((21,2), dtype=np.float32)
            else:
                pts = np.zeros((21,2), dtype=np.float32)
        else:
            pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
            pts = np.clip(pts, 0.0, 1.0)
        if label == "Left":
            left_pts = pts.copy()
        elif label == "Right":
            right_pts = pts.copy()
    if left_pts is None:
        left_pts = (last_left if (fill_missing and last_left is not None) else np.zeros((21,2), dtype=np.float32))
    if right_pts is None:
        right_pts = (last_right if (fill_missing and last_right is not None) else np.zeros((21,2), dtype=np.float32))
    combined[0:21] = left_pts
    combined[21:42] = right_pts
    if fill_missing:
        last_left = left_pts
        last_right = right_pts
    return combined, last_left, last_right, no_hand_printed

def process_keypoints(kp_buffer, seq_len, model, device, 
                      confidence_threshold=0.85,  # 기존 0.60 또는 0.80에서 0.85로 인상
                      ambiguous_margin=0.35,       # 기존 0.25 또는 0.30에서 0.35로 인상
                      end_of_gesture=False):
    kp_seq = np.array(kp_buffer)
    kp_seq = normalize_and_pad(kp_seq, seq_len)
    filtered = filter_static_hand(kp_seq, movement_threshold=0.01)
    if filtered is None:
        filtered = kp_seq
    smoothed = smooth_keypoints_ema(filtered, alpha=0.95)
    
    # (예: 추가적인 움직임 검증 또는 전처리 옵션이 있을 경우 여기에 삽입)
    
    if not check_static_hand_shape(smoothed):
        return None, None
    x_tensor = torch.tensor(smoothed, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_prob, pred = torch.max(probs, dim=1)
        top_prob_val = top_prob.item()
        pred_class = pred.item()
        logger.info(f"Model output: top_prob={top_prob_val:.2f}, diff={(torch.topk(probs,2).values[0][0] - torch.topk(probs,2).values[0][1]).item():.2f}")
        if top_prob_val < confidence_threshold:
            return None, None
        diff_prob = (torch.topk(probs,2).values[0][0] - torch.topk(probs,2).values[0][1]).item()
        if diff_prob < ambiguous_margin:
            return None, None
        return label_map[pred_class], top_prob_val


##############################################
# 모델 정의 및 초기화
##############################################
class CNN_LSTMModel(nn.Module):
    def __init__(self, cnn_out_dim, lstm_hidden_size, num_classes, dropout=0.5):
        super(CNN_LSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(32 * 42 * 2, cnn_out_dim),
            nn.BatchNorm1d(cnn_out_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=cnn_out_dim,
                            hidden_size=lstm_hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(B * T, -1)
        features = self.cnn_fc(cnn_out)
        features = features.view(B, T, -1)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out

def convert_to_sentence(words: str) -> str:
    if not words.strip():
        return ""
    # 프롬프트 수정:
    # 예를 들어 "가슴 아프다"라는 입력이 주어지면, 조사(예: "이")를 넣어 "가슴이 아프다"와 같이 자연스러운 문장을 만들어 달라는 요청
    prompt_text = (
      "다음에 나열된 단어들을 모두 포함하여, 단 하나의 자연스럽고 문법적으로 맞는 문장을 만들어줘. "
      "단어들 사이에는 필요한 조사, 어미, 연결어 등을 자유롭게 넣어도 되지만, 나열된 단어 외의 새로운 단어는 절대 추가하지 마. "
      "절대 추가되지 않은 명사, 동사, 형용사 등을 사용하지 마."
      "존댓말로 바꿔줘"
      "예를 들어, 입력 단어가 '학교 가다'일 경우, 결과는 '학교에 간다'처럼 만들어줘.\n"
      f"단어 목록: {words}\n"
      "자연스러운 문장:"
    )
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            GEMINI_API_URL, json=payload, headers=headers, timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Gemini API response: {data}")
            # 응답 구조는 API 문서에 따라 달라질 수 있기 때문에, candidate의 구조가 올바른지 확인합니다.
            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts:
                        sentence = parts[0].get("text", "").strip()
                        if sentence:
                            return sentence
                else:
                    logger.error("Gemini API response candidate에 content/parts 키가 없음.")
            else:
                logger.error("Gemini API response에 candidates가 없음.")
            return "문장 생성 실패."
        else:
            logger.error(f"Gemini API 호출 실패, status code: {response.status_code}, response: {response.text}")
            return "문장 생성 실패."
    except Exception as e:
        logger.error(f"Gemini API 호출 오류: {e}")
        return "문장 생성 실패."



def remove_adjacent_duplicates(sentence: str) -> str:
    words = sentence.split()
    if not words:
        return sentence
    result = [words[0]]
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    return " ".join(result)

####################################################
# 글로벌 설정 및 모델 초기화
####################################################
GEMINI_API_KEY = "AIzaSyCPi0gNCtfMFc2xVpsfHLyGFbWVVQoX8MU"  # 실제 API 키로 대체
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
label_map = {
    0:'3일전', 1: '가슴', 2: '감사합니다', 3: '구토', 4: '귀', 5: '기침',
    6: '다리', 7: '맞다', 8: '머리', 9: '목', 10: '무릎',
    11: '발', 12: '발목', 13: '복부', 14: '손목뼈', 15:'시작', 16: '아니',
    17: '아프다', 18: '안녕하세요', 19: '어깨', 20: '어지럽다',
    21: '열', 22: '팔', 23: '팔꿈치', 24: '허리'
}
num_classes = len(label_map)
cnn_out_dim = 256
lstm_hidden_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model_cnn_lstm.pth"
model = CNN_LSTMModel(cnn_out_dim, lstm_hidden_size, num_classes, dropout=0.5).to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {model_path}")
else:
    logger.error("모델 파일이 존재하지 않습니다.")
example_input = torch.randn(2, 64, 1, 42, 2).to(device)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.eval()

####################################################
# VideoCamera 클래스 (손이 1.5초 이상 미검출되면 flush하여 예측 수행)
####################################################
class VideoCamera:
    def __init__(self):
        # 외부 USB 카메라 사용: 인덱스를 1로 변경 (필요시 인덱스 조정)
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            logger.error("카메라 오픈 실패. 장치 번호와 권한을 확인하세요!")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.4
        )
        self.keypoints_buffer = deque()
        self.SEQUENCE_LENGTH = 40
        self.last_left = None
        self.last_right = None
        self.hand_in_frame = False
        self.last_hand_time = None
        self.predicted_words = []  # 최종 예측 단어 저장
        self.logged_buffer_insufficient = False  # 로그 플래그 초기화
        
        threading.Thread(target=self.update, daemon=True).start()

    def reset_gesture_detection(self):
        self.last_hand_time = None
        with self.lock:
            self.keypoints_buffer.clear()
        self.hand_in_frame = False
        logger.info("Gesture detection reset.")

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else np.zeros((480,640,3), dtype=np.uint8)

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("프레임 읽기 실패")
                continue
            display_frame = frame.copy()
            proc_frame = cv2.resize(frame, (320,240))
            frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                self.hand_in_frame = True
                self.last_hand_time = time.time()
                # 손이 감지되면 로그 플래그 초기화
                self.logged_buffer_insufficient = False
                logger.info("손 검출됨")
                kps, self.last_left, self.last_right, _ = process_frame(
                    proc_frame, results, self.last_left, self.last_right, False, fill_missing=True
                )
                if kps is not None:
                    with self.lock:
                        self.keypoints_buffer.append(kps)
            else:
                self.hand_in_frame = False
                # 손 검출 없이 즉시 flush를 실행하되, 키포인트 버퍼 길이 부족 로그는 한 번만 남김
                with self.lock:
                    buffer_length = len(self.keypoints_buffer)
                logger.info(f"Flush triggered immediately as hand is not detected, buffer length = {buffer_length}")
                if buffer_length >= 10:
                    prediction, conf = process_keypoints(
                        self.keypoints_buffer, self.SEQUENCE_LENGTH, scripted_model, device,
                        confidence_threshold=0.80, ambiguous_margin=0.30, end_of_gesture=True
                    )
                    if prediction is not None and conf is not None:
                        with self.lock:
                            self.predicted_words.append(prediction)
                        logger.info(f"Predicted word appended: {prediction}, Confidence: {conf:.2f}")
                    else:
                        logger.info("Flush triggered but no valid prediction")
                else:
                    if not self.logged_buffer_insufficient:
                        logger.info("Flush triggered but 키포인트 버퍼 길이 부족")
                        self.logged_buffer_insufficient = True
                self.reset_gesture_detection()

            # (여기서 디스플레이 처리 코드 유지)
            curr_kps, _, _, _ = process_frame(
                proc_frame, results, self.last_left, self.last_right, False, fill_missing=False
            )
            disp_kps = curr_kps.copy() if curr_kps is not None else np.zeros((42,2))
            h, w, _ = display_frame.shape
            for idx, (x, y) in enumerate(disp_kps):
                if x == 0 and y == 0:
                    continue
                cx, cy = int(x*w), int(y*h)
                cv2.circle(display_frame, (cx, cy), 3, (0,255,0), -1)
            with self.lock:
                text = " ".join(self.predicted_words)
            cv2.putText(display_frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            with self.lock:
                self.latest_frame = display_frame.copy()
            time.sleep(0.01)
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


##############################################
# FastAPI 엔드포인트
##############################################
@app.on_event("startup")
async def startup_event():
    global video_camera
    video_camera = VideoCamera()
    logger.info("VideoCamera started.")

@app.on_event("shutdown")
async def shutdown_event():
    global video_camera
    if video_camera is not None:
        video_camera.running = False
        logger.info("VideoCamera stopped.")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/camera", response_class=HTMLResponse)
async def camera(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = video_camera.get_frame()
            if frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("jpeg 인코딩 실패")
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.01)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/keywords")
def get_keywords(response: Response):
    with video_camera.lock:
        if video_camera.predicted_words:
            words = " ".join(video_camera.predicted_words)
            return {"keywords": words}
        else:
            response.status_code = 204  # No Content
            return {}

@app.get("/gen_sentence")
def gen_sentence(word: str = None):
    with video_camera.lock:
        words_str = " ".join(video_camera.predicted_words)
    if word and word.strip():
        words_str = word.strip()
    if not words_str:
        return {"sentence": "입력된 단어가 없습니다."}
    sentence = convert_to_sentence(words_str)
    processed_sentence = remove_adjacent_duplicates(sentence)
    if not processed_sentence:
        processed_sentence = "문장 생성 실패."
    with video_camera.lock:
        video_camera.predicted_words.clear()
    return {"sentence": processed_sentence}

@app.get("/clear_words")
def clear_words():
    with video_camera.lock:
        video_camera.predicted_words.clear()
    return {"result": "Predicted words cleared."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
