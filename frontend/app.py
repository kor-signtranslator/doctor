from flask import Flask, render_template, request, jsonify
from flask_compress import Compress

app = Flask(__name__)

Compress(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/camera')
def camera():
    return render_template("camera.html")


@app.route('/stt', methods=['POST'])
def receive_speech():
    data = request.get_json()
    text = data.get("text")
    print("받은 음성 텍스트:", text)
    # 여기서 필요한 처리 가능 (예: DB 저장, TTS 변환 등)
    return jsonify({"received": text})

if __name__ == '__main__':
    app.run(debug=True)

    
