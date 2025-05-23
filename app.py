from flask import Flask, render_template, jsonify, Response, request
from chatbot_engine import get_answer
import cv2
from detect import process_frame
from utils.tts import speak
import time
import threading


app = Flask(__name__)

# Status global
latest_status = {"label": None}
last_tts_time = 0

def generate_frames():
    global latest_status, last_tts_time
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r"http://192.168.1.7:81/stream", cv2.CAP_FFMPEG)  # URL kamera ESP32-CAM http://192.168.1.1:81/stream

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka!")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Gagal baca frame dari kamera")
            break
        
        detected_frame, label = process_frame(frame)
        latest_status["label"] = label
        
        current_time = time.time()
        if label == "smoking" and (current_time - last_tts_time) > 10:
            threading.Thread(target=speak, args=("smoking detected, don't smoking in this area!",)).start()
            last_tts_time = current_time
        else:
            pass

        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def detection_status():
    return jsonify(latest_status)

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

@app.route('/speak', methods=['POST'])
def trigger_speak():
    data = request.get_json()
    text = data.get('text', '')
    if text.strip() == '':
        return jsonify({'error': 'no text provided'}), 400
    threading.Thread(target=speak, args=(text,)).start()
    return jsonify({'status': 'Speaking Started'})

@app.route('/statistik')
def statistik():
    return render_template('statistik.html')  

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

@app.route('/detail_zat/<int:zat_id>')
def detail_zat(zat_id):
    details = {
        0: {"name": "Nikotin", "description": "Nikotin is a highly addictive substance found in cigarettes."},
        1: {"name": "Tar", "description": "Tar is a carcinogen found in cigarette smoke."},
        2: {"name": "Carbon Monoxide", "description": "A poisonous gas that can affect the lungs and heart."},
        3: {"name": "Formaldehyde", "description": "Formaldehyde is a toxic chemical used in preservatives."},
    }
    zat = details.get(zat_id, {"name": "Unknown", "description": "No details available."})
    return render_template('detail_zat.html', zat=zat)

@app.route('/get_response/<message>', methods=['GET'])
def get_response(message):
    answer = get_answer(message)
    return jsonify(response=answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)