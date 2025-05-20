from flask import Flask, render_template, jsonify, Response
from chatbot_engine import get_answer
import csv
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
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"http://192.168.18.48:81/stream", cv2.CAP_FFMPEG)  # URL kamera ESP32-CAM http://192.168.1.1:81/stream

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka!")
        return
    
    # tts_playing = False
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Gagal baca frame dari kamera")
            break
        
        detected_frame, label = process_frame(frame)
        latest_status["label"] = label
        
        current_time = time.time()
        if label == "smoking" and (current_time - last_tts_time) > 10:
            # if not tts_playing:
            #     speak("smoking detected, don't smoking in this area!")
            #     tts_playing = True
            # last_tts_time = current_time
            threading.Thread(target=speak, args=("smoking detected, don't smoking in this area!",), daemon=True).start()
            last_tts_time = current_time
        else:
            # tts_playing = False
            pass

        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# def load_qa_data(filename):
#     qa_pairs = {}
#     with open(filename, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             qa_pairs[row['pertanyaan'].lower()] = row['jawaban']
#     return qa_pairs
# qa_data = load_qa_data('static/assets/cleaned_data_qna.csv')

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

@app.route('/statistik')
def statistik():
    return render_template('statistik.html')  

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')  # Your about page

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

    # message = message.lower()
    # response = qa_data.get(message, "Maaf, saya tidak mengerti pertanyaan itu. Silakan coba tanyakan yang lain!")
    # return jsonify(response=response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)