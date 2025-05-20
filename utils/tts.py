# import pyttsx3
# import threading
# import time

# engine = pyttsx3.init()
# speak_lock = threading.Lock()  # Untuk mencegah akses paralel
# last_speak_time = 0  # Untuk tracking interval minimal antara dua ucapan

# def speak(text):
#     def run():
#         global last_speak_time
#         current_time = time.time()

#         # Jeda minimal antara dua ucapan
#         if current_time - last_speak_time < 5:
#             return

#         with speak_lock:
#             try:
#                 engine.say(text)
#                 engine.runAndWait()
#                 last_speak_time = current_time
#             except RuntimeError as e:
#                 if "run loop already started" in str(e):
#                     print("[TTS] Run loop sudah aktif. Lewati...")

#     threading.Thread(target=run, daemon=True).start()

import pyttsx3
import threading
import time

last_speak_time = 0
lock = threading.Lock()

def speak(text):
    global last_speak_time
    current_time = time.time()
    
    if current_time - last_speak_time < 10:
        return

    def run():
        engine = pyttsx3.init()  # Inisialisasi ulang tiap panggilan
        with lock:
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                engine.stop()
        
    threading.Thread(target=run, daemon=True).start()
    last_speak_time = current_time