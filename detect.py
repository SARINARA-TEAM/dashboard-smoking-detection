import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model Keras
MODEL_PATH = 'mobilenetv2_binary_classifier8.h5'

HAAR_CASCADE_PATH = 'haarcascade_frontalface_alt2.xml'

# Label class
CLASS_NAMES = ['not_smoking', 'smoking']

try:
    model = load_model(MODEL_PATH)
    body_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if body_cascade.empty():
        raise ValueError("Gagal memuat file haarcascade_frontalface_alt2.xml")
except Exception as e:
    raise ValueError(f"Gagal memuat model: {e}")

def process_frame(frame):
    # Salin frame untuk pengolahan
    output_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi tubuh
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    label = None

    for (x, y, w, h) in bodies:
        # Crop ROI tubuh
        roi = frame[y:y+h, x:x+w]

        # Resize sesuai input model
        input_size = (224, 224)
        resized_roi = cv2.resize(roi, input_size)
        normalized_roi = resized_roi / 255.0
        input_data = np.expand_dims(normalized_roi, axis=0).astype(np.float32)

        prediction = model.predict(input_data, verbose=0)
        confidence = prediction[0][0]
        predicted_class = int(confidence >= 0.7)
        label = CLASS_NAMES[predicted_class]

        # Warna kotak
        color = (0, 0, 255) if label == "smoking" else (0, 255, 0)

        # Gambar kotak dan teks
        cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output_frame, label