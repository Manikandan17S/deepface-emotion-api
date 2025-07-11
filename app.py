from flask import Flask, request, jsonify
from deepface import DeepFace
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "🎯 DeepFace Emotion API is Live!"

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        data = request.get_json()
        img_data = data['image']

        # Convert base64 to image
        img_bytes = base64.b64decode(img_data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze emotion
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]["dominant_emotion"]
        emotions = result[0]["emotion"]

        return jsonify({
            "dominant_emotion": emotion,
            "emotions": emotions
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
