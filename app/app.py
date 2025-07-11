from flask import Flask, request, jsonify
from deepface import DeepFace
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        image = request.files["image"]
        result = DeepFace.analyze(image.read(), actions=["emotion"], enforce_detection=False)
        return jsonify({"emotion": result[0]["dominant_emotion"], "details": result[0]["emotion"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "DeepFace Emotion API Running!"

if __name__ == "__main__":
    app.run(debug=True)
