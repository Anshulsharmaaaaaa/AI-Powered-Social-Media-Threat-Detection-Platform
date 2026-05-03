from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline("sentiment-analysis")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = classifier(text)

    return jsonify({
        "text": text,
        "prediction": result[0]
    })

if __name__ == "__main__":
    app.run(port=5000)
