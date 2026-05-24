from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")

    if len(text.strip()) < 50:
        return jsonify({"error": "Please enter longer text"}), 400

    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

if __name__ == "__main__":
    app.run(debug=True)
