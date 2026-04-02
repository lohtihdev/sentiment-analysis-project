"""
app.py
------
Flask web app to showcase the Sentiment Analysis project.
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

import sys
# Force UTF-8 on Windows (cp1252 can't encode emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory
from preprocessor import preprocess

os.makedirs("outputs", exist_ok=True)

app = Flask(__name__)

# ── Load saved model ──────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("outputs", "sentiment_model.pkl")
pipeline   = joblib.load(MODEL_PATH)

ICONS  = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}
EMOJIS = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
COLORS = {"Positive": "#2DD4BF", "Negative": "#F87171", "Neutral": "#A78BFA"}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty input"}), 400

    processed = preprocess(text)
    pred      = pipeline.predict([processed])[0]
    proba     = pipeline.predict_proba([processed])[0]
    classes   = pipeline.classes_.tolist()

    scores = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}

    return jsonify({
        "sentiment":  pred,
        "confidence": scores[pred],
        "scores":     scores,
        "icon":       EMOJIS[pred],
        "color":      COLORS[pred],
    })


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory("outputs", filename)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  >>> Sentiment Analysis Demo <<<")
    print("  Open:  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)
