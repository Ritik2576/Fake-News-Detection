from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load(r"ByteVerse 2025\Fake News Detector\fake_news_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Loads HTML from /templates

@app.route("/predict", methods=["POST"])

def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is empty"}), 400

    prediction = model.predict([text])[0]
    return jsonify({"result": "Real News" if prediction == 1 else "Fake News"})

if __name__ == "__main__":
    app.run(debug=True)