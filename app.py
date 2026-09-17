"""Small Flask web interface for spam detection."""

from pathlib import Path

from flask import Flask, render_template, request

from src.model import predict_email

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    email_text = ""
    if request.method == "POST":
        email_text = request.form.get("email_text", "")
        try:
            result = predict_email(MODEL_PATH, email_text)
        except (ValueError, FileNotFoundError) as exc:
            error = str(exc)
    return render_template("index.html", result=result, error=error, email_text=email_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
