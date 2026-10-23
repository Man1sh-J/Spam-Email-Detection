
from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

# Load our trained ML model
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            # Convert text into numerical features
            message_tfidf = vectorizer.transform([message])

            # Predict spam or ham
            result = model.predict(message_tfidf)[0]

            if result == 1:
                prediction = "Spam Message!"
            else:
                prediction = "Not Spam Message!"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
