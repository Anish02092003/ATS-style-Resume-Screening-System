from flask import Flask, render_template, request
import io

from src.inference import predict_resume, extract_text_from_pdf

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    decision = None
    score = None
    error = None

    if request.method == "POST":
        try:
            jd_text = request.form["jd"]
            resume_file = request.files["resume"]

            # Read file ONCE
            resume_bytes = resume_file.read()

            # PDF handling (text + OCR fallback handled inside)
            if resume_file.filename.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(
                    io.BytesIO(resume_bytes)
                )
            else:
                resume_text = resume_bytes.decode("utf-8")

            decision, score = predict_resume(resume_text, jd_text)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        decision=decision,
        score=score,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

