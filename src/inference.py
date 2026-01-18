import pickle
from src.text_preprocessing import preprocess

import PyPDF2
from PyPDF2.errors import PdfReadError
from pdf2image import convert_from_bytes
import pytesseract

from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Tesseract path (Windows)
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------------
# Load ML artifacts
# -------------------------------
model = pickle.load(open("models/resume_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))


# -------------------------------
# PDF TEXT EXTRACTION (TEXT + OCR)
# -------------------------------
def extract_text_from_pdf(file_stream):
    """
    1. Try normal text extraction
    2. If it fails, fallback to OCR
    """

    # ---- Attempt 1: Text-based PDF ----
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        if text.strip():
            return text

    except PdfReadError:
        pass
    except Exception:
        pass

    # ---- Attempt 2: OCR for scanned PDFs ----
    try:
        images = convert_from_bytes(file_stream.read())
        ocr_text = ""

        for img in images:
            ocr_text += pytesseract.image_to_string(img)

        if ocr_text.strip():
            return ocr_text
        else:
            raise ValueError("OCR failed: no readable text found")

    except Exception:
        raise ValueError("Unable to extract text from the uploaded PDF")


# -------------------------------
# PREDICTION LOGIC (ATS-CORRECT)
# -------------------------------
def predict_resume(resume_text, jd_text):
    """
    Returns:
    - decision: SHORTLIST / REJECT
    - match_percentage: ATS-style similarity score
    """

    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(jd_text)

    # ---------- ATS Match Percentage ----------
    corpus = [processed_jd, processed_resume]
    vectors = vectorizer.transform(corpus)

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    )[0][0]

    match_percentage = round(similarity * 100, 2)

    # ---------- HARD THRESHOLD (50%) ----------
    if match_percentage < 50:
        return "REJECT ❌", match_percentage

    # ---------- ML Classifier ----------
    combined = processed_jd + " " + processed_resume
    combined_vector = vectorizer.transform([combined])
    prediction = model.predict(combined_vector)[0]

    decision = "SHORTLIST ✅" if prediction == 1 else "REJECT ❌"

    return decision, match_percentage


# -------------------------------
# LOCAL TEST
# -------------------------------
if __name__ == "__main__":

    resume = """I have experience in Python, machine learning,
                Flask, and deploying ML models."""
    jd = """Looking for a Machine Learning Engineer with Python and NLP."""

    decision, score = predict_resume(resume, jd)
    print("Decision:", decision)
    print("Match Confidence:", score, "%")
