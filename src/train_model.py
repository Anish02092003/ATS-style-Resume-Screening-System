import os
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from text_preprocessing import preprocess


# ============================
# LOAD TEXT FILE
# ============================
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================
# DATA PREPARATION
# ============================
def prepare_data(resume_dir, jd_path):

    texts = []
    labels = []

    # Load and preprocess JD
    jd_text = preprocess(load_text(jd_path))

    resume_texts = []

    # Load resumes
    for file in os.listdir(resume_dir):
        if file.endswith(".txt"):
            resume_text = preprocess(load_text(os.path.join(resume_dir, file)))
            resume_texts.append(resume_text)

    # Build corpus for similarity scoring
    corpus = [jd_text] + resume_texts
    temp_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = temp_vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Generate labels using similarity threshold
    for resume_text, sim_score in zip(resume_texts, similarities):
        combined = jd_text + " " + resume_text
        texts.append(combined)

        # REAL ATS-style labeling
        labels.append(1 if sim_score >= 0.3 else 0)

    return texts, labels


# ============================
# TRAIN MODEL
# ============================
def train():

    texts, labels = prepare_data(
        resume_dir="data/resumes",
        jd_path="data/job_descriptions/jd.txt"
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Debug: label distribution
    print("Label distribution:", np.bincount(y))

    # 🚨 IMPORTANT:
    # No train-test split for tiny datasets
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save artifacts
    with open("models/resume_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nModel and vectorizer saved successfully 🚀")


if __name__ == "__main__":
    train()
