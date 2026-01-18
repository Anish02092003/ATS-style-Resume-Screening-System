import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_preprocessing import clean_text


# ==============================
# LOAD TEXT FILE
# ==============================
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ==============================
# MAIN PIPELINE
# ==============================
def compute_resume_scores(resume_dir, jd_path):

    # Load and clean JD
    jd_text = clean_text(load_text(jd_path))

    resume_texts = []
    resume_names = []

    # Load and clean resumes
    for file in os.listdir(resume_dir):
        if file.endswith(".txt"):
            text = load_text(os.path.join(resume_dir, file))
            resume_texts.append(clean_text(text))
            resume_names.append(file)

    # Combine JD + resumes
    corpus = [jd_text] + resume_texts

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Cosine similarity
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Create results
    results = list(zip(resume_names, similarities))
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results


# ==============================
# RUN SCRIPT
# ==============================
if __name__ == "__main__":
    results = compute_resume_scores(
        resume_dir="data/resumes",
        jd_path="data/job_descriptions/jd.txt"
    )

    print("\nResume Ranking:\n")
    for name, score in results:
        print(f"{name} → Match Score: {round(score * 100, 2)}%")
