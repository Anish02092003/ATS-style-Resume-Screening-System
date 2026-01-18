import re
import string

SKILLS = [
    "python", "machine learning", "deep learning", "nlp",
    "scikit-learn", "flask", "pandas", "numpy",
    "sql", "data analysis"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def boost_skills(text):
    """
    Repeat skill keywords to increase importance
    """
    boosted_text = text
    for skill in SKILLS:
        if skill in text:
            boosted_text += f" {skill} {skill}"
    return boosted_text


def preprocess(text):
    text = clean_text(text)
    text = boost_skills(text)
    return text
