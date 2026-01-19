# 🧠 Resume Screening AI (ATS-Style Machine Learning System)

## 📌 Overview
Recruiters often receive hundreds of resumes for a single job opening, making manual screening inefficient and error-prone.  
This project implements an **ATS-style Resume Screening System** that automatically evaluates resumes against a given Job Description (JD) using **NLP, Machine Learning, and OCR**.

The system computes an interpretable **match confidence score**, applies a **similarity threshold**, and makes a **shortlist/reject decision** using a trained ML classifier.

---

## 🎯 Problem Statement
Manual resume screening is:
- Time-consuming
- Subjective
- Difficult to scale

The objective of this project is to **automate resume shortlisting** by evaluating textual relevance between resumes and job descriptions while supporting real-world resume formats such as **PDFs (including scanned resumes)**.

---

## 🧠 Solution Approach

The system uses a **two-layer decision pipeline**, similar to real Applicant Tracking Systems (ATS):

### 1️⃣ Resume–JD Similarity (NLP Layer)
- Text preprocessing
- TF-IDF vectorization
- Cosine similarity
- Generates an interpretable **Match Confidence (%)**

### 2️⃣ Shortlisting Decision (ML Layer)
- Logistic Regression classifier
- Uses similarity-filtered data
- Final decision: **SHORTLIST / REJECT**

A **50% similarity threshold** is applied to filter irrelevant resumes before classification.

---

## 📂 Project Structure
resume-screening/
│── app.py
│── requirements.txt
│── models/
│ ├── resume_model.pkl
│ ├── tfidf_vectorizer.pkl
│── src/
│ ├── text_preprocessing.py
│ ├── resume_pipeline.py
│ ├── train_model.py
│ ├── inference.py
│── templates/
│ └── index.html


---

## 📄 Resume Format Support
- ✅ Text-based PDF resumes
- ✅ **Scanned PDF resumes (OCR using Tesseract)**
- ✅ TXT resumes
- ❌ Image-only resumes without readable text (future improvement)

---

## ⚙️ Technologies Used
- **Python**
- **scikit-learn**
- **Natural Language Processing (TF-IDF, Cosine Similarity)**
- **Flask**
- **PyPDF2**
- **Tesseract OCR**
- **HTML/CSS**

---

## 🚀 Web Application Features
- Upload resume (PDF / TXT)
- Paste job description
- ATS-style match confidence score
- Shortlist / Reject decision
- Graceful error handling for corrupted or scanned PDFs
   ## Deployed Here
  ---https://ats-style-resume-screening-system.onrender.com

---

## 📊 Decision Logic
If similarity < 50% → Reject
Else → ML Classifier → Shortlist / Reject

Decision: SHORTLIST ✅
Match Confidence: 78.4%


---

## 🧠 Key Learnings
- Building interpretable NLP-based ranking systems
- Handling real-world PDF parsing challenges
- OCR integration for scanned documents
- Separating similarity scoring from classification logic
- Designing ML systems with business constraints

---

## 🔮 Future Improvements
- Skill-level weighting (experience-based)
- Section-wise resume parsing (skills, education, projects)
- Explainable feedback (missing skills)
- Multi-role support (Data Analyst, Backend Engineer, etc.)
- Cloud OCR integration for higher accuracy

---

## 👨‍💻 Author
**Pritish Kumar Lenka**  
Electronics & Communication Engineering  
Machine Learning | Applied AI | NLP
