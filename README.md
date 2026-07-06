# 🏥 ClaimSense AI
### Agentic Clinical Decision Intelligence for Healthcare Payment Integrity

**Cotiviti Intern Assessment — Topic 2: Clinical Decision Making & Pattern Recognition**
**Submitted by:** Rohini Mallikarjunaiah | Stevens Institute of Technology | M.S. Business Intelligence & Analytics

---

## 📹 Video Walkthrough
[▶ Watch Demo Video](https://drive.google.com/file/d/1VJhBjJYWqACRSfQbfrT2u70I_goavy6H/view?usp=sharing)

---

## 🚀 What It Does
ClaimSense AI is a working agentic healthcare claims intelligence platform that:
- Detects anomalous claims using **Isolation Forest** (unsupervised)
- Scores fraud risk per claim using **XGBoost** (0–100%)
- Explains suspicious claims in plain clinical language using **LLaMA-3.3-70B via Groq**

## 🛠 Tech Stack
Python · Streamlit · scikit-learn · XGBoost · Groq API · LLaMA-3.3-70B

## 📂 Repository Contents
- `app.py` — Full working Streamlit application
- `Cotiviti_Report_Rohini.docx` — Written report (2 pages + bibliography)
- `ClaimSense_Presentation.pptx` — PowerPoint presentation (10 slides)

## ▶ How to Run
```bash
conda activate claimsense
streamlit run app.py
```
