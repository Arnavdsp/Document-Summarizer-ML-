# Multi-Format Document Summarizer & Q&A

An end-to-end multi-page Streamlit application that extracts text from multiple document formats (**PDF**, **TXT**, and **Images/Scans via OCR**), generates concise summaries using state-of-the-art transformer models, translates documents into Hindi, and provides question answering over document context.

---

## Features

- **Multi-Format Ingestion**: Upload PDFs (`pdfplumber`), plain text files, or image scans (`pytesseract` OCR).
- **Intelligent Summarization**:
  - Uses `sshleifer/distilbart-cnn-12-6` for fast, lightweight inference on CPU / Streamlit Cloud.
  - Supports `microsoft/Phi-3-mini-128k-instruct` in 4-bit precision when a GPU (CUDA) is detected.
- **Hindi Translation**: Fast, reliable English-to-Hindi translation using `deep-translator` with automatic fallback to `googletrans`.
- **Document Q&A Page**: Dedicated multi-page interface powered by `deepset/roberta-base-squad2` to ask natural language questions directly against your uploaded document.
- **Session Continuity**: Upload a document on the home page, and its text seamlessly persists across the app to the Q&A page.
- **Model Caching**: Uses `@st.cache_resource` for zero-redundancy model loading across user interactions.

---

##  Deploying to Streamlit Community Cloud

1. Fork or push this repository to your GitHub account (`Arnavdsp/Document-Summarizer-ML-`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app**:
   - **Repository:** `Arnavdsp/Document-Summarizer-ML-`
   - **Branch:** `main`
   - **Main file path:** `App.py`
4. Click **Deploy!**

> **Note on OCR:** Streamlit Cloud automatically uses [`packages.txt`](packages.txt) to install `tesseract-ocr` system packages for image scanning.

---

##  Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Arnavdsp/Document-Summarizer-ML-.git
cd Document-Summarizer-ML-

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional for OCR) Install Tesseract OCR:
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# 4. Run the Streamlit app
streamlit run App.py
```

---

## Contributors

- Jeel Savsani
- Prakrut Moon
- Yanshik Jada
- Yash Dodiya
- Arnav Deshpande

##  Demo & Resources
- **Demo Video**: [Google Drive Link](https://drive.google.com/file/d/1HtBseEKWRSSkWoOF55zxgdNTP_911rYC/view?usp=sharing)
- **Project Report**: [Google Docs Link](https://docs.google.com/document/d/1wgHxrRYpgwQ7mu4dB4HXAwyy4bE5EsPIEki4E_haft0/edit?usp=sharing)
