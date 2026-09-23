import os
import tempfile
import streamlit as st
from PIL import Image
import pdfplumber
import pytesseract
import torch

st.set_page_config(
    page_title="Document Q&A",
    page_icon="❓",
    layout="wide"
)

# ----------------- Model Loading with Caching ----------------- #

@st.cache_resource(show_spinner="Loading Question Answering model (RoBERTa)...")
def load_qa_model():
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        device=device
    )
    return qa_pipeline

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def extract_text_from_image(image_path):
    try:
        return pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception as e:
        return f"Error running OCR: {e}."

# ----------------- UI Layout ----------------- #

st.title("❓ Document Question & Answering")
st.markdown("Ask precise questions about your document using a fine-tuned RoBERTa QA model.")

# Check if document already loaded from home page
existing_text = st.session_state.get("shared_document_text", None)
existing_name = st.session_state.get("shared_document_name", None)

document_text = None

if existing_text:
    use_existing = st.checkbox(
        f"Use document already uploaded from home page (`{existing_name}`)",
        value=True
    )
    if use_existing:
        document_text = existing_text

if not document_text:
    uploaded_file = st.file_uploader(
        "Upload a document for Q&A",
        type=["pdf", "txt", "jpg", "jpeg", "png"],
        key="qa_uploader"
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type
        file_name = uploaded_file.name

        with st.spinner("Extracting content..."):
            if file_type == "application/pdf" or file_name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                document_text = extract_text_from_pdf(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            elif file_type == "text/plain" or file_name.endswith(".txt"):
                document_text = uploaded_file.read().decode("utf-8", errors="ignore")

            elif file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                document_text = extract_text_from_image(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

if document_text:
    st.success("Document context is loaded and ready for questions.")

    col_ctx, col_qa = st.columns([1, 1])

    with col_ctx:
        st.subheader("📑 Reference Context")
        st.text_area(
            "Context Preview",
            value=document_text,
            height=350,
            label_visibility="collapsed"
        )

    with col_qa:
        st.subheader("💬 Ask Questions")
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. What is the main finding of the report?"
        )

        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Please type a question first.")
            else:
                with st.spinner("Searching document for answer..."):
                    try:
                        qa_model = load_qa_model()
                        # Limit context window to first 4000 characters if too long
                        result = qa_model(question=question, context=document_text[:4000])

                        score = round(result.get("score", 0.0) * 100, 1)
                        answer = result.get("answer", "No answer found.")

                        st.markdown("#### Answer:")
                        st.info(f"**{answer}**")
                        st.caption(f"Model Confidence: **{score}%**")

                    except Exception as err:
                        st.error(f"Error answering question: {err}")
else:
    st.info("👆 Please upload a document above or on the home page to start asking questions.")
