import os
import tempfile
import string
import streamlit as st
from PIL import Image
import pdfplumber
import pytesseract
import torch

# Ensure NLTK data if needed
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

st.set_page_config(
    page_title="DocuSummarize & Translate",
    page_icon="📚",
    layout="wide"
)

# ----------------- Helper Functions ----------------- #

def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    try:
        return pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception as e:
        return f"Error running OCR: {e}. Please ensure tesseract-ocr is installed."

def translate_to_hindi(text):
    """Translate English text to Hindi with reliable fallback."""
    # First attempt: deep-translator (robust and rate-limit tolerant)
    try:
        from deep_translator import GoogleTranslator
        # deep-translator handles chunks up to 5000 chars
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated = [GoogleTranslator(source='en', target='hi').translate(c) for c in chunks]
        return "\n".join(translated)
    except Exception:
        pass

    # Fallback attempt: googletrans
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text[:4000], src='en', dest='hi')
        return result.text
    except Exception as e:
        return f"Translation error: {e}"

# ----------------- Model Loading with Caching ----------------- #

@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer(model_choice: str):
    """Load and cache the summarization pipeline."""
    from transformers import pipeline

    has_cuda = torch.cuda.is_available()

    if model_choice == "Phi-3-mini (GPU Required)" and has_cuda:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            device_map="cuda",
            quantization_config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        return {
            "type": "phi3",
            "pipeline": pipeline("text-generation", model=model, tokenizer=tokenizer)
        }
    else:
        # Default lightweight transformer for Cloud / CPU deployment
        device = 0 if has_cuda else -1
        pipe = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=device
        )
        return {
            "type": "bart",
            "pipeline": pipe
        }

def run_summarization(summarizer_obj, text: str, max_len: int = 150, min_len: int = 40):
    """Run summarization on the extracted text."""
    pipe = summarizer_obj["pipeline"]
    model_type = summarizer_obj["type"]

    if model_type == "phi3":
        messages = [
            {"role": "system", "content": "You are an expert AI summarizer. Provide a concise, clear, and informative summary of the document."},
            {"role": "user", "content": f"Please summarize the following document:\n\n{text}"}
        ]
        args = {
            "max_new_tokens": max_len,
            "return_full_text": False,
            "temperature": 0.2,
            "do_sample": False,
        }
        res = pipe(messages, **args)
        return res[0]["generated_text"]
    else:
        # Chunk text if too long for Bart's 1024 token limit
        max_chunk_chars = 3000
        text_to_summarize = text[:max_chunk_chars]
        res = pipe(text_to_summarize, max_length=max_len, min_length=min_len, do_sample=False)
        return res[0]["summary_text"]

# ----------------- UI Layout ----------------- #

st.title("📚 Intelligent Document Summarizer")
st.markdown("Upload documents (PDF, TXT, or Image) to automatically extract text, generate AI summaries, and translate to Hindi.")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    has_cuda = torch.cuda.is_available()
    device_label = "🟢 GPU (CUDA)" if has_cuda else "🟠 CPU (Streamlit Cloud)"
    st.caption(f"Detected Runtime: **{device_label}**")

    model_options = ["DistilBART (Cloud & CPU Fast)"]
    if has_cuda:
        model_options.append("Phi-3-mini (GPU Required)")
    model_choice = st.selectbox("Summarization Model", options=model_options)

    max_summary_length = st.slider("Max Summary Length (words)", min_value=50, max_value=300, value=150, step=10)
    st.markdown("---")
    st.markdown("Navigate to **Q&A** in the left sidebar to ask questions directly about your document.")

# Main File Upload Area
uploaded_file = st.file_uploader(
    "Choose a file to analyze",
    type=["pdf", "txt", "jpg", "jpeg", "png"],
    help="Supports PDF files, raw text files, and images (scanned text or photos)."
)

document_text = None

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    with st.spinner("Extracting content from uploaded file..."):
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
    word_count = len(document_text.split())
    st.success(f"Successfully extracted text ({word_count:,} words) from `{uploaded_file.name}`")

    # Store text in session state for cross-page persistence
    st.session_state["shared_document_text"] = document_text
    st.session_state["shared_document_name"] = uploaded_file.name

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Extracted Document Text")
        st.text_area(
            label="Extracted Text Preview",
            value=document_text,
            height=380,
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("✨ Actions & Results")
        tab_summary, tab_translate = st.tabs(["📝 Summary", "🌐 Hindi Translation"])

        with tab_summary:
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Generating summary with AI..."):
                    try:
                        summarizer_obj = load_summarizer(model_choice)
                        summary = run_summarization(summarizer_obj, document_text, max_len=max_summary_length)
                        st.session_state["last_summary"] = summary
                    except Exception as err:
                        st.error(f"Error during summarization: {err}")

            if "last_summary" in st.session_state:
                summary_text = st.session_state["last_summary"]
                st.markdown("#### Summary Result")
                st.info(summary_text)
                sum_words = len(summary_text.split())
                st.caption(f"Summary: {sum_words} words (Reduced by {max(0, round((1 - sum_words / max(1, word_count)) * 100))}%).")

                st.download_button(
                    label="📥 Download Summary",
                    data=summary_text,
                    file_name=f"summary_{uploaded_file.name}.txt",
                    mime="text/plain"
                )

        with tab_translate:
            if st.button("🔄 Translate Document to Hindi", use_container_width=True):
                with st.spinner("Translating text to Hindi..."):
                    hindi_text = translate_to_hindi(document_text)
                    st.session_state["last_translation"] = hindi_text

            if "last_translation" in st.session_state:
                st.markdown("#### Hindi Translation")
                st.text_area(
                    label="Translated text",
                    value=st.session_state["last_translation"],
                    height=250,
                    label_visibility="collapsed"
                )
                st.download_button(
                    label="📥 Download Hindi Translation",
                    data=st.session_state["last_translation"],
                    file_name=f"hindi_translation_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
else:
    st.info("👆 Upload a document above to get started. You can also switch to the **Q&A** page in the sidebar.")
