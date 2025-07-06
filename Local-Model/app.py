import streamlit as st
import tempfile
import os
import pdfplumber
import re
from pipeline import build_or_load_retriever, call_rag

# Loading the retriever once (on app start)
retriever = build_or_load_retriever()
rag_chain = call_rag(retriever)

# PDF text extractor
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    full_text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    full_text = re.sub(r"_+", "", full_text)
    os.remove(tmp_path)
    return full_text.strip()

# Streamlit App
st.set_page_config(page_title="Insurance Proposal Verifier", layout="centered")
st.title("📄 Proposal Form Validator")
st.write("Upload a life insurance proposal form (PDF), and we'll verify its format and completeness.")

uploaded_file = st.file_uploader("Upload your proposal form PDF", type=["pdf"])

if uploaded_file is not None:
    st.info("🔍 Processing your file...")
    extracted_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Running verification..."):
        response = rag_chain.invoke({"document": extracted_text})

    st.success("✅ Verification Completed!")
    st.markdown("### Result:")
    st.markdown(f"```text\n{response}\n```")

