import streamlit as st
from dotenv import load_dotenv
import tempfile
from pdf_utils import extract_text_from_pdf
from rag_pipeline import build_qa_pipeline

load_dotenv()

st.set_page_config(page_title="Chat with Your Notes", layout="wide")
st.title("Chat with Your Notes – PDF Q&A Bot")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("PDF uploaded successfully.")
    with st.spinner("Extracting text from PDF..."):
        document_text = extract_text_from_pdf(tmp_path)

    with st.spinner("Setting up the Q&A system..."):
        qa = build_qa_pipeline(document_text)

    question = st.text_input("Ask a question based on your PDF:")

    if question:
        with st.spinner("Generating answer..."):
            result = qa.run(question)
            st.markdown(f"**Answer:** {result}")