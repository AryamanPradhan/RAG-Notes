import streamlit as st
from utils.pdf_reader import load_pdfs
from ingest import create_vectorstore
from qa_chain import get_qa_chain
import os

st.set_page_config(page_title="Chat with PDF 📚")
st.title("📄 Upload your PDFs and ask questions!")

# Upload files
files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if files:
    st.info("Processing documents...")
    documents = load_pdfs(files)
    create_vectorstore(documents)
    st.success("PDFs processed and embedded!")

    qa_chain = get_qa_chain()

    question = st.text_input("Ask a question based on the uploaded PDFs:")
    if question:
        result = qa_chain.invoke({"input": question})
        st.markdown("### Answer:")
        st.write(result["answer"])
