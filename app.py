import streamlit as st
from utils.pdf_reader import load_pdfs
from ingest import create_vectorstore
from qa_chain import get_internal_answer ,get_external_answer, get_qa_chain
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
        with st.spinner("Getting  answer..."):

            # Get answers from both internal and external sources
            local_result = get_internal_answer(question)
            external_result = get_external_answer(question)

            #Display results
        st.markdown("### 📄 Answer from Your Notes:")
        st.write(local_result["answer"] if "answer" in local_result else local_result)

        st.markdown("### 🌍 Answer from External Sources:")
        st.write(external_result)

            

