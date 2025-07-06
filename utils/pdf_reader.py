import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(files):
    all_docs = []
    for file in files:
        # Save the uploaded file temporarily
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()
        all_docs.extend(docs)

        os.remove(file.name)  # Clean up
    return all_docs