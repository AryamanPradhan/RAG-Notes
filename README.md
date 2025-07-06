# RAG-Notes-Reader
#  Chat with Your PDFs – Secure RAG-Based Question Answering App

This project is an interactive **Retrieval-Augmented Generation (RAG)** app that allows users to upload PDF documents (e.g., class notes, books, reports) and ask natural language questions based on their content.

It uses a secure local stack with **FAISS vector search**, **HuggingFace embeddings**, and **Groq's Mixtral LLM**, and features a clean **Streamlit UI**. Unlike typical RAG apps, this one **avoids insecure pickle deserialization** by using safe FAISS index loading.

---

##  Features

-  Upload and process one or more PDF files
-  Semantic chunking and indexing with HuggingFace embeddings
-  Fast, vector-based retrieval using FAISS
-  Answer questions using Groq’s Mixtral LLM
-  Secure FAISS loading — no `allow_dangerous_deserialization`
-  Streamlit-powered web interface

---

##  Tech Stack

| Component      | Tool/Library                          |
|----------------|---------------------------------------|
| UI             | Streamlit                             |
| Embeddings     | HuggingFace (`all-mpnet-base-v2`)     |
| Vector Store   | FAISS (secure, local)                 |
| LLM            | Groq (Mixtral 8x7B via LangChain)     |
| Backend Logic  | LangChain v0.2+                       |
| PDF Loader     | PyPDFLoader (LangChain Community)     |
| Env Manager    | Conda                                 |

---

##  Installation

1. Clone the repository
2. Create virtual Environment
     conda create -n rag_pdf_env python=3.10 -y
     conda activate rag_pdf_env
3. Install Dependencies
     pip install -r requirements.txt
4. Run app.py


## Project Structure

 secure-rag-pdf-chatbot/
├── app.py                 # Streamlit app frontend
├── ingest.py              # Embeds documents and saves FAISS + JSON
├── retriever.py           # Loads FAISS index safely
├── qa_chain.py            # LLM + retrieval logic
├── utils/
│   └── pdf_loader.py      # PDF file extraction logic
├── vectorstore/           # FAISS index and documents (auto-generated)
├── requirements.txt       # Python dependencies
└── README.md


## Example Prompt Template

  You are a helpful assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say "I don’t know."
  
  Context:
  {context}
  
  Question:
  {input}
  
  Answer:

## 




