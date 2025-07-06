from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from retriever import get_retriever

load_dotenv()

def get_qa_chain(persist_path="vectorstore/faiss_index"):
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant. Use only the provided context to answer the question. If the answer is not in the context, say "I don’t know" — do not hallucinate.
    Context:
    {context}
    
    Question:
    {input}
    
    Answer:
    """)
    groq_api_key = os.environ["GROQ_API_KEY"]

    llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama3-70b-8192")
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)