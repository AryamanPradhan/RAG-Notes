from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

def get_retriever( index_path="vectorstore/faiss_index", model_name="all-MiniLM-L6-v2", doc_path="vectorstore/docs.json"):
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Load FAISS vectorstore from disk
    db = FAISS.load_local(index_path, embeddings,allow_dangerous_deserialization=True)

    # Convert to retriever
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # Serialize manually
    if hasattr(db, "docs"):
        serialized = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in db.docs
        ]
        with open(doc_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    return retriever
