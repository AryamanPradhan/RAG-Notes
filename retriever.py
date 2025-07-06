from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import serialize_docs

def get_retriever(docs, index_path="vectorstore/faiss_index", model_name="all-MiniLM-L6-v2",doc_path="vectorstore/docs.json"):
    # Load the same embedding model 
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Load the FAISS index from disk
    db = FAISS.load_local(index_path, embeddings )

    # Convert it to a retriever object for RAG
    retriever = db.as_retriever(search_kwargs={"k": 6})

    serialized = serialize_docs(db.docs)
    import json
    with open(doc_path, "w") as f:
        json.dump(serialized, f)

    return retriever
