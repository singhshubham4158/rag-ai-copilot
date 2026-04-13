from langchain_community.vectorstores import FAISS
import os

def get_db(docs, emb):
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)

    db = FAISS.from_documents(docs, emb)
    db.save_local("faiss_index")
    return db