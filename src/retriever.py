from langchain_community.retrievers import BM25Retriever

def get_retriever(db, docs):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5

    vector = db.as_retriever(search_kwargs={"k": 5})

    def retrieve(query):
        dense = vector.get_relevant_documents(query)
        sparse = bm25.get_relevant_documents(query)

        return list({d.page_content: d for d in dense + sparse}.values())

    return retrieve