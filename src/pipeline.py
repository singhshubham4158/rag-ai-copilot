from dotenv import load_dotenv
import os

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from src.ingestion import load_docs
from src.embeddings import get_embeddings
from src.vectordb import get_db
from src.retriever import get_retriever
from src.reranker import rerank
from src.memory import Memory

load_dotenv()

memory = Memory()

# 🔥 BUILD RAG COMPONENTS
docs = load_docs()
emb = get_embeddings()
db = get_db(docs, emb)
retriever = get_retriever(db, docs)


pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=512,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)


def run_pipeline(query):
    docs = retriever(query)
    docs = rerank(query, docs)

    context = "\n\n".join([
        f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)
    ])

    history = memory.get()

    prompt = f"""
You are a helpful AI assistant.

Answer the question using the context below.

Rules:
- Give a complete answer in 2-3 sentences
- Do NOT return only numbers like [1]
- Explain clearly in simple language

Context:
{context}

Question:
{query}
"""