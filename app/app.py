from flask import Flask, request, render_template
from src.pipeline import run_pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask")
def ask():
    query = request.args.get("q")

    if not query:
        return {"error": "Query required"}

    response = run_pipeline(query)

    return response

if __name__ == "__main__":
    app.run(debug=True)

from src.ingestion import load_docs
from src.embeddings import get_embeddings
from src.vectordb import get_db

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    path = f"data/{file.filename}"
    file.save(path)

    docs = load_docs()
    emb = get_embeddings()
    db = get_db(docs, emb)

    return {"msg": "File uploaded successfully"}