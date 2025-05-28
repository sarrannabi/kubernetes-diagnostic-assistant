from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_ollama import OllamaLLM

import requests
import os
import hashlib
from bs4 import BeautifulSoup

app = Flask(__name__)

# ----- CONFIG -----
k8s_urls = [
    "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/"
]
stackoverflow_url = "https://stackoverflow.com/search?q=[kubernetes]+errors"

qdrant_url = "http://localhost:6333"
model_name = "deepseek-r1:8b"
embedding_model_name = "all-MiniLM-L6-v2"
collection_name = "k8s_logs"

# ----- FONCTIONS -----
def get_hashed_filename(url):
    return hashlib.md5(url.encode()).hexdigest() + ".txt"

def download_k8s_docs():
    logs = []
    for url in k8s_urls:
        filename = get_hashed_filename(url)
        if not os.path.exists(filename):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        with open(filename, "r", encoding="utf-8") as f:
            logs.append(f.read())
    return logs

def fetch_stackoverflow():
    # Placeholder pour StackOverflow, à améliorer plus tard
    return ["Example StackOverflow error log 1", "Example StackOverflow error log 2"]

def collect_logs():
    return download_k8s_docs() + fetch_stackoverflow()

def store_logs_in_qdrant(logs):
    client = QdrantClient(url=qdrant_url)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    for idx, log in enumerate(logs[:50]):
        vector = embedding_model.embed_query(log)
        client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=idx + 1, vector=vector, payload={"log": log})]
        )
    return client, embedding_model

def query_ai(client, embedder, query_log):
    query_vector = embedder.embed_query(query_log)
    result = client.search(collection_name=collection_name, query_vector=query_vector, limit=1)
    retrieved_log = result[0].payload["log"] if result else query_log

    llm = OllamaLLM(model=model_name)
    prompt = f"""
You are a Kubernetes diagnostic assistant.

Given the following log:

{retrieved_log}

Return ONLY two sections exactly like this:

Root cause:
<one or two concise sentences explaining the root cause>

Fix:
<a precise and actionable solution including commands or YAML snippets if needed>

Do NOT add anything else, no explanations, no tags, no thoughts. Only these two labeled sections.
"""
    response = llm.invoke(prompt)

    # Parsing simple et robuste
    root_cause = ""
    fix = ""
    if "Root cause:" in response and "Fix:" in response:
        root_cause = response.split("Root cause:")[1].split("Fix:")[0].strip()
        fix = response.split("Fix:")[1].strip()
    else:
        # fallback: renvoyer tout en root_cause si format inattendu
        root_cause = response.strip()
        fix = ""

    return retrieved_log, root_cause, fix

# ----- INIT -----
print("🔄 Collecting and embedding logs at server start...")
logs = collect_logs()
client, embedder = store_logs_in_qdrant(logs)
print("✅ Qdrant ready with embedded logs.")

# ----- ROUTES -----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        retrieved, root_cause, fix = query_ai(client, embedder, query)
        return jsonify({"root_cause": root_cause, "fix": fix})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

