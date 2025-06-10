from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_ollama import OllamaLLM

import requests
import os
import hashlib
import json
from bs4 import BeautifulSoup
import datetime
import time

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

# ----- HISTORIQUE -----
history_file = "history.json"

def save_to_history(query, root_cause, fix, useful=None):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "root_cause": root_cause,
        "fix": fix,
        "useful": useful
    }
    try:
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.append(entry)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print("Erreur en sauvegardant l'historique:", e)

# ----- NOUVEAU: Gestion cache fichiers -----
def get_hashed_filename(url):
    return hashlib.md5(url.encode()).hexdigest() + ".txt"

def download_k8s_docs():
    logs = []
    for url in k8s_urls:
        filename = get_hashed_filename(url)
        if not os.path.exists(filename):
            print(f"🔁 Téléchargement {url} ...")
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(f"✅ Fichier déjà présent: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            logs.append(f.read())
    return logs

def fetch_stackoverflow():
    cache_file = "stackoverflow_cache.txt"
    if os.path.exists(cache_file):
        print("✅ Cache StackOverflow présent")
        with open(cache_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    else:
        print("🔁 Récupération StackOverflow...")
        response = requests.get(stackoverflow_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.select("div.question-summary h3 a")
        logs = ["https://stackoverflow.com" + link["href"] + " - " + link.text.strip() for link in results]
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
        return logs

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
You are a Kubernetes troubleshooting assistant. You are provided with context chunks from official documentation and community posts. Your role is to answer the user's query based only on that context.

Instructions:
- Do not hallucinate information.
- If the context does not contain the answer, reply: "I do not know based on the available information."
- If the issue is about a specific error (e.g., OOMKilled, CrashLoopBackOff), provide possible root causes and solutions based on the provided context.
You must return **only** the following two sections with no preamble, no explanations, and no tags. Format exactly like this:

Root cause:
<one or two concise sentences explaining the root cause>

Fix:
<precise and actionable solution with commands or YAML if needed>

User Query:
{query_log}
"""
    response = llm.invoke(prompt)

    root_cause = ""
    fix = ""
    if "Root cause:" in response and "Fix:" in response:
        root_cause = response.split("Root cause:")[1].split("Fix:")[0].strip()
        fix = response.split("Fix:")[1].strip()
    else:
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

    start_time = time.time()

    try:
        retrieved, root_cause, fix = query_ai(client, embedder, query)
        elapsed = time.time() - start_time
        save_to_history(query, root_cause, fix)
        return jsonify({
            "root_cause": root_cause,
            "fix": fix,
            "response_time_sec": round(elapsed, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    query = data.get("query")
    useful = data.get("useful")

    try:
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

        for entry in reversed(history):
            if entry["query"] == query and entry["useful"] is None:
                entry["useful"] = useful
                break

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

