from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
import subprocess
import json
from bs4 import BeautifulSoup
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- Initialisation ---
app = Flask(__name__)
CORS(app)

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
collection_name = "k8s_logs"
qdrant_url = "http://localhost:6333"

llm = OllamaLLM(model="deepseek-r1:14b")
client = QdrantClient(qdrant_url)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

k8s_urls = [
    "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/"
]

# --- Fonctions utilitaires ---
def download_k8s_doc():
    for idx, url in enumerate(k8s_urls):
        filename = f"k8s_doc_{idx+1}.txt"
        if not os.path.exists(filename):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

def fetch_stackoverflow_errors():
    filename = "stackoverflow_data.txt"
    if os.path.exists(filename):
        return
    url = "https://api.stackexchange.com/2.3/search?order=desc&sort=activity&tagged=kubernetes&site=stackoverflow"
    response = requests.get(url)
    data = response.json()
    with open(filename, "w", encoding="utf-8") as f:
        for item in data.get("items", []):
            f.write(item.get("title", "") + "\n")

def collect_cluster_logs():
    filename = "cluster_logs.txt"
    if os.path.exists(filename):
        return
    with open(filename, "w", encoding="utf-8") as f:
        pods = subprocess.getoutput("kubectl get pods --all-namespaces -o json")
        pod_data = json.loads(pods)
        for pod in pod_data.get("items", []):
            ns = pod["metadata"]["namespace"]
            name = pod["metadata"]["name"]
            try:
                logs = subprocess.getoutput(f"kubectl logs -n {ns} {name}")
                f.write(f"POD: {name}\n{logs}\n\n")
            except Exception:
                continue

def load_all_logs():
    logs = []
    for i in range(len(k8s_urls)):
        with open(f"k8s_doc_{i+1}.txt", "r", encoding="utf-8") as f:
            logs.append(f.read())
    with open("stackoverflow_data.txt", "r", encoding="utf-8") as f:
        logs.extend([line.strip() for line in f.readlines()])
    with open("cluster_logs.txt", "r", encoding="utf-8") as f:
        logs.append(f.read())
    return logs

def setup():
    download_k8s_doc()
    fetch_stackoverflow_errors()
    collect_cluster_logs()
    logs = load_all_logs()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        for idx, log in enumerate(logs[:100]):
            vector = embedding_model.embed_query(log)
            client.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=idx+1, vector=vector, payload={"log": log})]
            )
    return logs

# --- Routes Flask ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_vector = embedding_model.embed_query(query)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1,
    )
    matched_log = search_result[0].payload["log"] if search_result else query

    prompt = f"""
You are an expert Kubernetes diagnostics assistant.
Here is a log or issue to analyze:
{matched_log}

Give a precise, confident diagnosis and the exact fix. Use command examples if needed.
Return only the following:
- Root cause: ...
- Fix: ...
"""
    response = llm.invoke(prompt)

    # Supposons que la réponse est un texte contenant "Root cause: ..." et "Fix: ..."
    # On peut parser la réponse pour extraire ces deux infos (optionnel)
    root_cause = None
    fix = None
    for line in response.split('\n'):
        if line.lower().startswith("root cause:"):
            root_cause = line[len("root cause:"):].strip()
        elif line.lower().startswith("fix:"):
            fix = line[len("fix:"):].strip()

    # Fallback si pas trouvé
    if not root_cause:
        root_cause = "No root cause found."
    if not fix:
        fix = "No fix found."

    return jsonify({
        "matched_log": matched_log,
        "root_cause": root_cause,
        "fix": fix,
        "ai_diagnosis": response
    })

if __name__ == "__main__":
    setup()
    app.run(debug=True)

