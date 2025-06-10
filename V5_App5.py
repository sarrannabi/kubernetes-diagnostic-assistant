# Flask imports to handle the API, requests, JSON responses, and HTML rendering
from flask import Flask, request, jsonify, render_template

# CORS support to allow frontend-backend communication during development
from flask_cors import CORS

# Requests to fetch online data
import requests

# OS and subprocess to interact with the system and run kubectl commands
import os
import subprocess

# JSON module to handle data structures and save/load data
import json

# BeautifulSoup to extract text from HTML content (Kubernetes docs)
from bs4 import BeautifulSoup

# LangChain embedding model to convert logs into vectors
from langchain_community.embeddings import SentenceTransformerEmbeddings

# LangChain vector store wrapper for Qdrant
from langchain_community.vectorstores import Qdrant

# Interface for using the DeepSeek LLM via Ollama
from langchain_ollama import OllamaLLM

# Native Qdrant client for collection creation, search, and vector insertion
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- Initialization ---
# Create the Flask application
app = Flask(__name__)

# Enable CORS for cross-origin requests
CORS(app)

# Define the Qdrant collection name and connection URL
collection_name = "k8s_logs"
qdrant_url = "http://localhost:6333"

# Initialize the LLM using DeepSeek-R1 14B through Ollama
llm = OllamaLLM(model="deepseek-r1:14b")

# Create a Qdrant client instance to communicate with the vector DB
client = QdrantClient(qdrant_url)

# Load the sentence embedding model for vectorizing logs
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# List of Kubernetes documentation URLs to fetch log content from
k8s_urls = [
    "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
    "https://kubernetes.io/docs/tasks/debug/debug-cluster/"
]

# --- Utility Functions ---

# Download Kubernetes documentation and save them as local .txt files
def download_k8s_doc():
    for idx, url in enumerate(k8s_urls):
        filename = f"k8s_doc_{idx+1}.txt"
        if not os.path.exists(filename):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.get_text()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)

# Fetch recent StackOverflow questions tagged with Kubernetes and store them locally
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

# Collect logs from all running Kubernetes pods using kubectl and save them
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

# Load all previously saved logs (docs, StackOverflow, cluster logs) into one list
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

# Prepare the data: download logs, embed them, and store them in Qdrant if not already present
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

# --- Flask Routes ---

# Serve the HTML frontend
@app.route("/")
def home():
    return render_template("index.html")

# Handle POST requests from the frontend: accept a log, retrieve the closest match, query the LLM
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()  # Parse JSON input
    query = data.get("query", "")  # Extract the log or question
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Embed the query and perform vector search in Qdrant
    query_vector = embedding_model.embed_query(query)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1,
    )
    matched_log = search_result[0].payload["log"] if search_result else query

    # Construct the LLM prompt
    prompt = f"""
You are an expert Kubernetes diagnostics assistant.
Here is a log or issue to analyze:
{matched_log}

Give a precise, confident diagnosis and the exact fix. Use command examples if needed.
Return only the following:
- Root cause: ...
- Fix: ...
"""

    # Query the LLM with the prompt
    response = llm.invoke(prompt)

    # Try to extract structured parts from the LLM's response
    root_cause = None
    fix = None
    for line in response.split('\n'):
        if line.lower().startswith("root cause:"):
            root_cause = line[len("root cause:"):].strip()
        elif line.lower().startswith("fix:"):
            fix = line[len("fix:"):].strip()

    # If parsing failed, fall back to default placeholders
    if not root_cause:
        root_cause = "No root cause found."
    if not fix:
        fix = "No fix found."

    # Return all information as a JSON object
    return jsonify({
        "matched_log": matched_log,
        "root_cause": root_cause,
        "fix": fix,
        "ai_diagnosis": response
    })

# Entry point: run the setup and launch the Flask server
if __name__ == "__main__":
    setup()
    app.run(debug=True)

