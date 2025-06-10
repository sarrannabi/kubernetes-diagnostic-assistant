from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Step 1: Predefined Kubernetes logs
known_logs = [
    "Pod nginx crashed with CrashLoopBackOff",
    "Pod backend stuck in ContainerCreating",
    "Pod frontend restarted due to OOMKilled",
    "Pod database failed due to liveness probe failure",
    "Pod cache terminated with error code 137",
    "Pod api failed to pull image",
    "Pod ingress-controller CrashLoopBackOff",
    "Pod metrics-server readiness probe failed"
]

# Step 2: Setup vector DB and embedding
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient("http://localhost:6333")

client.recreate_collection(
    collection_name="k8s_logs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

for idx, log in enumerate(known_logs):
    vector = embedding_model.embed_query(log)
    client.upsert(
        collection_name="k8s_logs",
        points=[PointStruct(id=idx + 1, vector=vector, payload={"log": log})],
    )

# Step 3: Ask user for input
user_input = input("Describe your Kubernetes problem (e.g., 'my pod is in CrashLoopBackOff'): ")

# Step 4: Search most similar log
query_vector = embedding_model.embed_query(user_input)
result = client.search("k8s_logs", query_vector=query_vector, limit=1)
retrieved_log = result[0].payload["log"]

# Step 5: Send to LLM
llm = OllamaLLM(model="deepseek-r1:8b")
prompt = f"""
You are a helpful Kubernetes assistant.
A user is facing a problem similar to: "{retrieved_log}"
Based on your knowledge, please explain:
- Root cause
- Fix
"""

# Step 6: Get response
response = llm.invoke(prompt)
print("\n Assistant's Diagnosis:\n")
print(response)

