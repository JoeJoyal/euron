import json
import requests
import numpy as np
from qdrant_client import QdrantClient, models


# --- Config ---
EMBEDDING_DIM = 1536
EURI_API_KEY = "Bearer euri-6897392f64d349a7fdd3b577e63125855b1a3abd931fa3369add8bedb41498fb" # This API Key de-activated
COLLECTION_NAME = "python-Code"

def generate_embeddings(text, model="text-embedding-3-small"):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": EURI_API_KEY}
    if isinstance(text, str):
        text = [text]
    payload = {"input": text, "model": model}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return [np.array(d['embedding'], dtype=np.float32) for d in data['data']]

def embed_and_store(data, client, collection_name=COLLECTION_NAME):
    # Generate embeddings and prepare points
    vectors, payloads, ids = [], [], []
    for idx, code_obj in enumerate(data):
        embedding = generate_embeddings(code_obj['code'])[0]
        vectors.append(embedding.tolist())
        payload = {k: code_obj[k] for k in ["code_id", "tags", "description", "coding_language", "code"]}
        payloads.append(payload)
        ids.append(idx)  # Qdrant prefers integer or UUID ids.
    # Recreate collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)
    )
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=ids
    )

def retrieve_code(query, client, collection_name=COLLECTION_NAME, top_k=1):
    emb = generate_embeddings(query)[0].tolist()
    results = client.search(
        collection_name=collection_name,
        query_vector=emb,
        limit=top_k,
        with_payload=True
    )
    return [hit.payload['code'] for hit in results]

# --- Main Workflow ---
if __name__ == "__main__":
    # Load your code dataset
    with open("data/raw/python-code.json") as f:
        data = json.load(f)
    
    client = QdrantClient(
        url="https://d5ecbd2d-10c9-4d63-baa9-758083e15268.eu-west-1-0.aws.cloud.qdrant.io:6333",
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.XkvP8DFumSOX5FxJWxqm8ZprYT_-UXvqhWD8Iq9q2Os", # This API Key de-activated
        timeout=120 # set higher if needed
    )
    
    # 1. Embed and store snippets
    embed_and_store(data, client)
    # 2. Search
    query = 'binary search function'
    results = retrieve_code(query, client)
    print("Top match for your query:")
    print(results[0] if results else "No match found.")