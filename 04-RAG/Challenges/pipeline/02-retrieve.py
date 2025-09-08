import json
import faiss
import requests
import numpy as np

# --- Embedding Function ---
def generate_embeddings(text, model="text-embedding-3-small"):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer euri-1b4b28c5996dcf0e9f925888b5c5be28077420f6a1eb31ba3981c3dd14abad29" # This API Key de-activated
    }
    payload = {
        "input": text,
        "model": model
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    embedding = np.array(data['data'][0]['embedding'], dtype='float32')
    return embedding.reshape(1, -1)

# --- Load existing FAISS index ---
index = faiss.read_index("data/embeddings/processed_L2/indexflatL2.faiss")

# --- Load metadata records ---
metadata = []
with open("data/embeddings/processed_L2/metadata_L2.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

# --- Prepare and embed query ---
query = "Tell me about ageye?"
q_embedding = generate_embeddings(query)

# Sample Prompt Template
# Tell me about ageye?
# By LLM models in the future ageye used
# What is a vertical indoor farming
# What are the automation agencies used in this company?

# --- Retrieve top-5 most similar records ---
top_k = 5
scores, indices = index.search(q_embedding, top_k)

# --- Display results ---
print("Top 5 search results for your query:\n")
for rank, idx in enumerate(indices[0]):
    rec = metadata[idx]
    print(f"Rank {rank+1} (Score: {scores[0][rank]:.4f}):\nText: {rec['text']}\nLang: {rec['lang']} \n----\n")