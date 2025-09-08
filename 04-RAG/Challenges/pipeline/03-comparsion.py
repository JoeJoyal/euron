import json
import faiss
import requests
import numpy as np


def generate_embeddings(text, model="text-embedding-3-small", normalize=False):
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
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    return embedding.reshape(1, -1)

# --- Load metadata ---
metadata = []
with open("data/embeddings/processed_IP/metadata_IP.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))


# --- Load all stored vectors ---
embeds = np.load("data/embeddings/processed_IP/embedding_arrays.npy")  # shape: (n, dim)


# Build FAISS indices for both similarity types:
# Cosine similarity (IP + normalized vectors)
index_cosine = faiss.IndexFlatL2(embeds.shape[1])
norm_embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
index_cosine.add(norm_embeds)

# Inner product (IP + raw vectors, NOT normalized)
index_ip = faiss.IndexFlatIP(embeds.shape[1])
index_ip.add(embeds)

# ---- Query Section ----
query = "What is a vertical indoor farming?"

# --- Prompt ----
# 1.What is a vertical indoor farming?
# 2. Indoor farming yield? How can accelerate growth.
# 3. By LLM models in the future ageye used?
# 4. Tell me about ageye?
# 5. What are the automation agencies used in this company?

# Cosine similarity: use normalized embedding
q_embed_norm = generate_embeddings(query, normalize=True)
score_c, idx_c = index_cosine.search(q_embed_norm, 5)

print("=== Cosine Similarity (top-5) ===")
for rank, idx in enumerate(idx_c[0]):
    rec = metadata[idx]
    print(f"Rank {rank+1} (Score: {score_c[0][rank]:.4f}):\nText: {rec['text']}\nLang: {rec['lang']} \n----\n")

# Inner Product: use raw embedding
q_embed_raw = generate_embeddings(query, normalize=False)
score_ip, idx_ip = index_ip.search(q_embed_raw, 5)

print("\n=== Inner Product (top-5) ===")
for rank, idx in enumerate(idx_ip[0]):
    rec = metadata[idx]
    print(f"Rank {rank+1} (Score: {score_ip[0][rank]:.4f}):\nText: {rec['text']}\nLang: {rec['lang']} \n----\n")

# ---- Which is more relevant? ----
# print("""
# Q: Which method retrieves more relevant results?
# A: Cosine similarity is generally more relevant for text embeddings because it measures the angle (semantic similarity)
#    between vectors and is invariant to vector magnitude. Inner product can be dominated by vector norms, potentially
#    distorting relevance when vector magnitudes vary. For semantic search, cosine similarity (i.e., normalized vectors + IP index) 
#    is preferred.
# """)