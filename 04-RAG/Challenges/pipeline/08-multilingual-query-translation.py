import faiss
import requests
import numpy as np
from googletrans import Translator

# --- config ---
EMBEDDING_DIM = 1536

# --- Embedding function ---
def generate_embeddings(texts):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer euri-6897392f64d349a7fdd3b577e63125855b1a3abd931fa3369add8bedb41498fb" # This API Key de-activated
    }
    if isinstance(texts, str):
        texts = [texts]
    payload = {
        "input": texts,
        "model": "text-embedding-3-small"
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        embeds = [np.array(item['embedding'], dtype=np.float32) for item in data['data']]
        return np.vstack(embeds)
    except Exception as e:
        print(f"[Embedding API Error]: {e}")
        # fallback: empty or random
        return np.vstack([np.zeros(1536, dtype=np.float32)] * len(texts))

# --- Data ---
english_docs = [
    "Climate change is a major threat to our planet.",
    "Artificial Intelligence is transforming industries worldwide.",
    "The Taj Mahal is a famous monument in India.",
    "Python is a popular programming language for data science.",
    "Vaccines are crucial for public health."
]
hindi_queries = [
    "जलवायु परिवर्तन हमारे ग्रह के लिए एक बड़ा खतरा है।",
    "कृत्रिम बुद्धिमत्ता दुनिया भर में उद्योगों को बदल रही है।",
    "ताज महल भारत में एक प्रसिद्ध स्मारक है।",
    "पायथन डेटा विज्ञान के लिए एक लोकप्रिय प्रोग्रामिंग लैंग्वेज है।",
    "टीके सार्वजनिक स्वास्थ्य के लिए आवश्यक हैं।"
]
ground_truth = [i for i in range(len(english_docs))]


# --- Build FAISS index from English docs (cosine sim, so normalize) ---
doc_embeds = generate_embeddings(english_docs)
doc_embeds = doc_embeds / np.linalg.norm(doc_embeds, axis=1, keepdims=True)
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(doc_embeds)

# --- True Multilingual Retrieval ---
def semantic_search_multilingual(queries, index, k=3):
    q_embeds = generate_embeddings(queries)
    q_embeds = q_embeds / np.linalg.norm(q_embeds, axis=1, keepdims=True)
    D, I = index.search(q_embeds, k)
    return I

# --- Query-Translation to English, then retrieval ---
def translate_to_english(queries_hindi):
    translator = Translator()
    return [translator.translate(q, src='hi', dest='en').text for q in queries_hindi]

def semantic_search_query_translation(queries_hindi, index, k=3):
    queries_en = translate_to_english(queries_hindi)
    q_embeds = generate_embeddings(queries_en)
    q_embeds = q_embeds / np.linalg.norm(q_embeds, axis=1, keepdims=True)
    D, I = index.search(q_embeds, k)
    return I

# --- Evaluation: Recall@K ---
def recall_at_k(results_indices, ground_truth, k):
    recall = 0
    for i, row in enumerate(results_indices):
        if ground_truth[i] in row[:k]:
            recall += 1
    return recall / len(ground_truth)

# --- Run Both Methods ---
k = 1
print(f"\n--- Multilingual Semantic Search (Hindi→English index) ---")
indices_multi = semantic_search_multilingual(hindi_queries, index, k=k)
recall_multi = recall_at_k(indices_multi, ground_truth, k)
print(f"Recall@{k}: {recall_multi:.2f}")

print(f"\n--- Query-Translation Approach (Hindi→Translated→English index) ---")
indices_trans = semantic_search_query_translation(hindi_queries, index, k=k)
recall_trans = recall_at_k(indices_trans, ground_truth, k)
print(f"Recall@{k}: {recall_trans:.2f}")

# --- results ---
for i, q in enumerate(hindi_queries):
    print(f"\nHindi Query: {q}")
    print("Multilingual Top-K Matches:")
    for rank, idx in enumerate(indices_multi[i]):
        print(f"  {rank+1}: {english_docs[idx]}")
    print("\nQuery-Translation Top-K Matches:")
    for rank, idx in enumerate(indices_trans[i]):
        print(f"  {rank+1}: {english_docs[idx]}")