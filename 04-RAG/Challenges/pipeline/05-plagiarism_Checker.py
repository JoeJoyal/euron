import os
import re
import numpy as np
import faiss
import requests

# --- Settings/Constants ---
EXISTING_ESSAYS_FOLDER = "data/raw/blogs/" 
NEW_ESSAYS_FOLDER     = "data/raw/new_blogs/"
EMBEDDING_DIM         = 1536
THRESHOLD             = 0.85
EMBEDDING_MODEL       = "text-embedding-3-small"
FAISS_INDEX_FILE      = "data/embeddings/essays.faiss"
EMBEDDINGS_NPY_FILE   = "data/embeddings/essays.npy"

# --- Load all .txt essays from a folder ---
def load_essays_from_folder(folder_path):
    essays, fnames = [], []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    essays.append(content)
                    fnames.append(fname)
    print(f"Loaded {len(essays)} essays from {folder_path}")
    return essays, fnames

# --- Clean each essay ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Embedding Function---
def generate_embedding(text, model=EMBEDDING_MODEL):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer euri-6897392f64d349a7fdd3b577e63125855b1a3abd931fa3369add8bedb41498fb" # This API Key de-activated
    }
    payload = {"input": text, "model": model}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            print(f"[API ERROR] Status: {response.status_code} | Text: {response.text}")
            return None
        resp_json = response.json()
        if 'data' not in resp_json:
            print(f"[API ERROR] No 'data' in embedding response: {resp_json}")
            return None
        return np.array(resp_json['data'][0]['embedding'])
    except Exception as e:
        print(f"[API EXCEPTION]: {e}")
        return None

# --- Robust batch embedding generation for a list of essays ---
def embed_essays(essays):
    valid_essays, valid_embeddings = [], []
    for essay in essays:
        emb = generate_embedding(essay)
        if emb is not None:
            valid_essays.append(essay)
            valid_embeddings.append(emb)
        else:
            print("[WARNING] Skipped essay due to embedding API failure.")
    if len(valid_embeddings) == 0:
        raise ValueError("No valid embeddings could be generated!")
    return valid_essays, np.vstack(valid_embeddings).astype(np.float32)

# --- Build and normalize FAISS index ---
def build_faiss_index(embeddings):
    # Normalize for cosine similarity (required with IndexFlatIP)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / np.clip(norms, 1e-8, None)
    index = faiss.IndexFlatIP(norm_embeddings.shape[1])
    index.add(norm_embeddings)
    return index

def save_faiss_index(index, path): faiss.write_index(index, path)
def load_faiss_index(path): return faiss.read_index(path)
def save_embeddings(embeddings, path): np.save(path, embeddings)
def load_embeddings(path): return np.load(path)

# --- Batch check for plagiarism ---
def check_plagiarism_batch(new_essays, index, existing_essays, threshold=THRESHOLD):
    valid_new_essays, new_embeds = embed_essays(new_essays)
    # Normalize new essay embeddings before search!
    norms = np.linalg.norm(new_embeds, axis=1, keepdims=True)
    norm_new_embeds = new_embeds / np.clip(norms, 1e-8, None)
    D, I = index.search(norm_new_embeds, k=1)
    results = []
    for i, (score, match_idx) in enumerate(zip(D[:, 0], I[:, 0])):
        result = {
            'essay_index': i,
            'is_plagiarism': bool(score >= threshold),
            'similarity': float(score),
            'matched_essay_index': int(match_idx),
            'matched_essay': existing_essays[match_idx]
        }
        results.append(result)
    return results, valid_new_essays

# --- Main workflow ---
def main():
    # 1. Load and clean essays to build the reference database
    essays, essay_filenames = load_essays_from_folder(EXISTING_ESSAYS_FOLDER)
    essays = [clean_text(e) for e in essays]

    # 2. Build or load embeddings and FAISS index
    if not os.path.isfile(FAISS_INDEX_FILE) or not os.path.isfile(EMBEDDINGS_NPY_FILE):
        valid_essays, essay_embeddings = embed_essays(essays)
        save_embeddings(essay_embeddings, EMBEDDINGS_NPY_FILE)
        index = build_faiss_index(essay_embeddings)
        save_faiss_index(index, FAISS_INDEX_FILE)
        print("Built and saved FAISS index and embeddings.")
    else:
        essay_embeddings = load_embeddings(EMBEDDINGS_NPY_FILE)
        index = load_faiss_index(FAISS_INDEX_FILE)
        valid_essays = essays  # If index prebuilt, use original essay order
        print("Loaded FAISS index and embeddings.")

    # 3. Load, clean, and batch check new essays
    new_essays, new_filenames = load_essays_from_folder(NEW_ESSAYS_FOLDER)  # folder of new essays to check
    new_essays = [clean_text(e) for e in new_essays]
    results, checked_essays = check_plagiarism_batch(new_essays, index, valid_essays, THRESHOLD)

    # 4. Report
    for res, fname, essay in zip(results, new_filenames, checked_essays):
        print(f"\nEssay File: {fname}")
        print(f"Plagiarism detected: {res['is_plagiarism']} (similarity={res['similarity']:.3f})")
        if res['is_plagiarism']:
            print("Most similar essay in database (first 200 chars):")
            print(res['matched_essay'][:200] + ("..." if len(res['matched_essay']) > 200 else ""))
        else:
            print("No close match found.")

if __name__ == "__main__":
    main()