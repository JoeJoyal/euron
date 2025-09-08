import os
import re
import nltk
import json
import faiss
import string
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 

from docx import Document
from PyPDF2 import PdfReader
from langdetect import detect
from nltk.tokenize import word_tokenize

nltk.download('punkt')

INPUT_FOLDER = "data/raw/"
OUTPUT_FILE = "data/embeddings/processed_IP/clean_.jsonl"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 50

def load_csv(filepath):
    df = pd.read_csv(filepath, on_bad_lines="skip")
    df['clean_content']=df['text'].apply(clean_text)
    return df['clean_content'].tolist()

def load_xlsx(filepath):
    df = pd.read_excel(filepath, sheet_name='custom-data-GenAI')
    df['clean_content']= df['text'].apply(clean_text)
    return df['clean_content'].tolist()

def load_txt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]
    
def load_docx(filepath):
    doc = Document(filepath)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def load_pdf(filepath):
    reader = PdfReader(filepath)
    text =""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return [line.strip() for line in text.split("\n") if line.strip()]


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+',' ', text).strip()
    return text

def normalize(text):
    return ' '.join(text.strip().split())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
def deduplicate(text):
    return list(dict.fromkeys(text))

def chunk_text(text, max_tokens=CHUNK_SIZE):
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = words[i:i+max_tokens]
        chunks.append(' '.join(chunk))
    return chunks

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
    
def visualize_embeddings(embeddings, labels=None, method="pca", title="Embedding Visualization"):
    """
    embeddings: NxD array
    labels: list of N language codes, e.g. ['hi','en','ta',...]
    method: "pca" or "tsne"
    """
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        mtitle = "PCA"
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=0, perplexity=30, max_iter=1000)
        reduced = reducer.fit_transform(embeddings)
        mtitle = "t-SNE"
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    plt.figure(figsize=(10, 7))
    
    if labels is not None:
        unique_labels = list(sorted(set(labels)))
        color_map = plt.get_cmap('tab10')  # good for up to 10 languages
        colors = {label: color_map(i) for i, label in enumerate(unique_labels)}
        # Optionally set nice names:
        lang_map = {'hi': 'Hindi', 'en': 'English', 'fr': 'French', 'ta': 'Tamil'}
        for label in unique_labels:
            idx = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(reduced[idx, 0], reduced[idx, 1], 
                        label=lang_map.get(label, label), 
                        color=colors[label], alpha=0.7, s=40)
        plt.legend(title="Language")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.5)
    plt.title(f"{title} ({mtitle})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def main():
    # Load, clean, normalize, deduplicate, chunk, detect language
    # For demonstration, we'll use sample texts
    all_texts = []
    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        extn = fname.lower().split('.')[-1]
        if extn == 'csv':
            all_texts.extend(load_csv(path))

    all_texts = [clean_text(t) for t in all_texts]
    all_texts = [normalize(t) for t in all_texts]
    all_texts = deduplicate(all_texts)
    print(f"Loaded and deduplicated {len(all_texts)} items.")

    DESIRED_LANGS = {"en","hi","ta","fr"}

    processed = []
    id_counter = 0
    for text in all_texts:
        lang = detect_language(text)
        if lang not in DESIRED_LANGS:
            continue # skip this text
        for chunk_id, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
            if len(chunk) < 10:
                continue
            processed.append({
                "id": id_counter,
                "chunk_id": chunk_id,
                "lang": lang,
                "text": chunk
            })
            id_counter += 1
    print(f"Processed {len(processed)} text chunks.")
    print("Sample:", processed[:4])
   

    # save to clean.jsonl
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: 
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Processed {len(processed)} text chunks saved to {OUTPUT_FILE}")

    # Embeddings (total no.of token consumpation 35,513, 35,547)
    embeddings = []
    for rec in processed:
        emb = generate_embeddings(rec['text'])
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"Skipping record id={rec['id']} due to embedding failure.")

    if len(embeddings) == 0:
        print("No embeddings were created! Exiting.")
        return

    embeddings = np.vstack(embeddings).astype(np.float32)
    np.save("data/embeddings/processed_IP/embedding_arrays.npy", embeddings)

    
    # index = faiss.IndexFlatL2(embeddings.shape[1])
    # index.add(embeddings)
    # print(f"FAISS created for {index.ntotal} vectors.")

    # 5. FAISS: Cosine Index 35,547
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # np.save("data/embeddings/processed_IP/embedding_arrays.npy", embeddings)
    norm_embeddings = embeddings / np.clip(norms, 1e-8, None)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(norm_embeddings)
    print(f"FAISS:Cosine created for {index.ntotal} vectors.")
    

    # save faiss index and metadata
    faiss.write_index(index, "data/embeddings/processed_IP/indexflatIP.faiss")
    # faiss.write_index(index, "data/processed_ip/faiss_cosine.index")

    with open("data/embeddings/processed_IP/metadata_IP.jsonl", "w", encoding="utf-8") as f:
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("Pipeline run completed.")

    embedding_arrays = np.load("data/embeddings/processed_IP/embedding_arrays.npy")
    print(embedding_arrays.shape)
    labels = [rec['lang'] for rec in processed]
    visualize_embeddings(embedding_arrays, labels, method="pca", title="Text Embeddings Clustered by Language")
    visualize_embeddings(embedding_arrays, labels, method="tsne", title="Text Embeddings Clustered by Language")


if __name__ == "__main__":
    main()