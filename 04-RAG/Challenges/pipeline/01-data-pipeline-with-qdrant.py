import os
import re
import json
import string
import requests
import faiss
import nltk
nltk.download("punkt_tab")
import numpy as np
import pandas as pd

from docx import Document
from PyPDF2 import PdfReader
from langdetect import detect
from nltk.tokenize import word_tokenize, sent_tokenize
from vectordb.qdrant import QdrantClient, client, models

INPUT_FOLDER = "data/raw/"
OUTPUT_FILE = "data/embeddings/processed_IP/clean_.jsonl"
CHUNK_SIZE = 300
COLLECTION_NAME = "ageye"


def load_csv(filepath):
    df = pd.read_csv(filepath)
    df['clean_content']=df['text'].apply(clean_text)
    return df['clean_content'].tolist()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+',' ', text).strip()
    return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"
    
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
        "Authorization": "Bearer euri-6897392f64d349a7fdd3b577e63125855b1a3abd931fa3369add8bedb41498fb" # This API Key de-activated
    }
    payload = {
        "input": text,
        "model": model
    }

    response = requests.post(url, headers=headers, json=payload)
    return np.array(response.json()['data'][0]['embedding'])


def main():
    all_texts = []
    for fname in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, fname)
        extn = fname.lower().split('.')[-1]
        if extn == 'csv':
            all_texts.extend(load_csv(path))
    

    # all_texts = [normalize(t) for t in all_texts]  # Deduplicate and Normalize
    # print(all_texts)
    # all_texts = deduplicate(all_texts)
    # print("Deduplicate:", all_texts)
    # print(f"Loaded and deduplicated {len(all_texts)} items.")

    # Detect lang, chunk, pred-data
    processed = []
    id_counter = 0
    for text in all_texts:
        lang = detect_language(text)
        text_chunks = chunk_text(text, CHUNK_SIZE)
        for chunk_id, chunk in enumerate(text_chunks):
            if len(chunk) < 10: # skip empty or too small
                continue
            record = {
                "id": id_counter,
                "chunk_id": chunk_id,
                "lang": lang,
                "text": chunk
            }
            processed.append(record)
            id_counter +=1


    # save to clean.jsonl
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: 
        for rec in processed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Processed {len(processed)} text chunks saved to {OUTPUT_FILE}")

    # Embedding & FAISS Indexing
    embeddings = []
    for i in processed:
        emb = generate_embeddings(i['text']).tolist()
        embeddings.append(emb)
    print(len(embeddings))
    embedding_arrays = np.vstack(embeddings)
    dim = embedding_arrays.shape[1]
    print("Dimension:",dim)

    # Qdrant Integration
    try:
        client.create_collection(
            collection_name = COLLECTION_NAME,
            vectors_config={
                        "size": 1536,
                        "distance":"Cosine"
                        }
    )
    except Exception as e:
        if "already exists" in str(e):
            print(f"collection {COLLECTION_NAME} already exists, skipping creation.")
        else:
            raise

    # Prepare payloads
    payloads = [{"id": rec["id"], "lang": rec["lang"], "text": rec["text"]} for rec in processed]
    
    ids = [rec["id"] for rec in processed]



    points=[
        models.PointStruct(id=ids[i], vector=embedding_arrays[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserted {len(ids)} records into Qdrant collection '{COLLECTION_NAME}'.")


    collection_info = client.get_collection(COLLECTION_NAME)
    print("Total points in collection:", collection_info.points_count)

if __name__ == "__main__":
    main()

