import os
import numpy as np
import uuid
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient, models
import requests

# --- Configuration ---
collection_name = "resume-builder"
EMBEDDING_DIM = 1536
EURI_API_KEY = "Bearer euri-6897392f64d349a7fdd3b577e63125855b1a3abd931fa3369add8bedb41498fb" # This API Key de-activated
EURI_MODEL = "text-embedding-3-small"

# --- PDF to Text ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text.strip()

# --- Embedding Function (EURI API) ---
def generate_embedding_euri(text, model=EURI_MODEL):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": EURI_API_KEY
    }
    payload = {"input": text, "model": model}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200 and 'data' in response.json():
            data = response.json()
            return np.array(data['data'][0]['embedding'], dtype=np.float32)
        print(f"Embedding Error: {response.text}")
    except Exception as e:
        print(f"[API Exception] {e}")
    return None

# --- Upload Resumes to Qdrant ---
def store_resumes_in_qdrant(pdf_folder, client, collection_name):
    resumes, payloads, ids, vectors = [], [], [], []
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, fname)
            text = extract_text_from_pdf(file_path)
            emb = generate_embedding_euri(text)
            if emb is not None:
                resumes.append(text)
                payloads.append({"filename": fname, "text": text})
                ids.append(str(uuid.uuid4()))  # Use UUID for Qdrant point ID
                vectors.append(emb.tolist())
            else:
                print(f"Skipped {fname} due to embedding error.")
    # Check if collection exists, else create (do not use deprecated recreate_collection)
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' exists. Deleting and recreating...")
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
    )
    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=ids
    )
    print(f"Uploaded {len(vectors)} resumes to Qdrant collection '{collection_name}'.")
    return resumes, payloads, ids

# --- Retrieve Closest Resume(s) for a Job Description ---
def retrieve_closest_resumes(job_description, client, collection_name, top_k=1):
    emb = generate_embedding_euri(job_description)
    if emb is None:
        print("[ERROR] Could not get embedding for job description.")
        return []
    results = client.search(
        collection_name=collection_name,
        query_vector=emb.tolist(),
        limit=top_k,
        with_payload=True
    )
    # Each result contains .payload["filename"], .payload["text"]
    return results

# --- Recommend Job Descriptions for Resume ---
def recommend_jobs_for_resume(resume_text, job_descriptions, top_k=3):
    resume_emb = generate_embedding_euri(resume_text)
    job_vecs, descriptions = [], []
    for job in job_descriptions:
        emb = generate_embedding_euri(job)
        if emb is not None:
            job_vecs.append(emb)
            descriptions.append(job)
    job_vecs = np.vstack(job_vecs)
    # Normalize for cosine similarity
    norms = np.linalg.norm(job_vecs, axis=1, keepdims=True)
    job_vecs = job_vecs / np.clip(norms, 1e-8, None)
    emb_norm = resume_emb / (np.linalg.norm(resume_emb) + 1e-10)
    scores = job_vecs @ emb_norm
    top_indices = np.argsort(scores.flatten())[::-1][:top_k]
    return [descriptions[i] for i in top_indices]

# --- Main pipeline ---
def main():
    # 1. Qdrant client setup
    client = QdrantClient(
                    url="https://d5ecbd2d-10c9-4d63-baa9-758083e15268.eu-west-1-0.aws.cloud.qdrant.io:6333", 
                    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.XkvP8DFumSOX5FxJWxqm8ZprYT_-UXvqhWD8Iq9q2Os", # This API Key de-activated
                )
    print(client.get_collections())
    # 2. Store couple's resume PDFs
    pdf_folder = "data/raw/resumes"  # Put your PDF files here
    resumes_text, resume_payloads, resume_ids = store_resumes_in_qdrant(pdf_folder, client, collection_name)
    # 3. Retrieve: Enter job description and find best matching resume(s)
    user_job_desc = input("Enter job description or paste one: ")
    results = retrieve_closest_resumes(user_job_desc, client, collection_name, top_k=1)
    if results:
        closest_resume = results[0].payload
        print(f"\nBest-matching resume file: {closest_resume['filename']}\n")
        print(f"EXCERPT: {closest_resume['text'][:400]}...\n")
        # 4. (Optional) Recommend jobs for this resume
        job_descriptions = [
            "Senior Python developer with cloud experience.",
            "Data analyst with SQL and Power BI experience.",
            "AWS cloud engineer for DevOps automation projects."
            # Extend as needed
        ]
        print("Job description suggestions for this resume:")
        suggested_jobs = recommend_jobs_for_resume(closest_resume['text'], job_descriptions, top_k=2)
        for j in suggested_jobs:
            print(" -", j)
    else:
        print("No matching resumes found.")

if __name__ == "__main__":
    main()