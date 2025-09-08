import os
import requests
import numpy as np
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("EURI_API_KEY") # Use env var for security

def generate_embeddings(text, model="text-embedding-3-small"):
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": model
    }

    response = requests.post(url, headers=headers, json=payload)
    return np.array(response.json()['data'][0]['embedding'])

# def generate_embeddings(text, model="text-embedding-3-small"):
#     url = "https://api.euron.one/api/v1/euri/embeddings"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     payload = {"input": text,"model": model}
#     response = requests.post(url, headers=headers, json=payload)
#     return np.array(response.json()['data'][0]['embedding'])