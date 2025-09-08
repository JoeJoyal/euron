import numpy as np
import faiss
import requests

# --- Configuration ---
EMBEDDING_DIM = 1536
EURI_MODEL = "text-embedding-3-small"
EURI_API_URL = "https://api.euron.one/api/v1/euri/embeddings"
EURI_API_KEY = "Bearer euri-6897392f64d349a7fdd3b577e63125855b1a3abd931fa3369add8bedb41498fb" # This API Key de-activated

# --- Embedding Function ---
def generate_embedding_euri(text, model=EURI_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": EURI_API_KEY
    }
    payload = {"input": text, "model": model}
    try:
        response = requests.post(EURI_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                return np.array(data["data"][0]["embedding"], dtype=np.float32)
        print(f"[Embedding Error] {response.text}")
        return None
    except Exception as e:
        print(f"[Exception] {e}")
        return None

# --- Article Embedding and Indexing ---
def embed_articles(articles):
    embeddings, valid_articles = [], []
    for idx, art in enumerate(articles):
        text = (art["title"] + " " + art["content"]).strip()
        emb = generate_embedding_euri(text)
        if emb is not None:
            embeddings.append(emb)
            valid_articles.append(art)
        else:
            print(f"[Warning] Skipped article idx={idx} due to embedding failure.")
    if not embeddings:
        raise ValueError("No article embeddings created!")
    embeddings = np.vstack(embeddings)
    return embeddings, valid_articles

def build_faiss_cosine_index(embeddings):
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / np.clip(norms, 1e-8, None)
    idx = faiss.IndexFlatL2(norm_embeddings.shape[1])
    idx.add(norm_embeddings)
    return idx, norm_embeddings

# --- Recommendation Query ---
def recommend_news(query, index, articles, top_k=5):
    q_emb = generate_embedding_euri(query)
    if q_emb is None:
        print("Embedding error for query.")
        return []
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    D, I = index.search(q_norm.reshape(1, -1), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({"score": float(score), "article": articles[idx]})
    return results

# --- Pipeline ---
def main():
    # 1. Your in-memory article dataset
    articles = [
        {
            "title": "Big Tech’s AI Obsession is shaking Wall Street",
            "content": (
                "The AI bubble is stretching like a balloon one pin away from a pop and that has left Wall Street jittery for the last 18 months. "
                "Tech stocks were under pressure this week as Wall Street's AI enthusiasm slowed and investors adjusted portfolios after a strong summer rally. "
                "An AI-infused rally is once again sending the S&P 500 to record highs. "
                "Wall Street experts say the recent stocks surge is fueled by AI optimism and expectation the Fed will cut interest rates. "
                "U.S. tech stocks slide after Altman warns of 'bubble' in AI and MIT study doubts the hype. "
                "Nvidia dropped 3.5% and Palantir nearly 10% after an MIT study claimed 95% of companies investing in generative AI see no returns, "
                "and was potentially deepened by earlier comments from OpenAI’s Sam Altman suggesting investors may be caught in an AI bubble. "
                "The IMF notes that regulators will need new tools as AI transforms capital markets and impacts financial stability more broadly."
                "This downturn follows a period of strong gains fueled by AI enthusiasm, with major tech stocks like Nvidia, Microsoft, and Palantir experiencing steep declines, wiping out over $1 trillion in market value."
                "The sell-off is driven by growing concerns over the sustainability of the AI boom, highlighted by a blunt MIT study revealing that 95% of AI pilot projects fail, and comments from OpenAI's CEO Sam Altman suggesting the market may be overexcited."
                "In the week leading up to August 27, 2025, tech stocks faced significant pressure. The Nasdaq Composite fell 0.67% on August 20, following a 1.46% drop the previous day, and the S&P 500 posted its fourth consecutive day of losses."
                "Nvidia, a central figure in the AI boom, saw its share price drop from $182 to $169 within a single trading day.Palantir, another AI-focused stock, slumped nearly 9% over five days."
            )
        }
    ]

    # 2. Embed and index articles
    print("Embedding and indexing articles...")
    embeddings, valid_articles = embed_articles(articles)
    faiss_index, _ = build_faiss_cosine_index(embeddings)

    # 3. Real-time query
    query = "Recent Market Selloff"
    recommendations = recommend_news(query, faiss_index, valid_articles, top_k=3)
    print(f"\nTop results for: '{query}'")
    for res in recommendations:
        art = res["article"]
        print(f"\nScore: {res['score']:.3f} | Title: {art['title']}")
        print(art['content'][:500] + ("..." if len(art['content']) > 500 else ""))

if __name__ == "__main__":
    main()