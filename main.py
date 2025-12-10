from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import sys

app = FastAPI()

# Ensure prints are flushed immediately
print("Loading sentence transformer model...", flush=True)
# Same model as in embed_items.py
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!", flush=True)

@app.on_event("startup")
async def startup_event():
    print("FastAPI server is starting up...", flush=True)
    print("Server will be available at http://127.0.0.1:8000", flush=True)

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list[float]

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    print(f"Embedding text: {req.text}", flush=True)
    emb = model.encode(req.text, normalize_embeddings=True)
    print(f"Generated embedding of length {len(emb)}", flush=True)
    return EmbedResponse(embedding=emb.tolist())
