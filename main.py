from __future__ import annotations

import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from item_classifier import ClassifierArtifacts, load_classifier, predict_from_pil

_ML_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ML_DIR.parent
# Uvicorn does not load .env; Spring does. Mirror typical keys from ml-api, api, or repo root.
for _env_path in (
    _ML_DIR / ".env",
    _REPO_ROOT / "fitfolio-api" / ".env",
    _REPO_ROOT / ".env",
):
    if _env_path.is_file():
        load_dotenv(_env_path, override=False)


app = FastAPI()

print("Loading sentence transformer model...", flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully!", flush=True)

print("Loading item image classifier...", flush=True)
_classifier: ClassifierArtifacts | None = load_classifier(_ML_DIR)
if _classifier is not None:
    print("Item classifier loaded (POST /classify-item-image).", flush=True)
else:
    print(
        "Item classifier skipped: place resnet_item_classifier.pt and label_classes.json under "
        "fitfolio-ml-api/notebooks/exports or fitfolio-ml-api/artifacts, or set CLASSIFIER_DIR.",
        flush=True,
    )

_openai_client: OpenAI | None = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


@app.on_event("startup")
async def startup_event():
    print("FastAPI server is starting up...", flush=True)
    print("Server will be available at http://127.0.0.1:8000", flush=True)
    if os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is set (generate-insights enabled).", flush=True)
    else:
        print(
            "OPENAI_API_KEY is missing: add it to fitfolio-ml-api/.env or "
            "fitfolio-api/.env, or export it before starting uvicorn.",
            flush=True,
        )


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


# --- Item image classifier (ResNet) ----------------------------------------


class ClassifyItemImageResponse(BaseModel):
    category: str
    gender: str
    color: str
    categoryConfidence: float = Field(ge=0.0, le=1.0)
    genderConfidence: float = Field(ge=0.0, le=1.0)
    colorConfidence: float = Field(ge=0.0, le=1.0)


@app.post("/classify-item-image", response_model=ClassifyItemImageResponse)
async def classify_item_image(file: UploadFile = File(...)):
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not loaded: missing weights or label_classes.json (see server logs).",
        )
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        img = Image.open(io.BytesIO(raw))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    out = predict_from_pil(_classifier, img)
    return ClassifyItemImageResponse(
        category=out["category"],
        gender=out["gender"],
        color=out["color"],
        categoryConfidence=out["categoryConfidence"],
        genderConfidence=out["genderConfidence"],
        colorConfidence=out["colorConfidence"],
    )


# --- Review insights (LLM) -------------------------------------------------


class ReviewIn(BaseModel):
    rating: float = Field(ge=0.0, le=10.0)
    title: str = ""
    text: str = ""
    fit: str = ""
    comfort: str = ""
    quality: str = ""
    wouldRecommend: bool = False


class GenerateInsightsRequest(BaseModel):
    item_id: str
    reviews: list[ReviewIn]


class GenerateInsightsResponse(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]
    themes: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


INSIGHTS_SCHEMA = {
    "name": "review_insights",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "1-2 sentence summary from reviews only"},
            "pros": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short phrases, max ~8 items",
            },
            "cons": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short phrases, max ~8 items",
            },
            "themes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recurring short phrases, max ~8 items",
            },
            "confidence": {"type": "number", "description": "0-1"},
        },
        "required": ["summary", "pros", "cons", "themes", "confidence"],
        "additionalProperties": False,
    },
}


def _reviews_block(req: GenerateInsightsRequest) -> str:
    parts = []
    for i, r in enumerate(req.reviews, start=1):
        parts.append(
            f"{i}. rating={r.rating}/10 | recommend={r.wouldRecommend} | "
            f"fit={r.fit} | comfort={r.comfort} | quality={r.quality}\n"
            f"   title: {r.title}\n   text: {r.text}"
        )
    return "\n\n".join(parts)


SYSTEM_PROMPT = """You aggregate clothing item reviews for shoppers.
Rules:
- Use ONLY information supported by the review texts and structured fields below. Do not invent materials, sizing, colors, or brand claims.
- summary: 1-2 concise sentences.
- pros, cons, themes: short phrases (not full sentences), no duplicates, grounded in the reviews.
- Focus on fit, comfort, quality, durability, value, care, and real use when mentioned.
- confidence: float 0-1. Lower if reviews are sparse, mostly empty text, or strongly conflicting."""


@app.post("/generate-insights", response_model=GenerateInsightsResponse)
def generate_insights(req: GenerateInsightsRequest):
    if len(req.reviews) < 5:
        raise HTTPException(status_code=400, detail="At least 5 reviews are required")
    client = get_openai()
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    user_content = (
        f"item_id={req.item_id}\n\n"
        f"Reviews ({len(req.reviews)}):\n\n" + _reviews_block(req)
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_schema", "json_schema": INSIGHTS_SCHEMA},
    )
    raw = completion.choices[0].message.content
    if not raw:
        raise HTTPException(status_code=502, detail="Empty LLM response")
    try:
        data = json.loads(raw)
        return GenerateInsightsResponse.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid LLM JSON: {e}") from e
