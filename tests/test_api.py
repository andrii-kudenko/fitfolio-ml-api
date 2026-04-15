"""HTTP contract tests for fitfolio-ml-api (embed + generate-insights guards)."""

import os

import pytest

EXPECTED_MINILM_DIM = 384


def test_embed_returns_normalized_vector_of_expected_dimension(client):
    res = client.post("/embed", json={"text": "Nike Air Max running shoes"})
    assert res.status_code == 200
    data = res.json()
    assert "embedding" in data
    emb = data["embedding"]
    assert len(emb) == EXPECTED_MINILM_DIM
    assert all(isinstance(x, float) for x in emb)
    # normalize_embeddings=True => L2 norm ~ 1
    norm = sum(x * x for x in emb) ** 0.5
    assert 0.99 < norm < 1.01


def test_embed_rejects_invalid_json(client):
    res = client.post("/embed", json={})
    assert res.status_code == 422


def test_generate_insights_rejects_fewer_than_five_reviews(client):
    payload = {
        "item_id": "00000000-0000-0000-0000-000000000001",
        "reviews": [
            {
                "rating": 8.0,
                "title": "A",
                "text": "one",
                "fit": "",
                "comfort": "",
                "quality": "",
                "wouldRecommend": True,
            },
            {
                "rating": 7.0,
                "title": "B",
                "text": "two",
                "fit": "",
                "comfort": "",
                "quality": "",
                "wouldRecommend": True,
            },
            {
                "rating": 6.0,
                "title": "C",
                "text": "three",
                "fit": "",
                "comfort": "",
                "quality": "",
                "wouldRecommend": False,
            },
            {
                "rating": 8.0,
                "title": "D",
                "text": "four",
                "fit": "",
                "comfort": "",
                "quality": "",
                "wouldRecommend": True,
            },
        ],
    }
    res = client.post("/generate-insights", json=payload)
    assert res.status_code == 400
    assert "5" in res.json().get("detail", "")


def test_generate_insights_returns_503_when_openai_key_missing(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Ensure client does not reuse a cached OpenAI instance from another test module
    import main as app_main

    app_main._openai_client = None

    reviews = [
        {
            "rating": 8.0,
            "title": f"R{i}",
            "text": f"Review text {i}",
            "fit": "",
            "comfort": "",
            "quality": "",
            "wouldRecommend": True,
        }
        for i in range(5)
    ]
    payload = {"item_id": "00000000-0000-0000-0000-000000000002", "reviews": reviews}
    res = client.post("/generate-insights", json=payload)
    assert res.status_code == 503
    assert "OPENAI_API_KEY" in res.json().get("detail", "")


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY for live LLM integration test",
)
def test_generate_insights_live_openai_returns_structured_response(client):
    reviews = [
        {
            "rating": 8.0,
            "title": "Great",
            "text": "Soft fabric and comfortable fit.",
            "fit": "",
            "comfort": "",
            "quality": "",
            "wouldRecommend": True,
        },
        {
            "rating": 7.0,
            "title": "Good",
            "text": "Runs slightly large in the shoulders.",
            "fit": "",
            "comfort": "",
            "quality": "",
            "wouldRecommend": True,
        },
        {
            "rating": 6.0,
            "title": "OK",
            "text": "Color faded a bit after washing.",
            "fit": "",
            "comfort": "",
            "quality": "",
            "wouldRecommend": False,
        },
        {
            "rating": 8.0,
            "title": "Nice",
            "text": "Lightweight for summer days.",
            "fit": "",
            "comfort": "",
            "quality": "",
            "wouldRecommend": True,
        },
        {
            "rating": 7.0,
            "title": "Value",
            "text": "Fair price for the quality.",
            "fit": "",
            "comfort": "",
            "quality": "",
            "wouldRecommend": True,
        },
    ]
    payload = {"item_id": "00000000-0000-0000-0000-000000000099", "reviews": reviews}
    res = client.post("/generate-insights", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "summary" in data and data["summary"].strip()
    assert isinstance(data.get("pros"), list)
    assert isinstance(data.get("cons"), list)
    assert isinstance(data.get("themes"), list)
    assert 0.0 <= float(data.get("confidence", -1)) <= 1.0
