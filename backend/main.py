"""FastAPI: search Wikipedia titles, train LSTM, predict next words."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.predictor import predict_next_words, top_k_next_words
from backend.train import artifacts_exist, list_saved_models, load_artifacts, train_and_save
from backend.wikipedia import (
    WikipediaFetchError,
    WikipediaSearchError,
    fetch_page,
    search_wikipedia,
)

app = FastAPI(title="Next Word Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_active: dict[str, Any] = {
    "title": None,
    "model": None,
    "tokenizer": None,
    "config": None,
}


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=6, ge=1, le=20)


class TrainRequest(BaseModel):
    title: str
    epochs: int = Field(default=20, ge=5, le=30)
    retrain: bool = False


class PredictRequest(BaseModel):
    text: str
    num_words: int = Field(default=10, ge=1, le=50)
    top_k: int = Field(default=5, ge=1, le=20)


class SelectModelRequest(BaseModel):
    title: str


def _set_active(title: str, model, tokenizer, config: dict) -> None:
    _active["title"] = title
    _active["model"] = model
    _active["tokenizer"] = tokenizer
    _active["config"] = config


def _model_payload(title: str, config: dict, cached: bool | None = None) -> dict:
    payload = {
        "title": config.get("title", title),
        "vocab_size": int(config["num_classes"]) - 1,
        "max_len": int(config["max_len"]),
        "num_classes": int(config["num_classes"]),
        "epochs": int(config.get("epochs") or 0),
        "accuracy": config.get("accuracy"),
    }
    if cached is not None:
        payload["cached"] = cached
    return payload


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "active_title": _active["title"],
        "saved_models": len(list_saved_models()),
    }


@app.get("/models")
def models() -> dict:
    return {
        "models": list_saved_models(),
        "active_title": _active["title"],
    }


@app.post("/models/select")
def select_model(body: SelectModelRequest) -> dict:
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be empty.")
    if not artifacts_exist(title):
        raise HTTPException(
            status_code=404,
            detail=f"No saved model found for '{title}'. Train it first.",
        )
    try:
        model, tokenizer, config = load_artifacts(title)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc

    _set_active(title, model, tokenizer, config)
    return _model_payload(title, config)


@app.post("/search")
def search(body: SearchRequest) -> dict:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        titles = search_wikipedia(query, limit=body.limit)
    except WikipediaSearchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected search failure ({type(exc).__name__}): {exc}",
        ) from exc

    return {"query": query, "titles": titles}


@app.post("/train")
def train(body: TrainRequest) -> dict:
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be empty.")

    cached = False
    try:
        if artifacts_exist(title) and not body.retrain:
            model, tokenizer, config = load_artifacts(title)
            cached = True
        else:
            content = fetch_page(title)
            config = train_and_save(content, title=title, epochs=body.epochs)
            model, tokenizer, config = load_artifacts(title)
    except WikipediaFetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed ({type(exc).__name__}): {exc}",
        ) from exc

    _set_active(title, model, tokenizer, config)
    return _model_payload(title, config, cached=cached)


@app.post("/predict")
def predict(body: PredictRequest) -> dict:
    if _active["model"] is None or _active["tokenizer"] is None or _active["config"] is None:
        raise HTTPException(
            status_code=400,
            detail="No model is active. Search for an article and train first.",
        )

    seed = body.text.strip()
    if not seed:
        raise HTTPException(status_code=400, detail="Seed text must not be empty.")

    max_len = int(_active["config"]["max_len"])
    continuation = predict_next_words(
        _active["model"],
        _active["tokenizer"],
        max_len,
        seed,
        num_words_to_generate=body.num_words,
    )
    top_k = top_k_next_words(
        _active["model"],
        _active["tokenizer"],
        max_len,
        seed,
        top_k=body.top_k,
    )
    return {
        "title": _active["title"],
        "seed": seed,
        "continuation": continuation,
        "top_k": top_k,
    }
