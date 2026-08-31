"""Tokenizer, n-gram dataset, LSTM training, and artifact save/load."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"

# Longest history the model reads. Wikipedia sentences run 40+ tokens, and
# padding every n-gram out to the longest one leaves the LSTM chewing on zeros.
CONTEXT_WINDOW = 20
EMBED_DIM = 100
LSTM_UNITS = 150


def ensure_nltk() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    return slug[:80] or "untitled"


def artifact_dir(title: str) -> Path:
    return ARTIFACTS_ROOT / slugify(title)


def artifacts_exist(title: str) -> bool:
    directory = artifact_dir(title)
    return (
        (directory / "model.keras").is_file()
        and (directory / "tokenizer.pkl").is_file()
        and (directory / "config.json").is_file()
    )


def list_saved_models() -> list[dict]:
    """Return metadata for every complete artifact folder under artifacts/."""
    if not ARTIFACTS_ROOT.is_dir():
        return []

    models: list[dict] = []
    for directory in sorted(ARTIFACTS_ROOT.iterdir()):
        config_path = directory / "config.json"
        if not (
            directory.is_dir()
            and (directory / "model.keras").is_file()
            and (directory / "tokenizer.pkl").is_file()
            and config_path.is_file()
        ):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        num_classes = int(config.get("num_classes") or 1)
        models.append(
            {
                "slug": directory.name,
                "title": str(config.get("title") or directory.name),
                "vocab_size": num_classes - 1,
                "max_len": int(config.get("max_len") or 0),
                "epochs": int(config.get("epochs") or 0),
            }
        )
    return models


def _ngrams(sequences: list[list[int]], window: int) -> list[list[int]]:
    """Sliding-window n-grams so long sentences keep a full history."""
    grams: list[list[int]] = []
    for seq in sequences:
        for end in range(1, len(seq)):
            start = max(0, end + 1 - window)
            grams.append(seq[start : end + 1])
    return grams


def _build_sequences(content: str, context_window: int = CONTEXT_WINDOW) -> dict:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([content])
    num_classes = len(tokenizer.word_index) + 1
    if num_classes < 2:
        raise ValueError("The article produced no tokens.")

    sentences = sent_tokenize(content)
    tokenized = [seq for seq in tokenizer.texts_to_sequences(sentences) if len(seq) > 1]
    if not tokenized:
        raise ValueError("Not enough tokenized text to build training sequences.")

    window = context_window + 1
    grams = _ngrams(tokenized, window)
    if not grams:
        raise ValueError("Not enough tokenized text to build training sequences.")

    max_len = max(len(gram) for gram in grams)
    padded = pad_sequences(grams, maxlen=max_len, padding="pre", truncating="pre")
    return {
        "tokenizer": tokenizer,
        "X": padded[:, :-1],
        "y": padded[:, -1],
        "max_len": max_len,
        "num_classes": num_classes,
    }


def _build_model(num_classes: int, max_len: int):
    """Embedding, one LSTM, softmax. `mask_zero` keeps the padding out of the LSTM."""
    return Sequential(
        [
            Input(shape=(max_len - 1,)),
            Embedding(num_classes, EMBED_DIM, mask_zero=True),
            LSTM(LSTM_UNITS),
            Dense(num_classes, activation="softmax"),
        ]
    )


def train_and_save(
    content: str,
    title: str,
    epochs: int = 20,
    batch_size: int = 64,
) -> dict:
    """Train a dynamic LSTM on raw article text and persist artifacts."""
    ensure_nltk()
    data = _build_sequences(content)
    tokenizer = data["tokenizer"]
    max_len = data["max_len"]
    num_classes = data["num_classes"]

    model = _build_model(num_classes, max_len)
    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )
    history = model.fit(
        data["X"],
        data["y"],
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
    )

    directory = artifact_dir(title)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.keras")
    with open(directory / "tokenizer.pkl", "wb") as handle:
        pickle.dump(tokenizer, handle)

    accuracy = history.history.get("accuracy")
    config = {
        "title": title,
        "max_len": int(max_len),
        "num_classes": int(num_classes),
        "epochs": int(epochs),
        "context_window": int(max_len - 1),
        "accuracy": round(float(accuracy[-1]), 4) if accuracy else None,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def load_artifacts(title: str) -> tuple[object, Tokenizer, dict]:
    directory = artifact_dir(title)
    model = load_model(directory / "model.keras")
    with open(directory / "tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return model, tokenizer, config
