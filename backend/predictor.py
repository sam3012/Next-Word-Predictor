"""Next-word generation matching the notebook, plus top-k for the next token."""

from __future__ import annotations

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_next_words(model, tokenizer, max_len: int, seed_text: str, num_words_to_generate: int = 10) -> str:
    output_text = seed_text
    words = seed_text.split()

    for _ in range(num_words_to_generate):
        tokenized = tokenizer.texts_to_sequences([output_text])
        if not tokenized or not tokenized[0]:
            break

        padded = pad_sequences(tokenized, maxlen=max_len - 1, padding="pre", truncating="pre")
        preds = model.predict(padded, verbose=0)[0]

        # Greedy decoding stalls on "the the the"; step down the ranking until
        # the candidate is not an immediate repeat of the last word.
        previous = words[-1].lower() if words else ""
        next_word = ""
        for pos in np.argsort(preds)[::-1][:5]:
            candidate = tokenizer.index_word.get(int(pos), "")
            if candidate and candidate != previous:
                next_word = candidate
                break

        if not next_word:
            break

        words.append(next_word)
        output_text += " " + next_word

    return output_text


def top_k_next_words(model, tokenizer, max_len: int, seed_text: str, top_k: int = 5) -> list[dict]:
    tokenized = tokenizer.texts_to_sequences([seed_text])
    if not tokenized or not tokenized[0]:
        return []

    padded = pad_sequences(tokenized, maxlen=max_len - 1, padding="pre", truncating="pre")
    preds = model.predict(padded, verbose=0)[0]
    k = max(1, min(int(top_k), preds.shape[0]))
    top_indices = np.argsort(preds)[-k:][::-1]

    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(int(idx), "")
        if not word:
            continue
        results.append({"word": word, "probability": float(preds[int(idx)])})
    return results
