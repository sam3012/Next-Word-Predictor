# Next Word Predictor

A small language-model playground: search Wikipedia for an article, train a compact **LSTM** on that page’s raw text, then continue a sentence in the style of the topic. Built with **FastAPI** (training + inference) and **Streamlit** (UI). Each model lives on a single article — not the whole internet — so it stays small and trains in a few minutes on CPU.

---

## Features

- **Wikipedia search**: type a keyword (`earth`, `coding`, `space`) and pick an exact title from MediaWiki full-text search.
- **Per-article LSTM**: trains `Embedding → LSTM → Dense(softmax)` on the chosen page’s raw `page.text` (no preprocessing).
- **Saved models**: artifacts persist under `artifacts/<slug>/` so you can reload a topic without retraining.
- **Next-word generation**: seed a phrase, generate a continuation, and inspect the top-k next-token probabilities.
- **Dark Streamlit UI**: custom theme pinned in `.streamlit/config.toml`.

---

## How the Model Works

1. **Fetch** the Wikipedia article by exact title via `wikipedia-api`.
2. **Tokenize** with NLTK sentence splitting (`punkt`) and a Keras `Tokenizer` over the full article vocabulary.
3. **Build n-grams** with a sliding **20-token context window**. Each example is “history so far → next word.”
4. **Train** a Sequential Keras model:

   `Embedding(vocab, 100, mask_zero=True)` → `LSTM(150)` → `Dense(vocab, softmax)`

   - Adam at `lr=0.01`, batch size `64`, 5–30 epochs (default 20)
   - Integer labels with `SparseCategoricalCrossentropy`
   - Full article, no validation split — for a one-page model, fitting the page is the goal
5. **Generate** by padding the seed to `max_len - 1` and repeatedly taking the top candidate that is not an immediate repeat of the previous word.

The model only knows words that appear in that article. Seed text using out-of-vocab words will produce empty or weak predictions.

---

## Tech Stack

- **API**: Python, FastAPI, Uvicorn, Pydantic
- **UI**: Streamlit
- **Model**: TensorFlow / Keras LSTM
- **NLP**: NLTK (`punkt`, `punkt_tab`), Keras `Tokenizer`
- **Data**: Wikipedia (MediaWiki search + `wikipedia-api` page text)
- **Serialization**: Keras `.keras` model, Pickle tokenizer, JSON config

**Python 3.13+** (`requires-python` in `pyproject.toml`).

---

## Project Structure

```text
next-word-predictor/
│
├── backend/
│   ├── main.py              # FastAPI: search, train, load, predict
│   ├── wikipedia.py         # MediaWiki search + page fetch
│   ├── train.py             # Tokenizer, n-grams, LSTM, save/load
│   └── predictor.py         # Continuation + top-k next words
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── artifacts/               # Generated (gitignored): one folder per article
│   └── <slug>/
│       ├── model.keras
│       ├── tokenizer.pkl
│       └── config.json
│
├── .streamlit/
│   └── config.toml          # Dark theme, minimal toolbar
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/sam3012/Next-Word-Predictor.git
cd next-word-predictor
```

### 2. Create a virtual environment

**With [uv](https://docs.astral.sh/uv/) (recommended):**

```bash
uv sync
```

**With pip:**

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

pip install -e .
```

TensorFlow is a large install. First run also downloads NLTK `punkt` / `punkt_tab` automatically.

### 3. Optional: Wikipedia User-Agent

Wikimedia throttles generic clients. If search returns **HTTP 429**, set a User-Agent with a real contact address:

```bash
export NWP_USER_AGENT="NextWordPredictor/1.0 (you@example.com)"
```

Windows (PowerShell):

```powershell
$env:NWP_USER_AGENT = "NextWordPredictor/1.0 (you@example.com)"
```

---

## Run

Start both processes from the **project root** so `backend.main` imports resolve.

**Terminal 1 — API** (port 8000):

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — UI** (port 8501):

```bash
uv run streamlit run frontend/app.py
```

If you activated a venv and installed with pip, drop `uv run`:

```bash
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

Open Streamlit at [http://localhost:8501](http://localhost:8501). The sidebar API URL defaults to `http://127.0.0.1:8000`.

---

## Usage

1. **Search** — enter a keyword and click **Search**. Pick an exact Wikipedia title from the list.
2. **Train** — set epochs (5–30, default 20). Leave **Retrain** unchecked to reuse a cached `artifacts/<slug>/` model if it already exists. Click **Train model**. A 20–30 epoch run on a long article can take a few minutes on CPU.
3. **Predict** — load a saved model from the sidebar (or use the one you just trained). Enter seed text, how many words to generate, and top-k. Click **Generate**.

Example seed after training on *Earth*: `Earth is`

---

## API

| Method | Path | Body | Notes |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | — | Status, active title, saved-model count |
| `GET` | `/models` | — | List saved artifacts + active title |
| `POST` | `/models/select` | `{ "title" }` | Load a saved model into memory |
| `POST` | `/search` | `{ "query", "limit" }` | MediaWiki `list=search` |
| `POST` | `/train` | `{ "title", "epochs", "retrain" }` | Fetch page, train, cache under `artifacts/` |
| `POST` | `/predict` | `{ "text", "num_words", "top_k" }` | Continuation + top-k (400 if nothing is active) |

`/search` returns **502** when Wikipedia itself fails, and **200** with an empty `titles` list only when the query has no matches.

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Training details

| Setting | Value |
| :--- | :--- |
| Input | Wikipedia `page.text` as-is |
| Vocabulary | Full article (`num_classes = vocab_size + 1`) |
| Context | Sliding window of 20 tokens |
| Architecture | Embedding (100-d, `mask_zero`) → LSTM (150) → Softmax |
| Optimizer | Adam, learning rate `0.01` |
| Batch size | 64 |
| Epochs | 20 default (slider 5–30) |
| Artifacts | `model.keras`, `tokenizer.pkl`, `config.json` |

---

## Future Improvements

- [ ] Optional sampling temperature instead of greedy (near-greedy) decoding
- [ ] Deploy API + UI (e.g. Render, Railway, or Hugging Face Spaces)
- [ ] Progress streaming so the UI can show epoch logs while training
