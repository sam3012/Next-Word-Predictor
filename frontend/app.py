"""Streamlit UI: search Wikipedia titles, train LSTM, predict next words."""

from __future__ import annotations

from html import escape

import requests
import streamlit as st

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Outfit:wght@400;500;600&display=swap");

    html, body, [class*="css"] { font-family: "Outfit", sans-serif; }

    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, #2a2118 0%, transparent 55%),
            radial-gradient(900px 400px at 100% 0%, #1a2a28 0%, transparent 50%),
            #14110f;
        color: #f3eadb;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c1814 0%, #12100e 100%);
        border-right: 1px solid rgba(232, 197, 132, 0.12);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f3eadb !important;
    }

    h1, h2, h3 {
        font-family: "Fraunces", Georgia, serif !important;
        color: #f7efe2 !important;
        letter-spacing: -0.03em;
    }

    .hero-kicker {
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #e8c584;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-family: "Fraunces", Georgia, serif;
        font-size: 2.35rem;
        font-weight: 700;
        line-height: 1.15;
        color: #fff8ee;
        margin: 0 0 0.45rem 0;
    }
    .hero-sub {
        color: #c9bba8;
        font-size: 1.02rem;
        max-width: 42rem;
        margin: 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 248, 238, 0.04);
        border: 1px solid rgba(232, 197, 132, 0.16) !important;
        border-radius: 18px;
        padding: 0.35rem 0.15rem;
    }

    .panel-label {
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #e8c584;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .continuation {
        font-family: "Fraunces", Georgia, serif;
        font-size: 1.28rem;
        line-height: 1.55;
        color: #fff8ee;
        background: rgba(232, 197, 132, 0.08);
        border-left: 3px solid #e8c584;
        padding: 1rem 1.15rem;
        border-radius: 0 14px 14px 0;
        margin: 0.4rem 0 1rem 0;
    }
    .meta-chip {
        display: inline-block;
        background: rgba(232, 197, 132, 0.12);
        color: #e8c584;
        border: 1px solid rgba(232, 197, 132, 0.22);
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }
    .sidebar-brand {
        font-family: "Fraunces", Georgia, serif;
        font-size: 1.35rem;
        color: #fff8ee;
        margin: 0 0 0.15rem 0;
    }
    .sidebar-hint {
        color: #b8aa98 !important;
        font-size: 0.85rem;
        margin-bottom: 1.1rem;
    }

    .stButton > button {
        border-radius: 999px;
        font-weight: 600;
        letter-spacing: 0.01em;
        border: 1px solid rgba(232, 197, 132, 0.35);
        background: #e8c584;
        color: #1c140c;
    }
    .stButton > button:hover {
        background: #f3d7a0;
        border-color: #f3d7a0;
        color: #1c140c;
    }
    .stButton > button[kind="secondary"] {
        background: transparent;
        color: #f3eadb;
        border-color: rgba(243, 234, 219, 0.25);
    }

    [data-testid="stMetricValue"] { color: #e8c584; font-family: "Fraunces", Georgia, serif; }
    [data-testid="stMetricLabel"] { color: #c9bba8; }

    textarea, .stTextInput input, .stNumberInput input {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_API = "http://127.0.0.1:8000"

for key, default in (
    ("titles", []),
    ("train_result", None),
    ("active_title", None),
    ("predict_result", None),
    ("loaded_model_title", None),
    ("saved_models", []),
    ("model_choice", None),
    ("models_error", None),
    ("search_query", ""),
):
    if key not in st.session_state:
        st.session_state[key] = default


class ApiError(RuntimeError):
    """An API call that failed, carrying enough context to debug it."""

    def __init__(
        self,
        message: str,
        *,
        method: str,
        url: str,
        status: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.method = method
        self.url = url
        self.status = status
        self.body = body

    def details(self) -> str:
        lines = [f"{self.method} {self.url}"]
        if self.status is not None:
            lines.append(f"HTTP status: {self.status}")
        lines.append(f"Message: {self.message}")
        if self.body:
            lines.append(f"Response body:\n{self.body}")
        return "\n".join(lines)


def show_error(exc: Exception, context: str) -> None:
    """Render the exact failure, with the raw request/response behind a toggle."""
    if isinstance(exc, ApiError):
        st.error(f"{context}: {exc.message}")
        with st.expander("Error details"):
            st.code(exc.details(), language="text")
    else:
        st.error(f"{context}: {type(exc).__name__}: {exc}")


def _api_call(method: str, api_url: str, path: str, timeout: int, payload: dict | None) -> dict:
    url = f"{api_url}{path}"
    try:
        if payload is None:
            response = requests.get(url, timeout=timeout)
        else:
            response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(
            f"could not reach the API ({type(exc).__name__}): {exc}",
            method=method,
            url=url,
        ) from exc

    if response.status_code >= 400:
        detail = response.text
        try:
            payload_json = response.json()
            if isinstance(payload_json, dict):
                detail = str(payload_json.get("detail") or response.text)
        except ValueError:
            pass
        raise ApiError(
            detail.strip() or f"HTTP {response.status_code}",
            method=method,
            url=url,
            status=response.status_code,
            body=response.text[:2000],
        )

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError(
            "API returned a non-JSON response.",
            method=method,
            url=url,
            status=response.status_code,
            body=response.text[:2000],
        ) from exc


def api_get(api_url: str, path: str, timeout: int = 15) -> dict:
    return _api_call("GET", api_url, path, timeout, None)


def api_post(api_url: str, path: str, payload: dict, timeout: int = 30) -> dict:
    return _api_call("POST", api_url, path, timeout, payload)


def fetch_saved_models(api_url: str) -> list[dict]:
    try:
        data = api_get(api_url, "/models", timeout=8)
    except Exception as exc:
        st.session_state.models_error = (
            exc.details() if isinstance(exc, ApiError) else f"{type(exc).__name__}: {exc}"
        )
        return st.session_state.saved_models or []

    st.session_state.models_error = None
    models = data.get("models") or []
    st.session_state.saved_models = models
    if data.get("active_title") and not st.session_state.active_title:
        st.session_state.active_title = data.get("active_title")
    return models


def activate_model(api_url: str, title: str) -> dict:
    result = api_post(api_url, "/models/select", {"title": title}, timeout=120)
    st.session_state.active_title = result.get("title")
    st.session_state.loaded_model_title = result.get("title")
    st.session_state.train_result = result
    st.session_state.predict_result = None
    return result


with st.sidebar:
    st.markdown('<p class="sidebar-brand">✦ Word studio</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-hint">Pick a saved LSTM and generate, or train a new one from Wikipedia.</p>',
        unsafe_allow_html=True,
    )

    api_url = st.text_input("API URL", value=DEFAULT_API).rstrip("/")
    saved_models = fetch_saved_models(api_url)
    model_titles = [m["title"] for m in saved_models]

    st.markdown("#### Saved models")
    if model_titles:
        preferred = st.session_state.model_choice or st.session_state.active_title
        default_index = model_titles.index(preferred) if preferred in model_titles else 0
        selected_saved = st.selectbox(
            "Choose a trained model",
            model_titles,
            index=default_index,
            help="Models already trained and saved under artifacts/. Selecting one loads it for prediction.",
        )
        st.session_state.model_choice = selected_saved
        meta = next((m for m in saved_models if m["title"] == selected_saved), None)
        if meta:
            st.caption(
                f"vocab {meta['vocab_size']}  ·  context {max(meta['max_len'] - 1, 0)} words"
                f"  ·  {meta['epochs']} epochs"
            )

        already_active = selected_saved == st.session_state.active_title
        load_label = "Loaded" if already_active else "Load model"
        if st.button(load_label, disabled=already_active, use_container_width=True):
            with st.spinner(f"Loading '{selected_saved}'…"):
                try:
                    activate_model(api_url, selected_saved)
                    st.rerun()
                except Exception as exc:
                    show_error(exc, f"Loading '{selected_saved}' failed")
    else:
        st.info("No saved models yet. Search and train an article to create one.")

    if st.session_state.models_error:
        st.warning("Could not read the saved model list.")
        with st.expander("Error details"):
            st.code(st.session_state.models_error, language="text")

    active = st.session_state.active_title
    if active:
        st.markdown(
            f'<span class="meta-chip">Active · {active}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No model active.")

    with st.expander("API health"):
        if st.button("Check health"):
            try:
                data = api_get(api_url, "/health", timeout=5)
                st.session_state.active_title = data.get("active_title")
                st.success(
                    f"API ok · active: {data.get('active_title') or 'none'} · "
                    f"saved: {data.get('saved_models', 0)}"
                )
            except Exception as exc:
                show_error(exc, "Health check failed")


st.markdown(
    """
    <div class="hero-kicker">Wikipedia · LSTM · next-word</div>
    <h1 class="hero-title">Write the first words. The model continues.</h1>
    <p class="hero-sub">
        Search an article, train a small language model on its raw text, then generate a continuation.
        Saved models stay on disk — load them from the left panel anytime.
    </p>
    """,
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns((1.05, 1), gap="large")

with left:
    with st.container(border=True):
        st.markdown('<p class="panel-label">01 · Find an article</p>', unsafe_allow_html=True)
        st.subheader("Search Wikipedia")
        query = st.text_input(
            "Keyword",
            placeholder="coding  /  artificial  /  space",
            label_visibility="collapsed",
        )
        if st.button("Search", type="primary"):
            term = query.strip() or "coding"
            try:
                data = api_post(api_url, "/search", {"query": term, "limit": 6}, timeout=15)
                st.session_state.titles = data.get("titles") or []
                st.session_state.predict_result = None
                if not st.session_state.titles:
                    st.warning(
                        f"Wikipedia search succeeded but returned no articles for '{term}'."
                    )
            except Exception as exc:
                show_error(exc, f"Search for '{term}' failed")
                st.session_state.titles = []

        selected_title = None
        if st.session_state.titles:
            selected_title = st.radio(
                "Matching titles",
                st.session_state.titles,
                index=0,
            )
        else:
            st.caption("Search a keyword to list exact Wikipedia titles.")

    st.write("")
    with st.container(border=True):
        st.markdown('<p class="panel-label">02 · Train</p>', unsafe_allow_html=True)
        st.subheader("Fit an LSTM")
        epochs = st.slider("Epochs", min_value=5, max_value=30, value=20)
        retrain = st.checkbox("Retrain (ignore cached model)")
        train_disabled = selected_title is None
        if train_disabled:
            st.caption("Pick a title above to enable training.")

        if st.button("Train model", disabled=train_disabled, type="primary"):
            with st.spinner(
                f"Training on '{selected_title}' ({epochs} epochs). This can take several minutes…"
            ):
                try:
                    result = api_post(
                        api_url,
                        "/train",
                        {
                            "title": selected_title,
                            "epochs": epochs,
                            "retrain": retrain,
                        },
                        timeout=60 * 60,
                    )
                    st.session_state.train_result = result
                    st.session_state.active_title = result.get("title")
                    st.session_state.loaded_model_title = result.get("title")
                    st.session_state.predict_result = None
                    st.session_state.model_choice = result.get("title")
                    fetch_saved_models(api_url)
                    st.rerun()
                except Exception as exc:
                    show_error(exc, f"Training on '{selected_title}' failed")

        if st.session_state.train_result:
            result = st.session_state.train_result
            cached_note = "from disk" if result.get("cached") else "just trained"
            chips = [
                str(result.get("title")),
                f'vocab {result.get("vocab_size")}',
                f'context {int(result.get("max_len") or 1) - 1} words',
                cached_note,
            ]
            if result.get("accuracy") is not None:
                chips.append(f'acc {float(result["accuracy"]):.2f}')
            st.markdown(
                "".join(f'<span class="meta-chip">{escape(chip)}</span>' for chip in chips),
                unsafe_allow_html=True,
            )

with right:
    with st.container(border=True):
        st.markdown('<p class="panel-label">03 · Predict</p>', unsafe_allow_html=True)
        st.subheader("Continue a sentence")
        model_ready = bool(st.session_state.active_title)
        if model_ready:
            st.caption(f"Using **{st.session_state.active_title}**. The model only knows words from that article.")
        else:
            st.info("Load a saved model on the left, or train a new one, then predict.")

        seed = st.text_area("Seed text", placeholder="The history of", height=90)
        c1, c2 = st.columns(2)
        with c1:
            num_words = st.number_input("Words to generate", min_value=1, max_value=50, value=10)
        with c2:
            top_k = st.number_input("Top-k next words", min_value=1, max_value=20, value=5)

        if st.button("Generate", disabled=not model_ready, type="primary"):
            if not seed.strip():
                st.warning("Enter some starting words.")
            else:
                try:
                    st.session_state.predict_result = api_post(
                        api_url,
                        "/predict",
                        {
                            "text": seed.strip(),
                            "num_words": int(num_words),
                            "top_k": int(top_k),
                        },
                        timeout=60,
                    )
                except Exception as exc:
                    show_error(exc, "Prediction failed")

        if st.session_state.predict_result:
            pred = st.session_state.predict_result
            st.markdown("**Continuation**")
            st.markdown(
                f'<div class="continuation">{escape(str(pred.get("continuation", "")))}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Likely next words** (from the seed, before generation)")
            for item in pred.get("top_k") or []:
                word = item.get("word") or ""
                prob = float(item.get("probability") or 0)
                st.progress(min(max(prob, 0.0), 1.0), text=f"{word}  ·  {prob:.3f}")
