"""Professional, clean, and modern frontend for the Physics Book RAG system.
Run with: .venv/bin/python -m streamlit run app.py
"""
import json
from html import escape
import os
import threading
import time

import streamlit as st
from dotenv import dotenv_values

from answer_book import BookAnswer, DEFAULT_MODEL, build_request, retrieval_query
from answer_markdown import normalize_math_markdown
from embed_book import DATA, ROOT
from search_book import BookSearch
import notebook_ui as ui

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Physics Notebook · Stay curious",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# Notebook presentation
# ---------------------------------------------------------------------------
ui.load_styles()


# ---------------------------------------------------------------------------
# Backend Resource Caching & Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_search_instance(build_hash):
    """Load and cache the BookSearch instance and a thread lock."""
    return BookSearch(), threading.Lock()


def _secret(name, default=""):
    """Read from Streamlit Cloud secrets when available."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        return default
    return default if value is None else value


def load_app_settings():
    """Load config: Streamlit secrets → env vars → local .env."""
    local = dotenv_values(ROOT / ".env")
    api_key = (
        _secret("GROQ_API_KEY")
        or os.environ.get("GROQ_API_KEY", "")
        or local.get("GROQ_API_KEY", "")
        or ""
    )
    model = (
        _secret("GROQ_MODEL")
        or os.environ.get("GROQ_MODEL")
        or local.get("GROQ_MODEL")
        or DEFAULT_MODEL
    )
    return str(api_key).strip(), str(model).strip() or DEFAULT_MODEL


def execute_tutor_answer(question, top_k, api_key, model, build_hash, history=None):
    """Execute grounded RAG query through BookAnswer with thread safety."""
    search, lock = get_search_instance(build_hash)
    with lock:
        answerer = BookAnswer(api_key=api_key, model=model, search=search)
        return answerer.ask(question, top_k=top_k, history=history)


def execute_search_only(query, top_k, build_hash):
    """Retrieve nearest chunks without calling Groq API."""
    search, lock = get_search_instance(build_hash)
    with lock:
        return search.search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# App Initialization & State
# ---------------------------------------------------------------------------
api_key, default_model = load_app_settings()

index_error = None
manifest = {}
try:
    manifest_path = DATA / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception as err:
    index_error = f"The book index could not be loaded ({err}). Run embed_book.py first."

ready = bool(api_key) and not index_error

if "messages" not in st.session_state:
    st.session_state.messages = []
if "inspector_results" not in st.session_state:
    st.session_state.inspector_results = None

# ---------------------------------------------------------------------------
# Sidebar: Textbook Info & Testing Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    ui.brand()

    if st.button("＋  New notebook", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pop("failed_question", None)
        st.session_state.pop("preview_request", None)
        st.rerun()

    ui.library_card()

    with st.expander("Study preferences"):
        top_k = st.slider(
            "Passages per answer", min_value=1, max_value=10, value=5,
            help="Choose how many textbook passages to consult for each answer.",
        )
        model_choice = st.text_input("Groq model", value=default_model)
        preview_mode = st.checkbox(
            "Preview request without sending",
            help="Inspect the question and retrieved passages without calling Groq.",
        )

    with st.expander("Connection & index"):
        st.caption(f"Textbook passages: {manifest.get('chunks', 'Unavailable')}")
        st.caption(f"Embedding model: {manifest.get('model', 'Unavailable')}")
        st.caption(f"Answer service: {'Connected' if api_key else 'Key needed'}")
        if manifest.get("build_hash"):
            st.caption(f"Index build: {manifest['build_hash'][:12]}")

    st.html('<div class="sidebar-note"><em>A good place to wonder.</em>Ask freely. Follow the sources.<br>Make the ideas your own.</div>')

    if st.session_state.messages:
        transcript = "# Physics Notebook Conversation Transcript\n\n"
        for m in st.session_state.messages:
            transcript += f"### {m['role'].upper()}\n{m['text']}\n\n"
            if m.get("sources"):
                transcript += "**Cited Passages:**\n"
                for s in m["sources"]:
                    transcript += f"- Pages {s['page_start']}–{s['page_end']} ({s.get('section', 'Physics')}): {s['text'][:140]}...\n"
                transcript += "\n"
        st.download_button(
            "↓  Save your notes",
            data=transcript,
            file_name="physics-notebook-transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Main Header Banner
# ---------------------------------------------------------------------------
ui.masthead()
ui.hero(compact=bool(st.session_state.messages))

if index_error:
    st.error(index_error)
    st.stop()
elif not api_key:
    st.warning(
        "⚠️ **GROQ_API_KEY** is missing. Locally put it in `.env`. "
        "On Streamlit Cloud: **App settings → Secrets** and paste "
        '`GROQ_API_KEY = "your-key"` (plus optional `GROQ_MODEL`). '
        "You can still test vector search in Explore the textbook."
    )

# ---------------------------------------------------------------------------
# Navigation Tabs: Tutor Chat vs. Vector Search Inspector
# ---------------------------------------------------------------------------
tab_chat, tab_inspector = st.tabs(["Your notebook", "Explore the textbook"])

# ---------------------------------------------------------------------------
# Tab 1: Grounded Q&A Chat
# ---------------------------------------------------------------------------
with tab_chat:
    active_question = None

    # Curated Quick Prompts
    if not st.session_state.messages:
        ui.section_heading()
        with st.container(key="topic_grid"):
            cols = st.columns(4)
            for col, (number, category, title, subtitle, question, sketch) in zip(cols, ui.TOPICS):
                with col, st.container(key=f"topic_card_{number}"):
                    ui.topic_card(number, category, title, subtitle, sketch)
                    if st.button("Explore this idea  ↗", key=f"prompt_{number}",
                                 use_container_width=True,
                                 disabled=not (ready or (preview_mode and not index_error))):
                        active_question = question

    # Render Chat History
    for msg in st.session_state.messages:
        avatar = ":material/menu_book:" if msg["role"] == "assistant" else ":material/person:"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("error"):
                st.error(msg["text"])
                st.caption("Your question is saved. Use Try this question again below.")
            elif msg.get("is_preview"):
                st.info("ℹ️ **Offline Preview Mode (API was not called)**")
                st.json(msg["payload"])
            else:
                st.markdown(normalize_math_markdown(msg["text"]) if msg["role"] == "assistant" else msg["text"])

            if msg.get("elapsed") is not None:
                st.markdown(
                    f"<div class='latency-tag'>⚡ <b>{msg['elapsed']:.2f}s</b> response time · "
                    f"<b>{len(msg.get('sources', []))}</b> passages cited</div>",
                    unsafe_allow_html=True,
                )

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"Read the sources · {len(sources)} passages"):
                    st.caption("Exact excerpts retrieved from the PDF. Page numbers count from the first PDF page.")
                    for idx, s in enumerate(sources, 1):
                        p_start, p_end = s["page_start"], s["page_end"]
                        page_label = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"
                        section = s.get("section") or "General Physics"

                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">
                                <span>#{idx} · {page_label} · {escape(str(section))}</span>
                                <span class="passage-meta">Chunk ID: {escape(str(s.get('id', 'N/A')))}</span>
                            </div>
                            <div class="passage-text">{escape(s['text'])}</div>
                        </div>
                        """, unsafe_allow_html=True)

    if st.session_state.messages:
        latest = st.session_state.messages[-1]
        if latest["role"] == "assistant" and not latest.get("error") and not latest.get("is_preview"):
            if st.button("Show every calculation step", key="explain_calculations", disabled=not ready):
                active_question = "Explain the previous problem in detail, showing every calculation step."

    # Retry Button for failed attempts
    if st.session_state.get("failed_question"):
        if st.button("Try this question again", disabled=not (ready or preview_mode)):
            active_question = st.session_state.failed_question
            if st.session_state.messages and st.session_state.messages[-1].get("error"):
                st.session_state.messages.pop()
            st.session_state.retrying = True

    # Chat input
    user_input = st.chat_input(
        "What are you curious about?",
        disabled=not ready and not preview_mode,
        max_chars=2000,
    )
    st.html('<p class="input-hint">A question, a tricky concept, a “why does that happen?”</p>')
    final_query = user_input or active_question

    if final_query and final_query.strip():
        query_text = final_query.strip()
        if not st.session_state.pop("retrying", False):
            st.session_state.messages.append({"role": "user", "text": query_text})
            st.rerun()
        else:
            # We already have the user message; process directly
            pass

    # Process pending user message if last message is from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        current_q = st.session_state.messages[-1]["text"]
        history = st.session_state.messages[:-1]
        start_time = time.monotonic()

        if preview_mode:
            with st.spinner("Generating prompt preview without API call…"):
                search, lock = get_search_instance(manifest["build_hash"])
                with lock:
                    passages = search.search(retrieval_query(current_q, history), top_k=top_k)
                req_payload = build_request(current_q, passages, model=model_choice, history=history)
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": "Here is the JSON request that would be sent to Groq:",
                    "is_preview": True,
                    "payload": req_payload,
                    "sources": passages,
                    "elapsed": time.monotonic() - start_time,
                })
                st.rerun()
        else:
            try:
                with st.spinner("Searching textbook & preparing cited answer…"):
                    res = execute_tutor_answer(
                        question=current_q,
                        top_k=top_k,
                        api_key=api_key,
                        model=model_choice,
                        build_hash=manifest["build_hash"],
                        history=history,
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": res["answer"],
                    "sources": res.get("sources", []),
                    "elapsed": time.monotonic() - start_time,
                })
                st.session_state.pop("failed_question", None)
            except Exception as exc:
                err_msg = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": f"Error: {err_msg}",
                    "error": True,
                })
                st.session_state.failed_question = current_q
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Retrieval Inspector (Offline Vector Similarity Search)
# ---------------------------------------------------------------------------
with tab_inspector:
    st.markdown("#### A closer look at the textbook")
    st.caption("Find the original passages behind an idea. Search your book without generating an answer.")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        inspect_query = st.text_input(
            "Search query",
            placeholder="e.g. Newton's laws of motion, inertia, work and energy...",
            label_visibility="collapsed",
        )
    with col_k:
        inspect_k = st.number_input("Passages", min_value=1, max_value=15, value=5)

    if st.button("Find passages  →", type="secondary"):
        if inspect_query.strip():
            start_search = time.monotonic()
            results = execute_search_only(inspect_query.strip(), inspect_k, manifest["build_hash"])
            elapsed = time.monotonic() - start_search
            st.session_state.inspector_results = {
                "query": inspect_query.strip(),
                "results": results,
                "elapsed": elapsed,
            }
        else:
            st.warning("Please enter a query to test retrieval.")

    # Display Inspector Results
    if st.session_state.inspector_results:
        data = st.session_state.inspector_results
        st.markdown(f"**Showing top {len(data['results'])} passages for:** *\"{data['query']}\"* (took `{data['elapsed']*1000:.1f}ms`)")

        for i, item in enumerate(data["results"], 1):
            distance = item.get("distance", 0.0)
            # Cosine distance is in [0, 2], lower is closer. Similarity approx = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distance))
            p_start, p_end = item.get("page_start"), item.get("page_end")
            page_text = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"

            with st.container():
                st.markdown(f"""
                <div class="passage-card">
                    <div class="passage-header">
                        <span><b>#{i}</b> · {page_text} · <i>{escape(str(item.get('section') or 'Textbook section'))}</i></span>
                        <span class="passage-meta">Distance: <b>{distance:.4f}</b> · Sim: <b>{similarity*100:.1f}%</b></span>
                    </div>
                    <div style="background-color: #e6eeea; height: 5px; border-radius: 3px; margin-bottom: 8px;">
                        <div style="background-color: #246B58; width: {similarity*100}%; height: 100%; border-radius: 3px;"></div>
                    </div>
                    <div class="passage-text">{escape(item['text'])}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("View retrieval details"):
            st.json(data["results"])

ui.footer(manifest.get("chunks", 0))
