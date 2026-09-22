"""Professional, clean, and modern frontend for the Physics Book RAG system.
Run with: .venv/bin/python -m streamlit run app.py
"""
import json
import os
import threading
import time

import streamlit as st
from dotenv import dotenv_values

from answer_book import BookAnswer, DEFAULT_MODEL, build_request
from embed_book import DATA, ROOT
from search_book import BookSearch

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Physics Notebook · AI Textbook Assistant",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #142820;
}

/* Container spacing */
.block-container {
    max-width: 1080px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
}

/* App Header styling */
.hero-container {
    background: linear-gradient(135deg, #f0f7f3 0%, #ffffff 100%);
    border: 1px solid #dbe8e1;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 4px 20px -8px rgba(36, 107, 88, 0.08);
}
.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #e4f0ea;
    color: #1b5b4a;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 0.7rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #10241b;
    letter-spacing: -0.035em;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-desc {
    color: #536c61;
    font-size: 1.02rem;
    margin: 0;
    line-height: 1.6;
}

/* Pill badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ffffff;
    border: 1px solid #d4e2da;
    border-radius: 24px;
    padding: 3px 10px;
    font-size: 0.76rem;
    color: #38564a;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 6px;
}
.status-dot {
    width: 7px;
    height: 7px;
    background-color: #24a148;
    border-radius: 50%;
}

/* Chat Message Cards */
[data-testid="stChatMessage"] {
    background-color: #ffffff;
    border: 1px solid #e1ebe5;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    transition: border-color 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    border-color: #ccdcd3;
}

/* LLM-style Markdown inside assistant replies */
[data-testid="stChatMessage"] h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #10241b;
    margin: 1.05rem 0 0.45rem 0;
    letter-spacing: -0.02em;
}
[data-testid="stChatMessage"] h3:first-child {
    margin-top: 0;
}
[data-testid="stChatMessage"] p {
    line-height: 1.65;
    margin: 0.35rem 0 0.7rem 0;
}
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    margin: 0.25rem 0 0.85rem 0.2rem;
    padding-left: 1.2rem;
}
[data-testid="stChatMessage"] li {
    margin: 0.25rem 0;
    line-height: 1.55;
}
[data-testid="stChatMessage"] strong {
    color: #0f2a20;
}
[data-testid="stChatMessage"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.6rem 0 1rem 0;
    font-size: 0.92rem;
    overflow: hidden;
    border: 1px solid #d7e5de;
    border-radius: 10px;
}
[data-testid="stChatMessage"] thead th {
    background: #eaf3ee;
    color: #1b4d3e;
    font-weight: 700;
    text-align: left;
}
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {
    border: 1px solid #d7e5de;
    padding: 0.55rem 0.75rem;
    vertical-align: top;
}
[data-testid="stChatMessage"] tbody tr:nth-child(even) {
    background: #f7fbf9;
}
[data-testid="stChatMessage"] .katex-display {
    margin: 0.75rem 0;
    overflow-x: auto;
}

/* Citation Box */
.citation-badge {
    display: inline-block;
    background: #eaf3ee;
    color: #1e5e4d;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 2px 8px;
    border-radius: 6px;
    border: 1px solid #cfded6;
    margin-left: 4px;
}

/* Passage card for inspector & sources */
.passage-card {
    background: #fbfdfc;
    border: 1px solid #e2ebe6;
    border-left: 4px solid #246B58;
    border-radius: 10px;
    padding: 1.1rem;
    margin-bottom: 0.9rem;
}
.passage-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #1d352b;
    font-weight: 600;
}
.passage-meta {
    font-size: 0.76rem;
    color: #557266;
}
.passage-text {
    font-size: 0.86rem;
    line-height: 1.65;
    color: #283d34;
    white-space: pre-wrap;
    background: #ffffff;
    border: 1px solid #edf3f0;
    border-radius: 6px;
    padding: 0.75rem;
    font-family: inherit;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f7faf8;
    border-right: 1px solid #e1eae5;
}
.sidebar-box {
    background: #ffffff;
    border: 1px solid #dfe9e3;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.8rem 0;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: -0.01em;
    transition: all 0.15s ease;
}
.stButton button:hover {
    border-color: #246B58;
    color: #246B58;
}

/* Custom tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e1ece6;
    margin-bottom: 1.4rem;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 8px 8px 0 0;
    padding: 0 16px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #4b665a;
}
.stTabs [aria-selected="true"] {
    color: #175443 !important;
    border-bottom: 2px solid #246B58 !important;
}

/* Footer / status */
.latency-tag {
    font-size: 0.75rem;
    color: #647e72;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


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


def execute_tutor_answer(question, top_k, api_key, model, build_hash):
    """Execute grounded RAG query through BookAnswer with thread safety."""
    search, lock = get_search_instance(build_hash)
    with lock:
        answerer = BookAnswer(api_key=api_key, model=model, search=search)
        return answerer.ask(question, top_k=top_k)


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
    st.markdown("### ⚛️ Physics Notebook")
    st.caption("AI Study Assistant & RAG Evaluation Studio")

    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pop("failed_question", None)
        st.session_state.pop("preview_request", None)
        st.rerun()

    # Textbook library card
    st.markdown("""
    <div class="sidebar-box">
        <div style="font-size: 0.72rem; font-weight:700; color:#246B58; letter-spacing:0.08em; text-transform:uppercase;">Textbook Library</div>
        <div style="font-weight:700; font-size:1.05rem; margin-top:3px;">Physics · Grade 9</div>
        <div style="font-size:0.83rem; color:#5c766b; margin-top:2px;">Complete 200 PDF pages indexed with all-MiniLM-L6-v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ Test Settings")
    top_k = st.slider(
        "Passages to consult (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values retrieve more context for Groq to synthesize, but take slightly longer.",
    )

    model_choice = st.text_input(
        "Groq Model",
        value=default_model,
        help="Model ID used for grounded generation (e.g. openai/gpt-oss-120b).",
    )

    preview_mode = st.checkbox(
        "Offline Preview (Dry run)",
        value=False,
        help="View the generated prompt & context payload without sending an API request to Groq.",
    )

    st.divider()

    # Diagnostics
    with st.expander("🛠️ System Diagnostics"):
        st.markdown(f"**Index Chunks:** {manifest.get('chunks', 'N/A')}")
        st.markdown(f"**Embedding Model:** `{manifest.get('model', 'all-MiniLM-L6-v2')}`")
        st.markdown(f"**Collection:** `{manifest.get('collection', 'N/A')}`")
        st.markdown(f"**Build Hash:** `{manifest.get('build_hash', 'N/A')[:12]}...`" if manifest.get('build_hash') else "N/A")
        st.markdown(f"**API Key:** {'Configured ✅' if api_key else 'Missing ⚠️'}")

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
            "📥 Download transcript",
            data=transcript,
            file_name="physics-notebook-transcript.md",
            mime="text/markdown",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Main Header Banner
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-tag">Grade 9 Physics · Verified RAG</div>
    <h1 class="hero-title">Make sense of physics.</h1>
    <p class="hero-desc">Ask your textbook any question. Every answer is grounded directly in your book's pages with verified citations.</p>
    <div style="margin-top: 10px;">
        <span class="status-pill"><span class="status-dot"></span> 200 Pages Indexed</span>
        <span class="status-pill">📚 {manifest.get('chunks', 0)} Chunks in ChromaDB</span>
        <span class="status-pill">🤖 {model_choice}</span>
        <span class="status-pill">🔒 Offline Embedding</span>
    </div>
</div>
""", unsafe_allow_html=True)

if index_error:
    st.error(index_error)
    st.stop()
elif not api_key:
    st.warning(
        "⚠️ **GROQ_API_KEY** is missing. Locally put it in `.env`. "
        "On Streamlit Cloud: **App settings → Secrets** and paste "
        '`GROQ_API_KEY = "your-key"` (plus optional `GROQ_MODEL`). '
        "You can still test vector search in the Retrieval Inspector."
    )

# ---------------------------------------------------------------------------
# Navigation Tabs: Tutor Chat vs. Vector Search Inspector
# ---------------------------------------------------------------------------
tab_chat, tab_inspector = st.tabs(["💬 Ask the Book (Grounded Q&A)", "🔍 Retrieval Inspector (Vector Search)"])

# ---------------------------------------------------------------------------
# Tab 1: Grounded Q&A Chat
# ---------------------------------------------------------------------------
with tab_chat:
    active_question = None

    # Curated Quick Prompts
    if not st.session_state.messages:
        st.markdown("##### 💡 Try an example question")
        quick_questions = [
            "What is the difference between speed and velocity?",
            "What is centripetal force?",
            "How are mass and weight different?",
            "Explain Newton's third law of motion.",
        ]
        cols = st.columns(len(quick_questions))
        for col, qq in zip(cols, quick_questions):
            if col.button(qq, use_container_width=True, disabled=not ready):
                active_question = qq

    # Render Chat History
    for msg in st.session_state.messages:
        avatar = "⚛️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("error"):
                st.error(msg["text"])
                st.caption("Your question is saved. Use Retry last question below.")
            elif msg.get("is_preview"):
                st.info("ℹ️ **Offline Preview Mode (API was not called)**")
                st.json(msg["payload"])
            else:
                st.markdown(msg["text"])

            if msg.get("elapsed") is not None:
                st.markdown(
                    f"<div class='latency-tag'>⚡ <b>{msg['elapsed']:.2f}s</b> response time · "
                    f"<b>{len(msg.get('sources', []))}</b> passages cited</div>",
                    unsafe_allow_html=True,
                )

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📖 View Cited Book Sources ({len(sources)} passages)"):
                    st.caption("Exact excerpts retrieved from the PDF. Page numbers count from the first PDF page.")
                    for idx, s in enumerate(sources, 1):
                        p_start, p_end = s["page_start"], s["page_end"]
                        page_label = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"
                        section = s.get("section") or "General Physics"

                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">
                                <span>#{idx} · {page_label} · {section}</span>
                                <span class="passage-meta">Chunk ID: {s.get('id', 'N/A')}</span>
                            </div>
                            <div class="passage-text">{s['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Retry Button for failed attempts
    if st.session_state.get("failed_question"):
        if st.button("🔄 Retry last question", disabled=not ready):
            active_question = st.session_state.failed_question
            if st.session_state.messages and st.session_state.messages[-1].get("error"):
                st.session_state.messages.pop()
            st.session_state.retrying = True

    # Chat input
    user_input = st.chat_input(
        "Ask a question about 9th-grade physics…",
        disabled=not ready and not preview_mode,
        max_chars=2000,
    )
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
        start_time = time.monotonic()

        if preview_mode:
            with st.spinner("Generating prompt preview without API call…"):
                search, lock = get_search_instance(manifest["build_hash"])
                with lock:
                    passages = search.search(current_q, top_k=top_k)
                req_payload = build_request(current_q, passages, model=model_choice)
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
    st.markdown("#### 🔍 Real-Time Vector Retrieval Inspector")
    st.caption("Test ChromaDB embeddings and similarity ranking directly without calling Groq or spending API tokens.")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        inspect_query = st.text_input(
            "Search query",
            placeholder="e.g. Newton's laws of motion, inertia, work and energy...",
            label_visibility="collapsed",
        )
    with col_k:
        inspect_k = st.number_input("Results (k)", min_value=1, max_value=15, value=5)

    if st.button("🔎 Run Retrieval Search", type="secondary"):
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
                        <span><b>#{i}</b> · {page_text} · <i>{item.get('section') or 'Textbook section'}</i></span>
                        <span class="passage-meta">Distance: <b>{distance:.4f}</b> · Sim: <b>{similarity*100:.1f}%</b></span>
                    </div>
                    <div style="background-color: #e6eeea; height: 5px; border-radius: 3px; margin-bottom: 8px;">
                        <div style="background-color: #246B58; width: {similarity*100}%; height: 100%; border-radius: 3px;"></div>
                    </div>
                    <div class="passage-text">{item['text']}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📄 View Raw Results (JSON)"):
            st.json(data["results"])
