# Physics book RAG

## Current steps

1. OCR: `ocr_book.py`, saved in `Data/book_pages.json` (200 pages).
2. Cleaning and 3. chunking: `clean_and_chunk.py`.
4. Local embeddings and ChromaDB: `embed_book.py`.
5. Question search: `search_book.py`.
6. Groq answers with citations: `answer_book.py`.
7. Streamlit and 8. evaluations: still to build.

## Run Step 4

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python clean_and_chunk.py
.venv/bin/python embed_book.py
```

The first embedding run needs internet to download a free local model. No API
key is required. Later runs reuse the downloaded model. An unchanged completed
index is reused without embedding again. Changed chunks create a new collection;
`Data/index_manifest.json` identifies the current completed build.

On macOS, install Python 3.12 with `brew install python@3.12` before creating the
environment. This project uses Homebrew Python with OpenSSL, avoiding the
`NotOpenSSLWarning` from Apple's Python 3.9/LibreSSL. Installing OpenSSL alone
does not change the SSL library used by an existing Python environment.
The previous environment is preserved in `.venv-python39/` as a backup; use
`.venv/bin/python` for normal commands. In VS Code, choose that interpreter via
**Python: Select Interpreter** if the editor still selects Python 3.9.

The database is in `Data/chroma/` and the model is in `Data/model_cache/`.
Each record stores its text, 384-dimensional vector, chapter, section, and PDF
page range. Page numbers count from the first PDF page, not the printed labels.
Long chunks are embedded in token windows and their vectors averaged so the
model reads all the text. Search must use `embed_full_texts` with the same model.

Cleaning skips the front matter (pages 1–5) and index (198–200), while keeping
the glossary (195–197). OCR errors in formulas and diagrams still need review;
embedding does not fix them. Chapter and section detection remains heuristic.

## Run Step 5

Ask one question and retrieve the closest five passages:

```sh
.venv/bin/python search_book.py "What is the difference between speed and velocity?"
```

Choose how many passages to show, or enter several questions interactively:

```sh
.venv/bin/python search_book.py "What is centripetal force?" --top-k 3
.venv/bin/python search_book.py
```

Type `exit` to finish interactive mode. Add `--json` to a single-question command
for structured results. The `BookSearch` class can also be imported by the future
answer script and UI; keep one instance to avoid reloading the model each time.

Search uses the cached model locally with the same embedding method as Step 4.
It checks that the chunks still match the completed index. If you change chunks,
rerun `embed_book.py` before searching.

Results show the original passage, section, PDF page range, and cosine distance
(lower is closer; it is not a confidence percentage). Search always returns the
nearest passages, even for questions outside the book. Use Step 6 to turn the
retrieved passages into a cited answer.

## Run Step 6

Get a Groq API key from the [Groq Console](https://console.groq.com/keys).
Open the project's `.env` file and set `GROQ_API_KEY` to your key. Keep it local;
the file is excluded from Git. `.env.example` is a blank template for new setups.
The default model is `openai/gpt-oss-120b`; change `GROQ_MODEL` in `.env` if needed.
Existing shell environment variables take precedence over `.env`.

To list available model IDs from Groq:

```sh
.venv/bin/python answer_book.py --list-models
```

Set `GROQ_MODEL` in `.env` to any listed text model (such as `openai/gpt-oss-120b` or `openai/gpt-oss-20b`).

For temporary HTTP 500/502/503/504 failures, the script retries twice, waiting
about 2 then 4 seconds with a small random delay. Retry messages go to stderr,
so `--json` output stays valid. After three failed attempts it reports an error.
Authentication, quota, and invalid-request errors are not automatically retried.
You can test another listed text model with `--model MODEL_NAME` without editing
`.env`. No model switching happens automatically.

```sh
.venv/bin/python answer_book.py "What is the difference between speed and velocity?"
.venv/bin/python answer_book.py "Who won the world cup?"
```

The script retrieves five passages locally, sends them and your question to
Groq, and asks for an answer using only those passages. This API call requires
internet and uses your Groq account's rate limits/billing. It does not send the PDF
or API key as prompt text. Use `--top-k 3` to retrieve three passages, or `--json`
for structured output including the cited source text.

To inspect the request locally without a key or an API call:

```sh
.venv/bin/python answer_book.py "What is centripetal force?" --preview
```

Each answer statement must cite a retrieved chunk ID. The program checks the IDs
and attaches the saved PDF page ranges. If Groq reports insufficient evidence,
the program responds: "I don't know based on the retrieved book passages."
Invalid citations, blocked/truncated responses, and API failures are reported as
errors rather than shown as answers. These checks verify citation identity, not
whether the statement is actually supported: model grounding and abstention still
need live evaluation in Step 8, especially for garbled OCR and unrelated questions.

Run the offline tests (simulated API responses, no key or charges):

```sh
.venv/bin/python -m unittest test_answer_book.py -v
```

Groq integration follows Groq's [OpenAI-compatible Chat Completions API](https://console.groq.com/docs/openai).

API references: [Chroma persistent client](https://docs.trychroma.com/reference/python/client),
[Sentence Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
