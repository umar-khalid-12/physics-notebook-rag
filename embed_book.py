"""Step 4: embed saved chunks locally and persist them in ChromaDB.

Run: .venv/bin/python embed_book.py
The first run downloads the model. Matching completed builds are reused.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PIPELINE_VERSION = "token-windows-mean-v1"


def embed_full_texts(model, texts):
    """Average token-window vectors so long chunks are not silently truncated."""
    import numpy as np

    tokenizer = model.tokenizer
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide at least one nonempty text to embed.")
    # Tokenize, split into non-overlapping windows, and pad in one fast call.
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=model.max_seq_length,
        stride=0, return_overflowing_tokens=True, return_tensors="pt",
    )
    owners = encoded.pop("overflow_to_sample_mapping").numpy()
    import torch
    vectors = []
    model.eval()
    for offset in range(0, len(owners), 32):
        batch = {key: value[offset:offset + 32].to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            vectors.extend(model(batch)["sentence_embedding"].cpu().numpy())
    vectors = np.asarray(vectors)
    pooled = np.stack([vectors[owners == i].mean(axis=0) for i in range(len(texts))])
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return pooled.tolist()


def main():
    import chromadb
    from sentence_transformers import SentenceTransformer

    path = DATA / "chunks.json"
    if not path.exists():
        raise SystemExit("Run clean_and_chunk.py first to create Data/chunks.json.")
    raw = path.read_bytes()
    chunks = json.loads(raw)
    if not chunks or any(not c["text"].strip() for c in chunks):
        raise SystemExit("Chunks must contain nonempty text.")
    ids = [str(c["id"]) for c in chunks]
    if len(set(ids)) != len(ids):
        raise SystemExit("Chunk IDs must be unique.")

    # A new input/model gets its own collection; existing builds remain intact.
    digest = hashlib.sha256(raw + MODEL_NAME.encode() + PIPELINE_VERSION.encode()).hexdigest()
    name = "physics9-" + digest[:16]
    client = chromadb.PersistentClient(path=str(DATA / "chroma"))
    collection = client.get_or_create_collection(
        name=name, embedding_function=None,
        metadata={"hnsw:space": "cosine", "model": MODEL_NAME, "build_hash": digest},
    )
    if collection.count() != len(chunks):
        print(f"Loading {MODEL_NAME} (downloads on first run)...", flush=True)
        model = SentenceTransformer(MODEL_NAME, cache_folder=str(DATA / "model_cache"), device="cpu")
        for offset in range(0, len(chunks), 32):
            batch = chunks[offset:offset + 32]
            texts = [c["text"] for c in batch]
            collection.upsert(
                ids=[str(c["id"]) for c in batch],
                documents=texts,
                embeddings=embed_full_texts(model, texts),
                metadatas=[{
                    "source": "physics9.pdf", "page_start": c["page_start"],
                    "page_end": c["page_end"], "chapter": c["chapter"],
                    "section": c["section"], "page_numbering": "PDF (1-based)",
                } for c in batch],
            )
            print(f"Stored {min(offset + 32, len(chunks))}/{len(chunks)} chunks", flush=True)
    else:
        print("This exact build is already stored; skipping embedding.")

    assert collection.count() == len(chunks), "Stored chunk count does not match input"
    manifest = {"collection": name, "model": MODEL_NAME, "pipeline": PIPELINE_VERSION,
                "build_hash": digest, "chunks": len(chunks), "dimensions": 384}
    temporary = DATA / "index_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(DATA / "index_manifest.json")
    print(f"Step 4 complete: {collection.count()} chunks saved in {DATA / 'chroma'}")


if __name__ == "__main__":
    main()
