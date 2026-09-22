"""Step 5: retrieve book passages for a question using the saved local index.

Run with a quoted question, or without one for an interactive prompt.
"""
import argparse
import hashlib
import json
import sys

from embed_book import DATA, MODEL_NAME, PIPELINE_VERSION, embed_full_texts


class BookSearch:
    """Load once and reuse for multiple questions (and the future chat app)."""

    def __init__(self):
        manifest_path = DATA / "index_manifest.json"
        if not manifest_path.exists() or not (DATA / "chroma").is_dir():
            raise ValueError("No saved index found. Run .venv/bin/python embed_book.py first.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model"] != MODEL_NAME or manifest["pipeline"] != PIPELINE_VERSION:
            raise ValueError("The index uses a different embedding setup. Run embed_book.py again.")
        digest = hashlib.sha256(
            (DATA / "chunks.json").read_bytes() + MODEL_NAME.encode() + PIPELINE_VERSION.encode()
        ).hexdigest()
        if digest != manifest["build_hash"]:
            raise ValueError("Chunks changed since indexing. Run embed_book.py again before searching.")

        import chromadb
        from sentence_transformers import SentenceTransformer

        self.client = chromadb.PersistentClient(path=str(DATA / "chroma"))
        self.collection = self.client.get_collection(manifest["collection"], embedding_function=None)
        self.count = self.collection.count()
        if not self.count or self.count != manifest["chunks"]:
            raise ValueError("The saved index is incomplete. Run embed_book.py again.")
        if (self.collection.metadata or {}).get("build_hash") != digest:
            raise ValueError("Collection metadata does not match the saved build.")
        self.model = SentenceTransformer(
            MODEL_NAME, cache_folder=str(DATA / "model_cache"),
            device="cpu", local_files_only=True,
        )

    def search(self, question, top_k=5):
        question = question.strip()
        if not question:
            raise ValueError("Enter a nonempty question.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        response = self.collection.query(
            query_embeddings=embed_full_texts(self.model, [question]),
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": chunk_id, "text": text, "distance": float(distance), **metadata}
            for chunk_id, text, metadata, distance in zip(
                response["ids"][0], response["documents"][0],
                response["metadatas"][0], response["distances"][0],
            )
        ]


def print_results(results):
    print("\nClosest book passages (matches may not answer the question).")
    print("Page numbers count from the first PDF page. Lower distance means closer similarity.")
    for rank, result in enumerate(results, 1):
        start, end = result["page_start"], result["page_end"]
        pages = str(start) if start == end else f"{start}–{end}"
        section = result["section"] or "Section not identified"
        print(f"\n[{rank}] {result['source']} | PDF pages {pages} | {section}")
        print(f"Cosine distance: {result['distance']:.3f} | Chunk {result['id']}")
        print(result["text"])


def positive_integer(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer")
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question in quotes; omit for interactive mode")
    parser.add_argument("--top-k", type=positive_integer, default=5, help="Number of passages (default: 5)")
    parser.add_argument("--json", action="store_true", help="Output JSON for a single question")
    args = parser.parse_args()
    if args.question is not None and not args.question.strip():
        parser.error("Enter a nonempty question.")
    if args.json and args.question is None:
        parser.error("--json requires a question.")

    try:
        print("Loading the local book index...", file=sys.stderr, flush=True)
        book = BookSearch()
        if args.question is not None:
            results = book.search(args.question, args.top_k)
            if args.json:
                print(json.dumps({"question": args.question, "results": results}, ensure_ascii=False, indent=2))
            else:
                print_results(results)
        else:
            print("Ask about the physics book. Type 'exit' or 'quit' to stop.")
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() in {"exit", "quit"}:
                    break
                if question:
                    print_results(book.search(question, args.top_k))
    except (EOFError, KeyboardInterrupt):
        print("\nSearch closed.", file=sys.stderr)
    except Exception as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
