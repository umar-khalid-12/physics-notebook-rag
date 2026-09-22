"""
Steps 2 and 3: clean the OCR text, then cut it into chunks.

Input : data/book_pages.json   (made by ocr_book.py)
Output: data/chunks.json       (the pieces we will embed later)
        data/chunks_preview.txt (a readable copy so you can check it by eye)

Usage:
    python clean_and_chunk.py
"""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "Data"
IN_PATH = DATA_DIR / "book_pages.json"
OUT_PATH = DATA_DIR / "chunks.json"
FIRST_CONTENT_PAGE = 6   # skip the cover, preface and contents pages before this
FIRST_GLOSSARY_PAGE = 195
FIRST_INDEX_PAGE = 198  # index entries are pointers, not answer passages
PREVIEW_PATH = DATA_DIR / "chunks_preview.txt"

TARGET_WORDS = 250   # we start a new chunk once we pass this size
MAX_WORDS = 350      # hard limit for one chunk
OVERLAP_WORDS = 40   # last words of a chunk are repeated at the start of the next

HEADING_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+([A-Z][A-Za-z ,'&\-()]{2,80})$")
NEW_PARA_RE = re.compile(r"^(\(?[ivx]{1,4}\)|\(?[a-d]\)|[*•\-]\s|Q\d|Fig\.)")


# ---------------------------------------------------------------- cleaning
def is_junk(line):
    """True for lines that are OCR noise, like text read from a diagram."""
    if "=" in line:                      # keep formulas
        return False
    words = re.findall(r"[A-Za-z]{3,}", line)
    if not words:
        return True
    letters_in_words = sum(len(w) for w in words)
    if letters_in_words / max(len(line.replace(" ", "")), 1) < 0.5:
        return True
    tokens = line.split()
    if len(tokens) >= 4 and len(words) / len(tokens) < 0.4:
        return True
    return False


def page_to_paragraphs(text):
    """Turn raw page text into a list of clean paragraphs."""
    paragraphs, current = [], ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:                     # blank line = paragraph break
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if line.isdigit():               # lone page numbers
            continue
        if is_junk(line):
            continue
        line = re.sub(r"\s+", " ", line)
        starts_new = bool(HEADING_RE.match(line) or NEW_PARA_RE.match(line))
        if current and starts_new:
            paragraphs.append(current)
            current = ""
        if current.endswith("-") and line[:1].islower():
            current = current[:-1] + line          # fix "measure-\nment"
        else:
            current = (current + " " + line).strip()
        if HEADING_RE.match(current):    # a heading stands alone
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return paragraphs


def build_units(pages):
    """Make one list of paragraphs for the whole book, remembering page numbers."""
    units = []  # each: {"text", "page_start", "page_end"}
    for p in pages:
        paras = page_to_paragraphs(p["text"])
        for i, para in enumerate(paras):
            # A paragraph that continues from the previous page: glue them together
            if i == 0 and units and para[:1].islower() and not HEADING_RE.match(para):
                units[-1]["text"] += " " + para
                units[-1]["page_end"] = p["page"]
            else:
                units.append({"text": para, "page_start": p["page"], "page_end": p["page"]})
    return units


# ---------------------------------------------------------------- chunking
def split_long(unit):
    """Split a paragraph that is longer than MAX_WORDS into sentence groups."""
    if len(unit["text"].split()) <= MAX_WORDS:
        return [unit]
    sentences = re.split(r"(?<=[.!?])\s+", unit["text"])
    out, buf = [], []
    for s in sentences:
        if buf and len(" ".join(buf + [s]).split()) > TARGET_WORDS:
            out.append({**unit, "text": " ".join(buf)})
            buf = []
        buf.append(s)
    if buf:
        out.append({**unit, "text": " ".join(buf)})
    return out


def make_chunks(units):
    chunks = []
    chapter, sec_num, sec_title = "", "", ""
    buf = []            # paragraphs waiting to become a chunk
    real_words = 0      # words in buf that are NOT overlap
    buf_section = ("", "", "")

    def flush():
        nonlocal buf, real_words
        body = " ".join(u["text"] for u in buf)
        chap, num, title = buf_section
        label = f"{num} {title}".strip()
        header = f"Chapter {chap} | {label}" if (chap and label) else (f"Chapter {chap}" if chap else "")
        chunks.append({
            "id": len(chunks),
            "chapter": chap,
            "section": label,
            "page_start": min(u["page_start"] for u in buf),
            "page_end": max(u["page_end"] for u in buf),
            "text": (header + "\n" + body) if header else body,
        })
        # Keep each tail piece's source range, including cross-page paragraphs.
        tail_units, remaining = [], OVERLAP_WORDS
        for u in reversed(buf):
            words = u["text"].split()
            tail_units.insert(0, {**u, "text": " ".join(words[-remaining:]), "overlap": True})
            remaining -= min(remaining, len(words))
            if remaining == 0:
                break
        buf = tail_units
        real_words = 0

    for unit in units:
        m = HEADING_RE.match(unit["text"])
        if m:
            chapter, sec_num, sec_title = m.group(1), f"{m.group(1)}.{m.group(2)}", m.group(3)
            # Keep section labels and text aligned, even for short sections.
            if real_words:
                flush()
            buf = []
        for piece in split_long(unit):
            n = len(piece["text"].split())
            if real_words + n > MAX_WORDS and real_words > 0:
                flush()
            if real_words == 0:                      # first real paragraph of this chunk
                buf_section = (chapter, sec_num, sec_title)
            buf.append(piece)
            real_words += n
            if real_words >= TARGET_WORDS:
                flush()
    if real_words:                                   # preserve even a short final passage
        flush()
    return chunks


def main():
    if not IN_PATH.exists():
        raise SystemExit("Can't find data/book_pages.json. Run ocr_book.py first.")
    with open(IN_PATH, encoding="utf-8") as f:
        pages = json.load(f)

    pages = [p for p in pages if FIRST_CONTENT_PAGE <= p["page"] < FIRST_INDEX_PAGE]
    chunks = make_chunks(build_units([p for p in pages if p["page"] < FIRST_GLOSSARY_PAGE]))
    glossary = make_chunks(build_units([p for p in pages if p["page"] >= FIRST_GLOSSARY_PAGE]))
    for chunk in glossary:
        chunk.update(id=len(chunks), chapter="Glossary", section="Glossary")
        chunks.append(chunk)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(PREVIEW_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"===== chunk {c['id']} | pages {c['page_start']}-{c['page_end']} | "
                    f"{len(c['text'].split())} words =====\n{c['text']}\n\n")

    sizes = [len(c["text"].split()) for c in chunks]
    print(f"Pages read: {len(pages)}")
    print(f"Chunks made: {len(chunks)}")
    print(f"Words per chunk: smallest {min(sizes)}, average {sum(sizes)//len(sizes)}, biggest {max(sizes)}")
    print(f"Saved {OUT_PATH} and {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
