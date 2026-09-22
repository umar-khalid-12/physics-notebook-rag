"""
Step 1: Read a scanned PDF page by page with OCR and save the text
together with its page number.

Usage:
    python ocr_book.py                 # whole book
    python ocr_book.py --end 5         # only pages 1 to 5 (good for a quick test)

You can stop it any time (Ctrl+C) and run it again. It skips pages it already did.
"""
import argparse
import json
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path

DATA_DIR = Path(__file__).resolve().parent / "Data"
PDF_PATH = DATA_DIR / "physics9.pdf"
OUT_PATH = DATA_DIR / "book_pages.json"
DPI = 200  # higher = clearer text but slower. 200 is a good middle.


def load_existing():
    """If we already OCR'd some pages before, load them so we can continue."""
    if OUT_PATH.exists():
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            return {p["page"]: p["text"] for p in json.load(f)}
    return {}


def save(pages_dict):
    """Write everything to disk, sorted by page number."""
    data = [{"page": n, "text": t} for n, t in sorted(pages_dict.items())]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"Can't find {PDF_PATH}. Put your PDF in the data folder and name it physics9.pdf")

    total = pdfinfo_from_path(str(PDF_PATH))["Pages"]
    end = min(args.end or total, total)
    pages = load_existing()
    print(f"Book has {total} pages. Doing pages {args.start} to {end}.")

    for n in range(args.start, end + 1):
        if n in pages:
            continue  # already done
        # Convert just ONE page to an image (keeps memory low)
        image = convert_from_path(str(PDF_PATH), dpi=DPI, first_page=n, last_page=n)[0]
        text = pytesseract.image_to_string(image, lang="eng")
        pages[n] = text.strip()
        print(f"Page {n}/{end} done ({len(text.split())} words)")
        if n % 10 == 0:
            save(pages)  # save progress every 10 pages

    save(pages)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
