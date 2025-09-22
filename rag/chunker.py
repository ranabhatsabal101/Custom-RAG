import os, fitz
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

CHUNK_SIZE=400
OVERLAP = 200
RETRIES = 5
EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "")

def extract_text_pages(pdf_path: str):
    pages = []
    with fitz.open(pdf_path) as doc:
        for p in doc:
            text = p.get_text("text")
            pages.append(text or "")

    return pages

def make_chunks(pages, size=CHUNK_SIZE, overlap = OVERLAP):
    chunks, ordinal = [], 0

    for page_num, page in enumerate(pages, start=1):
        start = 0
        while start < len(page):
            end = min(len(page), start + size)
            chunk_text = page[start:end]
            chunks.append({
                "text": chunk_text,
                "ordinal": ordinal,
                "page_num": page_num,
                "start": start,
                "end": end,
                "embed_model": EMBED_MODEL
            })

            ordinal += 1
            if end == len(page):
                break

            start = end - overlap
    return chunks