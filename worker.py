import os, time, traceback
import numpy as np

from dotenv import load_dotenv, find_dotenv

from rag.chunker import extract_text_pages, make_chunks
from rag.indexer import add_to_flat_index, add_to_ivfpq_index
from rag.embedders import get_embedder
from rag.db import (init_schema, get_job, 
                    mark_job_done, mark_job_failed, update_document_status, 
                    get_document, insert_chunks, DocumentStatus)

load_dotenv(find_dotenv(), override=True)

EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
EMBEDDER = get_embedder(model=EMBED_MODEL)

def _get_embeddings(texts, ):
    return EMBEDDER.embed(texts)


def _process_index_document(document_id):
    update_document_status(document_id=document_id, status=DocumentStatus.PROCESSING.value)
    doc = get_document(document_id)
    if not doc:
        raise RuntimeError(f"Document {document_id} not found.")
    
    pages = extract_text_pages(doc["storage_path"])
    chunks = make_chunks(pages)
    chunk_ids = insert_chunks(document_id, chunks)

    texts = [c["text"] for c in chunks]
    vecs = _get_embeddings(texts)
    dim = vecs.shape[1]

    # Use Flat Index (Exhasutive Search) if not a lot of data
    add_to_flat_index(vecs, np.array(chunk_ids, dtype="int64"), dim)

    # Start using IVFPQ Index once the data size increases for better speed and memory usage
    add_to_ivfpq_index(dim, np.array(chunk_ids, dtype="int64"), vecs)

    update_document_status(document_id, DocumentStatus.INDEXED.value, pages=len(pages))

def main():
    init_schema()
    print("Worker started! Polling for jobs...")

    while True:
        job = get_job()
        if not job:
            time.sleep(1)
            continue
        try:
            if job["type"] == "INDEX_DOCUMENT":
                _process_index_document(job["document_id"])
            else:
                raise RuntimeError(f"Unknown job type: {job["type"]}")
            mark_job_done(job["id"])
            print("Job done!")
        except Exception as e:
            tb = traceback.format_exc()
            print("Job failed:", e, tb)
            mark_job_failed(job["id"], f"{e}\n{tb}")


if __name__ == '__main__':
    main()