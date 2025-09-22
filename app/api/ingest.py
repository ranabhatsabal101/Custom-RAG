from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from pypdf import PdfReader
from rag.db import init_schema, enqueue_index_job, create_document
from typing import List, Optional
import uuid, hashlib

ROUTER = APIRouter(prefix="/ingest", tags=["ingest"])

# Keeping it simple for now but there are safer and more dynamic ways to do this
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
init_schema()

class IngestResult(BaseModel):
    filename: str
    stored_as: str
    bytes: int
    pages: int
    sha256: str
    status: str
    message: Optional[str] = None

@ROUTER.post("/pdf_documents", response_model=List[IngestResult])
async def ingest_pdf_documents(files: List[UploadFile] = File(..., description="Single or more PDF files")):
    """
    Accepts one or more PDF files to be used as a source for RAG
    """
    results: List[IngestResult] = []
    
    for f in files:
        if f.content_type not in {"application/pdf", "application/x-pdf", "application/acrobat"} and not f.filename.lower().endswith(".pdf"):
            results.append(IngestResult(
                filename=f.filename,
                stored_as="",
                bytes=0,
                pages=0,
                sha256="",
                status="error",
                message="File format is not PDF."
            ))
            continue
    
        doc_id = uuid.uuid4().hex
        dest = UPLOAD_DIR / f"{doc_id}_{Path(f.filename).name}"
        sha = hashlib.sha256()
        nbytes = 0

        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    sha.update(chunk)
                    nbytes += len(chunk)

            pages = 0
            try:
                with dest.open("rb") as fh:
                    reader = PdfReader(fh)
                    if getattr(reader, "is_encrypted", False):
                        try:
                            reader.decrypt("")
                        except Exception:
                            pass
                    pages = len(reader.pages)
            except Exception as e:
                message = f"Stored but could not parse PDF: {str(e)}"
            else:
                message = "Stored PDF for indexing."

            create_document(
                id=doc_id,
                original_name=f.filename,
                storage_path=str(dest),
                sha256=sha.hexdigest(),
                bytes=nbytes,
                pages=pages
            )
            enqueue_index_job(doc_id, sha.hexdigest())

            results.append(IngestResult(
                filename=f.filename,
                stored_as=str(dest),
                bytes=nbytes,
                pages=pages,
                sha256=sha.hexdigest(),
                status="ok",
                message=message
            ))

        except Exception as e:
            if dest.exists():
                dest.unlink(missing_ok=True)
            results.append(IngestResult(
                filename = f.filename,
                stored_as="",
                bytes=nbytes,
                pages=0,
                sha256="",
                status="error",
                message=str(e)
            ))
        finally:
            await f.close()

    if not results:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    return results