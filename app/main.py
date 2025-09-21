from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

app = FastAPI(title="Custom RAG")
app.include_router(ingest_router)
app.include_router(query_router)