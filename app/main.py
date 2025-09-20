from fastapi import FastAPI
from app.api.ingest import router as ingest_router
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

app = FastAPI(title="Custom RAG")
app.include_router(ingest_router)
