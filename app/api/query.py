from fastapi import APIRouter, HTTPException
from rag.intent_service import get_intent_service
from rag.retriever import get_retriever
from rag.query_refiner import get_refiner
from rag.chat_assitant import get_chat_assistant
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any, Literal

RETRIEVER = get_retriever()
REFINER = get_refiner()
INTENT_SERVICE = get_intent_service()
CHAT_ASSISTANT = get_chat_assistant()

ROUTER = APIRouter(prefix="/query", tags=["query"])

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: int = Field(8, ge=1, le=50)
    rrf_k: int = Field(60, ge=1, le=200, description="RRF smoothing constant")
    history: List[Message] = []

class Source(BaseModel):
    rank: int
    chunk_id: int
    document_id: str
    document_name: str
    page_num: int
    text: str
    scores: Dict[str, Optional[float]] = {}

class Response(BaseModel):
    trigger: bool
    query_debug: Dict[str,Any]
    results: List[Source]
    answer: str


@ROUTER.post("", response_model=Response)
def query(request: QueryRequest):
    try:
        history = [m.model_dump() for m in request.history]
        refined_query = REFINER.refine(request.query, history)

        response = INTENT_SERVICE.analyze(refined_query)
        rag_trigger = bool(response.get("trigger", False))

        query_debug = {
            "original": request.query,
            "refined": refined_query,
            "meta": response,
        }

        results = []
        if rag_trigger:
            retrieved = RETRIEVER.search(
                refined_query,
                response,
                rerank=True,
                top_k=request.top_k,
                rrf_k=request.rrf_k)
            match = retrieved.get("results", [])
            for i, r in enumerate(match, start=1):
                text = r.get("text", "")
                results.append(Source(
                    rank=i,
                    chunk_id=r["chunk_id"],
                    document_id = r["document_id"],
                    document_name = r["document_name"],
                    page_num = r["page_num"],
                    text=(text[:1200] + "…") if len(text) > 1200 else text,
                    scores=r.get("scores", {})
                ))

        answer = CHAT_ASSISTANT.answer(rag_trigger, results, refined_query, temperature=0.3)
        return Response(trigger=rag_trigger, query_debug=query_debug, results=results, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))