from __future__ import annotations
from rag.llm_client import get_llm_client

import json

LLM_CLIENT = get_llm_client()

SYSTEM = (
  """You refine user queries for a RAG system using history. Use recent dialogue to 
  resolve pronouns and references and add context to the queries for better results; 
  keep the user's intent unchanged. Return the refined query.
"""
)

MAX_TURNS = 8
PER_MSG_CAP = 600
MAX_TOTAL_CHARS = 6000

def _trim_history(history):
    """Keep last N turns; cap each; drop oldest until under budget."""
    if not history:
        return []

    # Normalize shape and order: we want chronological (oldest→newest)
    hist = [{"role": m.get("role","user"), "content": (m.get("content") or "")}
            for m in history]
    # If client sent newest-first, flip to oldest-first
    if len(hist) >= 2 and hist[0]["role"] in ("user","assistant") and hist[-1]["role"] in ("user","assistant"):
        # No perfect test, but we'll reverse if the last item looks like the latest turn
        pass  # leave as-is; clients usually send oldest→newest
    # Cap each message
    hist = [{"role": m["role"], "content": m["content"][:PER_MSG_CAP]} for m in hist[-MAX_TURNS:]]
    def total_chars(msgs): return sum(len(m["content"]) for m in msgs)
    while hist and total_chars(hist) > MAX_TOTAL_CHARS:
        hist.pop(0)  # drop oldest
    return hist


class QueryRefinerClient:
    def __init__(self):
        self._client = LLM_CLIENT

    def refine(self, query, history):
        hist = _trim_history(history or [])

        payload = {
            "recent_dialogue": hist,
            "current_query": query
        }
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]

        try:
            txt = self._client.chat_query(msgs, temperature=0.0)
        except Exception:
            txt = ""

        return txt if txt else query
    

def get_refiner():
    return QueryRefinerClient()