from rag.llm_client import get_llm_client
from pydantic import BaseModel, ValidationError
from typing import List

import os, json, re

LLM_CLIENT = get_llm_client()

_SYSTEM = """You are an intent and query-rewriting assistant for a RAG system.
Decide if the user's message should trigger a knowledge-base search (documents).
Ensure that a generic question that could be easily answered through web search
is not supposed to trigger a knowledge-base search.
If yes, output high-quality rewrites:
- semantic_query: best semantic form (natural language) for semantic similarity checks
- keyword_query: FTS-friendly string; OR terms; keep quotes together
- must_terms: []  (exact terms that must appear; else empty)
- should_terms: [] (optional helpful terms; do not force any keyword here just to fill this)
Return STRICT JSON only:
{
  "trigger": boolean,
  "intent": "kb_search" | "smalltalk" | "nonsense" | "other",
  "reason": string,
  "semantic_query": string,
  "keyword_query": string,
  "must_terms": string[],
  "should_terms": string[]
}"""

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*)\s*```\s*$", re.DOTALL)

class QueryResponse(BaseModel):
    trigger: bool = False
    intent: str = ""
    reason: str = ""
    semantic_query: str = ""
    keyword_query: str = ""
    must_terms: List[str] = []
    should_terms: List[str] = []
    
class IntentService:
    def __init__(self):
        self.client = LLM_CLIENT

    def _strip_code_fences(self, s):
        m = _JSON_FENCE_RE.match(s)
        return m.group(1) if m else s
    
    def _extract_first_json_object(self, text):
        text = text.strip()
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None

    def _parse_query_response_from_completion(self, txt):
        try:
            return QueryResponse.model_validate_json(txt)
        except Exception:
            pass

        # Strip code fences and extract first
        txt = self._strip_code_fences(txt)
        jtxt = self._extract_first_json_object(txt)
        if jtxt:
            return QueryResponse.model_validate_json(jtxt)

        # Parse to dict then validate
        try:
            data = json.loads(txt)
            return QueryResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Safe fallback
            return QueryResponse(
                trigger=False, intent="other", reason=f"llm_failed: {str(e)}",
                semantic_query="", keyword_query="",
                must_terms=[], should_terms=[],
            )

    def analyze(self, query):
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"User query:\n{query}\nRespond with STRICT JSON only."}
        ]

        try:
            text = self.client.chat_query(msgs, temperature = 0.0, response_format={"type": "json_object"})
            response = self._parse_query_response_from_completion(text)
            return response.model_dump()
        except Exception as e:
            return QueryResponse(
                trigger=False,
                intent="other",
                reason=f"llm_failed: {str(e)}",
                semantic_query=query,
                keyword_query=query,
                must_terms=[],
                should_terms=[],
            )
        

def get_intent_service():
    return IntentService()