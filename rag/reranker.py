from rag.llm_client import get_llm_client
from pydantic import BaseModel
from typing import Optional, List

LLM_CLIENT = get_llm_client()
BATCH_SIZE = 16
MAX_CANDIDATE_CHARS = 1000

class RerankResult(BaseModel):
    index: int
    score: float
    reason: Optional[str] = None

class RerankResponse(BaseModel):
    scores: List[int]
    reasons: List[str]

class _BaseReranker:
    name: str = "base"
    def score(self, query: str, candidates: List[str]) -> List[RerankResult]:
        raise NotImplementedError

def _trim(p: str) -> str:
    """Simple char-based truncation to avoid very long inputs."""
    p = p or ""
    if len(p) <= MAX_CANDIDATE_CHARS: 
        return p
    return p[:MAX_CANDIDATE_CHARS] + "…"
    
class LLMReranker(_BaseReranker):
    SYSTEM = (
        "You are a reranker. Given a user query and a list of candidates, "
        "assign an integer score 0 to 3 to each candidate where 3=highly relevant, 1=somewhat relevant, 2=loosely relevant, 0=not relevant at all. "
        "Return STRICT JSON as: {\"scores\": [s0, s1, ...], \"reasons\": [\"...\", ...]} only."
    )

    def __init__(self):
        self.client = LLM_CLIENT

    def score(self, query: str, candidates: List[str]) -> List[RerankResult]:
        # Batch if many candidates to keep context small
        results: List[RerankResult] = []
        for start in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[start:start+BATCH_SIZE]
            lines = [f"Query: {query}", "Candidates:"]
            for i, p in enumerate(batch):
                lines.append(f"[{i}] {_trim(p)}")
            user = "\n".join(lines)

            msgs = [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": user}
            ]
            txt = self.client.chat_query(msgs, structured=True, temperature=0.0, response_format=RerankResponse)
            try:
                data = RerankResponse.model_validate_json(txt)
                scores = data.scores
                reasons = data.reasons
            except Exception:
                # if parsing fails, default to neutral 1.0
                scores = [1.0] * len(batch)
                reasons = ["llm_parse_failed"] * len(batch)

            # get final score between 0 and 1
            norm = [(float(s) / 3.0) if isinstance(s, (int, float)) else 0.0 for s in scores]
            for i, s in enumerate(norm):
                results.append(RerankResult(index=start+i, score=s, reason=reasons[i] if i < len(reasons) else None))
        return results
    
def build_reranker() -> _BaseReranker:
    return LLMReranker()