from rag.intent_service import get_intent_service
from rag.embedders import get_embedder
from rag.indexer import IVFPQ_INDEX_PATH, FLAT_INDEX_PATH
from rag.db import match_fts_query, get_chunk_meta
from dotenv import load_dotenv, find_dotenv

import re, faiss
import numpy as np

load_dotenv(find_dotenv(), override=True)
EMBED_MODEL = get_embedder()
INTENT_SERVICE = get_intent_service()

_WS_RE = re.compile(r"\s+")
_SMART_QUOTES = [
    ("\u201C", '"'), ("\u201D", '"'),  # smart double
    ("\u2018", "'"), ("\u2019", "'"),  # smart single
]

class Retriever:
    def __init__(self):
        self.embed_model = EMBED_MODEL

    def _normalize(self, query):
        return _WS_RE.sub(" ", (query or "").strip())
    
    def _clean(self, query):
        if not query:
            return ""
        
        for a, b in _SMART_QUOTES:
            query = query.replace(a, b)
        return re.sub(r"\s+", " ", query.strip())
    
    def _prep_term(self, term):
        t = self._clean(term)
        if not t:
            return ""
        if t.startswith('"') and t.endswith('"'):
            return t
        # if it has whitespace, quote it as a phrase
        if any(c.isspace() for c in t):
            # escape interior double quotes by doubling them
            inner = t.replace('"', '""')
            return f'"{inner}"'
        return t

    def _prep_terms(self, terms):
        if not terms:
            return []
        
        if isinstance(terms, str):
            parts = re.findall(r'"[^"]+"|[^,;]+', terms)
            items = [p.strip() for p in parts if p.strip()]
        else:
            items = [str(t) for t in terms if t is not None]
        out = []
        seen = set()
        for it in items:
            lit = self._prep_term(it)
            if lit and lit not in seen:
                seen.add(lit)
                out.append(lit)
        return out
    
    def _get_fts_query(self, query, must_terms, should_terms):
        cleaned_query = self._clean(query or "")
        groups = []
        optional_terms = self._prep_terms(should_terms)
        must_terms = self._prep_terms(must_terms)

        if cleaned_query:
            groups.append(f"({cleaned_query})")
        else:
            groups.append("(" + " OR ".join(must_terms) + ")")
            
        if optional_terms:
            groups.append("(" + " OR ".join(optional_terms) + ")")
        
        if not groups:
            return '""'
        return " OR ".join(groups)
    
    def _load_index(self, dim):
        if IVFPQ_INDEX_PATH.exists():
            index = faiss.read_index(str(IVFPQ_INDEX_PATH))
            if getattr(index, "is_trained", False) and getattr(index, "d", dim) == dim and index.ntotal > 0:
                return index, "ivfpq"
            
        if FLAT_INDEX_PATH.exists():
            index = faiss.read_index(str(FLAT_INDEX_PATH))
            if index.ntotal > 0:
                return index, "flat"
            
        return None, ""
    
    def _semantic_search(self, index, embedded_query, top_k):
        if index is None:
            return []
        
        distances, labels = index.search(embedded_query.astype("float32"), top_k)
        ids = labels[0]
        scores = distances[0]

        simple_score = np.clip(scores, -1.0, 1.0)
        simple_score = (simple_score + 1.0) / 2.0

        return [(int(i), float(s)) for i,s in zip(ids, simple_score) if i != -1]
    
    def _normalize_bm25_score(self, rows):
        # Invert the scores so that 
        if not rows:
            return []
        
        scores = [r[1] for r in rows]       
        smin, smax = min(scores), max(scores)
        if abs(smax - smin) < 1e-9:
            return [(int(r[0]), 1.0) for r in rows]
        inv = [smax - s for s in scores]      # invert so bigger = better
        imin, imax = min(inv), max(inv)
        return [(int(rows[i][0]), (inv[i] - imin) / (imax - imin + 1e-12)) for i in range(len(rows))]

    
    def _keyword_search(self, query, top_k):
        rows = match_fts_query(query, top_k)
        if not rows:
            return []
        
        return self._normalize_bm25_score(rows)

    def _rrf(self, semantic_similarity, keyword_similarity, k):
        #Reciprocal Rank Fusion method
        rD = {cid: r for r, (cid, _) in enumerate(semantic_similarity, start=1)}
        rS = {cid: r for r, (cid, _) in enumerate(keyword_similarity, start=1)}

        all_ids = set(rD) | set(rS)

        fused = {}
        for cid in all_ids:
            s = 0.0
            if cid in rD:
                s += 1.0 / (k + rD[cid])
            if cid in rS:
                s += 1.0 / (k + rS[cid])
            fused[cid] = s
        return fused
    
    def _get_full_chunk_info(self, ids):
        if not ids:
            return []
        
        ids_filter = ",".join(str(i) for i in ids)
        res = get_chunk_meta(ids_filter)
        by_id = {r["chunk_id"]: r for r in res}
        return [by_id[i] for i in ids if i in by_id]

    def _embed_query(self, query):
        embeddings = self.embed_model.embed(query)
        return embeddings, embeddings.shape[1]

    
    def search(self, query, query_meta, top_k = 8, rrf_k = 60):
        semantic_query = self._normalize(query_meta.get("semantic_query", "") or query)
        keyword_query = self._get_fts_query(
            query_meta.get("keyword_query", "") or query,
            query_meta.get("must_terms", []),
            query_meta.get("should_terms", []))

        embedded_query, dim = self._embed_query(semantic_query)

        index, type = self._load_index(dim)
        semantic_similarity = self._semantic_search(index, embedded_query, top_k) if index is not None else []
        keyword_similairty = self._keyword_search(keyword_query, top_k)

        if not semantic_similarity and not keyword_similairty:
            return {
                "index_type": type,
                "query": {"original": query, "semantic": semantic_query, "keyword": keyword_query},
                "results": []
            }
        
        merged_similarity = self._rrf(semantic_similarity, keyword_similairty, rrf_k)
        top_ids = [cid for cid, _ in sorted(merged_similarity.items(), key=lambda x: x[1], reverse=True)][:top_k]

        matches = self._get_full_chunk_info(top_ids)
        semantic_map = {cid: s for cid, s in semantic_similarity}
        keywords_map = {cid: s for cid, s in keyword_similairty}
        merged_map = {cid: s for cid, s in merged_similarity.items()}

        for m in matches:
            cid = m["chunk_id"]
            m["scores"] = {
                "semantic": semantic_map.get(cid),
                "keyword": keywords_map.get(cid),
                "merged": merged_map.get(cid, 0.0)
            }

        return {
            "index_type": type,
            "query": {"original": query, "semantic": semantic_query, "keyword": keyword_query},
            "results": matches
        }