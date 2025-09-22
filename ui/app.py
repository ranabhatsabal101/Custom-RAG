import streamlit as st
import io,requests, os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
_DEBUG = os.getenv("DEBUG", "False")

st.set_page_config(page_title="PDF Chat", layout="wide")

with st.sidebar:
    st.header("Settings")
    API_BASE = st.text_input("API base URL", value="http://localhost:8000")
    TOP_K = st.number_input("Top K (retrieve)", min_value=1, max_value=20, value=6)
    RRF_K = st.number_input("RRF k (fusion)", min_value=10, max_value=200, value=60)
    st.caption("Your FastAPI should expose /ingest/pdf_documents and /query")

st.title("PDF Chat")
st.write("Upload PDFs, then ask questions. The app will ingest and index, then use hybrid + rerank RAG to answer.")

def _pack_history(max_history = 8):
    if "chat_history" not in st.session_state:
        return []
    return [
        {"role": m["role"], "content": (m["content"] or "")}
        for m in st.session_state.chat_history[-max_history:]
        if m["role"] in ("user","assistant")
    ]

def _post_ingest_PDFs(api_base, files):
    url = f"{api_base.rstrip('/')}/ingest/pdf_documents"
    mp = []
    for f in files:
        mp.append(("files", (f.name, io.BytesIO(f.getvalue()), "application/pdf")))
    response = requests.post(url, files=mp, timeout=60)
    response.raise_for_status()
    return response.json()

def _post_query(api_base, query, top_k, rrf_k):
    url = f"{api_base.rstrip('/')}/query"
    payload = {
        "query": query, 
        "top_k": top_k, 
        "rrf_k": rrf_k,
        "history": _pack_history()
    }
    response = requests.post(url, json=payload, timeout=60)

    if not response.ok:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise RuntimeError(f"/query failed: {detail}")

    return response.json()


def _render_sources(sources):
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        with st.expander(f"[S{s.get('rank')}] {s.get('document_name')} {s.get('page_num')}"):
            st.write(s.get("text", ""))
            score = s.get("scores", {})
            if score:
                st.caption(
                     "scores – "
                    + ", ".join(f"{k}: {round(v,4) if isinstance(v,(int,float)) else v}"
                                for k, v in score.items() if v is not None)
                )

def _ensure_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

with st.container(border=True):
    st.subheader("Upload PDFs")
    files = st.file_uploader(
        "Drop one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        ingest_button = st.button("Ingest", disabled=not files)
    with col2:
        st.caption("After ingest, your background worker will parse/chunk/emb/index. You can start chatting right away.")

    if ingest_button and files:
        try:
            with st.spinner("Uploading..."):
                results = _post_ingest_PDFs(API_BASE, files)
            st.success(f"Uploaded {len(files)} file(s).")

            for r in results:
                status = r.get("status", "unknown")
                message = r.get("message", "")
                st.write(f"- **{r.get('filename')}** → `{status}` ({r.get('pages',0)} pages, {r.get('bytes',0)} bytes) {message}")
        except Exception as e:
            st.error(f"Ingest failed: {e}")

st.divider()

_ensure_history()

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("results"):
            _render_sources(m["results"])

        
user_query = st.chat_input("Ask a question about your documents")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = _post_query(API_BASE, user_query, TOP_K, RRF_K)
            answer = res.get("answer")
            sources = res.get("results", [])
            triggered = res.get("triggered", None)
            reason = res.get("reason", "")
            query_debug = res.get("query_debug", {})

            if not triggered and reason:
                st.caption(f"(No KB search: {reason}")

            if answer:
                st.markdown(answer)

            _render_sources(sources)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer or "",
                "sources": sources
            })

            if _DEBUG == "True":
                with st.expander("Debug (query)"):
                    st.json(query_debug if isinstance(query_debug, dict) else {})
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Query failed: {e}")

