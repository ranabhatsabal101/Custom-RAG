from rag.llm_client import get_llm_client
from rag.retriever import Retriever

CONTEXT_BUDGET_CHARS = 4000
LLM_CLIENT = get_llm_client()
RETRIEVER = Retriever()

_SYSTEM_WITH_CONTEXT = """You are a helpful assistant. Use ONLY the provided context to answer.
If the answer cannot be found in the context, say that you do not know and suggest next steps.
Cite sources inline as [S{N}] using the source numbers provided.
Be concise and directly helpful to the user's intent."""

_SYSTEM_NO_CONTEXT = """You are a helpful assistant. Answer clearly and concisely for the user."""


class ChatAssitant:
    def __init__(self):
        self.llm_client = LLM_CLIENT
        self.retriever = RETRIEVER

    def answer(self, rag_trigger, rag_vectors, query, **kwargs):
        if rag_trigger and rag_vectors:
            contexts = []
            for r in rag_vectors:
                contexts.append(f"[S{r.rank}] {r.document_name} {r.page_num}")
                contexts.append(r.text.strip())
                contexts.append("")
            ctx = "\n".join(contexts).strip()
            messages = [
                {"role": "system", "content": _SYSTEM_WITH_CONTEXT},
                {"role": "user", "content": f"Context:\n{ctx}\n\nUser question: {query}"},
            ]
        else:
            messages = [
                {"role": "system", "content": _SYSTEM_NO_CONTEXT},
                {"role": "user", "content": query},
            ]
        response = self.llm_client.chat_query(messages, **kwargs)
        return response
    
def get_chat_assistant():
    return ChatAssitant()