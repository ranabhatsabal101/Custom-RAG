from rag.llm_client import get_llm_client
from rag.retriever import get_retriever

LLM_CLIENT = get_llm_client()
RETRIEVER = get_retriever()

SYSTEM_WITH_RAG = """You are a helpful assistant that is triggered to answer queries that it is RAG
based. Hence, use ONLY the provided context to answer and do not use any outside information at all.
If the answer cannot be found in the context, say that you do not know and suggest that you only answer quesetions
based on uploaded documents and the information you have so far is not sufficient to answer that.
Cite sources inline as [document_name, page_num] using the document name and page num provided.
Be concise and directly helpful to the user's intent."""

SYSTEM_WITHOUT_RAG = """You are a helpful assistant that is triggered to do small talks. If the question is about
any factual information or anything else other than small talks, do not answer the question and mention you
only answer based off of your knowledge base, which are uploaded documents."""


class ChatAssitant:
    def __init__(self):
        self.llm_client = LLM_CLIENT
        self.retriever = RETRIEVER

    def answer(self, rag_trigger, rag_vectors, query, **kwargs):
        if rag_trigger and rag_vectors:
            contexts = []
            for r in rag_vectors:
                contexts.append(f"Rank=[S{r.rank}] Document_Name={r.document_name} Page_Num={r.page_num}")
                contexts.append(r.text.strip())
                contexts.append("")
            ctx = "\n".join(contexts).strip()
            messages = [
                {"role": "system", "content": SYSTEM_WITH_RAG},
                {"role": "user", "content": f"Context:\n{ctx}\n\nUser question: {query}"},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_WITHOUT_RAG},
                {"role": "user", "content": query},
            ]
        response = self.llm_client.chat_query(messages, **kwargs)
        return response
    
def get_chat_assistant():
    return ChatAssitant()