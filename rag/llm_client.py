import os
from dotenv import load_dotenv, find_dotenv
from mistralai import Mistral
from typing import Protocol, List, Dict, Any

load_dotenv(find_dotenv(), override=True)

API_KEY = os.getenv("MISTRAL_API_KEY")
CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL")

class LLMClient(Protocol):
    def chat_query(self, messages: List[Dict[str, str]], **kwargs) -> str: ...

class MistralChatClient:
    def __init__(self):
        if not CHAT_MODEL:
            raise RuntimeError("MISTRAL_CHAT_MODEL not set; cannot use Mistral Chat Client.")
        if not API_KEY:
            raise RuntimeError("MISTRAL_API_KEY not set; cannot use Mistral embeddings.")

        self.api_key = API_KEY
        self.chat_model = CHAT_MODEL
        self.client = Mistral(api_key=self.api_key)
    
    def _content_to_text(self, content: Any) -> str:
        # Mistral may return a string OR a list of chunk objects
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: List[str] = []
            for ch in content:
                # chunk objects usually have .type and .text
                typ = getattr(ch, "type", None)
                txt = getattr(ch, "text", None)
                if typ == "text" and txt:
                    texts.append(txt)
            return "\n".join(texts)
        return str(content)

    def chat_query(self, messages, structured=False, **kwargs) -> str:
        if structured:
            response = self.client.chat.parse(model=self.chat_model, messages=messages, **kwargs)
        else:
            response = self.client.chat.complete(model=self.chat_model, messages=messages, **kwargs)
        return self._content_to_text(response.choices[0].message.content)
    
def get_llm_client() -> LLMClient:
    return MistralChatClient()