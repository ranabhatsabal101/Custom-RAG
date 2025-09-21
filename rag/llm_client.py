import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from mistralai import Mistral
from typing import Protocol, List, Dict, Any


load_dotenv(find_dotenv(), override=True)

class LLMClient(Protocol):
    def chat_query(self, messages: List[Dict[str, str]], **kwargs) -> str: ...

class MistralChatClient:
    def __init__(self,
                 api_key = None,
                 chat_model = None):
        self.api_key = api_key or os.environ["MISTRAL_API_KEY"]
        self.chat_model = chat_model or os.getenv("MISTRAL_CHAT_MODEL",  "magistral-small-2509")
        self._client = Mistral(api_key=self.api_key)
    
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

    def chat_query(self, messages, **kwargs) -> str:
        response = self._client.chat.complete(model=self.chat_model, messages=messages, **kwargs)
        return self._content_to_text(response.choices[0].message.content)
    
def get_llm_client() -> LLMClient:
    return MistralChatClient()