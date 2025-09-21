from typing import Protocol, List
from mistralai import Mistral

import numpy as np
import os

class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...

class MistralEmbedder:
    def __init__(self, model: str | None = None, batch_size: int = 128):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set; cannot use Mistral embeddings.")
        
        self.client = Mistral(api_key=api_key)
        self.model = model or os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
        self.batch_size = batch_size

    def embed(self, texts) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            result = self.client.embeddings.create(model=self.model, inputs=batch)
            out.extend([d.embedding for d in result.data])

        arr = np.asarray(out, dtype="float32")

        # The output from Mistral should be normalized but just in case, it is normed here again
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr  = arr / np.clip(norms, 1e-12, None)
        return arr
    

def get_embedder(model = None) -> Embedder:
    return MistralEmbedder(model)