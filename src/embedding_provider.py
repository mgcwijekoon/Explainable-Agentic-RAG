from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.interfaces import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Sequence[str],
        normalize: bool = True
    ) -> np.ndarray:

        if not texts:
            raise ValueError("Text input cannot be empty!")

        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )

        return np.asarray(vectors, dtype=np.float32)