from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


# ==========================================
# 1. EMBEDDING PROVIDER INTERFACE
# ==========================================

class EmbeddingProvider(ABC):
    """
    Interface for converting text into numerical vectors.
    """

    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert text into embeddings.

        Args:
            texts: Input text collection.
            normalize: Whether to normalize vectors.

        Returns:
            NumPy array containing embeddings.
        """
        raise NotImplementedError


# ==========================================
# 2. VECTOR STORE INTERFACE
# ==========================================

class VectorStore(ABC):
    """
    Interface for persistent vector database operations.
    """

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of stored documents.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        top_k: int
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents.

        Args:
            embedding: User query embedding.
            top_k: Maximum number of documents.

        Returns:
            Retrieved documents with metadata and distances.
        """
        raise NotImplementedError


# ==========================================
# 3. RETRIEVER INTERFACE
# ==========================================

class RetrieverInterface(ABC):
    """
    Interface for dense and hybrid retrieval systems.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: User question.
            top_k: Number of results.

        Returns:
            Relevant document chunks.
        """
        raise NotImplementedError


# ==========================================
# 4. ANSWER SYNTHESIZER INTERFACE
# ==========================================

class AnswerSynthesizer(ABC):
    """
    Interface for generating answers from retrieved evidence.
    """

    @abstractmethod
    def synthesize(
        self,
        query: str,
        evidence: list[dict[str, Any]]
    ) -> str:
        """
        Generate an answer using retrieved evidence.

        Args:
            query: User question.
            evidence: Retrieved document chunks.

        Returns:
            Generated answer as a string.
        """
        raise NotImplementedError