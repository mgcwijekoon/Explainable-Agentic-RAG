from pathlib import Path
from typing import Any

import numpy as np
import chromadb

from src.interfaces import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ds205_documents"


class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME
    ):

        self.client = chromadb.PersistentClient(
            path=str(db_path)
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

    def count(self) -> int:

        return self.collection.count()

    def query(
        self,
        embedding: np.ndarray,
        top_k: int
    ) -> list[dict[str, Any]]:

        if top_k <= 0:
            raise ValueError("top_k must be positive!")

        if self.count() == 0:
            return []

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        if embedding.ndim != 1:
            raise ValueError(
                "Embedding must be a 1D vector!"
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                "Embedding contains invalid values!"
            )

        results = self.collection.query(
            query_embeddings=[
                embedding.tolist()
            ],
            n_results=min(top_k, self.count()),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved = []

        for index in range(len(results["ids"][0])):

            retrieved.append({

                "chunk_id": results["ids"][0][index],

                "text": results["documents"][0][index],

                "metadata": results["metadatas"][0][index],

                "distance": results["distances"][0][index]

            })

        return retrieved