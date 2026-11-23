import numpy as np

from src.interfaces import (
    EmbeddingProvider,
    VectorStore,
    RetrieverInterface
)

from src.embedding_provider import SentenceTransformerProvider
from src.chroma_vector_store import ChromaVectorStore


class DocumentRetriever(RetrieverInterface):

    def __init__(
        self,
        embedding_provider: EmbeddingProvider = None,
        vector_store: VectorStore = None,
        show_math_trace: bool = True
    ):

        # Dependency Injection
        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else SentenceTransformerProvider()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore()
        )

        self.show_math_trace = show_math_trace

        print("Document Retriever initialized!")
        print("Available documents:", self.vector_store.count())

    def search(self, query, top_k=3):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty!")

        if top_k <= 0:
            raise ValueError("top_k must be positive!")

        if self.vector_store.count() == 0:
            return []

        # STEP 1: Generate query embedding
        raw_query = self.embedding_provider.encode(
            [query],
            normalize=False
        )[0]

        raw_query = np.asarray(
            raw_query,
            dtype=np.float32
        )

        # STEP 2: Calculate query magnitude
        query_magnitude = float(
            np.linalg.norm(raw_query)
        )

        if query_magnitude == 0:
            raise ValueError("Query vector has zero magnitude!")

        # STEP 3: Normalize query vector
        normalized_query = (
            raw_query / query_magnitude
        )

        normalized_query_magnitude = float(
            np.linalg.norm(normalized_query)
        )

        # STEP 4: Retrieve from ChromaDB
        results = self.vector_store.query(
            normalized_query,
            top_k
        )

        if not results:
            return []

        # Calculate similarity directly from ChromaDB distance.
        # ChromaDB uses cosine distance = 1 - cosine similarity.

        for result in results:

            distance = float(result["distance"])

            result["cosine_similarity"] = float(
                np.clip(1.0 - distance, -1.0, 1.0)
            )

        # STEP 5: Mathematical tracing
        # Expensive document re-embedding happens ONLY
        # when mathematical tracing is enabled.

        if self.show_math_trace:

            documents = [
                result["text"]
                for result in results
            ]

            raw_documents = self.embedding_provider.encode(
                documents,
                normalize=False
            )

            print("\n========== MATHEMATICAL TRACE ==========")

            print("Query:", query)

            print(
                "Raw Query Magnitude:",
                round(query_magnitude, 6)
            )

            print(
                "Normalized Query Magnitude:",
                round(normalized_query_magnitude, 6)
            )

            # STEP 6: Document mathematical calculations

            for index, result in enumerate(results):

                doc_vector = np.asarray(
                    raw_documents[index],
                    dtype=np.float32
                )

                doc_magnitude = float(
                    np.linalg.norm(doc_vector)
                )

                if doc_magnitude == 0:
                    continue

                normalized_doc = (
                    doc_vector / doc_magnitude
                )

                normalized_doc_magnitude = float(
                    np.linalg.norm(normalized_doc)
                )

                # Dot Product
                dot_product = float(
                    np.dot(
                        normalized_query,
                        normalized_doc
                    )
                )

                # Cosine Similarity
                cosine_similarity = float(
                    np.clip(
                        dot_product / (
                            normalized_query_magnitude
                            * normalized_doc_magnitude
                        ),
                        -1.0,
                        1.0
                    )
                )

                calculated_distance = (
                    1.0 - cosine_similarity
                )

                result.update({

                    "query_magnitude": query_magnitude,

                    "document_magnitude": doc_magnitude,

                    "normalized_document_magnitude":
                        normalized_doc_magnitude,

                    "dot_product": dot_product,

                    "cosine_similarity": cosine_similarity,

                    "calculated_cosine_distance":
                        calculated_distance

                })

                print(f"\n--- Document {index + 1} ---")

                print(
                    "Document Magnitude:",
                    round(doc_magnitude, 6)
                )

                print(
                    "Normalized Document Magnitude:",
                    round(normalized_doc_magnitude, 6)
                )

                print(
                    "Dot Product:",
                    round(dot_product, 6)
                )

                print(
                    "Cosine Similarity:",
                    round(cosine_similarity, 6)
                )

                print(
                    "ChromaDB Distance:",
                    round(result["distance"], 6)
                )

                print(
                    "Calculated Distance:",
                    round(calculated_distance, 6)
                )

            print(
                "\n========== END MATHEMATICAL TRACE =========="
            )

        return results


if __name__ == "__main__":

    provider = SentenceTransformerProvider()

    store = ChromaVectorStore()

    retriever = DocumentRetriever(
        embedding_provider=provider,
        vector_store=store,
        show_math_trace=True
    )

    question = "What are the coursework deliverables?"

    results = retriever.search(
        question,
        top_k=3
    )

    print("\nFINAL RETRIEVAL RESULTS")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Chunk ID:", result["chunk_id"])

        print("Page:", result["metadata"]["page"])

        print(
            "Similarity:",
            round(result["cosine_similarity"], 4)
        )

        print("Text:", result["text"][:200])