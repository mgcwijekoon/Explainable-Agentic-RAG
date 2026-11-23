from pathlib import Path
import json

import numpy as np

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.interfaces import RetrieverInterface
from src.retriever import DocumentRetriever


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"


class HybridRetriever(RetrieverInterface):

    def __init__(
        self,
        dense_retriever=None,
        dense_weight=0.4,
        rrf_k=10
    ):

        if not 0 <= dense_weight <= 1:
            raise ValueError("Dense weight must be between 0 and 1!")

        if rrf_k <= 0:
            raise ValueError("RRF k must be positive!")

        # =====================================
        # CONFIGURATION
        # =====================================

        self.dense_weight = dense_weight

        self.tfidf_weight = 0.3

        self.bm25_weight = 0.3

        self.rrf_k = rrf_k

        # =====================================
        # DEPENDENCY INJECTION
        # =====================================

        self.dense_retriever = (
            dense_retriever
            if dense_retriever is not None
            else DocumentRetriever(show_math_trace=False)
        )

        # =====================================
        # LOAD DOCUMENT CHUNKS
        # =====================================

        if not CHUNKS_PATH.exists():
            raise FileNotFoundError("chunks.json not found!")

        with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if not self.chunks:
            raise ValueError("No document chunks found!")

        self.chunk_ids = [
            chunk["chunk_id"]
            for chunk in self.chunks
        ]

        self.chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        self.chunk_index = {
            chunk_id: index
            for index, chunk_id in enumerate(self.chunk_ids)
        }

        self.texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # =====================================
        # TF-IDF INITIALIZATION
        # =====================================

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            lowercase=True,
            dtype=np.float32
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.texts
        )

        # =====================================
        # BM25 INITIALIZATION
        # =====================================

        tokenized_corpus = [
            self.tokenize(text)
            for text in self.texts
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Hybrid Retriever Initialized!")
        print("Retrieval methods: Dense + TF-IDF + BM25")
        print("Total Documents:", len(self.chunks))

    # =========================================
    # TEXT TOKENIZATION
    # =========================================

    @staticmethod
    def tokenize(text):

        return [
            token.lower()
            for token in text.split()
            if token.strip()
        ]

    # =========================================
    # SEARCH
    # =========================================

    def search(self, query, top_k=3):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty!")

        if top_k <= 0:
            raise ValueError("top_k must be positive!")

        total_documents = len(self.chunks)

        candidate_k = min(
            total_documents,
            max(top_k * 5, 10)
        )

        # =====================================
        # STEP 1: DENSE RETRIEVAL
        # =====================================

        dense_results = self.dense_retriever.search(
            query,
            top_k=candidate_k
        )

        dense_rank = {
            result["chunk_id"]: rank
            for rank, result in enumerate(dense_results, start=1)
        }

        dense_lookup = {
            result["chunk_id"]: result
            for result in dense_results
        }

        # =====================================
        # STEP 2: TF-IDF RETRIEVAL
        # =====================================

        query_vector = self.vectorizer.transform([query])

        sparse_scores = (
            self.tfidf_matrix @ query_vector.T
        ).toarray().ravel()

        sparse_indices = np.argsort(-sparse_scores)

        sparse_indices = [
            index
            for index in sparse_indices
            if sparse_scores[index] > 0
        ][:candidate_k]

        sparse_rank = {
            self.chunk_ids[index]: rank
            for rank, index in enumerate(sparse_indices, start=1)
        }

        # =====================================
        # STEP 3: BM25 RETRIEVAL
        # =====================================

        tokenized_query = self.tokenize(query)

        bm25_scores = np.asarray(
            self.bm25.get_scores(tokenized_query),
            dtype=np.float32
        )

        bm25_indices = np.argsort(-bm25_scores)

        bm25_indices = [
            index
            for index in bm25_indices
            if bm25_scores[index] > 0
        ][:candidate_k]

        bm25_rank = {
            self.chunk_ids[index]: rank
            for rank, index in enumerate(bm25_indices, start=1)
        }

        # =====================================
        # STEP 4: COMBINE CANDIDATES
        # =====================================

        candidate_ids = (
            set(dense_rank)
            | set(sparse_rank)
            | set(bm25_rank)
        )

        hybrid_results = []

        for chunk_id in candidate_ids:

            d_rank = dense_rank.get(chunk_id)

            s_rank = sparse_rank.get(chunk_id)

            b_rank = bm25_rank.get(chunk_id)

            # ---------------------------------
            # Dense RRF
            # ---------------------------------

            dense_score = (
                self.dense_weight / (self.rrf_k + d_rank)
                if d_rank is not None
                else 0.0
            )

            # ---------------------------------
            # TF-IDF RRF
            # ---------------------------------

            tfidf_score = (
                self.tfidf_weight / (self.rrf_k + s_rank)
                if s_rank is not None
                else 0.0
            )

            # ---------------------------------
            # BM25 RRF
            # ---------------------------------

            bm25_score = (
                self.bm25_weight / (self.rrf_k + b_rank)
                if b_rank is not None
                else 0.0
            )

            # ---------------------------------
            # Final Combined Score
            # ---------------------------------

            hybrid_score = (
                dense_score
                + tfidf_score
                + bm25_score
            )

            chunk = self.chunk_lookup[chunk_id]

            dense_info = dense_lookup.get(chunk_id)

            dense_similarity = None

            if dense_info is not None:

                if "cosine_similarity" in dense_info:

                    dense_similarity = float(
                        dense_info["cosine_similarity"]
                    )

                elif "distance" in dense_info:

                    dense_similarity = float(
                        1.0 - dense_info["distance"]
                    )

            chunk_index = self.chunk_index[chunk_id]

            tfidf_similarity = float(
                sparse_scores[chunk_index]
            )

            bm25_similarity = float(
                bm25_scores[chunk_index]
            )

            hybrid_results.append({

                "chunk_id": chunk_id,

                "text": chunk["text"],

                "metadata": chunk["metadata"],

                "dense_rank": d_rank,

                "sparse_rank": s_rank,

                "bm25_rank": b_rank,

                "dense_similarity": dense_similarity,

                "tfidf_similarity": tfidf_similarity,

                "bm25_similarity": bm25_similarity,

                "dense_score": dense_score,

                "sparse_score": tfidf_score,

                "bm25_score": bm25_score,

                "hybrid_score": hybrid_score

            })

        # =====================================
        # STEP 5: FINAL RANKING
        # =====================================

        hybrid_results.sort(
            key=lambda item: (
                -item["hybrid_score"],
                item["chunk_id"]
            )
        )

        return hybrid_results[:top_k]


# =============================================
# TESTING
# =============================================

if __name__ == "__main__":

    retriever = HybridRetriever(
        dense_weight=0.4,
        rrf_k=10
    )

    question = "What is the maximum number of members in each group?"

    print("\nUSER QUESTION:")
    print(question)

    results = retriever.search(
        question,
        top_k=10
    )

    print("\nHYBRID RETRIEVAL RESULTS")

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Chunk:", result["chunk_id"])

        print("Page:", result["metadata"].get("page"))

        print("Dense Rank:", result["dense_rank"])

        print("TF-IDF Rank:", result["sparse_rank"])

        print("BM25 Rank:", result["bm25_rank"])

        print("Dense Similarity:", result["dense_similarity"])

        print("TF-IDF Similarity:", round(result["tfidf_similarity"], 4))

        print("BM25 Score:", round(result["bm25_similarity"], 4))

        print("Hybrid Score:", round(result["hybrid_score"], 5))

        print("Text:", result["text"][:400])