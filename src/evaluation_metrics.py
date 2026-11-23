import re
import numpy as np

from src.embedding_provider import SentenceTransformerProvider


class RAGEvaluationMetrics:
    """Sentence-level RAG evaluation metrics.

    The public evaluate() result keys are kept compatible with the existing
    evaluation_rag.py script.
    """

    FALLBACK_MARKERS = (
        "i found retrieved passages but could not identify sufficiently relevant",
        "i could not find sufficiently relevant information",
        "unable to identify meaningful query terms",
    )

    def __init__(self, embedding_provider=None, groundedness_threshold=0.50):
        if not 0.0 <= groundedness_threshold <= 1.0:
            raise ValueError("groundedness_threshold must be between 0 and 1")

        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else SentenceTransformerProvider()
        )
        self.groundedness_threshold = float(groundedness_threshold)

    # --------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------
    def clean_answer(self, text):
        """Remove presentation text and source citations before scoring."""
        if not isinstance(text, str):
            return ""

        text = re.sub(r"\[\s*Source\s*:[^\]]*\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(
            r"(?im)^\s*Based on the retrieved document evidence\s*:\s*", " ", text
        )
        text = re.sub(r"(?im)^\s*(?:Source|Page)\s*:\s*.*$", " ", text)
        text = re.sub(r"(?m)^\s*\d+[.)]\s*", "", text)
        text = re.sub(r"(?m)^\s*[-*•]\s*", "", text)
        return " ".join(text.split()).strip()

    def normalize_text(self, text):
        if not isinstance(text, str):
            return ""
        text = self.clean_answer(text).lower()
        text = re.sub(r"[^\w%]+", " ", text, flags=re.UNICODE)
        return " ".join(text.split())

    def is_fallback_answer(self, answer):
        normalized = self.normalize_text(answer)
        return any(marker in normalized for marker in self.FALLBACK_MARKERS)

    # --------------------------------------------------
    # TEXT SEGMENTATION
    # --------------------------------------------------
    def split_sentences(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [part.strip() for part in parts if len(part.strip()) >= 3]

    def extract_answer_claims(self, answer):
        if not isinstance(answer, str) or not answer.strip():
            return []
        if self.is_fallback_answer(answer):
            return []

        cleaned = self.clean_answer(answer)
        claims = []
        for sentence in self.split_sentences(cleaned):
            sentence = re.sub(r"^\s*\d+[.)]\s*", "", sentence).strip()
            if sentence:
                claims.append(sentence)

        return claims

    def extract_numbers(self, text):
        """Return numeric facts, preserving whether a value is a percentage."""
        normalized = str(text).lower()
        values = set()
        for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*(%)?", normalized):
            number = match.group(1)
            is_percent = bool(match.group(2))
            values.add(f"{number}%" if is_percent else number)
        return values

    # --------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------
    def generate_embeddings(self, texts):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.embedding_provider.encode(texts, normalize=True)
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def cosine_scores(self, left_vectors, right_vectors):
        if left_vectors.size == 0 or right_vectors.size == 0:
            return np.empty((left_vectors.shape[0], right_vectors.shape[0]))
        scores = np.matmul(left_vectors, right_vectors.T)
        return np.clip(scores, 0.0, 1.0)

    # --------------------------------------------------
    # 1. CONTEXT RELEVANCE
    # --------------------------------------------------
    def context_relevance_scores(self, query, contexts):
        if not isinstance(query, str) or not query.strip():
            return []

        valid_contexts = [
            context for context in contexts
            if isinstance(context, str) and context.strip()
        ]
        if not valid_contexts:
            return []

        # Score each retrieved chunk using its most relevant sentence.
        # This avoids diluting a relevant sentence with unrelated text in
        # the same long PDF chunk.
        sentence_groups = [self.split_sentences(context) for context in valid_contexts]
        flat_sentences = []
        group_ranges = []
        for sentences in sentence_groups:
            start = len(flat_sentences)
            flat_sentences.extend(sentences if sentences else [" "])
            group_ranges.append((start, len(flat_sentences)))

        vectors = self.generate_embeddings([query] + flat_sentences)
        query_vector = vectors[0:1]
        sentence_vectors = vectors[1:]
        similarities = self.cosine_scores(query_vector, sentence_vectors)[0]

        scores = []
        for start, end in group_ranges:
            scores.append(float(np.max(similarities[start:end])) if end > start else 0.0)
        return scores

    def context_relevance(self, query, contexts):
        scores = self.context_relevance_scores(query, contexts)
        return float(np.mean(scores)) if scores else 0.0

    # --------------------------------------------------
    # 2. GROUNDEDNESS / FAITHFULNESS
    # --------------------------------------------------
    def groundedness(self, answer, contexts):
        if not isinstance(answer, str) or not answer.strip() or self.is_fallback_answer(answer):
            return 0.0

        claims = self.extract_answer_claims(answer)
        valid_contexts = [
            context for context in contexts
            if isinstance(context, str) and context.strip()
        ]
        context_sentences = []
        for context in valid_contexts:
            context_sentences.extend(self.split_sentences(context))

        if not claims or not context_sentences:
            return 0.0

        vectors = self.generate_embeddings(claims + context_sentences)
        claim_vectors = vectors[:len(claims)]
        context_vectors = vectors[len(claims):]
        similarities = self.cosine_scores(claim_vectors, context_vectors)

        supported = 0
        for claim_index, claim in enumerate(claims):
            claim_numbers = self.extract_numbers(claim)
            best_supported_score = 0.0

            for context_index, context_sentence in enumerate(context_sentences):
                # Semantic similarity alone can incorrectly treat conflicting
                # figures (for example, 30% vs 40%) as equivalent. Require any
                # numbers in a claim to also occur in its supporting sentence.
                context_numbers = self.extract_numbers(context_sentence)
                if claim_numbers and not claim_numbers.issubset(context_numbers):
                    continue

                best_supported_score = max(
                    best_supported_score,
                    float(similarities[claim_index, context_index]),
                )

            if best_supported_score >= self.groundedness_threshold:
                supported += 1

        return float(supported / len(claims))

    # --------------------------------------------------
    # 3. ANSWER RELEVANCE
    # --------------------------------------------------
    def answer_relevance(self, query, answer):
        if not isinstance(query, str) or not query.strip():
            return 0.0
        cleaned_answer = self.clean_answer(answer)
        if not cleaned_answer or self.is_fallback_answer(answer):
            return 0.0

        vectors = self.generate_embeddings([query, cleaned_answer])
        return float(np.clip(np.dot(vectors[0], vectors[1]), 0.0, 1.0))

    # --------------------------------------------------
    # 4. REFERENCE ANSWER SIMILARITY
    # --------------------------------------------------
    def reference_answer_similarity(self, answer, ground_truth_answer):
        cleaned_answer = self.clean_answer(answer)
        cleaned_reference = self.clean_answer(ground_truth_answer)
        if not cleaned_answer or not cleaned_reference or self.is_fallback_answer(answer):
            return 0.0

        vectors = self.generate_embeddings([cleaned_answer, cleaned_reference])
        return float(np.clip(np.dot(vectors[0], vectors[1]), 0.0, 1.0))

    # --------------------------------------------------
    # 5. COMPLETE EVALUATION
    # --------------------------------------------------
    def evaluate(self, query, answer, contexts, ground_truth_answer):
        context_scores = self.context_relevance_scores(query, contexts)

        return {
            "context_relevance": round(
                float(np.mean(context_scores)) if context_scores else 0.0, 4
            ),
            "individual_context_scores": [round(score, 4) for score in context_scores],
            "groundedness": round(self.groundedness(answer, contexts), 4),
            "answer_relevance": round(self.answer_relevance(query, answer), 4),
            "reference_answer_similarity": round(
                self.reference_answer_similarity(answer, ground_truth_answer), 4
            ),
        }


if __name__ == "__main__":
    metrics = RAGEvaluationMetrics()
    print("RAG EVALUATION METRICS LOADED SUCCESSFULLY!")
