from pathlib import Path
import json
import time
from statistics import mean

from src.agent import ExplainableRAGAgent
from src.evaluation_metrics import RAGEvaluationMetrics


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation" / "ground_truth.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "evaluation_agent_results.json"

FALLBACK_MARKERS = (
    "i found retrieved passages, but could not identify sufficiently relevant evidence",
    "i could not find sufficiently relevant information",
    "unable to identify meaningful query terms",
)


def is_fallback_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return any(marker in normalized for marker in FALLBACK_MARKERS)


def average_metric(results, metric_name):
    values = [
        item["metrics"].get(metric_name, 0.0)
        for item in results
        if item.get("metrics") is not None
    ]
    return round(mean(values), 4) if values else 0.0


def run_agent_evaluation():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Ground-truth dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("Ground-truth dataset must be a non-empty JSON list.")

    print("\nInitializing Explainable Agent...")
    agent = ExplainableRAGAgent(
        initial_top_k=3,
        max_top_k=10,
        max_iterations=3,
    )

    print("\nInitializing RAG Evaluation Metrics...")
    metrics = RAGEvaluationMetrics(
        embedding_provider=agent.retriever.dense_retriever.embedding_provider
    )

    results = []
    print("\nSTARTING FULL AGENTIC RAG EVALUATION")
    print("=" * 70)

    for index, case in enumerate(test_cases, start=1):
        question = case.get("question", "")
        expected_answer = case.get("ground_truth_answer", "")
        expected_pages = case.get("expected_pages", [])
        test_id = case.get("id", index)

        if not question:
            print(f"\nTest {test_id}: skipped (question is missing)")
            continue

        print(f"\nTest {test_id}: {question}")
        started = time.perf_counter()
        run_error = ""

        try:
            answer = agent.run(question)
        except Exception as exc:
            answer = ""
            run_error = f"{type(exc).__name__}: {exc}"
            print("Agent execution error:", run_error)

        elapsed_seconds = time.perf_counter() - started
        state = agent.last_state
        evidence = getattr(state, "evidence", []) if state else []
        observations = getattr(state, "observations", []) if state else []

        contexts = [
            item.get("text", "")
            for item in evidence
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        retrieved_pages = [
            item.get("page", "Unknown")
            for item in evidence
            if isinstance(item, dict)
        ]

        matching_pages = [
            page for page in retrieved_pages
            if page in expected_pages
        ]
        page_match = bool(matching_pages)
        all_expected_pages_found = (
            all(page in retrieved_pages for page in expected_pages)
            if expected_pages else False
        )

        retrieval_events = [
            event for event in observations
            if isinstance(event, dict)
            and event.get("event") == "retrieval_completed"
        ]
        requested_top_k_sequence = [
            event.get("requested_top_k")
            for event in retrieval_events
        ]

        metric_results = metrics.evaluate(
            query=question,
            answer=answer or "",
            contexts=contexts,
            ground_truth_answer=expected_answer,
        )

        fallback = is_fallback_answer(answer or "")
        agent_status = getattr(state, "status", "error" if run_error else "unknown") if state else "unknown"

        result = {
            "test_id": test_id,
            "question": question,
            "ground_truth_answer": expected_answer,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "matching_expected_pages": matching_pages,
            "page_match": page_match,
            "all_expected_pages_found": all_expected_pages_found,
            "agent_status": agent_status,
            "retrieval_iterations": len(retrieval_events),
            "requested_top_k_sequence": requested_top_k_sequence,
            "evidence_count": len(evidence),
            "answer_fallback": fallback,
            "generated_answer": answer or "",
            "contexts": contexts,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "error": run_error or (getattr(state, "error", "") if state else ""),
            "metrics": metric_results,
        }
        results.append(result)

        print("Agent Status:", agent_status)
        print("Retrieval Top-K Sequence:", requested_top_k_sequence)
        print("Retrieved Pages:", retrieved_pages)
        print("Evidence Count:", len(evidence))
        print("Fallback Answer:", "YES" if fallback else "NO")
        print("Page Validation:", "PASS" if page_match else "FAIL")
        print("Groundedness:", metric_results["groundedness"])
        print("Answer Relevance:", metric_results["answer_relevance"])
        print("Reference Similarity:", metric_results["reference_answer_similarity"])
        if run_error:
            print("Error:", run_error)

    total = len(results)
    if total == 0:
        raise ValueError("No valid evaluation questions were processed.")

    successful_page_hits = sum(1 for item in results if item["page_match"])
    all_pages_hits = sum(1 for item in results if item["all_expected_pages_found"])
    fallback_count = sum(1 for item in results if item["answer_fallback"])
    evidence_found_count = sum(
        1 for item in results if item["agent_status"] == "evidence_found"
    )

    summary = {
        "system": "Explainable Agentic RAG",
        "evaluation_mode": "full_agent_end_to_end",
        "total_questions": total,
        "successful_page_hits": successful_page_hits,
        "page_hit_rate": round(successful_page_hits / total, 4),
        "all_expected_pages_found_count": all_pages_hits,
        "all_expected_pages_hit_rate": round(all_pages_hits / total, 4),
        "agent_evidence_found_count": evidence_found_count,
        "fallback_answer_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 4),
        "average_retrieval_iterations": round(
            mean(item["retrieval_iterations"] for item in results), 4
        ),
        "average_elapsed_seconds": round(
            mean(item["elapsed_seconds"] for item in results), 4
        ),
        "average_context_relevance": average_metric(results, "context_relevance"),
        "average_groundedness": average_metric(results, "groundedness"),
        "average_answer_relevance": average_metric(results, "answer_relevance"),
        "average_reference_answer_similarity": average_metric(
            results, "reference_answer_similarity"
        ),
        "individual_results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("FULL AGENTIC RAG EVALUATION COMPLETED")
    print("=" * 70)
    print("Total Questions:", total)
    print("Page Hit Rate:", round(summary["page_hit_rate"] * 100, 2), "%")
    print("All Expected Pages Hit Rate:", round(summary["all_expected_pages_hit_rate"] * 100, 2), "%")
    print("Fallback Answers:", fallback_count)
    print("Average Retrieval Iterations:", summary["average_retrieval_iterations"])
    print("Average Context Relevance:", summary["average_context_relevance"])
    print("Average Groundedness:", summary["average_groundedness"])
    print("Average Answer Relevance:", summary["average_answer_relevance"])
    print("Average Reference Answer Similarity:", summary["average_reference_answer_similarity"])
    print("\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    run_agent_evaluation()
