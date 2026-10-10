
from search import DocumentSearchEngine


# Small preliminary evaluation dataset.
# Each expected phrase should occur in a relevant passage
# from data/sample.pdf.
TEST_CASES = [
    {
        "query": "What role does RAG play in LLMs?",
        "expected_phrase": "cornerstone for future enhancements in LLMs",
    },
    {
        "query": "What does the RAG framework say about modular design?",
        "expected_phrase": "modular design",
    },
    {
        "query": "What research discusses RAG across heterogeneous knowledge?",
        "expected_phrase": "retrieval-augmented generation across heterogeneous knowledge",
    },
]


def is_relevant(document, expected_phrase):
    """Check whether a retrieved passage contains its labeled phrase."""
    return expected_phrase.casefold() in document.casefold()


def evaluate_search(search_engine, test_cases, k=5):
    total_recall = 0.0
    total_reciprocal_rank = 0.0

    print(f"\n===== RETRIEVAL EVALUATION (k={k}) =====")

    for case in test_cases:
        query = case["query"]
        expected_phrase = case["expected_phrase"]

        candidates = search_engine.hybrid_search(
            query,
            top_k=10,
        )

        results = search_engine.rerank(
            query,
            candidates,
            top_k=k,
        )

        relevant_ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if is_relevant(result["document"], expected_phrase)
        ]

        # One labeled relevant phrase per query.
        recall_at_k = 1.0 if relevant_ranks else 0.0
        reciprocal_rank = (
            1.0 / relevant_ranks[0]
            if relevant_ranks
            else 0.0
        )

        total_recall += recall_at_k
        total_reciprocal_rank += reciprocal_rank

        print(f"\nQuery: {query}")
        print(f"Expected phrase: {expected_phrase}")
        print(f"Relevant passage ranks: {relevant_ranks}")
        print(f"Recall@{k}: {recall_at_k:.2f}")
        print(f"Reciprocal rank: {reciprocal_rank:.2f}")

    count = len(test_cases)

    if count == 0:
        print("No evaluation queries were provided.")
        return

    print("\n===== SUMMARY =====")
    print(f"Queries evaluated: {count}")
    print(f"Recall@{k}: {total_recall / count:.3f}")
    print(f"MRR@{k}: {total_reciprocal_rank / count:.3f}")


if __name__ == "__main__":
    engine = DocumentSearchEngine()
    engine.index_document("data/sample.pdf")

    evaluate_search(engine, TEST_CASES, k=5)

