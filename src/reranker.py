
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        """Load a pretrained cross-encoder model."""

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, candidates, top_k=5):
        """Rerank retrieved chunks by query relevance."""

        if not candidates:
            return []

        # Create query-document pairs.
        pairs = [
            (query, candidate["document"])
            for candidate in candidates
        ]

        # Predict a relevance score for each pair.
        scores = self.model.predict(pairs)

        # Attach each reranking score to its candidate.
        reranked_results = []

        for candidate, score in zip(candidates, scores):
            result = candidate.copy()
            result["rerank_score"] = float(score)
            reranked_results.append(result)

        # Highest cross-encoder score first.
        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]
