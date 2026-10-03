from rank_bm25 import BM25Okapi


class BM25Search:

    def __init__(self, documents):
        """
        Build a BM25 index from document chunks.
        """

        self.documents = documents

        # Tokenize each document chunk.
        tokenized_documents = [
            document.lower().split()
            for document in documents
        ]

        # Create the BM25 index.
        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, top_k=3):
        """
        Search documents using keyword-based BM25 retrieval.
        """

        # Tokenize the query.
        tokenized_query = query.lower().split()

        # Calculate BM25 scores.
        scores = self.bm25.get_scores(tokenized_query)

        # Sort document indices by score.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "document": self.documents[index],
                "score": float(scores[index])
            })

        return results
