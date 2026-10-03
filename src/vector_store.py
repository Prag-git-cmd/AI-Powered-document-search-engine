import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension):
        """
        Create a FAISS index for storing document embeddings.
        """

        self.dimension = dimension

        # Inner product will be used to measure similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # Keep the original text chunks separately.
        self.documents = []

    def add_documents(self, embeddings, documents):
        """
        Add document embeddings and their corresponding text chunks.
        """

        embeddings = np.asarray(embeddings, dtype="float32")

        # Normalize vectors so inner product becomes cosine similarity.
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        self.documents.extend(documents)

    def search(self, query_embedding, top_k=3):
        """
        Search for the most similar document chunks.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        query_embedding = query_embedding.reshape(1, -1)

        # Normalize the query vector.
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index != -1:
                results.append({
                    "document": self.documents[index],
                    "score": float(score)
                })

        return results
