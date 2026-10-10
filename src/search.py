
from document_loader import load_document
from text_processor import clean_text, create_chunks
from embeddings import EmbeddingModel
from vector_store import VectorStore
from bm25_search import BM25Search
from reranker import Reranker
from rag import RAGGenerator


class DocumentSearchEngine:
    def __init__(self):
        """Initialize the document search engine."""

        self.embedding_model = EmbeddingModel()
        self.reranker = Reranker()
        self.rag_generator = RAGGenerator()

        self.vector_store = None
        self.bm25_search = None
        self.documents = []

    def index_document(self, file_path):
        """Load, clean, chunk, embed, and index a document."""

        text = load_document(file_path)
        cleaned_text = clean_text(text)

        chunks = create_chunks(
            cleaned_text,
            chunk_size=20,
            overlap=5
        )

        if not chunks:
            raise ValueError("The document contains no usable text.")

        self.documents = chunks
        print(f"Created {len(chunks)} document chunks.")

        embeddings = self.embedding_model.generate_embeddings(
            chunks
        )

        dimension = embeddings.shape[1]
        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(embeddings, chunks)

        self.bm25_search = BM25Search(chunks)

        print("FAISS index created.")
        print("BM25 index created.")

    def semantic_search(self, query, top_k=10):
        """Retrieve candidates using FAISS."""

        if self.vector_store is None:
            raise ValueError("Index a document before searching.")

        query_embedding = self.embedding_model.generate_embeddings(
            [query]
        )

        return self.vector_store.search(query_embedding, top_k)

    def keyword_search(self, query, top_k=10):
        """Retrieve candidates using BM25."""

        if self.bm25_search is None:
            raise ValueError("Index a document before searching.")

        return self.bm25_search.search(query, top_k)

    def hybrid_search(self, query, top_k=10):
        """Combine BM25 and FAISS rankings using RRF."""

        rrf_constant = 60
        combined_scores = {}

        semantic_results = self.semantic_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)

        for rank, result in enumerate(semantic_results, start=1):
            document = result["document"]

            if document not in combined_scores:
                combined_scores[document] = {
                    "score": 0.0,
                    "semantic_score": result["score"],
                    "keyword_score": 0.0
                }

            combined_scores[document]["score"] += (
                1.0 / (rrf_constant + rank)
            )

        for rank, result in enumerate(keyword_results, start=1):
            document = result["document"]

            if document not in combined_scores:
                combined_scores[document] = {
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "keyword_score": result["score"]
                }
            else:
                combined_scores[document]["keyword_score"] = (
                    result["score"]
                )

            combined_scores[document]["score"] += (
                1.0 / (rrf_constant + rank)
            )

        results = [
            {
                "document": document,
                "score": values["score"],
                "semantic_score": values["semantic_score"],
                "keyword_score": values["keyword_score"]
            }
            for document, values in combined_scores.items()
        ]

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]

    def rerank(self, query, candidates, top_k=5):
        """Improve candidate ordering with a cross-encoder."""

        return self.reranker.rerank(
            query,
            candidates,
            top_k
        )


if __name__ == "__main__":
    search_engine = DocumentSearchEngine()

    search_engine.index_document("data/sample.pdf")

    query = input("\nEnter your search query: ").strip()

    if not query:
        print("Please enter a non-empty search query.")
    else:
        # Stage 1: retrieve candidate chunks.
        candidates = search_engine.hybrid_search(
            query,
            top_k=10
        )

        # Stage 2: rerank the candidates.
        results = search_engine.rerank(
            query,
            candidates,
            top_k=5
        )

        print("\nReranked Search Results")
        print("=" * 60)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"RRF Score: {result['score']:.6f}")
            print(
                f"Reranker Score: "
                f"{result['rerank_score']:.4f}"
            )
            print(f"Text: {result['document']}")
