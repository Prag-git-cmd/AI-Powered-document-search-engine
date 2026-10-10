
from document_loader import load_document
from text_processor import clean_text, create_chunks
from embeddings import EmbeddingModel
from vector_store import VectorStore
from bm25_search import BM25Search


class DocumentSearchEngine:
    def __init__(self):
        """Initialize the document search engine."""
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self.bm25_search = None
        self.documents = []

    def index_document(self, file_path):
        """Load, clean, chunk, embed, and index a document."""

        # 1. Load the PDF or TXT document.
        text = load_document(file_path)

        # 2. Clean the extracted text.
        cleaned_text = clean_text(text)

        # 3. Create overlapping chunks.
        chunks = create_chunks(
            cleaned_text,
            chunk_size=20,
            overlap=5
        )

        if not chunks:
            raise ValueError("The document contains no usable text.")

        self.documents = chunks
        print(f"Created {len(chunks)} document chunks.")

        # 4. Generate embeddings for all chunks.
        embeddings = self.embedding_model.generate_embeddings(
            chunks
        )

        # 5. Create and populate the FAISS index.
        dimension = embeddings.shape[1]
        self.vector_store = VectorStore(dimension)

        self.vector_store.add_documents(
            embeddings,
            chunks
        )

        # 6. Create the BM25 keyword index.
        self.bm25_search = BM25Search(chunks)

        print("FAISS index created.")
        print("BM25 index created.")

    def semantic_search(self, query, top_k=10):
        """Retrieve chunks using FAISS semantic similarity."""

        if self.vector_store is None:
            raise ValueError("Index a document before searching.")

        query_embedding = self.embedding_model.generate_embeddings(
            [query]
        )

        return self.vector_store.search(
            query_embedding,
            top_k
        )

    def keyword_search(self, query, top_k=10):
        """Retrieve chunks using BM25 keyword matching."""

        if self.bm25_search is None:
            raise ValueError("Index a document before searching.")

        return self.bm25_search.search(
            query,
            top_k
        )

    def hybrid_search(self, query, top_k=5):
        """Combine BM25 and FAISS rankings using RRF."""

        retrieval_k = 10
        rrf_constant = 60
        combined_scores = {}

        # 1. Retrieve semantic search results.
        semantic_results = self.semantic_search(
            query,
            retrieval_k
        )

        # 2. Retrieve keyword search results.
        keyword_results = self.keyword_search(
            query,
            retrieval_k
        )

        # 3. Add FAISS reciprocal-rank contributions.
        for rank, result in enumerate(
            semantic_results, start=1
        ):
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

        # 4. Add BM25 reciprocal-rank contributions.
        for rank, result in enumerate(
            keyword_results, start=1
        ):
            document = result["document"]

            if document not in combined_scores:
                combined_scores[document] = {
                    "score": 0.0,
                    "semantic_score": 0.0,
                    "keyword_score": result["score"]
                }
            else:
                combined_scores[document][
                    "keyword_score"
                ] = result["score"]

            combined_scores[document]["score"] += (
                1.0 / (rrf_constant + rank)
            )

        # 5. Convert the combined scores into a list.
        results = []

        for document, values in combined_scores.items():
            results.append({
                "document": document,
                "score": values["score"],
                "semantic_score": values["semantic_score"],
                "keyword_score": values["keyword_score"]
            })

        # 6. Rank documents by their RRF scores.
        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]


if __name__ == "__main__":
    search_engine = DocumentSearchEngine()

    # Index the PDF stored in the data directory.
    search_engine.index_document("data/sample.pdf")

    query = input("\nEnter your search query: ").strip()

    if not query:
        print("Please enter a non-empty search query.")
    else:
        results = search_engine.hybrid_search(
            query,
            top_k=5
        )

        print("\nHybrid Search Results")
        print("=" * 60)

        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"RRF Score: {result['score']:.6f}")
            print(
                f"Semantic Score: "
                f"{result['semantic_score']:.4f}"
            )
            print(
                f"BM25 Score: "
                f"{result['keyword_score']:.4f}"
            )
            print(f"Text: {result['document']}")
