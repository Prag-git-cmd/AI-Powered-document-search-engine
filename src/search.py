from document_loader import load_document
from text_processor import clean_text, create_chunks
from embeddings import EmbeddingModel
from vector_store import VectorStore
from bm25_search import BM25Search


class DocumentSearchEngine:

    def __init__(self):
        """
        Initialize the embedding model.
        """

        self.embedding_model = EmbeddingModel()

        self.vector_store = None
        self.bm25_search = None

        self.documents = []

    def index_document(self, file_path):
        """
        Load, clean, chunk and index a document
        using both FAISS and BM25.
        """

        # 1. Load document
        text = load_document(file_path)

        # 2. Clean extracted text
        cleaned_text = clean_text(text)

        # 3. Create overlapping chunks
        chunks = create_chunks(
            cleaned_text,
            chunk_size=20,
            overlap=5
        )

        self.documents = chunks

        print(f"Created {len(chunks)} document chunks.")

        # 4. Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(
            chunks
        )

        # 5. Create FAISS vector store
        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(dimension)

        # 6. Store embeddings in FAISS
        self.vector_store.add_documents(
            embeddings,
            chunks
        )

        # 7. Create BM25 index
        self.bm25_search = BM25Search(chunks)

        print("FAISS index created.")
        print("BM25 index created.")

    def semantic_search(self, query, top_k=5):
        """
        Search using FAISS semantic similarity.
        """

        query_embedding = (
            self.embedding_model.generate_embeddings(
                [query]
            )
        )

        return self.vector_store.search(
            query_embedding,
            top_k
        )

    def keyword_search(self, query, top_k=5):
        """
        Search using BM25 keyword matching.
        """

        return self.bm25_search.search(
            query,
            top_k
        )

    def hybrid_search(self, query, top_k=5):
    """
    Combine BM25 and FAISS rankings using
    Reciprocal Rank Fusion (RRF).
    """

    # Retrieve more candidates from both systems
    # before performing the final ranking.
    retrieval_k = 10

    semantic_results = self.semantic_search(
        query,
        retrieval_k
    )

    keyword_results = self.keyword_search(
        query,
        retrieval_k
    )

    # RRF constant.
    k = 60

    combined_scores = {}

    # Process FAISS ranking.
    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        document = result["document"]

        rrf_score = 1 / (k + rank)

        if document not in combined_scores:
            combined_scores[document] = {
                "score": 0.0,
                "semantic_score": result["score"],
                "keyword_score": 0.0
            }

        combined_scores[document]["score"] += rrf_score

    # Process BM25 ranking.
    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        document = result["document"]

        rrf_score = 1 / (k + rank)

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

        combined_scores[document]["score"] += rrf_score

    # Convert dictionary into a list.
    results = []

    for document, values in combined_scores.items():

        results.append({
            "document": document,
            "score": values["score"],
            "semantic_score": values["semantic_score"],
            "keyword_score": values["keyword_score"]
        })

    # Highest RRF score first.
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    search_engine = DocumentSearchEngine()

    search_engine.index_document(
        "data/sample.pdf"
    )

    query = input(
        "\nEnter your search query: "
    )

    results = search_engine.hybrid_search(
        query,
        top_k=5
    )

    print("\nHybrid Search Results")
    print("=" * 60)

    for i, result in enumerate(results):

        print(f"\nResult {i + 1}")

        print(
            f"Hybrid Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Keyword Score: "
            f"{result['keyword_score']:.4f}"
        )

        print(
            f"Text: {result['document']}"
        )
