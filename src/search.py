from document_loader import load_document
from text_processor import clean_text, create_chunks
from embeddings import EmbeddingModel
from vector_store import VectorStore


class DocumentSearchEngine:

    def __init__(self):
        """
        Initialize the embedding model.
        """

        self.embedding_model = EmbeddingModel()
        self.vector_store = None

    def index_document(self, file_path):
        """
        Load, clean, chunk and index a document.
        """

        # Step 1: Load document
        text = load_document(file_path)

        # Step 2: Clean text
        cleaned_text = clean_text(text)

        # Step 3: Create chunks
        chunks = create_chunks(
            cleaned_text,
            chunk_size=20,
            overlap=5
        )

        # Step 4: Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(
            chunks
        )

        # Step 5: Create FAISS vector store
        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(dimension)

        # Step 6: Store embeddings
        self.vector_store.add_documents(
            embeddings,
            chunks
        )

        print(f"Indexed {len(chunks)} document chunks.")

    def search(self, query, top_k=3):
        """
        Search the indexed document using semantic similarity.
        """

        if self.vector_store is None:
            raise ValueError(
                "No document has been indexed yet."
            )

        # Convert query into an embedding
        query_embedding = (
            self.embedding_model.generate_embeddings(
                [query]
            )
        )

        # Search FAISS
        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        return results


if __name__ == "__main__":

    search_engine = DocumentSearchEngine()

    search_engine.index_document(
        "data/sample.txt"
    )

    query = input(
        "\nEnter your search query: "
    )

    results = search_engine.search(
        query,
        top_k=3
    )

    print("\nSearch Results")
    print("=" * 50)

    for i, result in enumerate(results):

        print(f"\nResult {i + 1}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Text: {result['document']}")
