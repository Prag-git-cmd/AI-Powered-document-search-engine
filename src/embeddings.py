from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        """
        Load the pre-trained sentence transformer model.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, texts):
        """
        Convert a list of text chunks into numerical vectors.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings
