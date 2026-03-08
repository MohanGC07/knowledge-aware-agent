from embeddings.embedding_model import EmbeddingModel
from vector_store.chroma_store import ChromaStore
from config import TOP_K


class Retriever:

    def __init__(self):

        self.embedder = EmbeddingModel()
        self.vector_store = ChromaStore()

    def retrieve(self, query):

        query_embedding = self.embedder.embed_query(query)

        documents = self.vector_store.query(query_embedding, TOP_K)

        return documents