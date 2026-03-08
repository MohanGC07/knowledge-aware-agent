import logging
from embeddings.embedding_model import EmbeddingModel
from vector_store.chroma_store import ChromaStore
from config import TOP_K

logger = logging.getLogger(__name__)

class Retriever:

    def __init__(self, top_k: int = TOP_K) -> None:
        self.top_k = top_k
        self._embedder = EmbeddingModel()
        self._vector_store = ChromaStore()

    def retrieve(self, query: str) -> list[str]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        logger.info("Retrieving top-%d chunks for query: '%s'", self.top_k, query[:80])
        query_embedding = self._embedder.embed_query(query)
        chunks = self._vector_store.query(query_embedding, self.top_k)
        logger.info("Retrieved %d chunk(s).", len(chunks))
        return chunks