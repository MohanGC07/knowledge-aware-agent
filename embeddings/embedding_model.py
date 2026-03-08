import logging
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)

class EmbeddingModel:

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer = _load_model(model_name)

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            logger.warning("embed_documents called with an empty list.")
            return []
        logger.info("Embedding %d chunk(s).", len(chunks))
        vectors = self._model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        vector = self._model.encode(query, show_progress_bar=False, convert_to_numpy=True)
        return vector.tolist()