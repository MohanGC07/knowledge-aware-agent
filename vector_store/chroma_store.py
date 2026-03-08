"""
vector_store/chroma_store.py
"""

import logging
import uuid

import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH, TOP_K

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "knowledge_base"


class ChromaStore:

    def __init__(
        self,
        collection_name: str = _COLLECTION_NAME,
        db_path: str = CHROMA_DB_PATH,
    ) -> None:
        self.collection_name = collection_name   # ← this line was missing
        self.db_path = db_path

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaStore initialised — collection '%s' at '%s'.",
            collection_name, db_path,
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            logger.warning("add_documents called with no chunks — nothing stored.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(ids=ids, documents=chunks, embeddings=embeddings)
        logger.info("Stored %d chunk(s) in collection '%s'.", len(chunks), self.collection_name)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K,
    ) -> list[str]:
        count = self._collection.count()
        if count == 0:
            logger.warning("Collection is empty — no documents have been ingested yet.")
            return []

        effective_k = min(top_k, count)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
        )
        documents: list[str] = results.get("documents", [[]])[0]
        logger.debug("Query returned %d chunk(s).", len(documents))
        return documents

    def reset_collection(self) -> None:
        self._client.delete_collection(self.collection_name)   # ← now works
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' has been reset.", self.collection_name)

    def document_count(self) -> int:
        return self._collection.count()