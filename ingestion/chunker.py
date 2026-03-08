import logging
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class TextChunker:

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def chunk_text(self, documents: list[str]) -> list[str]:
        all_chunks: list[str] = []
        for doc_index, doc in enumerate(documents):
            if not doc or not doc.strip():
                logger.debug("Document %d is empty — skipping.", doc_index)
                continue
            doc_chunks = self._chunk_single(doc)
            logger.debug("Document %d → %d chunks.", doc_index, len(doc_chunks))
            all_chunks.extend(doc_chunks)
        logger.info("Total chunks produced: %d", len(all_chunks))
        return all_chunks

    def _chunk_single(self, text: str) -> list[str]:
        chunks: list[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self._step
        return chunks