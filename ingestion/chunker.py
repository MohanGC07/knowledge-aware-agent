from config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:

    def chunk_text(self, documents):

        chunks = []

        for doc in documents:

            start = 0

            while start < len(doc):

                end = start + CHUNK_SIZE

                chunk = doc[start:end]

                chunks.append(chunk)

                start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks