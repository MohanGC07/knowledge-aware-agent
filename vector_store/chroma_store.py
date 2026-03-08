import chromadb
from config import CHROMA_DB_PATH


class ChromaStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        self.collection = self.client.get_or_create_collection(
            name="knowledge_base"
        )

    def add_documents(self, chunks, embeddings):

        ids = [str(i) for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

    def query(self, embedding, top_k):

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        return results["documents"][0]