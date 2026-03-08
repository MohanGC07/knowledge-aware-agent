from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, chunks):

        return self.model.encode(chunks).tolist()

    def embed_query(self, query):

        return self.model.encode(query).tolist()