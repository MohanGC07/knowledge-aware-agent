import streamlit as st
import os

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from embeddings.embedding_model import EmbeddingModel
from vector_store.chroma_store import ChromaStore
from retrieval.retriever import Retriever
from generation.generator import Generator

from config import GROQ_API_KEY

if not GROQ_API_KEY:
    st.error("GROQ API key is missing! Check .env or Streamlit secrets.")

UPLOAD_DIR = "data/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Knowledge Aware Agent")

uploaded_files = st.file_uploader(
    "Upload your files",
    accept_multiple_files=True,
    type=["pdf", "txt", "md", "py", "docx", "json", "csv"]
)

if uploaded_files:
    for file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
    st.success("Files uploaded successfully!")

if st.button("Process Documents"):

    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = EmbeddingModel()
    vector_store = ChromaStore()

    all_chunks = []

    for file_name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, file_name)
        text = loader.load_file(path)
        chunks = chunker.chunk_text([text])
        all_chunks.extend(chunks)

    embeddings = embedder.embed_documents(all_chunks)
    vector_store.add_documents(all_chunks, embeddings)

    st.success("Documents processed!")

query = st.text_input("Ask a question")

if query:

    retriever = Retriever()
    generator = Generator()

    docs = retriever.retrieve(query)
    context = "\n".join(docs)
    answer = generator.generate(query, context)

    st.write(answer)