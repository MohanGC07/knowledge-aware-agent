import logging
import os
import tempfile
import streamlit as st

from config import GROQ_API_KEY, ALLOWED_FILE_TYPES
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from embeddings.embedding_model import EmbeddingModel
from vector_store.chroma_store import ChromaStore
from retrieval.retriever import Retriever
from generation.generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Knowledge-Aware Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = tempfile.mkdtemp(prefix="kaa_docs_")

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()

@st.cache_resource(show_spinner=False)
def get_vector_store() -> ChromaStore:
    return ChromaStore()

def _init_session_state() -> None:
    defaults = {
        "documents_processed": False,
        "processed_file_names": [],
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()

def process_uploaded_files(uploaded_files) -> None:
    loader   = DocumentLoader()
    chunker  = TextChunker()
    embedder = get_embedding_model()
    vector_store = get_vector_store()
    vector_store.reset_collection()

    all_chunks: list[str] = []
    failed_files: list[str] = []
    progress = st.progress(0, text="Processing documents…")

    for idx, uploaded_file in enumerate(uploaded_files):
        progress.progress(idx / len(uploaded_files), text=f"Loading {uploaded_file.name}…")
        suffix = f".{uploaded_file.name.rsplit('.', 1)[-1].lower()}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            text = loader.load_file(tmp_path)
            if not text.strip():
                st.warning(f"⚠️ No text extracted from **{uploaded_file.name}** — skipping.")
                failed_files.append(uploaded_file.name)
                continue
            chunks = chunker.chunk_text([text])
            all_chunks.extend(chunks)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not all_chunks:
        progress.empty()
        st.error("No text could be extracted from any of the uploaded files.")
        return

    progress.progress(0.85, text="Generating embeddings…")
    embeddings = embedder.embed_documents(all_chunks)

    progress.progress(0.95, text="Storing in vector database…")
    vector_store.add_documents(all_chunks, embeddings)

    progress.progress(1.0, text="Done!")
    progress.empty()

    st.session_state.documents_processed = True
    st.session_state.processed_file_names = [
        f.name for f in uploaded_files if f.name not in failed_files
    ]

def answer_query(query: str) -> str:
    retriever = Retriever()
    generator = Generator()
    chunks = retriever.retrieve(query)
    if not chunks:
        return "I couldn't find relevant information. Try rephrasing or uploading more files."
    context = "\n\n---\n\n".join(chunks)
    return generator.generate(query, context)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Knowledge-Aware Agent")
    st.caption("A RAG system built from scratch — no LangChain.")
    st.divider()

    if not GROQ_API_KEY:
        st.error("Groq API key not found. Add it to .env or Streamlit secrets.", icon="🔑")
        st.stop()

    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        label="Drag and drop files here",
        accept_multiple_files=True,
        type=ALLOWED_FILE_TYPES,
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚙️ Process", disabled=not uploaded_files,
                                use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear", disabled=not st.session_state.documents_processed,
                              use_container_width=True)

    if process_btn and uploaded_files:
        process_uploaded_files(uploaded_files)
        if st.session_state.documents_processed:
            st.success(f"✅ Processed {len(st.session_state.processed_file_names)} file(s).")

    if clear_btn:
        get_vector_store().reset_collection()
        st.session_state.documents_processed = False
        st.session_state.processed_file_names = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    if st.session_state.documents_processed:
        st.subheader("📋 Ingested Files")
        for name in st.session_state.processed_file_names:
            st.markdown(f"- `{name}`")
        st.caption(f"Vector store: **{get_vector_store().document_count()}** chunks indexed")
    else:
        st.info("Upload documents and click **Process** to begin.", icon="ℹ️")

# ── Main panel ────────────────────────────────────────────────────────────────
st.header("💬 Ask a Question")

if not st.session_state.documents_processed:
    st.info("👈 Upload and process your documents using the sidebar.", icon="📂")
    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about your documents…"):
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = answer_query(query)
            except Exception as exc:
                logger.exception("Error: %s", exc)
                answer = "Something went wrong. Please try again."
        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})