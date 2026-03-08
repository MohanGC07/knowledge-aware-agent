<div align="center">

# 🧠 Knowledge-Aware Agent

### *A Production-Grade RAG System Built From Scratch*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![RAG](https://img.shields.io/badge/Architecture-RAG-6C63FF?style=for-the-badge&logo=databricks&logoColor=white)](#)
[![LLM](https://img.shields.io/badge/LLM-Llama3--70B-00A67E?style=for-the-badge&logo=meta&logoColor=white)](#)
[![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-F5A623?style=for-the-badge&logo=apache&logoColor=white)](#)

<br/>

> **Upload documents. Ask questions. Get grounded answers — powered by a full RAG pipeline built without LangChain.**

<br/>

[🚀 Live Demo](#-demo)https://knowledge-aware-agent.streamlit.app/ · [📦 Installation](#-installation) · [📖 How It Works](#-how-the-rag-pipeline-works) · [🛠️ Tech Stack](#️-tech-stack)

</div>

---

## 📌 Overview

**Knowledge-Aware Agent** is a Retrieval-Augmented Generation (RAG) system that lets users upload documents and ask natural language questions about their content. The system processes files, generates semantic embeddings, stores them in a vector database, and retrieves relevant context to produce grounded, accurate answers via a large language model.

**The key distinction:** this project is built **entirely from scratch** — no LangChain, no LlamaIndex — to demonstrate a deep understanding of how modern AI knowledge systems actually work under the hood.

This makes it an ideal reference project for engineers who want to understand the full RAG lifecycle from document ingestion to final answer generation.

---

## 🎬 Demo

> *Upload a research paper, a codebase, or a set of notes — and start querying instantly.*

```
📂 Upload: research_paper.pdf
❓ Query:  "What methodology did the authors use for evaluation?"
🤖 Answer: "The authors employed a mixed-methods approach, combining quantitative
            benchmarking across 5 datasets with qualitative expert review panels..."
https://knowledge-aware-agent.streamlit.app/            
```

*(Powered by llama-3.3-70b-versatile via Groq API — responses in under 2 seconds)*

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📄 **Multi-format Upload** | Supports PDF, DOCX, TXT, MD, PY, JSON, CSV |
| ✂️ **Smart Chunking** | Documents split into optimized overlapping chunks |
| 🔢 **Semantic Embeddings** | `all-MiniLM-L6-v2` via Sentence Transformers |
| 🗄️ **Vector Storage** | Persistent ChromaDB for efficient retrieval |
| 🔍 **Similarity Search** | Top-K cosine similarity retrieval |
| 💬 **Context-Grounded Answers** | Retrieved chunks injected into LLM prompts |
| ⚡ **Fast Inference** | Groq API with llama-3.3-70b-versatile for sub-second responses |
| 🖥️ **Clean UI** | Streamlit-based drag-and-drop interface |

---

## 🏗️ Architecture

The system is composed of two primary pipelines:

### Pipeline 1 — Document Ingestion

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│   User File  │───▶│  Text Extraction │───▶│   Chunking  │
│ (PDF/DOCX/…) │    │  (PyPDF/docx)    │    │  (overlap)  │
└──────────────┘    └──────────────────┘    └──────┬──────┘
                                                   │
                                                   ▼
                                      ┌────────────────────┐
                                      │ Embedding Generation│
                                      │  (MiniLM-L6-v2)    │
                                      └──────────┬─────────┘
                                                 │
                                                 ▼
                                      ┌────────────────────┐
                                      │   ChromaDB Storage  │
                                      │  (Vector Database)  │
                                      └────────────────────┘
```

### Pipeline 2 — Query & Generation

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐
│  User Query  │───▶│  Query Embedding │───▶│ Vector Search  │
│              │    │  (MiniLM-L6-v2)  │    │ (ChromaDB)     │
└──────────────┘    └──────────────────┘    └───────┬────────┘
                                                    │
                                                    ▼
                                       ┌────────────────────┐
                                       │   Top-K Retrieval  │
                                       │  (Relevant Chunks) │
                                       └──────────┬─────────┘
                                                  │
                                                  ▼
                                       ┌────────────────────┐
                                       │  Prompt Assembly   │
                                       │ (Context Injection)│
                                       └──────────┬─────────┘
                                                  │
                                                  ▼
                                       ┌────────────────────┐
                                       │   Groq / Llama-3   │
                                       │   Final Answer ✅   │
                                       └────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | [Streamlit](https://streamlit.io) | File upload UI & chat interface |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) `llama-3.3-70b-versatile` | Semantic vector generation |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Persistent embedding storage & retrieval |
| **LLM Inference** | [Groq API](https://groq.com/) | Ultra-fast LLM inference |
| **LLM Model** | `llama3-70b-8192` | Answer generation |
| **PDF Parsing** | [PyPDF](https://pypdf.readthedocs.io/) | Extract text from PDFs |
| **DOCX Parsing** | [python-docx](https://python-docx.readthedocs.io/) | Extract text from Word docs |
| **Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Manage API keys & env vars |

---

## 📁 Project Structure

```
knowledge-aware-agent/
│
├── app.py                      # 🚀 Streamlit entry point
├── config.py                   # ⚙️  Global configuration & constants
├── requirements.txt
├── .env                        # 🔑 API keys (not committed)
│
├── ingestion/
│   ├── document_loader.py      # Loads files by type (PDF, DOCX, TXT, etc.)
│   └── chunker.py              # Splits text into overlapping chunks
│
├── embeddings/
│   └── embedding_model.py      # Wraps Sentence Transformer model
│
├── vector_store/
│   └── chroma_store.py         # ChromaDB interface (add, query, reset)
│
├── retrieval/
│   └── retriever.py            # Similarity search & Top-K fetch
│
├── generation/
│   └── generator.py            # Prompt builder + Groq API caller
│
└── data/
    ├── documents/              # Uploaded files (temporary storage)
    └── chroma_db/              # Persistent ChromaDB collection
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/)

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/MohanGC07/knowledge-aware-agent.git
cd knowledge-aware-agent
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your environment**

Create a `.env` file in the project root:
```bash
touch .env
```

Add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

**5. Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🔬 How the RAG Pipeline Works

### Step 1 — Upload
User drags and drops one or more files into the Streamlit sidebar.

### Step 2 — Load
`document_loader.py` detects the file type and extracts raw text using PyPDF (for PDFs), python-docx (for Word files), or native Python file I/O for plain text formats.

### Step 3 — Chunk
`chunker.py` splits the extracted text into smaller, overlapping windows. Chunking ensures that semantically related sentences stay together, improving retrieval precision.

### Step 4 — Embed
`embedding_model.py` passes each chunk through `all-MiniLM-L6-v2`, a lightweight but powerful Sentence Transformer that converts text into 384-dimensional dense vectors.

### Step 5 — Store
`chroma_store.py` persists the embeddings and associated metadata in a local ChromaDB collection. This allows efficient cosine similarity search at query time.

### Step 6 — Query
The user types a question in the chat interface.

### Step 7 — Search
`retriever.py` embeds the query using the same model, then queries ChromaDB for the most semantically similar chunks.

### Step 8 — Retrieve
The top-K most relevant chunks are returned as candidate context.

### Step 9 — Prompt Assembly
`generator.py` assembles a structured prompt, injecting the retrieved chunks as context before the user's question.

### Step 10 — Generate
The prompt is sent to Groq's API running `llama-3.3-70b-versatile`. The LLM generates a final, grounded answer strictly based on the provided document context.

---

## ☁️ Deployment on Streamlit Cloud

Deploy this app publicly in minutes:

**1.** Push your project to a GitHub repository.

**2.** Visit [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.

**3.** Select your repository, set the main file path to `app.py`, and click **Deploy**.

**4.** In the **Advanced Settings** → **Secrets**, add your environment variable:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

**5.** Your app is live! 🌍

Users will be able to:
- Upload their own documents
- Query document content via natural language
- Interact with the knowledge-aware agent from any browser

---

## 💡 Example Use Cases

| Use Case | Example Query |
|----------|--------------|
| 📄 **Research Papers** | *"What were the key findings of this study?"* |
| 💻 **Code Analysis** | *"What does the `authenticate()` function do?"* |
| 📋 **Documentation** | *"How do I configure the deployment settings?"* |
| 📝 **Notes & Summaries** | *"Summarize the main takeaways from these meeting notes."* |
| 📊 **Data & Reports** | *"What were the revenue figures mentioned in Q3?"* |
| 📚 **Books & Articles** | *"What is the author's argument in chapter 3?"* |

---

## 🚀 Future Improvements

- [ ] **Semantic Chunking** — Use embedding similarity to create more coherent chunks rather than fixed windows
- [ ] **Hybrid Search** — Combine BM25 keyword search with vector similarity for better recall
- [ ] **Chat Memory** — Maintain conversation history for multi-turn Q&A sessions
- [ ] **Source Citations** — Display exact chunk sources and page numbers alongside answers
- [ ] **Streaming Responses** — Stream LLM tokens to the UI for a more responsive feel
- [ ] **Multi-document Ranking** — Re-rank retrieved chunks across multiple uploaded files
- [ ] **Authentication** — Add user login to keep uploaded document collections private
- [ ] **Evaluation Module** — Integrate RAGAS for automated RAG quality benchmarking

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

<div align="center">

**Built with curiosity and zero shortcuts.**

*If this project helped you understand RAG systems, consider giving it a ⭐*

</div>