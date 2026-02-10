# Endee RAG System

A complete **Retrieval-Augmented Generation (RAG)** system using Endee vector database with local embeddings and LLM - **100% free, no API keys required.**

## 🏗️ How It Works

```
User Question → Embedding Model → Vector Search (Endee) → Retrieve Chunks → LLM → Answer
```

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│  1. Document Upload (PDF, DOCX, TXT)                │
│     ↓                                                │
│  2. Text Chunking (~800 chars)                      │
│     ↓                                                │
│  3. Generate Embeddings (sentence-transformers)     │
│     ↓                                                │
│  4. Store Vectors in Endee (HNSW index)             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  5. User asks question                               │
│     ↓                                                │
│  6. Convert question to embedding                    │
│     ↓                                                │
│  7. Search Endee for similar vectors (cosine)       │
│     ↓                                                │
│  8. Retrieve top-K text chunks                       │
│     ↓                                                │
│  9. Send chunks + question to Ollama LLM            │
│     ↓                                                │
│  10. Get AI-generated answer                         │
└─────────────────────────────────────────────────────┘
```

**Components:**
- **Endee**: Vector database (C++, HNSW algorithm, 384-dim vectors)
- **sentence-transformers**: Local embedding model (all-MiniLM-L6-v2)
- **Ollama**: Local LLM (llama3.2, mistral, phi3, etc.)
- **Streamlit**: Web UI

## 🚀 Quick Start

### 1. Start Endee Vector Database

**macOS/Linux:**
```bash
cd endee
./install.sh  # First time only
./run.sh      # Starts server on port 8080
```

**Windows:**
```cmd
cd endee
install.bat   # First time only (or follow manual build instructions)
run.bat       # Starts server on port 8080
```

### 2. Install & Start Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from https://ollama.ai and install

**Start Ollama (all platforms):**
```bash
# Terminal 1
ollama serve

# Terminal 2
ollama pull llama3.2
```

### 3. Run RAG Application

**macOS/Linux:**
```bash
cd python_rag_streamlit
pip install -r requirements.txt
streamlit run rag_app.py
```

**Windows:**
```cmd
cd python_rag_streamlit
pip install -r requirements.txt
streamlit run rag_app.py
```

Opens at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**: Go to "Ingest documents" tab → Upload PDF/DOCX → Click "Ingest"
2. **Ask Questions**: Go to "Ask questions" tab → Type question → Click "Search & answer"
3. **View Results**: See AI answer + source chunks with similarity scores

## 🔧 Configuration

**Sidebar Settings:**
- **Top-K results**: 3 (default) - Number of chunks to retrieve
- **Ollama Model**: llama3.2 (default) - Change to mistral, phi3, etc.
- **Index name**: candidate_rag_index

**Try different models:**
```bash
ollama pull mistral    # Fast and capable
ollama pull phi3       # Very small (2GB)
```

## 🐛 Troubleshooting

**"Endee is not running"**

*macOS/Linux:*
```bash
cd endee && ./run.sh
```

*Windows:*
```cmd
cd endee && run.bat
```

**"Ollama not reachable"**

*All platforms:*
```bash
ollama serve
```

**Verify database:**

*macOS/Linux:*
```bash
cd python_rag_streamlit
python check_endee.py
```

*Windows:*
```cmd
cd python_rag_streamlit
python check_endee.py
```

**Virtual environment activation:**

*macOS/Linux:*
```bash
source .venv/bin/activate
```

*Windows:*
```cmd
.venv\Scripts\activate
```

## 📊 Performance

| Operation | Time |
|-----------|------|
| First load | ~30s (one-time model download) |
| Embed 1 page | ~0.5s |
| Vector search | ~50ms |
| LLM response | ~5-10s |

## 📁 Data Storage

- **Vectors**: `./data/endee/` (Endee database)
- **Text chunks**: `./data/doc_chunks/chunks.pkl` (persistent)
- **Metadata**: `./data/meta/`

## 🔐 Privacy

✅ Everything runs locally  
✅ No API keys needed  
✅ No data leaves your machine  
✅ Works offline (after setup)

---

**See [QUICKSTART.md](QUICKSTART.md) for 5-minute setup guide**
