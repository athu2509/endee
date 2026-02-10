# Free RAG Setup Guide (No API Keys Required!)

This guide shows you how to run the RAG application completely free using local models.

## Prerequisites

### 1. Install Ollama (Local LLM)

**macOS:**
```bash
brew install ollama
```

Or download from: https://ollama.ai

**Start Ollama:**
```bash
ollama serve
```

**Pull a model (in a new terminal):**
```bash
# Recommended: Small and fast
ollama pull llama3.2

# Or try other models:
# ollama pull mistral
# ollama pull phi3
```

### 2. Install Python Dependencies

```bash
cd python_rag_streamlit
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` - for local embeddings (no API key needed)
- `torch` - required by sentence-transformers
- Other dependencies

## Running the Application

### Step 1: Start the Vector Database (Endee)
In one terminal:
```bash
cd /path/to/endee
./run.sh
```

You should see: `Crow/master server is running at http://0.0.0.0:8080`

### Step 2: Start Ollama (if not already running)
In another terminal:
```bash
ollama serve
```

### Step 3: Run the Streamlit App
In a third terminal:
```bash
cd python_rag_streamlit
streamlit run rag_app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the Application

### 1. Check Health
- Click "Check Ollama" in the sidebar to verify Ollama is running
- Click "Check Endee health" to verify the vector database is running

### 2. Ingest Documents
- Go to the "📄 Ingest documents" tab
- Upload PDF, DOCX, or text files
- Click "Ingest into Endee"
- The app will:
  - Extract text from your files
  - Split into chunks
  - Generate embeddings locally (using sentence-transformers)
  - Store in the Endee vector database

### 3. Ask Questions
- Go to the "❓ Ask questions" tab
- Type your question
- Click "Search & answer"
- The app will:
  - Convert your question to an embedding
  - Search the vector database for relevant chunks
  - Send the chunks to Ollama (local LLM)
  - Display the answer

## Models Used

### Embeddings
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Size**: ~80MB
- **Speed**: Very fast
- **Quality**: Good for most use cases
- **Cost**: FREE (runs locally)

### LLM
- **Default**: `llama3.2` (via Ollama)
- **Size**: ~2GB
- **Speed**: Depends on your hardware
- **Quality**: Excellent for RAG tasks
- **Cost**: FREE (runs locally)

You can change the model in the sidebar. Other good options:
- `mistral` - Fast and capable
- `phi3` - Very small and fast
- `llama3.1` - Larger, more capable

## Troubleshooting

### "Ollama not reachable"
- Make sure Ollama is running: `ollama serve`
- Check if it's on the right port: `curl http://localhost:11434/api/tags`

### "Failed to reach Endee"
- Make sure the vector database is running: `./run.sh`
- Check if it's on port 8080: `curl http://localhost:8080/api/v1/health`

### Slow embedding/inference
- First run downloads models (one-time)
- Embeddings are fast even on CPU
- LLM inference is faster with GPU but works on CPU

### Out of memory
- Try a smaller Ollama model: `ollama pull phi3`
- Reduce the chunk size in the code
- Process fewer documents at once

## Advantages of This Setup

✅ **Completely free** - No API costs
✅ **Private** - Your data never leaves your machine
✅ **Offline** - Works without internet (after initial model download)
✅ **Customizable** - Swap models easily
✅ **Fast** - Local embeddings are very quick

## Next Steps

- Try different Ollama models for better/faster answers
- Adjust chunk size for your documents
- Experiment with different embedding models
- Add more document types
