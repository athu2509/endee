import io
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List

import streamlit as st
from sentence_transformers import SentenceTransformer
import requests

from endee_client import EndeeClient


# Persistent storage for document chunks
STORAGE_DIR = Path("./data/doc_chunks")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_FILE = STORAGE_DIR / "chunks.pkl"


def load_doc_store() -> Dict[str, str]:
    """Load document chunks from disk"""
    if STORAGE_FILE.exists():
        try:
            with open(STORAGE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    return {}


def save_doc_store(doc_store: Dict[str, str]):
    """Save document chunks to disk"""
    with open(STORAGE_FILE, 'wb') as f:
        pickle.dump(doc_store, f)


def extract_text_from_pdf(raw_bytes: bytes) -> str:
    """Extract plain text from a PDF file."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts).strip()


def extract_text_from_docx(raw_bytes: bytes) -> str:
    """Extract plain text from a DOCX file."""
    from docx import Document
    doc = Document(io.BytesIO(raw_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()


# -----------------------------
# Configuration / clients
# -----------------------------

st.set_page_config(
    page_title="Endee RAG Demo",
    page_icon="🔎",
    layout="wide",
)

# Initialize clients
endee_client = EndeeClient()

# Load embedding model (cached to avoid reloading)
@st.cache_resource
def load_embedding_model():
    """Load sentence-transformers model for embeddings (runs locally, free)"""
    with st.spinner("🔄 Loading embedding model (first time only, ~80MB download)..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Only load when needed
embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model"""
    global embedding_model
    if embedding_model is None:
        embedding_model = load_embedding_model()
    return embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Use sentence-transformers to convert texts into vectors (free, local).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.tolist()


def llm_answer(question: str, contexts: List[str], ollama_url: str = "http://localhost:11434", model: str = "llama3.2") -> str:
    """
    Simple RAG-style answer using Ollama (free, local LLM).
    """
    context_block = "\n\n---\n\n".join(contexts)
    system_prompt = (
        "You are a helpful assistant answering questions based only on the provided context. "
        "If the answer is not in the context, say you don't know instead of hallucinating."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer the question using only the context above. "
        "Cite which chunks you used if helpful."
    )

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": f"{system_prompt}\n\n{user_prompt}",
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "No response from LLM")
    except Exception as e:
        return f"Error calling Ollama: {e}. Make sure Ollama is running with: ollama serve"


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """
    Very simple character-based chunking.
    """
    text = text.replace("\r\n", "\n")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


if "doc_store" not in st.session_state:
    # Load from disk on first run
    st.session_state.doc_store = load_doc_store()


st.title("Endee RAG Demo (Python + Streamlit)")
st.markdown(
    "This demo uses **Endee** as the vector database, **sentence-transformers** for embeddings (local), "
    "and **Ollama** for LLM answers (local). Everything runs free on your machine!"
)

# System architecture indicator
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🗄️ Vector DB", "Endee", help="Vectors stored in Endee database")
with col2:
    st.metric("🧠 Embeddings", "Local", help="sentence-transformers (all-MiniLM-L6-v2)")
with col3:
    st.metric("💬 LLM", "Ollama", help="Local language model")

# Quick status check
try:
    health = endee_client.health()
    st.success("✅ Endee is connected and healthy!")
except:
    st.error("❌ Endee is not running! Start it with: ./run.sh")
    st.stop()

st.markdown("---")


with st.sidebar:
    st.header("Settings")
    
    st.markdown("### Local LLM (Ollama)")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama Model", value="llama3.2", help="Run 'ollama pull llama3.2' first")
    
    if st.button("Check Ollama"):
        try:
            resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if models:
                st.success(f"Ollama is running! Available models: {', '.join([m['name'] for m in models])}")
            else:
                st.warning("Ollama is running but no models found. Run: ollama pull llama3.2")
        except Exception as e:
            st.error(f"Ollama not reachable: {e}")
            st.info("Install Ollama from https://ollama.ai and run: ollama serve")

    st.markdown("### Vector Database")
    index_name = st.text_input("Endee index name", value="candidate_rag_index")
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=3)
    
    st.info(f"📊 Using **Endee** vector database at `{endee_client.base_url}`")

    st.markdown("**Endee server** is expected at `http://localhost:8080/api/v1` (from `./run.sh`).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check Endee health"):
            try:
                health = endee_client.health()
                st.success(f"✅ Endee is healthy!")
                st.json(health)
            except Exception as e:
                st.error(f"❌ Failed to reach Endee: {e}")
    
    with col2:
        if st.button("📚 View Stored Docs"):
            doc_count = len(st.session_state.doc_store)
            if doc_count > 0:
                st.success(f"Found {doc_count} chunks stored")
                # Group by file
                files = {}
                for vec_id in st.session_state.doc_store.keys():
                    filename = vec_id.split("::")[0]
                    files[filename] = files.get(filename, 0) + 1
                st.write("**Files indexed:**")
                for filename, count in files.items():
                    st.write(f"- {filename}: {count} chunks")
            else:
                st.warning("No documents stored yet")


tab_ingest, tab_ask = st.tabs(["📄 Ingest documents", "❓ Ask questions"])


def get_text_from_file(filename: str, raw_bytes: bytes) -> str:
    """Dispatch by extension: PDF, DOCX, or plain text."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(raw_bytes)
    if lower.endswith(".docx"):
        return extract_text_from_docx(raw_bytes)
    # .txt, .md, etc.
    return raw_bytes.decode("utf-8", errors="ignore")


with tab_ingest:
    st.subheader("Upload & ingest documents into Endee")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or text/markdown files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🗑️ Clear Index", help="Delete all vectors from the index"):
            try:
                # Delete the index to clear all data
                resp = requests.delete(
                    f"{endee_client.base_url}/index/{index_name}",
                    headers=endee_client._headers(),
                    timeout=endee_client.timeout
                )
                if resp.status_code in [200, 404]:
                    st.session_state.doc_store.clear()
                    save_doc_store(st.session_state.doc_store)
                    st.success("Index cleared! You can now upload fresh documents.")
                else:
                    st.warning(f"Clear returned status {resp.status_code}")
            except Exception as e:
                st.error(f"Failed to clear index: {e}")

    if st.button("Ingest into Endee") and uploaded_files:
        all_chunks: List[str] = []
        all_ids: List[str] = []
        for f in uploaded_files:
            try:
                raw_bytes = f.read()
                text = get_text_from_file(f.name, raw_bytes)
            except Exception as e:
                st.warning(f"Could not read file {f.name}: {e}")
                continue
            if not text.strip():
                st.warning(f"No text extracted from {f.name}; skipping.")
                continue

            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                vec_id = f"{f.name}::chunk-{i}"
                all_ids.append(vec_id)
                all_chunks.append(chunk)
                st.session_state.doc_store[vec_id] = chunk

        if not all_chunks:
            st.warning("No content to ingest.")
        else:
            with st.spinner("Embedding and indexing documents..."):
                try:
                    vectors = embed_texts(all_chunks)
                    dim = len(vectors[0])

                    # Ensure index exists
                    endee_client.create_index(index_name=index_name, dim=dim, space_type="cosine")

                    # Prepare batch payload
                    items = []
                    for vec_id, vec, chunk in zip(all_ids, vectors, all_chunks):
                        items.append(
                            {
                                "id": vec_id,
                                "vector": vec,
                                # Store the actual text in the filter field as JSON
                                "filter": json.dumps({
                                    "source": vec_id.split("::")[0],
                                    "text": chunk  # Store the chunk text here
                                }),
                            }
                        )

                    endee_client.insert_vectors(index_name=index_name, items=items)
                    
                    # Store text chunks in session state AND persist to disk
                    for vec_id, chunk in zip(all_ids, all_chunks):
                        st.session_state.doc_store[vec_id] = chunk
                    save_doc_store(st.session_state.doc_store)
                    
                    st.success(f"✅ Ingested {len(items)} chunks into Endee index '{index_name}'. Data persisted to disk.")
                    st.info(f"💾 Vectors stored in Endee database | 📝 Text stored locally for retrieval")
                except Exception as e:
                    st.error(f"Failed to ingest documents: {e}")


with tab_ask:
    st.subheader("Ask questions over your indexed documents")
    
    # Check if documents are indexed
    doc_count = len(st.session_state.doc_store)
    if doc_count == 0:
        st.warning("⚠️ No documents indexed yet!")
        st.info("""
        **To get started:**
        1. Go to the "📄 Ingest documents" tab
        2. Upload your PDF, DOCX, or text files
        3. Click "Ingest into Endee"
        4. Come back here to ask questions!
        """)
        st.stop()
    
    st.success(f"📚 {doc_count} chunks available for search")
    
    question = st.text_input("Your question", placeholder="e.g., What are the main features of the encrypted notes app?")

    if st.button("Search & answer") and question:
        with st.spinner("Searching Endee and generating answer..."):
            try:
                q_vec = embed_texts([question])[0]
                results = endee_client.search(
                    index_name=index_name,
                    query_vector=q_vec,
                    k=top_k,
                    ef=None,
                    include_vectors=False,
                    include_metadata=True,
                )

                if not results:
                    st.warning("No results returned from Endee for this query.")
                else:
                    # Build context from stored chunks
                    # Results format: [similarity, id, metadata, filter, label, extra]
                    contexts: List[str] = []
                    for r in results:
                        if isinstance(r, list) and len(r) >= 2:
                            vec_id = r[1]  # ID is at index 1
                        elif isinstance(r, dict):
                            vec_id = r.get("id", "")
                        else:
                            vec_id = str(r)
                        
                        # Get text from persistent storage
                        chunk_text_val = st.session_state.doc_store.get(
                            vec_id, f"[Text not found for {vec_id}. Please re-upload documents.]"
                        )
                        contexts.append(chunk_text_val)

                    answer = llm_answer(question, contexts, ollama_url, ollama_model)

                    st.markdown("---")
                    st.markdown("## 💡 Answer")
                    st.markdown(f"> {answer}")
                    
                    st.markdown("---")
                    st.markdown(f"## 📄 Retrieved Context (Top {len(results)} from Endee)")
                    
                    for idx, (r, ctx) in enumerate(zip(results, contexts), 1):
                        with st.expander(f"**Chunk {idx}** - Similarity: {r[0]:.3f}" if isinstance(r, list) else f"Chunk {idx}", expanded=(idx == 1)):
                            if isinstance(r, list) and len(r) >= 2:
                                similarity = r[0]
                                vec_id = r[1]
                                st.caption(f"📎 Source: `{vec_id.split('::')[0]}`")
                                st.caption(f"🔍 Similarity Score: `{similarity:.4f}`")
                            st.markdown("**Content:**")
                            st.text(ctx)
            except Exception as e:
                import traceback
                st.error(f"Failed to run search or generate answer: {e}")
                st.code(traceback.format_exc())


