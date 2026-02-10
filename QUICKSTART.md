# 🚀 Quick Start Guide - 5 Minutes to Your First Query

## Step 1: Start Endee (30 seconds)

**macOS/Linux:**
```bash
cd endee
./run.sh
```

**Windows:**
```cmd
cd endee
run.bat
```

✅ You should see: `Crow/master server is running at http://0.0.0.0:8080`

**Keep this terminal open!**

---

## Step 2: Start Ollama (30 seconds)

**Install Ollama:**
- **macOS:** `brew install ollama`
- **Linux:** `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows:** Download from https://ollama.ai

**Start Ollama (new terminal):**
```bash
ollama serve
```

**Pull a model (another new terminal):**
```bash
ollama pull llama3.2
```

✅ Wait for download to complete (~2GB)

---

## Step 3: Install Python Dependencies (1 minute)

**New terminal:**

**macOS/Linux:**
```bash
cd python_rag_streamlit
pip install -r requirements.txt
```

**Windows:**
```cmd
cd python_rag_streamlit
pip install -r requirements.txt
```

---

## Step 4: Run the App (30 seconds)

**macOS/Linux:**
```bash
streamlit run rag_app.py
```

**Windows:**
```cmd
streamlit run rag_app.py
```

✅ Browser opens at `http://localhost:8501`

**First time**: Wait ~30 seconds for embedding model download

---

## Step 5: Upload & Query (2 minutes)

### Upload Documents:
1. Click **"📄 Ingest documents"** tab
2. Click **"Browse files"**
3. Select your PDF/DOCX files
4. Click **"Ingest into Endee"**
5. Wait for "✅ Ingested X chunks..."

### Ask Questions:
1. Click **"❓ Ask questions"** tab
2. Type your question
3. Click **"Search & answer"**
4. Get AI-powered answers! 🎉

---

## 🎯 Example Questions

If you uploaded a resume:
- "What are the technical skills?"
- "Describe the work experience"
- "What projects are mentioned?"

If you uploaded research papers:
- "What is the main contribution?"
- "What methodology was used?"
- "What are the key findings?"

---

## ✅ Verification Checklist

- [ ] Endee running on port 8080
- [ ] Ollama running on port 11434
- [ ] Streamlit app open in browser
- [ ] Green checkmark: "✅ Endee is connected and healthy!"
- [ ] Documents uploaded successfully
- [ ] First query returns an answer

---

## 🐛 Quick Troubleshooting

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
```bash
ollama serve
```

**"No documents indexed"**
- Go to "Ingest documents" tab
- Upload files first

---

## 🎓 Next Steps

1. **Try different models**: `ollama pull mistral`
2. **Adjust Top-K**: Use slider in sidebar (1-3 is usually best)
3. **View stored docs**: Click "📚 View Stored Docs" button
4. **Check database**: Run `python check_endee.py`

---

**Need more help?** See the full [README.md](README.md)

**Ready to build?** Start uploading your documents! 🚀
