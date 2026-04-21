# 📚 Study Buddy RAG v2

> **A professional-grade, AI-powered academic tutor built with LangChain, ChromaDB, and Streamlit.**  
> Hybrid retrieval · Vision analysis · Chat memory · Interactive quizzes · Cloud & local backends.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Technology | Benefit |
|---|---|---|
| **Hybrid Search** | BM25 + ChromaDB `EnsembleRetriever` | Handles both exact technical terms & semantic meaning |
| **Advanced PDF Parsing** | PyMuPDF4LLM → Markdown | Tables, headers, and structured layout preserved |
| **Interactive Quiz** | `st.form` + `st.radio` + scoring | Instant feedback, score tracking, retry support |
| **Chat Memory** | Rolling summary injection | Follow-up questions work naturally |
| **Vision Analysis** | Gemini 1.5 Flash / phi3-vision | Explain diagrams, charts, handwritten notes |
| **Dual Backend** | Auto-detects Gemini vs Ollama | One codebase, two deployment targets |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyMuPDF4LLM ──► Markdown-aware chunks (tables intact)
                    │
                    ▼
          OllamaEmbeddings / Gemini Embeddings
                    │
                    ▼
              ChromaDB (persisted)
                    │
       ┌────────────┴────────────┐
       │                         │
  BM25Retriever           Dense Retriever
  (keyword / TF-IDF)     (semantic vector)
       │                         │
       └────────────┬────────────┘
                    │  EnsembleRetriever (50/50)
                    ▼
         Top-K relevant chunks
                    │
       ┌────────────┴────────────┐
       │                         │
  Chat History Summary     User Question
       │                         │
       └────────────┬────────────┘
                    ▼
          LLM (phi3.5 / Gemini 1.5 Flash)
                    │
            ┌───────┴────────┐
            │                │
         Answer          Quiz Generator
            │           (3 questions, scored)
            └───────┬────────┘
                    ▼
             Streamlit UI
```

---

## 🚀 Quick Start

### Option A — Streamlit Cloud (Gemini)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → Deploy
3. In **App Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Done — no GPU needed!

### Option B — Local (Ollama, GTX 1650 Ti / 4 GB VRAM)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh   # Linux
# Windows: https://ollama.com/download/OllamaSetup.exe

# 2. Pull models
ollama pull phi3.5
ollama pull nomic-embed-text
ollama pull phi3-vision

# 3. Clone & install
git clone https://github.com/yourusername/study-buddy-rag
cd study-buddy-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

---

## 📂 Project Structure

```
study-buddy-rag/
├── app.py              ← Main application (all features)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── .streamlit/
│   └── secrets.toml    ← GEMINI_API_KEY (git-ignored)
└── chroma_db/          ← Auto-created on first PDF upload
```

---

## 🧠 How It Works

### Hybrid Search
Traditional RAG uses only vector (semantic) search, which struggles with exact technical terms, acronyms, and formulas. Study Buddy combines **BM25 (TF-IDF keyword search)** and **dense vector search** in a 50/50 ensemble — getting the best of both worlds.

### PyMuPDF4LLM
Standard `PyPDFLoader` discards table structure and column layouts. `PyMuPDF4LLM` converts PDFs to clean Markdown first, preserving tables as pipe-delimited markdown, header hierarchies, and reading order — critical for lecture notes and textbooks.

### Chat Memory
After every 4+ turns, the conversation history is summarized and injected as context. This allows natural follow-up questions like *"Tell me more about the third point"* without exceeding the LLM's context window.

### Interactive Quiz
Every answer auto-generates 3 MCQ/True-False questions from the retrieved context. The quiz uses `st.form` for atomic submission, tracks scores in `st.session_state`, and provides per-question explanations with a retry option.

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap between chunks |
| `TOP_K_DEFAULT` | 4 | Chunks retrieved per query |
| `OLLAMA_LLM_MODEL` | `phi3.5` | Local LLM |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Local embeddings |
| `GEMINI_LLM_MODEL` | `gemini-1.5-flash` | Cloud LLM |

---

## 🖥️ Hardware Requirements

| Mode | Minimum | Recommended |
|---|---|---|
| **Local** | 4 GB VRAM (GTX 1650) | 8 GB VRAM |
| **Cloud** | Any machine | Any machine |

---

## 📄 License

MIT License — free for personal and commercial use.

---

*Built as a senior portfolio project · AI & Computer Vision specialization*
