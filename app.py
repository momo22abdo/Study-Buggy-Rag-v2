# ══════════════════════════════════════════════════════════════════════════════
#  Study Buddy RAG — v2.0  |  Senior Portfolio Build
#  Features: Hybrid Search · PyMuPDF4LLM · Interactive Quiz · Chat Memory
#            Multi-modal Vision · Dual Backend (Gemini Cloud / Ollama Local)
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import os, re, json, tempfile, time, base64, gc
from pathlib import Path

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Study Buddy · RAG v2",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND DETECTION  ·  Auto-detect Gemini (Cloud) vs Ollama (Local)
# ══════════════════════════════════════════════════════════════════════════════

def _get_gemini_key() -> str | None:
    """Return Gemini API key from Streamlit secrets or env, or None."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")

USE_GEMINI: bool = bool(_get_gemini_key())

# ── Model constants ───────────────────────────────────────────────────────────
OLLAMA_LLM_MODEL    = "phi3.5"
OLLAMA_EMBED_MODEL  = "nomic-embed-text"
OLLAMA_VISION_MODEL = "phi3-vision"

GEMINI_LLM_MODEL    = "gemini-1.5-flash"
GEMINI_EMBED_MODEL  = "models/text-embedding-004"
GEMINI_VISION_MODEL = "gemini-1.5-flash"   # native multimodal

# ── RAG constants ─────────────────────────────────────────────────────────────
CHROMA_DIR    = "./chroma_db"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120
TOP_K_DEFAULT = 4

# ══════════════════════════════════════════════════════════════════════════════
#  PREMIUM CSS  ·  Syne + DM Sans · SaaS dark theme
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root palette ────────────────────────────────────────── */
:root {
  --bg:       #080b12;
  --surface:  #0f1219;
  --surface2: #141820;
  --border:   #1d2236;
  --border2:  #252d42;
  --accent:   #4f8ef7;
  --accent2:  #9b72f7;
  --accent3:  #f472b6;
  --gold:     #f0c040;
  --green:    #2dd4a0;
  --red:      #f87171;
  --amber:    #fbbf24;
  --text:     #dde5f4;
  --text2:    #8899bb;
  --radius:   14px;
  --radius-sm:8px;
  --shadow:   0 4px 24px rgba(0,0,0,.55);
}

/* ── Global ──────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

/* ── Main area background noise texture ─────────────────── */
.main .block-container {
  background: radial-gradient(ellipse 120% 80% at 60% -20%,
    rgba(79,142,247,.07) 0%, transparent 70%),
    radial-gradient(ellipse 80% 60% at 90% 110%,
    rgba(155,114,247,.06) 0%, transparent 65%);
}

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSlider * { color: var(--text) !important; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg,var(--accent),var(--accent2)) !important;
  color:#fff !important; border:none !important;
  border-radius:var(--radius) !important;
  font-family:'Syne',sans-serif !important;
  font-weight:700 !important; letter-spacing:.4px;
  padding:.55rem 1.4rem !important;
  box-shadow: 0 2px 12px rgba(79,142,247,.3);
  transition: all .2s ease;
}
.stButton > button:hover { opacity:.85; transform:translateY(-2px); box-shadow:0 4px 20px rgba(79,142,247,.4); }
.stButton > button:active { transform:translateY(0); }

/* ── Secondary / ghost button ───────────────────────────── */
button[kind="secondary"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text2) !important;
}

/* ── File uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: var(--radius) !important;
  padding: 1rem !important;
  transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Chat bubbles ────────────────────────────────────────── */
.bubble-user {
  background: linear-gradient(135deg,#192545 0%,#131d38 100%);
  border: 1px solid #253558;
  border-radius: 18px 18px 4px 18px;
  padding: 1rem 1.25rem;
  margin: .7rem 0;
  max-width: 86%;
  margin-left: auto;
  box-shadow: var(--shadow);
}
.bubble-ai {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 18px 18px 18px 4px;
  padding: 1rem 1.25rem;
  margin: .7rem 0;
  max-width: 92%;
  box-shadow: var(--shadow);
}
.bubble-label {
  font-family: 'Syne', sans-serif;
  font-size: .68rem;
  font-weight: 800;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-bottom: .45rem;
  opacity: .55;
}
.bubble-ai .bubble-label  { color: var(--accent); }
.bubble-user .bubble-label { color: var(--accent2); }

/* ── Knowledge Check card ───────────────────────────────── */
.kc-card {
  background: linear-gradient(135deg,#12172a,#0e1320);
  border: 1px solid var(--accent2);
  border-left: 4px solid var(--gold);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  margin-top: 1.1rem;
  box-shadow: 0 4px 28px rgba(155,114,247,.12);
}
.kc-title {
  font-family: 'Syne', sans-serif; font-size: .72rem;
  font-weight: 800; letter-spacing: 1.6px; text-transform: uppercase;
  color: var(--gold); margin-bottom: 1rem;
}
.kc-score-bar {
  background: var(--border); border-radius: 999px; height: 6px;
  margin: .6rem 0 1.2rem;
}
.kc-score-fill {
  background: linear-gradient(90deg, var(--green), var(--accent));
  border-radius: 999px; height: 6px; transition: width .6s ease;
}
.kc-feedback-correct { color: var(--green); font-size:.84rem; margin-top:.25rem; }
.kc-feedback-wrong   { color: var(--red);   font-size:.84rem; margin-top:.25rem; }

/* ── Status pills ────────────────────────────────────────── */
.pill {
  display:inline-block; padding:.22rem .8rem;
  border-radius:999px; font-size:.72rem; font-weight:700;
  letter-spacing:.6px; font-family:'Syne',sans-serif;
}
.pill-green  { background:#0a211a; color:var(--green);  border:1px solid #133d2a; }
.pill-red    { background:#221010; color:var(--red);    border:1px solid #3d1515; }
.pill-blue   { background:#0d1729; color:var(--accent); border:1px solid #162640; }
.pill-purple { background:#170e2a; color:var(--accent2);border:1px solid #2a1a45; }
.pill-gold   { background:#221a04; color:var(--gold);   border:1px solid #3d3009; }

/* ── Metric boxes ────────────────────────────────────────── */
.metric-box {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: .9rem 1rem; text-align: center;
}
.metric-val {
  font-family:'Syne',sans-serif; font-size:1.6rem;
  font-weight:800; color:var(--accent);
}
.metric-lbl { font-size:.68rem; color:var(--text2); letter-spacing:.8px; text-transform:uppercase; margin-top:.15rem; }

/* ── Info / hint card ───────────────────────────────────── */
.hint-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius);
  padding: 1rem 1.3rem;
  margin-bottom: 1.3rem;
}

/* ── Vision card ────────────────────────────────────────── */
.vision-card {
  background: linear-gradient(135deg,#0e1a24,#091320);
  border: 1px solid #1a3040;
  border-left: 4px solid var(--accent3);
  border-radius: var(--radius);
  padding: 1.1rem 1.4rem;
  margin-top: .6rem;
}
.vision-title {
  font-family:'Syne',sans-serif; font-size:.7rem; font-weight:800;
  letter-spacing:1.5px; text-transform:uppercase; color:var(--accent3);
  margin-bottom:.5rem;
}

/* ── Chat input ──────────────────────────────────────────── */
.stChatInput textarea {
  background: var(--surface2) !important;
  border-color: var(--border2) !important;
  color: var(--text) !important;
  border-radius: var(--radius) !important;
}
.stChatInput textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(79,142,247,.2) !important; }

/* ── Expander ────────────────────────────────────────────── */
.st-expander { border-color: var(--border) !important; background: var(--surface) !important; }
details summary { font-family:'Syne',sans-serif !important; font-size:.85rem !important; }

/* ── Radio / form elements ───────────────────────────────── */
.stRadio label { font-size:.9rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size:.88rem; }
div[data-testid="stForm"] {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
}

/* ── Selectbox ───────────────────────────────────────────── */
.stSelectbox > div > div {
  background: var(--surface2) !important;
  border-color: var(--border2) !important;
  color: var(--text) !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.4rem !important; }
hr { border-color:var(--border) !important; margin:1.1rem 0 !important; }

/* ── Source chunk expander ───────────────────────────────── */
.src-chunk {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: .6rem .9rem;
  font-size: .8rem;
  color: var(--text2);
  margin-bottom: .4rem;
  line-height: 1.5;
}

/* ── Section divider label ───────────────────────────────── */
.section-label {
  font-family:'Syne',sans-serif; font-size:.68rem; font-weight:800;
  letter-spacing:1.8px; text-transform:uppercase; color:var(--text2);
  margin: 1rem 0 .5rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LAZY IMPORTS  ·  Load backend-specific libs only when needed
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_backend():
    """Return (llm, embeddings, vision_fn) based on detected backend."""
    if USE_GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        import google.generativeai as genai
        api_key = _get_gemini_key()
        genai.configure(api_key=api_key)
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL,
            google_api_key=api_key,
            temperature=0.3,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL,
            google_api_key=api_key,
        )
        def vision_fn(image_bytes: bytes, mime: str, prompt: str) -> str:
            import google.generativeai as genai2
            model = genai2.GenerativeModel(GEMINI_VISION_MODEL)
            import PIL.Image, io
            img = PIL.Image.open(io.BytesIO(image_bytes))
            resp = model.generate_content([prompt, img])
            return resp.text
    else:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.3)
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        def vision_fn(image_bytes: bytes, mime: str, prompt: str) -> str:
            import ollama
            b64 = base64.b64encode(image_bytes).decode()
            resp = ollama.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }],
            )
            return resp["message"]["content"]
    return llm, embeddings, vision_fn


def load_llm():
    llm, _, _ = _load_backend()
    return llm

def load_embeddings():
    _, emb, _ = _load_backend()
    return emb

def load_vision():
    _, _, vis = _load_backend()
    return vis


# ══════════════════════════════════════════════════════════════════════════════
#  PDF PROCESSING  ·  PyMuPDF4LLM → Markdown-aware chunks
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(uploaded_file) -> list:
    """Convert PDF to Markdown via PyMuPDF4LLM; return LangChain Documents."""
    import pymupdf4llm
    from langchain_core.documents import Document

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        md_text = pymupdf4llm.to_markdown(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Wrap as a single Document so the splitter handles it uniformly
    return [Document(page_content=md_text, metadata={"source": uploaded_file.name})]


# ══════════════════════════════════════════════════════════════════════════════
#  VECTOR STORE + HYBRID RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

def build_vectorstore(docs: list, embeddings) -> tuple:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vs = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vs, chunks, len(chunks)


def load_existing_vectorstore(embeddings):
    from langchain_community.vectorstores import Chroma
    if Path(CHROMA_DIR).exists():
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return None


def build_hybrid_retriever(vs, chunks: list, top_k: int = TOP_K_DEFAULT):
    """EnsembleRetriever: 50% BM25 (keyword) + 50% Chroma (semantic)."""
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    bm25 = BM25Retriever.from_documents(chunks, k=top_k)
    dense = vs.as_retriever(search_kwargs={"k": top_k})
    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

QA_SYSTEM = """You are Study Buddy, an expert academic tutor. Use ONLY the provided context to answer questions clearly and thoroughly.
If the answer is not in the context, say so honestly.
When relevant, use numbered lists, bullet points, or markdown tables to structure your answer."""

QUIZ_PROMPT = """You are an expert educator. Based ONLY on the context below, create a short comprehension quiz.

Context:
{context}

Generate exactly 3 quiz questions (mix MCQ and True/False) as a valid JSON array. No markdown, no extra text.
[
  {{
    "type": "mcq",
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "Brief explanation of why this is correct."
  }},
  {{
    "type": "true_false",
    "question": "...",
    "answer": "True",
    "explanation": "Brief explanation."
  }}
]
Return ONLY the JSON array."""

SUMMARY_PROMPT = """Summarize this conversation history in 3-5 sentences, focusing on the key topics and concepts discussed.
Preserve technical terms and important distinctions. Write in third person.

History:
{history}

Summary:"""


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT MEMORY  ·  Rolling summary to keep context window lean
# ══════════════════════════════════════════════════════════════════════════════

def get_history_summary(llm) -> str:
    """Summarize last N messages to inject as context."""
    messages = st.session_state.messages
    if len(messages) < 4:
        return ""
    # Use last 6 turns (12 messages) at most
    recent = messages[-12:]
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}" for m in recent
    )
    try:
        if USE_GEMINI:
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(history=history_text))])
            return response.content
        else:
            return llm.invoke(SUMMARY_PROMPT.format(history=history_text))
    except Exception:
        return ""


def build_qa_prompt_with_memory(context: str, question: str, history_summary: str) -> str:
    memory_block = ""
    if history_summary:
        memory_block = f"\n\nConversation history summary:\n{history_summary}\n"
    return f"""Context from lecture notes:
{context}
{memory_block}
Student Question: {question}

Answer (be clear, educational, and use structure where helpful):"""


# ══════════════════════════════════════════════════════════════════════════════
#  QUIZ GENERATION + INTERACTIVE RENDER
# ══════════════════════════════════════════════════════════════════════════════

def generate_quiz(llm, context: str) -> list:
    prompt = QUIZ_PROMPT.format(context=context[:3500])
    try:
        if USE_GEMINI:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = llm.invoke([
                SystemMessage(content="Return ONLY a valid JSON array. No markdown."),
                HumanMessage(content=prompt),
            ])
            raw = response.content
        else:
            raw = llm.invoke(prompt)
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```")
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return []


def render_quiz(questions: list, quiz_key: str):
    """Interactive quiz with st.form, radio buttons, scoring, and per-question feedback."""
    if not questions:
        return

    # Unique state keys per quiz instance
    submitted_key = f"quiz_submitted_{quiz_key}"
    answers_key   = f"quiz_answers_{quiz_key}"
    score_key     = f"quiz_score_{quiz_key}"

    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False
    if answers_key not in st.session_state:
        st.session_state[answers_key] = {}

    st.markdown('<div class="kc-card">', unsafe_allow_html=True)
    st.markdown('<div class="kc-title">⚡ Quick Knowledge Check</div>', unsafe_allow_html=True)

    if st.session_state[submitted_key]:
        # ── Show scored results ──────────────────────────────────────────────
        score = st.session_state.get(score_key, 0)
        pct   = int((score / len(questions)) * 100)
        color = "var(--green)" if pct >= 70 else ("var(--amber)" if pct >= 40 else "var(--red)")

        st.markdown(
            f'<div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:{color};">'
            f'Score: {score}/{len(questions)} ({pct}%)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="kc-score-bar"><div class="kc-score-fill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True,
        )

        for i, q in enumerate(questions):
            q_key    = f"q{i}"
            correct  = q.get("answer", "")
            given    = st.session_state[answers_key].get(q_key, "")
            is_right = given.strip().lower() == correct.strip().lower()

            st.markdown(f"**Q{i+1}. {q.get('question','')}**")
            if q.get("type") == "mcq" and "options" in q:
                opts = q["options"]
                for k, v in opts.items():
                    marker = " ✅" if k == correct else (" ❌" if k == given and not is_right else "")
                    st.markdown(
                        f"<span style='font-size:.88rem;color:{'var(--green)' if k==correct else 'var(--text2)'};'>"
                        f"{k}) {v}{marker}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f"<span style='font-size:.88rem;color:var(--text2);'>Your answer: "
                    f"<b style='color:{'var(--green)' if is_right else 'var(--red)'};'>{given}</b> "
                    f"| Correct: <b style='color:var(--green);'>{correct}</b></span>",
                    unsafe_allow_html=True,
                )

            expl = q.get("explanation", "")
            if expl:
                st.markdown(
                    f"<div class='{'kc-feedback-correct' if is_right else 'kc-feedback-wrong'}'>"
                    f"{'✓' if is_right else '✗'} {expl}</div>",
                    unsafe_allow_html=True,
                )
            st.divider()

        if st.button("🔄 Retry Quiz", key=f"retry_{quiz_key}"):
            st.session_state[submitted_key] = False
            st.session_state[answers_key]   = {}
            st.rerun()

    else:
        # ── Render answerable form ───────────────────────────────────────────
        with st.form(key=f"quiz_form_{quiz_key}"):
            user_answers = {}
            for i, q in enumerate(questions):
                q_type = q.get("type", "mcq")
                st.markdown(f"**Q{i+1}. {q.get('question','')}**")

                if q_type == "mcq" and "options" in q:
                    opts = q["options"]
                    choices = [f"{k}) {v}" for k, v in opts.items()]
                    sel = st.radio(
                        label=f"q{i+1}",
                        options=choices,
                        label_visibility="collapsed",
                        key=f"radio_{quiz_key}_{i}",
                    )
                    user_answers[f"q{i}"] = sel[0] if sel else ""
                else:
                    sel = st.radio(
                        label=f"q{i+1}",
                        options=["True", "False"],
                        label_visibility="collapsed",
                        key=f"tf_{quiz_key}_{i}",
                    )
                    user_answers[f"q{i}"] = sel or ""

            submitted = st.form_submit_button("✅ Submit Answers", use_container_width=True)

        if submitted:
            score = 0
            for i, q in enumerate(questions):
                correct = q.get("answer", "").strip().lower()
                given   = user_answers.get(f"q{i}", "").strip().lower()
                if given == correct:
                    score += 1
            st.session_state[submitted_key] = True
            st.session_state[answers_key]   = user_answers
            st.session_state[score_key]     = score
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "messages":     [],      # {role, content, quiz, quiz_key, sources}
        "vectorstore":  None,
        "raw_chunks":   [],      # for BM25 hybrid retriever
        "doc_count":    0,
        "chunk_count":  0,
        "doc_names":    [],
        "backend_ok":   None,
        "top_k":        TOP_K_DEFAULT,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_backend() -> bool:
    if st.session_state.backend_ok is None:
        try:
            llm = load_llm()
            if USE_GEMINI:
                from langchain_core.messages import HumanMessage
                llm.invoke([HumanMessage(content="Say OK.")])
            else:
                llm.invoke("Say OK in one word.")
            st.session_state.backend_ok = True
        except Exception:
            st.session_state.backend_ok = False
    return st.session_state.backend_ok


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.6rem;">
      <div style="font-family:'Syne',sans-serif;font-size:1.45rem;font-weight:800;
                  background:linear-gradient(90deg,#4f8ef7,#9b72f7,#f472b6);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  line-height:1.1;">
        📚 Study Buddy
      </div>
      <div style="font-size:.74rem;color:#64748b;margin-top:.2rem;letter-spacing:.6px;">
        RAG v2 · AI-Powered Tutor
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Backend status ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
    backend_ok = check_backend()
    backend_label = "Gemini Cloud" if USE_GEMINI else "Ollama Local"
    backend_model  = GEMINI_LLM_MODEL if USE_GEMINI else OLLAMA_LLM_MODEL

    if backend_ok:
        st.markdown(f'<span class="pill pill-green">● {backend_label} Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="pill pill-red">● {backend_label} Offline</span>', unsafe_allow_html=True)
        if not USE_GEMINI:
            st.error("Start Ollama:\n```\nollama serve\n```")
        else:
            st.error("Check `GEMINI_API_KEY` in secrets.")

    st.markdown(
        f'<span class="pill pill-purple" style="margin-top:.4rem;display:inline-block">'
        f'🤖 {backend_model}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Study Material</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Drop PDF lecture / notes",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded_pdf:
        if uploaded_pdf.name not in st.session_state.doc_names:
            with st.spinner("🔬 Parsing PDF with PyMuPDF4LLM…"):
                try:
                    embeddings = load_embeddings()
                    docs = process_pdf(uploaded_pdf)
                    vs, chunks, n = build_vectorstore(docs, embeddings)
                    st.session_state.vectorstore  = vs
                    st.session_state.raw_chunks   = st.session_state.raw_chunks + chunks
                    st.session_state.chunk_count += n
                    st.session_state.doc_count   += 1
                    st.session_state.doc_names.append(uploaded_pdf.name)
                    st.success(f"✓ Indexed {n} chunks")
                except Exception as e:
                    st.error(f"PDF Error: {e}")
        else:
            st.info("Already loaded.")

    # Try loading existing DB
    if st.session_state.vectorstore is None:
        try:
            embeddings = load_embeddings()
            vs = load_existing_vectorstore(embeddings)
            if vs:
                st.session_state.vectorstore = vs
                st.caption("📂 Loaded existing knowledge base.")
        except Exception:
            pass

    if st.session_state.doc_names:
        for name in st.session_state.doc_names:
            st.markdown(
                f"<div style='font-size:.78rem;color:#64748b;margin-top:.2rem;'>📄 {name}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Vision Upload ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">👁️ Vision Analysis</div>', unsafe_allow_html=True)
    st.caption("Upload a diagram, chart, or handwritten notes for AI explanation.")
    uploaded_img = st.file_uploader(
        "Drop image (PNG/JPG)",
        type=["png","jpg","jpeg","webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="vision_uploader",
    )

    if uploaded_img:
        st.image(uploaded_img, use_column_width=True)
        vision_prompt = st.text_input(
            "What would you like to know about this image?",
            value="Explain this diagram or image in detail.",
            key="vision_prompt",
        )
        if st.button("🔍 Analyze Image", key="vision_btn"):
            if backend_ok:
                with st.spinner("Analyzing image…"):
                    try:
                        vision_fn = load_vision()
                        img_bytes = uploaded_img.read()
                        mime = f"image/{uploaded_img.type.split('/')[-1]}"
                        result = vision_fn(img_bytes, mime, vision_prompt)
                        st.markdown(
                            f'<div class="vision-card">'
                            f'<div class="vision-title">👁️ Vision Analysis</div>'
                            f'{result}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        # Also add to chat as a message
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"[Image uploaded: {uploaded_img.name}] {vision_prompt}",
                            "quiz": None,
                            "quiz_key": None,
                            "sources": [],
                        })
                        st.session_state.messages.append({
                            "role": "ai",
                            "content": f"**Vision Analysis of `{uploaded_img.name}`:**\n\n{result}",
                            "quiz": None,
                            "quiz_key": None,
                            "sources": [],
                        })
                    except Exception as e:
                        st.error(f"Vision error: {e}")
            else:
                st.warning("Backend not connected.")

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Session Stats</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, st.session_state.doc_count,             "Docs"),
        (c2, st.session_state.chunk_count,            "Chunks"),
        (c3, len(st.session_state.messages) // 2,     "Turns"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Advanced Settings"):
        st.session_state.top_k = st.slider("Retrieved chunks (Top-K)", 2, 10, st.session_state.top_k)
        hybrid_enabled = st.checkbox("Enable Hybrid Search (BM25 + Dense)", value=True)
        memory_enabled = st.checkbox("Enable Chat Memory", value=True)
        show_sources   = st.checkbox("Show source chunks", value=False)
        st.caption(
            f"Backend: `{'Gemini' if USE_GEMINI else 'Ollama'}`  ·  "
            f"Chunks: {CHUNK_SIZE}/{CHUNK_OVERLAP}"
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("♻️ Reset KB"):
            import shutil
            st.session_state.vectorstore = None
            st.session_state.raw_chunks  = []
            gc.collect()
            time.sleep(.8)
            if Path(CHROMA_DIR).exists():
                try:
                    shutil.rmtree(CHROMA_DIR)
                    for k in ["doc_count","chunk_count","doc_names","messages"]:
                        st.session_state[k] = [] if k in ("doc_names","messages") else 0
                    st.success("Reset!")
                    st.rerun()
                except PermissionError:
                    st.error("File locked. Try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — Header
# ══════════════════════════════════════════════════════════════════════════════

backend_badge = (
    '<span class="pill pill-gold" style="font-size:.68rem;">☁️ Gemini Cloud</span>'
    if USE_GEMINI else
    '<span class="pill pill-blue" style="font-size:.68rem;">🖥️ Ollama Local</span>'
)

st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.4rem;">
  <div>
    <h1 style="margin:0;font-size:2.1rem;
               background:linear-gradient(90deg,#4f8ef7,#9b72f7,#f472b6);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               line-height:1.15;">
      Study Buddy RAG
    </h1>
    <p style="margin:.3rem 0 0;color:#64748b;font-size:.86rem;">
      Hybrid Search · Vision · Memory · Interactive Quizzes
    </p>
  </div>
  <div style="margin-top:.4rem;">{backend_badge}</div>
</div>
""", unsafe_allow_html=True)

# ── Onboarding hint ───────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="hint-card">
      <div style="font-family:'Syne',sans-serif;font-weight:800;margin-bottom:.4rem;">
        👋 Welcome to Study Buddy v2
      </div>
      <div style="font-size:.87rem;color:#94a3b8;line-height:1.8;">
        1. Upload a PDF lecture or notes in the sidebar (tables &amp; diagrams supported)<br>
        2. Upload an image in the Vision panel to explain diagrams<br>
        3. Ask any question below — hybrid search + memory = better answers<br>
        4. Take the auto-generated interactive quiz to reinforce learning
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CHAT HISTORY RENDER
# ══════════════════════════════════════════════════════════════════════════════

# Resolve settings from sidebar (safe defaults if expander not expanded)
try:
    _hybrid  = hybrid_enabled
    _memory  = memory_enabled
    _sources = show_sources
except NameError:
    _hybrid  = True
    _memory  = True
    _sources = False

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="bubble-user">'
            f'<div class="bubble-label">You</div>'
            f'{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bubble-ai">'
            f'<div class="bubble-label">Study Buddy</div>'
            f'{msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Source chunks
        if _sources and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source chunk(s)"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="src-chunk">{src[:500]}{"…" if len(src)>500 else ""}</div>',
                        unsafe_allow_html=True,
                    )
        # Interactive quiz
        if msg.get("quiz") and msg.get("quiz_key"):
            render_quiz(msg["quiz"], msg["quiz_key"])


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT + RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

query = st.chat_input(
    "Ask about your study material…",
    disabled=(not check_backend()),
)

if query:
    st.session_state.messages.append({
        "role":     "user",
        "content":  query,
        "quiz":     None,
        "quiz_key": None,
        "sources":  [],
    })

    if not st.session_state.vectorstore:
        st.session_state.messages.append({
            "role":     "ai",
            "content":  "⚠️ No study material loaded. Please upload a PDF in the sidebar first.",
            "quiz":     None,
            "quiz_key": None,
            "sources":  [],
        })
        st.rerun()

    with st.spinner("🔍 Thinking…"):
        try:
            llm        = load_llm()
            vs         = st.session_state.vectorstore
            top_k      = st.session_state.top_k

            # ── Choose retriever ────────────────────────────────────────────
            if _hybrid and st.session_state.raw_chunks:
                retriever = build_hybrid_retriever(vs, st.session_state.raw_chunks, top_k)
            else:
                retriever = vs.as_retriever(search_kwargs={"k": top_k})

            src_docs     = retriever.invoke(query)
            context_str  = "\n\n".join(d.page_content for d in src_docs)
            source_texts = [d.page_content for d in src_docs]

            # ── Chat memory ─────────────────────────────────────────────────
            history_summary = get_history_summary(llm) if _memory else ""

            # ── Build full prompt ───────────────────────────────────────────
            full_prompt = build_qa_prompt_with_memory(context_str, query, history_summary)

            # ── Invoke LLM ──────────────────────────────────────────────────
            if USE_GEMINI:
                from langchain_core.messages import SystemMessage, HumanMessage
                response = llm.invoke([
                    SystemMessage(content=QA_SYSTEM),
                    HumanMessage(content=full_prompt),
                ])
                answer = response.content
            else:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.output_parsers import StrOutputParser
                prompt_tmpl = ChatPromptTemplate.from_messages([
                    ("system", QA_SYSTEM),
                    ("human",  "{input}"),
                ])
                chain  = prompt_tmpl | llm | StrOutputParser()
                answer = chain.invoke({"input": full_prompt})

            # ── Generate quiz ───────────────────────────────────────────────
            quiz_context = "\n\n".join(d.page_content for d in src_docs[:3])
            quiz         = generate_quiz(llm, quiz_context)
            quiz_key     = str(int(time.time()))

            st.session_state.messages.append({
                "role":     "ai",
                "content":  answer,
                "quiz":     quiz,
                "quiz_key": quiz_key,
                "sources":  source_texts,
            })

        except Exception as e:
            st.session_state.messages.append({
                "role":     "ai",
                "content":  f"❌ Error: {e}\n\nCheck that your backend is running and models are available.",
                "quiz":     None,
                "quiz_key": None,
                "sources":  [],
            })

    st.rerun()
