import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Information Assistant",
    page_icon="🩺",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Root variables ── */
:root {
    --bg:         #0d1117;
    --surface:    #161b22;
    --surface2:   #1e2733;
    --border:     rgba(56, 139, 253, 0.15);
    --accent:     #3b82f6;
    --accent-glow:#1d4ed8;
    --teal:       #14b8a6;
    --text:       #e6edf3;
    --muted:      #8b949e;
    --danger:     #f87171;
    --radius:     14px;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(59,130,246,.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(20,184,166,.12) 0%, transparent 55%),
        var(--bg) !important;
    min-height: 100vh;
}

/* hide default header / toolbar */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
header { display: none !important; }

/* ── Main column width ── */
.block-container {
    max-width: 740px !important;
    padding: 2.5rem 1.5rem 4rem !important;
}

/* ── Hero header ── */
.med-hero {
    text-align: center;
    padding: 2.8rem 0 1.8rem;
    animation: fadeDown .7s ease both;
}
.med-hero .pulse-ring {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 72px;
    height: 72px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    box-shadow: 0 0 0 0 rgba(59,130,246,.5);
    animation: pulse 2.4s infinite;
    margin-bottom: 1rem;
    font-size: 2rem;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(59,130,246,.5); }
    70%  { box-shadow: 0 0 0 18px rgba(59,130,246,0); }
    100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); }
}
.med-hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.9rem, 5vw, 2.6rem);
    font-weight: 400;
    letter-spacing: -.02em;
    color: var(--text);
    margin: 0 0 .45rem;
    line-height: 1.15;
}
.med-hero h1 span { color: var(--accent); }
.med-hero p {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: .01em;
}

/* ── Divider ── */
.med-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.6rem 0;
}

/* ── Input card ── */
.stTextInput > div > div {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: .55rem 1rem !important;
    transition: border-color .25s, box-shadow .25s;
}
.stTextInput > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.15) !important;
}
.stTextInput input {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    background: transparent !important;
}
.stTextInput input::placeholder { color: var(--muted) !important; }
label[data-testid="stWidgetLabel"] {
    color: var(--muted) !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    margin-bottom: .3rem !important;
}

/* ── Answer card ── */
.answer-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
    position: relative;
    overflow: hidden;
    animation: fadeUp .5s ease both;
}
.answer-card::before {
    content: '';
    position: absolute;
    inset: 0 auto 0 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--teal));
    border-radius: 3px 0 0 3px;
}
.answer-label {
    display: flex;
    align-items: center;
    gap: .5rem;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .9rem;
}
.answer-label svg { flex-shrink: 0; }
.answer-body {
    color: var(--text);
    font-size: 1rem;
    line-height: 1.75;
    font-weight: 300;
}

/* ── Animations ── */
@keyframes fadeDown {
    from { opacity:0; transform: translateY(-16px); }
    to   { opacity:1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
}

/* ── Footer ── */
.med-footer {
    text-align: center;
    margin-top: 3rem;
    color: var(--muted);
    font-size: .78rem;
    letter-spacing: .03em;
}
.med-footer span { color: var(--danger); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="med-hero">
    <div class="pulse-ring">🩺</div>
    <h1>Medical <span>Information</span> Assistant</h1>
    <p>Ask any medical question — answers sourced directly from your documents.</p>
</div>
<div class="med-divider"></div>
""", unsafe_allow_html=True)
# ── PDF Upload ──────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📄 Upload a PDF",
    type=["pdf"]
)

if uploaded_file is None:
    st.info("Please upload a PDF first.")

# ── Load embeddings ───────────────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

retriever = None

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        retriever = db.as_retriever(
            search_kwargs={"k": 2}
        )

    st.success(
        f"PDF processed successfully! ({len(chunks)} chunks)"
    )

# ── Load Groq LLM ─────────────────────────────────────────────────────────────
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# ── User Input ────────────────────────────────────────────────────────────────
question = st.text_input(
    "Your question",
    placeholder="e.g. What are the symptoms of Type 2 Diabetes?",
)

if question and retriever:

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer ONLY from the provided context.

    If the answer is not available in the context,
    say:
    "This information is not available in the provided document."

    Context:
    {context}

    Question:
    {question}
    """

    with st.spinner("Searching the knowledge base…"):
        response = llm.invoke(prompt)

    # ── Answer card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-label">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 12l2 2 4-4"/>
                <circle cx="12" cy="12" r="10"/>
            </svg>
            Answer
        </div>
        <div class="answer-body">{response.content}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="med-footer">
    Powered by Groq · LangChain · ChromaDB &nbsp;|&nbsp; For informational purposes only.<br>
    Always consult a qualified healthcare professional for medical advice.
</div>
""", unsafe_allow_html=True)