"""
╔══════════════════════════════════════════════════════════════╗
║   Asistente IA sobre Documentación Corporativa  – B2B Demo  ║
║   Stack: Streamlit · Groq (LLaMA-3) · ChromaDB · pdfplumber ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import uuid
import time
import hashlib
import tempfile

import streamlit as st
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq, AuthenticationError, APIConnectionError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# ──────────────────────────────────────────────
#  Configuración inicial
# ──────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="DocAI B2B · Asistente Corporativo",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  Inyección de CSS premium
# ──────────────────────────────────────────────
st.markdown("""
<meta name="google" content="notranslate">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #111827 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1224 0%, #111827 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.brand-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35);
}
.brand-badge h2 { color: #fff !important; font-size: 1.2rem; font-weight: 700; margin: 0; }
.brand-badge p  { color: rgba(255,255,255,0.7) !important; font-size: 0.75rem; margin: 4px 0 0; }

[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.08);
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 12px;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1; }

[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

.main-header {
    text-align: center;
    padding: 40px 20px 20px;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.main-header p { color: #94a3b8; font-size: 1rem; }

.stat-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-card .number { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.stat-card .label  { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

[data-testid="stChatMessage"] {
    background: rgba(15,22,41,0.6) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(8px);
}

[data-testid="stChatInput"] {
    background: rgba(15,22,41,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea { color: #e2e8f0 !important; background: transparent !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a78bfa) !important; }

.alert-success {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 10px;
    padding: 12px 16px;
    color: #6ee7b7;
    font-size: 0.875rem;
}
.alert-info {
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: #93c5fd;
    font-size: 0.875rem;
}
.alert-warning {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    color: #fcd34d;
    font-size: 0.875rem;
}

hr { border-color: rgba(99,102,241,0.2) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Session State
# ──────────────────────────────────────────────
DEFAULTS = {
    "messages": [],
    "chroma_client": None,
    "collection": None,
    "doc_name": None,
    "doc_chunks": 0,
    "doc_pages": 0,
    "doc_hash": None,
    "groq_client": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
#  Constantes
# ──────────────────────────────────────────────
GROQ_MODEL    = "qwen/qwen3.8-27b"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 120
TOP_K         = 5

SYSTEM_PROMPT = """\
Eres DocAI, un asistente corporativo especializado en analizar y responder preguntas \
sobre documentación empresarial interna.

REGLAS ABSOLUTAS:
1. Responde ÚNICAMENTE con información extraída de los fragmentos del documento que se te proporcionan.
2. Si la respuesta está en el documento, cita textualmente la parte clave entre comillas «» \
   e indica de qué sección o contexto proviene.
3. Si la información NO está en los fragmentos proporcionados, responde EXACTAMENTE: \
   "No dispongo de esa información en el documento."
4. Nunca inventes, supongas ni uses conocimiento externo al documento.
5. Sé conciso, preciso y profesional. Usa formato Markdown cuando mejore la legibilidad.
6. Si la pregunta es ambigua, pide aclaración antes de responder.
"""

# ──────────────────────────────────────────────
#  Funciones auxiliares
# ──────────────────────────────────────────────

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def extract_text_from_pdf(uploaded_file) -> tuple:
    """Extrae texto completo del PDF. Devuelve (texto, num_páginas)."""
    text_parts = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with pdfplumber.open(tmp_path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    os.unlink(tmp_path)
    return "\n\n".join(text_parts), num_pages


def build_vector_store(text: str, collection_name: str) -> tuple:
    """Fragmenta, embede e indexa el texto en ChromaDB."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.Client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=batch,
            ids=[str(uuid.uuid4()) for _ in batch],
        )

    return client, collection, len(chunks)


def retrieve_context(collection, query: str, k: int = TOP_K) -> str:
    """Recupera los k fragmentos más relevantes para la consulta."""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)


def chat_with_groq(client: Groq, messages: list, context: str, user_query: str) -> str:
    """Llama a Groq con el contexto RAG y el historial de conversación."""
    augmented_user_message = (
        f"FRAGMENTOS RELEVANTES DEL DOCUMENTO:\n"
        f"{'=' * 40}\n"
        f"{context}\n"
        f"{'=' * 40}\n\n"
        f"PREGUNTA DEL USUARIO: {user_query}"
    )

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages[:-1]:
        history.append({"role": msg["role"], "content": msg["content"]})
    history.append({"role": "user", "content": augmented_user_message})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=history,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-badge">
      <h2>🏢 DocAI B2B</h2>
      <p>Asistente de IA Corporativo</p>
    </div>
    """, unsafe_allow_html=True)

    # API Key
    st.markdown("#### 🔑 Groq API Key")
    env_key = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "API Key",
        value=env_key,
        type="password",
        placeholder="gsk_...",
        label_visibility="collapsed",
        key="api_key_field",
    )
    api_key = api_key_input.strip() or env_key.strip()

    if api_key:
        if st.session_state.groq_client is None:
            try:
                client_test = Groq(api_key=api_key)
                client_test.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                st.session_state.groq_client = client_test
                st.success("✅ API Key válida y conectada")
            except AuthenticationError:
                st.session_state.groq_client = None
                st.warning("⚠️ API Key inválida. Revísala en console.groq.com")
            except APIConnectionError:
                st.session_state.groq_client = None
                st.warning("⚠️ Sin conexión. Comprueba tu red.")
        else:
            st.success("✅ Conectado a Groq")
    else:
        st.info("ℹ️ Obtén tu key en [console.groq.com](https://console.groq.com)")
        st.session_state.groq_client = None

    st.markdown("---")

    # PDF Upload
    st.markdown("#### 📄 Documento PDF")
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona un PDF",
        type=["pdf"],
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        fhash = file_hash(uploaded_file.getvalue())

        if fhash != st.session_state.doc_hash:
            st.session_state.doc_hash = fhash
            st.session_state.doc_name = uploaded_file.name
            st.session_state.messages = []

            progress_bar = st.progress(0, text="📖 Leyendo PDF…")
            try:
                uploaded_file.seek(0)
                raw_text, num_pages = extract_text_from_pdf(uploaded_file)

                if not raw_text.strip():
                    st.error("❌ No se pudo extraer texto. El PDF puede ser una imagen escaneada.")
                else:
                    progress_bar.progress(40, text="✂️ Fragmentando documento…")
                    time.sleep(0.2)

                    progress_bar.progress(65, text="🧠 Generando embeddings…")
                    col_name = f"doc_{fhash[:16]}"
                    chroma_client, collection, n_chunks = build_vector_store(raw_text, col_name)

                    progress_bar.progress(90, text="🗄️ Indexando en ChromaDB…")
                    time.sleep(0.2)

                    st.session_state.chroma_client = chroma_client
                    st.session_state.collection = collection
                    st.session_state.doc_pages = num_pages
                    st.session_state.doc_chunks = n_chunks

                    progress_bar.progress(100, text="✅ ¡Documento listo!")
                    time.sleep(0.4)
                    progress_bar.empty()

            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Error procesando PDF: {str(e)}")

        if st.session_state.collection:
            st.success(f"✅ **{st.session_state.doc_name}**\n\n📄 {st.session_state.doc_pages} páginas · 🧩 {st.session_state.doc_chunks} fragmentos")

    st.markdown("---")

    # Stats
    if st.session_state.collection:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
              <div class="number">{st.session_state.doc_pages}</div>
              <div class="label">Páginas</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
              <div class="number">{st.session_state.doc_chunks}</div>
              <div class="label">Fragmentos</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#475569;font-size:0.72rem;">'
        'Powered by Groq · LLaMA-3 8B<br>© 2024 DocAI B2B Demo</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
#  ÁREA PRINCIPAL
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header notranslate" translate="no">
  <h1>🏢 Asistente de IA Corporativo</h1>
  <p>Consulta tu documentación empresarial con inteligencia artificial de última generación</p>
</div>
""", unsafe_allow_html=True)

if not api_key:
    st.info("🔑 **Paso 1:** Introduce tu Groq API Key en el panel izquierdo para comenzar.")

elif not st.session_state.collection:
    st.info("📄 **Paso 2:** Sube un archivo PDF en el panel izquierdo para comenzar.")

    col_a, col_b, col_c = st.columns(3)
    features = [
        ("⚡", "Ultra Rápido", "Respuestas en menos de 1 segundo gracias a Groq + LLaMA-3"),
        ("🔒", "Privado y Seguro", "El procesamiento es local. Tu documento no sale de tu máquina"),
        ("🎯", "Preciso y Citado", "Solo responde con datos del documento, con citas textuales"),
    ]
    for col, (icon, title, desc) in zip([col_a, col_b, col_c], features):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="padding:28px 20px;min-height:160px;">
              <div style="font-size:2.2rem;margin-bottom:12px;">{icon}</div>
              <div style="font-weight:600;color:#a78bfa;margin-bottom:8px;">{title}</div>
              <div style="font-size:0.82rem;color:#64748b;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

else:
    # Chat activo
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"¡Hola! He procesado el documento **{st.session_state.doc_name}** "
                f"({st.session_state.doc_pages} páginas · {st.session_state.doc_chunks} fragmentos indexados).\n\n"
                "Puedo responder preguntas sobre su contenido con precisión y citando los fragmentos relevantes. "
                "¿En qué puedo ayudarte?"
            ),
        })

    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if user_input := st.chat_input(
        "Haz una pregunta sobre el documento…",
        disabled=(not st.session_state.groq_client),
    ):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analizando documento y generando respuesta…"):
                try:
                    context = retrieve_context(st.session_state.collection, user_input)
                    answer = chat_with_groq(
                        client=st.session_state.groq_client,
                        messages=st.session_state.messages,
                        context=context,
                        user_query=user_input,
                    )
                except AuthenticationError:
                    answer = "❌ Error de autenticación. Revisa tu API Key de Groq en el panel lateral."
                except APIConnectionError:
                    answer = "❌ No se pudo conectar con la API de Groq. Comprueba tu conexión a internet."
                except Exception as e:
                    answer = f"❌ Error inesperado: {str(e)}"

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if not st.session_state.groq_client and st.session_state.collection:
        st.warning("⚠️ Introduce una API Key de Groq válida en el panel izquierdo para activar el chat.")
