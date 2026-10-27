# 🏢 DocAI B2B — Asistente de IA sobre Documentación Corporativa

> Demo B2B de un asistente conversacional que responde preguntas sobre cualquier PDF corporativo usando RAG (Retrieval-Augmented Generation) con Groq + LLaMA-3.

---

## 🚀 Stack Tecnológico

| Capa | Tecnología |
|---|---|
| **Frontend / UI** | Streamlit |
| **Motor de IA** | Groq API · LLaMA-3 8B (`llama3-8b-8192`) |
| **Extracción PDF** | pdfplumber |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Base de datos vectorial** | ChromaDB (en memoria) |
| **Variables de entorno** | python-dotenv |

---

## 📁 Estructura del Proyecto

```
appIA B2B/
├── app.py               ← Aplicación principal (todo en un archivo)
├── requirements.txt     ← Dependencias Python
├── .env.example         ← Plantilla de variables de entorno
├── .env                 ← Tu API Key real (NO subir a git)
└── README.md            ← Este archivo
```

---

## ⚙️ Instalación y Puesta en Marcha

### 1. Crea el entorno virtual e instala dependencias

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (macOS/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configura tu API Key de Groq

```bash
# Copia la plantilla
cp .env.example .env
```

Edita `.env` y pega tu API Key:
```
GROQ_API_KEY=gsk_tu_api_key_aqui
```

> 💡 Obtén tu API Key **gratuita** en [console.groq.com](https://console.groq.com/) — no necesita tarjeta de crédito.

### 3. Lanza la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

---

## 🎯 Cómo Usar la Aplicación

1. **API Key** → Introduce tu Groq API Key en la barra lateral (o configúrala en `.env`)
2. **PDF** → Arrastra y suelta cualquier PDF corporativo en el sidebar
3. **Espera** → La barra de progreso te muestra el estado del procesamiento
4. **Chatea** → Haz preguntas en lenguaje natural sobre el documento

---

## 🔒 Privacidad y Seguridad

- El texto del PDF se procesa **localmente** en tu máquina
- ChromaDB se usa **en memoria** — no persiste datos entre sesiones
- Solo el texto de los fragmentos relevantes se envía a Groq para generar la respuesta
- **Nunca subas `.env` a un repositorio público**

---

## 🧠 Cómo Funciona el RAG

```
PDF ──► pdfplumber (extracción) ──► RecursiveCharacterTextSplitter (chunks de 800 tokens)
                                         │
                                         ▼
                                 SentenceTransformer (embeddings)
                                         │
                                         ▼
                                    ChromaDB (índice vectorial)
                                         │
                        Pregunta del usuario ──► Búsqueda por similitud (Top-5)
                                         │
                                         ▼
                              Groq API + LLaMA-3 8B
                              (responde solo con los fragmentos recuperados)
```

---

## 📦 Dependencias Principales

```
streamlit>=1.32.0
groq>=0.5.0
chromadb>=0.4.22
pdfplumber>=0.10.3
langchain-text-splitters>=0.0.1
sentence-transformers>=2.6.1
python-dotenv>=1.0.1
```

---

## ❓ Problemas Comunes

| Problema | Solución |
|---|---|
| `AuthenticationError` | La API Key es inválida — verifica en console.groq.com |
| PDF sin texto extraíble | El PDF puede ser una imagen escaneada — usa un OCR primero |
| `ModuleNotFoundError` | Asegúrate de haber activado el venv y ejecutado `pip install -r requirements.txt` |
| Puerto 8501 ocupado | Usa `streamlit run app.py --server.port 8502` |

---

*Desarrollado como demo B2B · Powered by [Groq](https://groq.com) + [Streamlit](https://streamlit.io)*
