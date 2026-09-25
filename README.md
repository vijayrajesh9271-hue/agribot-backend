# AI Agronomist — Backend

A multi-agent, multimodal AI agriculture advisory API. It combines computer-vision crop disease detection, a RAG knowledge base, satellite NDVI monitoring, and Gemini-powered conversational experts (Soil / Crop / Economics) into a single FastAPI backend — with support for English, Roman Urdu, and Urdu-script conversations.


## Download/access ui from that Repo:

LINK: https://github.com/jeetashkumar/Agronomy_application.git

---

## What This Does

A user (typically an agronomist or consultant) can:

- **Chat** in natural language about soil health, crop management, or agricultural economics.
- **Upload a crop photo** and get an automatic disease diagnosis (cotton, wheat, corn, or rice) with a confidence score and treatment advice.
- **Ask about a field's NDVI** (vegetation health index) using live Sentinel-2 satellite data.
- **Query a private knowledge base** of uploaded agronomy PDFs — answered directly from source material when a good match exists, otherwise routed to Gemini.
- **Speak instead of type** — voice messages are transcribed locally.
- **Get replies in their own language** — the AI mirrors English or Roman Urdu automatically, or can be forced into Urdu (Perso-Arabic) script.
- **Export a conversation as a PDF**, delete conversations, and pick up chat history across sessions.

---

## Architecture

| Layer | Technology | File |
|---|---|---|
| API server | FastAPI + Uvicorn | `api.py` |
| Crop disease detection | TensorFlow/Keras (CPU) | `vision.py` |
| Multi-agent orchestration | OpenAI Agents SDK + Gemini | `agronomist.py` |
| Knowledge base (RAG) | pgvector (Neon Postgres) + `sentence-transformers` | `rag.py` |
| Database | Async SQLAlchemy + Postgres | `db.py` |
| Satellite NDVI | Google Earth Engine | called from `agronomist.py` |
| Web research | Tavily Search API | called from `agronomist.py` |
| Speech-to-text | Whisper (local, CPU) | called from `api.py` |
| Abuse/safety filtering | — | `guards.py` |

**Startup order matters.** Heavy modules (`vision`, `agronomist`) are deliberately *not* imported at the top of `api.py` — they're loaded inside a staged FastAPI `startup` event (DB → RAG table/embedder → vision models → agronomist agents) so the server binds and starts responding to health checks quickly, with slow model loading happening in the background afterward.

---

## Prerequisites

- **Python 3.11** (the project was built and tested on this version)
- **PostgreSQL with the `pgvector` extension** — a free [Neon](https://neon.tech) project works well and is what this project was built against
- **`ffmpeg`** installed system-wide (used for audio preprocessing before transcription)
  ```bash
  sudo apt install ffmpeg
  ```
- API keys/accounts for:
  - **Google Gemini API** ([ai.google.dev](https://ai.google.dev)) — the core conversational model
  - **Tavily** ([tavily.com](https://tavily.com)) — web search tool used by the agents
  - **Google Earth Engine** ([earthengine.google.com](https://earthengine.google.com)) — for NDVI crop monitoring (optional — the app runs without it, just without that one tool)

---

## Step-by-Step Setup

### 1. Clone the repository

```bash
git clone https://github.com/jeetashkumar/Agronomist-AI-Chatbot.git
cd Agronomist-AI-Chatbot
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, here's the core set this project depends on:

> **Note on `openai-agents`:** this is the package providing `from agents import Agent, Runner, ...` used in `agronomist.py`.

### 4. Set up your `.env` file

Create a `.env` file in the project root (see [Environment Variables Reference](#environment-variables-reference) below for what each value means):

```bash
touch .env
```

Populate it with your actual values — **never commit this file.** It's already covered by `.gitignore`.

### 5. Set up Google Earth Engine authentication (optional, for NDVI monitoring)

```bash
earthengine authenticate
```
Follow the browser prompt to log in with the Google account tied to your Earth Engine project, then add your `GEE_PROJECT_ID` to `.env`.

### 6. Add your trained model files

Place your 5 Keras model files and their class-index JSON files inside a `models/` folder in the project root:

```
models/
├── crops_classifier.keras
├── class_indices.json
├── cottonbest.keras
├── cotton_class_indices.json
├── best_model.keras              (wheat)
├── wheat_class_indices.json
├── cornbest.keras
├── corn_class_indices.json
├── rice_best_model.keras
└── rice_class_indices.json
```

If any class-index JSON is missing, `vision.py` falls back to a hardcoded label mapping, so the app will still run — but confirm the fallback labels actually match your trained model's output order.

### 7. Initialize the database

The database schema is created automatically on server startup (`init_db()` in `db.py` calls `Base.metadata.create_all`) — no separate migration step is needed for a fresh database. Similarly, the RAG knowledge base table (`rag_chunks`) is created automatically via `init_vector_db()` during startup.

### 8. (Optional) Add PDFs to the knowledge base

Place any agronomy reference PDFs into a `knowledge_base/` folder, or upload them once the server is running via:

```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@/path/to/your/document.pdf" \
  -F "email=you@example.com"
```

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ Yes | Postgres connection string (Neon or any Postgres with `pgvector` installed). Format: `postgresql://user:password@host/dbname` |
| `GEMINI_API_KEY` | ✅ Yes | Your Google AI Studio / Gemini API key |
| `TAVILY_API_KEY` | ✅ Yes | Tavily Search API key, used by the `agri_search` tool |
| `GEE_PROJECT_ID` | ⛔ Optional | Your Google Earth Engine Cloud project ID. Without it, NDVI monitoring is disabled but the rest of the app works normally |
| `OPENAI_API_KEY` | ⛔ Optional | Only used for Agents SDK tracing/telemetry — not needed for core functionality |

---

## Running the Server

```bash
python api.py
```

The server starts on `http://0.0.0.0:8000` by default (see `main()` at the bottom of `api.py` to change host/port). Interactive API docs are available at:

```
http://localhost:8000/docs
```

**Watch the startup log** — you should see each stage complete in order:
```
✅ Database initialized successfully
✅ RAG table ready
✅ Embedding model preloaded
✅ Vision models ready
✅ Agronomist agents ready
🎉 Startup sequence complete
```
If any stage logs a `⚠️` warning instead of `✅`, that specific feature will be degraded (e.g., Earth Engine failing to initialize just disables NDVI — it won't crash the whole server).

## Deploying On Railway

Create a Railway service from this repository and set its **Root Directory** to `Agronomist-AI-Chatbot`. The included `railway.json` uses `requirements-railway.txt` and starts Uvicorn on Railway's `$PORT` automatically.

Add these Railway variables (do not commit `.env` or secret values):

```text
DATABASE_URL=your-postgres-url
GEMINI_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
OPENAI_API_KEY=your-openai-key       # optional tracing
GEE_PROJECT_ID=your-earth-engine-project  # optional NDVI
SKIP_RAG_STARTUP=false
```

After deployment, verify `https://YOUR-RAILWAY-DOMAIN/api/health` returns `status: healthy`. Use that Railway domain as `AGRONOMIST_API_URL` in the WhatsApp bot service.

---

## Project Structure

```
.
├── api.py              # FastAPI app, all HTTP routes, startup sequence
├── agronomist.py        # Multi-agent orchestration, Gemini integration, RAG synthesis
├── vision.py             # Crop classifier + 4 disease detection models
├── rag.py                # PDF ingestion, embeddings, pgvector similarity search
├── db.py                 # SQLAlchemy models and all database operations
├── guards.py              # Abuse/safety content filtering (referenced, not detailed here)
├── models/                # Trained .keras model files + class index JSONs (not committed if large — see Git LFS note below)
├── knowledge_base/         # Uploaded PDFs for the RAG system
└── .env                    # Your local secrets (never committed)
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/auth/signup` | Create an account |
| `POST` | `/api/auth/login` | Log in |
| `GET` | `/api/auth/me` | Fetch current user info |
| `POST` | `/api/auth/forgot-password` | Request a password reset code |
| `POST` | `/api/auth/verify-reset-code` | Verify a reset code |
| `POST` | `/api/auth/reset-password` | Set a new password |
| `DELETE` | `/api/auth/delete-account` | Permanently delete an account (archived first) |
| `POST` | `/api/chat` | Send a message (+ optional images) to the AI |
| `POST` | `/api/chat/new` | Start a new conversation |
| `GET` | `/api/history/{email}` | Get conversation list + recent messages |
| `GET` | `/api/conversation/{email}/{id}` | Get all messages in one conversation |
| `DELETE` | `/api/conversation/{email}/{id}` | Delete a conversation (archived first) |
| `GET` | `/api/conversation/{email}/{id}/export` | Export a conversation as PDF |
| `POST` | `/api/transcribe` | Transcribe a voice recording to text |
| `POST` | `/api/rag/upload` | Upload a PDF to the knowledge base |
| `GET` | `/api/rag/status` | Check how many chunks are indexed |
| `POST` | `/api/message/feedback` | Submit thumbs up/down on a message |
| `GET` | `/api/health` | Health check |

Full interactive documentation with request/response schemas is available at `/docs` once the server is running.

---

## Known Limitations

- **Password hashing** currently uses plain SHA-256 without a salt — acceptable for development, should be upgraded to `bcrypt`/`argon2` before handling real user data in production.
- **No rate limiting** — nothing currently prevents one user from exhausting the shared Gemini API quota.
- **No automated test suite** yet.
- **Google Earth Engine's free tier is noncommercial-only.** If this app is used with paying customers, a paid Earth Engine commercial account is required — see Earth Engine's terms of service.
- **Gemini model names change.** Google periodically deprecates model versions (this project has already migrated once, from `gemini-2.5-flash`). If you see a `404 model not found` error, check [ai.google.dev](https://ai.google.dev/gemini-api/docs/models) for the current model lineup.
