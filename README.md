# CreatorOS — AI Service

A Python (FastAPI) microservice that powers CreatorOS's AI features using a Retrieval-Augmented Generation (RAG) pipeline built with LangChain. It is called exclusively by the Node.js backend — it is never exposed directly to the frontend/browser.

## Tech Stack

- **FastAPI** + **Uvicorn**
- **LangChain** (`langchain-core`, `langchain-google-genai`, `langchain-huggingface`, `langchain-chroma`)
- **Chroma** — local vector database
- **Google Generative AI Embeddings** (`gemini-embedding-001`) — text embeddings
- **HuggingFace Inference API** (`Qwen/Qwen2.5-1.5B-Instruct` via Featherless) — chat/completion model, called via API (no local model download)
- **PyMongo** — direct MongoDB access for reading cached video data

## Features

Three AI-powered endpoints, each backed by the creator's own YouTube video data (fetched from MongoDB, scoped per-user):

1. **Chat (`/chat`)** — RAG-based Q&A. Retrieves the most relevant videos for a question (MMR retrieval) and answers using only that context.
2. **Trend Insights (`/api/trends`)** — Ranks the creator's videos by views and asks the LLM to identify patterns (topics, title styles, tags) behind top performers.
3. **Suggest New Topic (`/api/suggest-topic`)** — Takes a short multi-step questionnaire (category, target audience, tone, goal) plus the creator's past titles, and generates a tailored new video idea.

## Project Structure

```
AI/
├── app.py                 # FastAPI app: routes, prompts, RAG pipeline
├── videoservice.py         # MongoDB connection + raw video queries
├── get_data.py              # Wraps the /api/data route logic for in-process reuse
├── requirements.txt
├── .env
└── chroma_db/               # Local Chroma persistence directory (per-creator collections)
```

## Environment Variables

```env
MONGODB_URI=mongodb+srv://...
GOOGLE_API_KEY=your_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

## Getting Started

```bash
# Create and activate a virtual environment inside the AI/ folder
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

uvicorn app:app --reload --port 8000
```

The service runs on `http://localhost:8000`. Interactive API docs are available at `http://localhost:8000/docs`.

> If using `uv` as the package manager, prefix commands with `uv run`, e.g. `uv run uvicorn app:app --reload --port 8000`.

## Endpoints

| Method | Path | Body / Query | Description |
|---|---|---|---|
| GET | `/` | — | Health check |
| GET | `/health` | — | Health check |
| GET | `/api/data` | `?creatorId=` | Raw video list for a creator (from MongoDB) |
| POST | `/chat` | `{ creatorId, question }` | RAG chat response |
| GET | `/api/trends` | `?creatorId=` | AI-generated trend insights |
| POST | `/api/suggest-topic` | `{ creatorId, category, target_audience, tone, goal }` | AI-generated topic suggestion |

All responses follow `{ success: boolean, ...payload }`. On failure, `{ success: false, message: string }`.

## How the RAG Pipeline Works

1. **Fetch** — `get_data(creatorId)` pulls the creator's video documents from MongoDB (title, description, stats, publish date).
2. **Document conversion** — Each video is converted into a readable text block (`build_documents`) rather than being embedded as raw JSON — this produces better embeddings and retrieval quality.
3. **Embed + store** — Documents are embedded via Gemini (`gemini-embedding-001`) and stored in a **per-creator Chroma collection** (`creator_{creatorId}`), so one user's data is never mixed with another's.
4. **Retrieve** — For a given question, an MMR (Maximal Marginal Relevance) retriever pulls the top 4 most relevant, diverse video documents.
5. **Generate** — The retrieved context is injected into a system prompt instructing the model to answer *only* from the provided context, then passed to the HuggingFace-hosted chat model.

## Security Notes

- `creatorId` is **never** accepted from the browser directly. The Node.js backend resolves it from a verified JWT and passes it to this service — this service trusts whatever `creatorId` it receives, because only the backend is allowed to call it (not exposed publicly in production; should sit behind Railway's internal network or equivalent).
- Each creator's vector data is isolated in its own Chroma collection, keyed by their MongoDB user ID.

## Known Limitations / Future Work

- **No response caching for AI endpoints yet** — every `/chat`, `/api/trends` call re-embeds the creator's documents into Chroma unless a collection already exists with that name. For larger channels this adds latency; a documents-already-embedded check (skip re-embedding if the collection has entries) is a planned optimization.
- **Chroma persistence in production** — `./chroma_db` is a local folder. On platforms without persistent disk (e.g. ephemeral containers), this data will not survive a restart. Needs a Railway volume (or equivalent) mounted at `./chroma_db`, or a move to a hosted vector DB.
- **LLM response length** — `max_new_tokens=200` on the HuggingFace endpoint may truncate longer answers (e.g. detailed trend breakdowns); increase if responses are getting cut short.

## Deployment

Deployed on **Railway**, ideally in the same project as the Node.js backend to use Railway's private networking (so this service does not need to be publicly reachable). Start command:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Set all environment variables above in the Railway service settings.
