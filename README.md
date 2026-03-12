# Multimodal RAG API

Production-ready Multimodal Retrieval-Augmented Generation API built with FastAPI, Docling, LangChain, FAISS, and IBM Granite models.

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd multimodal-rag-production

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Set up environment
cp .env.example .env
# Edit .env — set your OpenRouter API key and MOCK_MODE=true for testing

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Method | Path      | Description                              |
|--------|-----------|------------------------------------------|
| GET    | /health   | Liveness check + index status            |
| POST   | /ingest   | Upload and ingest a PDF into vector store|
| POST   | /query    | Ask a question against ingested documents|
| GET    | /docs     | Swagger UI (auto-generated)              |

## Architecture

```
PDF → Docling Parser → [Text | Table | Image] chunks
                            ↓
                  Granite Embedding 30M (local)
                            ↓
                      FAISS Index
                            ↓
                  Query → Retriever (top-k=4)
                            ↓
                  Granite 8B Instruct (via OpenRouter) + Grounding Prompt
                            ↓
                      Answer + Sources
```

## Running Tests

```bash
pytest tests/ -v
```

## Environment Variables

| Variable                | Default                                    | Description                    |
|------------------------|--------------------------------------------|--------------------------------|
| OPENROUTER_API_KEY     | (required for real mode)                   | OpenRouter API key             |
| MOCK_MODE              | false                                      | Enable mock mode for demos     |
| EMBEDDING_MODEL_NAME   | ibm-granite/granite-embedding-30m-english  | HuggingFace embedding model    |
| GRANITE_INSTRUCT_MODEL | ibm-granite/granite-3.1-8b-instruct       | Text generation model          |
| GRANITE_VISION_MODEL   | ibm-granite/granite-vision-3.1-2b         | Vision model for image summary |
| FAISS_INDEX_PATH       | faiss_index                                | FAISS index directory          |
| RETRIEVER_TOP_K        | 4                                          | Number of chunks to retrieve   |
| MAX_UPLOAD_SIZE_MB     | 50                                         | Max PDF upload size            |

## Tech Stack

- **FastAPI** — async web framework
- **Docling** — PDF parsing (text, tables, images)
- **FAISS** — vector similarity search
- **IBM Granite** — embedding (30M) and generation (8B Instruct) models
- **LangChain** — RAG chain orchestration
- **OpenRouter** — LLM API gateway
- **Pydantic** — data validation and settings
