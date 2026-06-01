# SmartPub — Biomedical Literature QA Chatbot

SmartPub is a Retrieval-Augmented Generation (RAG) chatbot for querying biomedical literature. It retrieves relevant PubMed abstracts from a Pinecone vector database and feeds them to a Llama 2 language model to answer natural-language questions.

![CI](https://github.com/jasmine95dn/SmartPub-INLPT-WS2023/actions/workflows/ci.yml/badge.svg?branch=re-work)

---

## Architecture

```
User question
     |
     v
SentenceTransformer (all-MiniLM-L6-v2)
     |  embed question
     v
Pinecone vector index  -->  top-k relevant documents (KG triplets or raw text)
     |
     v
Llama 2 (13B-chat, 4-bit quantised via bitsandbytes)
     |  RAG prompt
     v
Answer
```

The pipeline is implemented with **LangChain 1.x LCEL** (`create_retrieval_chain` + `create_stuff_documents_chain`).

---

## Project Structure

```
SmartPub-INLPT-WS2023/
├── .github/workflows/
│   ├── ci.yml            # Lint (ruff) + unit/e2e tests on every push
│   └── cd.yml            # Docker image pushed to ghcr.io on merge to main
├── .pre-commit-config.yaml
├── docker-compose.yml
└── smartpub_app/
    ├── Dockerfile
    ├── pyproject.toml    # Poetry dependencies (Python 3.11+)
    ├── app.py            # Flask web server (port 8080)
    ├── store_index.py    # One-shot Pinecone indexing script
    ├── model/
    │   ├── model.py                         # Top-level pipeline entry point
    │   ├── db_retriever.py                  # DBRetriever: Pinecone + LLM RAG chain
    │   ├── db_loader.py                     # PineconeVDB / PineconeVDBRawText loaders
    │   ├── qa_inference.py                  # QA: Llama 2 pipeline wrapper
    │   └── RelevantInformationExtractor.py  # Standalone similarity search util
    ├── src/
    │   └── prompt.py     # RAG prompt template
    └── tests/
        ├── unit/         # Fast tests with stubbed ML dependencies
        ├── e2e/          # Pipeline integration tests (mocked LLM/Pinecone)
        └── system/       # Full HTTP tests against Flask test client
```

---

## Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/docs/#installation) **or** Docker + Docker Compose
- A [Pinecone](https://www.pinecone.io/) account with an index named `smartpub`
- A [Hugging Face](https://huggingface.co/) account with access to `meta-llama/Llama-2-13b-chat-hf`

---

## Environment Variables

Create a `.env` file in `smartpub_app/` (or export these in your shell):

```env
PIPELINE_API_KEY=<your-pinecone-api-key>
HF_AUTH=<your-huggingface-token>
```

---

## Setup & Running

### Option A — Docker Compose (recommended)

```bash
# Build and start the chat server
docker compose up --build web

# Run the one-shot indexer (only needed once, or after new data)
docker compose --profile tools up indexer
```

The chat UI is served at `http://localhost:8080`.

### Option B — Poetry (local)

```bash
cd smartpub_app

# Install dependencies (CPU-only torch by default)
poetry install --without dev

# Start the Flask server
poetry run python app.py
```

### Index new documents

```bash
cd smartpub_app
poetry run python store_index.py <path-to-data-dir>
```

---

## Development

```bash
cd smartpub_app
poetry install --with dev

# Linting (ruff)
pre-commit run --all-files

# Tests (unit + e2e, no heavy ML deps required)
poetry run pytest -m "not system and not slow" -v

# All tests including system (requires running Flask server)
poetry run pytest -v
```

> **Note:** Unit and e2e tests stub all ML dependencies (`torch`, `transformers`,
> `langchain`, `pinecone`) so they run without a GPU or network access.

---

## CI/CD

| Workflow | Trigger | Steps |
|----------|---------|-------|
| `ci.yml` | Push / PR to `main`, push to `re-work` | ruff lint, unit + e2e tests |
| `cd.yml` | Push to `main` | Build & push Docker image to `ghcr.io` |

---

## Security — Dependabot Alerts

All 5 alerts present on `main` are **resolved in the `re-work` branch** and will be
closed once that branch is merged:

| # | Package | Severity | Fixed version in `re-work` |
|---|---------|----------|---------------------------|
| 5 | langchain | High | `^1.0` (>= 1.0.0) |
| 4 | langchain-openai | Low | `1.1.14` |
| 3 | langchain | Low | `^1.0` (>= 1.0.0) |
| 2 | langchain | Medium | `^1.0` (>= 1.0.0) |
| 1 | pydantic | Medium | `>=2.7.4` (pydantic v2) |

---

## Acknowledgements

Developed as part of the **Information Retrieval and Natural Language Processing
for Text (INLPT) WS 2023** course at Saarland University.
