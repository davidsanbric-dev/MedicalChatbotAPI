# Medical Conditions API

A **FastAPI-based REST API** that powers a RAG (Retrieval-Augmented Generation) chatbot for medical conditions. This API provides intelligent, conversational responses about medical conditions using vector similarity search, cross-encoder re-ranking, and LLM generation. The LLM layer uses **OpenRouter** as the default inference provider with **Google Gemini** as an automatic fallback.

## 🏗️ Architecture Overview

The API is the **serving layer** of the 3-stage ETL pipeline:

```
PostgreSQL Database (with pgvector)
    ↓
FastAPI Service (this component)
    ├─ Vector similarity search
    ├─ Cross-encoder re-ranking
    ├─ LLM generation (OpenRouter → Gemini fallback)
    └─ Conversational memory
    ↓
Streamlit Frontend
```

## 🚀 Features

### RAG Pipeline with Re-Ranking
- **Conversational Context**: Rewrites queries based on conversation history
- **Vector Similarity Search**: Retrieves relevant chunks using pgvector cosine similarity
- **Cross-Encoder Re-Ranking**: Scores and re-ranks results with `ms-marco-MiniLM-L-6-v2`
- **Relevance Filtering**: Filters chunks below relevance threshold (-2.0)
- **LLM Generation**: Generates responses via OpenRouter (default), automatically falling back to Google Gemini 2.5 Flash if the OpenRouter request fails

### Conversational Memory
- **Session Management**: PostgreSQL-backed chat history per session
- **Context Awareness**: Maintains conversation context across multiple turns
- **Message Persistence**: Stores and retrieves conversation history

### Dataset Insights
- **Condition Names**: Lists all available medical conditions from database
- **Top Symptoms**: Analyzes most common symptoms across conditions
- **Statistics**: Provides insights into the knowledge base

### Health Checks
- **Container Orchestration**: `/health` endpoint for Docker health checks
- **Service Monitoring**: Quick status verification

## 📦 Installation

### Prerequisites
- Python 3.12+
- Docker and Docker Compose (recommended)
- PostgreSQL with pgvector extension
- An OpenRouter API key (default provider) and/or a Google Gemini API key (fallback)

### Environment Variables

Create a `.env` file (or set in docker-compose.yml):

```env
DATABASE_URL=postgresql://postgres:password@postgres:5432/medical_conditions_db

# Inference providers — OpenRouter is the default/primary provider; Gemini is
# the automatic fallback used when an OpenRouter request fails. If
# OPENROUTER_API_KEY is unset, Gemini is used as the sole provider.
OPENROUTER_API_KEY=your_openrouter_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional overrides (defaults shown):
# OPENROUTER_MODEL=openai/gpt-oss-120b:free
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# GEMINI_MODEL=gemini-2.5-flash

# Rate limiting on /chat/ and /chat/stream (defaults shown; 0 disables a limit).
# See "Rate Limiting" below.
# CHAT_RATE_LIMIT_PER_MINUTE=6
# CHAT_RATE_LIMIT_PER_DAY=60
# CHAT_GLOBAL_LIMIT_PER_DAY=500
# TRUSTED_PROXY_HOPS=0
```

> **Note:** The default `OPENROUTER_MODEL` (`openai/gpt-oss-120b:free`) works on a
> free OpenRouter account. To use a paid model (e.g. `google/gemini-2.5-flash`
> routed through OpenRouter), add credits at openrouter.ai/credits and set
> `OPENROUTER_MODEL` accordingly.

### Docker Setup (Recommended)

1. **Build the API container**:
   ```bash
   docker compose build api
   ```

2. **Start the API service**:
   ```bash
   docker compose up api
   ```

3. **Verify health**:
   ```bash
   curl http://localhost:8000/health
   ```

### Local Development Setup

1. **Navigate to API directory**:
   ```bash
   cd medical_conditions_api
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   export DATABASE_URL="postgresql://postgres:password@localhost:5434/medical_conditions_db"
   export GEMINI_API_KEY="your_api_key"
   ```

4. **Run the API**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🔌 API Endpoints

### Health Check

#### `GET /health`
Check API service health status.

**Response:**
```json
{
  "status": "healthy"
}
```

### Conversational Chat

#### `POST /chat/`
Generate conversational responses using RAG pipeline with re-ranking.

**Request Body:**
```json
{
  "query": "What are the symptoms of acne?",
  "session_id": "uuid-session-id",
  "top_k": 10
}
```

**Response:**
```json
{
  "response": "Acne symptoms typically include...",
  "session_id": "uuid-session-id"
}
```

**Pipeline Flow:**
1. Retrieves conversation history
2. Rewrites query based on context
3. Performs vector similarity search (top 10 chunks)
4. Re-ranks with cross-encoder
5. Filters by relevance threshold
6. Generates response with Gemini LLM
7. Saves conversation to history

#### Rate Limiting

Each chat turn costs two LLM calls (rephrase + answer), so `/chat/` and
`/chat/stream` are rate limited. `/health` and `/insights/*` are not. There is
no authentication yet.

| Limit | Default | Env var |
|---|---|---|
| Per client, per minute | 6 | `CHAT_RATE_LIMIT_PER_MINUTE` |
| Per client, per day | 60 | `CHAT_RATE_LIMIT_PER_DAY` |
| All clients together, per day | 500 | `CHAT_GLOBAL_LIMIT_PER_DAY` |

The windows slide, and a rejected request uses no quota. Over a limit, the API
answers `429 Too Many Requests` with a `Retry-After` header (seconds) and a
human-readable `detail`. Counters live in process memory: they reset on restart,
and running several replicas would need a shared store such as Redis.

**Identifying clients behind proxies.** By default (`TRUSTED_PROXY_HOPS=0`) a
client is the direct peer address. Behind reverse proxies that is the proxy, so
every visitor would share one bucket. Set `TRUSTED_PROXY_HOPS` to the number of
proxies in front of the API that append to `X-Forwarded-For`. The client is then
read that many entries from the right, so entries a caller forges on the left
are ignored. For the Dokploy deployment, where the API is reached by the frontend's
nginx over the internal Docker network (browser → Traefik → nginx → API), that
is `2`. This is only sound if the API can't be reached by a shorter path: if it
also has a public domain, a caller can pad the header. The global daily cap
bounds cost either way.

### Dataset Insights

#### `GET /insights/condition-names`
Get all available medical condition names.

**Response:**
```json
[
  "Acne",
  "Allergies",
  "Asthma",
  "Diabetes",
  ...
]
```

#### `GET /insights/top-symptoms?top_n=10`
Get most common symptoms across conditions.

**Query Parameters:**
- `top_n` (optional): Number of top symptoms to return (default: 10)

**Response:**
```json
[
  ["Fever", 45],
  ["Headache", 38],
  ["Fatigue", 35],
  ...
]
```

## 🏛️ Project Structure

```
medical_conditions_api/
├── app/
│   ├── main.py                      # FastAPI application entry point
│   ├── schemas.py                   # Pydantic request/response models
│   ├── core/                        # Core functionality
│   │   ├── config.py                # Environment configuration
│   │   ├── embedding.py             # Embedding generation utilities
│   │   ├── exceptions.py            # Custom exception classes
│   │   └── memory.py                # Conversational memory implementation
│   ├── routes/                      # API route handlers
│   │   ├── rag_re_ranker.py        # RAG chatbot endpoint
│   │   └── dataset_insights.py     # Dataset statistics endpoints
│   └── services/                    # Business logic layer
│       ├── svc_document.py         # Document retrieval service
│       ├── svc_chat_history.py     # Chat history management
│       └── svc_processed_data.py   # Processed data queries
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file

# Shared database configuration (mounted from medical_conditions_db)
/app/database/                       # Volume-mounted shared database module
├── database.py                      # Database connection and SessionLocal
└── models.py                        # SQLAlchemy models
```

## 🔧 Configuration

### Embedding Model

```python
# Default model for query embeddings
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### Cross-Encoder Re-Ranking

```python
# Cross-encoder model for re-ranking
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RELEVANCE_THRESHOLD = -2.0  # Filter chunks below this score
```

### LLM Configuration

The LLM is configured in [`app/core/llm.py`](app/core/llm.py). OpenRouter
(OpenAI-compatible) is the primary provider; Gemini is wired in as a fallback via
LangChain's `with_fallbacks`, so if an OpenRouter call fails (auth, rate limit,
402, outage), the same request is transparently retried against Gemini. If
`OPENROUTER_API_KEY` is unset, the API degrades cleanly to Gemini-only.

```python
# Primary provider (OpenRouter, OpenAI-compatible)
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"   # env: OPENROUTER_MODEL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"  # env: OPENROUTER_BASE_URL

# Fallback provider (Google Gemini)
GEMINI_MODEL = "gemini-2.5-flash"               # env: GEMINI_MODEL
```

### Database Configuration

Configured via `DATABASE_URL` environment variable. Uses shared database models from `medical_conditions_db/`:

```yaml
# From docker-compose.yml
volumes:
  - ./medical_conditions_db:/app/database
```

## 🔄 Integration with System

### Service Dependencies

```yaml
# From docker-compose.yml
api:
  depends_on:
    postgres:
      condition: service_healthy
    db:
      condition: service_completed_successfully
```

### Data Flow

1. **Worker** populates `documents` table with embeddings
2. **API** queries `documents` for vector similarity
3. **API** re-ranks results with cross-encoder
4. **API** generates responses with Gemini LLM
5. **Frontend** displays responses to users

### Shared Components

- **Database Models**: Via volume mount from `medical_conditions_db/`
- **SessionLocal**: Database session management
- **Models**: Document, ProcessedData, ChatHistory

## 📊 RAG Pipeline Details

### 1. Query Rewriting
```
User Query + Conversation History
    ↓
Gemini LLM (conversational awareness)
    ↓
Standalone Query (context-independent)
```

### 2. Vector Retrieval
```
Standalone Query
    ↓
Embedding Model (all-MiniLM-L6-v2)
    ↓
Query Embedding (384 dimensions)
    ↓
pgvector Cosine Similarity Search
    ↓
Top K Chunks (default: 10)
```

### 3. Cross-Encoder Re-Ranking
```
Retrieved Chunks + Query
    ↓
Cross-Encoder (ms-marco-MiniLM-L-6-v2)
    ↓
Relevance Scores
    ↓
Sort by Relevance (descending)
    ↓
Filter by Threshold (> -2.0)
```

### 4. Response Generation
```
Relevant Chunks + Conversation History + User Query
    ↓
OpenRouter LLM (default) ──fails──▶ Gemini LLM (fallback)
    ↓
Conversational Response
```

## 🚨 Error Handling

### Common Issues

#### Database Connection Errors
```bash
# Problem: Cannot connect to PostgreSQL
# Solution: Ensure postgres service is running
docker compose up -d postgres
docker compose logs postgres

# Verify DATABASE_URL
echo $DATABASE_URL
```

#### Missing / Invalid Provider API Keys
```bash
# Problem: LLM generation fails
# Solution: Set the provider keys. OpenRouter is primary, Gemini is the fallback.
export OPENROUTER_API_KEY="your_openrouter_key"
export GEMINI_API_KEY="your_gemini_key"

# Or in docker-compose.yml
environment:
  OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
  GEMINI_API_KEY: ${GEMINI_API_KEY}
```

> If OpenRouter requests fail (e.g. a 402 "insufficient credits" on a paid model,
> or a transient 429 rate limit), the API automatically falls back to Gemini —
> so a valid `GEMINI_API_KEY` keeps the service working even when OpenRouter is
> unavailable.

#### No Documents Found
```bash
# Problem: Vector search returns no results
# Solution: Ensure worker has populated documents table
docker compose run --rm worker python main.py

# Verify documents table
docker compose exec postgres psql -U postgres -d medical_conditions_db \
  -c "SELECT COUNT(*) FROM documents;"
```

#### CORS Issues (Development)
```python
# Add CORS middleware in main.py if needed
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📈 Performance Considerations

### Response Times
- **Vector Search**: ~50-100ms. At the current size (hundreds of chunks) Postgres scans every row, which is exact and cheaper than any index. The `documents_embedding_idx` HNSW index (cosine) takes over automatically as the table grows, and it needs no rebuild after loading, unlike ivfflat. Startup replaces the ivfflat index that older databases carry.
- **Cross-Encoder**: ~100-200ms (depends on chunk count)
- **LLM Generation**: ~1-3s (Gemini API latency)
- **Total**: ~1.5-3.5s per request

### Optimization Strategies

1. **Reduce top_k**: Lower initial retrieval count (e.g., 5 instead of 10)
2. **Batch Processing**: Process multiple queries in parallel if needed
3. **Caching**: Cache common queries/responses (not implemented)
4. **Connection Pooling**: SQLAlchemy handles connection pooling automatically

### Scaling

- **Horizontal**: Run multiple API instances behind load balancer
- **Vertical**: Increase CPU/memory allocation in docker-compose.yml
- **Database**: Optimize pgvector index parameters (lists parameter)
- **LLM**: Consider switching to faster models or local deployment

## 🧪 Testing

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# List condition names
curl http://localhost:8000/insights/condition-names

# Get top symptoms
curl http://localhost:8000/insights/top-symptoms?top_n=5

# RAG query (POST with JSON)
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of acne?",
    "session_id": "test-session-123",
    "top_k": 5
  }'
```

### Integration Testing

```bash
# Start full stack
docker compose up -d

# Wait for services to be healthy
docker compose ps

# Test API health
curl http://localhost:8000/health

# Test frontend connection
curl http://localhost:8503
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

### Development Workflow

1. Make changes to code
2. Test locally with `--reload` flag
3. Update schemas if API contracts change
4. Test with curl or interactive docs
5. Build and test in Docker
6. Update this README if needed

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and returns
- Document functions with docstrings
- Implement proper error handling with HTTPException
- Use dependency injection where appropriate

### Architecture Patterns

- **Service Layer**: Business logic in `services/`
- **Route Handlers**: Thin controllers in `routes/`
- **Core Utilities**: Shared functionality in `core/`
- **Schemas**: Pydantic models for validation

## 📝 Dependencies

### Core Framework
- **FastAPI** (0.116.2): Modern, fast web framework
- **Uvicorn** (0.32.0): ASGI server
- **Pydantic** (2.11.9): Data validation

### Machine Learning
- **sentence-transformers** (5.1.0): Embedding generation
- **transformers** (4.56.2): HuggingFace models
- **torch** (2.8.0): CPU-only PyTorch

### LLM Integration
- **langchain** (0.3.27): LLM orchestration
- **langchain-openai** (0.3.35): OpenRouter (OpenAI-compatible) integration — primary provider
- **langchain-google-genai** (2.0.4): Gemini integration — fallback provider
- **langchain-huggingface** (0.3.1): HuggingFace embeddings

### Database
- **SQLAlchemy** (2.0.43): ORM
- **psycopg2-binary** (2.9.9): PostgreSQL adapter
- **pgvector** (0.2.5): Vector similarity search

## 🆘 Support

### Debug Commands

```bash
# View API logs
docker compose logs api -f

# Check API container status
docker compose ps api

# Access API container
docker compose exec api bash

# Test database connection from API
docker compose exec api python -c "
import sys
sys.path.insert(0, '/app/database')
from database import SessionLocal
db = SessionLocal()
print('Database connected!')
db.close()
"
```

### Common Troubleshooting

1. **Check service health**: `curl http://localhost:8000/health`
2. **Verify environment variables**: `docker compose config`
3. **Check database connection**: Review API logs for connection errors
4. **Test embeddings**: Ensure worker has generated embeddings
5. **Verify provider APIs**: Check OpenRouter and Gemini keys are valid and have quota/credits

---

**Note**: This API provides the intelligent serving layer for medical condition queries, combining vector search, re-ranking, and LLM generation to deliver accurate, conversational responses based on the processed knowledge base.
