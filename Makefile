.PHONY: help install test db-up db-down api eval-retrieval eval-agent docker-up docker-down

# Load variables from .env (POSTGRES_*, API_PORT, GEMINI_API_KEY, ...)
ifneq (,$(wildcard .env))
include .env
export
endif

help:
	@echo "Available targets:"
	@echo "  install     Install Python dependencies with uv"
	@echo "  test        Run the unit tests (no database or API keys needed)"
	@echo "  db-up       Start the local Postgres (pgvector) container"
	@echo "  db-down     Stop the local Postgres container"
	@echo "  api         Run the API locally with uv + uvicorn (auto-reload)"
	@echo "  eval-retrieval  Score retrieval on evals/retrieval_cases.jsonl (local DB, no LLM)"
	@echo "  eval-agent  Check the agent's tool choices on evals/agent_cases.jsonl (local DB + LLM)"
	@echo "  docker-up   Build and start the full stack (postgres + api) in Docker"
	@echo "  docker-down Stop and remove the full stack"

install:
	uv sync

test:
	uv run pytest

db-up:
	docker compose up -d postgres
	@echo "Waiting for Postgres to be ready..."
	@until docker compose exec -T postgres pg_isready -U $(POSTGRES_USER) >/dev/null 2>&1; do sleep 1; done

db-down:
	docker compose stop postgres

# Run the API locally (outside Docker), against the Postgres container
# exposed on localhost:$(POSTGRES_PORT).
api: db-up
	DATABASE_URL=postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/$(POSTGRES_DB) \
		uv run uvicorn app.main:app --host 0.0.0.0 --port $(API_PORT) --reload

# Evaluations run against the Postgres container on localhost, like `make api`.
LOCAL_DATABASE_URL = postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/$(POSTGRES_DB)

eval-retrieval: db-up
	@DATABASE_URL=$(LOCAL_DATABASE_URL) uv run python -m evals.retrieval

eval-agent: db-up
	@DATABASE_URL=$(LOCAL_DATABASE_URL) uv run python -m evals.agent

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
