.PHONY: help install db-up db-down api docker-up docker-down

# Load variables from .env (POSTGRES_*, API_PORT, GEMINI_API_KEY, ...)
ifneq (,$(wildcard .env))
include .env
export
endif

help:
	@echo "Available targets:"
	@echo "  install     Install Python dependencies with uv"
	@echo "  db-up       Start the local Postgres (pgvector) container"
	@echo "  db-down     Stop the local Postgres container"
	@echo "  api         Run the API locally with uv + uvicorn (auto-reload)"
	@echo "  docker-up   Build and start the full stack (postgres + api) in Docker"
	@echo "  docker-down Stop and remove the full stack"

install:
	uv sync

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

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down
