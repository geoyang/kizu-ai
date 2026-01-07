# Kizu AI - Makefile

.PHONY: help setup install dev docker-up docker-down test lint clean download-models

help:
	@echo "Kizu AI - Available Commands"
	@echo ""
	@echo "  setup          - Full setup (venv, deps, models)"
	@echo "  install        - Install dependencies only"
	@echo "  dev            - Run development server"
	@echo "  docker-up      - Start Docker containers"
	@echo "  docker-down    - Stop Docker containers"
	@echo "  download-models - Download AI models"
	@echo "  test           - Run tests"
	@echo "  lint           - Run linters"
	@echo "  clean          - Clean cache files"
	@echo ""

setup:
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

install:
	pip install -r api/requirements.txt

dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

download-models:
	python scripts/download_models.py --all

download-llava:
	python scripts/download_models.py --llava

test:
	pytest tests/ -v

lint:
	ruff check api/
	mypy api/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

# Database
migrate:
	@echo "Copy migrations/001_ai_tables.sql to Supabase SQL editor"
	@echo "File: migrations/001_ai_tables.sql"

# Worker
worker:
	celery -A api.workers.celery_app worker --loglevel=info

# Redis (local)
redis:
	docker run -d -p 6379:6379 --name kizu-redis redis:7-alpine
