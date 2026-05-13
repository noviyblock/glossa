.PHONY: help install dev build up down logs test lint format typecheck check clean \
        dvc-pull mlflow-ui docs shell-gateway shell-cv shell-asr shell-nlp \
        shell-tts shell-max gpu-up migrate-qdrant prod-up prod-down deploy \
        backup restore health security-scan

SHELL := /bin/bash
COMPOSE      := docker compose
COMPOSE_PROD := docker compose -f docker-compose.yml -f docker-compose.prod.yml
COMPOSE_GPU  := docker compose -f docker-compose.yml -f docker-compose.gpu.yml

## ── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

## ── Local Dev ────────────────────────────────────────────────────────────────
install: ## Install dev dependencies + pre-commit hooks
	pip install --upgrade pip
	pip install pre-commit ruff black mypy bandit[toml] pytest pytest-asyncio pytest-cov
	pip install -e libs/common -e libs/max_sdk
	pre-commit install --install-hooks
	@if [[ ! -f .secrets.baseline ]]; then \
		detect-secrets scan > .secrets.baseline; \
		echo "Created .secrets.baseline"; \
	fi

dev: ## Start all services in development mode (hot-reload)
	$(COMPOSE) -f docker-compose.yml -f docker-compose.override.yml up

up: ## Start all services (dev mode)
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

gpu-up: ## Start all services with GPU support
	$(COMPOSE_GPU) up -d

restart: ## Restart a specific service: make restart svc=cv-service
	$(COMPOSE) restart $(svc)

logs: ## Follow logs for all services
	$(COMPOSE) logs -f --tail=100

logs-%: ## Follow logs for a specific service: make logs-api-gateway
	$(COMPOSE) logs -f --tail=100 $*

## ── Production ───────────────────────────────────────────────────────────────
prod-up: ## Start all services in production mode with resource limits
	$(COMPOSE_PROD) up -d

prod-down: ## Stop production stack
	$(COMPOSE_PROD) down

deploy: ## Rolling production deployment (runs scripts/deploy.sh)
	@bash scripts/deploy.sh

## ── Backup & Restore ─────────────────────────────────────────────────────────
backup: ## Backup Redis, Qdrant, MLflow, Grafana (runs scripts/backup.sh)
	@bash scripts/backup.sh

restore: ## Restore from backup: make restore path=/opt/glossa/backups/20240101T120000Z
	@[[ -n "$(path)" ]] || (echo "Usage: make restore path=<backup-dir>"; exit 1)
	@bash scripts/restore.sh "$(path)"

health: ## Run comprehensive health check against all services
	@bash scripts/health-check.sh

## ── Build ────────────────────────────────────────────────────────────────────
build: ## Build all Docker images
	$(COMPOSE) build --parallel

build-%: ## Build a specific service image: make build-cv-service
	$(COMPOSE) build $*

push: ## Push images to registry
	$(COMPOSE) push

## ── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check services/ libs/ --fix

format: ## Run black + ruff formatter
	black services/ libs/
	ruff format services/ libs/

typecheck: ## Run mypy type checker
	mypy libs/ services/ --config-file=pyproject.toml

security-scan: ## Run bandit static security analysis
	bandit -r services/ libs/ -c pyproject.toml -x "*/tests/*" -ll

check: lint format typecheck security-scan ## Run all code quality checks

## ── Testing ──────────────────────────────────────────────────────────────────
test: ## Run all tests
	pytest services/ libs/ --cov=services --cov=libs --cov-report=term-missing

test-%: ## Run tests for a specific service: make test-max-adapter
	pytest services/$*/tests/ -v

test-watch: ## Run tests in watch mode
	pytest services/ libs/ -f

## ── MLflow & DVC ─────────────────────────────────────────────────────────────
mlflow-ui: ## Open MLflow UI
	$(COMPOSE) up -d mlflow
	@echo "MLflow UI: http://localhost:5000"

dvc-pull: ## Pull model artifacts from DVC remote
	dvc pull --run-cache

dvc-push: ## Push model artifacts to DVC remote
	dvc push

## ── Infrastructure ───────────────────────────────────────────────────────────
migrate-qdrant: ## Initialize Qdrant collections
	python scripts/init_qdrant.py

redis-cli: ## Open Redis CLI
	$(COMPOSE) exec redis redis-cli

## ── Shell Access ─────────────────────────────────────────────────────────────
shell-gateway: ## Shell into api-gateway container
	$(COMPOSE) exec api-gateway /bin/bash

shell-cv: ## Shell into cv-service container
	$(COMPOSE) exec cv-service /bin/bash

shell-asr: ## Shell into asr-service container
	$(COMPOSE) exec asr-service /bin/bash

shell-nlp: ## Shell into nlp-service container
	$(COMPOSE) exec nlp-service /bin/bash

shell-tts: ## Shell into tts-service container
	$(COMPOSE) exec tts-service /bin/bash

shell-max: ## Shell into max-adapter container
	$(COMPOSE) exec max-adapter /bin/bash

## ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

clean-docker: ## Remove all Docker containers, volumes, images for this project
	$(COMPOSE) down -v --rmi local
