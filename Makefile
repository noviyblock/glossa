.PHONY: help dev up down build logs test lint format check models exp-dry \
        prod-up prod-down mlflow-ui dvc-pull dvc-push health clean clean-docker \
        shell-gateway shell-cv shell-asr shell-nlp shell-tts redis-cli

SHELL        := /bin/bash
COMPOSE      := docker compose
COMPOSE_PROD := docker compose -f docker-compose.yml -f docker-compose.prod.yml

## ── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

## ── Local Dev ────────────────────────────────────────────────────────────────
dev: ## Start all services with hot-reload (docker-compose.override.yml applied automatically)
	$(COMPOSE) up

up: ## Start all services in detached mode
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

restart: ## Restart a single service: make restart svc=cv-service
	$(COMPOSE) restart $(svc)

logs: ## Follow logs for all services
	$(COMPOSE) logs -f --tail=100

logs-%: ## Follow logs for one service: make logs-nlp-service
	$(COMPOSE) logs -f --tail=100 $*

## ── Production ───────────────────────────────────────────────────────────────
prod-up: ## Start with production resource limits
	$(COMPOSE_PROD) up -d

prod-down: ## Stop production stack
	$(COMPOSE_PROD) down

## ── Build ────────────────────────────────────────────────────────────────────
build: ## Build all Docker images in parallel
	$(COMPOSE) build --parallel

build-%: ## Build a single service image: make build-cv-service
	$(COMPOSE) build $*

## ── Models ───────────────────────────────────────────────────────────────────
models: ## Download all model files from Google Drive (requires gdown)
	@pip install -q gdown
	python scripts/download_models.py

## ── Experiments ──────────────────────────────────────────────────────────────
exp-dry: ## Run all experiments in dry-run mode (no GPU/models required)
	bash experiments/run_all.sh --dry-run

## ── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Run ruff linter with auto-fix
	ruff check services/ experiments/ mlops/ --fix

format: ## Run black formatter
	black services/ experiments/ mlops/

check: lint format ## Run lint + format

## ── Testing ──────────────────────────────────────────────────────────────────
# Top-level tests/ referenced here never existed in this repo; services/*/tests
# (the only test suites that did) were archived to
# archive/legacy_tests_pre_flatfile_refactor/ -- they import from a deleted
# services/*/app/ package and don't run against current code. Real tests
# against current flat-file modules are only starting to exist (see
# services/api_gateway/tests/) -- most services still have none.
test: ## Run pytest test suite
	pytest services/ -v --tb=short

## ── MLflow & DVC ─────────────────────────────────────────────────────────────
mlflow-ui: ## Start MLflow and open UI at http://localhost:5000
	$(COMPOSE) up -d mlflow
	@echo "MLflow UI → http://localhost:5000"

dvc-pull: ## Pull model artifacts from DVC remote (DagsHub)
	dvc pull --run-cache

dvc-push: ## Push model artifacts to DVC remote
	dvc push

## ── Health ───────────────────────────────────────────────────────────────────
health: ## Check health endpoints for all running services
	@for svc in "api-gateway:8000" "cv-service:8001" "asr-service:8002" "nlp-service:8003" "tts-service:8004"; do \
		name=$${svc%%:*}; port=$${svc##*:}; \
		status=$$(curl -sf http://localhost:$$port/health/live -o /dev/null -w "%{http_code}" 2>/dev/null || echo "ERR"); \
		printf "%-20s %s\n" "$$name" "$$status"; \
	done

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

redis-cli: ## Open Redis CLI
	$(COMPOSE) exec redis redis-cli

## ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove Python bytecode and cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-docker: ## Remove all project containers and volumes
	$(COMPOSE) down -v --rmi local
