.PHONY: install install-dev test test-unit test-integration test-e2e test-all lint format typecheck docker docker-down clean help

# Default Python version
PYTHON := python3.11

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PYTHON) -m pip install --upgrade pip
	pip install -e .

install-dev: ## Install development dependencies and pre-commit hooks
	$(PYTHON) -m pip install --upgrade pip
	pip install -e .[dev]
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: test-unit ## Run unit tests (alias)

test-unit: ## Run unit tests
	pytest tests/unit -v

test-integration: ## Run integration tests (requires Qdrant)
	pytest tests/integration -v -m integration

test-e2e: ## Run end-to-end tests
	pytest tests/e2e -v -m e2e

test-all: ## Run all tests with coverage
	pytest tests -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated: htmlcov/index.html$(NC)"

test-fast: ## Run tests without slow markers
	pytest tests -v -m "not slow"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run all linters (ruff, black, mypy)
	@echo "$(YELLOW)Running Ruff...$(NC)"
	ruff check src tests
	@echo "$(YELLOW)Running Black...$(NC)"
	black --check src tests
	@echo "$(YELLOW)Running MyPy...$(NC)"
	mypy src --ignore-missing-imports
	@echo "$(GREEN)All checks passed!$(NC)"

format: ## Format code with ruff and black
	ruff check --fix src tests
	black src tests
	@echo "$(GREEN)Code formatted!$(NC)"

typecheck: ## Run type checking with mypy
	mypy src --ignore-missing-imports

# =============================================================================
# Docker
# =============================================================================

docker: ## Start Docker services (Qdrant)
	docker compose up -d
	@echo "$(YELLOW)Waiting for Qdrant to be ready...$(NC)"
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -s http://localhost:6333/health > /dev/null && break || sleep 2; \
	done
	@echo "$(GREEN)Docker services ready!$(NC)"

docker-down: ## Stop Docker services
	docker compose down

docker-build: ## Build the application Docker image
	docker build -t gri-rag-api:latest .

docker-logs: ## Show Docker logs
	docker compose logs -f

# =============================================================================
# Application
# =============================================================================

run: ## Run the API server
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ingest: ## Run the ingestion pipeline
	gri-ingest --input data/raw/gri.docx --output data/processed/

eval: ## Run the evaluation pipeline
	gri-eval --dataset data/golden_dataset.json --output reports/

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleaned!$(NC)"

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

check: lint test-unit ## Run lint and unit tests (quick check)
	@echo "$(GREEN)Quick check passed!$(NC)"
