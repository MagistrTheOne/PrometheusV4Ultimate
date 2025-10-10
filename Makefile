# PrometheusULTIMATE v4 Makefile

.PHONY: help up down test e2e fmt clean install dev

help: ## Show this help message
	@echo "PrometheusULTIMATE v4 - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies in virtual environment
	python -m venv .venv
	.venv\Scripts\activate && python -m pip install --upgrade pip wheel
	.venv\Scripts\activate && pip install -r requirements.txt

dev: ## Start development environment
	.venv\Scripts\activate && python -m uvicorn apps.gateway.main:app --host 0.0.0.0 --port 8090 --reload

up: ## Start all services with docker-compose
	docker-compose -f infra/docker-compose.yml up -d --build

down: ## Stop all services
	docker-compose -f infra/docker-compose.yml down

logs: ## Show logs from all services
	docker-compose -f infra/docker-compose.yml logs -f

test: ## Run unit tests
	.venv\Scripts\activate && pytest tests/unit -v

e2e: ## Run end-to-end tests
	.venv\Scripts\activate && pytest tests/e2e -v

fmt: ## Format code with black and ruff
	.venv\Scripts\activate && ruff check --fix .
	.venv\Scripts\activate && black .

lint: ## Run linting
	.venv\Scripts\activate && ruff check .
	.venv\Scripts\activate && mypy libs/ apps/

clean: ## Clean up containers and volumes
	docker-compose -f infra/docker-compose.yml down -v
	docker system prune -f

health: ## Check health of all services
	@echo "Checking service health..."
	@curl -f http://localhost:8090/health || echo "Gateway: DOWN"
	@curl -f http://localhost:6333/health || echo "Qdrant: DOWN"

models: ## Show available models
	curl -s http://localhost:8090/models | python -m json.tool

task: ## Create a test task (usage: make task GOAL="your goal")
	curl -X POST http://localhost:8090/task \
		-H "Content-Type: application/json" \
		-d "{\"goal\": \"$(GOAL)\", \"project_id\": \"test\"}"

# Development shortcuts
gateway: ## Start only gateway service
	.venv\Scripts\activate && python -m uvicorn apps.gateway.main:app --host 0.0.0.0 --port 8090 --reload

memory: ## Start only memory service
	.venv\Scripts\activate && python -m uvicorn libs.memory.main:app --host 0.0.0.0 --port 8003 --reload

# Database operations
db-init: ## Initialize database
	.venv\Scripts\activate && python -c "from libs.memory.kv_store import kv_store; print('KV store initialized')"

# Model operations
model-list: ## List all models in registry
	@find .promu/models -name "model.json" -exec echo "=== {} ===" \; -exec cat {} \;

model-fetch: ## Fetch model info from HuggingFace
	.venv\Scripts\activate && python tools/fetch_radon_models.py

model-prepare: ## Prepare weights directory structure
	bash tools/prepare_weights.sh

model-download: ## Download all model weights (requires HF token)
	@echo "Downloading RadonSAI models..."
	@echo "This will download ~50GB of model weights"
	@echo "Make sure you have enough disk space and bandwidth"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	git clone https://huggingface.co/MagistrTheOne/RadonSAI-Small weights/radon/small-0.1b
	git clone https://huggingface.co/MagistrTheOne/RadonSAI weights/radon/base-0.8b
	git clone https://huggingface.co/MagistrTheOne/RadonSAI-Balanced weights/radon/balanced-3b
	git clone https://huggingface.co/MagistrTheOne/RadonSAI-Efficient weights/radon/efficient-3b
	git clone https://huggingface.co/MagistrTheOne/RadonSAI-Ultra weights/radon/ultra-13b
	git clone https://huggingface.co/MagistrTheOne/RadonSAI-Mega weights/radon/mega-70b

# Documentation
docs: ## Generate documentation
	@echo "Generating API documentation..."
	@echo "Gateway API: http://localhost:8090/docs"

skills.test: ## Test all skills
	@echo "Testing skills..."
	.venv\Scripts\activate && python -m pytest examples/skills/ -v

skills.lint: ## Lint skills code
	@echo "Linting skills..."
	.venv\Scripts\activate && ruff check examples/skills/
	.venv\Scripts\activate && mypy examples/skills/

ui: ## Open UI dashboard
	@echo "Opening UI dashboard..."
	@echo "Dashboard available at: http://localhost:8096"

cli: ## Run CLI
	@echo "Running CLI..."
	.venv\Scripts\activate && python apps/cli/main.py --help

observability: ## Check observability
	@echo "Checking observability..."
	@curl -s http://localhost:8095/health | python -m json.tool
	@echo "Memory API: http://localhost:8003/docs"

# CI/CD helpers
ci-test: ## Run tests for CI
	.venv\Scripts\activate && pytest tests/unit tests/integration --cov=libs --cov=apps --cov-report=xml

ci-build: ## Build Docker images for CI
	docker-compose -f infra/docker-compose.yml build

# Monitoring
monitor: ## Show resource usage
	docker stats --no-stream

# Backup and restore
backup: ## Backup data
	@echo "Backing up data..."
	@mkdir -p backups
	@cp -r .promu backups/promu-$(shell date +%Y%m%d-%H%M%S)

restore: ## Restore data (usage: make restore BACKUP=backups/promu-YYYYMMDD-HHMMSS)
	@echo "Restoring from $(BACKUP)..."
	@cp -r $(BACKUP) .promu
