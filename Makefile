# Variables
DOCKER_COMPOSE = docker compose
SERVICE ?= dev   # change to your service name if needed

.PHONY: help build up down down-v down-rmi restart logs shell

help: ## Show available commands
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@echo "  build      Build all services"
	@echo "  up         Start services in background"
	@echo "  down       Stop and remove containers, networks"
	@echo "  down-v     Stop and remove containers + volumes"
	@echo "  down-rmi   Stop and remove containers + images"
	@echo "  restart    Restart services"
	@echo "  logs       Show logs (tail -f)"
	@echo "  shell      Enter a shell in the '$(SERVICE)' container"

build: ## Build docker images
	$(DOCKER_COMPOSE) build dev -t kg-builder:dev .

up: ## Start containers in background
	$(DOCKER_COMPOSE) up -d frontend

down: ## Stop and remove containers
	$(DOCKER_COMPOSE) down

down-v: ## Stop and remove containers + volumes
	$(DOCKER_COMPOSE) down -v

down-rmi: ## Stop and remove containers + images
	$(DOCKER_COMPOSE) down --rmi local

down-all: ## Stop and remove containers, volumes, images, and networks
	$(DOCKER_COMPOSE) down -v --rmi all --remove-orphans

restart: down up ## Restart services

logs: ## Follow logs
	$(DOCKER_COMPOSE) logs -f

shell: ## Open a shell inside the app container
	$(DOCKER_COMPOSE) exec $(SERVICE) bash
