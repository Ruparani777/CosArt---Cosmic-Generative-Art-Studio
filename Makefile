# CosArt Makefile
# Quick commands for development and deployment

.PHONY: help install run test clean docker-up docker-down format lint

# Default target
help:
	@echo "CosArt Development Commands:"
	@echo "  install     - Install Python dependencies"
	@echo "  run         - Run development server"
	@echo "  test        - Run all tests"
	@echo "  clean       - Clean cache and temporary files"
	@echo "  docker-up   - Start all services with Docker"
	@echo "  docker-down - Stop all Docker services"
	@echo "  format      - Format code with black"
	@echo "  lint        - Lint code with flake8"
	@echo "  frontend    - Start frontend development server"
	@echo "  build       - Build for production"

# Installation
install:
	pip install -r requirements.txt
	cd frontend && npm install

# Development
run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm start

# Testing
test:
	pytest tests/ -v --cov=cosart

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

# Code Quality
format:
	black .
	cd frontend && npm run format

lint:
	flake8 .
	cd frontend && npm run lint

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf dist/ build/ .eggs/

# Production build
build:
	docker-compose -f docker-compose.prod.yml build

# Database
db-init:
	alembic upgrade head

db-migrate:
	alembic revision --autogenerate -m "$(msg)"

# Models
train-model:
	python scripts/train_cosmic_gan.py

# Deployment
deploy:
	docker-compose -f docker-compose.prod.yml up -d

# Logs
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f cosart-api

# Health check
health:
	curl http://localhost:8000/health

# Full development setup
dev: install
	@echo "Starting development environment..."
	@echo "API will be available at: http://localhost:8000"
	@echo "Frontend will be available at: http://localhost:3000"
	@echo "API Docs will be available at: http://localhost:8000/docs"
	@make run &
	@sleep 2
	@make frontend