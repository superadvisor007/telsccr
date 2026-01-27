# Makefile for TelegramSoccer

.PHONY: help install test lint format run docker-build docker-up docker-down clean

help:
	@echo "TelegramSoccer - Make Commands"
	@echo "=============================="
	@echo "install      - Install dependencies"
	@echo "install-dev  - Install dev dependencies"
	@echo "test         - Run tests with coverage"
	@echo "lint         - Run linters (black, isort, flake8, mypy)"
	@echo "format       - Format code with black and isort"
	@echo "run          - Run the application"
	@echo "pipeline     - Run daily pipeline only"
	@echo "docker-build - Build Docker image"
	@echo "docker-up    - Start Docker containers"
	@echo "docker-down  - Stop Docker containers"
	@echo "clean        - Clean cache and temp files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	black --check src tests
	isort --check-only src tests
	flake8 src tests --max-line-length=100 --extend-ignore=E203,W503
	mypy src --ignore-missing-imports

format:
	black src tests
	isort src tests

run:
	python src/main.py

pipeline:
	python src/pipeline.py

docker-build:
	docker build -t telegramsoccer:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	rm -rf build/ dist/
