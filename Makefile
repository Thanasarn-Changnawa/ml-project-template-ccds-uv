# Makefile for ML Project Template

.PHONY: help install install-dev test lint format clean train notebook

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	uv venv
	uv pip install -e .

install-dev:  ## Install project and development dependencies
	uv venv
	uv pip install -e ".[dev]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting checks
	ruff check .

format:  ## Format code with ruff
	ruff format .
	ruff check --fix .

clean:  ## Clean up temporary files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

train:  ## Run training script
	python src/train.py

notebook:  ## Start Jupyter notebook
	jupyter notebook

check:  ## Run all checks (lint and test)
	make lint
	make test
