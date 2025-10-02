.PHONY: help install dev-install test lint format docs serve-docs clean

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  dev-install  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo "  clean        Clean build artifacts"

install:
	pip install .

dev-install:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=bookwyrm --cov-report=html --cov-report=term

lint:
	flake8 bookwyrm tests
	mypy bookwyrm
	black --check bookwyrm tests

format:
	black bookwyrm tests
	isort bookwyrm tests

docs:
	mkdocs build

serve-docs:
	mkdocs serve

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
