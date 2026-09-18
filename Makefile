.PHONY: install install-dev test lint format clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest ruff black

test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m ruff check src/ tests/ scripts/

format:
	python -m ruff format src/ tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf *.egg-info/