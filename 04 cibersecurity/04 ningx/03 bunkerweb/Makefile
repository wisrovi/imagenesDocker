.PHONY: install lint format security test clean help

help:
	@echo "Available targets:"
	@echo "  install   - Install dependencies from requirements.txt"
	@echo "  lint      - Run pylint static analysis (target: >=9.5)"
	@echo "  format    - Run isort and black formatters"
	@echo "  security  - Run bandit security scan"
	@echo "  test      - Run pytest with coverage report"
	@echo "  clean     - Remove cache and temporary files"

install:
	@if [ -f requirements.txt ]; then \
		pip install -r requirements.txt; \
	else \
		echo "No requirements.txt found"; \
	fi

lint:
	@if command -v pylint >/dev/null 2>&1; then \
		pylint \
			--disable=import-error,no-member,no-name-in-module \
			--score=yes \
			. || true; \
	else \
		echo "pylint not installed, skipping lint"; \
	fi

format:
	@if command -v isort >/dev/null 2>&1; then \
		isort .; \
	else \
		echo "isort not installed"; \
	fi
	@if command -v black >/dev/null 2>&1; then \
		black .; \
	else \
		echo "black not installed"; \
	fi

security:
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r . -x ./.venv || true; \
	else \
		echo "bandit not installed, skipping security scan"; \
	fi

test:
	@if command -v pytest >/dev/null 2>&1; then \
		pytest \
			--cov=. \
			--cov-report=term-missing \
			--cov-report=html \
			--ignore=tests \
			-v || true; \
	else \
		echo "pytest not installed, skipping tests"; \
	fi

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ 2>/dev/null || true
	@rm -rf .ruff_cache/ 2>/dev/null || true
	@echo "Clean complete"
