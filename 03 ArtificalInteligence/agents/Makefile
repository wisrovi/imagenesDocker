# Makefile for AI Tools Installation Scripts project
#

.PHONY: help install install-gemini install-opencode docs serve-docs clean all

# Default target
all: install docs

# Install all tools
install: install-gemini install-opencode

# Install Google Gemini CLI
install-gemini:
	@echo "Installing Google Gemini CLI..."
	@./install/gemini.sh

# Install OpenCode
install-opencode:
	@echo "Installing OpenCode..."
	@./install/opencode.sh

# Build documentation
docs:
	@echo "Building documentation..."
	@cd docs && make build

# Serve documentation locally
serve-docs:
	@echo "Serving documentation..."
	@cd docs && make serve

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf docs/_build

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Install tools and build docs (default)"
	@echo "  install      - Install all AI tools"
	@echo "  install-gemini   - Install Google Gemini CLI"
	@echo "  install-opencode - Install OpenCode"
	@echo "  docs         - Build Sphinx documentation"
	@echo "  serve-docs   - Build and serve documentation locally"
	@echo "  clean        - Remove build files"
	@echo "  help         - Show this help message"