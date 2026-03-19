# Makefile for Jenkins Pipelines for Train Service

.PHONY: build-docs serve-docs

# Build the Sphinx documentation
build-docs:
	cd docs && make html

# Serve the built documentation on localhost:8000
serve-docs:
	cd docs/_build/html && python3 -m http.server 8000