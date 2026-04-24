# PyPI Traffic Generator - Remote Deployment Makefile

SCRIPT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CONTROL_SCRIPT = $(SCRIPT_DIR)/control.sh
DEPLOY_SCRIPT = $(SCRIPT_DIR)/deploy.sh

.PHONY: help deploy start stop status logs clean

help:
	@echo "PyPI Traffic Generator - Remote Commands"
	@echo "=========================================="
	@echo ""
	@echo "  make deploy        - Deploy files to all PCs"
	@echo "  make start N=10   - Start N replicas on each PC"
	@echo "  make stop         - Stop all replicas"
	@echo "  make status       - Show replica count per PC"
	@echo "  make logs         - Tail logs from all PCs"
	@echo "  make clean        - Remove all traces from all PCs"
	@echo ""
	@echo "Edit pcs.conf to add/remove PCs"

deploy:
	@echo "Deploying to all PCs..."
	@bash $(DEPLOY_SCRIPT)

start:
	@if [ -z "$(N)" ]; then echo "Usage: make start N=10"; exit 1; fi
	@echo "Starting $(N) replicas on each PC..."
	@bash $(CONTROL_SCRIPT) start $(N)

stop:
	@echo "Stopping all replicas..."
	@bash $(CONTROL_SCRIPT) stop

status:
	@bash $(CONTROL_SCRIPT) status

logs:
	@echo "Following logs (Ctrl+C to exit)..."
	@bash $(CONTROL_SCRIPT) logs

clean:
	@echo "Cleaning all PCs..."
	@bash $(CONTROL_SCRIPT) clean