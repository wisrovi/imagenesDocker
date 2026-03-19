# Makefile for Kind GPU Cluster

.PHONY: help install create verify gpu-setup test clean pdf clean-latex

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install Kind
	./install_kind.txt

create: ## Create cluster
	kind create cluster --config kind-config.yaml

verify: ## Verify cluster
	kubectl get nodes
	kubectl get nodes --show-labels

gpu-setup: ## Setup GPU
	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml
	kubectl label node kind-control-plane nvidia.com/gpu.present=true
	kubectl label node kind-worker kind-worker2 kind-worker3 kind-worker4 nvidia.com/gpu.present=true

test: ## Test GPU
	./gpu_test.txt

clean: ## Cleanup cluster
	kind delete cluster

pdf: ## Compile LaTeX documentation to PDF
	cd docs && pdflatex documentation.tex

clean-latex: ## Clean LaTeX auxiliary files
	cd docs && rm -f *.aux *.log *.toc *.lof *.lot *.out *.bbl *.blg

all: install create verify gpu-setup test ## Run full setup