.PHONY: install uninstall

# Target to deploy all Milvus resources to Kubernetes in the correct order
install:
	@echo "🚀 Deploying Milvus stack to Kubernetes..."
	@echo "Applying Namespace..."
	@kubectl apply -f 00-namespace.yaml
	@echo "Applying Storage..."
	@kubectl apply -f 01-storage.yaml
	@echo "Applying Configs and Secrets..."
	@kubectl apply -f 02-configs.yaml
	@echo "Applying etcd..."
	@kubectl apply -f 10-etcd.yaml
	@echo "Applying minio..."
	@kubectl apply -f 11-minio.yaml
	@echo "Applying milvus..."
	@kubectl apply -f 20-milvus.yaml
	@echo "Applying prometheus..."
	@kubectl apply -f 30-prometheus.yaml
	@echo "Applying grafana..."
	@kubectl apply -f 31-grafana.yaml
	@echo "Applying attu..."
	@kubectl apply -f 40-attu.yaml
	@echo "✅ Milvus stack deployed. Use 'kubectl get pods -n milvus -w' to check the status."

# Target to delete all Milvus resources from Kubernetes in reverse order
uninstall:
	@echo "🔥 Deleting Milvus stack from Kubernetes..."
	@kubectl delete -f 40-attu.yaml --ignore-not-found=true
	@kubectl delete -f 31-grafana.yaml --ignore-not-found=true
	@kubectl delete -f 30-prometheus.yaml --ignore-not-found=true
	@kubectl delete -f 20-milvus.yaml --ignore-not-found=true
	@kubectl delete -f 11-minio.yaml --ignore-not-found=true
	@kubectl delete -f 10-etcd.yaml --ignore-not-found=true
	@kubectl delete -f 02-configs.yaml --ignore-not-found=true
	@kubectl delete -f 01-storage.yaml --ignore-not-found=true
	@kubectl delete -f 00-namespace.yaml --ignore-not-found=true
	@echo "✅ Milvus stack deleted."
