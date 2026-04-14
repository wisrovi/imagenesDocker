#!/bin/bash
# Script para desplegar Kafka en Kubernetes con Strimzi
# Ejecutar desde el directorio kafka_kube

set -e

echo "=== Desplegando Kafka en Kubernetes ==="

echo "[0/7] Instalando operador Strimzi..."
kubectl apply -f https://strimzi.io/install/latest?namespace=kafka

echo "[1/7] Creando namespace..."
kubectl apply -f 00-namespace.yaml

echo "[2/7] Creando secret para SMB..."
kubectl apply -f 01-secret.yaml

echo "[3/7] Creando PersistentVolumes..."
kubectl apply -f 02-pv.yaml

echo "[4/7] Creando PersistentVolumeClaims..."
kubectl apply -f 03-pvc.yaml

echo "[5/7] Esperando a que los PVCs estén bound..."
kubectl wait --for=condition=Bound pvc -l app=kafka --timeout=300s -n kafka 2>/dev/null || true

echo "[6/7] Desplegando Kafka Cluster..."
kubectl apply -f 04-kafka.yaml

echo "[7/7] Esperando a que Kafka esté ready..."
kubectl wait kafka kafka-cluster -n kafka --for=condition=Ready --timeout=600s

echo "=== Despliegue completado ==="
echo ""
echo "Para ver el estado:"
echo "  kubectl get pods -n kafka -w"
echo ""
echo "Kafka bootstrap:"
echo "  kafka-cluster-kafka-bootstrap.kafka.svc:9092"
