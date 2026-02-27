#!/bin/bash

# 1. Borrar el release fallido para limpiar el historial de Helm
helm uninstall ollama

# 2. Reinstalar con el values.yaml corregido
helm install ollama ollama-helm/ollama -f values.yaml


