#!/bin/bash

# Usage: sh scripts/run_analysis.sh <repo_owner> <repo_name>
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sh scripts/run_analysis.sh <repo_owner> <repo_name>"
    exit 1
fi

REPO_OWNER=$1
REPO_NAME=$2
IMAGE_NAME="wisrovi/pr_analizer:latest"

echo "Running analysis for ${REPO_OWNER}/${REPO_NAME}..."

docker run --rm \
    -v ./report:/report \
    -v ~/.ssh:/root/.ssh \
    -w /usr/src/app \
    --env-file config/secrets.env \
    -e REPO_OWNER=$REPO_OWNER \
    -e REPO_NAME=$REPO_NAME \
    ${IMAGE_NAME} \
        bash -c "python -m src.pr_analyzer.main"

echo "Analysis complete."
