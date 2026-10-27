#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-credit-default-api}"
cd "$(dirname "$0")/.."

if [ ! -f models/model.joblib ]; then
    echo "models/model.joblib not found. Get it first: poetry run dvc pull (or poetry run dvc repro)." >&2
    exit 1
fi

docker build -t "$IMAGE" .
echo "Built image: $IMAGE"
