#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-credit-default-api}"
PORT="${2:-8000}"
NAME=credit-default-api

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null

for _ in $(seq 1 30); do
    if curl -fs "http://127.0.0.1:$PORT/health"; then
        echo
        echo "API: http://127.0.0.1:$PORT (docs: /docs). Stop: docker stop $NAME"
        exit 0
    fi
    sleep 1
done

echo "API did not start in 30 s. Logs:" >&2
docker logs "$NAME" >&2
docker rm -f "$NAME" >/dev/null
exit 1
