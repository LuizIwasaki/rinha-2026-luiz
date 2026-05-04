#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Copying references.json.gz to build context..."
cp "$SCRIPT_DIR/../resources/references.json.gz" "$SCRIPT_DIR/references.json.gz" 2>/dev/null || true

echo "Building and starting containers..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" up --build -d

echo "Waiting for APIs to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:9999/ready > /dev/null 2>&1; then
        echo "API is ready!"
        exit 0
    fi
    sleep 2
done

echo "ERROR: API did not become ready in time"
exit 1
