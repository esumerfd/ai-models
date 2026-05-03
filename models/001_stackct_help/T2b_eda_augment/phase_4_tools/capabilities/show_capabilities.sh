#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-stackslm}"

response=$(curl -s http://localhost:11434/api/show -d "{\"model\": \"$MODEL\"}")

echo "Capabilities for model: $MODEL"
echo "---"
echo "$response" | jq '.capabilities'
