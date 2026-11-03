#!/usr/bin/env bash
# Run after logging back in and activating qwen-local.
# No downloads or model generation. Runs a cached Python container if available.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Ollama service environment:"
systemctl show ollama --property=Environment
echo "7B model parameters:"
ollama show progrouter-coder:7b --parameters
echo "Docker client/server access:"
docker version
echo "Testing an already-cached Python image (no automatic pull):"
docker run --rm --pull=never --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --user 65534:65534 --memory 256m --pids-limit 32 \
  python:3.12-slim python -I -B -c 'print("Docker Python execution passed")'

echo "Testing the actual workflow encoder using cached weights:"
cd "$project_dir"
python -B - <<'PY'
from progress import SentenceEncoder
encoder = SentenceEncoder(local_files_only=True)
assert encoder.model.device.type == "cpu", encoder.model.device
vectors = encoder.encode(["Public examples passed."])
assert len(vectors) == 1 and len(vectors[0]) == 384
print("Workflow MiniLM device:", encoder.model.device)
print("Workflow MiniLM embedding check passed")
PY
