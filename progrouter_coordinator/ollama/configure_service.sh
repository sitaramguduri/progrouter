#!/usr/bin/env bash
# Run explicitly: bash ollama/configure_service.sh
set -euo pipefail

config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v ollama >/dev/null
test -f "$config_dir/override.conf"

echo "Installing a dedicated Ollama systemd drop-in; restarting Ollama."
echo "Finish any active inference before running this script."
sudo install -d -m 0755 /etc/systemd/system/ollama.service.d
sudo install -m 0644 "$config_dir/override.conf" \
  /etc/systemd/system/ollama.service.d/99-progrouter.conf
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl restart ollama
systemctl show ollama --property=Environment
systemctl is-active ollama

echo "Expected settings: OLLAMA_NUM_PARALLEL=1, OLLAMA_MAX_LOADED_MODELS=1,"
echo "OLLAMA_CONTEXT_LENGTH=4096, OLLAMA_HOST=127.0.0.1:11434."
echo "Inspect all drop-ins if values differ: systemctl cat ollama"
