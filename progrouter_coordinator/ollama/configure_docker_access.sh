#!/usr/bin/env bash
# Run as your regular user: bash ollama/configure_docker_access.sh
set -euo pipefail

if [[ "$EUID" -eq 0 ]]; then
  echo "Run this script as your regular user, without sudo." >&2
  exit 1
fi
command -v docker >/dev/null
docker_user="$(id -un)"
echo "Adding $docker_user to the docker group."
echo "Docker group membership grants effectively root-level host access."
echo "Only continue if this is permitted on your machine."
read -r -p "Type ADD to continue: " docker_answer
if [[ "$docker_answer" != "ADD" ]]; then
  echo "No changes made."
  exit 0
fi

if ! getent group docker >/dev/null; then
  sudo groupadd docker
fi
sudo usermod -aG docker "$docker_user"
sudo systemctl enable --now docker
echo "Log out of your desktop/SSH session completely and log back in."
echo "Then reopen your terminal and IDE so they inherit the new group."
echo "Do not change Docker socket permissions to make it world-writable."
