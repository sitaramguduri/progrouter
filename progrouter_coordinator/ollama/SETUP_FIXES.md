# Finish the local runtime setup

Run from the coordinator project directory. These scripts are not executed
automatically. They require Ubuntu/systemd and an existing Ollama/Docker installation.

## 1. Apply Ollama settings

Finish any active inference first: this restarts Ollama.

```bash
bash ollama/configure_service.sh
```

Installs `override.conf` as `/etc/systemd/system/ollama.service.d/99-progrouter.conf`.
Re-running replaces that dedicated file; other drop-ins are left in place.
Verify the printed effective environment. No model weights are downloaded.

## 2. Grant your account Docker access

The current Python launchers call Docker directly. The following grants the current
user effectively root-level access through the Docker daemon. On a managed machine,
use this only if authorized by its administrator. Do not run the whole workflow as root.

```bash
bash ollama/configure_docker_access.sh
```

The script asks you to type ADD before applying the change.
Log out completely and log back in, then reopen your IDE/terminal. Existing processes
do not automatically inherit new group membership.

## 3. CPU MiniLM

`progress.py` now defaults `SentenceEncoder` to `device="cpu"`, including instances
created by `run.py`. Advanced callers can explicitly select another device.
The model weights must already be cached in your user environment.

## Verify all three

```bash
conda activate qwen-local
bash ollama/verify_setup.sh
```

This runs a harmless Python container and one CPU embedding through the real workflow
encoder. It does not download images/weights or run LLM inference. If the Python image
is missing, explicitly download it first:

```bash
docker pull python:3.12-slim
```

Then repeat verification. The scripts stop at the first failed command.

## Undo the system changes if needed

Remove only the dedicated ProgRouter drop-in, then reload/restart Ollama:

```bash
sudo rm /etc/systemd/system/ollama.service.d/99-progrouter.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

To revoke Docker group membership, use your actual username:

```bash
sudo gpasswd -d YOUR_USERNAME docker
```

Log out and back in after revoking membership too.
