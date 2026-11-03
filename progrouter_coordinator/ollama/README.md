# Local Ollama configuration

Four worker models: Qwen2.5-Coder 0.5B, 1.5B, 3B, 7B.
Coordinator: the same 7B model, with a different role prompt.
Each Modelfile sets 4096 context tokens, 1024 maximum output tokens, and temperature 0.
The FROM tags use Ollama's packaged defaults; inspect quantization and record model digests before experiments.

These files do not install, download, start, or configure services automatically.

## Server configuration

For an existing systemd Ollama installation, run `sudo systemctl edit ollama`
and paste the contents of `override.conf`. Then run:

```bash
sudo systemctl restart ollama
```

The service settings allow one parallel request and one loaded model.
One Ollama service supplies all roles. GPU placement still needs verification.

## Provision models

From the coordinator project directory, run these commands yourself.
Creating the aliases can download the underlying model if it is missing.

```bash
ollama create progrouter-coder:0.5b -f ollama/0.5b.Modelfile
ollama create progrouter-coder:1.5b -f ollama/1.5b.Modelfile
ollama create progrouter-coder:3b -f ollama/3b.Modelfile
ollama create progrouter-coder:7b -f ollama/7b.Modelfile
ollama list
ollama show progrouter-coder:7b
```

## Application configuration

`examples/ollama_config.json` is a fixed-7B smoke-test configuration compatible
with the current CLI. It deliberately has no routing cost profiles: calibrated
energy, latency and budgets are still required before enabling the four-model router.
All four aliases above are the agreed future routing pool.

Once the service, model, task, and Python dependencies are ready:

```bash
python run.py --task examples/task.json --config examples/ollama_config.json --output runs/ollama-smoke-001
```

Visible code execution is disabled unless `--enable-tests` is supplied.
The Docker Python image and Docker socket access must be available for that flag.

The prompt character limit is only a preliminary guard, not a token-aware
4096-token context guarantee. Token admission remains to be implemented.
This setup does not configure CPU MiniLM: the current SentenceEncoder still
needs an explicit device option in code. GPU energy calibration and the
HumanEval Plus progress adapter also remain outstanding.
