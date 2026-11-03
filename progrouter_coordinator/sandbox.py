"""Disposable Docker execution for independent benchmark evaluators, never host exec."""
from pathlib import Path
import math
import subprocess
import tempfile
import time
import uuid


class DockerSandbox:
    def __init__(self, image='python:3.12-slim', timeout=60, max_output_bytes=1048576):
        if not math.isfinite(timeout) or timeout <= 0 or max_output_bytes <= 0:
            raise ValueError('Sandbox limits must be positive')
        self.image, self.timeout, self.max_output_bytes = image, timeout, max_output_bytes

    def run(self, files):
        name = 'progrouter-eval-'+uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix='progrouter-eval-') as folder:
            root = Path(folder)
            root.chmod(0o755)
            for filename, content in files.items():
                if Path(filename).name != filename:
                    raise ValueError('Sandbox files must have plain filenames')
                path = root/filename
                path.write_text(content, encoding='utf-8')
                path.chmod(0o644)
            cmd = ['docker', 'run', '--rm', '--pull=never', '--name', name,
                '--network', 'none', '--read-only', '--cpus', '1', '--memory', '2g',
                '--memory-swap', '2g', '--pids-limit', '128', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges', '--user', '65534:65534',
                '--tmpfs', '/tmp:rw,nosuid,size=512m', '--env', 'HOME=/tmp',
                '--env', 'XDG_CACHE_HOME=/tmp/.cache',
                '--env', 'HF_HUB_OFFLINE=1', '--env', 'TRANSFORMERS_OFFLINE=1',
                '--mount', f'type=bind,src={root},dst=/work,readonly',
                self.image, 'python', '-B', '/work/grade.py']
            process = None
            try:
                with (root/'output').open('wb') as output:
                    process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT)
                    end = time.monotonic()+self.timeout
                    status = 'completed'
                    while process.poll() is None:
                        if time.monotonic() >= end:
                            status = 'timeout'
                            break
                        if (root/'output').stat().st_size > self.max_output_bytes:
                            status = 'output_limit'
                            break
                        try:
                            process.wait(timeout=.05)
                        except subprocess.TimeoutExpired:
                            pass
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    elif (root/'output').stat().st_size > self.max_output_bytes:
                        status = 'output_limit'
                    elif process.returncode:
                        status = 'error'
                with (root/'output').open('rb') as output:
                    output.seek(0, 2)
                    output.seek(max(0, output.tell()-self.max_output_bytes))
                    text = output.read().decode('utf-8', errors='replace')
                return {'status': status, 'stdout': text}
            except FileNotFoundError:
                return {'status': 'unavailable', 'stdout': ''}
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
                try:
                    subprocess.run(['docker', 'rm', '-f', name], timeout=5,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
