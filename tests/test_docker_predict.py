"""Exercise the Docker entry point without a GPU: real shell, fake Python workers."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docker/predict.sh"


@pytest.mark.parametrize("worker_status,phase,expected_status", [(0, "", 0), (7, "", 1), (124, "", 0), (0, "merge", 9), (0, "fill-logs", 9), (0, "check", 9)])
def test_worker_and_validation_status(tmp_path, worker_status, phase, expected_status):
    code, data, out, bin_dir = [tmp_path / name for name in ("code", "data", "result", "bin")]
    for path in (code, data, bin_dir):
        path.mkdir()
    (code / "flags.sh").write_bytes((SCRIPT.parents[1] / "scripts/flags.sh").read_bytes())
    (data / "questions.jsonl").write_text('{"id":"q_1"}\n', encoding="utf-8")
    # Replace only the container's absolute working directory; run every other line unchanged.
    script = code / "predict.sh"
    script.write_text(SCRIPT.read_text().replace("cd /code\n", f'cd "{code}"\n'), encoding="utf-8")
    stub = bin_dir / "python3"
    stub.write_text('''#!/usr/bin/env bash
if [[ "$2" == datamate.predict ]]; then
  echo "worker $CUDA_VISIBLE_DEVICES" >> "$RESULT_DIR/steps"
  if [[ "$CUDA_VISIBLE_DEVICES" == 0 ]]; then exit "$TEST_WORKER_STATUS"; fi
  exit 0
fi
echo "$3" >> "$RESULT_DIR/steps"
if [[ "$3" == "$TEST_FAIL_PHASE" ]]; then exit 9; fi
''', encoding="utf-8")
    stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DATA_DIR": str(data), "RESULT_DIR": str(out), "NUM_GPUS": "2",
           "TIMEOUT_SECONDS": "1000", "TEST_WORKER_STATUS": str(worker_status), "TEST_FAIL_PHASE": phase}
    run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert run.returncode == expected_status, run.stdout + run.stderr
    steps = (out / "steps").read_text().splitlines()
    assert {"worker 0", "worker 1", "merge"} <= set(steps)
    if phase != "merge":
        assert "fill-logs" in steps
    if phase not in ("merge", "fill-logs"):
        assert "check" in steps
    if worker_status:
        assert "shard0" in run.stdout + run.stderr and str(worker_status) in run.stdout + run.stderr
