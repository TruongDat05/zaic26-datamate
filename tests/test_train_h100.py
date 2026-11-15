"""Run the actual H100 launcher with fake Python stages; no GPU or model execution."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "training/train_h100.sh"


@pytest.fixture
def launcher_env(tmp_path):
    data, model, init = [tmp_path / n for n in ("data", "model", "init")]
    for p in (data, model, init):
        p.mkdir()
    for p in (data / "train.jsonl", data / "dev.jsonl", model / "config.json", init / "adapter_config.json", init / "adapter_model.safetensors"):
        p.write_text("{}\n")
    stub = tmp_path / "python"
    stub.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
a = sys.argv[1:]
stage = ("memory" if a[0] == "-" else "kernel" if a[0].endswith("check_fla.py") else
         "data" if "--validate-only" in a else "preflight" if "--longest-first" in a else "train")
record = {"stage": stage, "args": a, "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}
if "--config" in a:
    record["config"] = json.loads(Path(a[a.index("--config") + 1]).read_text())
with open(os.environ["TEST_CALLS"], "a") as f:
    f.write(json.dumps(record) + "\\n")
if stage == os.environ.get("TEST_FAIL"):
    sys.exit(17)
if stage == "kernel":
    print("fla check FAIL" if os.environ.get("TEST_FAIL") == "kernel_message" else "fla check PASS")
if stage == "preflight":
    out = Path(a[a.index("--out") + 1])
    out.mkdir(exist_ok=True)
    reserved = 79 if os.environ.get("TEST_FAIL") == "memory_limit" else 40
    (out / "run_manifest.json").write_text(json.dumps({"gpu": "fake H100", "peak_gpu_reserved_gib": reserved, "gpu_total_gib": 80}))
if stage == "memory":
    sys.argv = ["-", *a[1:]]
    exec(compile(sys.stdin.read(), "memory-check", "exec"))
''')
    stub.chmod(0o755)
    return {**os.environ, "PYTHON": str(stub), "DATA_DIR": str(data), "MODEL_DIR": str(model), "INIT_ADAPTER": str(init),
            "OUT": str(tmp_path / "run"), "TEST_CALLS": str(tmp_path / "calls.jsonl"), "TEST_FAIL": "", "RESUME": "0", "ONLY": "",
            "TRAIN_CONFIG": str(REPO / "training/configs/qwen35_sft_h100_v2.json"), "TOKENIZER_DIR": "", "WORLD_SIZE": "1", "CUDA_VISIBLE_DEVICES": "0"}


@pytest.mark.parametrize("fail", ["", "data", "kernel", "kernel_message", "preflight", "memory", "memory_limit", "train"])
def test_h100_profile_and_stage_failures(launcher_env, fail):
    env = {**launcher_env, "TEST_FAIL": fail}
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)
    assert (run.returncode == 0) == (not fail), run.stdout + run.stderr
    calls = [json.loads(l) for l in Path(env["TEST_CALLS"]).read_text().splitlines()]
    stages = [c["stage"] for c in calls]
    order = ["data", "kernel", "preflight", "memory", "train"]
    failed_stage = {"kernel_message": "kernel", "memory_limit": "memory"}.get(fail, fail)
    end = order.index(failed_stage) + 1 if fail else len(order)
    assert stages == order[:end]
    for c in calls:
        assert c["cuda"] == "0"
        if "config" in c:
            cfg = c["config"]
            assert (cfg["epochs"], cfg["micro_batch"], cfg["grad_accum"], cfg["dev_max_rows"], cfg["early_stop"]) == (1, 1, 16, 64, 0)
            assert (cfg["eval_steps"], cfg["save_steps"]) == (100, 100)
            assert cfg["limit_rows"] is None and cfg["max_steps"] == -1
            assert cfg["init_adapter"] == "runs/lora/qwen35_sft_4xl40s/adapter"
            assert (cfg["max_len"], cfg["lr"], cfg["r"], cfg["alpha"]) == (16384, 5e-5, 16, 32)
    if not fail:
        assert "--resume" not in calls[-1]["args"]
        assert "--dev" not in calls[2]["args"]
        assert calls[2]["args"][calls[2]["args"].index("--grad-accum") + 1] == "1"


def test_h100_data_only_does_not_need_model_or_adapter(launcher_env):
    env = {**launcher_env, "ONLY": "data", "MODEL_DIR": "missing-model", "INIT_ADAPTER": "missing-adapter", "TOKENIZER_DIR": "adapters/sql_rft_v3"}
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr
    calls = [json.loads(l) for l in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert [c["stage"] for c in calls] == ["data"] and "--tokenizer" in calls[0]["args"]


@pytest.mark.parametrize("resume", ["0", "1"])
def test_h100_existing_run_requires_explicit_resume(launcher_env, resume):
    env = {**launcher_env, "RESUME": resume}
    (Path(env["OUT"]) / "checkpoint-50").mkdir(parents=True)
    run = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=10)
    if resume == "0":
        assert run.returncode != 0 and "RESUME=1" in run.stdout
        assert not Path(env["TEST_CALLS"]).exists()
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        calls = [json.loads(l) for l in Path(env["TEST_CALLS"]).read_text().splitlines()]
        assert [c["stage"] for c in calls] == ["data", "kernel", "train"]
        assert "--resume" in calls[-1]["args"]
