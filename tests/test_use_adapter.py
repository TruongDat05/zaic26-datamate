"""scripts/use_adapter.py: switching the submitted LoRA rewrites every reference, and refuses adapters that cannot replace it."""
import json
import shutil
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import use_adapter  # noqa: E402
from rule_check import manifest_adapter  # noqa: E402

REFS = ["docker/Dockerfile", ".dockerignore", "scripts/run_public.sh", "scripts/run_holdout.sh", "docker/model_manifest.json"]


def weights(path: Path, n: int):
    head = json.dumps({"__metadata__": {}, "w": {"dtype": "BF16", "shape": [n], "data_offsets": [0, 2 * n]}}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0" * (2 * n))


@pytest.fixture
def repo(tmp_path):
    for rel in REFS + ["adapters/sql_rft_v3/adapter_config.json"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, tmp_path / rel)
    new = tmp_path / "run/adapter"
    new.mkdir(parents=True)
    shutil.copy2(REPO / "adapters/sql_rft_v3/adapter_config.json", new / "adapter_config.json")
    weights(new / "adapter_model.safetensors", 1000)
    return tmp_path, new


def test_manifest_names_the_current_adapter():
    m = json.loads((REPO / "docker/model_manifest.json").read_text())
    assert manifest_adapter(m) == "adapters/sql_rft_v3"
    assert manifest_adapter({"lora_adapter": "none"}) is None


def test_switch_rewrites_every_reference(repo):
    root, new = repo
    before = {rel: (root / rel).read_text() for rel in REFS}
    use_adapter.main([str(new), "--name", "sql_bf16_v1", "--root", str(root), "--dry-run"])
    assert {rel: (root / rel).read_text() for rel in REFS} == before and not (root / "adapters/sql_bf16_v1").exists()
    use_adapter.main([str(new), "--name", "sql_bf16_v1", "--root", str(root)])
    for rel in REFS:
        text = (root / rel).read_text()
        assert "adapters/sql_rft_v3" not in text and "adapters/sql_bf16_v1" in text, rel
    m = json.loads((root / "docker/model_manifest.json").read_text())
    assert m["lora_adapter_parameters"] == 1000 and m["total_parameters_with_adapter"] == m["safetensors_total_parameters"] + 1000
    assert (root / "adapters/sql_bf16_v1/adapter_model.safetensors").read_bytes() == (new / "adapter_model.safetensors").read_bytes()
    assert "COPY adapters/sql_bf16_v1 /code/models/lora" in (root / "docker/Dockerfile").read_text()


def test_refuses_other_rank_and_lfs_pointer(repo):
    root, new = repo
    cfg = json.loads((new / "adapter_config.json").read_text())
    (new / "adapter_config.json").write_text(json.dumps({**cfg, "r": 32}))
    with pytest.raises(SystemExit, match="r=32"):
        use_adapter.main([str(new), "--name", "x_v1", "--root", str(root)])
    (new / "adapter_config.json").write_text(json.dumps(cfg))
    (new / "adapter_model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(SystemExit, match="LFS pointer"):
        use_adapter.main([str(new), "--name", "x_v1", "--root", str(root)])
    assert "adapters/sql_rft_v3" in (root / "docker/Dockerfile").read_text()
