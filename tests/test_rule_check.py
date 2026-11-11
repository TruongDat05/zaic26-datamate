"""scripts/rule_check.py: what it treats as inference code, how it reads strings, parameter counts, and a clean run on this repo."""
import json
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import rule_check  # noqa: E402


def test_inference_modules_follow_predict_imports():
    names = {p.stem for p in rule_check.inference_modules()}
    assert {"predict", "pipeline", "llm", "ingest", "views", "debts", "voting", "sqlexec"} <= names
    assert not names & {"scoring", "notebook", "values"}  # evaluation and notebook helpers are not imported by inference


def test_strings_tell_docstrings_cli_and_code_apart(tmp_path):
    f = tmp_path / "m.py"
    f.write_text('"""Module doc biz_101."""\nimport argparse\n\n\ndef g():\n    """Func doc."""\n    ap = argparse.ArgumentParser()\n'
                 '    ap.add_argument("--q", default="data/public_test/q.jsonl", help="pro-rata")\n    return "mik dev_002"\n', encoding="utf-8")
    kinds = {s: k for _, s, k in rule_check.strings(f)}
    assert kinds["Module doc biz_101."] == "doc" and kinds["Func doc."] == "doc"
    assert kinds["data/public_test/q.jsonl"] == "cli" and kinds["pro-rata"] == "cli"
    assert kinds["mik dev_002"] == "code"


def test_safetensors_params_counts_shapes_and_spots_lfs_pointers(tmp_path):
    head = json.dumps({"__metadata__": {}, "a": {"dtype": "BF16", "shape": [3, 4], "data_offsets": [0, 24]},
                       "b": {"dtype": "BF16", "shape": [5], "data_offsets": [24, 34]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + b"\0" * 34)
    assert rule_check.safetensors_params(tmp_path) == 17
    ptr = tmp_path / "lfs"
    ptr.mkdir()
    (ptr / "adapter_model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    assert rule_check.safetensors_params(ptr) is None


def test_repo_has_no_rule_error_without_data(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts/rule_check.py"), "--data", str(tmp_path), "--json", str(tmp_path / "out.json")],
                       capture_output=True, text=True)
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert r.returncode == 0 and out["LỖI"] == [], out["LỖI"]
    assert any("chưa kiểm tra ngân hàng ví dụ" in x for x in out["CẦN XEM"])  # rule 6 says it skipped the data checks
