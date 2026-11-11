"""Switch the LoRA adapter of the submission (the one the image copies to /code/models/lora) in one step.

    python3 scripts/use_adapter.py runs/lora/sql_bf16_r16/adapter --name sql_bf16_r16 [--dry-run]

Checks the new adapter against the current one (PEFT LoRA, same rank, alpha and target modules, same base model, real weights and not
a Git LFS pointer, base + adapter <= 15B parameters), copies it to adapters/NAME/, then points every reference of the submission at it:
docker/Dockerfile (COPY), .dockerignore (allow-list), docker/model_manifest.json (path and parameter counts), scripts/run_public.sh and
scripts/run_holdout.sh (default LORA_DIR). scripts/rule_check.py and scripts/pack_training_code.py read the adapter from the manifest.
Switch only after the new adapter scored higher on public than the current one. To go back: run it again with the old adapter folder.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
from rule_check import LIMIT, manifest_adapter, safetensors_params  # noqa: E402

FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md")
REFS = ("docker/Dockerfile", ".dockerignore", "scripts/run_public.sh", "scripts/run_holdout.sh")
SAME = ("peft_type", "r", "lora_alpha", "target_modules")


def check(src: Path, cur: Path) -> int:
    """Parameter count of the new adapter; SystemExit with the reason when it cannot replace the current one."""
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        if not (src / f).is_file():
            raise SystemExit(f"{src} has no {f}: not a PEFT adapter folder")
    new, old = json.loads((src / "adapter_config.json").read_text()), json.loads((cur / "adapter_config.json").read_text())
    diff = [k for k in SAME if new.get(k) != old.get(k)]
    if diff:
        raise SystemExit(f"{src}: {', '.join(f'{k}={new.get(k)!r} (current {old.get(k)!r})' for k in diff)}: vLLM and the manifest expect the current setup")
    base = lambda c: str(c.get("base_model_name_or_path", "")).rstrip("/").split("/")[-1]
    if base(new) != base(old):
        raise SystemExit(f"{src}: trained on {base(new)}, the submission runs {base(old)}")
    n = safetensors_params(src)
    if n is None:
        raise SystemExit(f"{src}/adapter_model.safetensors is a Git LFS pointer or empty: git lfs pull first")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="adapter folder (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--name", required=True, help="folder name under adapters/, e.g. sql_bf16_r16")
    ap.add_argument("--root", type=Path, default=R, help=argparse.SUPPRESS)   # tests run on a copy of the repo
    ap.add_argument("--dry-run", action="store_true", help="check and show what would change, write nothing")
    a = ap.parse_args(argv)
    root = a.root
    if not re.fullmatch(r"[a-z0-9_]+", a.name):
        raise SystemExit("--name: lowercase letters, digits and _ only")
    man_path = root / "docker/model_manifest.json"
    m = json.loads(man_path.read_text(encoding="utf-8"))
    old = manifest_adapter(m)
    if old is None:
        raise SystemExit("docker/model_manifest.json: lora_adapter does not name an adapters/ folder")
    new = f"adapters/{a.name}"
    n = check(a.src, root / old)
    total = m["safetensors_total_parameters"] + n
    if total > LIMIT:
        raise SystemExit(f"base + adapter = {total:,} parameters > {LIMIT:,}")
    dst = root / new
    if dst.resolve() != a.src.resolve() and dst.exists() and new != old:
        raise SystemExit(f"{new} exists: pick another --name or remove it")
    edits = {}
    for rel in REFS:
        text = (root / rel).read_text(encoding="utf-8")
        hits = len(re.findall(re.escape(old) + r"(?![\w])", text))
        if not hits:
            raise SystemExit(f"{rel} does not mention {old}: edit it by hand")
        edits[rel] = (re.sub(re.escape(old) + r"(?![\w])", new, text), hits)
    print(f"{old} -> {new}: {n:,} adapter parameters, base + adapter {total / 1e9:.3f}B (limit 15B)")
    for rel, (_, hits) in edits.items():
        print(f"  {rel}: {hits} reference(s)")
    print(f"  docker/model_manifest.json: lora_adapter, lora_adapter_parameters {m.get('lora_adapter_parameters'):,} -> {n:,}, total {total:,}")
    if a.dry_run:
        print("dry run: nothing written")
        return
    if dst.resolve() != a.src.resolve():
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            if (a.src / f).is_file():
                shutil.copy2(a.src / f, dst / f)
    for rel, (text, _) in edits.items():
        (root / rel).write_text(text, encoding="utf-8")
    m["lora_adapter"] = m["lora_adapter"].replace(old, new, 1)
    m["lora_adapter_parameters"], m["total_parameters_with_adapter"] = n, total
    man_path.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done. Next: git add {new} (weights go through Git LFS), review the 'note' of the manifest, "
          "python3 scripts/rule_check.py, python3 scripts/pack_training_code.py, docker build")


if __name__ == "__main__":
    main()
