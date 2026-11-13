"""Nạp các module runtime của upstream (Zaic26-Foxel/datamate) — chỉ import, không sửa.

Repo sinh có package `datamate/` cũ trùng tên; module này đặt upstream lên đầu sys.path và kiểm đường dẫn,
nên mọi file trong eval/sqlv1 phải `from sqlv1.up import ...` trước khi đụng tới datamate.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlv1.paths import SYNTH_ROOT, upstream_root

UPSTREAM = upstream_root()
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
_old = sys.modules.get("datamate")
if _old is not None and not str(getattr(_old, "__file__", "")).startswith(str(UPSTREAM)):
    raise ImportError(f"package datamate đã nạp từ {_old.__file__}, không phải upstream {UPSTREAM}")
for p in (str(SYNTH_ROOT / "eval"),):
    if p not in sys.path:
        sys.path.append(p)  # synth_world, llm, synth_checks: thêm CUỐI để không che upstream

from datamate import debts, fewshot, hints, ingest, prompts, scoring, sqlexec, views  # noqa: E402

assert Path(ingest.__file__).resolve().is_relative_to(UPSTREAM), ingest.__file__


def upstream_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


__all__ = ["debts", "fewshot", "hints", "ingest", "prompts", "scoring", "sqlexec", "views", "UPSTREAM", "SYNTH_ROOT", "upstream_commit"]
