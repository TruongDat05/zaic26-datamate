"""Read-only checks of local dependencies and source files; never calls a model API."""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import sys

from sqlv1.paths import SYNTH_ROOT, controller_root, upstream_root

STAGES = ("env", "source", "sql", "export")
RUNTIME_MODULES = ("__init__", "debts", "fewshot", "hints", "ingest", "prompts", "scoring", "sqlexec", "views")


def missing_files(stage: str, root: Path, controller: Path, upstream: Path) -> list[Path]:
    required = []
    if STAGES.index(stage) >= STAGES.index("source"):
        required += [root / "data/train" / name for name in ("questions.jsonl", "answers.jsonl", "calendar.csv", "rulebook.md")]
    if STAGES.index(stage) >= STAGES.index("sql"):
        required += [upstream / "datamate" / f"{name}.py" for name in RUNTIME_MODULES]
    if stage == "export":
        required += [upstream / "datamate/snapshot.py"]
        required += [upstream / "training/make_synthetic_fixture.py"]
        required += [controller / "harness_control" / f"{name}.py" for name in ("evaluate", "exec_guard", "scorer")]
        required += [controller / "training" / f"{name}.py" for name in ("sft_format", "make_synthetic_fixture", "offline_candidates", "build_sft")]
    return [path for path in required if not path.is_file()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="env", help="env, source inputs, SQL runtime, or exporter/controller")
    args = parser.parse_args(argv)
    errors = []
    if sys.version_info < (3, 12):
        errors.append("Python >= 3.12 required")
    for name in ("pandas", "numpy", "openpyxl", "xlrd", "python-calamine", "pyyaml", "openai"):
        try:
            version = metadata.version(name)
            if name == "pandas" and version != "2.2.3":
                errors.append(f"pandas==2.2.3 required, found {version}")
        except metadata.PackageNotFoundError:
            errors.append(f"missing dependency: {name}")
    if args.stage in ("sql", "export"):
        try:
            if metadata.version("duckdb") != "2.0.0":
                errors.append("SQL runtime requires duckdb==2.0.0")
        except metadata.PackageNotFoundError:
            errors.append("missing dependency: duckdb==2.0.0 (use SQL_PY)")
    controller, upstream = controller_root(), upstream_root()
    errors += [f"missing file: {path}" for path in missing_files(args.stage, SYNTH_ROOT, controller, upstream)]
    if args.stage != "env" and not (SYNTH_ROOT / "data/train/data").is_dir():
        errors.append(f"missing shop directory: {SYNTH_ROOT / 'data/train/data'}")
    if args.stage == "export":
        exporter = controller / "training/build_sft.py"
        if exporter.is_file() and "--offline-candidates" not in exporter.read_text(encoding="utf-8"):
            errors.append(f"incompatible exporter: {exporter} has no --offline-candidates contract")
    print(f"stage={args.stage}\nsynth={SYNTH_ROOT}")
    if args.stage in ("sql", "export"):
        print(f"controller={controller}\nruntime={upstream}")
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print("PASS: dependency/file checks only; runtime execution and API access are not tested")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
