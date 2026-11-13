"""Export independently verified pilot records through the existing inference runtime.

Raw/T4 is not SFT. Writes the five-key training contract only after runtime,
semantic mutations, source-family split, tokenizer and loss-mask checks pass.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from datamate import loader
from datamate.llm import extract_code
from synth_tools import Kit
from synth_pilot import TODAY, digest, implementation_hash, mutations, prepare, reference, render, write_json
from synth_execution import preflight

META_FIELDS = {"schema_version": str, "family": str, "source_id": str, "source_hash": str, "seed": int,
               "split": str, "slot": str, "answer_type": str, "answer_json": str, "difficulty": int,
               "skills": list, "relative_time": bool, "as_of_date": str, "runtime_path": str,
               "candidate": str, "prompt_sha256": str, "code_sha256": str, "tokens": int}


def code_problem(code: str | None, typ: str) -> str | None:
    if not code:
        return "missing_code"
    if issue := preflight(code):
        return issue
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax"
    if any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):
        return "placeholder"
    deps = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = n.value
            names = {x.id for x in ast.walk(value) if isinstance(x, ast.Name)} if value else set()
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "T":
                    return "redefined_input_tables"
                for x in ast.walk(target):
                    if isinstance(x, ast.Name):
                        deps.setdefault(x.id, set()).update(names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            deps.setdefault(n.name, set()).update(x.id for x in ast.walk(n) if isinstance(x, ast.Name))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("exec", "eval", "open", "compile", "__import__"):
            return "external_execution"
    if "ANSWER" not in deps:
        return "missing_ANSWER"
    active = set(deps["ANSWER"])
    while True:
        expanded = active | set().union(*(deps.get(x, set()) for x in active))
        if expanded == active:
            break
        active = expanded
    if typ != "unanswerable" and "T" not in active:
        return "constant_answer"
    return None


def training_rows(path: Path):
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if set(r) != {"id", "kind", "shop", "messages", "meta"} or r["kind"] != "code_synth":
            raise ValueError("invalid SFT outer contract")
        if not all(isinstance(r[k], str) and r[k] for k in ("id", "kind", "shop")) or r["id"] in seen:
            raise ValueError("invalid/duplicate id")
        seen.add(r["id"])
        if [m["role"] for m in r["messages"]] != ["system", "user", "assistant"] or any(set(m) != {"role", "content"} or not isinstance(m["content"], str) or not m["content"] for m in r["messages"]):
            raise ValueError("invalid messages")
        meta = r["meta"]
        if set(meta) != set(META_FIELDS) or any(type(meta[k]) is not t for k, t in META_FIELDS.items()):
            raise ValueError("metadata schema mismatch")
        if any(not isinstance(x, str) for x in meta["skills"]):
            raise ValueError("invalid skills")
        json.loads(meta["answer_json"])
        if code_problem(extract_code(r["messages"][-1]["content"]), meta["answer_type"]):
            raise ValueError("invalid assistant solution")
        yield r


def mutation_kinds(slot):
    return {"gross": ["date_boundary", "join_fanout"], "net": ["return_boundary", "date_boundary"],
            "quantity": ["unit_factor", "join_fanout"], "cancel": ["status_alias", "join_fanout"],
            "top": ["tie", "unit_factor"], "best": ["tie", "unit_factor"], "day": ["date_boundary", "status_alias"],
            "exists": ["status_alias", "no_today"], "table": ["date_boundary", "join_fanout"], "unknown": ["join_fanout"],
            "zero": ["status_alias", "unsold_item"], "average": ["status_alias", "date_boundary", "rounding_boundary"]}[slot]


def semantic_checks(rows, work: Path):
    """Apply the same gold/mutation evaluator to every candidate, including baseline."""
    import model_bakeoff as mb
    from datamate import formatter, verify
    (work / "execution").mkdir(parents=True, exist_ok=True)
    out = []
    for row in rows:
        row = dict(row)
        row["reject"] = list(row["reject"])
        problem = code_problem(row["runtime_code"], row["gold"]["type"])
        if problem:
            row["reject"].append(problem)
        checks = []
        if not row["reject"]:
            f = prepare(work, row["shop"], row["family"], row["seed"])
            kit = Kit(data=f["root"], cache=Path(f["root"]) / "cache")
            if kit.contaminated(row["question"])["hit"] or kit.contaminated(row["question_canonical"])["hit"]:
                row["reject"].append("contaminated")
            for kind, truth in mutations(f["truth"]):
                if kind not in mutation_kinds(row["slot"]):
                    continue
                name = row["shop"] + "_" + kind
                root = work / "mutations" / name
                path = root / "data" / name
                path.mkdir(parents=True, exist_ok=True)
                frames = render(truth, row["family"])
                for k, df in frames.items():
                    df.to_csv(path / (k + ".csv"), index=False)
                (path / "README.md").write_text((Path(f["root"]) / "data" / row["shop"] / "README.md").read_text())
                for file in ("rulebook.md", "calendar.csv"):
                    (root / file).write_text((Path(f["root"]) / file).read_text())
                mk = Kit(data=root, cache=root / "cache")
                gold = reference(truth, row["slot"])
                ex = mk.execute(name, row["runtime_code"], TODAY, row["question"])
                loaded = mk.biz(name)
                pkl = root / "truth.pkl"
                import pickle
                pkl.write_bytes(pickle.dumps(loaded.tables))
                role_results = {}
                for role, code in (("teacher", row["teacher_code"]), ("verifier", row.get("verifier_code"))):
                    if not code:  # old raw records retained, never quietly claim independent mutation coverage
                        role_results[role] = {"pass": False, "error": "missing_saved_code"}
                        continue
                    a, err = mb.run_code(code, pkl, work / "execution", name + role)
                    answer = mb.typed(a, gold["type"], row["question"]) if err is None else None
                    role_results[role] = {"pass": err is None and mk.check(verify.round_money(row["question"], answer), gold) == 1, "error": err, "answer": a}
                ok = ex["ok"] and not ex["issues"] and mk.check(ex["answer"], gold) == 1 and all(x["pass"] for x in role_results.values())
                checks.append({"kind": kind, "gold": gold, "runtime": ex, "independent": role_results, "pass": ok})
            if not all(c["pass"] for c in checks):
                row["reject"].append("semantic_mutation")
        row["mutations"] = checks
        out.append(row)
    return out


def audit_hash(row):
    return digest({k: row[k] for k in ("id", "candidate", "phase", "shop", "family", "seed", "source_hash", "question", "gold", "plan", "teacher_code", "verifier_code", "runtime_code")})


def recheck_for_export(rows, work):
    """Saved pass flags are evidence for comparison, never an export authority."""
    checked = iter(semantic_checks([r for r in rows if not r["reject"]], work))
    result = []
    for row in rows:
        if not row["reject"]:
            row = next(checked)
            kinds = [c["kind"] for c in row["mutations"]]
            if row["reject"] or sorted(kinds) != sorted(mutation_kinds(row["slot"])) or not all(c["pass"] for c in row["mutations"]):
                raise ValueError("fresh semantic mutation replay failed at export")
        result.append(row)
    return result


def require_pilot_coverage(rows):
    """Certify the declared pilot plan, including every fresh final-test slot."""
    from synth_pilot import plan
    slots = {s["slot"] for s in plan()}
    accepted = [r for r in rows if not r["reject"]]
    for family in ("joined", "flat", "events"):
        if {r["slot"] for r in accepted if r["family"] == family} != slots:
            raise ValueError("incomplete pilot family coverage: " + family)
    final = [r for r in rows if r["phase"] == "final"]
    if not final:
        raise ValueError("missing fresh final test")
    for shop in {r["shop"] for r in final}:
        group = [r for r in final if r["shop"] == shop]
        if len(group) != len(slots) or {r["slot"] for r in group} != slots or any(r["reject"] for r in group):
            raise ValueError("incomplete fresh final test: " + shop)


def build(rows, out: Path, tokenizer_python: str, tokenizer: Path, audit: Path):
    t0 = time.monotonic()
    if pd.__version__ != "2.2.3":
        raise ValueError("SFT certification requires pandas 2.2.3")
    if out.exists():
        raise FileExistsError("do not overwrite earlier experiments")
    if len({r["candidate"] for r in rows}) != 1:
        raise ValueError("export must use one selected configuration")
    require_pilot_coverage(rows)
    rows = recheck_for_export(rows, out.parent)
    audit_data = json.loads(audit.read_text())
    audited = {(r["id"], r["content_sha256"]) for r in audit_data["rows"] if r["pass"]}
    kept, seen, split_source, split_shop, rejected = [], set(), {}, {}, []
    for row in rows:
        reasons = list(row["reject"])
        if (row["id"], audit_hash(row)) not in audited:
            reasons.append("manual_audit_required")
        key = digest([row["source_hash"], row["slot"], row["plan"]])
        if key in seen:
            reasons.append("duplicate")
        seen.add(key)
        if reasons:
            rejected.append({"id": row["id"], "reasons": reasons})
            continue
        split = "val" if row["family"] == "flat" else "train"
        old = split_source.setdefault(row["source_hash"], split)
        old_shop = split_shop.setdefault(row["shop"], split)
        if old != split or old_shop != split:
            raise ValueError("source leakage")
        kit = Kit(data=row["data_root"], cache=Path(row["data_root"]) / "cache")
        q = {"id": row["id"], "business_id": row["shop"], "question": row["question"], "as_of_date": row["as_of_date"]}
        messages = kit.prompt(q, "base0")
        # LOCAL ONLY: runtime messages may contain BTC fewshots and are never sent to llm.chat.
        if code_problem(row["runtime_code"], row["gold"]["type"]):
            raise ValueError("invalid code at export")
        import model_bakeoff as mb
        f = prepare(out.parent, row["shop"], row["family"], row["seed"])
        if digest(f["truth"]) != row["source_hash"] or reference(f["truth"], row["slot"]) != row["gold"]:
            raise ValueError("source hash or independent gold conflict")
        for role in ("teacher", "verifier"):
            raw, error = mb.run_code(row[role + "_code"], f["tables"], out.parent / "execution", row["id"] + role)
            if error or not mb.same_answer(raw, row["gold"].get("answer"), row["gold"]["type"], row["question"]):
                raise ValueError("independent replay failed at export: " + role)
        if not row.get("mutations") or not all(c["pass"] for c in row["mutations"]):
            raise ValueError("missing semantic mutation evidence")
        ex = kit.execute(row["shop"], row["runtime_code"], row["as_of_date"], row["question"])
        if not ex["ok"] or ex["issues"] or kit.check(ex["answer"], row["gold"]) != 1:
            raise ValueError("runtime replay failed at export")
        meta = {"schema_version": "1", "family": row["family"], "source_id": row["shop"], "source_hash": row["source_hash"], "seed": row["seed"],
                "split": split, "slot": row["slot"], "answer_type": row["gold"]["type"], "answer_json": json.dumps(row["gold"].get("answer"), ensure_ascii=False),
                "difficulty": row["plan"]["difficulty"], "skills": row["plan"]["skills"], "relative_time": row["plan"]["period"]["relative"],
                "as_of_date": row["as_of_date"], "runtime_path": "base0", "candidate": row["candidate"], "prompt_sha256": digest(messages),
                "code_sha256": digest(row["runtime_code"]), "tokens": 0}
        kept.append({"id": row["id"], "kind": "code_synth", "shop": row["shop"], "messages": messages + [{"role": "assistant", "content": "```python\n" + row["runtime_code"].strip() + "\n```"}], "meta": meta})
    exported_ids = {r["id"] for r in kept}
    require_pilot_coverage([dict(r, reject=r["reject"] or ([] if r["id"] in exported_ids else ["excluded_at_export"])) for r in rows])
    if not kept or not {r["meta"]["split"] for r in kept} == {"train", "val"}:
        raise ValueError("need accepted rows in both family-disjoint splits")
    # Quarantine intermediate rows until all gates pass.
    stage = out.parent / (out.name + "_checks")
    stage.mkdir(parents=True, exist_ok=False)
    provisional = stage / "pending.jsonl"
    provisional.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    subprocess.run([tokenizer_python, str(ROOT / "eval/check_sft_tokens.py"), "--input", str(provisional), "--tokenizer", str(tokenizer), "--out", str(stage / "tokens.json")], check=True, capture_output=True, text=True)
    tok = json.loads((stage / "tokens.json").read_text())
    counts = {r["id"]: r["tokens"] for r in tok["rows"]}
    for r in kept:
        r["meta"]["tokens"] = counts[r["id"]]
    out.mkdir()
    for split in ("train", "val"):
        path = out / (split + ".jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept if r["meta"]["split"] == split))
        assert list(training_rows(path))
    types = {str: "string", int: "integer", bool: "boolean", list: "array"}
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
              "required": ["id", "kind", "shop", "messages", "meta"], "properties": {
                  "id": {"type": "string", "minLength": 1}, "kind": {"const": "code_synth"}, "shop": {"type": "string"},
                  "messages": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False,
                      "required": ["role", "content"], "properties": {"role": {"enum": ["system", "user", "assistant"]}, "content": {"type": "string"}}}},
                  "meta": {"type": "object", "additionalProperties": False, "required": list(META_FIELDS), "properties": {
                      k: {"type": types[t], **({"items": {"type": "string"}} if t is list else {})} for k, t in META_FIELDS.items()}}}}
    write_json(out / "schema.json", schema)
    manifest = {"scope": "pilot", "ready_for_runtime_sft": True, "production_data_ready": False, "rows": len(kept),
                "splits": dict(collections.Counter(r["meta"]["split"] for r in kept)), "shops": sorted({r["shop"] for r in kept}),
                "families": sorted({r["meta"]["family"] for r in kept}), "family_split": {"joined": "train", "events": "train", "flat": "val"},
                "environment": {"python": platform.python_version(), "pandas": pd.__version__}, "tokenizer": tok,
                "implementation_sha256": implementation_hash(), "candidate": rows[0]["candidate"],
                "manual_audit": str(audit), "export_s": time.monotonic() - t0,
                "coverage": {k: dict(collections.Counter(str(r["meta"][k]) for r in kept)) for k in ("answer_type", "difficulty", "relative_time", "family", "slot")},
                "hashes": {p.name: digest(p.read_bytes()) for p in out.iterdir()},
                "limits": ["toy pilot; no stage-1 shops", "Agent default drop_types excludes unanswerable; sandbox contract supports it", "no fine-tuning or model accuracy evaluation", "automatic TRL assistant_only_loss not certified; use prefix mask checked here"]}
    write_json(out / "manifest.json", manifest)
    write_json(stage / "rejected.json", rejected)
    (out / "README.md").write_text("# Pilot SFT Qwen3-14B\n\nChỉ dữ liệu đồ chơi tổng hợp, không phải corpus chính thức. "
                                    "Messages dựng cục bộ bằng Kit.prompt(base0), có thể chứa few-shot của repo: không gửi ra API. "
                                    "Python ANSWER đã replay bằng pandas 2.2.3, formatter/score và mutation oracle Python độc lập.\n\n"
                                    "Chia theo family nguồn trước biến thể: joined/events train, flat val. Tokenizer Qwen3-14B, max_length=32768, "
                                    "enable_thinking=False, không truncation. Dùng collator prefix-masked của eval/check_sft_tokens.py; "
                                    "không bật assistant_only_loss tự động nếu chưa dùng template hỗ trợ generation mask. "
                                    "Pipeline inference mặc định drop_types=[unanswerable] và có zero_suspect; đây là chứng nhận code SFT/sandbox, không chứng nhận Agent chọn nộp mọi đáp án.\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--tokenizer-python", required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = [r for p in a.rows for r in json.loads(p.read_text())]
    print(json.dumps({k: v for k, v in build(rows, a.out, a.tokenizer_python, a.tokenizer, a.audit).items() if k in ("scope", "rows", "splits", "export_s")}))


if __name__ == "__main__":
    main()
