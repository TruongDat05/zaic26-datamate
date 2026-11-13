"""Bundle offline_candidates/1 cho exporter SFT v2 (upstream training/build_sft.py --offline-candidates): ứng viên của gói +
bằng chứng gốc. Không gọi API, không sửa ứng viên.

    DATAMATE_UPSTREAM=<snapshot> python eval/sqlv1/bundle.py --snapshot <snapshot> [--out build/synthetic_sql_v1/offline_bundle]

Bằng chứng mỗi ứng viên:
- lời giải model: bản ghi request đã cache (outputs/synth_sql_v1/api/{solve,repair}/<shop>/<key>.json: messages, settings, response)
  mà response (qua parse_answers + completion của solve.py) cho ĐÚNG completion của ứng viên; dòng ledger
  outputs/synth_sql_v1/requests.sqlite có cùng prompt_sha (sha256 json.dumps(messages)), provider, model và usage. Lời giải sửa
  kèm request lượt đầu (cha) mà request sửa trích lại.
- SQL mẫu: dòng sự kiện trong solutions/pool_*.template.jsonl (file, số dòng, sha256 dòng) + chạy lại generator
  (sqlgen.for_spec) ở revision hiện tại và so từng byte.
QA của bộ sinh (loại trước, ghi vào excluded.jsonl): SQL phân loại trạng thái bằng chuỗi con một từ ('%huy%', '%tra%'...),
mẫu mà lỗi prompt upstream đã gợi ý; không tìm được bằng chứng.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

FOXEL = controller_root()


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--pkg", type=Path, default=FOXEL / "build" / "synthetic_sql_v1")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--exclude", type=Path, help="producer QA: JSON list of {question_id, reason, detail}; every candidate of those questions -> excluded.jsonl")
    ap.add_argument("--exclude-candidates", type=Path, help="producer QA per candidate: JSON list of {candidate_id, reason, detail} (eval/sqlv1/clean_q35.py)")
    ap.add_argument("--mutation-report", type=Path, help="default: sft_v2/mutation_report_probe.jsonl (generic + targeted mutants) if present")
    return ap.parse_args()


A = _args() if __name__ == "__main__" else None
if A is not None:
    os.environ["DATAMATE_UPSTREAM"] = str(A.snapshot.resolve())
from sqlv1 import render as RD, shops as SH, solve as SV, sqlgen as G  # noqa: E402
from sqlv1.mutation_replay import OUT as MOUT, candidate_id, load_candidates  # noqa: E402

sys.path.append(str(FOXEL / "training"))
import sft_format as F  # noqa: E402
from make_synthetic_fixture import content_hash  # noqa: E402
import offline_candidates as OC  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
LIT = re.compile(r"(?i)(?:i?like|similar\s+to)\s*'([^']*)'")
UNSAFE = {"huy", "huỷ", "hủy", "tra", "trả"}


def unsafe_status_predicate(sql: str) -> list[str]:
    """Mẫu LIKE '%<một từ trạng thái>%': khớp cả chuỗi con ('huy' trong 'vận chuyển', 'tra' trong 'kiểm tra', 'trả bằng thẻ')."""
    return sorted({p for p in LIT.findall(sql) if p.startswith("%") and p.strip("%").strip().lower() in UNSAFE})


_CODES = {}
# rulebook_api.md breaks ties of dates/months by 'earlier first'; the competition rulebook the runtime prompt carries has no such rule
# (checked locally: no sentence with a tie word and 'sớm'). A trace citing it would teach the model to cite a rule it is not shown.
ABSENT_RULE = re.compile(r"(?im)^(?=[^\n]*(?:\btie|bằng nhau|rulebook|\brule\b))[^\n]*(?:sớm hơn|\bearlier\b)[^\n]*$")  # a tie/rule line that says 'earlier'


# section headings of rulebook_api.md (and 'phần Khách', the cross-reference think.py wrote for 'mục B') and the generator's templates:
# the runtime rulebook has other headings, so a trace naming them names a document the model is not shown
GENERATOR_REF = re.compile("|".join([re.escape(h) for h in ("Đơn và doanh thu", "Số lượng, đơn vị", "Kỳ thời gian", "Làm tròn, định dạng",
                                                              "Xếp hạng và liệt kê", "Nhập hàng, giá, kho", "Không trả lời được",
                                                              "Cách ghi trong file", "phần Khách")] + [r"(?i:\btemplates?\b)"]))
# DISTINCT over v_sales keeping qty but neither amount nor a line id: merges two real lines of one order (independent audit, q001892)
PARTIAL_DISTINCT = re.compile(r'(?is)select\s+distinct\s+((?:(?!\bfrom\b).){0,300}?)\s+from\s+"?v_sales"?\b')


def partial_line_distinct(completion: str) -> bool:
    sql = completion.split("```sql")[-1]
    return any("qty" in m.group(1).lower() and "amount" not in m.group(1).lower() and "line" not in m.group(1).lower()
               for m in PARTIAL_DISTINCT.finditer(sql))


def price_question_says_revenue(question: str, slot: str) -> bool:
    """A price question worded with 'doanh thu' (dthu): read as revenue it has another answer (independent audit, q000522)."""
    import unicodedata
    f = "".join(ch for ch in unicodedata.normalize("NFD", question.lower().replace("đ", "d")) if unicodedata.category(ch) != "Mn")
    return slot in ("price_on_day", "price_change_pct") and bool(re.search(r"\b(dthu|doanh thu|doanh so|dso)\b", f))


def item_code_literals(c: dict) -> list[str]:
    """Item codes of the shop written as SQL literals that the question does not name. The runtime prompt never pairs an item's
    name with its code (VALUES MATCHING lists names, codes only sit in schema value lists), so such a literal is a guess the model
    would learn. Templates look items up by name (sqlgen._item_in)."""
    b = c["business_id"]
    if b not in _CODES:
        import gzip
        import pickle
        rec = pickle.load(gzip.open(WORK / "shops" / b / "rec.pkl.gz"))
        _CODES[b] = {it["item"] for it in rec["items"]}
    sql = c["completion"].split("</think>")[-1].split("```sql")[-1]
    q = c.get("_question", "")
    return sorted({x for x in re.findall(r"'([^']*)'", sql) if x in _CODES[b] and x not in q})


def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(SH.SYNTH_ROOT), *args], text=True).strip()


def prompt_sha(msgs) -> str:
    return hashlib.sha256(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()


class ApiIndex:
    """Bản ghi request theo (vai trò, shop, provider) + dòng ledger khớp từng bản ghi."""

    def __init__(self):
        self.by = collections.defaultdict(list)
        self.db = sqlite3.connect(f"file:{WORK / 'requests.sqlite'}?mode=ro", uri=True)
        for role in ("solve", "repair", "think"):
            for f in sorted((WORK / "api" / role).glob("*/*.json")):
                rec = json.loads(f.read_text(encoding="utf-8"))
                self.by[(role, rec["shop"], rec["settings"]["provider"])].append((f, rec))
        self._items = {}

    def items(self, f, rec):
        if f not in self._items:
            self._items[f] = SV.parse_answers(rec["response"].get("content") or "")
        return self._items[f]

    def ledger(self, rec):
        rows = self.db.execute("SELECT id, ts, provider, requested_model, response_model, status, usage FROM requests WHERE prompt_sha = ? AND status = 'success' "
                               "ORDER BY id", (prompt_sha(rec["messages"]),)).fetchall()
        usage = rec["response"].get("usage") or {}
        ok = [r for r in rows if r[2] == rec["settings"]["provider"] and r[3] == rec["settings"]["model"] and json.loads(r[6] or "{}") == usage]
        return ok[0][0] if ok else None, len(rows)


def find_model_evidence(ix: ApiIndex, c: dict) -> tuple[dict | None, str]:
    m = c["metadata"]
    k, prov, b = m["question_key"], m["provider"], c["business_id"]
    mark = f"### id: {k}\n"
    user = lambda rec: "\n".join(x["content"] for x in rec["messages"] if x["role"] == "user")  # noqa: E731
    firsts = [(f, rec) for f, rec in ix.by[("solve", b, prov)] if mark in user(rec)]
    if m["attempt"] == "thinking":  # one question per request: reasoning + runtime-format answer -> OC.THINK_TRANSFORM
        hits, parent = [], None
        for f, rec in ix.by[("think", b, prov)]:
            resp = rec["response"]
            if mark not in user(rec) or resp.get("finish") != "stop" or OC.think_problem(resp.get("reasoning")):
                continue
            it = OC.runtime_answer(resp.get("content"))
            if it and it["sql"] and it["type"] and F.canonical_thinking(resp["reasoning"].strip(), OC.layout(it["type"], it["sql"], it["notes"])) == c["completion"]:
                hits.append((f, rec))
    elif m["attempt"] == "first":
        hits = [(f, rec) for f, rec in firsts if (it := ix.items(f, rec).get(k)) and it["sql"] and it["type"] and SV.completion(it) == c["completion"]]
        parent = None
    else:
        hits, parent = [], None
        for f, rec in ix.by[("repair", b, prov)]:
            if mark not in user(rec) or not (it := ix.items(f, rec).get(k)) or not it["sql"]:
                continue
            for pf, prec in firsts:
                pit = ix.items(pf, prec).get(k)
                if not pit or f"YOUR ANSWER:\nTYPE: {pit['type']}\n```sql\n{pit['sql']}\n```" not in user(rec):
                    continue
                it2 = dict(it) if it["notes"] else {**it, "notes": pit["notes"]}
                if SV.completion(it2) == c["completion"]:
                    hits.append((f, rec))
                    parent = (pf, prec)
                    break
    if not hits:
        return None, "no cached request whose reply derives this completion"
    hits.sort(key=lambda fr: fr[1].get("ts") or "")
    f, rec = hits[0]
    lid, n = ix.ledger(rec)
    if lid is None:
        return None, f"no ledger row with this prompt_sha/provider/model/usage ({n} rows share the prompt_sha)"
    ev = {"request_file": f, "ledger_id": lid, "item_id": k, "n_matching_requests": len(hits)}
    if parent:
        plid, _ = ix.ledger(parent[1])
        if plid is None:
            return None, "the first attempt of this repair has no ledger row"
        ev["parent"] = {"request_file": parent[0], "ledger_id": plid}
    return ev, ""


def template_events(cands: list[dict], sel: dict, head: str) -> tuple[dict, list[dict]]:
    """{candidate_id: event} của SQL mẫu: dòng gốc trong solutions/*.template.jsonl + chạy lại generator, so từng byte."""
    lines = collections.defaultdict(list)
    for f in sorted((WORK / "solutions").glob("pool_*.template.jsonl")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            x = json.loads(line)
            if x.get("ok"):
                lines[(x["qid"], x["completion"])].append((f, n, line))
    gen_sha = F.file_hash(Path(G.__file__))
    cache, out, miss = {}, {}, []
    for c in cands:
        if c["metadata"]["solution_source"] != "template":
            continue
        k, b = c["metadata"]["question_key"], c["business_id"]
        src = lines.get((k, c["completion"]))
        if not src:
            miss.append({"candidate_id": candidate_id(c), "question_id": c["id"], "reason": "evidence_not_found", "detail": "no template event line"})
            continue
        if b not in cache:
            info, rec = SH.load_shop(WORK, b)
            cache[b] = (info, rec, RD.raw_info(info["roles"], RD.relations(info["layers"]["N"])))
        info, rec, raw = cache[b]
        spec = sel[c["id"]]["spec"]
        try:
            kind, sql, notes = G.for_spec(spec, rec, raw)
            regen = SV.completion({"type": kind, "sql": sql, "notes": notes})
        except Exception as e:  # noqa: BLE001
            regen = f"error: {e}"
        f, n, line = src[0]
        out[candidate_id(c)] = {"event_id": hashlib.sha256(f"{f.name}:{n}:{line}".encode()).hexdigest()[:32], "item_id": k, "business_id": b,
                                "completion": c["completion"], "seed": info["plan"]["seed"], "file": f"outputs/synth_sql_v1/solutions/{f.name}", "line": n,
                                "line_sha256": hashlib.sha256(line.encode()).hexdigest(), "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                                "regenerated_equal": regen == c["completion"],
                                "generator": {"module": "eval/sqlv1/sqlgen.py (for_spec) + eval/sqlv1/solve.py (completion)", "module_sha256": gen_sha,
                                              "commit": head, "repo": "zaic26-datamate-synth"}}
    return out, miss


def main(a):
    pkg, out = a.pkg, (a.out or a.pkg / "offline_bundle")
    if out.exists():
        shutil.rmtree(out)
    (out / "evidence" / "api").mkdir(parents=True)
    head = git("rev-parse", "HEAD")
    data_root = pkg / "data_root"
    man = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    cands = load_candidates(pkg)
    sel = {x["id"]: x for x in map(json.loads, (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines()) if x.get("id")}
    ix = ApiIndex()
    tev, excluded = template_events(cands, sel, head)
    keep, used_files, stats = [], {}, collections.Counter()
    qa = {x["question_id"]: x for x in json.loads(a.exclude.read_text(encoding="utf-8"))} if a.exclude else {}
    qc = {x["candidate_id"]: x for x in json.loads(a.exclude_candidates.read_text(encoding="utf-8"))} if a.exclude_candidates else {}
    for c in cands:
        cid = candidate_id(c)
        if cid in qc:
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "clean:" + qc[cid]["reason"], "detail": qc[cid].get("detail")})
            stats["excluded:clean:" + qc[cid]["reason"]] += 1
            continue
        if price_question_says_revenue(sel[c["id"]]["question"], sel[c["id"]]["spec"].get("slot")):
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "ambiguous_price_question_says_revenue"})
            stats["excluded:ambiguous_price_question_says_revenue"] += 1
            continue
        if partial_line_distinct(c["completion"]):
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "distinct_merges_sale_lines"})
            stats["excluded:distinct_merges_sale_lines"] += 1
            continue
        codes = item_code_literals({**c, "_question": sel[c["id"]]["question"]})
        if c["metadata"].get("attempt") == "thinking":
            item_code_literals({**c, "_question": ""})  # loads the shop's codes
            reasoning = c["completion"].split("</think>")[0]
            guessed = sorted(k for k in _CODES[c["business_id"]] if re.search(r"\b" + re.escape(k) + r"\b", reasoning) and k not in sel[c["id"]]["question"])
            if GENERATOR_REF.search(reasoning):
                excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "thinking_names_generator_document",
                                 "detail": GENERATOR_REF.search(reasoning).group(0)[:60]})
                stats["excluded:thinking_names_generator_document"] += 1
                continue
            if ABSENT_RULE.search(reasoning):  # a generator-rulebook rule the runtime rulebook does not have: a cited rule that is not there
                excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "thinking_cites_rule_absent_at_runtime",
                                 "detail": ABSENT_RULE.search(reasoning).group(0)[:80]})
                stats["excluded:thinking_cites_rule_absent_at_runtime"] += 1
                continue
            if guessed:  # the trace pairs names with codes the prompt never pairs (value lists only): a guessing habit, even if the SQL is right
                excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "thinking_guesses_item_code", "detail": ", ".join(guessed[:6])})
                stats["excluded:thinking_guesses_item_code"] += 1
                continue
        if codes:
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "item_code_literal_not_in_prompt", "detail": ", ".join(codes)})
            stats["excluded:item_code_literal_not_in_prompt"] += 1
            continue
        if c["id"] in qa:
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": qa[c["id"]]["reason"], "detail": qa[c["id"]].get("detail")})
            stats["excluded:" + qa[c["id"]]["reason"]] += 1
            continue
        sql = c["completion"].split("```sql")[-1]
        bad = unsafe_status_predicate(sql)
        if bad:
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "unsafe_status_substring_predicate", "detail": ", ".join(bad)})
            stats["excluded:unsafe_status_substring_predicate"] += 1
            continue
        base = {"candidate_id": cid, "question_id": c["id"], "business_id": c["business_id"], "completion": c["completion"],
                "metadata": c["metadata"]}  # the package's own check claims: listed, ignored by the importer
        if c["metadata"]["solution_source"] == "template":
            if cid not in tev:
                continue
            keep.append({**base, "origin": "program_template", "attempt": "template", "evidence": {"event_id": tev[cid]["event_id"], "item_id": tev[cid]["item_id"]}})
            stats["template"] += 1
            continue
        ev, why = find_model_evidence(ix, c)
        if ev is None:
            excluded.append({"candidate_id": cid, "question_id": c["id"], "reason": "evidence_not_found", "detail": why})
            stats["excluded:evidence_not_found"] += 1
            continue
        stats["multi_request_match"] += ev.pop("n_matching_requests") > 1
        for ref in [ev] + ([ev["parent"]] if ev.get("parent") else []):
            f = ref["request_file"]
            rel = "evidence/api/" + f.relative_to(WORK / "api").as_posix()
            used_files[rel] = f
            ref["request_file"] = rel
        keep.append({**base, "origin": "model_api", "attempt": c["metadata"]["attempt"], "evidence": ev})
        stats["model:" + c["metadata"]["attempt"]] += 1
    for rel, f in used_files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out / rel)
    src = sqlite3.connect(f"file:{WORK / 'requests.sqlite'}?mode=ro", uri=True)
    dst = sqlite3.connect(out / "evidence" / "requests.sqlite")
    src.backup(dst)
    dst.close()
    src.close()
    layers = {}
    for b in sorted({c["business_id"] for c in keep}):
        info = json.loads((WORK / "shops" / b / "shop.json").read_text(encoding="utf-8"))
        L, (n, k) = info["layers"], info["k_agree"]
        layers[b] = {"canon_names": True, "views_sql": L["view_sql"] if L["view_ok"] else None, "debt_sqls": L["debt_sqls"] if L["debt_ok"] else None,
                     "views_agree": n, "views_k": k, "debt_agree": n, "debt_k": k}
    mrep = a.mutation_report or next(p for p in (MOUT / "mutation_report_probe.jsonl", MOUT / "mutation_report.jsonl") if p.exists())
    base = MOUT / "mutation_report.jsonl"
    if mrep.name == "mutation_report_probe.jsonl" and base.exists() and base.stat().st_mtime > mrep.stat().st_mtime:
        raise SystemExit("mutation_report_probe.jsonl is older than mutation_report.jsonl: rerun eval/sqlv1/probe.py")
    shutil.copy2(mrep, out / "mutation_report.jsonl")
    for name, rows in (("candidates.jsonl", keep), ("excluded.jsonl", excluded), ("evidence/template_events.jsonl", list(tev.values()))):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "layers.json").write_text(json.dumps(layers, ensure_ascii=False), encoding="utf-8")
    files = {p.relative_to(out).as_posix(): F.file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    bundle = {"kind": OC.BUNDLE_KIND, "contract": OC.CONTRACT, "dataset_id": man["dataset_id"], "data_hash": content_hash(data_root),
              "transform": OC.TRANSFORM, "ledger_name": "synth_sql_v1", "candidates": "candidates.jsonl", "layers": "layers.json",
              "mutation_report": "mutation_report.jsonl", "ledger": "evidence/requests.sqlite", "template_events": "evidence/template_events.jsonl",
              "excluded": "excluded.jsonl",
              **({"thinking_transform": OC.THINK_TRANSFORM} if any(k["attempt"] == "thinking" for k in keep) else {}),
              "producer": {"repo": "zaic26-datamate-synth", "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": head,
                           "scripts": {p: F.file_hash(SH.SYNTH_ROOT / p) for p in ("eval/sqlv1/bundle.py", "eval/sqlv1/mutation_replay.py", "eval/sqlv1/solve.py",
                                                                                     "eval/sqlv1/sqlgen.py", "eval/llm.py", "eval/sqlv1/think.py")
                                       if (SH.SYNTH_ROOT / p).exists()},
                           "ledger_source": str(WORK / "requests.sqlite"), "ledger_source_sha256": F.file_hash(WORK / "requests.sqlite"),
                           "mutation_report_source": str(mrep), "created": dt.datetime.now().isoformat(timespec="seconds"),
                           **({"exclusions": str(a.exclude), "exclusions_sha256": F.file_hash(a.exclude)} if a.exclude else {}),
                           **({"candidate_exclusions": str(a.exclude_candidates), "candidate_exclusions_sha256": F.file_hash(a.exclude_candidates)}
                              if a.exclude_candidates else {})},
              "files": files}
    (out / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    summ = {"out": str(out), "candidates_in": len(cands), "candidates_bundled": len(keep), "excluded": dict(collections.Counter(x["reason"] for x in excluded)),
            "stats": dict(stats), "evidence_files": len(files), "bundle_json_sha256": F.file_hash(out / "bundle.json")}
    (MOUT / "bundle_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(A)
