"""Khai thác log một lượt chạy (datamate.run --out DIR) thành dữ liệu cho bước sinh: nhóm kết quả từng câu, mẫu đúng
(kể cả hội thoại sửa lỗi), mẫu sai để chẩn đoán. Dùng sau mỗi lượt chạy model (gốc hoặc đã fine-tune).

    python eval/mine_errors.py OUT/collect_ft --name collect_ft
    python eval/mine_errors.py OUT/hard_tune_ft --name hard_tune_ft --data /content/hard_tune --cache /content/hard_tune_cache

Lượt chạy nên dùng: shop mining của train, agent.templates=false, agent.collect_all=true và đường bổ sung (p3, quote, sql, think)
để mỗi câu có nhiều phiếu. --data/--cache phải là bộ dữ liệu của chính lượt chạy (bộ đổi tên cột thì trỏ vào bộ đó), để dựng lại
đúng phản hồi sửa lỗi và prompt.

Đầu ra data/synth_source/errors/<name>/:
- questions.jsonl: mỗi câu một dòng: nhóm (T_ok, T_wrong, E1, E2, E3, E4, E6, E6b, no_gold), nhãn phụ, từng phiếu đúng/sai, tỉ lệ đúng, mã lỗi
- positives.jsonl: mỗi phiếu ra đáp án đúng: các lượt model trả lời và phản hồi sửa lỗi dựng lại đúng như agent đã gửi → mẫu train
  (prompt đầu dựng lại bằng synth_tools.Kit.prompt(q, path) với cùng --data/--cache)
- negatives.jsonl: mỗi phiếu ra đáp án sai: code cuối, đáp án, đáp án chuẩn → chẩn đoán (prompts/5_chan_doan_loi.md), cặp đúng/sai
- summary.json: đếm theo nhóm, nhãn, vai trò, quyết định, nhóm kỹ năng
Chỉ dòng usable_as_seed = true (shop mining) được dùng để sinh; log của tune/holdout chỉ để xem số tổng hợp.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from datamate import formatter, repair, verify  # noqa: E402
import show_trace  # noqa: E402

# Nguyên văn các phản hồi trong datamate/agent.py (Agent._attempt); tests/test_synth_source.py so với agent thật
NO_CODE_MSG = ("Chưa thấy khối ```python``` hoàn chỉnh (có thể câu trả lời bị cắt vì quá dài). "
               "Viết lại đúng một khối code ngắn, ít comment, gán ANSWER.")
ZERO_MSG = ("Kết quả bằng 0, gần như chắc chắn do điều kiện lọc không khớp dòng nào. Kiểm tra: tên trong "
            "câu hỏi so với giá trị thật của cột (dm.match_one, các cột __ten, __ho_ten, __loai), kỳ "
            "thời gian, cột đúng chưa. Nếu shop không ghi thông tin này thì dùng dm.unanswerable(...).")
FORMAT_CODES = {"TYPE_MISMATCH", "TOPK_TOO_LONG", "FORMAT_INVALID"}


def feedback(turn: dict, q: dict, biz, typed_repair: bool) -> str | None:
    """Tin nhắn user mà agent gửi sau lượt này (None nếu lượt này được nhận)."""
    if turn["code"] is None:
        return NO_CODE_MSG
    if turn["ok"]:
        ans = turn["answer"]
        final = verify.round_money(q["question"], formatter.finalize(q["id"], ans))
        if final["type"] == "abstain" and ans.get("type") != "abstain":
            err = f"Đáp án sai format: {json.dumps(ans, ensure_ascii=False)[:300]}"
        else:
            issues = verify.check(q["question"], final)
            if issues:
                err = "Đáp án chạy được nhưng chưa đúng yêu cầu câu hỏi:\n" + "\n".join(f"- {i.message}" for i in issues)
            elif verify.zero_suspect(q["question"], final):
                err = ZERO_MSG
            else:
                return None
    else:
        err = turn["error"] if turn["error"] is not None else "lỗi không rõ"
        if typed_repair:
            kind, tip = repair.classify(err, biz)
            err = f"{err[:2200]}\n\nLoại lỗi: {kind}. Cần sửa: {tip}"
    return f"Code lỗi:\n```\n{err[:2500]}\n```\nSửa code (vẫn một khối ```python```, gán ANSWER)."


def classify(t: dict, gold: dict | None) -> tuple[str, list[str]]:
    """Nhóm kết quả của một câu và nhãn phụ (xem data/synth_source/taxonomy/error_groups.json)."""
    tags = []
    turns = [x for p in t["paths"].values() for x in p["turns"]]
    if any(x["error_code"] in FORMAT_CODES or set(x.get("verify") or []) & FORMAT_CODES for x in turns):
        tags.append("E5_format")
    if t["decision"] == "dropped_type":
        tags.append("dropped_type")
    if gold is None:
        return "no_gold", tags
    answered = [p for p in t["paths"].values() if p["answer"] and p["answer"].get("type") not in (None, "abstain")]
    gu = gold["type"] == "unanswerable"
    if any((p["answer"]["type"] == "unanswerable") != gu for p in answered):
        tags.append("E7_answerability")
    if t["decision"] == "template":
        return ("T_ok" if t["mark"] == 1 else "T_wrong"), tags
    correct = [p for p in answered if p["mark"] == 1]
    if t["mark"] == 1:
        return ("E6" if len(correct) == len(answered) else "E6b"), tags
    if t["mark"] == -1:
        return "E1", tags
    return ("E2" if correct else "E3" if answered else "E4"), tags


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="thư mục --out của datamate.run")
    ap.add_argument("--name", help="tên thư mục đầu ra (mặc định: tên thư mục run)")
    ap.add_argument("--gold", default=str(ROOT / "data/train/answers.jsonl"))
    ap.add_argument("--data", default=str(ROOT / "data/train"), help="bộ dữ liệu của lượt chạy")
    ap.add_argument("--cache", default=str(ROOT / "outputs/cache"))
    ap.add_argument("--source", default=str(ROOT / "data/synth_source"))
    ap.add_argument("--out", help="thư mục đầu ra (mặc định <source>/errors/<name>)")
    a = ap.parse_args(argv)
    run, source = Path(a.run), Path(a.source)
    meta_path = run / "logs" / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    cfg_agent = (meta.get("config") or {}).get("agent") or {}
    seed0, typed = int(cfg_agent.get("seed", 42)), bool(cfg_agent.get("typed_repair", True))
    gold = {g["id"]: g for g in (json.loads(l) for l in Path(a.gold).read_text(encoding="utf-8").splitlines() if l.strip())} if a.gold else {}
    splits = json.loads((source / "splits.json").read_text(encoding="utf-8")) if (source / "splits.json").exists() else {}
    role_of = {b: r for r in ("mining", "tune", "holdout") for b in splits.get(r, [])}
    seeds = {}
    if (source / "seeds/questions.jsonl").exists():
        for l in (source / "seeds/questions.jsonl").read_text(encoding="utf-8").splitlines():
            s = json.loads(l)
            seeds[s["id"]] = s
    data, cache = Path(a.data), Path(a.cache)
    bizs: dict = {}

    def biz_of(b):
        if b not in bizs:
            try:
                from datamate import loader
                bizs[b] = loader.load_cached(data / "data" / b, cache)
            except Exception:  # noqa: BLE001 — không nạp được thì gợi ý tên cột trong phản hồi có thể khác bản gốc
                bizs[b] = None
        return bizs[b]

    qrows, pos, neg = [], [], []
    files = sorted(p for p in (run / "logs").glob("*.json") if p.name != "run_meta.json" and not p.name.startswith("contract_"))
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if "id" not in r or "steps" not in r:
            continue
        g = gold.get(r["id"])
        t = show_trace.build(r, gold, seed0)
        group, tags = classify(t, g)
        b = r.get("business_id")
        role = role_of.get(b, "khac")
        q = {"id": r["id"], "business_id": b, "question": r.get("question"), "as_of_date": r.get("as_of_date")}
        sd = seeds.get(r["id"], {})
        codes = collections.Counter(x["error_code"] for p in t["paths"].values() for x in p["turns"] if x["error_code"])
        rtypes = collections.Counter(x["repair_type"] for p in t["paths"].values() for x in p["turns"] if x.get("repair_type"))
        answered = {n: p for n, p in t["paths"].items() if p["answer"] and p["answer"].get("type") not in (None, "abstain")}
        n_ok = sum(p["mark"] == 1 for p in answered.values())
        base = {"run": run.name, **q, "role": role, "usable_as_seed": role == "mining", "group": group,
                "skill_family": sd.get("skill_family"), "k_tags": sd.get("k_tags", []), "template_on_train": (sd.get("route") or {}).get("template")}
        qrows.append({**base, "tags": tags, "decision": t["decision"], "template": t["template"], "final": t["final"], "mark": t["mark"],
                      "gold": {k: g[k] for k in ("type", "answer", "alias") if k in g} if g else None,
                      "n_paths": len(t["paths"]), "n_answered": len(answered), "n_correct": n_ok,
                      "pass_rate": round(n_ok / len(t["paths"]), 3) if t["paths"] else None,
                      "paths": {n: {"answer": p["answer"], "mark": p["mark"], "n_turns": len(p["turns"])} for n, p in t["paths"].items()},
                      "step_codes": dict(codes), "repair_types": dict(rtypes), "llm_calls": t["llm_calls"], "elapsed_s": t["elapsed_s"]})
        if not g:
            continue
        for name, p in t["paths"].items():
            if not p["turns"] or name not in answered:
                continue
            turns, exact = [], True
            for i, x in enumerate(p["turns"]):
                fb = feedback(x, q, biz_of(b), typed) if i < len(p["turns"]) - 1 else None
                if fb and x.get("repair_type") and f"Loại lỗi: {x['repair_type']}." not in fb:
                    exact = False
                turns.append({"assistant": x["response"], "user_feedback": fb, "error_code": x["error_code"], "repair_type": x.get("repair_type")})
            row = {**base, "path": name, "prompt_ref": {"path": name, "data": str(data), "cache": str(cache)}, "turns": turns,
                   "n_turns": len(turns), "is_repair": len(turns) > 1, "feedback_exact": exact, "answer": p["answer"],
                   "gold": {k: g[k] for k in ("type", "answer", "alias") if k in g}}
            if p["mark"] == 1:
                pos.append(row)
            elif p["mark"] == -1:
                neg.append({**{k: v for k, v in row.items() if k not in ("turns", "is_repair", "feedback_exact")},
                            "last_code": p["turns"][-1]["code"], "codes": [x["error_code"] for x in p["turns"] if x["error_code"]]})

    out = Path(a.out) if a.out else source / "errors" / (a.name or run.name)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("questions", qrows), ("positives", pos), ("negatives", neg)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in rows), encoding="utf-8")
    cnt = lambda key, rows: dict(collections.Counter(key(x) for x in rows).most_common())  # noqa: E731
    summary = {"run": str(run), "code_version": meta.get("code_version"), "n_questions": len(qrows), "n_positives": len(pos), "n_negatives": len(neg),
               "groups": cnt(lambda x: x["group"], qrows), "groups_mining": cnt(lambda x: x["group"], [x for x in qrows if x["usable_as_seed"]]),
               "tags": dict(collections.Counter(t for x in qrows for t in x["tags"])), "roles": cnt(lambda x: x["role"], qrows),
               "decisions": cnt(lambda x: x["decision"], qrows), "skill_family": cnt(lambda x: x["skill_family"], [x for x in qrows if x["group"] in ("E1", "E2", "E3", "E4")]),
               "k_tags_loi": dict(collections.Counter(k for x in qrows if x["group"] in ("E1", "E2", "E3", "E4") for k in x["k_tags"]).most_common()),
               "step_codes": dict(sum((collections.Counter(x["step_codes"]) for x in qrows), collections.Counter())),
               "repair_types": dict(sum((collections.Counter(x["repair_types"]) for x in qrows), collections.Counter())),
               "positives_repair": sum(p["is_repair"] for p in pos), "positives_feedback_not_exact": sum(not p["feedback_exact"] for p in pos)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
