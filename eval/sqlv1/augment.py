"""Bổ sung (chỉ NỐI THÊM, không đổi dòng cũ) vào gói syn_sales_sql_v1 hai mảng còn mỏng so với BTC train. Không gọi API.

    DATAMATE_UPSTREAM=<snapshot> python eval/sqlv1/augment.py [--unans 60] [--long-train 100] [--long-dev 10]

1. Câu "cửa hàng không ghi thông tin này" (unanswerable): lấy từ kho câu đã sinh (teacher viết, verifier khác họ kiểm, solver API giải
   và đã qua kiểm cục bộ) nhưng chưa được chọn: bước chọn cũ phạt chúng vì nhãn kỹ năng của câu unanswerable BTC phần lớn để trống.
2. Câu hỏi giá thay đổi bao nhiêu phần trăm giữa hai ngày, viết DÀI như họ câu dài của BTC (đo gộp trên BTC: 84 câu, trung vị 36 từ,
   có dấu, hai ngày d/m, tăng/giảm, lời nhờ; đáp án khác 0): đặc tả mới từ specs.sample (gold = oracle), câu chữ từ template viết tay
   (nguồn 'template_text'), SQL từ sqlgen.for_spec (SQL mẫu tất định, ghi sự kiện generator để exporter kiểm tái tạo).

Ghi: data_root/{questions,answers}.jsonl (+ thư mục shop nếu thiếu), candidates/solutions.jsonl, data_root/manifest.json,
outputs/synth_sql_v1/selected.jsonl (nối), questions/aug_long.jsonl, solutions/pool_aug.template.jsonl, sft_v2/augment_summary.json.
Bản cũ của mọi file bị sửa được giữ cạnh nó với hậu tố .pre_augment.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=controller_root() / "build/sft_v2/snapshot_8b07ba9")
    ap.add_argument("--pkg", type=Path, default=controller_root() / "build/synthetic_sql_v1")
    ap.add_argument("--unans", type=int, default=60)
    ap.add_argument("--long-train", type=int, default=100)
    ap.add_argument("--long-dev", type=int, default=10)
    return ap.parse_args()


A = _args() if __name__ == "__main__" else None
if A is not None:
    os.environ["DATAMATE_UPSTREAM"] = str(A.snapshot.resolve())
from sqlv1 import package as PK, render as RD, shops as SH, solve as SV, specs as SP, sqlgen as G, verify as V, world as WB  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
# Họ câu dài của BTC (chỉ số gộp): giá giữa hai ngày tăng/giảm bao nhiêu phần trăm, nói dài, có lời nhờ. Câu chữ viết mới, không chép BTC.
LONG = [
    "Cho mình hỏi giá của {item} vào ngày {d2} so với giá của chính sản phẩm đó vào ngày {d1} thì đã tăng hay giảm bao nhiêu phần trăm vậy, bạn tính giúp mình với nhé",
    "Chị đang rà lại bảng giá của cửa hàng, nhờ em xem giúp chị giá {item} ngày {d2} đã tăng hay giảm bao nhiêu phần trăm so với ngày {d1} để chị điều chỉnh lại cho phù hợp nhé",
    "Em ơi, từ ngày {d1} đến ngày {d2} thì giá của {item} bên mình đã thay đổi bao nhiêu phần trăm vậy, nếu giá giảm thì em ghi số âm giúp anh với nha",
    "Anh muốn biết giá {item} ngày {d2} tăng bao nhiêu phần trăm so với giá ngày {d1}, nếu có giảm thì cũng báo anh con số phần trăm luôn, em kiểm tra trong sổ bán hàng giúp anh với",
    "Chị cần so sánh giá của {item} giữa ngày {d1} và ngày {d2}, em tính giúp chị xem tới ngày sau thì giá đã tăng hay giảm đi bao nhiêu phần trăm nhé, chị cảm ơn em nhiều",
    "Nhờ bạn kiểm tra giúp mình là giá {item} hôm {d2} so với hôm {d1} thì tăng hay giảm, và tăng hoặc giảm bao nhiêu phần trăm vậy, mình cần gấp để chốt giá với khách",
    "Mấy hôm nay khách hỏi giá {item} nhiều quá, em xem giúp anh giá ngày {d2} đã tăng hay giảm bao nhiêu phần trăm so với ngày {d1} để anh còn trả lời khách nha em",
    "Em xem giúp chị với, giá của {item} ngày {d1} là một mức, sang ngày {d2} thì giá đã tăng hay giảm bao nhiêu phần trăm so với lúc trước vậy em",
    "Bạn ơi cho mình hỏi giá {item} ngày {d2} so với ngày {d1} đã tăng thêm hay giảm bớt bao nhiêu phần trăm vậy, bạn tính kỹ giúp mình rồi báo lại nhé, cảm ơn bạn",
    "Cho chị hỏi chút xíu, so với ngày {d1} thì đến ngày {d2} giá bán của {item} ở tiệm mình đã tăng hay giảm bao nhiêu phần trăm rồi em, em tính giúp chị nhé",
]


def dmy(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{int(d)}/{int(m)}/{y}"


def jl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def keep_copy(p: Path):
    b = p.with_name(p.name + ".pre_augment")
    if p.exists() and not b.exists():
        shutil.copy2(p, b)


def long_text(spec: dict, rec: dict, rng: random.Random) -> str:
    name = next(it["name"] for it in rec["items"] if it["item"] == spec["params"]["item"])
    t = rng.choice(LONG).format(item=name, d1=dmy(spec["period"]["d1"]), d2=dmy(spec["period"]["d2"]))
    if rng.random() < 0.46:
        t = t[:1].lower() + t[1:]
    q = rng.random() < 0.49 and not t.endswith(("cảm ơn em nhiều", "cảm ơn bạn"))
    return t + ("?" if q else "")


def main(a):
    pkg, root = a.pkg, a.pkg / "data_root"
    rng = random.Random(2026_11_11)
    qs = jl(root / "questions.jsonl")
    sel = jl(WORK / "selected.jsonl")
    plan = {s["business_id"]: s for s in json.loads((WORK / "plan.json").read_text(encoding="utf-8"))}
    have_q = {x["qid"] for x in sel}
    texts = {PK.fold(x["question"]) for x in sel}
    shops_in = {q["business_id"] for q in qs}
    next_id = max(int(q["id"].split(":q")[1]) for q in qs if int(q["id"].split(":q")[1]) < 900000) + 1
    new_sel, new_cands, events, longq = [], [], [], []

    # ---- 1. unanswerable từ kho đã kiểm
    sols = collections.defaultdict(list)
    for f in sorted((WORK / "solutions").glob("pool_*.jsonl")):
        for x in jl(f):
            sols[x["qid"]].append(x)
    pool = jl(WORK / "questions" / "pool_train.jsonl") + jl(WORK / "questions" / "pool_train2.jsonl")
    un = [q for q in pool if q["gold"]["type"] == "unanswerable" and q["qid"] not in have_q and q["split"] == "train" and q["business_id"] in shops_in
          and not V.question_ok(q) and PK.fold(q["question"]) not in texts]
    rng.shuffle(un)
    per_shop, picked = collections.Counter(), []
    for q in sorted(un, key=lambda q: (q["industry"] not in WB.CORE, rng.random())):
        best = PK.best_solutions([s for s in sols.get(q["qid"], []) if s.get("ok")])
        if not best or per_shop[q["business_id"]] >= 2 or len(picked) >= a.unans:
            continue
        per_shop[q["business_id"]] += 1
        texts.add(PK.fold(q["question"]))
        picked.append((q, best))
    for q, best in picked:
        q = dict(q)
        q["id"] = f"syn_sales_v1:q{next_id:06d}"
        next_id += 1
        q["augment"] = "pool_unanswerable_unselected"
        new_sel.append(q)
        for s in best:
            new_cands.append((q, s))

    # ---- 2. câu giá thay đổi % viết dài (đặc tả mới, gold = oracle, SQL mẫu)
    for split, n_want in (("train", a.long_train), ("dev", a.long_dev)):
        shops = [b for b in sorted(shops_in) if plan[b]["split"] == split and plan[b]["family"] not in SP.NO_PRICE_FAMILIES]
        shops.sort(key=lambda b: (plan[b]["industry"] not in WB.CORE, rng.random()))
        made, sign = 0, collections.Counter()
        for rnd in range(4):
            for b in shops:
                if made >= n_want:
                    break
                info, rec = SH.load_shop(WORK, b)
                specs = SP.sample(rec, plan[b]["family"], 10, f"aug_long:{rnd}", force_slots=["price_change_pct"])
                specs = [s for s in specs if s["gold"]["type"] == "percent" and s["gold"]["answer"] != 0]
                want_pos = sign["pos"] <= sign["neg"]
                specs.sort(key=lambda s: ((s["gold"]["answer"] > 0) != want_pos, rng.random()))
                if not specs:
                    continue
                spec = specs[0]
                gold = spec.pop("gold")
                raw = RD.raw_info(info["roles"], RD.relations(info["layers"]["N"]))
                try:
                    kind, sql, notes = G.for_spec(spec, rec, raw)
                except G.Unsupported:
                    continue
                text = long_text(spec, rec, rng)
                if PK.fold(text) in texts:
                    continue
                texts.add(PK.fold(text))
                qid = f"aug:{b}:price_change_long:{rnd}"
                q = {"key": qid, "origin": "template_text", "teacher": None, "verifier": None, "split": split, "business_id": b, "industry": plan[b]["industry"],
                     "family": plan[b]["family"], "question": text, "as_of_date": spec["as_of"], "spec": spec, "gold": gold, "style": {}, "qid": qid,
                     "id": f"syn_sales_v1:q{next_id:06d}", "augment": "long_price_change_pct"}
                next_id += 1
                completion = SV.completion({"type": kind, "sql": sql, "notes": notes})
                ev = {"qid": qid, "provider": "template", "attempt": "template", "completion": completion, "sql": sql, "type": kind, "ok": True, "stage": "ok",
                      "route": V.route(sql), "skeleton": V.skeleton(sql), "generator": "augment.py long_price_change_pct"}
                events.append(ev)
                longq.append(q)
                new_sel.append(q)
                new_cands.append((q, ev))
                sign["pos" if gold["answer"] > 0 else "neg"] += 1
                made += 1

    # ---- ghi (giữ bản cũ)
    for p in (root / "questions.jsonl", root / "answers.jsonl", root / "manifest.json", pkg / "candidates" / "solutions.jsonl", WORK / "selected.jsonl"):
        keep_copy(p)
    recs = {}
    with open(root / "questions.jsonl", "a", encoding="utf-8") as fq, open(root / "answers.jsonl", "a", encoding="utf-8") as fa:
        for q in new_sel:
            fq.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "question": q["question"], "as_of_date": q["as_of_date"]}, ensure_ascii=False) + "\n")
            g = q["gold"]
            row = {"id": q["id"], "type": g["type"]} if g["type"] == "unanswerable" else {"id": q["id"], "type": g["type"], "answer": g["answer"]}
            if g["type"] != "unanswerable":
                if q["business_id"] not in recs:
                    recs[q["business_id"]] = SH.load_shop(WORK, q["business_id"])[1]
                al = PK.alias_of(recs[q["business_id"]], g)
                if al:
                    row["alias"] = al
            fa.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(pkg / "candidates" / "solutions.jsonl", "a", encoding="utf-8") as f:
        for q, s in new_cands:
            f.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "split": q["split"], "completion": s["completion"], "thinking_mode": "off",
                                "metadata": {"solution_source": "template" if s["provider"] == "template" else "model", "provider": s["provider"],
                                             "attempt": s.get("attempt"), "route": s.get("route") or V.route(s["sql"]),
                                             "skeleton_sha": hashlib.sha256((s.get("skeleton") or V.skeleton(s["sql"])).encode()).hexdigest()[:16],
                                             "checks": {"mutants": s.get("mutants"), "static": s.get("static") or []}, "slot": q["spec"]["slot"],
                                             "origin": q["origin"], "question_key": q["qid"], "augment": q["augment"]}}, ensure_ascii=False) + "\n")
    with open(WORK / "selected.jsonl", "a", encoding="utf-8") as f:
        for q in new_sel:
            f.write(json.dumps({k: v for k, v in q.items() if k not in ("verify",)}, ensure_ascii=False, default=str) + "\n")
    with open(WORK / "questions" / "aug_long.jsonl", "w", encoding="utf-8") as f:
        for q in longq:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(WORK / "solutions" / "pool_aug.template.jsonl", "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    man["n_questions"] = len(qs) + len(new_sel)
    man.setdefault("augmentations", []).append({"by": "eval/sqlv1/augment.py", "unanswerable_from_pool": len(picked), "long_price_change_pct": len(longq),
                                                "ids": [new_sel[0]["id"], new_sel[-1]["id"]] if new_sel else None})
    (root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    summ = {"unanswerable_added": len(picked), "unanswerable_rows": sum(1 for q, _ in new_cands if q["augment"].startswith("pool")),
            "long_added": {s: sum(1 for q in longq if q["split"] == s) for s in ("train", "dev")},
            "long_words": sorted(len(q["question"].split()) for q in longq)[::max(1, len(longq) // 8)],
            "long_signs": dict(collections.Counter("pos" if q["gold"]["answer"] > 0 else "neg" for q in longq)),
            "new_ids": [new_sel[0]["id"], new_sel[-1]["id"]] if new_sel else None, "new_candidates": len(new_cands)}
    (WORK / "sft_v2" / "augment_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(A)
