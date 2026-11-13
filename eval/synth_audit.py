"""Audit độc lập giai đoạn 2 cho corpus synthetic: kiểm CẢ nghĩa câu và code, phân tầng, có mẫu mồi (canary) đã biết là sai để đo độ nhạy.

    python eval/synth_audit.py --work outputs/synth_corpus_20261110 --ledger <requests.sqlite> select   # chọn mẫu (không gọi API)
    python eval/synth_audit.py ... run                                                                  # gọi auditor qua eval/llm.py
    python eval/synth_audit.py ... summary                                                              # tỷ lệ đạt có trọng số + Wilson CI

Chỉ gửi nội dung tự sinh (README/quy ước shop giả lập, câu hỏi sinh, diễn giải đặc tả, code sinh). Auditor khác họ verifier của mẫu:
mẫu cấu hình A (verifier GPT) → DeepSeek; cấu hình B/C (verifier DeepSeek) → GPT. Teacher (Gemini) và coder (Qwen) không làm auditor.
Mã mẫu được thay bằng mã mù (a001…) để auditor không phân biệt mẫu thật và mồi."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import os
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import llm  # noqa: E402
import synth_corpus as C  # noqa: E402
import synth_world as W  # noqa: E402

N_TRAINVAL = 450
MIN_PER_SLOT = 5


def config_of(s: dict) -> str:
    if s["round_created"] >= 27:
        return "C"
    return "B" if s["round_created"] >= 12 else "A"


def auditor_of(s: dict) -> str:
    return "deepseek" if (s.get("verifier") or {}).get("provider", "gpt") == "gpt" else "gpt"


def select(work: Path, require_cert=True, min_round=0, n_trainval=None, tag="") -> dict:
    """Mẫu phân tầng theo slot (tối thiểu MIN_PER_SLOT, phần còn lại tỷ lệ cỡ slot), rải họ schema/ngành/cấu hình; cộng toàn bộ final_b, final_c."""
    st = json.loads((work / "state" / "samples.json").read_text(encoding="utf-8"))
    ver = C.cert_version()
    # require_cert=False: chọn sớm trên pool accepted khi chứng nhận đang chạy song song (bản 4 giờ); mẫu trượt certify sau đó → rejected → summary bỏ qua
    pool = [s for s in st.values() if s["status"] == "accepted" and s["split"] in ("train", "val") and (not require_cert or C.cert_current(s, ver))
            and s["round_created"] >= min_round and (require_cert or not C.code_issues(s["coder"]["code"], s["spec"], s["question"]))]
    rng = random.Random(4242)
    by = collections.defaultdict(list)
    for s in pool:
        by[s["slot"]].append(s)
    rest = (n_trainval or N_TRAINVAL) - MIN_PER_SLOT * len(by)
    alloc = {k: min(len(v), MIN_PER_SLOT + int(rest * len(v) / len(pool))) for k, v in by.items()}
    chosen = []
    for slot in sorted(by):
        xs = sorted(by[slot], key=lambda s: s["id"])
        rng.shuffle(xs)
        seen = collections.Counter()
        picked = []
        for _ in range(alloc[slot]):  # tham lam: ưu tiên (họ schema, ngành, cấu hình) ít xuất hiện nhất
            best = min((s for s in xs if s not in picked), key=lambda s: (seen[("f", s["family"])] + seen[("i", s["industry"])] + seen[("c", config_of(s))], rng.random()))
            picked.append(best)
            seen.update([("f", best["family"]), ("i", best["industry"]), ("c", config_of(best))])
        chosen += picked
    fins = [s for s in st.values() if s["split"] in ("final_b", "final_c") and s["status"] == "accepted" and (not require_cert or C.cert_current(s, ver))
            and s["round_created"] >= min_round]
    rows = [{"id": s["id"], "split": s["split"], "slot": s["slot"], "family": s["family"], "industry": s["industry"], "config": config_of(s),
             "stratum_pop": len(by[s["slot"]]) if s["split"] in ("train", "val") else None, "stratum_n": alloc.get(s["slot"]) if s["split"] in ("train", "val") else None,
             "cert_at_select": C.cert_current(s, ver)}
            for s in chosen + fins]
    C.write_json(work / "audit" / f"audit_sample_phase2{tag}.json", rows)
    return {"trainval": len(chosen), "final_b_c": len(fins), "slots": len(by), "pool": len(pool)}


# ---------------------------------------------------------------- mồi (đã biết sai): đo độ nhạy auditor
def _swap_period(q: str, spec: dict) -> str | None:
    p = spec.get("period") or {}
    if p.get("kind") == "month":
        m2 = p["m"] % 12 + 1
        out = re.sub(rf"(tháng\s*0?){p['m']}(\s*(/|-|năm)\s*{p['y']})", rf"\g<1>{m2}\2", q, count=1, flags=re.I)
        out = re.sub(rf"(?<!\d)0?{p['m']}/{p['y']}", f"{m2}/{p['y']}", out, count=1) if out == q else out
        return out if out != q else None
    rel = {"hôm qua": "hôm nay", "tháng trước": "tháng này", "tuần trước": "tuần này", "tháng này": "tháng trước", "tuần này": "tuần trước"}
    for a, b in rel.items():
        if a in q.lower():
            return re.sub(a, b, q, count=1, flags=re.I)
    return None


def canaries(st: dict, rng: random.Random, n_code=30, n_meaning=20) -> list[dict]:
    """Mồi code: câu + code của mẫu bị loại vì code sai thật (sai ở đối chứng runtime, sai đáp án, ghi cứng). Mồi nghĩa: câu đã đổi kỳ
    nhưng giữ diễn giải/đáp án cũ (dung_nghia phải sai), hoặc câu văn phong bị loại vì lệch nghĩa."""
    out = []
    bad_code = [s for s in st.values() if s["status"] == "rejected" and s["coder"].get("code") and s["split"] in ("train", "val")
                and ((s["reject"] or [""])[0].split(":")[0] in ("mutation_runtime", "coder_wrong") or (s["reject"] or [""])[0] == "hardcode:hardcoded_key")]
    # bỏ hardcode:missing_TODAY: gần như toàn câu không trả lời được, code dm.unanswerable(...) ĐÚNG — không phải mồi sai thật (thử pilot 10/11)
    bad_code.sort(key=lambda s: s["id"])
    rng.shuffle(bad_code)
    for s in bad_code[:n_code]:
        out.append({"base": s["id"], "kind": "code:" + s["reject"][0].split(":")[0], "expect": "code_ok", "question": s["question"],
                    "question_canonical": s["question_canonical"], "code": s["coder"]["code"]})
    good = [s for s in st.values() if s["status"] == "accepted" and s["split"] in ("train", "val") and s["slot"] != "unanswerable"]
    good.sort(key=lambda s: s["id"])
    rng.shuffle(good)
    k = 0
    for s in good:
        if k >= n_meaning:
            break
        q2, q2c = _swap_period(s["question"], s["spec"]), _swap_period(s["question_canonical"], s["spec"])
        if q2 and q2c:
            out.append({"base": s["id"], "kind": "meaning:period_swap", "expect": "dung_nghia", "question": q2, "question_canonical": q2c,
                        "code": s["coder"]["code"]})
            k += 1
    return out


AUDIT_RULES = ("Bạn kiểm định độc lập dữ liệu huấn luyện: câu hỏi của chủ cửa hàng nhỏ (dữ liệu giả lập tự tạo), đáp án chuẩn do chương trình tính, "
               "và code pandas trả lời câu hỏi. Với MỖI mẫu, đánh giá riêng từng tiêu chí (true/false):\n"
               "- ro_rang: câu văn phong có một cách hiểu hợp lý duy nhất theo quy ước của cửa hàng.\n"
               "- dung_nghia: diễn giải phép tính + đáp án chuẩn đúng là thứ câu hỏi hỏi (đúng số đo, kỳ thời gian, đối tượng, kiểu đáp án, quy ước).\n"
               "- giu_nghia: câu văn phong giữ nguyên nghĩa câu gốc (viết tắt, không dấu, lỗi gõ không làm đổi nghĩa).\n"
               "- code_ok: code tính ĐÚNG điều câu hỏi hỏi theo README và quy ước: đúng bảng/cột, lọc trạng thái, kỳ (kể cả TODAY và trọn ngày cuối), "
               "dấu tiền hoàn, quy đổi đơn vị, bỏ khách lẻ, đồng hạng; không ghi cứng đáp án, mã nội bộ hay ngày suy từ hôm nay; không có dòng nháp. "
               "Biến T là dict bảng đã nạp, dm là thư viện tiện ích (dm.month, dm.this_week(TODAY)…, .mask(cột) lọc trọn kỳ), TODAY là ngày hôm nay.\n"
               "Không cần tính lại con số. Nếu có tiêu chí sai, nêu ngắn lỗi cụ thể trong ly_do.\n"
               "Trả về đúng một JSON: [{\"id\": \"<mã>\", \"ro_rang\": true/false, \"dung_nghia\": true/false, \"giu_nghia\": true/false, "
               "\"code_ok\": true/false, \"ly_do\": \"...\"}, ...]")


def prompt(groups) -> str:
    parts = [AUDIT_RULES]
    for sh, items in groups:
        parts.append(f"\n## CỬA HÀNG {sh.rec['shop_name']} — hôm nay (TODAY) = {sh.today}\n{W.render(sh.rec, sh.s['family'])[1]}\n"
                     f"{W.rules_text(sh.rec, sh.s['family'])}")
        for it in items:
            parts.append(f"- id: {it['blind']}\n  câu văn phong: {it['question']}\n  câu gốc: {it['question_canonical']}\n  diễn giải: {it['explain']}\n"
                         f"  đáp án chuẩn: {it['gold']}\n  code:\n```python\n{it['code']}\n```")
    return "\n".join(parts)


def run(work: Path, ledger: Path, per_call=12, shops_per_call=3, providers=("gpt", "deepseek"), workers=8, tag="", n_code=30, n_meaning=20) -> dict:
    """providers: auditor đang dùng được (smoke test trước khi chạy). Mẫu có auditor không nằm trong danh sách → ghi missing=provider_unavailable
    (KHÔNG đổi sang auditor cùng họ verifier); mồi chia vòng tròn cho các auditor đang dùng."""
    llm.configure_run(ledger, limit=llm.HARD_CAP,
                      provider_limits=json.loads(os.environ.get("SYNTH_PROVIDER_LIMITS") or '{"gemini": 4, "gpt": 4, "qwen": 4, "deepseek": 2}'))
    providers = tuple(p for p in ("gpt", "deepseek") if p in providers)
    runner = C.Runner(work, ledger, "main", llm.HARD_CAP)
    st = runner.samples
    plan = {s["name"]: s for s in C.shop_plan()}
    rows = json.loads((work / "audit" / f"audit_sample_phase2{tag}.json").read_text(encoding="utf-8"))
    rng = random.Random(77)
    cans = canaries(st, rng, n_code=n_code, n_meaning=n_meaning)
    items = []
    for r in rows:
        s = st[r["id"]]
        items.append({"id": s["id"], "shop": s["shop"], "auditor": auditor_of(s), "question": s["question"], "question_canonical": s["question_canonical"],
                      "code": s["coder"]["code"], "spec": s["spec"], "canary": None})
    for i, c in enumerate(cans):
        s = st[c["base"]]
        items.append({"id": f"canary{i:03d}", "shop": s["shop"], "auditor": providers[i % len(providers)], "question": c["question"],
                      "question_canonical": c["question_canonical"], "code": c["code"], "spec": s["spec"], "canary": c})
    order = list(range(len(items)))
    rng.shuffle(order)
    for k, i in enumerate(order):
        items[i]["blind"] = f"a{k:04d}"
    blind = {it["blind"]: it for it in items}
    by = collections.defaultdict(list)
    for it in items:
        sh = runner.shop(plan[it["shop"]])
        g = it["spec"]["gold"]
        it["explain"] = C.explain(sh.rec, it["spec"])
        it["gold"] = json.dumps({k: g[k] for k in g if k in ("type", "answer")}, ensure_ascii=False)
        by[(it["auditor"], it["shop"])].append(it)
    calls, verdicts = collections.Counter(), {}
    jobs = []
    for prov in providers:
        keys = sorted(k for k in by if k[0] == prov)
        batch, n = [], 0
        batches = []
        for k in keys:
            its = by[k]
            for j in range(0, len(its), per_call):
                chunk = its[j:j + per_call]
                if batch and (len(batch) >= shops_per_call or n + len(chunk) > per_call):
                    batches.append(batch)
                    batch, n = [], 0
                batch.append((runner.shop(plan[k[1]]), chunk))
                n += len(chunk)
        if batch:
            batches.append(batch)
        jobs += [(prov, b) for b in batches]

    run_tag = tag

    def call(job):
        prov, b = job
        tag = "+".join(sh.name for sh, _ in b)[:120]
        r, hit = C.model_call(work, "audit2" + run_tag, tag, prov, prompt(b), think="off", max_tokens=12000)
        return prov, b, r, hit
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for prov, b, r, hit in ex.map(call, jobs):
            calls[prov] += not hit
            calls[f"{prov}:{r.get('finish')}"] += 1
            try:
                out = llm.extract_json(r["content"]) if r.get("finish") == "stop" else []
            except ValueError:
                out = []
            want = {it["blind"] for _, its in b for it in its}
            for x in out if isinstance(out, list) else []:
                if isinstance(x, dict) and x.get("id") in want:
                    verdicts[x["id"]] = {k: x.get(k) for k in ("ro_rang", "dung_nghia", "giu_nghia", "code_ok", "ly_do")}
    res, can = {}, []
    for b, it in blind.items():
        v = verdicts.get(b) or {"missing": True if it["auditor"] in providers else "provider_unavailable"}
        v = dict(v, auditor=it["auditor"], blind=b)
        if it["canary"]:
            can.append({"id": it["id"], **{k: it["canary"][k] for k in ("base", "kind", "expect")}, "verdict": v,
                        "caught": (not v.get("missing")) and not all(v.get(k) for k in ("ro_rang", "dung_nghia", "giu_nghia", "code_ok"))})
        else:
            res[it["id"]] = v
            if not v.get("missing"):
                st[it["id"]]["audit_llm"] = dict(v, phase=2, tag=tag, code_sha256=C.digest(st[it["id"]]["coder"]["code"]))
    runner.save()
    C.write_json(work / "audit" / f"audit_llm_phase2{tag}.json", res)
    C.write_json(work / "audit" / f"canary_phase2{tag}.json", can)
    return {"audited": len(res), "answered": sum(1 for v in res.values() if not v.get("missing")),
            "provider_unavailable": sum(1 for v in res.values() if v.get("missing") == "provider_unavailable"), "providers": list(providers),
            "canaries": len(can), "canary_answered": sum(1 for c in can if not c["verdict"].get("missing")),
            "canary_caught": sum(c["caught"] for c in can), "calls": dict(calls)}


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - r) / d, 4), round((c + r) / d, 4))


def summary(work: Path, tag="") -> dict:
    """Tỷ lệ đạt (nghĩa / code / toàn phần) theo LLM, theo kiểm tay, và kết hợp (kiểm tay ghi đè LLM) — có trọng số tầng slot cho train/val."""
    st = json.loads((work / "state" / "samples.json").read_text(encoding="utf-8"))
    rows = json.loads((work / "audit" / f"audit_sample_phase2{tag}.json").read_text(encoding="utf-8"))
    llm_v = json.loads((work / "audit" / f"audit_llm_phase2{tag}.json").read_text(encoding="utf-8"))
    man = C.load_manual_audits(work)
    can = json.loads((work / "audit" / f"canary_phase2{tag}.json").read_text(encoding="utf-8")) if (work / "audit" / f"canary_phase2{tag}.json").exists() else []
    out = {}
    ex_p = work / "audit" / "exclude_selection_4h.json"  # lỗi hệ thống (luật tĩnh) — tính thêm bộ số sau khi loại
    sys_ex = set(json.loads(ex_p.read_text(encoding="utf-8"))["ids"]) if ex_p.exists() else set()
    q_p = work / "audit" / f"manual_queue_phase2{tag}.json"
    why = {r["id"]: r["why"] for r in json.loads(q_p.read_text(encoding="utf-8"))} if q_p.exists() else {}
    for split_set, name, drop in ((("train", "val"), "trainval", set()), (("final_b", "final_c"), "final", set()),
                                  (("train", "val"), "trainval_after_systematic", sys_ex), (("final_b", "final_c"), "final_after_systematic", sys_ex)):
        rr = [r for r in rows if r["split"] in split_set and st.get(r["id"], {}).get("status") == "accepted" and r["id"] not in drop]  # trượt certify sau khi chọn → bỏ
        agg = collections.defaultdict(lambda: [0.0, 0.0, 0, 0])
        for r in rr:
            s = st[r["id"]]
            v = llm_v.get(r["id"], {})
            mv = man.get(r["id"])
            mverd = C.manual_verdict(s, mv)
            meaning = all(v.get(k) for k in ("ro_rang", "dung_nghia", "giu_nghia")) if not v.get("missing") else None
            code = bool(v.get("code_ok")) if not v.get("missing") else None
            if mverd in ("pass", "fail"):
                meaning = all(mv.get(k, True) for k in ("ro_rang", "dung_nghia", "giu_nghia"))
                code = bool(mv.get("code_ok", mv["pass"]))
            if meaning is None:
                continue
            w = (r["stratum_pop"] / r["stratum_n"]) if r.get("stratum_pop") else 1.0
            for key, ok in (("meaning", meaning), ("code", code), ("overall", meaning and code)):
                a = agg[key]
                a[0] += w * ok
                a[1] += w
                a[2] += ok
                a[3] += 1
        out[name] = {k: {"n": a[3], "pass": a[2], "rate_unweighted": round(a[2] / a[3], 4), "rate_weighted": round(a[0] / a[1], 4),
                         "wilson95_unweighted": wilson(a[2], a[3])} for k, a in agg.items()}
    # kiểm tay riêng (giai đoạn 2, đúng code hiện tại): nghĩa / code / toàn phần, kèm Wilson 95%
    m2 = [(k, v) for k, v in man.items() if v.get("phase", "") == "2" + tag and k in st and st[k]["status"] == "accepted"
          and C.manual_verdict(st[k], v) in ("pass", "fail")]
    def _rate(xs):
        k = sum(xs)
        return {"n": len(xs), "pass": k, "rate": round(k / max(1, len(xs)), 4), "wilson95": wilson(k, len(xs))}
    out["manual_only"] = {"meaning": _rate([all(v.get(x, True) for x in ("ro_rang", "dung_nghia", "giu_nghia")) for _, v in m2]),
                          "code": _rate([bool(v.get("code_ok", v["pass"])) for _, v in m2]), "overall": _rate([bool(v["pass"]) for _, v in m2]),
                          "note": "toàn bộ kiểm tay giai đoạn 2 — gồm cả mẫu LLM gắn cờ (hàng đợi làm giàu lỗi) nên chệch thấp; dùng manual_random làm ước lượng"}
    m2r = [(k, v) for k, v in m2 if why.get(k, "").startswith("random")]
    out["manual_random"] = {"overall": _rate([bool(v["pass"]) for _, v in m2r]), "code": _rate([bool(v.get("code_ok", v["pass"])) for _, v in m2r]),
                            "meaning": _rate([all(v.get(x, True) for x in ("ro_rang", "dung_nghia", "giu_nghia")) for _, v in m2r]),
                            "overall_after_systematic": _rate([bool(v["pass"]) for k, v in m2r if k not in sys_ex]),
                            "note": "mẫu ngẫu nhiên phân tầng (train/val vs final × cấu hình A/B/C) trong số LLM cho đạt — ước lượng không chệch phần LLM đạt"}
    out["manual_flagged"] = {"overall": _rate([bool(v["pass"]) for k, v in m2 if why.get(k) == "llm_flag"]),
                             "note": "tỷ lệ mẫu LLM gắn cờ mà kiểm tay xác nhận ĐẠT (= tỷ lệ cờ nhầm)"}
    out["systematic_excluded_ids"] = len(sys_ex)
    # ước lượng hiệu chỉnh hai pha (mẫu chưa kiểm tay dùng tỷ lệ kiểm tay theo nhóm LLM đạt / LLM gắn cờ thay vì tin LLM)
    def _calib(split_set, drop):
        rr = [r for r in rows if r["split"] in split_set and st.get(r["id"], {}).get("status") == "accepted" and r["id"] not in drop
              and r["id"] in llm_v and not llm_v[r["id"]].get("missing")]
        flag = [r for r in rr if not all(llm_v[r["id"]].get(x) for x in ("ro_rang", "dung_nghia", "giu_nghia", "code_ok"))]
        rp = [bool(v["pass"]) for k, v in m2 if why.get(k, "").startswith("random") and k not in drop]
        rf = [bool(v["pass"]) for k, v in m2 if why.get(k) == "llm_flag" and k not in drop]
        if not rr or not rp:
            return None
        pf = len(flag) / len(rr)
        r_pass, r_flag = sum(rp) / len(rp), (sum(rf) / len(rf) if rf else 0.0)
        lo_p, lo_f = wilson(sum(rp), len(rp))[0], (wilson(sum(rf), len(rf))[0] if rf else 0.0)
        return {"n_audited": len(rr), "p_llm_flag": round(pf, 4), "manual_pass_rate_llm_pass": round(r_pass, 4), "manual_pass_rate_llm_flag": round(r_flag, 4),
                "estimate": round((1 - pf) * r_pass + pf * r_flag, 4), "estimate_low95": round((1 - pf) * lo_p + pf * lo_f, 4),
                "n_manual_random": len(rp), "n_manual_flag": len(rf)}
    out["calibrated"] = {"trainval_full_pool": _calib(("train", "val"), set()), "trainval_after_systematic": _calib(("train", "val"), sys_ex),
                         "final_full": _calib(("final_b", "final_c"), set()),
                         "note": "estimate = P(LLM đạt)·tỷ lệ đạt kiểm tay ngẫu nhiên + P(LLM cờ)·tỷ lệ đạt kiểm tay mẫu cờ; low95 dùng cận dưới Wilson của hai tỷ lệ"}
    out["manual_only"]["agree_with_llm"] = _rate([bool(v["pass"]) == all(llm_v.get(k, {}).get(x) for x in ("ro_rang", "dung_nghia", "giu_nghia", "code_ok"))
                                                   for k, v in m2 if k in llm_v and not llm_v[k].get("missing")])
    n_all = len(can)
    can = [c for c in can if not c["verdict"].get("missing")]  # độ nhạy chỉ tính trên mồi auditor đã trả lời
    kinds = collections.Counter((c["kind"], c["caught"]) for c in can)
    out["canary"] = {"n_total": n_all, "n": len(can), "caught": sum(c["caught"] for c in can), "recall": round(sum(c["caught"] for c in can) / max(1, len(can)), 3),
                     "recall_wilson95": wilson(sum(c["caught"] for c in can), len(can)),
                     "by_kind": {f"{k}:{'caught' if ok else 'missed'}": n for (k, ok), n in sorted(kinds.items())},
                     "by_auditor": {a: [sum(1 for c in can if c["verdict"]["auditor"] == a and c["caught"]), sum(1 for c in can if c["verdict"]["auditor"] == a)]
                                    for a in ("gpt", "deepseek")}}
    C.write_json(work / "audit" / f"summary_phase2{tag}.json", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("cmd", choices=["select", "run", "summary"])
    ap.add_argument("--providers", default="gpt,deepseek", help="auditor đang dùng được, ví dụ deepseek nếu GPT lỗi smoke test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cert-filter", action="store_true", help="chọn trên pool accepted khi chứng nhận chưa xong (ghi rõ trong báo cáo)")
    ap.add_argument("--min-round", type=int, default=0, help="chỉ audit mẫu sinh từ vòng này (audit bổ sung)")
    ap.add_argument("--n", type=int, default=None, help="cỡ mẫu train/val (mặc định N_TRAINVAL)")
    ap.add_argument("--tag", default="", help="hậu tố file audit, ví dụ 'b' → audit_*_phase2b.json")
    ap.add_argument("--canaries", default="30,20", help="số mồi code,nghĩa")
    a = ap.parse_args(argv)
    nc, nm = (int(x) for x in a.canaries.split(","))
    f = {"select": lambda: select(a.work, require_cert=not a.no_cert_filter, min_round=a.min_round, n_trainval=a.n, tag=a.tag),
         "run": lambda: run(a.work, a.ledger, providers=tuple(a.providers.split(",")), workers=a.workers, tag=a.tag, n_code=nc, n_meaning=nm),
         "summary": lambda: summary(a.work, tag=a.tag)}[a.cmd]
    print(json.dumps(f(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
