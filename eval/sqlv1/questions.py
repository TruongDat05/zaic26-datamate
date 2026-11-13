"""Câu hỏi + gold cho syn_sales_sql_v1.

- reuse: câu đã qua kiểm của bộ cũ (accepted + chỉ bị loại vì code pandas) trên 132 thế giới cũ; gold TÍNH LẠI bằng oracle btc_v1;
  bỏ câu có nghĩa phụ thuộc quy ước cũ (doanh thu thuần, tiền hoàn theo thời điểm hoàn, ca tối).
- gen: câu mới — teacher viết câu từ đặc tả (không có đáp án), verifier KHÁC HỌ đọc ngược câu thành đặc tả (không thấy gold, không thấy
  đặc tả gốc); chỉ giữ câu đọc ngược khớp đặc tả. Gold luôn do world.gold tính.

    python eval/sqlv1/questions.py reuse
    python eval/sqlv1/questions.py gen --n-per-shop 24 [--shops ...] [--teacher gemini --verifier deepseek]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1 import shops as SH, specs as SP, world as WB  # noqa: E402
import synth_world as W  # noqa: E402

WORK = SH.SYNTH_ROOT / "outputs" / "synth_sql_v1"
OLD = SH.SYNTH_ROOT / "outputs" / "synth_corpus_20261110" / "state" / "samples.json"
CODE_ONLY = {"certify", "hardcode", "mutation_runtime", "coder_wrong"}
OLD_CONVENTION = re.compile(r"thuần|thuan|hoàn trả|hoan tra|ca tối|ca toi|hoàn tiền|hoan tien", re.I)
SHIFT = {"ca sáng": "sang", "ca chiều": "chieu"}


def fold(s: str) -> str:
    t = unicodedata.normalize("NFKD", s.replace("đ", "d").replace("Đ", "D"))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", "".join(c for c in t if not unicodedata.combining(c)).lower()).split())


# ------------------------------------------------------------------ dùng lại câu cũ
def reuse(work: Path = WORK) -> dict:
    st = json.loads(OLD.read_text(encoding="utf-8"))
    plan = {s.get("old_name"): s for s in json.loads((work / "plan.json").read_text(encoding="utf-8")) if s.get("old_name")}
    recs, out, drop = {}, [], collections.Counter()
    for s in sorted(st.values(), key=lambda x: x["id"]):
        rj = (s.get("reject") or [""])[0].split(":")[0]
        if not (s["status"] == "accepted" or (s["status"] == "rejected" and rj in CODE_ONLY)):
            continue
        p = plan[s["shop"]]
        spec = {k: s["spec"][k] for k in ("slot", "period", "base_period", "params", "as_of")}
        spec["params"] = dict(spec["params"] or {})
        if spec["slot"] in ("revenue_net", "refund_total"):
            drop["quy_uoc_cu:" + spec["slot"]] += 1
            continue
        if OLD_CONVENTION.search(s["question"]) or OLD_CONVENTION.search(s.get("question_canonical") or ""):
            drop["quy_uoc_cu:wording"] += 1
            continue
        if "shift" in spec["params"]:
            if spec["params"]["shift"] not in SHIFT:
                drop["quy_uoc_cu:ca_toi"] += 1
                continue
            spec["params"]["shift"] = SHIFT[spec["params"]["shift"]]
        bid = p["business_id"]
        if bid not in recs:
            recs[bid] = SH.load_shop(work, bid)[1]
        try:
            g = WB.gold(recs[bid], spec)
        except WB.Degenerate as e:
            drop["suy_bien:" + str(e)[:30]] += 1
            continue
        except (ValueError, KeyError, StopIteration) as e:
            drop["loi_dac_ta:" + type(e).__name__] += 1
            continue
        out.append({"key": "old:" + s["id"], "origin": "reuse", "old_id": s["id"], "old_split": s["split"], "split": p["split"], "business_id": bid,
                    "industry": p["industry"], "family": p["family"], "question": s["question"], "as_of_date": spec["as_of"], "spec": spec, "gold": g,
                    "old_gold_changed": g != s["spec"]["gold"], "style_variant": s.get("style_variant")})
    (work / "questions").mkdir(parents=True, exist_ok=True)
    with open(work / "questions" / "reuse.jsonl", "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    rep = {"kept": len(out), "dropped": dict(drop), "by_split": dict(collections.Counter(x["split"] for x in out)),
           "gold_changed_vs_old": sum(x["old_gold_changed"] for x in out)}
    (work / "questions" / "reuse_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


# ------------------------------------------------------------------ đặc tả → lời cho teacher
MEASURE = {
    "revenue": "tổng doanh thu", "revenue_item": "doanh thu của mặt hàng {item}", "revenue_group": "doanh thu của nhóm hàng {group}",
    "revenue_channel": "doanh thu qua kênh bán {channel}", "revenue_payment": "doanh thu của các đơn thanh toán bằng hình thức {payment}",
    "n_orders": "số đơn bán được", "n_orders_item": "số đơn có mua {item}", "n_cancelled": "số đơn bị hủy", "n_returned": "số đơn khách trả hàng",
    "cancel_rate": "tỷ lệ đơn bị hủy (phần trăm)", "avg_order": "giá trị trung bình mỗi đơn", "qty_base": "số {base} {item} đã bán ra",
    "top_items": "{k} mặt hàng bán chạy nhất (theo số lượng)", "top_item": "mặt hàng bán chạy nhất (một mặt hàng, theo số lượng)",
    "best_item_rev": "mặt hàng mang lại doanh thu cao nhất (một mặt hàng)", "best_day": "ngày có doanh thu cao nhất",
    "best_month": "tháng có doanh thu cao nhất", "min_month": "tháng có doanh thu thấp nhất", "top_customer": "khách mua nhiều tiền nhất (một khách)",
    "top_customers": "{k} khách mua nhiều tiền nhất", "top_customer_orders": "khách mua nhiều đơn nhất (một khách)", "n_customers": "số khách đã mua hàng",
    "cust_ge_n": "số khách mua từ {n} đơn trở lên", "exists_item_day": "có bán được {item} không", "group_table": "doanh thu của từng nhóm hàng (bảng: nhóm, doanh thu)",
    "unsold_items": "số mặt hàng bị ế (trước đó có bán mà trong kỳ không bán được cái nào)", "rev_change_pct": "doanh thu {period} tăng/giảm bao nhiêu phần trăm so với {base_p}",
    "rev_change": "doanh thu {period} chênh lệch bao nhiêu tiền so với {base_p} (kỳ hỏi trừ kỳ so)", "g_rev_change": "doanh thu nhóm {group} {period} tăng/giảm bao nhiêu phần trăm so với {base_p}",
    "share_group": "nhóm {group} chiếm bao nhiêu phần trăm doanh thu", "share_channel": "kênh {channel} chiếm bao nhiêu phần trăm doanh thu",
    "shift_share": "{shift} chiếm bao nhiêu phần trăm doanh thu", "days_over": "số ngày có doanh thu từ {threshold} đồng trở lên",
    "orders_over": "số đơn có giá trị từ {threshold} đồng trở lên", "items_ge": "số mặt hàng {group_text}bán được từ {n} đơn vị cơ bản trở lên",
    "rev_list": "doanh thu từng tháng (danh sách số theo thứ tự tháng)", "avg_daily_rev": "doanh thu trung bình mỗi ngày", "shift_rev": "doanh thu {shift}",
    "shift_orders": "số đơn {shift}", "staff_rev": "doanh thu của nhân viên {staff}", "staff_best": "nhân viên bán được nhiều tiền nhất (một người)",
    "rx_rev": "doanh thu thuốc kê đơn", "rx_share": "thuốc kê đơn chiếm bao nhiêu phần trăm doanh thu", "debt_total": "tổng số tiền khách còn nợ",
    "debt_n_cust": "số khách còn nợ", "top_debtor": "khách đang nợ nhiều nhất (một khách)", "debt_cust": "số tiền khách {customer} còn nợ",
    "debt_older": "số tiền nợ của các đơn đã quá {days} ngày (tính từ ngày đặt)", "debt_collected": "số tiền thu nợ được (khách trả nợ cũ)",
    "debt_new": "số tiền khách nợ thêm (mua chịu mới)", "purchase_money": "tổng tiền nhập hàng", "avg_cost": "giá nhập bình quân một {base} {item}",
    "lots_expiring": "số lô hàng còn tồn sẽ hết hạn", "price_on_day": "giá bán một {base} {item}", "price_change_pct": "giá bán một {base} {item} tăng/giảm bao nhiêu phần trăm",
    "profit_gross": "lãi gộp", "stock_qty": "số {base} {item} còn tồn kho", "unanswerable": "{missing} (cửa hàng không ghi thông tin này)"}
ANS = {"number": "number (một số)", "percent": "percent (bao nhiêu phần trăm)", "date": "date", "string": "string (một tên)", "list": "list",
       "table": "table (liệt kê kèm số)", "boolean": "boolean (có/không)", "unanswerable": "unanswerable (hỏi tự nhiên)"}
WORD_BINS = (("<=8", 0.30, "6-8"), ("9-12", 0.45, "9-12"), ("13-16", 0.18, "13-16"), ("17-20", 0.05, "17-20"), (">20", 0.02, "21-26"))  # bù kho câu cũ (lệch dài)


def _dmy(d):
    d = W._d(d)
    return f"{d.day}/{d.month}/{d.year}"


def spec_text(rec, spec) -> dict:
    items = {it["item"]: it for it in rec["items"]}
    p = spec.get("params") or {}
    fill = {"item": items[p["item"]]["name"] if "item" in p else "", "base": items[p["item"]]["base"] if "item" in p else "",
            "group": p.get("group", ""), "channel": p.get("channel", ""), "payment": p.get("payment", ""), "k": p.get("k", ""), "n": p.get("n", ""),
            "threshold": f"{p['threshold']:,}".replace(",", ".") if "threshold" in p else "", "shift": {"sang": "ca sáng", "chieu": "ca chiều"}.get(p.get("shift"), ""),
            "staff": p.get("staff", ""), "customer": p.get("customer", ""), "days": p.get("days", ""), "missing": p.get("missing", ""),
            "group_text": f"thuộc nhóm {p['group']} " if "group" in p else "",
            "period": W.period_text(spec["period"], spec["as_of"]) if spec.get("period") else "",
            }
    base_txt = W.period_text(spec["base_period"], spec["as_of"]) if spec.get("base_period") else ""
    m = MEASURE[spec["slot"]].format(**fill, base_p=base_txt)
    out = {"so_do": m, "ky": fill["period"], "kieu_dap_an": ANS[WB.SLOT_TYPE[spec["slot"]]]}
    if spec["slot"] in ("rev_change_pct", "rev_change", "g_rev_change"):
        out["ky"] = f"{fill['period']} so với {base_txt}"
    if spec["slot"] == "price_change_pct":
        out["ky"] = f"ngày {_dmy(spec['period']['d2'])} so với ngày {_dmy(spec['period']['d1'])} (nêu đủ hai ngày)"
    if spec["slot"] == "lots_expiring":
        out["ky"] = f"trong {p['n']} ngày tới tính từ hôm nay"
    if spec["slot"] in ("best_month", "min_month"):
        out["kieu_dap_an"] = "date (tháng nào)"
    if spec["slot"] == "best_day":
        out["kieu_dap_an"] = "date (ngày nào)"
    if spec["slot"] == "top_items":
        out["kieu_dap_an"] = f"list ({p['k']} tên)"
    if spec["slot"] == "top_customers":
        out["kieu_dap_an"] = f"list ({p['k']} tên khách)"
    return out


def style_of(key: str) -> dict:
    r = random.Random("style:" + key)
    lab, _, rng_ = r.choices(WORD_BINS, [w for _, w, _ in WORD_BINS])[0]
    return {"so_tu": rng_, "khong_dau": r.random() < 0.27, "viet_tat": r.random() < 0.44, "tu_dem": r.random() < 0.26,
            "go_telex": r.random() < 0.07, "chu_thuong_dau": r.random() < 0.53}


def style_words(st: dict) -> str:
    parts = [f"khoảng {st['so_tu']} từ"]
    parts.append("viết KHÔNG dấu" if st["khong_dau"] else "có dấu")
    if st["viet_tat"]:
        parts.append("có viết tắt kiểu nhắn tin (dthu, bn, bnhiu, sl, dc, hqua, hnay, sp...)")
    if st["tu_dem"]:
        parts.append("có từ đệm/lời nhờ (cho hỏi, xem giúp, giùm, nhỉ, ạ, với)")
    if st["go_telex"]:
        parts.append("gõ sai kiểu telex một từ (vd. thangs, tieenf, nhieeu)")
    parts.append("viết thường chữ đầu" if st["chu_thuong_dau"] else "viết hoa chữ đầu")
    return "; ".join(parts)


def shop_brief(rec, plan) -> str:
    units = collections.defaultdict(list)
    for u in rec["units"]:
        if u["factor"] > 1:
            units[u["item"]].append(u["unit"])
    t = W._d(rec["today"])
    lines = [f"Cửa hàng: {rec['shop_name']} ({W.INDUSTRIES[rec['industry']]['label']}), dữ liệu giả lập tự tạo, từ {_dmy(rec['start'])} đến {_dmy(rec['data_end'])}.",
             "Mặt hàng (tên; nhóm; đơn vị cơ bản; đơn vị lớn): " + "; ".join(f"{it['name']} ({it['group']}; {it['base']}" + (f"; {'/'.join(units[it['item']])}" if units[it["item"]] else "") + ")" for it in rec["items"]),
             "Kênh bán: " + ", ".join(sorted({o["channel"] for o in rec["orders"]})) + ". Hình thức thanh toán: " + ", ".join(sorted({o["payment"] for o in rec["orders"]})) + "."]
    if rec.get("staff"):
        lines.append("Nhân viên: " + ", ".join(rec["staff"]) + ".")
    return "\n".join(lines)


def teacher_prompt(rec, plan, items) -> str:
    """items: [(i, spec, style)] cùng shop, có thể khác ngày hôm nay."""
    head = ["Bạn soạn câu hỏi tiếng Việt mà chủ một cửa hàng nhỏ nhắn hỏi về sổ bán hàng của chính mình (dữ liệu giả lập tự tạo).", shop_brief(rec, plan), "",
            "Với MỖI đặc tả viết đúng MỘT câu hỏi. Yêu cầu:",
            "- Nghĩa đúng hệt đặc tả: cùng số đo, cùng đối tượng (giữ nguyên tên hàng/nhóm/kênh/khách), cùng kỳ, cùng kiểu đáp án; không thêm hay bớt điều kiện.",
            "- Kỳ: tháng/quý/năm/ngày cụ thể ghi đủ năm (vd. 'tháng 5/2025', 'T5/2025', 'tháng 5 năm 2025', 'quý 2/2025', 'năm 2025', '12/3/2025'). Kỳ tương đối "
            "(hôm nay, hôm qua, tháng này, tuần trước, N ngày qua, tính đến hôm nay...) giữ nguyên là tương đối; bỏ phần giải thích trong ngoặc của đặc tả.",
            "- Văn phong theo trường 'van_phong' của từng đặc tả (giống tin nhắn thật: không dấu, viết tắt, từ đệm...), nhưng số liệu và tên riêng phải đọc được đúng nghĩa.",
            "- Không nhắc quy ước tính (không viết 'không tính khách lẻ', 'trừ đơn hủy', 'quy đổi', 'làm tròn'), không nêu tên cột/bảng/mã, không đưa đáp án vào câu.",
            "- Kiểu đáp án: number → 'bao nhiêu'; percent → 'bao nhiêu %' hoặc 'tỷ lệ'; date ngày → 'ngày nào'; date tháng → 'tháng nào'; boolean → 'có ... không'; "
            "string → '... nào' (một đối tượng); list → nêu rõ số lượng ('top 3', '3 ... nào'); table → yêu cầu liệt kê từng nhóm kèm số.",
            "- Câu hỏi số lượng nói rõ đơn vị cơ bản ('bao nhiêu viên', 'bao nhiêu kg').",
            f'Trả về đúng một JSON: [{{"i": <số>, "question": "<câu>"}}, ...] đủ {len(items)} phần tử, không giải thích.', "", "ĐẶC TẢ:"]
    body = [{"i": i, "hom_nay": _dmy(spec["as_of"]), **spec_text(rec, spec), "van_phong": style_words(st)} for i, spec, st in items]
    return "\n".join(head) + "\n" + json.dumps(body, ensure_ascii=False, indent=0)


MENU = {k: v.replace("{item}", "<mặt hàng>").replace("{group}", "<nhóm>").replace("{channel}", "<kênh>").replace("{payment}", "<hình thức>").replace(
    "{k}", "<k>").replace("{n}", "<n>").replace("{threshold}", "<ngưỡng>").replace("{shift}", "<ca>").replace("{staff}", "<nhân viên>").replace(
    "{customer}", "<khách>").replace("{days}", "<số ngày>").replace("{base}", "<đơn vị>").replace("{base_p}", "<kỳ so>").replace("{period}", "<kỳ hỏi>").replace("{group_text}", "").replace(
    "{missing}", "<thông tin cửa hàng không ghi>") for k, v in MEASURE.items()}
PERIOD_KINDS = ("month(y,m) | quarter(y,q) | year(y) | day(d) | range(d1,d2) | today | yesterday | this_month | last_month | this_week | last_week | "
                "last_n(n) | this_year | until(d) | until_today | prev_same_days | week_before_last | prev_n(n) | none")


def verifier_prompt(rec, plan, items) -> str:
    """items: [(i, question, today)]. Đọc ngược câu thành đặc tả; không thấy đặc tả gốc hay đáp án."""
    head = ["Đọc từng câu hỏi của chủ cửa hàng và chuyển thành đặc tả có cấu trúc (bạn KHÔNG trả lời câu hỏi). Dữ liệu giả lập tự tạo.", shop_brief(rec, plan), "",
            "Mã số đo (chọn đúng một cho mỗi câu):", json.dumps(MENU, ensure_ascii=False, indent=0),
            f"Loại kỳ (trường 'loai') và tham số của nó: {PERIOD_KINDS}. Ngày ghi YYYY-MM-DD. Kỳ tương đối giữ loại tương đối (không đổi ra ngày). "
            'Ví dụ: {"loai": "month", "y": 2025, "m": 5}; {"loai": "day", "d": "2025-03-28"}; {"loai": "range", "d1": "2025-03-01", "d2": "2025-03-15"}; '
            '{"loai": "last_n", "n": 7}; {"loai": "until", "d": "2025-06-30"}.',
            "Với câu so sánh hai kỳ: 'ky' là kỳ hỏi, 'ky_so' là kỳ đem so (vd. tháng này so với cùng kỳ tháng trước → this_month và prev_same_days; "
            "tuần trước so với tuần trước nữa → last_week và week_before_last; N ngày qua so với N ngày liền trước → last_n và prev_n).",
            "Tham số (null nếu không có): mat_hang, nhom, kenh, thanh_toan, khach, nhan_vien (tên y như câu hỏi nhắc, viết lại đúng tên trong danh sách nếu nhận ra), "
            "k (số đối tượng), nguong (số đồng), n, ca ('sang'|'chieu'), so_ngay.",
            "Nếu câu mơ hồ (hiểu được nhiều cách) đặt \"ro_rang\": false.",
            'Trả về đúng một JSON: [{"i": <số>, "so_do": "<mã>", "ky": {"loai": "...", ...}, "ky_so": {...} hoặc null, "tham_so": {...}, "ro_rang": true}, ...], không giải thích.',
            "", "CÂU HỎI:"]
    body = [{"i": i, "hom_nay": _dmy(t), "cau_hoi": q} for i, q, t in items]
    return "\n".join(head) + "\n" + json.dumps(body, ensure_ascii=False, indent=0)


ALIAS = {"y": ("y", "nam", "year"), "m": ("m", "thang", "month"), "q": ("q", "quy", "quarter"), "n": ("n", "so_ngay", "days"),
         "d": ("d", "ngay", "date", "day"), "d1": ("d1", "tu", "tu_ngay", "start", "from"), "d2": ("d2", "den", "den_ngay", "end", "to")}


def _period_from(v: dict | None):
    if not v or not isinstance(v, dict):
        return None
    k = v.get("loai") or v.get("kind")
    if k in (None, "none"):
        return None
    out = {"kind": k}
    for key, names in ALIAS.items():
        val = next((v[x] for x in names if v.get(x) is not None), None)
        if val is not None:
            out[key] = int(val) if key in ("y", "m", "q", "n") else str(val)[:10]
    if k == "until" and "d" not in out and "d2" in out:
        out["d"] = out["d2"]
    return out


def matches(rec, spec, interp) -> tuple[bool, str]:
    """Đặc tả đọc ngược có cùng nghĩa với đặc tả gốc không (so khoảng ngày đã giải, số đo, tham số)."""
    if not isinstance(interp, dict):
        return False, "no_interp"
    if interp.get("ro_rang") is False:
        return False, "ambiguous"
    slot = interp.get("so_do")
    same = {("top_item", "top_items"), ("top_items", "top_item")}
    if slot != spec["slot"] and (slot, spec["slot"]) not in same:
        return False, f"slot:{slot}"
    if spec["slot"] != "lots_expiring":
        try:
            a = W.resolve(spec["period"], spec["as_of"]) if spec.get("period") else None
            pb = _period_from(interp.get("ky"))
            b = W.resolve(pb, spec["as_of"]) if pb else None
        except (ValueError, KeyError, TypeError):
            return False, "period_unparsed"
        if spec["slot"] in ("debt_total", "debt_n_cust", "top_debtor", "debt_cust", "debt_older", "stock_qty"):
            a, b = (a[1] if a else None), (b[1] if b else None)
        if a != b and spec["slot"] != "unanswerable":
            return False, "period"
        if spec.get("base_period"):
            try:
                bb = _period_from(interp.get("ky_so"))
                if W.resolve(spec["base_period"], spec["as_of"]) != (W.resolve(bb, spec["as_of"]) if bb else None):
                    return False, "base_period"
            except (ValueError, KeyError, TypeError):
                return False, "base_unparsed"
    p, t = spec.get("params") or {}, interp.get("tham_so") or {}
    items = {it["item"]: it for it in rec["items"]}
    eq = lambda x, y: fold(str(x or "")) == fold(str(y or ""))  # noqa: E731
    checks = [("item", "mat_hang", lambda v: items[v]["name"]), ("group", "nhom", str), ("channel", "kenh", str), ("payment", "thanh_toan", str),
              ("customer", "khach", str), ("staff", "nhan_vien", str), ("shift", "ca", str)]
    for k, kk, f in checks:
        if k in p and not eq(f(p[k]), t.get(kk)):
            return False, f"param:{k}"
    for k, kk in (("k", "k"), ("threshold", "nguong"), ("n", "n"), ("days", "so_ngay")):
        if k in p and spec["slot"] != "lots_expiring" or (k == "n" and spec["slot"] == "lots_expiring"):
            try:
                if int(float(t.get(kk))) != int(p[k]):
                    return False, f"param:{k}"
            except (TypeError, ValueError):
                return False, f"param:{k}"
    return True, "ok"


# ------------------------------------------------------------------ sinh câu mới
def gen_shop(args):
    from sqlv1 import api
    import llm
    bid, n, teacher, verifier, seed, month_only = args
    info, rec = SH.load_shop(WORK, bid)
    plan = info["plan"]
    specs = (SP.sample(rec, plan["family"], n, f"gen{seed}", period_w=SP.MONTH_PERIOD_W, slot_w=SP.MONTH_SLOT_W) if month_only
             else SP.sample(rec, plan["family"], n, f"gen{seed}", period_w=SP.GEN_PERIOD_W))
    out, stats = [], collections.Counter()
    for b0 in range(0, len(specs), 12):
        batch = specs[b0:b0 + 12]
        items = [(i + 1, sp, style_of(f"{bid}:{seed}:{b0 + i}")) for i, sp in enumerate(batch)]
        r, _ = api.call(WORK, "teacher", bid, teacher, teacher_prompt(rec, plan, items), think="off", max_tokens=6000)
        try:
            qs = {int(x["i"]): str(x["question"]).strip() for x in llm.extract_json(r.get("content") or "")}
        except (ValueError, TypeError, KeyError):
            stats["teacher_bad_json"] += len(batch)
            continue
        vitems = [(i, qs[i], sp["as_of"]) for i, sp, _ in items if qs.get(i)]
        if not vitems:
            continue
        rv, _ = api.call(WORK, "verifier", bid, verifier, verifier_prompt(rec, plan, vitems), think="off", max_tokens=8000)
        try:
            interp = {int(x["i"]): x for x in llm.extract_json(rv.get("content") or "")}
        except (ValueError, TypeError, KeyError):
            stats["verifier_bad_json"] += len(vitems)
            continue
        for i, sp, st in items:
            q = qs.get(i)
            if not q:
                stats["no_question"] += 1
                continue
            ok, why = matches(rec, sp, interp.get(i))
            stats["ok" if ok else "rej:" + why.split(":")[0]] += 1
            rec_out = {"key": f"new:{bid}:{seed}:{b0 + i}", "origin": "teacher", "teacher": teacher, "verifier": verifier, "split": plan["split"], "business_id": bid,
                       "industry": plan["industry"], "family": plan["family"], "question": q, "as_of_date": sp["as_of"],
                       "spec": {k: sp[k] for k in ("slot", "period", "base_period", "params", "as_of")}, "gold": sp["gold"], "style": st,
                       "verify": {"ok": ok, "why": why, "interp": interp.get(i)}}
            out.append(rec_out)
    return bid, out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reuse", "gen"])
    ap.add_argument("--shops")
    ap.add_argument("--n-per-shop", type=int, default=24)
    ap.add_argument("--teacher", default="gemini")
    ap.add_argument("--verifier", default="deepseek")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="gen")
    ap.add_argument("--month-only", action="store_true", help="đợt bù: chỉ kỳ tháng/năm cụ thể, slot theo nhóm kỹ năng MINING")
    a = ap.parse_args()
    if a.cmd == "reuse":
        print(json.dumps(reuse(), ensure_ascii=False, indent=1))
        return
    plan = json.loads((WORK / "plan.json").read_text(encoding="utf-8"))
    if a.shops in ("core_train", "dev", "final"):
        shops = [s["business_id"] for s in plan if (a.shops == "core_train" and s["split"] == "train" and s["industry"] in WB.CORE)
                 or (a.shops != "core_train" and s["split"] == a.shops)]
    else:
        shops = a.shops.split(",")
    path = WORK / "questions" / f"{a.out}.jsonl"
    tot = collections.Counter()
    with cf.ThreadPoolExecutor(a.workers) as ex, open(path, "a", encoding="utf-8") as f:
        for bid, out, st in ex.map(gen_shop, [(b, a.n_per_shop, a.teacher, a.verifier, a.seed, a.month_only) for b in shops]):
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            tot.update(st)
            print(bid, dict(st), flush=True)
    print("total", dict(tot))


if __name__ == "__main__":
    main()
