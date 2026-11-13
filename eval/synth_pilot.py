"""Small, explicitly synthetic pilot. Never reads questions, logs or tables from BTC.

Independent truth uses Python records/loops; model code uses loaded pandas tables.
This is deliberately not the stage-1 shop simulator. Artifacts are resumable by hash.
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import llm
import model_bakeoff as mb
from datamate import loader
from synth_tools import Kit

TODAY = "2025-05-16"
RULES = """Quy ước riêng của pilot tự tạo:
Đơn bán được chấp nhận gồm các trạng thái 'Xong', 'xong', 'Đã giao', 'da giao'; 'Hủy', 'huỷ' là hủy, 'Chờ' chưa bán.
Doanh thu bán là cộng tiền dòng của đơn được chấp nhận theo ngày bán, chưa trừ hoàn tiền.
Doanh thu thuần bằng doanh thu bán trừ hoàn tiền theo NGÀY HOÀN TIỀN, kể cả đơn gốc nằm ở kỳ khác.
Số lượng bán cơ bản bằng số lượng bán nhân hệ số theo cặp mã hàng và đơn vị; không trừ hàng trả.
Đếm đơn theo mã đơn duy nhất; tỷ lệ hủy lấy mọi đơn đặt trong kỳ làm mẫu số.
Xếp hàng bán chạy theo số lượng cơ bản giảm dần, đồng hạng theo mã hàng tăng dần; ngày doanh thu cao nhất đồng hạng lấy ngày sớm nhất.
Tiền tính đồng, trung bình làm tròn nửa lên; phần trăm làm tròn hai chữ số. Ngày cuối kỳ bao gồm toàn bộ ngày.
TODAY cố định 2025-05-16. 'Tháng trước' là toàn bộ tháng 4/2025; 'tháng này đến hôm nay' là 1–16/5/2025.
Hàng ế là hàng trong danh mục không có số lượng bán dương trong kỳ. Không có bảng lương hoặc dữ liệu nhân viên.
"""
SLOTS = [
    ("gross", "number", 1, "doanh_thu_ky", False),
    ("net", "number", 3, "tra_hang", True),
    ("quantity", "number", 3, "quy_doi_don_vi", False),
    ("cancel", "percent", 2, "dem_don_khach", True),
    ("top", "list", 3, "xep_hang", False),
    ("best", "string", 3, "xep_hang", False),
    ("day", "date", 2, "doanh_thu_ky", False),
    ("exists", "boolean", 1, "loc_trang_thai", True),
    ("table", "table", 3, "noi_bang", False),
    ("unknown", "unanswerable", 1, "thieu_du_lieu", False),
    ("zero", "number", 2, "hang_e", False),
    ("average", "number", 2, "gia_tri_trung_binh", True),
]


def digest(x) -> str:
    b = x if isinstance(x, bytes) else json.dumps(x, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()


def write_json(p: Path, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def implementation_hash():
    """Cache identity includes the actual loader, prompts, runtime and scorer."""
    paths = ["eval/synth_pilot.py", "eval/model_bakeoff.py", "eval/llm.py", "eval/synth_tools.py", "eval/synth_spec.py",
             "eval/synth_execution.py", "eval/build_sft_data.py", "eval/check_sft_tokens.py", "eval/score.py", "requirements.txt",
             "datamate/loader.py", "datamate/clean.py", "datamate/agent.py", "datamate/config.yaml",
             "datamate/sandbox.py", "datamate/dm.py", "datamate/formatter.py", "datamate/verify.py",
             "datamate/prompts/system.md"]
    import numpy as np
    return digest({"files": {p: digest((ROOT / p).read_bytes()) for p in paths},
                   "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__})


def sample_schema(c, spec, candidate):
    required = ("question", "code", "loai", "spec", "slot")
    if not isinstance(c, dict) or not all(k in c for k in required):
        return False
    if not all(isinstance(c[k], str) and c[k].strip() for k in ("question", "code")):
        return False
    if c["slot"] != spec["slot"] or c["loai"] != spec["answer_type"]:
        return False
    if candidate != "baseline":
        keys = ("measure", "filters", "period", "grouping", "ranking", "unit", "tie_policy", "answer_type")
        return c["spec"] == {k: spec[k] for k in keys}
    return isinstance(c["spec"], (str, dict)) and bool(c["spec"])


def records(seed: int) -> dict:
    rng = random.Random(seed)
    data = {"items": [{"item": "P01", "name": "Vở Sao", "base": "quyển"},
                      {"item": "P02", "name": "Bút Mây", "base": "cây"},
                      {"item": "P03", "name": "Túi Lá", "base": "cái"}],
            "units": [{"item": c, "unit": u, "factor": f} for c, u, f in
                      [("P01", "quyển", 1), ("P01", "lốc", 6), ("P02", "cây", 1), ("P02", "hộp", 10), ("P03", "cái", 1)]],
            "orders": [], "lines": [], "refunds": []}
    days = ["2025-03-31 23:59:59", "2025-04-01 00:00:00", "2025-04-05 12:00:00", "2025-04-30 23:59:59",
            "2025-05-01 00:00:00", "2025-05-16 23:59:59", "2025-05-17 00:00:00"]
    for i, day in enumerate(days):
        for j, status in enumerate(["Xong", "Đã giao", "Hủy", "Chờ"]):
            oid = f"S{i:02d}{j}"
            data["orders"].append({"order": oid, "sold_at": day, "status": status})
            for k, item in enumerate(data["items"]):
                unit = rng.choice([u for u in data["units"] if u["item"] == item["item"]])
                qty = rng.randint(1, 4)
                data["lines"].append({"order": oid, "line": k + 1, "item": item["item"], "unit": unit["unit"],
                                      "qty": qty, "amount": qty * unit["factor"] * (1000 + k * 2000)})
    data["refunds"] = [{"refund": "R01", "order": "S000", "refunded_at": "2025-04-01 12:00:00", "amount": 3000},
                       {"refund": "R02", "order": "S010", "refunded_at": "2025-05-01 00:00:00", "amount": 5000}]
    return data


def plan() -> list[dict]:
    out = []
    for key, typ, hard, skill, relative in SLOTS:
        start, end = ("2025-05-01", "2025-05-16") if key in ("cancel", "average") else (TODAY, TODAY) if key == "exists" else ("2025-04-01", "2025-04-30")
        measure = {"gross": "doanh thu bán", "net": "doanh thu thuần", "quantity": "số lượng Vở Sao theo quyển",
                   "cancel": "tỷ lệ đơn hủy", "top": "top 2 tên hàng theo lượng bán cơ bản", "best": "một tên hàng bán chạy nhất",
                   "day": "ngày doanh thu bán cao nhất", "exists": "có đơn bán thành công hôm nay không",
                   "table": "doanh thu bán từng tên hàng, gồm tên và tiền", "unknown": "tổng tiền lương nhân viên",
                   "zero": "số mặt hàng ế", "average": "doanh thu bán trung bình trên mỗi đơn được chấp nhận"}[key]
        out.append({"slot": key, "measure": measure, "filters": ["item=P01"] if key == "quantity" else [],
                    "period": {"start": start, "end": end, "relative": relative, "as_of_date": TODAY},
                    "grouping": "item" if key in ("top", "best", "table") else "day" if key == "day" else "none",
                    "ranking": "quantity desc, item asc" if key in ("top", "best") else "none",
                    "unit": "VND" if key in ("gross", "net", "average", "table", "unknown") else "percent" if key == "cancel" else "base",
                    "tie_policy": "item asc; date asc", "answer_type": typ, "difficulty": hard, "skills": [skill]})
    return out


def reference(data: dict, slot: str) -> dict:
    """Independent oracle: no pandas, Kit.oracle, model code or exporter implementation."""
    from decimal import Decimal, ROUND_HALF_UP
    spec = next(s for s in plan() if s["slot"] == slot)
    start, end = spec["period"]["start"], spec["period"]["end"]
    inside = lambda d: start <= str(d)[:10] <= end
    orders = {o["order"]: o for o in data["orders"]}
    accepted = {k for k, o in orders.items() if o["status"] in ("Xong", "xong", "Đã giao", "da giao") and inside(o["sold_at"])}
    lines = [r for r in data["lines"] if r["order"] in accepted]
    amounts, quantity, dates = collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)
    factors = {(u["item"], u["unit"]): u["factor"] for u in data["units"]}
    names = {i["item"]: i["name"] for i in data["items"]}
    for r in lines:
        amounts[r["item"]] += r["amount"]
        quantity[r["item"]] += r["qty"] * factors[r["item"], r["unit"]]
        dates[orders[r["order"]]["sold_at"][:10]] += r["amount"]
    rank = sorted(names, key=lambda c: (-quantity[c], c))
    if slot == "unknown":
        return {"type": "unanswerable"}
    value = {
        "gross": lambda: sum(amounts.values()),
        "net": lambda: sum(amounts.values()) - sum(r["amount"] for r in data["refunds"] if inside(r["refunded_at"])),
        "quantity": lambda: quantity["P01"],
        "cancel": lambda: round(100 * sum(o["status"] in ("Hủy", "huỷ") for o in orders.values() if inside(o["sold_at"])) / sum(inside(o["sold_at"]) for o in orders.values()), 2),
        "top": lambda: [names[c] for c in rank[:2]], "best": lambda: names[rank[0]],
        "day": lambda: min(dates, key=lambda d: (-dates[d], d)), "exists": lambda: bool(accepted),
        "table": lambda: {"columns": ["Tên hàng", "Doanh thu"], "rows": [[names[c], amounts[c]] for c in sorted(names)]},
        "zero": lambda: sum(quantity[c] <= 0 for c in names),
        "average": lambda: int((Decimal(sum(amounts.values())) / Decimal(len(accepted))).quantize(Decimal(1), rounding=ROUND_HALF_UP)),
    }[slot]()
    return {"type": spec["answer_type"], "answer": value}


def mutations(data: dict):
    for kind in ("join_fanout", "status_alias", "return_boundary", "unit_factor", "tie", "date_boundary", "no_today", "unsold_item", "rounding_boundary"):
        d = copy.deepcopy(data)
        if kind == "join_fanout":
            d["units"].append({"item": "P01", "unit": "kiện", "factor": 30})
        elif kind == "status_alias":
            for o in d["orders"]:
                o["status"] = {"Xong": "xong", "Đã giao": "da giao", "Hủy": "huỷ"}.get(o["status"], o["status"])
        elif kind == "return_boundary":
            d["refunds"][1]["refunded_at"] = "2025-04-30 23:59:59"
        elif kind == "unit_factor":
            d["units"][1]["factor"] = 9
        elif kind == "tie":
            # Exactly equal base quantities for all items, regardless of seed.
            for r in d["lines"]:
                r["qty"], r["unit"] = 2, next(i["base"] for i in d["items"] if i["item"] == r["item"])
        elif kind == "date_boundary":
            d["orders"].append({"order": "BOUND", "sold_at": "2025-04-30 23:59:59", "status": "Xong"})
            d["lines"].append({"order": "BOUND", "line": 1, "item": "P01", "unit": "quyển", "qty": 101, "amount": 999000})
        elif kind == "no_today":
            for o in d["orders"]:
                if o["sold_at"][:10] == TODAY:
                    o["status"] = "Chờ"
        elif kind == "unsold_item":
            d["lines"] = [r for r in d["lines"] if r["item"] != "P03"]
        else:
            # Four completed May orders: +2 VND produces a .5 mean whose
            # integer part is even. Half-up and bankers rounding must differ.
            next(r for r in d["lines"] if r["order"] == "S040")["amount"] += 2
        yield kind, d


def render(data: dict, family: str) -> dict[str, pd.DataFrame]:
    if family == "joined":
        return {k: pd.DataFrame(v) for k, v in data.items()}
    orders = {o["order"]: o for o in data["orders"]}
    sales = [dict(r, **{k: v for k, v in orders[r["order"]].items() if k != "order"}) for r in data["lines"]]
    if family == "flat":
        return {"catalog": pd.DataFrame(data["items"]), "packs": pd.DataFrame(data["units"]),
                "sales": pd.DataFrame(sales), "refunds": pd.DataFrame(data["refunds"])}
    # New final family: sale and refund events in one ledger, split header state.
    events = [dict(r, event="sale", event_at=orders[r["order"]]["sold_at"]) for r in data["lines"]]
    events += [{"order": r["order"], "line": 0, "item": "", "unit": "", "qty": 0, "amount": -r["amount"],
                "event": "refund", "event_at": r["refunded_at"]} for r in data["refunds"]]
    return {"products": pd.DataFrame(data["items"]), "conversion": pd.DataFrame(data["units"]),
            "headers": pd.DataFrame(data["orders"]), "events": pd.DataFrame(events)}


def prepare(work: Path, name: str, family: str, seed: int) -> dict:
    root = work / "fixtures" / name
    truth = records(seed)
    shop = root / "data" / name
    shop.mkdir(parents=True, exist_ok=True)
    for k, df in render(truth, family).items():
        df.to_csv(shop / f"{k}.csv", index=False)
    (shop / "README.md").write_text(f"Pilot cửa hàng văn phòng phẩm {name}. Schema family {family}.\n" + RULES +
                                   "\nMọi ngày dạng ISO có giờ. amount ở dòng bán là tiền dòng; event refund có amount âm.\n" +
                                   "\n".join(f"{k}: " + ", ".join(df.columns) for k, df in render(truth, family).items()), encoding="utf-8")
    (root / "rulebook.md").write_text(RULES, encoding="utf-8")
    (root / "calendar.csv").write_text("date,lunar_year,lunar_month,lunar_day,lunar_leap\n2025-05-16,2025,4,19,false\n", encoding="utf-8")
    biz = loader.load_cached(shop, root / "cache")
    b = {"T": biz.tables, "quy_uoc": RULES, "khoang": "Giao dịch từ 31/3/2025 đến 17/5/2025. TODAY=2025-05-16."}
    path = root / "tables.pkl"
    with path.open("wb") as f:
        pickle.dump(b["T"], f)
    return {"name": name, "family": family, "seed": seed, "root": str(root), "truth": truth, "b": b, "tables": path}


def teacher_prompt(b: dict, candidate: str, baseline: Path) -> str:
    p = mb.prompt_t4(b, baseline)
    shared = ("\nKẾ HOẠCH PILOT: thay phân bố trong lời dặn trên bằng đúng 12 slot dưới đây, giữ nguyên thứ tự. "
              "Mỗi mẫu thêm slot. Kỳ relative=true phải dùng câu thời gian tương đối (TODAY nêu trong ngữ cảnh); không chèn đáp án. "
              "Riêng table: answer={'columns': [...], 'rows': [...]}; unanswerable: answer=None. "
              "Câu table dùng cột 'Tên hàng', 'Doanh thu'. Quy ước riêng của pilot ở trên được ưu tiên.\n" + json.dumps(plan(), ensure_ascii=False))
    if candidate == "baseline":
        return p + shared
    return p + shared + ("\nĐặc tả spec phải là object có đúng measure, filters, period, grouping, ranking, unit, tie_policy, answer_type. "
                         "Giữ các giá trị của slot kế hoạch. Mỗi code tự tính từ T, không ghi sẵn kết quả. "
                         "Ngày cuối kỳ lọc bằng mốc đầu ngày kế tiếp; TODAY không dùng ngày máy. "
                         "Đếm đơn trên bảng đơn hoặc drop_duplicates theo order trước khi tính tỷ lệ. "
                         "Quy đổi chỉ nối bằng cả item và unit; không nối bảng hệ số khi tính tiền. "
                         "Không đổi 'doanh thu bán' thành thuần. Hỏi thuần thì lấy hoàn tiền theo ngày hoàn, không ngày đơn gốc. "
                         "Phân biệt không có dữ liệu lương với số 0 hợp lệ của hàng ế. Không đổi ý slot để né code khó.")


def model_call(work: Path, candidate: str, phase: str, shop: str, role: str, provider: str, prompt: str, *, thinking="off", max_tokens=16000, source_hash=""):
    settings = {"provider": provider, "model": llm.config(provider)["model"], "think": thinking, "max_tokens": max_tokens, "temperature": 0, "retries": 1}
    key = digest({"prompt": prompt, "settings": settings, "candidate": candidate, "phase": phase,
                  "implementation": implementation_hash(), "source_hash": source_hash})
    path = work / "api" / candidate / phase / shop / f"{role}_{key[:16]}.json"
    if path.exists():
        return json.loads(path.read_text())["response"], True
    r = llm.chat(provider, prompt, think=thinking, max_tokens=max_tokens, temperature=0, retries=1, timeout=240,
                 tag=f"{phase}:{role}:{candidate}:{shop}", check_leak=True)
    write_json(path, {"prompt": prompt, "prompt_sha256": digest(prompt), "settings": settings, "source_hash": source_hash,
                      "implementation_sha256": implementation_hash(), "response": r})
    return r, False


def solver_prompt(b, questions):
    return mb.prompt_giai(b, questions) + "\nTODAY cố định 2025-05-16. table dùng dict columns/rows; unanswerable gán answer=None."


def coder_prompt(b, questions):
    # Authored API prompt, intentionally NOT Kit.prompt or runtime/few-shot content.
    return ("Viết Python cho bộ dữ liệu tổng hợp này. T là dict DataFrame, có pd, np, dm, TODAY (datetime.date 2025-05-16). "
            "Không nhận hoặc đoán đáp án mẫu. Đáp án cuối phải gán ANSWER bằng dm.money/number/percent/string/list_/date/boolean, "
            "dm.table(columns,rows) hoặc dm.unanswerable(reason). dm.money làm tròn đồng. "
            "Mỗi câu đúng một khối python bắt đầu '# cau k', tự tính độc lập từ T, không đọc file, không hardcode kết quả.\n" +
            mb._desc(b["T"]) + "\n" + b["quy_uoc"] + "\nCâu hỏi:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)))


def strict_output():
    return ("\nBẮT BUỘC: từng câu chạy ở tiến trình mới, không tồn tại biến của câu trước. Mỗi câu một khối python riêng. "
            "Mỗi code tối đa 35 dòng, không diễn giải dài. Sao chép nguyên tên bảng/cột. Hỏi tên phải nối danh mục trả name, không trả item. "
            "Hỏi bao nhiêu quyển phải quy đổi cả lốc sang quyển theo units, không chỉ lọc unit='quyển'. "
            "Bảng tên hàng và tiền có columns=['Tên hàng','Doanh thu'], rows gồm tên, số tiền. "
            "Mỗi chương trình phải tự đọc T từ đầu, không dùng biến của khối trước. Không chép kết quả thành hằng số.")


def runtime_output():
    return strict_output().replace("Mỗi chương trình phải tự đọc T từ đầu", "Mỗi chương trình dùng T đã được runtime cấp sẵn") + (
        "\nQUAN TRỌNG: môi trường ĐÃ CÓ T, pd, np, dm, TODAY. TUYỆT ĐỐI KHÔNG gán T=..., không tạo bảng dữ liệu mẫu, "
        "không sao chép các dòng minh họa, không tạo DataFrame thay thế dữ liệu. T chỉ được đọc. "
        "Khối câu k không được dùng cả biến lẫn import của câu khác; ưu tiên pd.to_datetime thay import thêm. "
        "Luôn bắt đầu khối bằng đúng '# cau k'. Mỗi khối tự lọc bảng thật trong T, rồi gán ANSWER=dm.<kiểu>(kết quả).")


def money_output():
    return runtime_output() + ("\nTiền trung bình: truyền trực tiếp giá trị chưa làm tròn cho dm.money(), để runtime làm tròn nửa lên. "
                               "Không dùng round(), np.round(), int() trước dm.money() với giá trị trung bình: banker rounding và cắt phần lẻ sai ở mốc nửa đồng. "
                               "Các năm, ngưỡng và top-k trong câu hỏi vẫn là hằng số hợp lệ.")


def style(q: str, index: int) -> str:
    # Deterministic self-authored transformation, never copies train examples.
    return q.replace("Doanh thu", "dthu").replace("bao nhiêu", "bnhiu") if index % 3 == 0 else q


def raw_execute(code, fixture, work, tag):
    if not code:
        return None, "missing_code"
    return mb.run_code(code, fixture["tables"], work, tag)


def run(work: Path, candidate: str, phase: str, fixtures: list[dict], baseline: Path) -> list[dict]:
    from datamate import formatter, verify
    (work / "execution").mkdir(exist_ok=True)
    (work / "execution/harness.py").write_text(mb.HARNESS)
    result = []
    for f in fixtures:
        checkpoint = work / "runs" / candidate / phase / (f["name"] + ".json")
        config_hash = digest({"candidate": candidate, "fixture": f["truth"], "family": f["family"], "prompt": teacher_prompt(f["b"], candidate, baseline),
                              "implementation": implementation_hash(), "models": {p: llm.config(p)["model"] for p in llm.PROVIDERS}})
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            if old["config_hash"] != config_hash:
                raise ValueError("checkpoint config changed: choose new output run")
            result.extend(old["rows"])
            continue
        t0 = time.monotonic()
        teacher, _ = model_call(work, candidate, phase, f["name"], "teacher", "gemini", teacher_prompt(f["b"], candidate, baseline), thinking="high", max_tokens=24000, source_hash=digest(f["truth"]))
        try:
            items = llm.extract_json(teacher["content"]) if teacher.get("finish") == "stop" else []
        except ValueError:
            items = []
        items = items if isinstance(items, list) else []
        questions = [style(str(c.get("question", "")), i) for i, c in enumerate(items[:12])]
        if questions:
            suffix = strict_output() if candidate != "baseline" else ""
            gpt, _ = model_call(work, candidate, phase, f["name"], "verifier", "gpt", solver_prompt(f["b"], questions) + suffix, source_hash=digest(f["truth"]))
            coder_suffix = money_output() if candidate == "runtime_money" else runtime_output() if candidate == "runtime" else suffix
            qwen, _ = model_call(work, candidate, phase, f["name"], "coder", "qwen", coder_prompt(f["b"], questions) + coder_suffix, source_hash=digest(f["truth"]))
            subset = [i for i in (1, 2, 4, 8) if i < len(questions)]
            weak, _ = model_call(work, candidate, phase, f["name"], "weak", "deepseek", solver_prompt(f["b"], [questions[i] for i in subset]), source_hash=digest(f["truth"]))
            gs = mb.tach_khoi(gpt["content"], len(questions)) if gpt.get("finish") == "stop" else [None] * len(questions)
            qs = mb.tach_khoi(qwen["content"], len(questions)) if qwen.get("finish") == "stop" else [None] * len(questions)
            ws = mb.tach_khoi(weak["content"], len(subset)) if weak.get("finish") == "stop" else [None] * len(subset)
        else:
            gs, qs, ws, subset = [], [], [], []
        kit = Kit(data=f["root"], cache=Path(f["root"]) / "cache")
        rows = []
        for i, spec in enumerate(plan()):
            c = items[i] if i < len(items) and isinstance(items[i], dict) else {}
            q = questions[i] if i < len(questions) else ""
            gold = reference(f["truth"], spec["slot"])
            row = {"id": digest([f["name"], spec["slot"], q])[:24], "candidate": candidate, "phase": phase, "shop": f["name"], "family": f["family"],
                   "seed": f["seed"], "data_root": f["root"], "source_hash": digest(f["truth"]), "slot": spec["slot"], "plan": spec,
                   "question": q, "question_canonical": c.get("question", ""), "as_of_date": TODAY, "gold": gold,
                   "teacher_code": c.get("code", ""), "runtime_code": qs[i] if i < len(qs) else None,
                   "verifier_code": gs[i] if i < len(gs) else None,
                   "schema_pass": sample_schema(c, spec, candidate),
                   "reject": [], "mutations": [], "weak_status": "not_measured"}
            for role, code in [("teacher", c.get("code")), ("verifier", gs[i] if i < len(gs) else None)]:
                a, err = raw_execute(code, f, work / "execution", f"{candidate}_{phase}_{f['name']}_{role}_{i}")
                typed = mb.typed(a, gold["type"], q) if err is None else None
                row[role + "_pass"] = bool(typed and kit.check(verify.round_money(q, typed), gold) == 1)
                row[role + "_answer"], row[role + "_error"] = a, err
            if i in subset:
                a, err = raw_execute(ws[subset.index(i)], f, work / "execution", f"weak_{f['name']}_{i}")
                typed = mb.typed(a, gold["type"], q) if err is None else None
                row["weak_status"] = "invalid_output" if err or not typed or typed["type"] == "abstain" else "executed_correct" if kit.check(typed, gold) == 1 else "executed_wrong"
            if row["runtime_code"]:
                from synth_execution import preflight
                issue = preflight(row["runtime_code"])
                ex = {"ok": False, "error": issue, "issues": [], "answer": None} if issue else kit.execute(f["name"], row["runtime_code"], TODAY, q)
                row["runtime_pass"] = ex["ok"] and not ex["issues"] and kit.check(ex["answer"], gold) == 1
                row["runtime_result"] = ex
            else:
                row["runtime_pass"] = False
            for key in ("schema_pass", "teacher_pass", "verifier_pass", "runtime_pass"):
                if not row[key]:
                    row["reject"].append(key)
            row["style_pass"] = row["verifier_pass"]
            row["wall_s_shop"] = round(time.monotonic() - t0, 3)
            rows.append(row)
        write_json(checkpoint, {"config_hash": config_hash, "teacher": teacher, "rows": rows, "wall_s": time.monotonic() - t0})
        result.extend(rows)
        print(json.dumps({"candidate": candidate, "phase": phase, "shop": f["name"], "requested": 12, "base_pass": sum(not r["reject"] for r in rows),
                          "rejects": dict(collections.Counter(e for r in rows for e in r["reject"]))}), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, default=llm.DEFAULT_LEDGER, help="same HTTP ledger for every generation entry point")
    ap.add_argument("--candidate", choices=["baseline", "structured", "runtime", "runtime_money"], required=True)
    ap.add_argument("--phase", choices=["dev", "validation", "final", "expanded"], required=True)
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    llm.configure_run(a.ledger, limit=llm.HARD_CAP, provider_limits={p: 2 for p in llm.PROVIDERS})
    import sqlite3
    with sqlite3.connect(a.ledger) as db:
        if a.phase in ("dev", "validation") and db.execute("select count(*) from requests").fetchone()[0] >= 650:
            raise ValueError("650 requests reached: reserve remaining budget for confirmation")
    seeds = {"dev": [("pilot_dev_joined", "joined", 41011)], "validation": [("pilot_val_flat", "flat", 52023)],
             "final": [("pilot_final_events", "events", 93071)],
             "expanded": [("pilot_train_joined_2", "joined", 61003), ("pilot_val_flat_2", "flat", 71009), ("pilot_train_events_2", "events", 81017)]}
    fs = [prepare(a.work, *s) for s in seeds[a.phase]]
    rows = run(a.work, a.candidate, a.phase, fs, a.work / "prompt8_baseline.md")
    write_json(a.work / f"{a.candidate}_{a.phase}_rows.json", rows)


if __name__ == "__main__":
    main()
