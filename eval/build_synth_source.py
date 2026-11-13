"""Dựng data/synth_source: một nguồn duy nhất để sinh synthetic data. Chỉ đọc data/train (luật 6), không cần GPU.

    python eval/build_synth_source.py                      # ghi data/synth_source
    python eval/build_synth_source.py --out /tmp/x --workers 8 --cache outputs/cache

Nội dung (xem data/synth_source/README.md sau khi dựng):
- shops/: hồ sơ 30 shop (ngành, kiểu sổ, bảng, cột, vai trò cột, khả năng), mô tả dữ liệu đúng như prompt model thấy
- catalogs/: danh mục hàng, nhóm, quy đổi đơn vị theo ngành
- seeds/: 3.000 câu train kèm đáp án chuẩn, đường đi (lớp tính toán hay model), kỹ năng, kỳ, cách viết;
  câu khó tính sẵn (lớp tính toán bỏ qua, bỏ qua khi đổi tên cột / làm nhiễu, oracle sai); kho code mẫu đã chạy lại
- rules/: rulebook tách dòng, dòng nào lớp tính toán canh, biến thể luật soạn sẵn
- taxonomy/, style/, guard/, prompts/, errors/ (eval/mine_errors.py ghi vào), splits.json, manifest.json
Phải chạy bằng pandas==2.2.3 (requirements.txt): loader ra dữ liệu khác nếu sai phiên bản.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

import synth_spec as spec  # noqa: E402

DATA = ROOT / "data" / "train"
NOISE_KINDS = ["bo_dau", "viet_tat", "telex", "dem", "dao_ky"]
SCHEMA_MODES = ["style", "syn", "syn_table"]
SECTION_SKILLS = [("1.", ["K2", "K3", "K7"]), ("2.", ["K1"]), ("3.", ["K6", "K3", "K4"]), ("4.", ["K8"]), ("5.", ["K9"]),
                  ("Cà phê", ["K10"]), ("Nhà thuốc", ["K10"]), ("Vật liệu", ["K10"]), ("6.", ["K10"]), ("7.", ["K11"]),
                  ("8.", ["K12", "K5"]), ("9.", ["K13"]), ("10.", ["K4"])]
AUX = [("nhap_hang", r"nhap"), ("cong_no", r"cong_no|congno|thanh_toan|thanhtoan|phieu_thu|so_quy|soquy|\bno\b"), ("ton_kho", r"ton|kho|\blo\b|lo_hang"),
       ("danh_muc", r"danh_muc|danhmuc|hang_hoa|hanghoa|san_pham|sanpham|thuoc|quy_doi|quydoi|bang_gia"), ("khach", r"khach"),
       ("giao", r"giao"), ("tra_hang", r"tra_hang|trahang"), ("chi_phi", r"chi_phi|chiphi")]


def jl(path: Path, rows) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def jw(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def splits() -> dict:
    """Cùng cách chia với eval/make_hard_set.py: holdout cố định, tune = 8 shop ngẫu nhiên (hạt 'hard-tune') ngoài holdout."""
    hold = (ROOT / "eval/holdout_businesses.txt").read_text().split()
    shops = sorted(p.name for p in (DATA / "data").iterdir() if p.is_dir())
    rest = sorted({q["business_id"] for q in load_jsonl(DATA / "questions.jsonl")} - set(hold))
    tune = sorted(random.Random("hard-tune").sample(rest, 8))
    mining = sorted(set(shops) - set(hold) - set(tune))
    return {"mining": mining, "tune": tune, "holdout": sorted(hold),
            "allowed_for_generation": ["mining"],
            "quy_tac": {"mining": "được dùng làm seed, lấy dữ liệu shop, đưa câu hỏi vào train",
                        "tune": "chỉ để chọn ngưỡng/cấu hình (make_hard_set.py --split tune); không đưa câu hỏi vào train",
                        "holdout": "chỉ đo một lần cuối (make_hard_set.py); không dùng câu hỏi, dữ liệu shop hay log chi tiết để sinh"},
            "cach_chia": "holdout: eval/holdout_businesses.txt; tune: random.Random('hard-tune').sample(shop ngoài holdout, 8) như eval/make_hard_set.py"}


# ---------------------------------------------------------------- việc của từng shop (chạy trong tiến trình con)
def _aux_kinds(table: str) -> list[str]:
    from datamate.clean import norm_key
    k = norm_key(table).replace(" ", "_")
    return [name for name, pat in AUX if re.search(pat, k)]


def _table_info(df, roles: dict) -> dict:
    inv = {c: r for r, c in roles.items()}
    cols = []
    for c in df.columns:
        if str(c).startswith("_"):
            continue
        s = df[c]
        nn = s.dropna()
        ex = [str(x)[:40] for x in nn.drop_duplicates().head(3).tolist()]
        cols.append({"name": str(c), "dtype": str(s.dtype), "role": inv.get(c), "null_pct": round(100 * (1 - len(nn) / max(len(s), 1)), 1),
                     "n_unique": int(nn.astype(str).nunique()), "examples": ex})
    return {"rows": int(len(df)), "columns": cols}


def _conversions(biz, roles: dict) -> list[dict]:
    out = []
    for line in biz.readme.splitlines():
        m = re.match(r"^\s*-\s*(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", line)
        if m:
            out.append({"item": m[1].strip(), "big_unit": m[2].strip(), "factor": float(m[3].replace(",", ".")), "base_unit": m[4].strip(), "source": "README"})
    for t, r in roles.items():
        if {"item", "factor"} <= set(r) and ("big_unit" in r or "unit" in r):
            df = biz.tables[t]
            bu = r.get("big_unit") or r.get("unit")
            for _, row in df[[r["item"], bu, r["factor"]]].dropna().iterrows():
                try:
                    f = float(str(row[r["factor"]]).replace(",", "."))
                except ValueError:
                    continue
                out.append({"item": str(row[r["item"]]), "big_unit": str(row[bu]), "factor": f, "base_unit": None, "source": t})
    return out


def shop_work(arg):
    biz_id, cache, rulebook, role = arg
    import pandas as pd
    from datamate import agent as agent_mod, loader, shop as shop_mod, templates, templates_ext
    from score import match
    from stress_noise import noise
    from stress_schema import rename_all
    warnings.filterwarnings("ignore")
    templates_ext.RULEBOOK = rulebook
    gold = {g["id"]: g for g in load_jsonl(DATA / "answers.jsonl")}
    qs = [q for q in load_jsonl(DATA / "questions.jsonl") if q["business_id"] == biz_id]
    biz = loader.load_cached(DATA / "data" / biz_id, cache)
    roles = shop_mod._table_roles(biz)
    s = shop_mod.build(biz)

    def route(shop, text, q):
        if shop is None:
            return None, None
        try:
            a = templates.answer(text, q["as_of_date"], shop)
        except Exception:  # noqa: BLE001
            return None, None
        if a is None:
            return None, None
        return a["template"], match({"id": q["id"], **a}, gold[q["id"]]) > 0

    variants, build_errors = {}, {}
    for m in SCHEMA_MODES:
        try:
            variants[m] = shop_mod.build(rename_all(loader.load_cached(DATA / "data" / biz_id, cache), m))
        except Exception as e:  # noqa: BLE001 — shop.build lỗi với tên cột lạ: lớp tính toán coi như bỏ qua cả shop
            variants[m], build_errors[m] = None, f"{type(e).__name__}: {e}"[:200]
    qrows = []
    for q in qs:
        t, ok = route(s, q["question"], q)
        row = {"id": q["id"], "template": t, "ok": ok, "schema": {}, "noise": {}}
        for m in SCHEMA_MODES:
            row["schema"][m] = route(variants[m], q["question"], q)[0]
        for k in NOISE_KINDS:
            text = noise(q["question"], k, random.Random(q["id"] + k))
            row["noise"][k] = {"question": text, "template": route(s, text, q)[0]}
        qrows.append(row)

    lines = s.lines if s is not None else pd.DataFrame()
    sales_roles = sorted(c for c in lines.columns if c not in ("order", "amount") and lines[c].notna().any()) if s is not None else []
    aux = sorted({k for t in biz.tables for k in _aux_kinds(t)})
    has = set(sales_roles)
    caps = {"doanh_thu": s is not None, "dem_don": bool(s is not None and s.has_orders), "mat_hang": bool(s is not None and s.items),
            "nhom_hang": bool(s is not None and s.groups), "so_luong": "qty_base" in has, "khach": bool({"cust", "cust_code"} & has),
            "nhom_khach": "cust_group" in has, "ca": "time" in has, "kenh": "channel" in has, "thanh_toan": "pay" in has, "ke_don": "rx" in has,
            "gia_von": "cost" in has, "nhan_vien": "staff" in has, "nhap_hang": "nhap_hang" in aux, "ton_kho": "ton_kho" in aux,
            "cong_no": "cong_no" in aux, "giao_hang": "giao" in aux, "chi_phi": "chi_phi" in aux}
    dates = lines["date"].dropna() if "date" in lines else pd.Series(dtype="datetime64[ns]")
    cust = lines["cust"].dropna().astype(str) if "cust" in lines else pd.Series(dtype=str)
    status = lines["status"].value_counts().to_dict() if "status" in lines else {}
    templ = collections.Counter(r["template"] for r in qrows if r["template"] and r["ok"])
    profile = {
        "business_id": biz_id, "role": role, "industry": spec.industry_of(s.items if s is not None else []),
        "archetype": spec.archetype_of(biz.readme), "readme_head": biz.readme.splitlines()[2] if len(biz.readme.splitlines()) > 2 else "",
        "files": sorted(f.name for f in (DATA / "data" / biz_id).iterdir() if f.name != "README.md"),
        "tables": {t: _table_info(df, roles.get(t, {})) for t, df in biz.tables.items()},
        "table_roles": roles, "sales_roles": sales_roles, "aux_tables": aux, "capabilities": caps,
        "S": s is not None, "has_orders": bool(s.has_orders) if s is not None else None, "reliable": bool(s.reliable) if s is not None else None,
        "date_min": str(dates.min())[:10] if len(dates) else None, "date_max": str(dates.max())[:10] if len(dates) else None,
        "n_sales_lines": int(len(lines)), "n_items": len(s.items) if s is not None else 0, "n_groups": len(s.groups) if s is not None else 0,
        "n_customers": int(cust.nunique()), "status_counts": {str(k): int(v) for k, v in status.items()},
        "oracle_templates": dict(sorted(templ.items())), "n_questions": len(qs),
        "schema_variant_build_errors": build_errors,
    }
    catalog = {"items": list(s.items) if s is not None else [], "groups": list(s.groups) if s is not None else [],
               "conversions": _conversions(biz, roles)}
    return {"profile": profile, "block": agent_mod.business_block(biz), "catalog": catalog, "questions": qrows}


# ---------------------------------------------------------------- phần chạy trong tiến trình chính
def k_tags(question: str, template: str | None, gold_type: str, periods: list[str]) -> list[str]:
    tags = set(spec.TEMPLATES[template][1]) if template in spec.TEMPLATES else set()
    if template is None:
        t = question.lower()
        tags |= {k for k, pat in spec.KEYWORD_SKILLS if re.search(pat, t)}
    if set(periods) & {"tuan_nay", "tuan_truoc", "thang_nay", "quy", "n_ngay", "tet", "cung_ky", "den_het", "hom_qua"}:
        tags.add("K1")
    if gold_type == "unanswerable":
        tags.add("K11")
    if gold_type in ("list", "table", "date", "boolean"):
        tags.add("K13")
    return sorted(tags, key=lambda x: int(x[1:]))


def style_of(question: str, known=frozenset()) -> dict:
    """Văn phong một câu: không dấu, viết tắt (gồm từ tiếng Anh), từ đệm, gõ telex sót, chữ thường đầu câu, thiếu dấu hỏi, số từ."""
    t = question.lower()
    abbr = [a for a in spec.ABBREVIATIONS if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", t)]
    fill = [f for f in spec.FILLERS if re.search(rf"(?<!\w){re.escape(f)}(?!\w)", t)]
    return {"no_diacritic": not spec.DIACRITIC.search(t), "abbreviations": abbr, "fillers": fill,
            "telex": spec.telex_words(question, known), "lower_start": question.strip()[:1].islower(), "no_qmark": "?" not in question,
            "n_words": len(t.split())}


def known_words(cats: dict) -> frozenset:
    """Từ ASCII trong tên hàng/nhóm và trong danh sách viết tắt: không tính là gõ telex sót."""
    text = " ".join(k for c in cats.values() for k in list(c["items"]) + list(c["groups"])).lower()
    return frozenset(re.findall(r"[a-z]+", text)) | {w for a in spec.ABBREVIATIONS for w in a.split()}


def periods_of(question: str) -> list[str]:
    t = question.lower()
    return [k for k, p in spec.PERIODS if re.search(p, t)]


def parse_rulebook(text: str) -> list[dict]:
    """Mỗi dòng luật thành một bản ghi: dòng bảng, gạch đầu dòng và đoạn văn có nội dung luật."""
    out, section = [], ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            section = s.lstrip("#").strip()
            continue
        if s.startswith("|"):
            if set(s) <= set("|-: "):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells[0] in ("Luật", "Cách nói", "Cách ghi", "Từ", "type"):
                continue
            out.append({"section": section, "kind": "table_row", "name": re.sub(r"[*`]", "", cells[0]).strip(), "text": s})
        elif s.startswith("- "):
            out.append({"section": section, "kind": "bullet", "name": re.sub(r"[*`]", "", s[2:].split(":")[0])[:80].strip(), "text": s})
        elif section:
            out.append({"section": section, "kind": "paragraph", "name": re.sub(r"[*`]", "", s)[:60].strip(), "text": s})
    for i, r in enumerate(out, 1):
        r["id"] = f"r{i:03d}"
    return out


def verify_code_bank(cache: Path, questions: dict, gold: dict, role_of: dict, workers: int) -> list[dict]:
    from datamate import formatter, loader, verify
    from datamate.sandbox import run_code
    from score import match
    bank = load_jsonl(ROOT / "data/code_bank.jsonl")

    def one(r):
        q = questions[r["id"]]
        pkl = loader.cache_file(DATA / "data" / r["business_id"], cache)
        res = run_code(r["code"], pkl, DATA / "calendar.csv", q["as_of_date"], 120)
        ans, ok = None, False
        if res.get("ok"):
            ans = verify.round_money(q["question"], formatter.finalize(q["id"], res["answer"]))
            ok = match(ans, gold[q["id"]]) > 0
        code = r["code"]
        smells = [name for name, pat in (("dem_dong_len", r"\blen\(\s*\w+\s*[\[\)]"), ("dem_dong_shape", r"\.shape\[0\]"),
                                          ("so_chuoi_bang_eq", r"==\s*['\"][^'\"]{3,}['\"]"), ("ham_main", r"def main\(")) if re.search(pat, code)]
        return {"id": r["id"], "business_id": r["business_id"], "role": role_of[r["business_id"]], "question": q["question"], "as_of_date": q["as_of_date"],
                "code": code, "verified": ok, "answer": ans, "error": None if res.get("ok") else str(res.get("error", ""))[:500],
                "uses_S": bool(re.search(r"\bS\.", code)), "smells": smells}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, bank))


def write_readme(out: Path, stats: dict) -> None:
    s = stats
    text = f"""# Nguồn sinh dữ liệu (synth_source)

Một chỗ duy nhất chứa mọi thứ cần để sinh synthetic data cho Qwen3-14B. Dựng lại bằng
`python eval/build_synth_source.py` (pandas 2.2.3). Chỉ lấy từ `data/train`; **không** đọc public/private (luật 6).
Dựng lúc {s['created_at']}, code `{s['code_version']}` (commit `{s['git_commit']}`).

Hai giai đoạn:
1. **Sinh dữ liệu shop** (file thô + README + lớp sự thật): đặc tả và kế hoạch ở `data_spec/` (dựng bằng `python eval/build_data_spec.py`).
2. **Sinh câu hỏi + đáp án** từ dữ liệu đã kiểm: `seeds/`, `prompts/1–6`, `rules/`, `taxonomy/`, `style/`, `guard/`, `errors/`, `eval/synth_tools.py`.

## Quy tắc dùng
1. Chỉ sinh từ shop có `role = "mining"` ({len(s['splits']['mining'])} shop). Shop `tune` ({len(s['splits']['tune'])}) và `holdout` ({len(s['splits']['holdout'])}) để đo; xem `splits.json`.
2. Mọi mẫu sinh ra phải qua lọc trùng với `guard/eval_questions.jsonl` (câu của tune và holdout) trước khi đưa vào train.
3. Đáp án lấy từ oracle (`eval/synth_tools.py`, `Kit.oracle`) hoặc từ chạy code đã khớp đáp án chuẩn. Dạng câu có `oracle_status != "ok"`
   trong `taxonomy/templates.jsonl` thì không dùng oracle một mình.
4. Biến thể luật (`rules/counterfactual.jsonl`): oracle **không** tính được luật mới. Khi bật biến thể, `Kit.oracle` tự từ chối các câu
   khớp `trigger` hoặc thuộc `affected_templates`; đáp án cho câu chịu ảnh hưởng phải lấy từ ≥2 lời giải độc lập khớp nhau và được kiểm tay.

## Thư mục
| Đường dẫn | Nội dung | Số dòng |
|---|---|---|
| `data_spec/` | **Giai đoạn 1.** Đặc tả dữ liệu trích từ train: kiểu sổ, tên cột theo vai trò, định dạng giá trị, bẫy, cách viết README, thống kê hiệu chỉnh theo ngành, độ phức tạp shop nối với kết quả log; lược đồ lớp sự thật, khuôn danh mục ngành mới, công thức xuất; ngành mới, khối lượng, chia việc cho 3 model, kết quả đo model; kế hoạch (README) | – |
| `splits.json` | Vai trò shop: mining / tune / holdout | – |
| `shops/index.jsonl` | Hồ sơ shop: ngành, kiểu sổ, bảng, cột (kiểu, % trống, ví dụ, vai trò), khả năng, dạng câu oracle trả lời được | {s['n_shops']} |
| `shops/<biz>.md` | Phần "DỮ LIỆU CỬA HÀNG" đúng như prompt model thấy (README + mô tả bảng) | {s['n_shops']} |
| `catalogs/<ngành>.json` | Danh mục hàng, nhóm hàng, quy đổi đơn vị theo ngành | {s['n_catalogs']} |
| `seeds/questions.jsonl` | 3.000 câu train: đáp án chuẩn, vai trò shop, đường đi (dạng câu lớp tính toán hoặc `null` = chuyển model), nhóm kỹ năng, nhãn K, kỳ, cách viết, dòng luật liên quan | {s['n_questions']} |
| `seeds/hard_cases.jsonl` | Câu khó tính được không cần model: lớp tính toán bỏ qua, bỏ qua khi đổi tên cột hoặc làm nhiễu, oracle sai; kèm câu đã làm nhiễu | {s['n_hard']} |
| `seeds/code_bank.jsonl` | Kho code mẫu chạy lại trên code hiện tại: `verified` = khớp đáp án chuẩn; `usable_for_training` = khớp và thuộc shop mining; `smells` là dấu hiệu dò bằng regex (đếm dòng bằng len/shape, so chuỗi bằng ==), cần xem tay | {s['n_code']} ({s['n_code_ok']} khớp) |
| `rules/rulebook_train.md` | Bản sao rulebook train | – |
| `rules/rules.jsonl` | Rulebook tách dòng: mục, nhãn K, lớp tính toán có canh dòng này không (`guarded`), dạng câu liên quan | {s['n_rules']} |
| `rules/counterfactual.jsonl` | Biến thể luật soạn sẵn (thay hoặc thêm dòng), trigger, dạng câu bị ảnh hưởng | {s['n_cf']} |
| `taxonomy/skills.json` | Nhãn kỹ năng K1–K14, nhóm dạng câu | – |
| `taxonomy/templates.jsonl` | 68 dạng câu của oracle: nhóm, nhãn K, dòng luật, số câu theo vai trò, số sai, ví dụ (chỉ mining) | {s['n_templates']} |
| `taxonomy/error_groups.json` | Nhóm kết quả E1–E6b, nhãn phụ, mã lỗi trong log, loại sửa lỗi | – |
| `taxonomy/known_errors.jsonl` | Mẫu lỗi đã biết và cách đúng | {s['n_known']} |
| `style/style_profile.json` | Phân bố cách viết, viết tắt, từ đệm, kỳ, độ dài, câu mẫu (đo trên câu mining) | – |
| `guard/eval_questions.jsonl` | Câu tune + holdout, **chỉ để lọc trùng** | {s['n_guard']} |
| `prompts/` | Prompt cho teacher: 7 = danh mục ngành mới (giai đoạn 1); 1–6 = câu hỏi từ đặc tả, viết lại, đọc ngược, biến thể luật, chẩn đoán lỗi, code (giai đoạn 2) | – |
| `errors/<lượt>/` | Log chạy thật đã khai thác (`eval/ingest_real_logs.py` từ `data/real_logs/`, hoặc `eval/mine_errors.py`): questions, positives, negatives, summary, by_shop; lượt public chỉ có summary thống kê; danh sách ở `errors/index.json` | – |
| `manifest.json` | Phiên bản, hash đầu vào và từng file | – |

## Số liệu chính
- Câu train chuyển cho model (lớp tính toán bỏ qua): {s['n_model_routed']} / 3.000; trong shop mining: {s['n_model_routed_mining']}.
- Câu mining dùng làm seed được: {s['n_seed']}. Câu khó trong mining: {s['n_hard_mining']}.
- Ngành: {s['industries']}.
- Kiểu sổ: {s['archetypes']}.

## Cách dùng nhanh
```python
import sys; sys.path[:0] = ["eval", "."]
from synth_tools import Kit
kit = Kit()                                   # đọc data/synth_source + data/train + outputs/cache
seeds = kit.seeds(roles=["mining"])           # câu seed
kit.oracle("biz_101", "Doanh thu tháng 5/2026 là bao nhiêu?", "2026-08-24")  # đáp án oracle hoặc None
kit.execute("biz_101", code, "2026-08-24")    # chạy code teacher trong sandbox
kit.prompt(q)                                 # messages đúng như lúc chạy (system + user)
kit.rulebook(["cf_tuan_chu_nhat"])            # rulebook đã áp biến thể
kit.contaminated("câu hỏi mới")               # trùng câu tune/holdout?
```
Khai thác log: đặt lượt chạy vào `data/real_logs/<lượt>/` rồi `python eval/ingest_real_logs.py`, hoặc `python eval/mine_errors.py OUT/<lượt> --name <lượt>` → `errors/<lượt>/`.
Gọi teacher: `eval/llm.py` (khoá trong `.env`, kiểm bằng `python eval/llm.py check`).
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/synth_source"))
    ap.add_argument("--cache", default=str(ROOT / "outputs/cache"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    import pandas as pd
    if pd.__version__ != "2.2.3":
        print(f"CẢNH BÁO: pandas {pd.__version__}, requirements.txt ghi 2.2.3; dữ liệu nạp có thể khác bản đã chấm", file=sys.stderr)
    from datamate import __version__, loader, rules as rules_mod, templates_ext
    from datamate.clean import norm_key
    from datamate.normq import normalize

    out, cache = Path(a.out), Path(a.cache)
    for sub in ("shops", "catalogs", "seeds", "rules", "taxonomy", "style", "guard", "prompts"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    out.mkdir(parents=True, exist_ok=True)
    sp = splits()
    role_of = {b: r for r in ("mining", "tune", "holdout") for b in sp[r]}
    jw(out / "splits.json", sp)
    rulebook = (DATA / "rulebook.md").read_text(encoding="utf-8")
    questions = {q["id"]: q for q in load_jsonl(DATA / "questions.jsonl")}
    gold = {g["id"]: g for g in load_jsonl(DATA / "answers.jsonl")}

    # 1) từng shop: hồ sơ, mô tả như prompt, đường đi của câu hỏi (gốc, đổi tên cột, làm nhiễu)
    for b in sorted(role_of):  # nạp tuần tự trước để các tiến trình con chỉ đọc cache
        loader.load_cached(DATA / "data" / b, cache)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(shop_work, [(b, cache, rulebook, role_of[b]) for b in sorted(role_of)]))
    profiles = [r["profile"] for r in res]
    jl(out / "shops/index.jsonl", profiles)
    for r in res:
        (out / "shops" / f"{r['profile']['business_id']}.md").write_text(r["block"] + "\n", encoding="utf-8")
    routes = {q["id"]: q for r in res for q in r["questions"]}
    prof = {p["business_id"]: p for p in profiles}

    # 2) danh mục theo ngành
    cats = collections.defaultdict(lambda: {"shops": [], "items": collections.Counter(), "groups": collections.Counter(), "conversions": {}})
    for r in res:
        ind = r["profile"]["industry"]
        c = cats[ind]
        c["shops"].append(r["profile"]["business_id"])
        c["items"].update(r["catalog"]["items"])
        c["groups"].update(r["catalog"]["groups"])
        for cv in r["catalog"]["conversions"]:
            c["conversions"].setdefault((cv["item"], cv["big_unit"]), cv)
    for ind, c in cats.items():
        jw(out / "catalogs" / f"{ind}.json", {"industry": ind, "ten": spec.INDUSTRY_NAMES.get(ind, ind), "shops": sorted(c["shops"]),
                                              "rule_section": spec.INDUSTRY_RULE_SECTION.get(ind),
                                              "items": [{"name": k, "n_shops": v} for k, v in sorted(c["items"].items())],
                                              "groups": [{"name": k, "n_shops": v} for k, v in sorted(c["groups"].items())],
                                              "conversions": sorted(c["conversions"].values(), key=lambda x: (x["item"], x["big_unit"]))})

    known = known_words(cats)

    # 3) seeds
    rb_rules = rules_mod.parse(rulebook)
    unknown = sorted({r["template"] for r in routes.values() if r["template"] and r["template"] not in spec.TEMPLATES})
    if unknown:
        raise SystemExit(f"Dạng câu chưa có trong synth_spec.TEMPLATES: {unknown}")
    seeds, hard = [], []
    for qid, q in questions.items():
        rt, g, b = routes[qid], gold[qid], q["business_id"]
        per = periods_of(q["question"])
        t = rt["template"]
        role = role_of[b]
        seeds.append({
            "id": qid, "business_id": b, "role": role, "usable_as_seed": role == "mining",
            "industry": prof[b]["industry"], "archetype": prof[b]["archetype"],
            "question": q["question"], "question_norm": normalize(q["question"]), "as_of_date": q["as_of_date"],
            "gold": {"type": g["type"], "answer": g.get("answer"), "alias": g.get("alias", [])},
            "route": {"template": t, "template_correct": rt["ok"], "to_model": t is None},
            "skill_family": spec.TEMPLATES[t][0] if t else None, "k_tags": k_tags(q["question"], t, g["type"], per),
            "periods": per, "style": style_of(q["question"], known),
            "rule_hints": [r.name for r in rules_mod.relevant(rb_rules, q["question"], 3)],
        })
        reasons = []
        if t is None:
            reasons.append("lop_tinh_toan_bo_qua")
        elif not rt["ok"]:
            reasons.append("oracle_sai")
        miss_schema = [m for m in SCHEMA_MODES if t and rt["schema"][m] is None]
        miss_noise = [k for k in NOISE_KINDS if t and rt["noise"][k]["template"] is None]
        reasons += [f"bo_qua_khi_doi_ten_cot:{m}" for m in miss_schema] + [f"bo_qua_khi_lam_nhieu:{k}" for k in miss_noise]
        if reasons:
            hard.append({"id": qid, "business_id": b, "role": role, "usable_as_seed": role == "mining", "question": q["question"],
                         "as_of_date": q["as_of_date"], "gold": {"type": g["type"], "answer": g.get("answer")}, "template": t, "reasons": reasons,
                         "noisy_variants": {k: rt["noise"][k]["question"] for k in miss_noise},
                         "k_tags": k_tags(q["question"], t, g["type"], per)})
    seeds.sort(key=lambda r: (r["business_id"], r["id"]))
    hard.sort(key=lambda r: (r["business_id"], r["id"]))
    jl(out / "seeds/questions.jsonl", seeds)
    jl(out / "seeds/hard_cases.jsonl", hard)
    bank = verify_code_bank(cache, questions, gold, role_of, a.workers)
    tmpl_of = {s["id"]: s["route"]["template"] for s in seeds}
    for r in bank:
        r["template"] = tmpl_of[r["id"]]
        r["usable_for_training"] = r["verified"] and r["role"] == "mining"
    jl(out / "seeds/code_bank.jsonl", bank)

    # 4) rulebook
    (out / "rules").mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / "rulebook.md", out / "rules/rulebook_train.md")
    rule_lines = json.loads((ROOT / "datamate/rule_lines.json").read_text(encoding="utf-8"))
    tcount = collections.Counter(s["route"]["template"] for s in seeds if s["route"]["template"])
    rrows = parse_rulebook(rulebook)
    for r in rrows:
        full_guards = [v for v in rule_lines.values() if v.strip() and v.strip() in r["text"]]
        r["guarded"] = bool(full_guards)
        r["templates"] = sorted(t for t, frag in templates_ext.RULE_OF.items() if rule_lines.get(frag, frag).strip() in r["text"])
        r["k_tags"] = next((ks for pre, ks in SECTION_SKILLS if r["section"].startswith(pre)), [])
        r["n_train_questions"] = sum(tcount[t] for t in r["templates"])
    jl(out / "rules/rules.jsonl", rrows)
    cf_rows = []
    for v in spec.COUNTERFACTUAL:
        key = v["old"] if v["kind"] == "replace" else v["anchor"]
        if key not in rulebook:
            raise SystemExit(f"Biến thể {v['id']}: không tìm thấy dòng gốc trong rulebook train")
        cf_rows.append({**v, "oracle_supported": False,
                        "note": "Oracle chỉ dùng cho câu đối chứng (không khớp trigger, không thuộc affected_templates)."})
    jl(out / "rules/counterfactual.jsonl", cf_rows)

    # 5) phân loại
    jw(out / "taxonomy/skills.json", {"skills": spec.SKILLS, "families": spec.FAMILIES})
    trows = []
    for t, (fam, ks, desc) in spec.TEMPLATES.items():
        rows = [s for s in seeds if s["route"]["template"] == t]
        wrong = [s["id"] for s in rows if not s["route"]["template_correct"]]
        frag = templates_ext.RULE_OF.get(t)
        trows.append({"template": t, "family": fam, "k_tags": ks, "mo_ta": desc, "rule_line": rule_lines.get(frag, frag) if frag else None,
                      "n_questions": dict(collections.Counter(s["role"] for s in rows)), "n_wrong": len(wrong), "wrong_ids": wrong,
                      "oracle_status": "ok" if not wrong else "caution",
                      "answer_types": dict(collections.Counter(s["gold"]["type"] for s in rows)),
                      "shops": sorted({s["business_id"] for s in rows}),
                      "examples_mining": [s["question"] for s in rows if s["role"] == "mining"][:3]})
    jl(out / "taxonomy/templates.jsonl", trows)
    repair_src = (ROOT / "datamate/repair.py").read_text(encoding="utf-8")
    verify_src = (ROOT / "datamate/verify.py").read_text(encoding="utf-8")
    jw(out / "taxonomy/error_groups.json", {"groups": spec.ERROR_GROUPS, "tags": spec.ERROR_TAGS, "step_codes": spec.STEP_CODES,
                                            "repair_types": sorted(set(re.findall(r'return "([A-Z_0-9]+)"', repair_src))),
                                            "verify_codes": sorted(set(re.findall(r'Issue\("([A-Z_]+)"', verify_src))),
                                            "decisions": ["template", "template_rejected", "plan", "plan_agree", "consensus", "escalated", "no_consensus",
                                                          "no_candidate", "dropped_type", "plan_conflict", "judge_rejected", "consensus_rejected"]})
    jl(out / "taxonomy/known_errors.jsonl", [{"id": i, "k_tag": k, "loi": w, "cach_dung": r, "nguon": src} for i, k, w, r, src in spec.KNOWN_ERRORS])

    # 6) phong cách (chỉ câu mining)
    ms = [s for s in seeds if s["role"] == "mining"]
    n = len(ms)
    words = sorted(s["style"]["n_words"] for s in ms)
    abbr = collections.Counter(x for s in ms for x in s["style"]["abbreviations"])
    fill = collections.Counter(x for s in ms for x in s["style"]["fillers"])
    opener = collections.Counter(" ".join(norm_key(s["question"]).split()[:2]) for s in ms)
    rnd = random.Random(7)

    def pick(pred, k=6):
        xs = [s["question"] for s in ms if pred(s)]
        return rnd.sample(xs, min(k, len(xs)))
    jw(out / "style/style_profile.json", {
        "n": n, "ti_le": {"khong_dau": round(sum(s["style"]["no_diacritic"] for s in ms) / n, 3),
                          "co_viet_tat": round(sum(bool(s["style"]["abbreviations"]) for s in ms) / n, 3),
                          "co_tu_dem": round(sum(bool(s["style"]["fillers"]) for s in ms) / n, 3),
                          "go_telex": round(sum(bool(s["style"]["telex"]) for s in ms) / n, 3),
                          "chu_thuong_dau": round(sum(s["style"]["lower_start"] for s in ms) / n, 3),
                          "khong_dau_hoi": round(sum(s["style"]["no_qmark"] for s in ms) / n, 3)},
        "go_telex": dict(collections.Counter(w for s in ms for w in s["style"]["telex"]).most_common(30)),
        "so_tu": {"p10": words[n // 10], "p50": words[n // 2], "p90": words[9 * n // 10], "max": words[-1]},
        "viet_tat": dict(abbr.most_common()), "tu_dem": dict(fill.most_common()), "mo_dau": dict(opener.most_common(25)),
        "ky": dict(collections.Counter(p for s in ms for p in s["periods"]).most_common()),
        "kieu_dap_an": dict(collections.Counter(s["gold"]["type"] for s in ms).most_common()),
        "cau_mau": {"day_du_dau": pick(lambda s: not s["style"]["no_diacritic"] and not s["style"]["abbreviations"] and not s["style"]["fillers"]),
                    "khong_dau": pick(lambda s: s["style"]["no_diacritic"]), "viet_tat": pick(lambda s: len(s["style"]["abbreviations"]) >= 2),
                    "tu_dem_lich_su": pick(lambda s: bool(s["style"]["fillers"])), "go_telex": pick(lambda s: bool(s["style"]["telex"]))}})

    # 7) lọc trùng
    guard = [{"id": s["id"], "business_id": s["business_id"], "role": s["role"], "question": s["question"], "question_key": norm_key(s["question"])}
             for s in seeds if s["role"] in ("tune", "holdout")]
    jl(out / "guard/eval_questions.jsonl", guard)

    # 8) prompt cho teacher, thư mục log lỗi
    shutil.copytree(ROOT / "eval/synth_prompts", out / "prompts")
    (out / "errors").mkdir(exist_ok=True)
    if not (out / "errors/README.md").exists():
        (out / "errors/README.md").write_text(
            "# Lỗi từ log chạy\n\nTạo bằng `python eval/mine_errors.py OUT/<lượt> --name <lượt>`; mỗi lượt một thư mục con.\n"
            "Xem `taxonomy/error_groups.json` cho ý nghĩa nhóm E1–E6b. Chỉ dòng `usable_as_seed = true` (shop mining) được dùng để sinh.\n", encoding="utf-8")

    # 9) manifest + README
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = branch = "?"
    import numpy as np
    stats = {"created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "code_version": __version__, "git_commit": commit, "splits": sp,
             "n_shops": len(profiles), "n_catalogs": len(cats), "n_questions": len(seeds), "n_hard": len(hard),
             "n_hard_mining": sum(h["role"] == "mining" for h in hard), "n_code": len(bank), "n_code_ok": sum(r["verified"] for r in bank),
             "n_rules": len(rrows), "n_cf": len(cf_rows), "n_templates": len(trows), "n_known": len(spec.KNOWN_ERRORS), "n_guard": len(guard),
             "n_model_routed": sum(s["route"]["to_model"] for s in seeds), "n_model_routed_mining": sum(s["route"]["to_model"] for s in ms),
             "n_seed": len(ms), "industries": dict(collections.Counter(p["industry"] for p in profiles)),
             "archetypes": dict(collections.Counter(p["archetype"] for p in profiles))}
    write_readme(out, stats)
    files = {str(p.relative_to(out)): {"sha256": sha(p)[:16], "rows": sum(1 for _ in p.open(encoding="utf-8")) if p.suffix == ".jsonl" else None}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json" and not str(p.relative_to(out)).startswith("errors/")}
    jw(out / "manifest.json", {**{k: v for k, v in stats.items() if k != "splits"}, "git_branch": branch, "pandas": pd.__version__, "numpy": np.__version__,
                               "loader_code_hash": loader._code_hash(),
                               "inputs": {f: sha(DATA / f)[:16] for f in ("questions.jsonl", "answers.jsonl", "rulebook.md", "calendar.csv")}
                               | {"data/code_bank.jsonl": sha(ROOT / "data/code_bank.jsonl")[:16], "eval/synth_spec.py": sha(ROOT / "eval/synth_spec.py")[:16]},
                               "files": files})
    print(json.dumps({k: v for k, v in stats.items() if k != "splits"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
