"""Đo 3 model trên chính các việc sinh dữ liệu, theo 3 mức khó, để chia việc (synth_spec.MODEL_ROUTING).

    python eval/model_bakeoff.py                     # chạy đủ: ~78 lần gọi, vài phút đến vài chục phút
    python eval/model_bakeoff.py --dry-run           # chỉ dựng prompt + kiểm chặn rò dữ liệu, không gọi API
    python eval/model_bakeoff.py --tasks T3 --providers qwen,gpt --samples 1
    python eval/model_bakeoff.py --providers deepseek,qwen --tasks T2,T3 --think T2=off,T3=off --merge outputs/bakeoff/<lượt trước>
    python eval/model_bakeoff.py --rescore outputs/bakeoff/<a>,outputs/bakeoff/<b>   # chấm lại trả lời đã lưu bằng bộ kiểm hiện tại
    python eval/model_bakeoff.py --tasks T4 --providers gemini,qwen --think qwen.T4=off --samples 3 --bo dev   # đo prompt 8 (--bo holdout, xac_nhan)

Việc:
    T1 (dễ)  hồ sơ shop: 8 hồ sơ/lần, tắt suy nghĩ; kiểm synth_checks.check_profiles
    T2 (vừa) danh mục ngành mới theo prompts/7 (40 mặt hàng), mỗi ngành trong synth_spec.NEW_INDUSTRIES; kiểm check_catalog
    T4 (khó) sinh mẫu dữ liệu train theo prompts/8: câu hỏi gốc + đặc tả + code, 12 mẫu/lô trên một bộ đồ chơi khó tự sinh (dev, holdout
             — cả hai đã dùng khi chọn prompt — và xac_nhan: hạt mới, chỉ để xác nhận bản chốt). Một mẫu (case) hợp lệ khi code chạy, ra đúng kiểu khai báo qua formatter, không rỗng, kiểu hợp cách hỏi
             (verify.check), không lộ đáp án số, không trùng, và model giải chính (khác họ model soạn, chỉ đọc câu hỏi) ra cùng đáp án theo
             score.match. Đo thêm: khớp cả hai model giải, "khó thật" (câu khó model yếu sai mà hai model mạnh khớp), phủ kỹ năng, trùng giữa
             lô, cảnh báo (thiếu kỳ, mã nội bộ, gợi ý), câu viết lại qua lớp văn phong còn đúng nghĩa, văn phong so với câu train, giây/USD.
    T3 (khó) viết code pandas trên bảng chuẩn đồ chơi (tự sinh ở đây, không phải dữ liệu BTC), 6 câu từ dễ đến khó gồm các bẫy
             thấy trong log hard42: lọc trạng thái nhiều cách viết, bỏ khách lẻ khi xếp hạng, quy đổi đơn vị, trừ trả hàng,
             ngày hai định dạng; chấm bằng cách chạy code (tiến trình riêng, quá 60 giây là hỏng)

"Hoàn chỉnh" = gọi được + đọc được (JSON/code) + không lỗi nặng sau khi chuẩn hoá kiểu (synth_checks.normalize_catalog);
lỗi nhẹ (synth_checks.SOFT: số nhóm, độ lệch phổ biến, khuôn luật) vẫn đếm riêng ("khong_loi_nao"). Không gửi dữ liệu BTC: mọi prompt qua llm.guard().
Ra: data/synth_source/data_spec/model_bakeoff.json (tổng hợp) và outputs/bakeoff/<giờ>/ (trả lời thô, danh mục đạt).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import tempfile
import functools
import json
import os
import pickle
import re
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT)]
import llm  # noqa: E402
import synth_checks  # noqa: E402
from stress_noise import VanPhong  # noqa: E402
from synth_spec import CATALOG_EXAMPLE, CATALOG_SCHEMA, MODEL_ROUTING, NEW_INDUSTRIES  # noqa: E402

SPEC = ROOT / "data/synth_source/data_spec"
PROMPT7 = ROOT / "eval/synth_prompts/7_danh_muc_nganh_moi.md"
N_ITEMS, N_PROFILES = 40, 8
THINK = {"T1": "off", "T2": "high", "T3": "high", "T4": "high"}
MAX_TOKENS = {"T1": 4000, "T2": 24000, "T3": 12000, "T4": 32000, "T4g": 24000}  # T4g: một model giải lại cả lô
N_CASES = 12
SKILLS_T4 = ["doanh_thu_ky", "loc_trang_thai", "tra_hang", "quy_doi_don_vi", "khach_le", "xep_hang", "so_sanh_ky", "ty_le", "dem_don_khach",
             "gia_tri_trung_binh", "cong_no", "lo_han_dung", "noi_bang"]
SKILLS_T4_V1 = ["doanh_thu_ky", "loc_trang_thai", "xep_hang_khach", "xep_hang_mat_hang", "quy_doi_don_vi", "tra_hang_doanh_thu_thuan",
                "so_sanh_ky", "dem_don_khach", "gia_tri_trung_binh"]  # nhãn của lô v1 (bảng tạp hoá), chỉ để chấm lại
# Giải lại độc lập: hai model mạnh đầu tiên khác họ model soạn (model đầu quyết định hợp lệ, model sau để đo "khớp cả hai");
# model yếu giải cùng câu để đánh dấu "khó thật" = câu khó mà model yếu sai trong khi hai model mạnh cùng khớp bên soạn.
MANH = next(r["model"] for r in MODEL_ROUTING if r["viec"].startswith("2.4"))  # gpt:off, qwen:off, gemini:high
YEU = "deepseek:off"


# ---------------------------------------------------------------- T1, T2: prompt
def prompt_t1(ten: str) -> str:
    return (f"Tạo {N_PROFILES} hồ sơ cửa hàng {ten} nhỏ ở Việt Nam (hộ kinh doanh), khác nhau về vùng miền và quy mô, để mô phỏng dữ liệu bán hàng.\n"
            f"Trả về đúng một JSON là danh sách {N_PROFILES} phần tử, mỗi phần tử có các khoá:\n"
            '"ten_shop" (tên biển hiệu), "tinh_thanh", "quan_huyen", "chu_tiem" (họ tên đầy đủ, có dấu), "mo_ta" (một câu, không quá 30 từ), '
            '"gio_mo" ([giờ mở, giờ đóng], số nguyên 0–24), "kenh_ban" (danh sách kênh bán), "so_nhan_vien" (số nguyên 0–20), "quy_mo" ("nhỏ" hoặc "vừa").\n'
            "Không giải thích thêm.")


def prompt_version(path: Path = PROMPT7) -> str:
    m = re.search(r"phiên bản:\s*(v\d+)", path.read_text(encoding="utf-8"))
    return m.group(1) if m else "v1"


def prompt_t2(ten: str) -> str:
    t = re.sub(r"<!--.*?-->\n?", "", PROMPT7.read_text(encoding="utf-8"), flags=re.S)
    return (t.replace("{ten_nganh}", ten).replace("{n_items}", str(N_ITEMS))
            .replace("{catalog_schema}", json.dumps(CATALOG_SCHEMA, ensure_ascii=False, indent=1))
            .replace("{catalog_example}", json.dumps(CATALOG_EXAMPLE, ensure_ascii=False, indent=1)))


# ---------------------------------------------------------------- T3: bảng đồ chơi + câu hỏi + đáp án chuẩn
DONE = ("hoàn thành", "Hoàn thành", "xong")
CANCEL = ("đã huỷ", "Huỷ", "huy")
ITEMS = [("SP01", "Nước ngọt có ga lon 330ml", "lon", 10000, ("thùng", 24)), ("SP02", "Bia lon 330ml", "lon", 15000, ("thùng", 24)),
         ("SP03", "Mì gói tôm chua cay", "gói", 4500, ("thùng", 30)), ("SP04", "Sữa tươi hộp 180ml", "hộp", 8000, ("lốc", 4)),
         ("SP05", "Gạo thơm túi 5kg", "túi", 120000, None), ("SP06", "Dầu ăn chai 1 lít", "chai", 52000, ("thùng", 12)),
         ("SP07", "Nước suối chai 500ml", "chai", 5000, ("thùng", 24)), ("SP08", "Bánh quy hộp 300g", "hộp", 45000, None),
         ("SP09", "Đường trắng gói 1kg", "gói", 25000, None), ("SP10", "Cà phê hoà tan hộp 20 gói", "hộp", 60000, None)]
HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương"]
DEM = ["Văn", "Thị", "Minh", "Ngọc", "Đức", "Thanh", "Quốc", "Hữu", "Thu", "Gia"]
TEN = ["An", "Bình", "Châu", "Dũng", "Giang", "Hà", "Hải", "Hùng", "Khánh", "Lan", "Linh", "Long", "Mai", "Nam", "Phúc", "Quân", "Sơn", "Tâm",
       "Thảo", "Trang", "Tuấn", "Vy", "Yến"]


def toy_tables(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    items = pd.DataFrame([{"item_code": c, "name": n, "base_unit": u, "base_price": p} for c, n, u, p, _ in ITEMS])
    units = pd.DataFrame([{"item_code": c, "unit": b[0], "factor": b[1]} for c, _, _, _, b in ITEMS if b])
    names = set()
    while len(names) < 30:
        names.add(f"{rng.choice(HO)} {rng.choice(DEM)} {rng.choice(TEN)}")
    customers = pd.DataFrame([{"cust_code": "KL", "name": "Khách lẻ"}] + [{"cust_code": f"KH{i:03d}", "name": n} for i, n in enumerate(sorted(names), 1)])
    w = 1 / np.arange(1, 31) ** 0.8
    w /= w.sum()
    orders, lines, rets = [], [], []
    oid = 0
    for day in pd.date_range("2025-01-01", "2025-09-30"):
        for _ in range(int(rng.integers(5, 12))):
            oid += 1
            o = f"HD{oid:05d}"
            cust = "KL" if rng.random() < 0.4 else f"KH{int(rng.choice(30, p=w)) + 1:03d}"
            st = rng.choice(CANCEL) if rng.random() < 0.04 else rng.choice(DONE, p=[0.6, 0.3, 0.1])
            orders.append({"order_id": o, "order_date": day.strftime("%d/%m/%Y"), "cust_code": cust, "status": st})
            for ln, k in enumerate(rng.choice(len(ITEMS), size=int(rng.integers(1, 5)), replace=False), 1):
                code, _, bu, price, big = ITEMS[k]
                if big and rng.random() < 0.3:
                    unit, factor, qty = big[0], big[1], int(rng.integers(1, 4))
                    unit_price = int(round(price * factor * 0.95, -2))
                else:
                    unit, factor, qty = bu, 1, int(rng.integers(1, 10))
                    unit_price = price
                lines.append({"order_id": o, "line_no": ln, "item_code": code, "unit": unit, "qty": qty, "unit_price": unit_price,
                              "amount": qty * unit_price})
                if st in DONE and rng.random() < 0.02:
                    q = int(rng.integers(1, qty * factor + 1))
                    rets.append({"return_id": f"TH{len(rets) + 1:04d}", "order_id": o, "item_code": code,
                                 "return_date": (day + pd.Timedelta(days=int(rng.integers(0, 10)))).strftime("%Y-%m-%d"),
                                 "qty_base": q, "refund_amount": int(round(q * unit_price / factor, -2))})
    return {"items": items, "units": units, "customers": customers, "orders": pd.DataFrame(orders),
            "order_lines": pd.DataFrame(lines), "returns": pd.DataFrame(rets)}


def _gold(T: dict) -> list[dict]:
    o = T["orders"].assign(d=pd.to_datetime(T["orders"]["order_date"], format="%d/%m/%Y"))
    li = T["order_lines"].merge(o, on="order_id")
    done = li[li["status"].isin(DONE)]
    r = T["returns"].assign(d=pd.to_datetime(T["returns"]["return_date"], format="%Y-%m-%d"))

    def rev(m):
        return int(done.loc[done["d"].dt.to_period("M") == m, "amount"].sum())

    def net(m):
        return rev(m) - int(r.loc[r["d"].dt.to_period("M") == m, "refund_amount"].sum())

    om = o[o["d"].dt.to_period("M") == "2025-05"]
    jun = done[(done["d"].dt.to_period("M") == "2025-06") & (done["cust_code"] != "KL")]
    top = jun.groupby("cust_code")["amount"].sum().sort_values(ascending=False).head(3)
    name = T["customers"].set_index("cust_code")["name"]
    fac = {(u.item_code, u.unit): u.factor for u in T["units"].itertuples()}
    apr = done[(done["d"].dt.to_period("M") == "2025-04") & (done["item_code"] == "SP02")]
    qb = int(sum(x.qty * fac.get((x.item_code, x.unit), 1) for x in apr.itertuples()))
    n7, n8 = net("2025-07"), net("2025-08")
    return [
        {"qid": "q1_doanh_thu", "kho": 1, "q": "Doanh thu tháng 3/2025 là bao nhiêu?", "type": "number", "gold": rev("2025-03")},
        {"qid": "q2_ti_le_huy", "kho": 2, "q": "Tỉ lệ đơn bị huỷ trong tháng 5/2025 là bao nhiêu phần trăm?", "type": "number",
         "gold": round(om["status"].isin(CANCEL).mean() * 100, 2)},
        {"qid": "q3_top_khach", "kho": 3, "q": "3 khách hàng mua nhiều tiền nhất tháng 6/2025 là ai (từ cao xuống thấp)?", "type": "list",
         "gold": [name[c] for c in top.index]},
        {"qid": "q4_quy_doi", "kho": 3, "q": "Tháng 4/2025 cửa hàng bán được tổng cộng bao nhiêu lon Bia lon 330ml?", "type": "number", "gold": qb},
        {"qid": "q5_dt_thuan", "kho": 4, "q": "Doanh thu thuần tháng 7/2025 là bao nhiêu?", "type": "number", "gold": n7},
        {"qid": "q6_tang_truong", "kho": 5, "q": "Doanh thu thuần tháng 8/2025 tăng hay giảm bao nhiêu phần trăm so với tháng 7/2025?",
         "type": "number", "gold": round((n8 - n7) / n7 * 100, 2)},
    ]


RULES_T3 = """Quy ước tính (của cửa hàng này):
1. Trạng thái đơn ghi nhiều kiểu: 'hoàn thành', 'Hoàn thành', 'xong' là đơn thành công; 'đã huỷ', 'Huỷ', 'huy' là đơn huỷ.
2. Doanh thu của một kỳ = tổng cột amount của mọi dòng thuộc đơn thành công có order_date nằm trong kỳ. Đơn huỷ không tính.
3. Doanh thu thuần của một kỳ = doanh thu của kỳ trừ tổng refund_amount của các phiếu trả có return_date nằm trong kỳ.
4. Số lượng theo đơn vị cơ bản = qty × factor; factor lấy ở bảng units theo (item_code, unit); dòng ghi bằng đơn vị cơ bản thì factor = 1.
5. Khách lẻ (cust_code 'KL') không đưa vào xếp hạng khách hàng.
6. Tỉ lệ huỷ = số đơn huỷ / tổng số đơn có order_date trong kỳ × 100. Phần trăm thay đổi = (kỳ này − kỳ trước) / kỳ trước × 100.
   Kết quả phần trăm làm tròn 2 chữ số sau dấu phẩy; tiền và số lượng là số nguyên.
order_date dạng ngày/tháng/năm (dd/mm/yyyy); return_date dạng năm-tháng-ngày (yyyy-mm-dd)."""


def prompt_t3(T: dict, q: dict) -> str:
    desc = []
    for k, df in T.items():
        desc.append(f"T['{k}']: {len(df)} dòng; cột: " + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
                    + "\n" + df.head(3).to_string(index=False))
    return ("Bạn viết code Python (pandas) trả lời câu hỏi về dữ liệu bán hàng của một cửa hàng tạp hoá nhỏ.\n"
            "Biến có sẵn: T là dict các DataFrame; pd và np đã import. Không đọc file, không in bảng.\n\n"
            + "\n\n".join(desc) + "\n\n" + RULES_T3 + f"\n\nCâu hỏi: {q['q']}\n\n"
            "Gán kết quả cuối vào biến `answer`: một số (int hoặc float) với câu hỏi số; list tên (list[str]) với câu hỏi danh sách. "
            "Chỉ trả về một khối ```python```.")


# ---------------------------------------------------------------- T4: bộ đồ chơi khó — dev, holdout (dev thứ hai) và xac_nhan (hạt mới)
# Tự sinh từ hạt cố định, không đọc file. Tái hiện cấu trúc làm model học sai (data_spec/README.md §5): nhiều bảng nối qua mã, lô/hạn dùng,
# quy đổi nhiều cấp, công nợ, trạng thái và khách lẻ ghi nhiều kiểu, trả hàng là dòng âm, tên cột tiếng Việt không chuẩn.
HANG_DEV = [  # mã, tên, nhóm, đơn vị gốc, giá, kê đơn, [(đơn vị lớn, số đơn vị gốc)]
    ("T01", "Paracetamol 500mg", "Giảm đau - hạ sốt", "viên", 800, "Không", [("vỉ", 10), ("hộp", 100)]),
    ("T02", "Amoxicillin 500mg", "Kháng sinh", "viên", 1500, "Có", [("vỉ", 10), ("hộp", 100)]),
    ("T03", "Cefuroxim 250mg", "Kháng sinh", "viên", 4500, "Có", [("vỉ", 10), ("hộp", 20)]),
    ("T04", "Omeprazol 20mg", "Tiêu hoá", "viên", 1200, "Có", [("vỉ", 14), ("hộp", 28)]),
    ("T05", "Vitamin C 500mg", "Vitamin", "viên", 600, "Không", [("lọ", 100)]),
    ("T06", "Men vi sinh gói 1g", "Tiêu hoá", "gói", 5000, "Không", [("hộp", 20)]),
    ("T07", "Oresol bù nước", "Tiêu hoá", "gói", 3000, "Không", [("hộp", 20)]),
    ("T08", "Siro ho thảo dược 100ml", "Hô hấp", "chai", 45000, "Không", [("thùng", 24)]),
    ("T09", "Nước muối sinh lý 500ml", "Vật tư y tế", "chai", 6000, "Không", [("lốc", 10), ("thùng", 30)]),
    ("T10", "Khẩu trang y tế 4 lớp", "Vật tư y tế", "cái", 1000, "Không", [("hộp", 50), ("thùng", 2000)]),
    ("T11", "Băng cá nhân", "Vật tư y tế", "miếng", 500, "Không", [("hộp", 100)]),
    ("T12", "Kem bôi da 15g", "Da liễu", "tuýp", 35000, "Không", []),
]
HANG_HOLDOUT = [  # mã, tên, loại, đơn vị nhỏ nhất, giá, bán theo lô, [(đơn vị lớn, số đơn vị nhỏ nhất)]
    ("SP001", "Phân NPK 16-16-8", "Phân bón", "kg", 18000, False, [("bao 25kg", 25), ("bao 50kg", 50)]),
    ("SP002", "Phân Urê hạt đục", "Phân bón", "kg", 14000, False, [("bao", 50)]),
    ("SP003", "Phân Kali clorua", "Phân bón", "kg", 16000, False, [("bao", 50)]),
    ("SP004", "Vôi bột nông nghiệp", "Cải tạo đất", "kg", 3000, False, [("bao", 25)]),
    ("SP005", "Thuốc trừ sâu sinh học 450ml", "Thuốc BVTV", "chai", 95000, True, [("lốc", 6), ("thùng", 24)]),
    ("SP006", "Thuốc trừ cỏ 1 lít", "Thuốc BVTV", "chai", 160000, True, [("thùng", 12)]),
    ("SP007", "Thuốc trừ bệnh 100ml", "Thuốc BVTV", "chai", 42000, True, [("hộp", 10), ("thùng", 100)]),
    ("SP008", "Hạt giống lúa thơm", "Hạt giống", "kg", 22000, True, [("bao", 40)]),
    ("SP009", "Hạt giống rau muống 50g", "Hạt giống", "gói", 12000, True, [("hộp", 20), ("thùng", 200)]),
    ("SP010", "Phân bón lá vi lượng 500ml", "Phân bón", "chai", 55000, True, [("thùng", 20)]),
    ("SP011", "Bình phun 16 lít", "Dụng cụ", "cái", 450000, False, []),
    ("SP012", "Màng phủ nông nghiệp", "Dụng cụ", "cuộn", 380000, False, []),
]
QUY_UOC_DEV = """Quy ước tính của nhà thuốc này:
1. Trạng thái hoá đơn ghi nhiều kiểu (hoa/thường, có/không dấu, viết tắt); mỗi cách ghi thuộc một trong hai nhóm: hoàn thành hoặc huỷ.
   Hoá đơn huỷ không tính vào doanh thu, số lượng bán, xếp hạng, công nợ; chỉ dùng khi hỏi về hoá đơn huỷ.
2. Doanh thu của một kỳ = tổng Thành tiền các dòng có SL dương thuộc hoá đơn hoàn thành có Ngày HĐ trong kỳ. Một đơn là một Số HĐ.
3. Trả hàng ghi thành dòng âm (SL và Thành tiền âm) ngay trong hoá đơn gốc và tính vào kỳ của hoá đơn đó. Tiền trả hàng = tổng trị tuyệt
   đối Thành tiền các dòng âm; doanh thu thuần = doanh thu − tiền trả hàng. Số lượng bán chỉ cộng dòng dương; số lượng bán ròng cộng cả dòng âm.
4. Số lượng theo ĐVT gốc = SL × hệ số ở bảng quy đổi (theo mã thuốc và ĐVT); dòng ghi bằng ĐVT gốc có hệ số 1. Hỏi theo đơn vị lớn thì
   chia số lượng theo ĐVT gốc cho hệ số của đơn vị đó, làm tròn 2 chữ số.
5. Cột Khách hàng của hoá đơn ghi mã khách có trong bảng khách hàng; khách lẻ ghi nhiều kiểu hoặc để trống. Khách lẻ không phải khách hàng:
   không xếp hạng, không đếm khi hỏi số khách.
6. Nợ của một khách đến hết ngày D = tổng Phát sinh nợ − tổng Đã thu của khách đó trong sổ công nợ có Ngày ≤ D. Tiền thu nợ của một kỳ =
   tổng Đã thu có Ngày trong kỳ.
7. Dòng bán thuốc ghi số lô (vật tư y tế không theo lô); hạn dùng ở bảng lô. Lô hết hạn tại ngày D khi Hạn dùng trước ngày D. Tồn của lô
   đến hết ngày D = SL nhập − số lượng bán ròng (ĐVT gốc) của lô đó trong các hoá đơn hoàn thành có Ngày HĐ ≤ D.
8. Tỉ lệ huỷ của kỳ = số hoá đơn huỷ / số hoá đơn có Ngày HĐ trong kỳ × 100. Phần trăm thay đổi = (kỳ này − kỳ trước) / kỳ trước × 100.
   Phần trăm làm tròn 2 chữ số; tiền và số lượng theo ĐVT gốc là số nguyên (trung bình tiền làm tròn về đồng).
9. Xếp hạng bằng nhau thì mã nhỏ hơn đứng trước."""
QUY_UOC_HOLDOUT = """Quy ước của cửa hàng vật tư nông nghiệp:
1. Tình trạng đơn có nhiều cách ghi; mỗi cách ghi thuộc một trong ba nhóm: đã giao xong, đang giao, đã huỷ. Doanh thu, số lượng bán và
   xếp hạng chỉ tính đơn đã giao xong (đơn đang giao chưa tính, đơn huỷ không tính).
2. Doanh thu kỳ = tổng Tiền các dòng Số lượng > 0 của đơn đã giao xong có Ngày đặt trong kỳ. Đếm đơn theo Mã đơn, không theo dòng.
3. Hàng trả là dòng âm (Số lượng, Tiền âm) nằm trong đơn gốc, có Ngày trả. Tiền hàng trả của kỳ = tổng |Tiền| các dòng âm có Ngày trả trong
   kỳ. Doanh thu thuần kỳ = doanh thu kỳ − tiền hàng trả của kỳ. Số lượng bán chỉ cộng dòng dương.
4. Số lượng theo đơn vị nhỏ nhất = Số lượng × Hệ số trong bảng quy cách (theo Mã SP và ĐVT); dòng ghi bằng đơn vị nhỏ nhất có hệ số 1.
   Hỏi theo đơn vị lớn thì chia cho hệ số của đơn vị đó, làm tròn 2 chữ số.
5. Khách vãng lai (cột Khách ghi nhiều kiểu hoặc để trống) không phải khách hàng: không xếp hạng, không đếm số khách.
6. Đơn ghi nợ là đơn có Thanh toán thuộc nhóm nợ (ghi sổ, nợ…). Nợ của khách đến hết ngày D = tổng Tiền các dòng dương của đơn ghi nợ
   không huỷ có Ngày đặt ≤ D − tổng Số tiền phiếu thu của khách có Ngày thu ≤ D. Hàng trả không làm giảm nợ.
7. Thuốc và hạt giống bán theo lô (Mã lô); HSD ở bảng lô. Lô hết hạn tại ngày D khi HSD trước ngày D. Tồn lô đến hết ngày D = SL nhập − số
   lượng (đơn vị nhỏ nhất) dòng dương của lô trong các đơn không huỷ (kể cả đang giao) có Ngày đặt ≤ D + số lượng dòng âm có Ngày trả ≤ D.
8. Tỉ lệ huỷ = số đơn huỷ / số đơn đặt trong kỳ (mọi tình trạng) × 100; phần trăm thay đổi = (kỳ này − kỳ trước) / kỳ trước × 100;
   làm tròn 2 chữ số. Tiền, số lượng theo đơn vị nhỏ nhất là số nguyên (trung bình tiền làm tròn về đồng).
9. Xếp hạng bằng nhau thì mã nhỏ hơn đứng trước."""


def _pick(rng, xs, p=None):
    return xs[int(rng.choice(len(xs), p=p))]


def _ten_khach(rng, n: int) -> list[str]:
    names = set()
    while len(names) < n:
        names.add(f"{rng.choice(HO)} {rng.choice(DEM)} {rng.choice(TEN)}")
    return [str(x) for x in rng.permutation(sorted(names))]


def _mo_phong(rng, hang, ngay, n_kh, don_ngay, p_no, co_lo, dang_giao=False) -> dict:
    """Lớp sự thật chung: lô (bán lô hạn gần nhất), đơn, dòng, trả hàng một phần dòng, phiếu thu nợ."""
    days = pd.date_range(*ngay)
    lots = collections.defaultdict(list)
    for k, h in enumerate(hang):
        d = days[0] - pd.Timedelta(days=int(rng.integers(30, 120)))
        while co_lo(h) and d <= days[-1]:
            lots[h[0]].append({"lot": f"{d:%y%m%d}{k + 1:02d}", "nhap": d, "han": d + pd.Timedelta(days=int(rng.choice([150, 240, 365, 540])))})
            d += pd.Timedelta(days=int(rng.integers(40, 110)))
    wi, wk = rng.permutation(1 / np.arange(1, len(hang) + 1) ** 0.7), 1 / np.arange(1, n_kh + 1) ** 0.9
    orders, lines, rets, used = [], [], [], collections.Counter()
    for day in days:
        for _ in range(int(rng.integers(*don_ngay))):
            o = {"id": len(orders) + 1, "day": day, "cust": None if rng.random() < 0.4 else int(rng.choice(n_kh, p=wk / wk.sum())),
                 "st": "huy" if rng.random() < 0.04 else "giao" if dang_giao and day > days[-8] and rng.random() < 0.5 else "ok"}
            o["pay"] = "no" if o["cust"] is not None and rng.random() < p_no else "tm" if rng.random() < 0.6 else "ck"
            orders.append(o)
            for k in rng.choice(len(hang), size=int(rng.integers(1, 4)), replace=False, p=wi / wi.sum()):
                code, price, big = hang[k][0], hang[k][4], hang[k][6]
                if big and rng.random() < 0.35:
                    unit, f = big[int(rng.integers(len(big)))]
                    qty, up = int(rng.integers(1, 4)), int(round(price * f * 0.97, -2))
                else:
                    unit, f, qty, up = hang[k][3], 1, int(rng.integers(1, max(2, min(30, 300000 // price)) + 1)), price
                ok = [x for x in lots[code] if x["nhap"] <= day < x["han"]]
                ln = {"order": o["id"], "item": code, "unit": unit, "f": f, "qty": qty, "price": up,
                      "lot": min(ok, key=lambda x: x["han"])["lot"] if ok else None}
                lines.append(ln)
                used[ln["lot"]] += qty * f if o["st"] != "huy" else 0
                if o["st"] == "ok" and rng.random() < 0.03:
                    rets.append(dict(ln, qty=int(rng.integers(1, qty + 1)), day=min(day + pd.Timedelta(days=int(rng.integers(0, 15))), days[-1])))
                    used[ln["lot"]] -= rets[-1]["qty"] * f
    for h in hang:
        m = max([f for _, f in h[6]] + [10])
        for x in lots[h[0]]:
            x["sl"] = (max(used[x["lot"]], 0) // m + 1 + int(rng.integers(0, 4))) * m
    amt = collections.Counter()
    for ln in lines:
        amt[ln["order"]] += ln["qty"] * ln["price"]
    no = collections.defaultdict(list)
    for o in orders:
        if o["pay"] == "no" and o["st"] != "huy":
            no[o["cust"]].append((o["day"], amt[o["id"]]))
    pays = []
    for c, ev in sorted(no.items()):
        d, paid = ev[0][0] + pd.Timedelta(days=int(rng.integers(10, 40))), 0
        while d <= days[-1]:
            x = int((sum(a for t, a in ev if t <= d) - paid) * rng.uniform(0.4, 1.0)) // 1000 * 1000
            if x >= 1000:
                pays.append({"cust": c, "day": d, "amount": x})
                paid += x
            d += pd.Timedelta(days=int(rng.integers(20, 45)))
    return {"lots": lots, "orders": orders, "lines": lines, "rets": rets, "amt": amt, "pays": sorted(pays, key=lambda p: p["day"])}


def bo_dev(seed: int = 11) -> dict:
    """Nhà thuốc 01/10/2024–30/06/2025: 7 bảng nối qua mã; trả hàng là dòng âm tính theo kỳ hoá đơn; nợ ở sổ công nợ hai cột."""
    rng = np.random.default_rng(seed)
    s = _mo_phong(rng, HANG_DEV, ("2024-10-01", "2025-06-30"), 25, (12, 26), 0.3, lambda h: h[2] != "Vật tư y tế")
    kh, hd = [f"KH{i:02d}" for i in range(1, 26)], lambda i: f"HD{i:06d}"
    le, xong, huy = ["Khách lẻ", "khách lẻ", "KL", "kl", "Khach le", None], ["Hoàn thành", "hoàn thành", "HT", "Đã thanh toán", "xong"], ["Đã hủy", "đã huỷ", "Huỷ", "HUY"]
    don = [{"Số HĐ": hd(o["id"]), "Ngày HĐ": o["day"], "Khách hàng": kh[o["cust"]] if o["cust"] is not None else _pick(rng, le, [.35, .2, .2, .1, .1, .05]),
            "Trạng thái": _pick(rng, huy, [.4, .3, .2, .1]) if o["st"] == "huy" else _pick(rng, xong, [.5, .2, .1, .1, .1]),
            "Hình thức TT": {"tm": "Tiền mặt", "ck": "Chuyển khoản"}.get(o["pay"]) or _pick(rng, ["Ghi nợ", "ghi nợ"])} for o in s["orders"]]
    dong = [{"Số HĐ": hd(x["order"]), "Mã thuốc": x["item"], "Số lô": x["lot"], "ĐVT": x["unit"], "SL": sg * x["qty"], "Đơn giá": x["price"],
             "Thành tiền": sg * x["qty"] * x["price"]} for sg, xs in ((1, s["lines"]), (-1, s["rets"])) for x in xs]
    so_no = ([{"Ngày": o["day"], "Mã KH": kh[o["cust"]], "Diễn giải": f"Ghi nợ {hd(o['id'])}", "Phát sinh nợ": s["amt"][o["id"]], "Đã thu": np.nan}
              for o in s["orders"] if o["pay"] == "no" and o["st"] != "huy"]
             + [{"Ngày": p["day"], "Mã KH": kh[p["cust"]], "Diễn giải": _pick(rng, ["Thu nợ", "thu tiền nợ", "Khách trả nợ"]), "Phát sinh nợ": np.nan,
                 "Đã thu": p["amount"]} for p in s["pays"]])
    T = {"DanhMucThuoc": pd.DataFrame([{"Mã thuốc": c, "Tên thuốc": n, "Nhóm": g, "ĐVT gốc": u, "Kê đơn": rx, "Giá bán lẻ": p}
                                        for c, n, g, u, p, rx, _ in HANG_DEV]),
         "QuyDoiDVT": pd.DataFrame([{"Mã thuốc": h[0], "ĐVT": u, "Quy ra ĐVT gốc": f} for h in HANG_DEV for u, f in h[6]]),
         "KhachHang": pd.DataFrame({"Mã KH": kh, "Tên KH": _ten_khach(rng, 25),
                                    "Loại khách": [_pick(rng, ["Khách quen", "Phòng khám", "Nhà thuốc nhỏ"], [.6, .25, .15]) for _ in kh]}),
         "HoaDon": pd.DataFrame(don), "ChiTietHD": pd.DataFrame(dong).sort_values("Số HĐ", kind="stable", ignore_index=True),
         "LoThuoc": pd.DataFrame([{"Số lô": x["lot"], "Mã thuốc": c, "Ngày nhập": x["nhap"], "Hạn dùng": x["han"], "SL nhập": x["sl"]}
                                  for c, xs in s["lots"].items() for x in xs]),
         "SoCongNo": pd.DataFrame(so_no).sort_values("Ngày", kind="stable", ignore_index=True)}
    return {"T": T, "quy_uoc": QUY_UOC_DEV, "khoang": "Ngày HĐ có từ 01/10/2024 đến 30/06/2025."}


def bo_holdout(seed: int = 23) -> dict:
    """Vật tư nông nghiệp 01/03/2025–31/12/2025: thêm nhóm 'đang giao', hàng trả có Ngày trả riêng, nợ = đơn ghi nợ − phiếu thu."""
    rng = np.random.default_rng(seed)
    s = _mo_phong(rng, HANG_HOLDOUT, ("2025-03-01", "2025-12-31"), 30, (6, 15), 0.45, lambda h: h[5], dang_giao=True)
    kh, md = [f"K{i:03d}" for i in range(1, 31)], lambda i: f"DB-{i:05d}"
    st = {"ok": (["Đã giao", "đã giao", "Giao xong", "Hoàn tất", "hoan tat"], [.45, .2, .15, .12, .08]),
          "huy": (["Hủy", "đã huỷ", "Khách hủy", "HỦY"], [.4, .3, .2, .1]), "giao": (["Đang giao", "đang giao"], [.6, .4])}
    don = [{"Mã đơn": md(o["id"]), "Ngày đặt": o["day"],
            "Khách": kh[o["cust"]] if o["cust"] is not None else _pick(rng, ["Vãng lai", "vãng lai", "KVL", "khách vãng lai", "-", None], [.3, .2, .2, .1, .1, .1]),
            "Tình trạng": _pick(rng, *st[o["st"]]),
            "Thanh toán": {"tm": "Tiền mặt", "ck": "Chuyển khoản"}.get(o["pay"]) or _pick(rng, ["Ghi sổ", "ghi nợ", "Nợ"], [.5, .3, .2])} for o in s["orders"]]
    dong = [{"Mã đơn": md(x["order"]), "Mã SP": x["item"], "Mã lô": x["lot"], "ĐVT": x["unit"], "Số lượng": sg * x["qty"], "Giá": x["price"],
             "Tiền": sg * x["qty"] * x["price"], "Ngày trả": x["day"] if sg < 0 else pd.NaT} for sg, xs in ((1, s["lines"]), (-1, s["rets"])) for x in xs]
    T = {"san_pham": pd.DataFrame([{"Mã SP": h[0], "Tên sản phẩm": h[1], "Loại": h[2], "Đơn vị nhỏ nhất": h[3]} for h in HANG_HOLDOUT]),
         "quy_cach": pd.DataFrame([{"Mã SP": h[0], "ĐVT": u, "Hệ số": f} for h in HANG_HOLDOUT for u, f in h[6]]),
         "khach": pd.DataFrame({"Mã khách": kh, "Tên khách": _ten_khach(rng, 30),
                                "Ấp/xã": [_pick(rng, ["Mỹ Hoà", "Tân Phú", "Bình Long", "An Thạnh", "Phú Mỹ"]) for _ in kh]}),
         "don_ban": pd.DataFrame(don), "dong_don": pd.DataFrame(dong).sort_values("Mã đơn", kind="stable", ignore_index=True),
         "lo_hang": pd.DataFrame([{"Mã lô": x["lot"], "Mã SP": c, "Ngày nhập": x["nhap"], "HSD": x["han"], "SL nhập": x["sl"]}
                                  for c, xs in s["lots"].items() for x in xs]),
         "phieu_thu": pd.DataFrame([{"Số phiếu": f"PT{i:04d}", "Ngày thu": p["day"], "Mã khách": kh[p["cust"]], "Số tiền": p["amount"]}
                                    for i, p in enumerate(s["pays"], 1)])}
    return {"T": T, "quy_uoc": QUY_UOC_HOLDOUT, "khoang": "Ngày đặt có từ 01/03/2025 đến 31/12/2025."}


SHOPS: dict[str, dict] = {}  # shop thật của giai đoạn 2, eval/gen_data.py đăng ký vào đây


@functools.lru_cache(maxsize=None)
def bo(name: str) -> dict:
    """Bộ dữ liệu cho T4: {"T": bảng, "quy_uoc": quy ước tính, "khoang": khoảng ngày có dữ liệu}. toy = bảng tạp hoá của T3 (số đo v1)."""
    if name in SHOPS:
        return SHOPS[name]
    if name == "toy":
        return {"T": toy_tables(), "quy_uoc": RULES_T3, "khoang": "order_date có từ 01/01/2025 đến 30/09/2025."}
    return {"dev": bo_dev, "holdout": bo_holdout, "xac_nhan": lambda: bo_holdout(seed=2611)}[name]()  # xac_nhan: hạt mới, chưa dùng khi chọn prompt


def _desc(T: dict) -> str:
    """Mô tả bảng như phần DỮ LIỆU model thấy: cột + kiểu, khoảng giá trị, các cách ghi hay gặp của cột chữ, ô trống, 3 dòng đầu."""
    out = []
    for k, df in T.items():
        cols = []
        for c in df.columns:
            s, x = df[c], f"- {c} ({df[c].dtype})"
            if pd.api.types.is_datetime64_any_dtype(s):
                x += f": {s.min():%d/%m/%Y} → {s.max():%d/%m/%Y}"
            elif pd.api.types.is_numeric_dtype(s):
                x += ": " + " → ".join(f"{v:.0f}" if v == int(v) else f"{v:g}" for v in (s.min(), s.max()))
            elif 12 < s.nunique() and s.nunique() > 0.5 * len(s):
                x += f": {s.nunique()} giá trị khác nhau, ví dụ " + ", ".join(repr(v) for v in s.dropna().unique()[:2])
            else:
                vc = s.value_counts()
                x += (": " if len(vc) <= 12 else f": {len(vc)} giá trị, hay gặp: ") + ", ".join(f"{v!r} ×{n}" for v, n in vc.head(12).items())
            cols.append(x + (f"; {int(s.isna().sum())} ô trống" if s.isna().any() else ""))
        out.append(f"T['{k}']: {len(df)} dòng\n" + "\n".join(cols) + "\n" + df.head(3).to_string(index=False))
    return "\n\n".join(out)


PROMPT8 = ROOT / "eval/synth_prompts/8_generate_samples.md"
KIEU_DAP_AN = ("Gán kết quả cuối vào biến `answer` đúng kiểu câu hỏi cần: số tiền, số lượng, số đơn… là số; phần trăm là số phần trăm "
               "(12.5 nghĩa là 12,5%); một tên là chuỗi; nhiều phần tử là list theo thứ tự câu hỏi yêu cầu; ngày là chuỗi 'YYYY-MM-DD', "
               "tháng là chuỗi 'YYYY-MM'; câu có/không là True hoặc False; table là dict columns và rows; unanswerable gán None.")


def prompt_t4(b: dict, path: Path = PROMPT8, n_cases: int = N_CASES) -> str:
    """Prompt 8 đã điền cho bộ dữ liệu b."""
    t = re.sub(r"<!--.*?-->\n?", "", path.read_text(encoding="utf-8"), flags=re.S)
    fill = {"khoang_du_lieu": b["khoang"], "mo_ta_bang": _desc(b["T"]), "quy_uoc": b["quy_uoc"], "n_cases": str(n_cases), "ky_nang": ", ".join(SKILLS_T4)}
    for k, v in fill.items():
        t = t.replace("{" + k + "}", v)
    return t


def prompt_giai(b: dict, questions: list[str]) -> str:
    """Giải độc lập cả lô trong một lần gọi: chỉ thấy câu hỏi + mô tả bảng + quy ước, không thấy code, đáp án, đặc tả của bên soạn."""
    qs = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return ("Bạn viết code Python (pandas) trả lời câu hỏi của chủ cửa hàng về dữ liệu bán hàng.\n"
            f"Biến có sẵn: T là dict các DataFrame; pd và np đã import. Không đọc file, không in bảng. {b['khoang']}\n\n{_desc(b['T'])}\n\n"
            f"{b['quy_uoc']}\n\nCâu hỏi:\n{qs}\n\nViết {len(questions)} khối ```python``` theo đúng thứ tự câu, khối thứ k mở đầu bằng dòng "
            f"`# cau k`; mỗi khối tự tính lại từ T, không dùng biến của khối khác. Tên bảng, tên cột chép nguyên văn như mô tả (cả dấu, "
            f"khoảng trắng). {KIEU_DAP_AN}")


def tach_khoi(text: str, n: int) -> list[str | None]:
    """Khối code thứ k của trả lời giải cả lô (theo dòng '# cau k'; không đánh số mà đủ n khối thì theo thứ tự)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    # Some providers put all numbered programs in one fenced block. Split only
    # explicit whole-line markers; each segment still executes in isolation.
    blocks = [part for b in blocks for part in re.split(r"(?=^\s*#\s*c[aâ]u\s*\d+\s*$)", b, flags=re.M) if part.strip()]
    out = [None] * n
    for b in blocks:
        m = re.match(r"\s*#\s*c[aâ]u\s*(\d+)", b, re.I)
        if m and 1 <= int(m.group(1)) <= n and out[int(m.group(1)) - 1] is None:
            out[int(m.group(1)) - 1] = b
    return blocks if not any(out) and len(blocks) == n else out


HARNESS = """import sys, json, pickle, runpy
import numpy as np, pandas as pd
T = pickle.load(open(sys.argv[1], "rb"))
code = open(sys.argv[2], encoding="utf-8").read()
restrict = runpy.run_path(sys.argv[3])["restrict_process"]
env = {"T": T, "pd": pd, "np": np}
restrict()
exec(compile(code, "<code>", "exec"), env)
a = env.get("answer")
if isinstance(a, (pd.Series, pd.Index, np.ndarray)):
    a = list(a)
if hasattr(a, "item"):
    a = a.item()
if isinstance(a, list):
    a = [x.item() if hasattr(x, "item") else x for x in a]
print(json.dumps({"answer": a}, ensure_ascii=False, default=str))
"""


# Each child imports pandas and numpy, so bound their combined memory by default.
_EXEC_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get("SYNTH_SUBPROCESS_LIMIT", "2"))))


def run_code(code: str, tables_pkl: Path, work: Path, tag: str) -> tuple[object, str | None]:
    from synth_execution import preflight, child_environment
    if problem := preflight(code):
        return None, "unsafe_code: " + problem
    # A unique directory prevents concurrent jobs with identical tags from overwriting code or harness.
    with tempfile.TemporaryDirectory(prefix="synth_exec_", dir=work) as tmp:
        tmp = Path(tmp)
        f = tmp / "code.py"
        f.write_text(code, encoding="utf-8")
        harness = tmp / "harness.py"
        harness.write_text(HARNESS, encoding="utf-8")
        try:
            with _EXEC_SEM:
                p = subprocess.run([sys.executable, "-I", str(harness), str(tables_pkl.resolve()), str(f), str(ROOT / "eval" / "synth_execution.py")], capture_output=True, text=True,
                                   timeout=60, cwd=tmp, env=child_environment())
        except subprocess.TimeoutExpired:
            return None, "qua_gio"
        if p.returncode != 0:
            last = (p.stderr.strip().splitlines() or ["?"])[-1]
            return None, "loi_chay: " + last[:160]
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])["answer"], None
        except (ValueError, KeyError, IndexError):
            return None, "khong_doc_duoc_ket_qua"


def run_many(items: list[tuple[str | None, str]], tables_pkl: Path, work: Path) -> list[tuple[object, str | None]]:
    """[(code, tag)] → [(đáp án thô, lỗi)]: mỗi code một tiến trình, chạy song song."""
    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(lambda x: run_code(x[0], tables_pkl, work, x[1]) if x[0] else (None, "khong_co_code"), items))


# ---------------------------------------------------------------- chấm như pipeline: formatter + làm tròn tiền + score.match
def typed(raw, loai: str, question: str = "") -> dict | None:
    """Đáp án thô của code → đáp án pipeline nộp: formatter.finalize theo kiểu khai báo rồi verify.round_money. None khi formatter bỏ."""
    from datamate import formatter, verify
    if loai == "unanswerable" and raw is not None:
        return None
    a = formatter.finalize("x", {"type": loai, "answer": raw})
    return verify.round_money(question, {k: v for k, v in a.items() if k != "id"}) if a["type"] == loai else None


def same_answer(a, g, loai: str, question: str = "") -> bool:
    """So như bộ chấm BTC sau formatter (score.match): cùng kiểu khai báo; số làm tròn 2 chữ số (tiền về đồng nếu câu hỏi về tiền);
    chuỗi chuẩn hoá như score.norm; list đúng thứ tự. Khác dạng (list một phần tử với chuỗi, Timestamp với chuỗi ngày) là sai."""
    from score import match
    ta, tg = typed(a, loai, question), typed(g, loai, question)
    return bool(ta and tg) and match({"id": "x", **ta}, {"alias": [], **tg}) == 1


def same(ans, q: dict) -> bool:
    return same_answer(ans, q["gold"], q["type"], q["q"])


# ---------------------------------------------------------------- T4: một lô = soạn → kiểm (sửa một lượt) → văn phong → giải lại → chấm
LOAI = ("number", "percent", "list", "string", "date", "boolean", "table", "unanswerable")
SUA = ("Trả lời trên chưa dùng được:\n{loi}\nTrả về đúng một JSON là danh sách {n} phần tử theo khuôn đã nêu (question, do_kho, ky_nang, loai, "
       "spec, code){them}.")
_HINT_KIEU = ("hỏi %/tỉ lệ → percent; 'tháng nào' → date 'YYYY-MM'; 'ngày nào' → date 'YYYY-MM-DD'; 'có … không' → boolean; "
              "'bao nhiêu tiền' → number nguyên; 'top k' → list không quá k phần tử")


def _mau(i: int, c) -> dict:
    if not (isinstance(c, dict) and isinstance(c.get("question"), str) and isinstance(c.get("code"), str) and c.get("loai") in LOAI):
        return {"i": i, "loi_sinh": ["sai_khuon"], "loi_chay": "thiếu question/code hoặc loai không thuộc " + "|".join(LOAI)}
    return {"i": i, "question_goc": c["question"].strip(), "spec": c.get("spec"), "do_kho": c.get("do_kho"), "ky_nang": c.get("ky_nang") or [],
            "loai": c["loai"], "code": c["code"], "loi_sinh": []}


def _so(x) -> int:
    return int(x) if str(x).strip().isdigit() else 0


def _doc_lo(text: str) -> tuple[list | None, str | None]:
    try:
        x = llm.extract_json(text)
    except ValueError as e:
        return None, f"không đọc được JSON ({str(e)[:100]})"
    if isinstance(x, dict):  # {"khoá": [mẫu…]} thì lấy danh sách; một mẫu lẻ, các object rời không có [ ] là sai khuôn → lượt sửa JSON
        x = next((v for v in x.values() if isinstance(v, list) and v and all(isinstance(e, dict) for e in v)), None)
    return (x, None) if isinstance(x, list) else (None, "JSON phải là một danh sách các mẫu [ {...}, {...} ]")


def kiem_mau(c: dict) -> list[tuple[str, str]]:
    """Lỗi nặng của một mẫu (mã, lời nhắn để teacher sửa). Trừ lỗi lúc sinh (sai khuôn, code lỗi), mọi lỗi tính lại được từ đáp án thô."""
    from datamate import verify
    if c["loi_sinh"]:
        return [(e, c.get("loi_chay") or e) for e in c["loi_sinh"]]
    a = typed(c["raw"], c["loai"], c["question_goc"])
    if a is None:
        return [("sai_loai", f"khai báo loai={c['loai']} nhưng code ra {str(c['raw'])[:80]!r}")]
    v, err = a.get("answer"), []
    if verify.check(c["question_goc"], a):
        err.append(("kieu_lech_cau_hoi", "kiểu đáp án không hợp cách hỏi: " + _HINT_KIEU))
    if c["loai"] in ("number", "percent") and synth_checks.lo_dap_an(c["question_goc"], a):
        err.append(("lo_dap_an", "câu hỏi chứa sẵn đáp án"))
    return err


def _chay(cs: list[dict], tables: Path, work: Path, tag: str) -> None:
    todo = [c for c in cs if not c["loi_sinh"]]
    for c, (ans, err) in zip(todo, run_many([(c["code"], f"{tag}_{c['i']}") for c in todo], tables, work)):
        c["raw"] = ans
        if err:
            c["loi_sinh"], c["loi_chay"] = ["code_loi"], err


def _goi(j: dict, vai: str, provider: str, think: str, messages, max_tokens: int) -> dict:
    """Một lần gọi model trong lô; lỗi API trả nội dung rỗng (không làm hỏng cả lô). Ghi thời gian, chi phí vào j['goi']."""
    try:
        r = llm.chat(provider, messages, think=think, max_tokens=max_tokens, tag=f"bakeoff:T4_{vai.split(':')[0]}", timeout=900, retries=2)
    except (llm.LeakError, llm.BudgetExceeded):
        raise
    except Exception as e:  # noqa: BLE001
        r = {"content": "", "usage": {}, "latency_s": None, "cost_usd": None, "finish": "loi_api", "error": str(e)[:300]}
    j["goi"].append({"vai": vai, "provider": provider, "think": think, "latency_s": r.get("latency_s"), "cost_usd": r.get("cost_usd"),
                     "usage": r.get("usage", {}), "finish": r.get("finish"), "error": r.get("error")})
    return r


def giai_lo(j: dict, vai: str, spec: str, b: dict, questions: list[str], tables: Path, work: Path) -> list[dict]:
    """Một model giải cả lô câu hỏi (một lần gọi); trả [{answer, loi}] theo thứ tự câu."""
    prov, think = spec.split(":")
    r = _goi(j, vai, prov, think, prompt_giai(b, questions), MAX_TOKENS["T4g"])
    if r["finish"] == "loi_api" or not r["content"]:
        return [{"answer": None, "loi": "loi_api", "code": None} for _ in questions]
    codes = tach_khoi(r["content"], len(questions))
    tag = f"g{j['sample']}_{j['provider']}_{vai.replace(':', '_')}"
    return [{"answer": a, "loi": e, "code": code} for code, (a, e) in zip(codes, run_many([(c, f"{tag}_{i}") for i, c in enumerate(codes)], tables, work))]


def solvers_for(teacher: str) -> list[str]:
    """Hai model mạnh khác họ model soạn; model đầu quyết định hợp lệ."""
    return [s for s in MANH if s.split(":")[0] != teacher][:2]


def t4_batch(j: dict, work: Path, vp) -> dict:
    """Một lô T4. j: task, provider, think, sample, bo, prompt. vp: lớp văn phong (stress_noise.VanPhong). Trả bản ghi để chấm lại."""
    t0, b, tables = time.time(), bo(j["bo"]), work / f"{j['bo']}.pkl"
    n_req = j.get("n_req", N_CASES)
    j["goi"], j["sua"] = [], None
    msgs = [{"role": "user", "content": j["prompt"]}]
    r = _goi(j, "soan", j["provider"], j["think"], msgs, MAX_TOKENS["T4"])
    items, err = _doc_lo(r["content"]) if r["finish"] != "length" else (None, "hết token trước khi xong JSON")
    if err and r["finish"] != "loi_api":  # sửa một lượt: JSON hỏng
        j["sua"] = "json"
        msgs += [{"role": "assistant", "content": r["content"][-6000:]}, {"role": "user", "content": SUA.format(loi=err, n=n_req, them="")}]
        r = _goi(j, "sua", j["provider"], j["think"], msgs, MAX_TOKENS["T4"])
        items, err = _doc_lo(r["content"])
    j.update(content=r["content"], loi_lo=["loi_api" if r["finish"] == "loi_api" else "khong_doc_duoc_json"] if items is None
             else ["thieu_mau"] if len(items) < n_req else [])
    cs = [_mau(i, c) for i, c in enumerate((items or [])[:n_req])]
    _chay(cs, tables, work, f"t{j['sample']}_{j['provider']}")
    bad = {c["i"]: e for c in cs if (e := kiem_mau(c))}
    if bad and not j["sua"]:  # sửa một lượt: code lỗi, sai kiểu, rỗng, lộ đáp án (kèm thông báo lỗi)
        j["sua"] = "code"
        loi = "\n".join(f"- mẫu {i + 1}" + (f" ({cs[i]['question_goc']!r})" if "question_goc" in cs[i] else "") + ": " + "; ".join(m for _, m in e)
                        for i, e in bad.items())
        msgs += [{"role": "assistant", "content": r["content"]},
                 {"role": "user", "content": SUA.format(loi=loi, n=len(bad), them=" — chỉ gồm các mẫu đã sửa, mỗi phần tử thêm khoá \"i\" là số mẫu cũ")}]
        fix, _ = _doc_lo(_goi(j, "sua", j["provider"], j["think"], msgs, MAX_TOKENS["T4"])["content"])
        new = [dict(_mau(k, x), sua=True) for x in fix or [] if isinstance(x, dict) and (k := _so(x.get("i")) - 1) in bad]
        _chay(new, tables, work, f"t{j['sample']}_{j['provider']}_sua")
        for c in new:
            cs[c["i"]] = c
    j["cases"] = cs
    todo = [c for c in judge_t4(j | {"cases": cs, "solvers": [], "giai_xong": False})["cases"] if not c["errors"] and "question_goc" in c]
    for c in todo:
        c["question"] = vp.render(c["question_goc"])
    sv = solvers_for(j["provider"])
    j.update(solvers=sv, weak=YEU if j["provider"] != YEU.split(":")[0] else None)
    calls = [("giai", s, "question_goc") for s in sv] + [("yeu", YEU, "question_goc")] * bool(j["weak"]) + [("vp", sv[0], "question")]
    with ThreadPoolExecutor(max_workers=len(calls)) as ex:  # các model giải song song, mỗi model một lần gọi cho cả lô
        res = list(ex.map(lambda x: giai_lo(j, f"{x[0]}:{x[1]}", x[1], b, [c[x[2]] for c in todo], tables, work), calls if todo else []))
    for k, c in enumerate(todo):
        c["giai"] = {s: x[k] for (v, s, _), x in zip(calls, res) if v != "vp"}
        c["vp"] = dict(res[-1][k], lan=1)
    lai = [c for c in todo if same_answer(c["giai"][sv[0]]["answer"], c["raw"], c["loai"], c["question_goc"])
           and not same_answer(c["vp"]["answer"], c["raw"], c["loai"], c["question_goc"])]
    if lai:  # câu viết lại lệch nghĩa: viết lại một lần nữa rồi kiểm lại, vẫn lệch thì bỏ
        for c in lai:
            c["question_1"], c["question"] = c["question"], vp.render(c["question_goc"])
        for c, x in zip(lai, giai_lo(j, f"vp:{sv[0]}", sv[0], b, [c["question"] for c in lai], tables, work)):
            c["vp"] = dict(x, lan=2)
    j["t_tong"] = round(time.time() - t0, 1)
    return judge_t4(j | {"giai_xong": True})


def _trung(a: dict, b: dict) -> bool:
    """Hai mẫu trùng chữ (Jaccard 3-gram ≥ 0,7); đáp án số giống nhau chưa đủ."""
    sa, sb = a["_sh"], b["_sh"]
    if len(sa & sb) / max(1, len(sa | sb)) >= 0.7:
        return True
    return False


def _shingles(q: str) -> set:
    # Remove only conversational framing, preserving measures, negation and dates.
    # This catches "cho biết ... nhé" paraphrases without using equal answers.
    q = re.sub(r"\b(?:cho biết|bao nhiêu|là|nhé)\b", " ", q, flags=re.I)
    q = re.sub(r"\bcủa\s+(?=tháng|quý|năm|ngày)", "", q, flags=re.I)
    w = synth_checks.fold(q).strip(" ?!.,;:").split()
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def judge_t4(rec: dict) -> dict:
    """Chấm lô từ đáp án thô đã lưu (bên soạn, các bên giải, câu viết lại) — chấm lại được mà không gọi API.
    Lỗi nặng (mẫu không hợp lệ): sai khuôn, code lỗi, sai kiểu, rỗng, kiểu lệch cách hỏi, lộ đáp án số, trùng, model giải chính lệch.
    Cảnh báo (vẫn hợp lệ, đếm riêng): thiếu kỳ rõ, mã nội bộ, gợi ý cách giải, lộ đáp án chữ."""
    from build_synth_source import style_of
    sv, weak, b = rec.get("solvers") or [], rec.get("weak"), bo(rec["bo"])
    codes, known = synth_checks.codes_of(b["T"]), _known(rec["bo"])
    seen, n_req = [], rec.get("n_req", N_CASES)
    for c in rec["cases"]:
        if "question_goc" not in c:
            c["errors"] = list(c["loi_sinh"])
            continue
        c["errors"] = [e for e, _ in kiem_mau(c)]
        c["_typed"] = typed(c.get("raw"), c["loai"], c["question_goc"]) if not c["loi_sinh"] else None
        c["_sh"] = _shingles(c["question_goc"])
        if any(_trung(c, o) for o in seen):
            c["errors"].append("trung")
        seen.append(c)
        q, a = c["question_goc"], c["_typed"]
        c["canh_bao"] = ([] if synth_checks.ky_ro(q) else ["thieu_ky"]) + (["ma_noi_bo"] if synth_checks.ma_noi_bo(q, codes) else []) \
            + (["goi_y"] if synth_checks.goi_y(q) else []) + (["lo_dap_an_chu"] if a and a["type"] not in ("number", "percent") and synth_checks.lo_dap_an(q, a) else [])
        if not rec.get("giai_xong") or c["errors"]:
            continue
        for v in c.get("giai", {}).values():
            error = v.get("loi")
            if error:
                v["status"] = "infrastructure_error" if error == "loi_api" or error == "qua_gio" else "invalid_output"
            elif typed(v.get("answer"), c["loai"], q) is None:
                v["status"] = "invalid_output"
            else:
                v["status"] = "executed_correct" if same_answer(v.get("answer"), c["raw"], c["loai"], q) else "executed_wrong"
            v["khop"] = v["status"] == "executed_correct"
        for solver in [*sv, *([weak] if weak else [])]:
            c.setdefault("giai", {}).setdefault(solver, {"status": "not_measured", "khop": False})
        if not sv or not c.get("giai", {}).get(sv[0], {}).get("khop"):
            c["errors"].append("giai_lai_lech")
        c["khop_ca_hai"] = len(sv) == 2 and all(c["giai"].get(s, {}).get("khop") for s in sv)
        c["that_kho"] = not c["errors"] and c.get("do_kho") == 3 and c["khop_ca_hai"] and bool(weak) and c["giai"].get(weak, {}).get("status") == "executed_wrong"
        if "vp" in c:
            c["vp"]["khop"] = c["vp"].get("loi") is None and same_answer(c["vp"].get("answer"), c["raw"], c["loai"], q)
    valid = [c for c in rec["cases"] if "question_goc" in c and not c["errors"] and rec.get("giai_xong")]
    dung = [c for c in valid if c.get("vp", {}).get("khop")]
    sty = lambda q: VanPhong.net(style_of(q, known)) | {"so_tu": len(q.split())}  # noqa: E731
    for c in rec["cases"]:
        c.pop("_sh", None), c.pop("_typed", None)
    return rec | {"n_req": n_req, "n_valid": len(valid), "n_khop_ca_hai": sum(bool(c.get("khop_ca_hai")) for c in valid),
                  "n_that_kho": sum(bool(c.get("that_kho")) for c in valid), "n_dung_duoc": len(dung),
                  "n_vp_kiem": sum("vp" in c for c in valid), "n_sua": sum(bool(c.get("sua")) for c in rec["cases"]),
                  "n_sua_dat": sum(bool(c.get("sua")) for c in valid),
                  "ky_nang_hop_le": sorted({k for c in valid for k in c.get("ky_nang") or [] if k in rec.get("ky_nang_ds", SKILLS_T4)}),
                  "do_kho_hop_le": dict(collections.Counter(str(c.get("do_kho")) for c in valid)),
                  "loai_hop_le": dict(collections.Counter(c["loai"] for c in valid)),
                  "canh_bao": dict(collections.Counter(w for c in valid for w in c["canh_bao"])),
                  "van_phong": [sty(c["question"]) for c in dung], "van_phong_goc": [sty(c["question_goc"]) for c in valid],
                  "ok": len(valid) >= 0.75 * n_req, "errors": rec.get("loi_lo", []) + [e for c in rec["cases"] for e in c["errors"]]}


@functools.lru_cache(maxsize=None)
def _known(name: str) -> frozenset:
    """Từ ASCII trong tên hàng/nhóm/khách của bộ dữ liệu và trong danh sách viết tắt: không tính là gõ telex sót."""
    from synth_spec import ABBREVIATIONS
    words = {w for df in bo(name)["T"].values() for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
             and not pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].nunique() <= 200
             for v in df[c].dropna().unique() for w in re.findall(r"[a-z]+", str(v).lower())}
    return frozenset(words | {w for a in ABBREVIATIONS for w in a.split()})


def ban_ghi_t4(rec: dict) -> dict:
    """Bản ghi một lô để lưu raw.jsonl và tổng hợp: bỏ prompt; thời gian, token, chi phí của lần soạn như các việc khác."""
    return {k: v for k, v in rec.items() if k != "prompt"} | {"api_ok": rec["loi_lo"] != ["loi_api"], "ky_nang_ds": SKILLS_T4,
                                                             "latency_s": rec["goi"][0]["latency_s"], "usage": rec["goi"][0]["usage"],
                                                             "cost_usd": rec["goi"][0]["cost_usd"]}


def mau_finetune(c: dict, rec: dict, b: dict) -> dict:
    """Legacy name: return RAW T4, not runtime SFT. Code still assigns lower-case answer.

    Use build_sft_data after independent gold, runtime conversion and export gates.
    """
    a = typed(c["raw"], c["loai"], c["question_goc"])
    return {"kind": "t4_mau", "shop": rec["bo"], "question": c["question"], "question_goc": c["question_goc"], "spec": c.get("spec"),
            "type": a["type"], "answer": a.get("answer"), "code": c["code"], "do_kho": c.get("do_kho"), "ky_nang": c.get("ky_nang"),
            "teacher": f"{rec['provider']}:{rec['think']}", "pv": rec.get("pv"), "giai_lai": {s: v["khop"] for s, v in c["giai"].items()},
            "khop_ca_hai": c["khop_ca_hai"], "that_kho": c["that_kho"], "canh_bao": c["canh_bao"],
            "messages": [{"role": "system", "content": "Bạn viết code Python (pandas) trả lời câu hỏi của chủ cửa hàng về dữ liệu bán hàng. "
                                                       f"Biến có sẵn: T là dict các DataFrame; pd và np đã import. {b['khoang']}\n\n{b['quy_uoc']}"},
                         {"role": "user", "content": f"{_desc(b['T'])}\n\nCâu hỏi: {c['question']}\n\n{KIEU_DAP_AN}"},
                         {"role": "assistant", "content": f"```python\n{c['code'].strip()}\n```"}]}


# ---------------------------------------------------------------- chạy
def jobs(tasks, providers, samples, industries, T, qs, bo_name="dev"):
    out = []
    for p in providers:
        for s in range(samples):
            if "T1" in tasks:
                for ind in industries[:2]:
                    out.append({"task": "T1", "provider": p, "sample": s, "key": ind["code"], "prompt": prompt_t1(ind["ten"])})
            if "T3" in tasks:
                for q in qs:
                    out.append({"task": "T3", "provider": p, "sample": s, "key": q["qid"], "prompt": prompt_t3(T, q), "q": q})
            if "T4" in tasks:
                out.append({"task": "T4", "provider": p, "sample": s, "key": f"{bo_name}_lo{N_CASES}", "bo": bo_name, "prompt": prompt_t4(bo(bo_name)),
                            "pv": prompt_version(PROMPT8)})
        if "T2" in tasks:
            for ind in industries:
                out.append({"task": "T2", "provider": p, "sample": 0, "key": ind["code"], "prompt": prompt_t2(ind["ten"]), "pv": prompt_version()})
    return out


def evaluate(j: dict, r: dict, work: Path, tables_pkl: Path) -> tuple[bool, list[str]]:
    text = r["content"]
    if r.get("finish") == "length":
        return False, ["het_token"]
    if j["task"] == "T3":
        ans, err = run_code(llm.extract_code(text), tables_pkl, work, f"{j['provider']}_{j.get('think', '')}_{j['key']}_{j['sample']}")
        if err:
            return False, [err.split(":")[0]]
        return (True, []) if same(ans, j["q"]) else (False, ["sai_dap_an"])
    try:
        obj = llm.extract_json(text)
    except ValueError:
        return False, ["khong_doc_duoc_json"]
    if j["task"] == "T1":
        errs = synth_checks.check_profiles(obj, N_PROFILES)
    else:
        obj, _ = synth_checks.normalize_catalog(obj)  # bộ sinh cũng chuẩn hoá kiểu dữ liệu trước khi dùng
        errs = synth_checks.check_catalog(obj, N_ITEMS)
    ok = not synth_checks.hard_errors(errs)  # hoàn chỉnh = không lỗi nặng; lỗi nhẹ (số nhóm, khuôn luật…) vẫn ghi lại
    if j["task"] == "T2" and ok:
        (work / "catalogs").mkdir(exist_ok=True)
        (work / "catalogs" / f"{j['key']}__{j['provider']}_{j.get('think', '')}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    return ok, errs


def _hop_le(r: dict) -> list[dict]:
    return [c for c in r.get("cases", []) if "question_goc" in c and not c["errors"] and r.get("giai_xong", True)]


def _ti_le_net(rows: list[dict], prof: dict) -> dict:
    """Tỉ lệ từng nét văn phong so với style_profile.json (ti_le) và trung vị số từ (so_tu.p50)."""
    if not rows:
        return {}
    out = {t: {"ti_le": round(sum(x[t] for x in rows) / len(rows), 3), "muc_tieu": prof["ti_le"][t],
               "lech": round(sum(x[t] for x in rows) / len(rows) - prof["ti_le"][t], 3)} for t in VanPhong.NET}
    return out | {"so_tu_p50": statistics.median(x["so_tu"] for x in rows), "muc_tieu_so_tu": prof["so_tu"]["p50"], "n": len(rows)}


def t4_summary(rs: list[dict]) -> dict:
    """Tổng hợp các lô T4 cùng (bản prompt, bộ dữ liệu, model): hợp lệ, khớp cả hai model giải, khó thật, phủ kỹ năng, trùng giữa lô,
    cảnh báo, kiểm nghĩa câu viết lại, văn phong so với train, sửa lỗi, giây và USD mỗi mẫu hợp lệ."""
    prof = json.loads((ROOT / "data/synth_source/style/style_profile.json").read_text(encoding="utf-8"))
    req, valid = sum(r.get("n_req", N_CASES) for r in rs), [c for r in rs for c in _hop_le(r)]
    nv = max(1, len(valid))
    seen, dup = [], 0
    for r in sorted(rs, key=lambda r: (r.get("run", ""), r["sample"])):  # trùng với mẫu hợp lệ của lô trước (cùng model, bản prompt, bộ dữ liệu)
        cur = [dict(c, _sh=_shingles(c["question_goc"]), _typed=typed(c["raw"], c["loai"], c["question_goc"])) for c in _hop_le(r)]
        dup += sum(any(_trung(c, o) for o in seen) for c in cur)
        seen += cur
    goi = [g for r in rs for g in r.get("goi", [])]
    usd = sum(g["cost_usd"] or 0 for g in goi)
    skills = [set(r.get("ky_nang_hop_le", [])) for r in rs]
    return {"mau_hop_le": f"{len(valid)}/{req}", "ti_le_hop_le": round(len(valid) / req, 3), "hop_le_moi_lo": round(len(valid) / len(rs), 2),
            "ti_le_khop_ca_hai": round(sum(bool(c.get("khop_ca_hai")) for c in valid) / req, 3),
            "ti_le_kho_that": round(sum(bool(c.get("that_kho")) for c in valid) / req, 3), "kho_that": sum(bool(c.get("that_kho")) for c in valid),
            "ky_nang_tb": round(statistics.mean(len(s) for s in skills), 1), "ky_nang_phu": len(set().union(*skills)),
            "do_kho_hop_le": dict(sorted(collections.Counter(str(c.get("do_kho")) for c in valid).items())),
            "loai_hop_le": dict(collections.Counter(c["loai"] for c in valid).most_common()), "trung_giua_lo": dup,
            "canh_bao": {k: round(v / nv, 3) for k, v in collections.Counter(w for c in valid for w in c.get("canh_bao", [])).most_common()},
            "kiem_nghia_vp": f"{sum(bool(c.get('vp', {}).get('khop')) for c in valid)}/{sum('vp' in c for c in valid)}",
            "van_phong": _ti_le_net([x for r in rs for x in r.get("van_phong", [])], prof),
            "van_phong_goc": _ti_le_net([x for r in rs for x in r.get("van_phong_goc", [])], prof),
            "sua": {"lo_sua_json": sum(r.get("sua") == "json" for r in rs), "mau_da_sua": sum(r.get("n_sua", 0) for r in rs),
                    "mau_sua_hop_le": sum(r.get("n_sua_dat", 0) for r in rs)},
            "giay_moi_mau_hop_le": round(sum(r.get("t_tong") or 0 for r in rs) / nv, 1) if any(r.get("t_tong") for r in rs) else None,
            "usd_moi_mau_hop_le": round(usd / nv, 5) if goi else None,
            "gia_chua_ro": sorted({g["provider"] for g in goi if g["cost_usd"] is None and g["finish"] != "loi_api"}),
            "token_moi_mau_hop_le": round(sum((g["usage"] or {}).get("total_tokens", 0) for g in goi) / nv) if goi else None,
            "goi_api": len(goi) or None,
            "loi_case": dict(collections.Counter(e for r in rs for c in r.get("cases", []) for e in c.get("errors", [])).most_common(8))}


def summarize(rows: list[dict]) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"] + (f"/{r['pv']}" if r.get("pv") else "") + (f"/{r['bo']}" if r.get("bo") else ""), r["provider"], r["think"])].append(r)
    table = []
    for (task, prov, think), rs in sorted(by.items()):
        lat = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        out_tok = [r["usage"].get("completion_tokens", 0) for r in rs if r.get("usage")]
        costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        fails = collections.Counter(e for r in rs for e in r["errors"])
        table.append({"task": task, "provider": prov, "model": llm.PROVIDERS[prov]["model"], "think": think, "n": len(rs),
                      **(t4_summary(rs) if task.startswith("T4") else {}),
                      "goi_duoc": sum(r["api_ok"] for r in rs), "hoan_chinh": sum(r["ok"] for r in rs),
                      "ti_le_hoan_chinh": round(sum(r["ok"] for r in rs) / len(rs), 3),
                      "khong_loi_nao": sum(r["ok"] and not r["errors"] for r in rs),
                      "giay_p50": round(statistics.median(lat), 1) if lat else None, "giay_max": round(max(lat), 1) if lat else None,
                      "token_ra_tb": round(statistics.mean(out_tok)) if out_tok else None,
                      "chi_phi_usd": round(sum(costs), 4) if costs else None,
                      "usd_moi_mau_dat": round(sum(costs) / max(1, sum(r["ok"] for r in rs)), 5) if costs else None,
                      "loi_hay_gap": dict(fails.most_common(6))})
    detail = collections.defaultdict(dict)
    for r in rows:
        if r["task"] in ("T2", "T3"):
            d = detail[f"{r['task']}{('/' + r['pv']) if r.get('pv') else ''}:{r['key']}"].setdefault(f"{r['provider']}:{r['think']}", {"dat": 0, "n": 0, "loi": []})
            d["n"] += 1
            d["dat"] += r["ok"]
            d["loi"] += r["errors"]
    return {"theo_viec_model": table, "theo_nganh_cau": dict(detail)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="T1,T2,T3")
    ap.add_argument("--providers", default="deepseek,qwen,gpt,gemini")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4, help="số lời gọi song song mỗi model")
    ap.add_argument("--think", default="", help="đổi mức suy nghĩ: T2=off (mọi model) hoặc deepseek.T4=off (một model); off|low|high")
    ap.add_argument("--bo", default="dev", choices=["dev", "holdout", "xac_nhan", "toy"],
                    help="bộ dữ liệu T4: dev và holdout (đã dùng khi quyết định giữ/bỏ: coi là dev thứ hai); xac_nhan: hạt mới, chỉ chạy bản chốt")
    ap.add_argument("--merge", default="", help="thư mục outputs/bakeoff/<lượt> trước đó, gộp vào bảng tổng hợp (chấm lại)")
    ap.add_argument("--rescore", default="", help="chỉ chấm lại các thư mục đã có (cách nhau dấu phẩy), không gọi API")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger", default=str(llm.DEFAULT_LEDGER), help="shared HTTP attempt ledger, cap llm.HARD_CAP")
    ap.add_argument("--out", default=str(SPEC / "model_bakeoff.json"))
    a = ap.parse_args(argv)
    llm.configure_run(a.ledger, limit=llm.HARD_CAP, provider_limits={p: a.workers for p in llm.PROVIDERS})
    tasks, providers = a.tasks.split(","), a.providers.split(",")
    think = {**THINK, **dict(x.split("=") for x in a.think.split(",") if x)}
    T = toy_tables()
    qs = _gold(T)
    if a.rescore:
        return finish(a, [Path(d) for d in a.rescore.split(",")], T, qs, think)
    js = jobs(tasks, providers, a.samples, NEW_INDUSTRIES, T, qs, a.bo)
    for j in js:
        llm.guard(j["prompt"])
    print(f"{len(js)} việc (T4: mỗi lô 1 lần soạn + ≤1 lần sửa + 4–5 lần giải); prompt qua kiểm chặn rò dữ liệu", flush=True)
    if a.dry_run:
        for t in ("T1", "T2", "T3", "T4"):
            j = next((x for x in js if x["task"] == t), None)
            if j:
                print(f"\n===== {t} ({len(j['prompt'])} ký tự) =====\n{j['prompt'][:6000]}")
        print("\nĐáp án chuẩn T3:", json.dumps([(q["qid"], q["gold"]) for q in qs], ensure_ascii=False))
        return 0
    work = ROOT / "outputs/bakeoff" / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"  # hai lượt chạy song song không trùng thư mục
    work.mkdir(parents=True, exist_ok=False)
    (work / "harness.py").write_text(HARNESS, encoding="utf-8")
    tables_pkl = work / "toy_tables.pkl"
    pickle.dump(T, open(tables_pkl, "wb"))
    pickle.dump(bo(a.bo)["T"], open(work / f"{a.bo}.pkl", "wb"))
    vp = VanPhong(seed=0, known=_known(a.bo))
    sem = {p: threading.Semaphore(a.workers) for p in providers}
    lock, rows = threading.Lock(), []
    raw, train = open(work / "raw.jsonl", "a", encoding="utf-8"), open(work / "raw_samples.jsonl", "a", encoding="utf-8")

    def one(j):
        j["think"] = llm.PROVIDERS[j["provider"]].get("think_co_dinh") or think.get(f"{j['provider']}.{j['task']}") or think[j["task"]]
        j.setdefault("pv", "")  # bản prompt đọc lúc dựng prompt (sửa file prompt giữa lượt không làm sai nhãn)
        if j["task"] == "T4":
            with sem[j["provider"]]:
                rec = ban_ghi_t4(t4_batch(j, work, vp))
            msg = f"hợp lệ {rec['n_valid']}/{rec['n_req']} khớp 2 {rec['n_khop_ca_hai']} khó thật {rec['n_that_kho']} dùng được {rec['n_dung_duoc']}"
            with lock:
                for c in _hop_le(rec):
                    if c.get("vp", {}).get("khop"):
                        train.write(json.dumps(mau_finetune(c, rec, bo(rec["bo"])), ensure_ascii=False, default=str) + "\n")
                train.flush()
        else:
            with sem[j["provider"]]:
                try:
                    r = llm.chat(j["provider"], j["prompt"], think=j["think"], max_tokens=MAX_TOKENS[j["task"]], tag=f"bakeoff:{j['task']}",
                                 timeout=900, retries=2)
                    api_ok = True
                except (llm.LeakError, llm.BudgetExceeded):
                    raise
                except Exception as e:  # noqa: BLE001
                    r, api_ok = {"content": "", "usage": {}, "latency_s": None, "cost_usd": None, "error": str(e)[:300]}, False
            ok, errs = evaluate(j, r, work, tables_pkl) if api_ok else (False, ["loi_api"])
            rec = {k: j[k] for k in ("task", "provider", "think", "sample", "key", "pv")} | {
                "api_ok": api_ok, "ok": ok, "errors": errs, "latency_s": r.get("latency_s"), "usage": r.get("usage", {}),
                "cost_usd": r.get("cost_usd"), "finish": r.get("finish"), "reasoning_chars": len(r.get("reasoning", "")), "error": r.get("error"),
                "content": r.get("content", "")}
            msg = ("ĐẠT " if ok else "hỏng") + " " + ",".join(errs)[:80]
        with lock:
            rows.append(rec)
            raw.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            raw.flush()
            print(f"[{len(rows)}/{len(js)}] {j['task']} {j['provider']:<8} {j['key']:<22} {msg} ({rec.get('t_tong') or rec.get('latency_s')}s)", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers * len(providers)) as ex:
        for f in as_completed([ex.submit(one, j) for j in js]):
            f.result()
    raw.close()
    train.close()
    return finish(a, [Path(d) for d in a.merge.split(",") if d] + [work], T, qs, think)


def _t4_v1(r: dict) -> dict:
    """Bản ghi T4 cũ (v1: 8 mẫu/lô trên bảng tạp hoá, giải từng câu) → khuôn mới để chấm lại bằng bộ chấm hiện tại."""
    for c in r.get("cases", []):
        if "question" in c:
            c.update(question_goc=c.pop("question"), raw=c.pop("answer", None), giai=c.pop("giai_lai", {}),
                     loi_sinh=[e for e in c.get("errors", []) if e == "code_loi"])
        else:
            c["loi_sinh"] = ["sai_khuon"]
    return r | {"bo": "toy", "solvers": r.get("giai_lai_boi") or [], "weak": None, "giai_xong": True, "ky_nang_ds": SKILLS_T4_V1,
                "loi_lo": [e for e in r.get("errors", []) if e in ("het_token", "khong_doc_duoc_json", "khong_phai_list", "loi_api")]}


def rescore(dirs: list[Path], T: dict, qs: list[dict]) -> list[dict]:
    """Chấm lại mọi trả lời đã lưu (raw.jsonl) bằng bộ kiểm hiện tại; T3 chạy lại code, T4 chấm lại từ đáp án thô đã lưu."""
    work = ROOT / "outputs/bakeoff/_rescore"
    work.mkdir(parents=True, exist_ok=True)
    (work / "harness.py").write_text(HARNESS, encoding="utf-8")
    pickle.dump(T, open(work / "toy_tables.pkl", "wb"))
    qby = {q["qid"]: q for q in qs}
    rows = []
    for d in dirs:
        for line in (Path(d) / "raw.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                print(f"bỏ dòng hỏng trong {d}/raw.jsonl", file=sys.stderr)
                continue
            r["think"] = r.get("think") or THINK[r["task"]]
            r["pv"] = r.get("pv") or ("v1" if r["task"] in ("T2", "T4") else "")
            if r["task"] == "T4":  # không gọi lại API: chấm lại từ đáp án đã lưu
                r = judge_t4(r if "bo" in r else _t4_v1(r))
            else:
                j = {k: r[k] for k in ("task", "provider", "think", "sample", "key")} | {"q": qby.get(r["key"])}
                r["ok"], r["errors"] = evaluate(j, r, work, work / "toy_tables.pkl") if r["api_ok"] else (False, ["loi_api"])
            rows.append({k: v for k, v in r.items() if k != "content"} | {"run": Path(d).name})
    # cùng (việc, model, mức suy nghĩ, bản prompt, bộ dữ liệu, câu/ngành, lần) xuất hiện ở nhiều thư mục: giữ bản của thư mục sau
    keep = {(r["task"], r["provider"], r["think"], r["pv"], r.get("bo"), r["key"], r["sample"], r["run"] if r["task"] == "T4" else ""): r
            for r in rows}
    return list(keep.values())


def finish(a, dirs: list[Path], T: dict, qs: list[dict], think: dict) -> int:
    rows = rescore(dirs, T, qs)
    res = {"ngay": dt.date.today().isoformat(), "thu_muc_tho": [str(Path(d).resolve().relative_to(ROOT)) for d in dirs],
           "cau_hinh": {"providers": {p: llm.PROVIDERS[p]["model"] for p in llm.PROVIDERS}, "samples": a.samples, "think_mac_dinh": THINK,
                        "max_tokens": MAX_TOKENS, "n_items": N_ITEMS, "n_profiles": N_PROFILES, "t4_mau_moi_lo": N_CASES,
                        "t4_giai_lai": {"manh": MANH, "yeu": YEU}},
           "dap_an_t3": [{k: q[k] for k in ("qid", "kho", "q", "gold")} for q in qs], **summarize(rows)}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for t in res["theo_viec_model"]:
        extra = (f"hợp lệ {t['mau_hop_le']} khớp2 {t['ti_le_khop_ca_hai']} khó thật {t['kho_that']} vp {t['kiem_nghia_vp']} "
                 f"{t['giay_moi_mau_hop_le']}s/mẫu {t['loi_case']}") if t["task"].startswith("T4") else f"{t['loi_hay_gap']}"
        print(f"{t['task']} {t['provider']:<9}{t['think']:<5} {t['hoan_chinh']:>3}/{t['n']:<3} = {t['ti_le_hoan_chinh']:.2f}  "
              f"p50 {t['giay_p50']}s  ${t['chi_phi_usd']}  {extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
