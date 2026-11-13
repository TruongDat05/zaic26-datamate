"""Excel files in every format a shop may ship (.xlsx, .xlsm, .xls) load into the same tables."""
import datetime
import shutil
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from datamate.ingest import load_business, read_rows, to_number

FIXTURE = Path(__file__).parent / "fixtures" / "so_ban_hang.xls"  # written with xlwt: two monthly sheets, a title row above the header
SHEETS = {"T01-2025": [["HD1", datetime.date(2025, 1, 2), "Cà phê sữa", 2, "55k"], [None, None, "Bạc xỉu", 1, 30000]],
          "T02-2025": [["HD2", datetime.date(2025, 2, 3), "Trà đào", 1, 35000]]}


def write_openpyxl(path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in SHEETS.items():
        ws = wb.create_sheet(name)
        ws.append(["SỔ BÁN HÀNG"])
        ws.append(["Mã đơn", "Ngày", "Món", "SL", "Thành tiền"])
        for r in rows:
            ws.append(r)
    wb.save(path)


def shop(tmp_path, ext):
    folder = tmp_path / ext / "data" / "shop"
    folder.mkdir(parents=True)
    target = folder / f"so_ban_hang{ext}"
    shutil.copy(FIXTURE, target) if ext == ".xls" else write_openpyxl(target)
    (folder / "README.md").write_text("Sổ bán hàng, mỗi sheet một tháng.", encoding="utf-8")
    return tmp_path / ext


@pytest.mark.parametrize("ext", [".xlsx", ".xlsm", ".xls"])
def test_excel_formats_load_the_same(tmp_path, ext):
    root = shop(tmp_path, ext)
    assert [sheet for _, sheet, _ in read_rows(root / "data/shop" / f"so_ban_hang{ext}")] == list(SHEETS)
    db, meta = load_business(root, "shop")
    names = {t["name"] for t in meta["tables"]}
    assert "so_ban_hang_all" in names, names  # the two monthly sheets share their columns: one union view
    rows = db.execute('SELECT "Mã đơn", CAST("Ngày" AS DATE), "Món", "SL", "Thành tiền" FROM so_ban_hang_all ORDER BY 3').fetchall()
    assert rows == [("HD1", datetime.date(2025, 1, 2), "Bạc xỉu", 1, 30000), ("HD1", datetime.date(2025, 1, 2), "Cà phê sữa", 2, 55000),
                    ("HD2", datetime.date(2025, 2, 3), "Trà đào", 1, 35000)]  # title row skipped, '55k' read, order id and date filled down


def test_signed_money_shorthand():
    values = pd.Series(["8tr600", "-8tr600", "-0tr600", "+8tr600", "-25k500", "-8triệu6", "-1,27tr", "-25k", "55.000đ", "1,234,000"], dtype="string")
    assert to_number(values).tolist() == [8_600_000, -8_600_000, -600_000, 8_600_000, -25_500, -8_600_000, -1_270_000, -25_000, 55_000, 1_234_000]
