"""Question-text hints: regexes on synthetic phrases (never copied from test questions), date windows, glosses."""
from datamate.hints import LIST_WORDS, LOT, TRIPS, lot_hint, date_anchors, grain_hints, outside, period_hints
from datamate.ingest import norm
from datamate.prompts import money_gloss


def test_trip_regex():
    assert not TRIPS.search("rua_bao_nhieu_luot_xe_may")
    assert TRIPS.search("xe_cho_duoc_bn_dot_hang")
    assert TRIPS.search("tuan_nay_giao_tong_cong_may_dot")


def test_lot_regex():
    assert not LOT.search("san_pham_nao_het_hang")
    assert not LOT.search("tra_no_dung_han_dung_bua_nay")  # 'hạn đúng hôm nay' becomes 'han_dung' after norm
    assert LOT.search("lo_nao_co_han_dung_tu_thang_5")


def test_money_gloss():
    assert money_gloss("khoảng 2tr5, 30tr, 7 củ, 250k, 2,5 triệu") == [
        "2tr5 = 2500000 đồng", "30tr = 30000000 đồng", "7 củ = 7000000 đồng", "250k = 250000 đồng", "2,5 triệu = 2500000 đồng"]


def test_month_without_year_is_this_year_unless_later():
    hints = period_hints("Doanh thu tháng 3 là bao nhiêu?", "2026-05-10")
    assert "tháng 3/2026 = [2026-03-01, 2026-04-01)" in hints
    hints = period_hints("Doanh thu tháng 11 là bao nhiêu?", "2026-05-10")
    assert "tháng 11/2025 = [2025-11-01, 2025-12-01)" in hints


def test_quarter_and_last_n_days():
    assert "quý 2/2026 = [2026-04-01, 2026-07-01)" in period_hints("lợi nhuận quý 2 năm 2026", "2026-08-01")
    assert "7 ngày gần nhất (incl. today) = [2026-05-04, 2026-05-11)" in period_hints("7 ngày gần nhất bán được bao nhiêu", "2026-05-10")
    assert not any("gần nhất" in h for h in period_hints("so với 7 ngày trước đó", "2026-05-10"))  # a window before another day


def test_n_days_from_a_date():
    hints = period_hints("số lượng 5 ngày kể từ 10/03/2026 so với 5 ngày trước", "2026-05-10")
    assert "5 ngày kể từ 2026-03-10 (incl. 2026-03-10) = [2026-03-10, 2026-03-15)" in hints
    assert "5 ngày ngay trước 2026-03-10 = [2026-03-05, 2026-03-10)" in hints


def test_date_anchors_name_today_and_yesterday():
    text = date_anchors("2026-05-10")
    assert "today (hôm nay, bữa nay, hnay) = 2026-05-10 (Sun)" in text
    assert "yesterday (hôm qua, bữa qua, hqua) = 2026-05-09 = [2026-05-09, 2026-05-10)" in text


def test_grain_hint_counts_days():
    assert grain_hints("có mấy ngày bán được từ 5 triệu trở lên")[0].startswith("GRAIN: count DAYS")
    assert grain_hints("doanh thu hôm qua") == []


def test_outside_data_dates():
    span = ("2025-01-01", "2025-12-31")
    assert outside(["tháng 3/2027 = [2027-03-01, 2027-04-01)"], span)
    assert not outside(["tháng 3/2025 = [2025-03-01, 2025-04-01)"], span)
    assert not outside([], span)


def test_norm_folds_accents_and_spacing():
    assert norm("Tháng  3/2026") == "thang_3_2026"


LOT_META = {"tables": [{"name": "lo_nhap", "columns": [("Lot", "VARCHAR"), ("Ma", "VARCHAR"), ("Han", "DATE"), ("SL", "BIGINT")]},
                       {"name": "xuat_lo", "columns": [("Lot", "VARCHAR"), ("SL lay", "BIGINT"), ("So HD", "VARCHAR")]},
                       {"name": "ban", "columns": [("So HD", "VARCHAR"), ("Ngay", "DATE")]}]}


def test_list_order_hint_only_on_lot_lists():
    many = lot_hint("liệt kê các thuốc còn tồn có hạn dùng trong tháng 9", LOT_META)
    one = lot_hint("lô thuốc ho gần hết hạn nhất có hạn dùng ngày nào", LOT_META)
    assert len(many) == 2 and many[1].startswith("LIST ORDER (rulebook)") and "MIN(expiry), then item code" in many[1]
    assert len(one) == 1 and one[0].startswith("LOT STOCK")
    assert lot_hint("liệt kê các món bán chạy tuần này", LOT_META) == []          # not a lot question
    assert lot_hint("những thuốc nào còn hàng có hạn dùng tới cuối năm", {"tables": LOT_META["tables"][:1]}) == []  # no lot link table
    assert LIST_WORDS.search(norm("Những SP nào")) and not LIST_WORDS.search(norm("cacao sữa"))
