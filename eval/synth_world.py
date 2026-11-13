"""Thế giới shop tự tạo cho sinh dữ liệu đa dạng: danh mục tự soạn, mô phỏng bản ghi, 6 họ schema, oracle độc lập, đối chứng.

Không đọc dữ liệu BTC. Danh mục, tên hàng, tên khách, quy ước đều tự soạn trong file này.
Oracle tính từ bản ghi Python (vòng lặp + Decimal), không dùng pandas, dm hay code model: đáp án không phụ thuộc runtime.
Đối chứng (mutation) là bản ghi đã sửa có chủ đích; đáp án được tính lại bằng cùng oracle.
"""
from __future__ import annotations

import calendar
import collections
import copy
import re
import datetime as dt
import random
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

# ---------------------------------------------------------------- danh mục tự soạn
# (tên, nhóm, đơn vị gốc, giá/đơn vị gốc, [(đơn vị lớn, số đơn vị gốc)]) — đơn vị lớn tính theo đơn vị gốc
INDUSTRIES = {
    "nha_sach_vpp": {"label": "nhà sách – văn phòng phẩm", "shop": "Nhà sách Hoa Mai", "prefix": "VP", "rate": 9, "lines": 2.4, "named": 0.45,
                     "channels": ["Tại quầy", "Giao hàng", "Zalo"], "kinds": ["Khách quen", "Trường học", "Công ty"],
                     "debt": True, "lots": False, "purchases": True, "refund": 0.04, "missing": ["tiền lương nhân viên", "tiền thuê mặt bằng"],
                     "items": [("Bút bi Hoa Mai 0.5", "Bút viết", "cây", 4000, [("hộp", 20), ("thùng", 400)]),
                               ("Bút gel Lam Ngọc", "Bút viết", "cây", 7000, [("hộp", 12)]),
                               ("Bút chì gỗ Sóc Nâu 2B", "Bút viết", "cây", 3500, [("hộp", 10)]),
                               ("Vở kẻ ngang Mây Trắng 96 trang", "Vở – giấy", "quyển", 9000, [("lốc", 10)]),
                               ("Vở ô ly Cánh Diều 48 trang", "Vở – giấy", "quyển", 6000, [("lốc", 10)]),
                               ("Giấy in A4 Tân Phú 70gsm", "Vở – giấy", "ram", 72000, [("thùng", 5)]),
                               ("Thước kẻ nhựa 20cm", "Dụng cụ học tập", "cái", 5000, [("hộp", 24)]),
                               ("Gôm tẩy Bông Gòn", "Dụng cụ học tập", "cục", 3000, [("hộp", 30)]),
                               ("Băng keo trong 5cm", "Dụng cụ học tập", "cuộn", 12000, [("cây", 6)]),
                               ("Sách tô màu Vườn Thú", "Sách", "cuốn", 25000, []),
                               ("Truyện tranh Cậu Bé Mây tập 1", "Sách", "cuốn", 22000, [])]},
    "tiem_banh": {"label": "tiệm bánh", "shop": "Tiệm bánh Mật Ong", "prefix": "TB", "rate": 12, "lines": 1.9, "named": 0.3,
                  "channels": ["Tại quầy", "GrabFood", "Đặt trước"], "kinds": ["Khách quen", "Văn phòng"],
                  "debt": False, "lots": False, "purchases": True, "refund": 0.03, "missing": ["tiền điện", "tiền lương thợ làm bánh"],
                  "items": [("Bánh kem bắp size 16", "Bánh kem", "cái", 180000, []),
                            ("Bánh kem dâu size 20", "Bánh kem", "cái", 260000, []),
                            ("Bánh bông lan trứng muối", "Bánh ngọt", "hộp", 65000, []),
                            ("Bánh su kem", "Bánh ngọt", "cái", 6000, [("hộp", 10)]),
                            ("Bánh tart trứng", "Bánh ngọt", "cái", 15000, [("hộp", 6)]),
                            ("Bánh mì hoa cúc", "Bánh mì", "ổ", 45000, []),
                            ("Bánh croissant bơ", "Bánh mì", "cái", 22000, [("hộp", 4)]),
                            ("Bánh quy bơ hạnh nhân", "Bánh quy", "hũ", 85000, []),
                            ("Bánh flan caramen", "Bánh ngọt", "hũ", 12000, [("khay", 6)]),
                            ("Trà đào đóng chai", "Đồ uống", "chai", 25000, [("lốc", 6)])]},
    "my_pham": {"label": "mỹ phẩm", "shop": "Mỹ phẩm Sen Hồng", "prefix": "MP", "rate": 8, "lines": 1.8, "named": 0.5,
                "channels": ["Tại quầy", "Website", "Livestream", "Sàn TMĐT"], "kinds": ["Khách quen", "Cộng tác viên"],
                "debt": False, "lots": True, "purchases": True, "refund": 0.05, "missing": ["chi phí quảng cáo", "tiền lương nhân viên"],
                "items": [("Sữa rửa mặt Lá Trà 100ml", "Chăm sóc da", "tuýp", 95000, []),
                          ("Kem chống nắng Mây Xanh SPF50", "Chăm sóc da", "tuýp", 185000, []),
                          ("Kem dưỡng ẩm Yến Mạch 50g", "Chăm sóc da", "hũ", 245000, []),
                          ("Mặt nạ giấy Nha Đam", "Chăm sóc da", "miếng", 18000, [("hộp", 10)]),
                          ("Son lì Hồng Đào số 03", "Trang điểm", "thỏi", 159000, []),
                          ("Phấn phủ Ngọc Trai", "Trang điểm", "hộp", 210000, []),
                          ("Nước tẩy trang Sen Trắng 400ml", "Trang điểm", "chai", 135000, []),
                          ("Dầu gội Bồ Kết Xanh 300ml", "Chăm sóc tóc", "chai", 89000, [("thùng", 24)]),
                          ("Sữa tắm Hoa Nhài 500ml", "Cơ thể", "chai", 120000, []),
                          ("Bông tẩy trang 80 miếng", "Cơ thể", "gói", 25000, [("bịch", 10)])]},
    "thoi_trang": {"label": "thời trang – giày dép", "shop": "Shop thời trang Gió Nam", "prefix": "TT", "rate": 7, "lines": 1.7, "named": 0.4,
                   "channels": ["Tại cửa hàng", "Facebook", "Sàn TMĐT"], "kinds": ["Khách quen", "Khách sỉ"],
                   "debt": False, "lots": False, "purchases": True, "refund": 0.07, "missing": ["tiền thuê mặt bằng", "chi phí quảng cáo"],
                   "items": [("Áo thun cổ tròn Bông Cotton", "Áo", "cái", 129000, []),
                             ("Áo sơ mi trắng công sở", "Áo", "cái", 239000, []),
                             ("Quần jean ống suông", "Quần", "cái", 329000, []),
                             ("Quần short kaki", "Quần", "cái", 189000, []),
                             ("Váy hoa nhí dáng dài", "Váy", "cái", 299000, []),
                             ("Giày thể thao trắng Gió Nam", "Giày dép", "đôi", 459000, []),
                             ("Dép quai ngang Biển Xanh", "Giày dép", "đôi", 99000, []),
                             ("Tất cổ ngắn", "Phụ kiện", "đôi", 15000, [("lốc", 5)]),
                             ("Mũ lưỡi trai thêu chữ", "Phụ kiện", "cái", 119000, []),
                             ("Thắt lưng da bò", "Phụ kiện", "cái", 279000, [])]},
    "vat_tu_nn": {"label": "vật tư nông nghiệp", "shop": "Đại lý vật tư Đồng Xanh", "prefix": "NN", "rate": 6, "lines": 2.1, "named": 0.7,
                  "channels": ["Tại cửa hàng", "Giao tận ruộng"], "kinds": ["Nông hộ", "Hợp tác xã"],
                  "debt": True, "lots": True, "purchases": True, "refund": 0.03, "missing": ["tiền lương nhân viên", "diện tích ruộng của khách"],
                  "items": [("Phân NPK Đồng Xanh 16-16-8", "Phân bón", "kg", 18000, [("bao", 50)]),
                            ("Phân Urê Hạt Vàng", "Phân bón", "kg", 14000, [("bao", 50)]),
                            ("Phân hữu cơ Đất Mẹ", "Phân bón", "kg", 9000, [("bao", 25)]),
                            ("Thuốc trừ sâu Lúa Vàng 240ml", "Thuốc BVTV", "chai", 65000, [("thùng", 40)]),
                            ("Thuốc diệt cỏ Ruộng Sạch 480ml", "Thuốc BVTV", "chai", 85000, [("thùng", 20)]),
                            ("Hạt giống rau muống Mùa Xuân", "Hạt giống", "gói", 10000, [("túi", 10)]),
                            ("Hạt giống dưa leo F1", "Hạt giống", "gói", 25000, []),
                            ("Bình xịt tay 2 lít", "Dụng cụ", "cái", 95000, []),
                            ("Dây tưới nhỏ giọt", "Dụng cụ", "mét", 3000, [("cuộn", 100)]),
                            ("Kéo cắt cành", "Dụng cụ", "cái", 75000, [])]},
    "gas_nuoc": {"label": "gas – nước uống", "shop": "Đại lý gas nước Suối Mát", "prefix": "GN", "rate": 10, "lines": 1.4, "named": 0.65,
                 "channels": ["Giao tận nhà", "Tại cửa hàng", "Zalo"], "kinds": ["Hộ gia đình", "Quán ăn", "Văn phòng"],
                 "debt": True, "lots": False, "purchases": True, "refund": 0.02, "missing": ["tiền xăng xe giao hàng", "tiền lương nhân viên giao"],
                 "items": [("Bình gas 12kg Lửa Xanh", "Gas", "bình", 420000, []),
                           ("Bình gas 6kg Lửa Xanh", "Gas", "bình", 230000, []),
                           ("Nước tinh khiết bình 20 lít Suối Mát", "Nước uống", "bình", 55000, []),
                           ("Nước suối chai 500ml Suối Mát", "Nước uống", "chai", 5000, [("thùng", 24)]),
                           ("Nước khoáng chai 1.5 lít", "Nước uống", "chai", 11000, [("thùng", 12)]),
                           ("Nước suối chai 350ml", "Nước uống", "chai", 4000, [("thùng", 24)]),
                           ("Van điều áp gas", "Phụ kiện", "cái", 160000, []),
                           ("Dây dẫn gas 1.5m", "Phụ kiện", "sợi", 60000, []),
                           ("Bếp gas mặt kính", "Phụ kiện", "cái", 650000, [])]},
    "hoa_tuoi": {"label": "tiệm hoa tươi", "shop": "Tiệm hoa Nắng Hạ", "prefix": "HT", "rate": 8, "lines": 1.8, "named": 0.35,
                 "channels": ["Tại tiệm", "Giao hoa", "Facebook"], "kinds": ["Khách quen", "Công ty"],
                 "debt": False, "lots": True, "purchases": True, "refund": 0.03, "missing": ["tiền thuê mặt bằng", "chi phí xăng giao hoa"],
                 "items": [("Hoa hồng đỏ Đà Lạt", "Hoa cắt cành", "cành", 12000, [("bó", 10)]),
                           ("Hoa cúc họa mi", "Hoa cắt cành", "cành", 5000, [("bó", 20)]),
                           ("Hoa ly trắng", "Hoa cắt cành", "cành", 35000, [("bó", 5)]),
                           ("Hoa cẩm chướng", "Hoa cắt cành", "cành", 8000, [("bó", 10)]),
                           ("Bó hoa sinh nhật mẫu Nắng Hạ", "Bó hoa", "bó", 350000, []),
                           ("Bó hoa hồng 20 bông", "Bó hoa", "bó", 450000, []),
                           ("Lẵng hoa chúc mừng", "Lẵng – kệ", "lẵng", 850000, []),
                           ("Kệ hoa khai trương 2 tầng", "Lẵng – kệ", "kệ", 1500000, []),
                           ("Giấy gói hoa", "Phụ kiện", "tờ", 5000, [("xấp", 20)]),
                           ("Thiệp chúc mừng", "Phụ kiện", "tấm", 10000, [])]},
    "dien_gia_dung": {"label": "điện gia dụng", "shop": "Điện máy Sáng Mãi", "prefix": "DG", "rate": 5, "lines": 1.5, "named": 0.55,
                      "channels": ["Tại cửa hàng", "Giao lắp đặt", "Website"], "kinds": ["Khách quen", "Thợ điện"],
                      "debt": True, "lots": False, "purchases": True, "refund": 0.04, "missing": ["tiền công lắp đặt của thợ", "tiền thuê kho"],
                      "items": [("Nồi cơm điện Gạo Thơm 1.8L", "Nhà bếp", "cái", 690000, []),
                                ("Ấm siêu tốc 1.7L", "Nhà bếp", "cái", 290000, []),
                                ("Máy xay sinh tố Bão Nhỏ", "Nhà bếp", "cái", 790000, []),
                                ("Quạt đứng 5 cánh Gió Mát", "Làm mát", "cái", 550000, []),
                                ("Quạt treo tường", "Làm mát", "cái", 480000, []),
                                ("Bóng đèn LED 12W Sáng Mãi", "Chiếu sáng", "bóng", 45000, [("hộp", 10)]),
                                ("Ổ cắm điện 6 lỗ", "Phụ kiện điện", "cái", 120000, []),
                                ("Dây điện đôi 2x1.5", "Phụ kiện điện", "mét", 9000, [("cuộn", 100)]),
                                ("Pin tiểu AA", "Phụ kiện điện", "viên", 6000, [("vỉ", 4), ("hộp", 40)]),
                                ("Bàn ủi hơi nước", "Nhà bếp", "cái", 420000, [])]},
    "tap_hoa": {"label": "tạp hóa", "shop": "Tạp hóa Cô Ba", "prefix": "TH", "rate": 14, "lines": 2.2, "named": 0.35,
                "channels": ["Tại quầy", "Giao hàng", "Zalo"], "kinds": ["Khách quen", "Quán ăn"],
                "debt": True, "lots": False, "purchases": True, "refund": 0.02, "missing": ["tiền điện", "tiền lương người bán"],
                "items": [("Mì ly Hảo Vị tôm", "Thực phẩm khô", "ly", 9000, [("thùng", 24)]),
                          ("Gạo thơm Hạt Ngọc", "Thực phẩm khô", "kg", 22000, [("bao", 25)]),
                          ("Bánh gạo vị phô mai", "Thực phẩm khô", "gói", 12000, [("thùng", 30)]),
                          ("Nước mắm cá cơm Biển Đông 500ml", "Gia vị", "chai", 38000, [("thùng", 12)]),
                          ("Dầu đậu nành Vàng Ươm 1L", "Gia vị", "chai", 52000, [("thùng", 12)]),
                          ("Đường cát trắng", "Gia vị", "kg", 24000, []),
                          ("Sữa hộp Đồng Cỏ 180ml", "Đồ uống", "hộp", 8000, [("lốc", 4), ("thùng", 48)]),
                          ("Trứng gà ta", "Thực phẩm tươi", "quả", 3500, [("vỉ", 10)]),
                          ("Bột giặt Nắng Sớm 3kg", "Hóa phẩm", "túi", 115000, []),
                          ("Nước rửa chén Chanh Xả 750ml", "Hóa phẩm", "chai", 32000, [("thùng", 12)])]},
    "ca_phe": {"label": "quán cà phê", "shop": "Cà phê Góc Phố", "prefix": "CF", "rate": 18, "lines": 1.6, "named": 0.15,
               "channels": ["Tại quán", "GrabFood", "Mang đi"], "kinds": ["Khách quen", "Văn phòng"],
               "debt": False, "lots": False, "purchases": True, "refund": 0.01, "missing": ["tiền lương nhân viên pha chế", "tiền điện nước"],
               "items": [("Cà phê sữa đá", "Cà phê", "ly", 25000, []),
                         ("Cà phê đen đá", "Cà phê", "ly", 20000, []),
                         ("Bạc xỉu", "Cà phê", "ly", 29000, []),
                         ("Trà đào cam sả", "Trà", "ly", 35000, []),
                         ("Trà sữa trân châu", "Trà", "ly", 32000, []),
                         ("Nước cam ép", "Nước ép – sinh tố", "ly", 38000, []),
                         ("Sinh tố bơ", "Nước ép – sinh tố", "ly", 42000, []),
                         ("Bánh croissant", "Bánh", "cái", 25000, []),
                         ("Bánh flan", "Bánh", "hũ", 15000, []),
                         ("Cà phê hạt rang xay 500g", "Cà phê gói", "gói", 160000, [])]},
    "vlxd": {"label": "vật liệu xây dựng", "shop": "Vật liệu xây dựng Trường Sơn", "prefix": "XD", "rate": 5, "lines": 2.0, "named": 0.7,
             "channels": ["Tại bãi", "Giao công trình"], "kinds": ["Thầu xây dựng", "Hộ gia đình"],
             "debt": True, "lots": False, "purchases": True, "refund": 0.02, "missing": ["tiền công thợ hồ", "chi phí xe cẩu"],
             "items": [("Xi măng Sông Lam PCB40", "Xi măng – cát đá", "bao", 92000, [("tấn", 20)]),
                       ("Cát xây tô", "Xi măng – cát đá", "khối", 350000, []),
                       ("Đá 1x2", "Xi măng – cát đá", "khối", 420000, []),
                       ("Thép thanh phi 12", "Sắt thép", "cây", 185000, [("bó", 10)]),
                       ("Thép cuộn phi 6", "Sắt thép", "kg", 17000, [("cuộn", 200)]),
                       ("Gạch ống 4 lỗ", "Gạch", "viên", 1300, [("kiện", 500)]),
                       ("Gạch men 60x60", "Gạch", "viên", 65000, [("thùng", 4)]),
                       ("Sơn nước nội thất 18L", "Sơn", "thùng", 1450000, []),
                       ("Sơn lót chống kiềm 5L", "Sơn", "lon", 520000, []),
                       ("Ống nhựa PVC phi 27", "Điện nước", "cây", 38000, [("bó", 20)])]},
    "nha_thuoc": {"label": "nhà thuốc", "shop": "Nhà thuốc An Khang Phúc", "prefix": "NT", "rate": 12, "lines": 1.8, "named": 0.3,
                  "channels": ["Tại quầy", "Giao thuốc", "Zalo"], "kinds": ["Khách quen", "Phòng khám"],
                  "debt": False, "lots": True, "purchases": True, "refund": 0.02, "missing": ["tiền lương dược sĩ", "tiền thuê mặt bằng"],
                  "items": [("Viên hạ sốt Thanh Nhiệt 500mg", "Giảm đau – hạ sốt", "viên", 1000, [("vỉ", 10), ("hộp", 100)]),
                            ("Viên kẽm Sức Sống", "Vitamin", "viên", 2500, [("vỉ", 10), ("hộp", 30)]),
                            ("Viên sủi vitamin C Cam Vàng", "Vitamin", "tuýp", 45000, [("hộp", 6)]),
                            ("Siro ho Lá Hẹ 100ml", "Ho – cảm", "chai", 55000, []),
                            ("Men tiêu hóa Bụng Êm", "Tiêu hóa", "gói", 6000, [("hộp", 20)]),
                            ("Khẩu trang y tế 4 lớp", "Dụng cụ y tế", "cái", 1500, [("hộp", 50)]),
                            ("Nước muối sinh lý 500ml", "Dụng cụ y tế", "chai", 8000, [("thùng", 20)]),
                            ("Băng cá nhân", "Dụng cụ y tế", "miếng", 800, [("hộp", 100)]),
                            ("Nhiệt kế điện tử", "Dụng cụ y tế", "cái", 85000, []),
                            ("Dầu gió xanh Bạc Hà", "Ho – cảm", "chai", 28000, [("hộp", 12)])]},
    "quan_nhau": {"label": "quán nhậu", "shop": "Quán nhậu Sông Hàn", "prefix": "QN", "rate": 9, "lines": 3.2, "named": 0.2,
                  "channels": ["Tại quán", "Mang về"], "kinds": ["Khách quen", "Công ty"],
                  "debt": True, "lots": False, "purchases": True, "refund": 0.01, "missing": ["tiền lương nhân viên phục vụ", "tiền thuê mặt bằng"],
                  "items": [("Gà nướng muối ớt", "Món nhậu", "con", 260000, []),
                            ("Mực một nắng nướng", "Món nhậu", "đĩa", 180000, []),
                            ("Khô bò lá chanh", "Món nhậu", "đĩa", 90000, []),
                            ("Lẩu thái hải sản", "Lẩu", "nồi", 320000, []),
                            ("Lẩu dê", "Lẩu", "nồi", 350000, []),
                            ("Bia chai Sao Vàng", "Đồ uống", "chai", 18000, [("két", 20)]),
                            ("Bia lon Sông Hàn", "Đồ uống", "lon", 17000, [("thùng", 24)]),
                            ("Nước ngọt lon", "Đồ uống", "lon", 12000, [("thùng", 24)]),
                            ("Rau muống xào tỏi", "Món thêm", "đĩa", 45000, []),
                            ("Đậu phộng rang", "Món thêm", "đĩa", 25000, [])]},
    # 6 ngành bổ sung ngày 10/11 cho mục tiêu ≥3.000 dòng train (chỉ dùng ở train, danh mục tự soạn)
    "the_thao": {"label": "đồ thể thao", "shop": "Thể thao Sao Băng", "prefix": "ST", "rate": 7, "lines": 1.8, "named": 0.4,
                 "channels": ["Tại cửa hàng", "Sàn TMĐT", "Facebook"], "kinds": ["Khách quen", "Câu lạc bộ"],
                 "debt": True, "lots": False, "purchases": True, "refund": 0.05, "missing": ["tiền thuê sân tập", "tiền lương nhân viên"],
                 "items": [("Bóng đá số 5 Sao Băng", "Bóng – vợt", "quả", 280000, [("bao", 10)]),
                           ("Vợt cầu lông Lông Vũ", "Bóng – vợt", "cây", 450000, []),
                           ("Quả cầu lông lông vịt", "Bóng – vợt", "quả", 9000, [("ống", 12), ("thùng", 240)]),
                           ("Áo thể thao thun lạnh", "Quần áo thể thao", "cái", 120000, []),
                           ("Quần short chạy bộ", "Quần áo thể thao", "cái", 150000, []),
                           ("Giày chạy bộ Gió Lốc", "Giày thể thao", "đôi", 690000, []),
                           ("Bình nước thể thao 750ml", "Phụ kiện tập", "cái", 85000, [("thùng", 24)]),
                           ("Dây nhảy lõi thép", "Phụ kiện tập", "sợi", 65000, []),
                           ("Thảm tập yoga 6mm", "Phụ kiện tập", "tấm", 220000, []),
                           ("Băng quấn cổ tay", "Phụ kiện tập", "đôi", 35000, [("hộp", 10)])]},
    "nhac_cu": {"label": "nhạc cụ", "shop": "Nhạc cụ Cung Đàn", "prefix": "NC", "rate": 4, "lines": 1.6, "named": 0.55,
                "channels": ["Tại cửa hàng", "Website", "Zalo"], "kinds": ["Học viên", "Trung tâm âm nhạc"],
                "debt": True, "lots": False, "purchases": True, "refund": 0.04, "missing": ["học phí lớp đàn", "tiền lương giáo viên"],
                "items": [("Đàn guitar acoustic gỗ thông", "Đàn", "cây", 1850000, []),
                          ("Đàn ukulele soprano", "Đàn", "cây", 650000, []),
                          ("Sáo trúc Đông Hồ", "Sáo – kèn", "cây", 120000, []),
                          ("Dây đàn guitar bộ 6 sợi", "Phụ kiện đàn", "bộ", 90000, [("hộp", 10)]),
                          ("Phím gảy đàn", "Phụ kiện đàn", "cái", 5000, [("túi", 12), ("hộp", 144)]),
                          ("Capo kẹp đàn", "Phụ kiện đàn", "cái", 75000, []),
                          ("Bao đàn guitar", "Phụ kiện đàn", "cái", 250000, []),
                          ("Chân đàn gấp gọn", "Phụ kiện đàn", "cái", 180000, []),
                          ("Sách học đàn cơ bản", "Sách nhạc", "cuốn", 85000, [])]},
    "hai_san_kho": {"label": "hải sản khô", "shop": "Hải sản khô Biển Xanh", "prefix": "HS", "rate": 9, "lines": 2.0, "named": 0.5,
                    "channels": ["Tại cửa hàng", "Giao hàng", "Zalo"], "kinds": ["Khách quen", "Quán ăn"],
                    "debt": True, "lots": True, "purchases": True, "refund": 0.03, "missing": ["tiền thuê kho lạnh", "tiền lương người phơi khô"],
                    "items": [("Mực khô câu loại 1", "Mực – tôm", "kg", 950000, [("thùng", 5)]),
                              ("Tôm khô đất", "Mực – tôm", "kg", 780000, []),
                              ("Cá khô dứa một nắng", "Cá khô", "kg", 320000, [("thùng", 10)]),
                              ("Khô cá lóc", "Cá khô", "kg", 380000, []),
                              ("Cá cơm khô", "Cá khô", "gói", 45000, [("thùng", 20)]),
                              ("Mắm tôm chua", "Mắm", "hũ", 65000, [("thùng", 24)]),
                              ("Nước mắm nhĩ cá cơm", "Mắm", "chai", 120000, [("thùng", 12)]),
                              ("Khô gà lá chanh", "Đồ khô khác", "hũ", 75000, [])]},
    "cay_canh": {"label": "cây cảnh", "shop": "Vườn cây Xanh Mát", "prefix": "CC", "rate": 6, "lines": 1.9, "named": 0.35,
                 "channels": ["Tại vườn", "Giao hàng", "Facebook"], "kinds": ["Khách quen", "Quán cà phê"],
                 "debt": False, "lots": False, "purchases": True, "refund": 0.03, "missing": ["tiền nước tưới", "tiền công chăm cây"],
                 "items": [("Cây kim tiền chậu sứ", "Cây trong nhà", "chậu", 250000, []),
                           ("Cây lưỡi hổ", "Cây trong nhà", "chậu", 150000, []),
                           ("Cây bonsai mini", "Cây trong nhà", "chậu", 320000, []),
                           ("Sen đá mix", "Sen đá – xương rồng", "chậu", 25000, [("khay", 12)]),
                           ("Đất trồng hữu cơ 5dm3", "Vật tư trồng", "bao", 40000, []),
                           ("Phân trùn quế 2kg", "Vật tư trồng", "túi", 35000, [("bao", 10)]),
                           ("Chậu nhựa 30cm", "Chậu – dụng cụ", "cái", 30000, [("chồng", 10)]),
                           ("Bình tưới 5 lít", "Chậu – dụng cụ", "cái", 70000, []),
                           ("Hạt giống hoa hướng dương", "Vật tư trồng", "gói", 15000, [("túi", 10)])]},
    "do_an_vat": {"label": "đồ ăn vặt", "shop": "Tiệm ăn vặt Bé Na", "prefix": "AV", "rate": 15, "lines": 2.1, "named": 0.25,
                  "channels": ["Tại tiệm", "ShopeeFood", "Zalo"], "kinds": ["Khách quen", "Sinh viên"],
                  "debt": False, "lots": True, "purchases": True, "refund": 0.02, "missing": ["tiền điện", "tiền thuê ki-ốt"],
                  "items": [("Bánh tráng trộn", "Bánh tráng", "bịch", 20000, [("thùng", 30)]),
                            ("Bánh tráng nướng", "Bánh tráng", "cái", 15000, []),
                            ("Khô gà xé", "Đồ khô", "hũ", 60000, []),
                            ("Hạt hướng dương", "Hạt – snack", "gói", 12000, [("bịch", 10)]),
                            ("Snack khoai tây", "Hạt – snack", "gói", 10000, [("thùng", 24)]),
                            ("Me lắc muối ớt", "Đồ khô", "hũ", 35000, []),
                            ("Kẹo dẻo trái cây", "Kẹo", "gói", 18000, [("hộp", 20)]),
                            ("Cơm cháy chà bông", "Đồ khô", "gói", 30000, [("thùng", 20)]),
                            ("Trà sữa đóng chai", "Đồ uống", "chai", 22000, [("lốc", 6)])]},
    "vai_may": {"label": "vải – phụ liệu may", "shop": "Tiệm vải Chỉ Hồng", "prefix": "VM", "rate": 6, "lines": 2.3, "named": 0.6,
                "channels": ["Tại tiệm", "Giao hàng", "Zalo"], "kinds": ["Tiệm may", "Khách quen"],
                "debt": True, "lots": False, "purchases": True, "refund": 0.03, "missing": ["tiền công may", "tiền thuê mặt bằng"],
                "items": [("Vải kate trơn", "Vải", "mét", 35000, [("cây", 50)]),
                          ("Vải lụa tơ tằm", "Vải", "mét", 180000, [("cây", 30)]),
                          ("Vải jean co giãn", "Vải", "mét", 75000, [("cây", 40)]),
                          ("Chỉ may polyester", "Chỉ – kim", "cuộn", 12000, [("hộp", 10)]),
                          ("Kim may tay", "Chỉ – kim", "vỉ", 8000, []),
                          ("Nút áo sơ mi", "Phụ liệu", "cái", 1000, [("vỉ", 12), ("hộp", 144)]),
                          ("Dây kéo 20cm", "Phụ liệu", "sợi", 4000, [("bó", 100)]),
                          ("Thun luồn bản 2cm", "Phụ liệu", "mét", 3000, [("cuộn", 50)]),
                          ("Kéo cắt vải", "Dụng cụ may", "cây", 95000, [])]},
    # hai ngành chỉ dùng cho final test (chưa xuất hiện ở train/val)
    "thu_cung": {"label": "cửa hàng thú cưng", "shop": "Pet shop Mèo Mun", "prefix": "PC", "rate": 7, "lines": 1.9, "named": 0.55,
                 "channels": ["Tại cửa hàng", "Giao hàng", "Sàn TMĐT"], "kinds": ["Khách quen", "Phòng khám thú y"],
                 "debt": False, "lots": True, "purchases": True, "refund": 0.03, "missing": ["tiền lương nhân viên tắm thú", "tiền thuê mặt bằng"],
                 "items": [("Hạt cho chó vị bò 1.5kg", "Thức ăn", "túi", 165000, [("bao", 8)]),
                           ("Pate mèo cá ngừ 85g", "Thức ăn", "lon", 18000, [("lốc", 6), ("thùng", 48)]),
                           ("Hạt cho mèo vị cá hồi 1kg", "Thức ăn", "túi", 135000, []),
                           ("Cát vệ sinh mèo 5L", "Vệ sinh", "túi", 85000, [("bao", 4)]),
                           ("Sữa tắm thú cưng hương táo", "Vệ sinh", "chai", 95000, []),
                           ("Vòng cổ phản quang", "Phụ kiện", "cái", 45000, []),
                           ("Đồ chơi chuột bông", "Đồ chơi", "con", 25000, [("hộp", 12)]),
                           ("Bát ăn inox đôi", "Phụ kiện", "cái", 75000, []),
                           ("Xương gặm canxi", "Thức ăn", "cái", 12000, [("túi", 10)])]},
    "trai_cay": {"label": "trái cây – nông sản", "shop": "Vựa trái cây Bốn Mùa", "prefix": "TC", "rate": 11, "lines": 2.0, "named": 0.45,
                 "channels": ["Tại vựa", "Giao hàng", "Zalo"], "kinds": ["Khách quen", "Quán chè", "Siêu thị mini"],
                 "debt": True, "lots": False, "purchases": True, "refund": 0.03, "missing": ["tiền thuê xe tải", "tiền lương bốc xếp"],
                 "items": [("Xoài cát Hòa Lộc", "Trái cây nội", "kg", 65000, [("thùng", 10)]),
                           ("Bưởi da xanh", "Trái cây nội", "quả", 55000, [("sọt", 12)]),
                           ("Thanh long ruột đỏ", "Trái cây nội", "kg", 30000, [("thùng", 15)]),
                           ("Táo Envy nhập khẩu", "Trái cây nhập", "kg", 145000, [("thùng", 9)]),
                           ("Nho xanh không hạt", "Trái cây nhập", "kg", 120000, []),
                           ("Cam sành", "Trái cây nội", "kg", 28000, [("sọt", 20)]),
                           ("Dừa xiêm", "Trái cây nội", "quả", 15000, [("chục", 12)]),
                           ("Hộp quà trái cây", "Quà tặng", "hộp", 450000, [])]},
}
FINAL_ONLY = ("thu_cung", "trai_cay")
ADDED_TRAIN = ("the_thao", "nhac_cu", "hai_san_kho", "cay_canh", "do_an_vat", "vai_may")  # chỉ train, thêm sau khi đóng băng final
ORIGINAL = ("nha_sach_vpp", "tiem_banh", "my_pham", "thoi_trang", "vat_tu_nn", "gas_nuoc", "hoa_tuoi", "dien_gia_dung", "tap_hoa", "ca_phe",
            "vlxd", "nha_thuoc", "quan_nhau")  # thứ tự ngành gốc: shop_plan cũ dựa vào thứ tự này, không đổi

# Cách ghi trạng thái, khách lẻ: đều thuộc nhóm mà datamate.clean.norm_status / is_walkin nhận được.
STATUS_TEXT = {"done": ["Hoàn thành", "hoàn thành", "Xong", "xong", "Đã giao", "Thành công", "Hoàn tất", "Đã lấy"],
               "cancel": ["Hủy", "huỷ", "Đã hủy", "Khách hủy", "Giao thất bại"],
               "pending": ["Chờ xác nhận", "Đang xử lý", "Đang giao"]}
WALKIN_TEXT = ["Khách lẻ", "khách lẻ", "KL", "kl", "Khách vãng lai", ""]
HO = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý", "Mai", "Trịnh"]
DEM = ["Văn", "Thị", "Minh", "Ngọc", "Đức", "Thanh", "Quốc", "Hữu", "Thu", "Gia", "Bảo", "Kim", "Hoài", "Xuân"]
TEN = ["An", "Bình", "Châu", "Dũng", "Giang", "Hà", "Hải", "Hùng", "Khánh", "Lan", "Linh", "Long", "Mai", "Nam", "Nga", "Phúc", "Quân",
       "Sơn", "Tâm", "Thảo", "Trang", "Tuấn", "Uyên", "Việt", "Yến", "Đạt", "Hiền", "Lộc", "Nhung", "Thịnh"]
FAMILIES = ("joined", "flat", "vn_cols", "neg_lines", "events", "monthly")
WEEKDAY_VN = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]


def _fold_ascii(s: str) -> str:
    """Bỏ dấu, chữ thường (synth_checks.fold)."""
    import synth_checks
    return synth_checks.fold(s)


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) and not isinstance(x, dt.datetime) else dt.date.fromisoformat(str(x)[:10])


def _ts(x) -> dt.datetime:
    return x if isinstance(x, dt.datetime) else dt.datetime.fromisoformat(str(x))


def _month_add(y: int, m: int, k: int) -> tuple[int, int]:
    i = y * 12 + (m - 1) + k
    return i // 12, i % 12 + 1


def _month_end(y: int, m: int) -> dt.date:
    return dt.date(y, m, calendar.monthrange(y, m)[1])


# ---------------------------------------------------------------- mô phỏng bản ghi
RX_EXTRA = [("Kháng sinh Amoxi-Sao 500mg", "Thuốc kê đơn", "viên", 2500, [("vỉ", 10), ("hộp", 100)]),
            ("Thuốc huyết áp Bình An 5mg", "Thuốc kê đơn", "viên", 3000, [("vỉ", 10), ("hộp", 30)])]


def simulate(industry: str, seed: int, start: str, today: str, after_days: int = 0, variant: bool = False, complex_: bool = False) -> dict:
    """Bản ghi chuẩn của một shop: từ `start` đến hết `today` (+ `after_days` ngày dữ liệu sau TODAY).
    variant=True (chỉ shop thêm sau): lấy tập con danh mục và đổi nhịp bán theo hạt riêng; shop cũ (variant=False) giữ nguyên bản ghi."""
    ind, rng = dict(INDUSTRIES[industry]), random.Random(seed)
    if variant:
        vr = random.Random(seed * 7 + 1)
        keep = sorted(vr.sample(range(len(ind["items"])), max(7, int(round(len(ind["items"]) * vr.uniform(0.75, 0.95))))))
        ind["items"] = [ind["items"][i] for i in keep]
        ind["rate"] = max(3, int(round(ind["rate"] * vr.uniform(0.7, 1.4))))
        ind["named"] = min(0.8, max(0.15, ind["named"] + vr.uniform(-0.15, 0.15)))
    if complex_ and industry == "nha_thuoc":
        ind["items"] = list(ind["items"]) + RX_EXTRA
    t0, t1 = _d(start), _d(today) + dt.timedelta(days=after_days)
    items, units = [], []
    for i, (name, group, base, price, big) in enumerate(ind["items"], 1):
        code = f"{ind['prefix']}{i:03d}"
        p = int(round(price * rng.uniform(0.9, 1.1), -2))
        items.append({"item": code, "name": name, "group": group, "base": base, "price": p})
        units.append({"item": code, "unit": base, "factor": 1})
        units += [{"item": code, "unit": u, "factor": f} for u, f in big]
    pop = {it["item"]: rng.uniform(0.3, 3.0) for it in items}
    names = set()
    customers = []
    while len(customers) < rng.randint(14, 22):
        n = f"{rng.choice(HO)} {rng.choice(DEM)} {rng.choice(TEN)}"
        if n not in names:
            names.add(n)
            customers.append({"cust": f"KH{len(customers) + 1:03d}", "name": n, "kind": rng.choice(ind["kinds"])})
    cweight = {c["cust"]: 1 / (k + 1) ** 0.8 for k, c in enumerate(customers)}
    st = {k: rng.sample(v, min(len(v), 3 if k == "done" else 2)) for k, v in STATUS_TEXT.items()}
    walk = rng.sample(WALKIN_TEXT, 3)
    rec = {"industry": industry, "shop_name": ind["shop"], "seed": seed, "start": t0.isoformat(), "today": _d(today).isoformat(),
           "data_end": t1.isoformat(), "items": items, "units": units, "customers": customers, "status_text": st, "walkin_text": walk,
           "orders": [], "lines": [], "refunds": [], "payments": [], "purchases": [], "lots": [],
           "features": {k: bool(ind[k]) for k in ("debt", "lots", "purchases")}}
    fac = {(u["item"], u["unit"]): u["factor"] for u in units}
    price = {it["item"]: it["price"] for it in items}
    cr = random.Random(seed * 13 + 5)  # thuộc tính chỉ shop phức tạp có; hạt riêng
    hist = {}
    if complex_:
        rec["features"].update(price_history=True, cost=True, stock=True, staff=True, rx=industry == "nha_thuoc")
        for it in items:
            it["cost"] = int(round(it["price"] * cr.uniform(0.55, 0.8), -2))
            if industry == "nha_thuoc":
                it["rx"] = it["group"] == "Thuốc kê đơn"
            if cr.random() < 0.65:  # 1–2 lần đổi giá trong kỳ dữ liệu
                days = sorted(cr.sample(range(20, max(21, (t1 - t0).days - 5)), cr.choice([1, 2])))
                p, ch = it["price"], []
                for k in days:
                    p = int(round(p * cr.choice([0.9, 0.95, 1.05, 1.1, 1.15]), -2))
                    ch.append(((t0 + dt.timedelta(days=k)).isoformat(), p))
                hist[it["item"]] = ch
        rec["price_history"] = [{"item": c, "from": d, "price": pr} for c, ch in sorted(hist.items()) for d, pr in ch]
        used = set(names)
        staff = []
        while len(staff) < cr.randint(3, 5):
            nm = f"{cr.choice(HO)} {cr.choice(DEM)} {cr.choice(TEN)}"
            if nm not in used:
                used.add(nm)
                staff.append(nm)
        rec["staff"] = staff
        h1 = cr.choice([11, 12, 13])
        h2 = cr.choice([17, 18])
        rec["shifts"] = [{"shift": "ca sáng", "from": 0, "to": h1}, {"shift": "ca chiều", "from": h1, "to": h2}, {"shift": "ca tối", "from": h2, "to": 24}]
        sweight = [cr.uniform(0.5, 2) for _ in staff]

    def price_at(code, day):
        p = price[code]
        for d, pr in hist.get(code, []):
            if _d(d) <= day:
                p = pr
        return p
    big_of = {it["item"]: [u for u in units if u["item"] == it["item"] and u["factor"] > 1] for it in items}
    n = 0

    def add_order(at: dt.datetime, status: str, cust, kinds_lines=None, force_items=None, scale=1):
        nonlocal n
        n += 1
        oid = f"DH{n:05d}"
        pay = "Tiền mặt" if rng.random() < 0.55 else "Chuyển khoản"
        if ind["debt"] and cust and rng.random() < 0.3:
            pay = "Ghi nợ"
        rec["orders"].append({"order": oid, "at": at.isoformat(sep=" "), "status": status, "status_text": rng.choice(st[status]),
                              "channel": rng.choices(ind["channels"], [3] + [1] * (len(ind["channels"]) - 1))[0], "cust": cust,
                              "walkin_text": rng.choice(walk) if cust is None else None, "payment": pay})
        if complex_:
            rec["orders"][-1]["staff"] = cr.choices(rec["staff"], sweight)[0]
        k = max(1, min(len(items), int(round(rng.expovariate(1 / ind["lines"]))) or 1))
        chosen = force_items or _weighted_sample(rng, [it["item"] for it in items], [pop[it["item"]] for it in items], k)
        for j, code in enumerate(chosen, 1):
            big = big_of[code]
            pc = price_at(code, at.date()) if complex_ else price[code]
            if big and rng.random() < 0.3:
                u = rng.choice(big)
                unit, q = u["unit"], rng.randint(1, 3)
                up = int(round(pc * u["factor"] * rng.choice([1, 1, 0.97, 0.95]), -2))
            else:
                unit, q, up = next(it["base"] for it in items if it["item"] == code), rng.randint(1, 6) * scale, pc
            rec["lines"].append({"order": oid, "line": j, "item": code, "unit": unit, "qty": q, "price": up, "amount": q * up})
        return oid

    day = t0
    while day <= t1:
        wk = [1.0, 0.9, 0.95, 1.0, 1.15, 1.35, 1.3][day.weekday()]
        cnt = max(1, int(rng.gauss(ind["rate"] * wk, ind["rate"] * 0.25)))
        for _ in range(cnt):
            at = dt.datetime.combine(day, dt.time(rng.randint(7, 21), rng.randint(0, 59), rng.randint(0, 59)))
            recent = (t1 - day).days <= 3
            status = rng.choices(["done", "cancel", "pending"], [0.88, 0.08, 0.04 if recent else 0.012])[0]
            cust = rng.choices(list(cweight), list(cweight.values()))[0] if rng.random() < ind["named"] else None
            add_order(at, status, cust)
        day += dt.timedelta(days=1)
    # bẫy có chủ đích: đơn sát mốc nửa đêm đầu/cuối tháng, đơn khuya đúng TODAY, đơn lớn của khách lẻ mỗi tháng
    y, m = t0.year, t0.month
    while dt.date(y, m, 1) <= t1:
        end = _month_end(y, m)
        if end <= t1:
            add_order(dt.datetime.combine(end, dt.time(23, 59, rng.randint(50, 59))), "done", rng.choice([None, customers[0]["cust"]]))
            nxt = end + dt.timedelta(days=1)
            if nxt <= t1:
                add_order(dt.datetime.combine(nxt, dt.time(0, 0, rng.randint(0, 9))), "done", None)
        big_day = dt.date(y, m, min(rng.randint(5, 25), end.day))
        if t0 <= big_day <= _d(today):
            add_order(dt.datetime.combine(big_day, dt.time(rng.randint(9, 18), 0)), "done", None, scale=8)
        y, m = _month_add(y, m, 1)
    add_order(dt.datetime.combine(_d(today), dt.time(23, 58, rng.randint(0, 59))), "done", rng.choice([None, customers[1]["cust"]]))
    rec["orders"].sort(key=lambda o: (o["at"], o["order"]))
    _refunds(rec, rng, ind["refund"])
    if ind["debt"]:
        _payments(rec, rng)
    if ind["purchases"]:
        _purchases(rec, rng, fac)
    if ind["lots"]:
        _lots(rec, rng)
    if complex_:  # tồn đầu kỳ đủ để tồn không âm: tồn(D) = tồn đầu + nhập đến D − bán (đơn hoàn thành) đến D
        mv = collections.defaultdict(list)
        by_order = collections.defaultdict(list)
        for ln in rec["lines"]:
            by_order[ln["order"]].append(ln)
        for o in rec["orders"]:
            if o["status"] == "done":
                for ln in by_order[o["order"]]:
                    mv[ln["item"]].append((o["at"], -ln["qty"] * fac[(ln["item"], ln["unit"])]))
        for x in rec["purchases"]:
            mv[x["item"]].append((x["at"], x["qty"] * fac[(x["item"], x["unit"])]))
        rec["opening_stock"] = []
        for it in items:
            run, low = 0, 0
            for _, dq in sorted(mv[it["item"]]):
                run += dq
                low = min(low, run)
            rec["opening_stock"].append({"item": it["item"], "qty": -low + cr.randint(10, 80)})
    return rec


def _weighted_sample(rng, xs, ws, k):
    xs, ws, out = list(xs), list(ws), []
    for _ in range(min(k, len(xs))):
        i = rng.choices(range(len(xs)), ws)[0]
        out.append(xs.pop(i))
        ws.pop(i)
    return out


def _refunds(rec, rng, p):
    """Hoàn tiền một phần dòng hàng; ngày hoàn sau ngày bán 0–12 ngày (có ca vắt qua tháng), mỗi tháng ít nhất một ca chéo kỳ."""
    end = _d(rec["data_end"])
    lines = {}
    for ln in rec["lines"]:
        lines.setdefault(ln["order"], []).append(ln)
    done = [o for o in rec["orders"] if o["status"] == "done" and o["payment"] != "Ghi nợ"]
    picked = [o for o in done if rng.random() < p]
    by_month = {}
    for o in done:
        d = _d(o["at"])
        if (_month_end(d.year, d.month) - d).days <= 3:
            by_month.setdefault((d.year, d.month), []).append(o)
    cross_pick = {rng.choice(by_month[key])["order"] for key in sorted(by_month)}  # mỗi tháng một ca hoàn sang đầu tháng sau
    picked += [o for o in done if o["order"] in cross_pick]
    seen = set()
    for o in sorted(picked, key=lambda o: o["order"]):
        if o["order"] in seen:
            continue
        seen.add(o["order"])
        sold, d = _ts(o["at"]), _d(o["at"])
        if o["order"] in cross_pick:
            delay = (_month_end(d.year, d.month) - d).days + rng.randint(1, 4)
        else:
            delay = min(int(rng.expovariate(1 / 3)), 12)
        at = dt.datetime.combine(d + dt.timedelta(days=delay), dt.time(rng.randint(8, 20), rng.randint(0, 59)))
        if at <= sold:
            at = sold + dt.timedelta(minutes=rng.randint(20, 90))
        if at.date() > end:
            continue
        ln = rng.choice(lines[o["order"]])
        q = rng.randint(1, ln["qty"])
        rec["refunds"].append({"refund": f"TH{len(rec['refunds']) + 1:04d}", "order": o["order"], "at": at.isoformat(sep=" "),
                               "item": ln["item"], "unit": ln["unit"], "qty": q, "amount": q * ln["price"]})
    rec["refunds"].sort(key=lambda r: (r["at"], r["refund"]))


def _payments(rec, rng):
    end = _d(rec["data_end"])
    lines = {}
    for ln in rec["lines"]:
        lines[ln["order"]] = lines.get(ln["order"], 0) + ln["amount"]
    owed = {}
    for o in rec["orders"]:
        if o["status"] == "done" and o["payment"] == "Ghi nợ":
            owed.setdefault(o["cust"], []).append((_d(o["at"]), lines[o["order"]]))
    for cust, debts in sorted(owed.items()):
        for d, amt in debts:
            if rng.random() < 0.6:
                pay_day = d + dt.timedelta(days=rng.randint(3, 40))
                if pay_day <= end:
                    amt_paid = int(round(amt * rng.choice([1, 1, 0.5, 0.3]), -3))
                    if amt_paid > 0:
                        rec["payments"].append({"payment": f"TN{len(rec['payments']) + 1:04d}", "cust": cust,
                                                "at": dt.datetime.combine(pay_day, dt.time(rng.randint(8, 19), 0)).isoformat(sep=" "),
                                                "amount": amt_paid})
    rec["payments"].sort(key=lambda r: (r["at"], r["payment"]))


def _purchases(rec, rng, fac):
    d, end = _d(rec["start"]), _d(rec["data_end"])
    while d <= end:
        if d.weekday() == rng.choice([0, 2, 4]):
            for it in rng.sample(rec["items"], 2):
                units = [u for u in rec["units"] if u["item"] == it["item"]]
                u = max(units, key=lambda x: x["factor"])
                q = rng.randint(1, 5) if u["factor"] > 1 else rng.randint(5, 30)
                cost = int(round(q * u["factor"] * it["price"] * rng.uniform(0.6, 0.8), -3))
                rec["purchases"].append({"receipt": f"PN{len(rec['purchases']) + 1:04d}",
                                         "at": dt.datetime.combine(d, dt.time(rng.randint(7, 10), 0)).isoformat(sep=" "),
                                         "item": it["item"], "unit": u["unit"], "qty": q, "amount": cost})
        d += dt.timedelta(days=1)


def _lots(rec, rng):
    today = _d(rec["today"])
    for it in rec["items"]:
        for _ in range(rng.randint(2, 3)):
            exp = today + dt.timedelta(days=rng.randint(-20, 150))
            rec["lots"].append({"lot": f"LO{len(rec['lots']) + 1:04d}", "item": it["item"], "expiry": exp.isoformat(),
                                "qty_left": rng.choice([0, rng.randint(3, 60), rng.randint(3, 60)])})


# ---------------------------------------------------------------- kỳ thời gian (độc lập với dm)
def resolve(period: dict, today) -> tuple[dt.date, dt.date]:
    """Kỳ → (ngày đầu, ngày cuối), cả hai đầu tính trọn ngày."""
    t, k = _d(today), period["kind"]
    if k == "day":
        d = _d(period["d"])
        return d, d
    if k == "range":
        return _d(period["d1"]), _d(period["d2"])
    if k == "month":
        return dt.date(period["y"], period["m"], 1), _month_end(period["y"], period["m"])
    if k == "quarter":
        m0 = 3 * (period["q"] - 1) + 1
        return dt.date(period["y"], m0, 1), _month_end(period["y"], m0 + 2)
    if k == "year":
        return dt.date(period["y"], 1, 1), dt.date(period["y"], 12, 31)
    if k == "today":
        return t, t
    if k == "yesterday":
        return t - dt.timedelta(days=1), t - dt.timedelta(days=1)
    if k == "this_month":
        return t.replace(day=1), t
    if k == "last_month":
        y, m = _month_add(t.year, t.month, -1)
        return dt.date(y, m, 1), _month_end(y, m)
    if k == "this_week":
        return t - dt.timedelta(days=t.weekday()), t
    if k == "last_week":
        mon = t - dt.timedelta(days=t.weekday() + 7)
        return mon, mon + dt.timedelta(days=6)
    if k == "last_n":
        return t - dt.timedelta(days=period["n"] - 1), t
    if k == "this_year":
        return dt.date(t.year, 1, 1), t
    if k == "until":
        return dt.date(1900, 1, 1), _d(period["d"])
    if k == "until_today":
        return dt.date(1900, 1, 1), t
    if k == "prev_same_days":  # cùng số ngày đầu tháng trước (so với 'tháng này')
        y, m = _month_add(t.year, t.month, -1)
        return dt.date(y, m, 1), min(dt.date(y, m, 1) + dt.timedelta(days=t.day - 1), _month_end(y, m))
    if k == "week_before_last":
        mon = t - dt.timedelta(days=t.weekday() + 14)
        return mon, mon + dt.timedelta(days=6)
    if k == "prev_n":  # N ngày liền trước 'N ngày qua'
        return t - dt.timedelta(days=2 * period["n"] - 1), t - dt.timedelta(days=period["n"])
    raise ValueError("kỳ không hỗ trợ: " + k)


RELATIVE = {"today", "yesterday", "this_month", "last_month", "this_week", "last_week", "last_n", "this_year", "until_today",
            "prev_same_days", "week_before_last", "prev_n"}


def period_text(period: dict, today) -> str:
    """Cách nói kỳ trong câu hỏi (teacher phải giữ nguyên nghĩa)."""
    k = period["kind"]
    f = lambda d: f"{d.day}/{d.month}/{d.year}"  # noqa: E731
    return {"day": lambda: f"ngày {f(_d(period.get('d', today)))}", "range": lambda: f"từ ngày {f(_d(period.get('d1', today)))} đến ngày {f(_d(period.get('d2', today)))}",
            "month": lambda: f"tháng {period.get('m')}/{period.get('y')}", "quarter": lambda: f"quý {period.get('q')}/{period.get('y')}",
            "year": lambda: f"năm {period.get('y')}", "today": lambda: "hôm nay", "yesterday": lambda: "hôm qua",
            "this_month": lambda: "tháng này (từ ngày 1 đến hôm nay)", "last_month": lambda: "tháng trước",
            "this_week": lambda: "tuần này (từ thứ Hai đến hôm nay)", "last_week": lambda: "tuần trước",
            "last_n": lambda: f"{period.get('n')} ngày qua (tính cả hôm nay)", "this_year": lambda: "năm nay (từ đầu năm đến hôm nay)",
            "until": lambda: f"tính đến hết ngày {f(_d(period.get('d', today)))}", "until_today": lambda: "tính đến hôm nay",
            "prev_same_days": lambda: "cùng kỳ tháng trước (cùng số ngày đầu tháng)", "week_before_last": lambda: "tuần trước nữa",
            "prev_n": lambda: f"{period.get('n')} ngày liền trước đó"}[k]()


# ---------------------------------------------------------------- oracle độc lập
class View:
    """Chỉ mục bản ghi cho oracle; bỏ mọi bản ghi sau hết ngày TODAY."""

    def __init__(self, rec: dict, today):
        self.rec, self.today = rec, _d(today)
        self.items = {it["item"]: it for it in rec["items"]}
        self.fac = {(u["item"], u["unit"]): u["factor"] for u in rec["units"]}
        self.cust = {c["cust"]: c for c in rec["customers"]}
        self.orders = [o for o in rec["orders"] if _d(o["at"]) <= self.today]
        self.lines = {}
        for ln in rec["lines"]:
            self.lines.setdefault(ln["order"], []).append(ln)
        self.refunds = [r for r in rec["refunds"] if _d(r["at"]) <= self.today]

    def orders_in(self, s, e, status=None, channel=None):
        return [o for o in self.orders if s <= _d(o["at"]) <= e and (status is None or o["status"] == status)
                and (channel is None or o["channel"] == channel)]

    def done_lines(self, s, e, channel=None):
        return [(o, ln) for o in self.orders_in(s, e, "done", channel) for ln in self.lines.get(o["order"], [])]

    def base_qty(self, ln) -> int:
        return ln["qty"] * self.fac[(ln["item"], ln["unit"])]


def _money(x: Fraction) -> int:
    return int((Decimal(x.numerator) / Decimal(x.denominator)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _pct(num, den) -> float:
    v = Fraction(num) * 100 / Fraction(den)
    return float((Decimal(v.numerator) / Decimal(v.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def half_tie(num, den) -> bool:
    """Phần trăm đúng nằm đúng mốc …5 ở chữ số thứ ba: formatter làm tròn kiểu float có thể lệch bộ chấm, không dùng làm mẫu."""
    return (Fraction(num) * 100 * 100 / Fraction(den)).denominator == 2


def gold(rec: dict, spec: dict) -> dict:
    """Đáp án chuẩn của đặc tả trên bản ghi. Trả {'type', 'answer'} hoặc {'type': 'unanswerable'}."""
    v = View(rec, spec["as_of"])
    s, e = resolve(spec["period"], spec["as_of"]) if spec.get("period") else (None, None)
    slot, p = spec["slot"], spec.get("params", {})
    typ = SLOTS[slot]["type"]
    if slot == "unanswerable":
        return {"type": "unanswerable"}

    def revenue(s, e, item=None, group=None, channel=None):
        return sum(ln["amount"] for o, ln in v.done_lines(s, e, channel)
                   if (item is None or ln["item"] == item) and (group is None or v.items[ln["item"]]["group"] == group))

    def refunds(s, e):
        return sum(r["amount"] for r in v.refunds if s <= _d(r["at"]) <= e)

    def item_qty(s, e):
        q = {c: 0 for c in v.items}
        for _, ln in v.done_lines(s, e):
            q[ln["item"]] += v.base_qty(ln)
        return q

    def cust_spend(s, e):
        out = {}
        for o, ln in v.done_lines(s, e):
            if o["cust"]:
                name = v.cust[o["cust"]]["name"]
                out[name] = out.get(name, 0) + ln["amount"]
        return out

    def debts(d):
        out = {}
        for o in v.orders:
            if o["status"] == "done" and o["payment"] == "Ghi nợ" and _d(o["at"]) <= d:
                name = v.cust[o["cust"]]["name"]
                out[name] = out.get(name, 0) + sum(ln["amount"] for ln in v.lines[o["order"]])
        for pmt in rec["payments"]:
            if _d(pmt["at"]) <= d:
                name = v.cust[pmt["cust"]]["name"]
                out[name] = out.get(name, 0) - pmt["amount"]
        return out

    def daily(s, e):
        out = {}
        for o, ln in v.done_lines(s, e):
            k = _d(o["at"])
            out[k] = out.get(k, 0) + ln["amount"]
        return out

    if slot == "revenue":
        val = revenue(s, e)
    elif slot == "revenue_item":
        val = revenue(s, e, item=p["item"])
    elif slot == "revenue_group":
        val = revenue(s, e, group=p["group"])
    elif slot == "revenue_channel":
        val = revenue(s, e, channel=p["channel"])
    elif slot == "revenue_net":
        val = revenue(s, e) - refunds(s, e)
    elif slot == "refund_total":
        val = refunds(s, e)
    elif slot == "n_orders":
        val = len({o["order"] for o in v.orders_in(s, e, "done", p.get("channel"))})
    elif slot == "n_cancelled":
        val = len({o["order"] for o in v.orders_in(s, e, "cancel")})
    elif slot == "cancel_rate":
        allo = v.orders_in(s, e)
        n_cancel = sum(o["status"] == "cancel" for o in allo)
        if not allo or half_tie(n_cancel, len(allo)):
            raise Degenerate("no orders or half tie")
        val = _pct(n_cancel, len(allo))
    elif slot == "avg_order":
        n = len(v.orders_in(s, e, "done"))
        if not n:
            raise Degenerate("no orders")
        val = _money(Fraction(revenue(s, e), n))
    elif slot == "qty_base":
        val = item_qty(s, e)[p["item"]]
    elif slot in ("top_items", "best_item_rev"):
        if slot == "top_items":
            q = item_qty(s, e)
            ranked = sorted((c for c in q if q[c] > 0), key=lambda c: (-q[c], c))
            if len(ranked) < p["k"]:
                raise Degenerate("not enough items")
            val = [v.items[c]["name"] for c in ranked[:p["k"]]]
        else:
            r = {c: revenue(s, e, item=c) for c in v.items}
            ranked = sorted((c for c in r if r[c] > 0), key=lambda c: (-r[c], c))
            if not ranked:
                raise Degenerate("no sales")
            val = v.items[ranked[0]]["name"]
    elif slot == "best_day":
        dd = daily(s, e)
        if not dd:
            raise Degenerate("no sales")
        val = min(dd, key=lambda d: (-dd[d], d)).isoformat()
    elif slot == "best_month":
        mm = {}
        for d, x in daily(s, e).items():
            mm[(d.year, d.month)] = mm.get((d.year, d.month), 0) + x
        if len(mm) < 2:
            raise Degenerate("one month")
        y, m = min(mm, key=lambda k: (-mm[k], k))
        val = f"{y:04d}-{m:02d}"
    elif slot in ("top_customer", "top_customers"):
        sp = cust_spend(s, e)
        ranked = sorted(sp, key=lambda n: (-sp[n], n))
        k = 1 if slot == "top_customer" else p["k"]
        if len(ranked) < k:
            raise Degenerate("not enough customers")
        val = ranked[0] if slot == "top_customer" else ranked[:k]
    elif slot == "n_customers":
        val = len({o["cust"] for o in v.orders_in(s, e, "done") if o["cust"]})
    elif slot == "exists_item_day":
        val = any(ln["item"] == p["item"] for _, ln in v.done_lines(s, e))
    elif slot == "group_table":
        groups = sorted({it["group"] for it in rec["items"]})
        rows = [[g, revenue(s, e, group=g)] for g in groups]
        rows = [r for r in rows if r[1] > 0]
        if len(rows) < 2:
            raise Degenerate("table too small")
        val = {"columns": ["Nhóm hàng", "Doanh thu"], "rows": rows}
    elif slot == "unsold_items":
        q = item_qty(s, e)
        val = sum(1 for c in q if q[c] == 0)
    elif slot == "rev_change_pct":
        bs, be = resolve(spec["base_period"], spec["as_of"])
        old, new = revenue(bs, be), revenue(s, e)
        if old == 0 or new == old or half_tie(new - old, old):
            raise Degenerate("base zero, no change or half tie")
        val = _pct(new - old, old)
    elif slot == "share_group":
        tot = revenue(s, e)
        part = revenue(s, e, group=p["group"])
        if tot == 0 or part == 0 or part == tot or half_tie(part, tot):
            raise Degenerate("share degenerate")
        val = _pct(part, tot)
    elif slot == "days_over":
        dd = daily(s, e)
        val = sum(1 for x in dd.values() if x >= p["threshold"])
    elif slot == "debt_total":
        val = sum(x for x in debts(e).values() if x > 0)
    elif slot == "top_debtor":
        db = {k: x for k, x in debts(e).items() if x > 0}
        if not db:
            raise Degenerate("no debt")
        val = min(db, key=lambda n: (-db[n], n))
    elif slot == "purchase_money":
        val = sum(x["amount"] for x in rec["purchases"] if s <= _d(x["at"]) <= e and _d(x["at"]) <= v.today)
    elif slot == "lots_expiring":
        val = sum(1 for lot in rec["lots"] if lot["qty_left"] > 0 and v.today <= _d(lot["expiry"]) <= v.today + dt.timedelta(days=p["n"]))
    else:
        val = _gold_ext(rec, spec, v, s, e, p, revenue, refunds, item_qty, debts, daily)
    return {"type": typ, "answer": val}


def _shift_of(rec, at):
    h = _ts(at).hour
    return next(x["shift"] for x in rec["shifts"] if x["from"] <= h < x["to"])


def price_on(rec, item, day):
    """Giá một đơn vị gốc có hiệu lực ngày `day` (lần đổi gần nhất có ngày áp dụng ≤ day, chưa đổi thì giá danh mục)."""
    p = next(it["price"] for it in rec["items"] if it["item"] == item)
    for x in sorted(rec.get("price_history", []), key=lambda x: x["from"]):
        if x["item"] == item and _d(x["from"]) <= _d(day):
            p = x["price"]
    return p


def _gold_ext(rec, spec, v, s, e, p, revenue, refunds, item_qty, debts, daily):
    """Slot của shop phức tạp: ca, nhân viên, giá theo ngày, giá vốn/lãi gộp, tồn kho, công nợ chi tiết, nhiều kỳ, đếm có điều kiện, kê đơn."""
    slot = spec["slot"]
    lines_in = v.done_lines(s, e) if s else []
    if slot in ("shift_rev", "shift_share"):
        part = sum(ln["amount"] for o, ln in lines_in if _shift_of(rec, o["at"]) == p["shift"])
        if slot == "shift_rev":
            return part
        tot = revenue(s, e)
        if not tot or not part or part == tot or half_tie(part, tot):
            raise Degenerate("shift share degenerate")
        return _pct(part, tot)
    if slot in ("staff_rev", "staff_best"):
        by = collections.Counter()
        for o, ln in lines_in:
            if o.get("staff"):
                by[o["staff"]] += ln["amount"]
        if slot == "staff_rev":
            return by[p["staff"]]
        if not by:
            raise Degenerate("no staff sales")
        return min(by, key=lambda n: (-by[n], n))
    if slot == "price_on_day":
        return price_on(rec, p["item"], s)
    if slot == "price_change_pct":
        a, b = price_on(rec, p["item"], s), price_on(rec, p["item"], e)
        if a == b or half_tie(b - a, a):
            raise Degenerate("no price change")
        return _pct(b - a, a)
    fac = v.fac
    if slot == "avg_cost":
        rows = [x for x in rec["purchases"] if x["item"] == p["item"] and s <= _d(x["at"]) <= e and _d(x["at"]) <= v.today]
        qty = sum(x["qty"] * fac[(x["item"], x["unit"])] for x in rows)
        if not qty:
            raise Degenerate("no purchases")
        return _money(Fraction(sum(x["amount"] for x in rows), qty))
    if slot == "profit_gross":
        cost = {it["item"]: it["cost"] for it in rec["items"]}
        return revenue(s, e) - sum(v.base_qty(ln) * cost[ln["item"]] for _, ln in lines_in)
    if slot == "stock_qty":
        it = p["item"]
        q = next(x["qty"] for x in rec["opening_stock"] if x["item"] == it)
        q += sum(x["qty"] * fac[(x["item"], x["unit"])] for x in rec["purchases"] if x["item"] == it and _d(x["at"]) <= e)
        q -= sum(v.base_qty(ln) for o, ln in v.done_lines(dt.date(1900, 1, 1), e) if ln["item"] == it)
        return q
    if slot == "debt_collected":
        return sum(x["amount"] for x in rec["payments"] if s <= _d(x["at"]) <= e and _d(x["at"]) <= v.today)
    if slot == "debt_n_cust":
        return sum(1 for x in debts(e).values() if x > 0)
    if slot == "debt_cust":
        x = debts(e).get(p["customer"], 0)
        if x <= 0:
            raise Degenerate("customer owes nothing")
        return x
    if slot == "g_rev_change":
        bs, be = resolve(spec["base_period"], spec["as_of"])
        old, new = revenue(bs, be, group=p["group"]), revenue(s, e, group=p["group"])
        if not old or new == old or half_tie(new - old, old):
            raise Degenerate("group change degenerate")
        return _pct(new - old, old)
    if slot in ("rev_list", "min_month"):
        mm = collections.Counter()
        for d0, x in daily(s, e).items():
            mm[(d0.year, d0.month)] += x
        months = []
        y, m = s.year, s.month
        while dt.date(y, m, 1) <= e:
            months.append((y, m))
            y, m = _month_add(y, m, 1)
        if len(months) < 2:
            raise Degenerate("one month")
        if slot == "rev_list":
            return [mm[k] for k in months]
        y, m = min(months, key=lambda k: (mm[k], k))
        return f"{y:04d}-{m:02d}"
    if slot == "avg_daily_rev":
        return _money(Fraction(revenue(s, e), (e - s).days + 1))
    if slot == "orders_over":
        tot = collections.Counter()
        for o, ln in lines_in:
            tot[o["order"]] += ln["amount"]
        return sum(1 for x in tot.values() if x >= p["threshold"])
    if slot == "items_ge":
        q = item_qty(s, e)
        return sum(1 for c, x in q.items() if x >= p["n"] and ("group" not in p or v.items[c]["group"] == p["group"]))
    if slot == "cust_ge_n":
        cnt = collections.Counter(o["cust"] for o in v.orders_in(s, e, "done") if o["cust"])
        return sum(1 for x in cnt.values() if x >= p["n"])
    if slot in ("rx_rev", "rx_share"):
        rx = {it["item"] for it in rec["items"] if it.get("rx")}
        part = sum(ln["amount"] for _, ln in lines_in if ln["item"] in rx)
        if slot == "rx_rev":
            return part
        tot = revenue(s, e)
        if not tot or not part or half_tie(part, tot):
            raise Degenerate("rx share degenerate")
        return _pct(part, tot)
    raise ValueError(slot)


class Degenerate(ValueError):
    """Đặc tả cho đáp án suy biến (rỗng, chia 0, mốc làm tròn): lấy tham số khác."""


# ---------------------------------------------------------------- danh sách kỹ năng (slot)
# type, độ khó gốc, kỹ năng, cần gì, cách nói số đo cho teacher
SLOTS = {
    "revenue": {"type": "number", "hard": 1, "skill": "doanh_thu_ky", "needs": (), "measure": "tổng doanh thu bán"},
    "revenue_item": {"type": "number", "hard": 2, "skill": "loc_mat_hang", "needs": (), "measure": "doanh thu bán của mặt hàng {item_name}"},
    "revenue_group": {"type": "number", "hard": 2, "skill": "nhom_hang", "needs": (), "measure": "doanh thu bán của nhóm hàng {group}"},
    "revenue_channel": {"type": "number", "hard": 2, "skill": "kenh_ban", "needs": (), "measure": "doanh thu bán qua kênh {channel}"},
    "revenue_net": {"type": "number", "hard": 3, "skill": "tra_hang", "needs": (), "measure": "doanh thu thuần (doanh thu bán trừ tiền hoàn trả của kỳ)"},
    "refund_total": {"type": "number", "hard": 2, "skill": "tra_hang", "needs": (), "measure": "tổng số tiền đã hoàn trả cho khách"},
    "n_orders": {"type": "number", "hard": 1, "skill": "dem_don", "needs": (), "measure": "số đơn bán được"},
    "n_cancelled": {"type": "number", "hard": 2, "skill": "loc_trang_thai", "needs": (), "measure": "số đơn bị hủy"},
    "cancel_rate": {"type": "percent", "hard": 2, "skill": "ty_le_huy", "needs": (), "measure": "tỷ lệ đơn bị hủy (phần trăm)"},
    "avg_order": {"type": "number", "hard": 2, "skill": "gia_tri_trung_binh", "needs": (), "measure": "giá trị trung bình mỗi đơn bán"},
    "qty_base": {"type": "number", "hard": 3, "skill": "quy_doi_don_vi", "needs": ("multi_unit",), "measure": "số {base} {item_name} đã bán (tính theo {base})"},
    "top_items": {"type": "list", "hard": 3, "skill": "xep_hang_mat_hang", "needs": (), "measure": "{k} mặt hàng bán chạy nhất theo số lượng, xếp từ cao xuống"},
    "best_item_rev": {"type": "string", "hard": 2, "skill": "xep_hang_mat_hang", "needs": (), "measure": "mặt hàng mang lại doanh thu bán cao nhất (một mặt hàng)"},
    "best_day": {"type": "date", "hard": 2, "skill": "ngay_cao_nhat", "needs": (), "measure": "ngày có doanh thu bán cao nhất (trả lời một ngày)"},
    "best_month": {"type": "date", "hard": 2, "skill": "thang_cao_nhat", "needs": (), "measure": "tháng có doanh thu bán cao nhất (trả lời tháng nào)"},
    "top_customer": {"type": "string", "hard": 3, "skill": "xep_hang_khach", "needs": (), "measure": "khách hàng mua nhiều tiền nhất (một khách)"},
    "top_customers": {"type": "list", "hard": 3, "skill": "xep_hang_khach", "needs": (), "measure": "{k} khách hàng mua nhiều tiền nhất, xếp từ cao xuống"},
    "n_customers": {"type": "number", "hard": 2, "skill": "dem_khach", "needs": (), "measure": "số khách hàng đã mua hàng"},
    "exists_item_day": {"type": "boolean", "hard": 2, "skill": "kiem_tra_co_khong", "needs": (), "measure": "có bán được {item_name} không (có/không)"},
    "group_table": {"type": "table", "hard": 3, "skill": "bang_theo_nhom", "needs": (), "measure": "liệt kê từng nhóm hàng có bán kèm doanh thu bán của nhóm (bảng hai cột: nhóm hàng, doanh thu)"},
    "unanswerable": {"type": "unanswerable", "hard": 1, "skill": "thieu_du_lieu", "needs": (), "measure": "{missing} (cửa hàng không ghi thông tin này)"},
    "unsold_items": {"type": "number", "hard": 2, "skill": "hang_e", "needs": (), "measure": "số mặt hàng ế (không bán được cái nào)"},
    "rev_change_pct": {"type": "percent", "hard": 3, "skill": "so_sanh_ky", "needs": (), "measure": "doanh thu bán {period} tăng hay giảm bao nhiêu phần trăm so với {base_text}"},
    "share_group": {"type": "percent", "hard": 2, "skill": "ty_trong", "needs": (), "measure": "nhóm hàng {group} chiếm bao nhiêu phần trăm tổng doanh thu bán"},
    "days_over": {"type": "number", "hard": 2, "skill": "dem_ngay", "needs": (), "measure": "số ngày có doanh thu bán từ {threshold_text} đồng trở lên"},
    "debt_total": {"type": "number", "hard": 3, "skill": "cong_no", "needs": ("debt",), "measure": "tổng số tiền khách còn nợ cửa hàng"},
    "top_debtor": {"type": "string", "hard": 3, "skill": "cong_no", "needs": ("debt",), "measure": "khách đang nợ cửa hàng nhiều nhất (một khách)"},
    "purchase_money": {"type": "number", "hard": 2, "skill": "nhap_hang", "needs": ("purchases",), "measure": "tổng tiền nhập hàng"},
    "lots_expiring": {"type": "number", "hard": 3, "skill": "lo_han_dung", "needs": ("lots",), "measure": "số lô hàng còn tồn sẽ hết hạn trong {n} ngày tới (tính từ hôm nay)"},
    # slot của shop phức tạp (bù nhóm nghiệp vụ còn thiếu so với taxonomy gốc)
    "shift_rev": {"type": "number", "hard": 2, "skill": "doanh_thu_ca", "needs": ("staff",), "measure": "doanh thu bán của {shift}"},
    "shift_share": {"type": "percent", "hard": 2, "skill": "doanh_thu_ca", "needs": ("staff",), "measure": "{shift} chiếm bao nhiêu phần trăm doanh thu bán"},
    "staff_rev": {"type": "number", "hard": 2, "skill": "nhan_vien", "needs": ("staff",), "measure": "doanh thu bán của nhân viên {staff}"},
    "staff_best": {"type": "string", "hard": 3, "skill": "nhan_vien", "needs": ("staff",), "measure": "nhân viên bán được nhiều tiền nhất (một người)"},
    "price_on_day": {"type": "number", "hard": 2, "skill": "gia_ban", "needs": ("price_history",), "measure": "giá bán một {base} {item_name}"},
    "price_change_pct": {"type": "percent", "hard": 3, "skill": "gia_ban", "needs": ("price_history",), "measure": "giá bán một {base} {item_name} ngày {d2_text} tăng hay giảm bao nhiêu phần trăm so với ngày {d1_text}"},
    "avg_cost": {"type": "number", "hard": 3, "skill": "gia_nhap", "needs": ("stock",), "measure": "giá nhập bình quân một {base} {item_name}"},
    "profit_gross": {"type": "number", "hard": 3, "skill": "lai_gop", "needs": ("cost",), "measure": "lãi gộp (doanh thu bán trừ giá vốn hàng bán)"},
    "stock_qty": {"type": "number", "hard": 3, "skill": "ton_kho", "needs": ("stock",), "measure": "số {base} {item_name} còn tồn kho"},
    "debt_collected": {"type": "number", "hard": 2, "skill": "cong_no", "needs": ("debt",), "measure": "tổng tiền khách trả nợ"},
    "debt_n_cust": {"type": "number", "hard": 3, "skill": "cong_no", "needs": ("debt",), "measure": "số khách còn nợ cửa hàng"},
    "debt_cust": {"type": "number", "hard": 3, "skill": "cong_no", "needs": ("debt",), "measure": "số tiền khách {customer} còn nợ"},
    "g_rev_change": {"type": "percent", "hard": 3, "skill": "so_sanh_ky_nhom", "needs": (), "measure": "doanh thu bán của nhóm {group} {period} tăng hay giảm bao nhiêu phần trăm so với {base_text}"},
    "rev_list": {"type": "list", "hard": 3, "skill": "nhieu_ky", "needs": (), "measure": "doanh thu bán của từng tháng, liệt kê theo thứ tự tháng (một danh sách số)"},
    "min_month": {"type": "date", "hard": 2, "skill": "nhieu_ky", "needs": (), "measure": "tháng có doanh thu bán thấp nhất (trả lời tháng nào)"},
    "avg_daily_rev": {"type": "number", "hard": 2, "skill": "nhieu_ky", "needs": (), "measure": "doanh thu bán trung bình mỗi ngày"},
    "orders_over": {"type": "number", "hard": 2, "skill": "dem_co_dieu_kien", "needs": (), "measure": "số đơn bán có giá trị từ {threshold_text} đồng trở lên"},
    "items_ge": {"type": "number", "hard": 3, "skill": "dem_co_dieu_kien", "needs": (), "measure": "số mặt hàng {group_text}bán được từ {n} đơn vị gốc trở lên"},
    "cust_ge_n": {"type": "number", "hard": 3, "skill": "dem_co_dieu_kien", "needs": (), "measure": "số khách hàng mua từ {n} đơn trở lên"},
    "rx_rev": {"type": "number", "hard": 2, "skill": "nganh_ke_don", "needs": ("rx",), "measure": "doanh thu bán thuốc kê đơn"},
    "rx_share": {"type": "percent", "hard": 2, "skill": "nganh_ke_don", "needs": ("rx",), "measure": "thuốc kê đơn chiếm bao nhiêu phần trăm doanh thu bán"},
}
COMPLEX_SLOTS = ("shift_rev", "shift_share", "staff_rev", "staff_best", "price_on_day", "price_change_pct", "avg_cost", "profit_gross", "stock_qty",
                 "debt_collected", "debt_n_cust", "debt_cust", "g_rev_change", "rev_list", "min_month", "avg_daily_rev", "orders_over", "items_ge",
                 "cust_ge_n", "rx_rev", "rx_share")


# ---------------------------------------------------------------- đối chứng có chủ đích (mức shop, dùng chung cho mọi mẫu của shop)
MUTATIONS = ("m_amount", "m_cancel", "m_alias", "m_boundary", "m_refund", "m_units", "m_walkin", "m_after", "m_unsold", "m_debt", "m_lots", "m_purchase")


def mutate(rec: dict, kind: str) -> dict:
    """Bản ghi đã sửa có chủ đích. Mỗi kiểu nhắm một lỗi đã biết; đáp án tính lại bằng gold()."""
    d, rng = copy.deepcopy(rec), random.Random(f"{rec['seed']}:{kind}")
    today, end = _d(d["today"]), _d(d["data_end"])
    by_order = {}
    for ln in d["lines"]:
        by_order.setdefault(ln["order"], []).append(ln)

    def new_order(at, status, cust, item, qty, unit=None, walk=None):
        oid = f"DX{len(d['orders']) + 1:05d}"
        it = next(i for i in d["items"] if i["item"] == item)
        d["orders"].append({"order": oid, "at": at.isoformat(sep=" "), "status": status, "status_text": rng.choice(d["status_text"][status]),
                            "channel": d["orders"][0]["channel"], "cust": cust, "walkin_text": walk if cust is None else None,
                            "payment": "Tiền mặt"})
        u = unit or it["base"]
        f = next(x["factor"] for x in d["units"] if x["item"] == item and x["unit"] == u)
        d["lines"].append({"order": oid, "line": 1, "item": item, "unit": u, "qty": qty, "price": it["price"] * f, "amount": qty * it["price"] * f})
        return oid

    if kind == "m_amount":  # đổi giá vài dòng: bắt đáp án ghi cứng
        for ln in d["lines"]:
            if rng.random() < 0.15:
                ln["price"] += 1000
                ln["amount"] = ln["qty"] * ln["price"]
    elif kind == "m_cancel":  # huỷ thêm đơn, ghi trạng thái kiểu khác
        for o in d["orders"]:
            if o["status"] == "done" and rng.random() < 0.08:
                o["status"], o["status_text"] = "cancel", rng.choice(d["status_text"]["cancel"])
        _drop_refunds_of_cancelled(d)
    elif kind == "m_alias":  # bất biến: đảo tần suất các cách ghi trạng thái / khách lẻ của chính shop (cách ghi hiếm thành phổ biến)
        for o in d["orders"]:
            opts = d["status_text"][o["status"]]
            o["status_text"] = opts[-1] if rng.random() < 0.7 else rng.choice(opts)
            if o["cust"] is None:
                o["walkin_text"] = d["walkin_text"][-1] if rng.random() < 0.7 else rng.choice(d["walkin_text"])
    elif kind == "m_boundary":  # đơn đúng 23:59:59 cuối tháng/TODAY và 00:00:00 đầu tháng/ngày sau TODAY
        hot = max(d["items"], key=lambda it: sum(ln["qty"] for ln in d["lines"] if ln["item"] == it["item"]))["item"]
        y, m = _d(d["start"]).year, _d(d["start"]).month
        while dt.date(y, m, 1) <= end:
            me = _month_end(y, m)
            if me <= today:
                new_order(dt.datetime.combine(me, dt.time(23, 59, 59)), "done", d["customers"][2]["cust"], hot, 7)
            if me + dt.timedelta(days=1) <= end:
                new_order(dt.datetime.combine(me + dt.timedelta(days=1), dt.time(0, 0, 0)), "done", None, hot, 5, walk=d["walkin_text"][0])
            y, m = _month_add(y, m, 1)
        new_order(dt.datetime.combine(today, dt.time(23, 59, 59)), "done", d["customers"][3]["cust"], hot, 9)
    elif kind == "m_refund":  # hoàn đầu tháng cho đơn tháng trước, hoàn cuối tháng cho đơn giữa tháng
        done = [o for o in d["orders"] if o["status"] == "done" and o["payment"] != "Ghi nợ"]
        y, m = _d(d["start"]).year, _d(d["start"]).month
        while dt.date(y, m, 1) <= today:
            py, pm = _month_add(y, m, -1)
            prev = [o for o in done if _d(o["at"]).year == py and _d(o["at"]).month == pm]
            if prev:
                o = rng.choice(prev)
                ln = by_order[o["order"]][0]
                d["refunds"].append({"refund": f"TX{len(d['refunds']) + 1:04d}", "order": o["order"],
                                     "at": dt.datetime.combine(dt.date(y, m, 1), dt.time(8, 30)).isoformat(sep=" "),
                                     "item": ln["item"], "unit": ln["unit"], "qty": 1, "amount": ln["price"]})
            y, m = _month_add(y, m, 1)
        d["refunds"].sort(key=lambda r: (r["at"], r["refund"]))
    elif kind == "m_units":  # đổi hệ số quy đổi đơn vị lớn
        for u in d["units"]:
            if u["factor"] > 1:
                u["factor"] += 2 if u["factor"] < 20 else 10
    elif kind == "m_walkin":  # đơn rất lớn của khách lẻ mỗi tháng: lọt top nếu quên bỏ khách lẻ
        top = max(d["items"], key=lambda it: it["price"])["item"]
        y, m = _d(d["start"]).year, _d(d["start"]).month
        while dt.date(y, m, 1) <= today:
            day = dt.date(y, m, min(15, _month_end(y, m).day))
            if day <= today:
                new_order(dt.datetime.combine(day, dt.time(10, 15)), "done", None, top, 40, walk=rng.choice(d["walkin_text"]))
            y, m = _month_add(y, m, 1)
    elif kind == "m_after":  # bất biến: đơn sau TODAY coi như chưa xảy ra
        hot = d["items"][0]["item"]
        for k in range(1, 4):
            new_order(dt.datetime.combine(today + dt.timedelta(days=k), dt.time(9, 0)), "done", d["customers"][0]["cust"], hot, 30)
        new_order(dt.datetime.combine(today + dt.timedelta(days=1), dt.time(0, 0, 1)), "done", None, hot, 30, walk=d["walkin_text"][0])
        d["data_end"] = (today + dt.timedelta(days=3)).isoformat()
    elif kind == "m_unsold":  # bỏ toàn bộ dòng bán của hai mặt hàng
        gone = {it["item"] for it in rng.sample(d["items"], 2)}
        d["lines"] = [ln for ln in d["lines"] if ln["item"] not in gone]
        keep = {ln["order"] for ln in d["lines"]}
        d["orders"] = [o for o in d["orders"] if o["order"] in keep]
        d["refunds"] = [r for r in d["refunds"] if r["order"] in keep and r["item"] not in gone]
    elif kind == "m_debt":  # thêm khoản trả nợ và một đơn ghi nợ mới
        if d["payments"]:
            p0 = d["payments"][len(d["payments"]) // 2]
            d["payments"].append({"payment": f"TX{len(d['payments']) + 1:04d}", "cust": p0["cust"], "at": p0["at"], "amount": 50000})
        cust = d["customers"][4]["cust"]
        oid = new_order(dt.datetime.combine(today - dt.timedelta(days=2), dt.time(11, 0)), "done", cust, d["items"][1]["item"], 20)
        next(o for o in d["orders"] if o["order"] == oid)["payment"] = "Ghi nợ"
        d["payments"].sort(key=lambda r: (r["at"], r["payment"]))
    elif kind == "m_lots":  # dời hạn dùng vài lô vào/ra khỏi cửa sổ
        for lot in d["lots"]:
            if rng.random() < 0.3:
                lot["expiry"] = (today + dt.timedelta(days=rng.choice([0, 3, 9, 29, 31, 61]))).isoformat()
                lot["qty_left"] = rng.choice([0, 12])
    elif kind == "m_purchase":
        y, m = _d(d["start"]).year, _d(d["start"]).month
        while dt.date(y, m, 1) <= today:
            it = d["items"][0]
            d["purchases"].append({"receipt": f"PX{len(d['purchases']) + 1:04d}", "at": dt.datetime.combine(dt.date(y, m, 2), dt.time(7, 0)).isoformat(sep=" "),
                                   "item": it["item"], "unit": it["base"], "qty": 10, "amount": 10 * it["price"]})
            y, m = _month_add(y, m, 1)
    else:
        raise ValueError(kind)
    d["orders"].sort(key=lambda o: (o["at"], o["order"]))
    d["mutation"] = kind
    return d


def _drop_refunds_of_cancelled(d):
    ok = {o["order"] for o in d["orders"] if o["status"] == "done"}
    d["refunds"] = [r for r in d["refunds"] if r["order"] in ok]


# ---------------------------------------------------------------- 6 họ schema (cách xuất file)
def _vnd(x: int) -> str:
    return f"{x:,}".replace(",", ".")


def _dmy(ts: str, with_time=True) -> str:
    t = _ts(ts)
    return t.strftime("%d/%m/%Y %H:%M:%S") if with_time else t.strftime("%d/%m/%Y")


def render(rec: dict, family: str) -> tuple[dict[str, list[dict]], str]:
    """Bản ghi → {tên file: các dòng} + README (mô tả file/cột, cách ghi dấu tiền hoàn). Mọi họ biểu diễn cùng một sự thật."""
    items = {it["item"]: it for it in rec["items"]}
    cust = {c["cust"]: c for c in rec["customers"]}
    feats = rec["features"]
    walk = lambda o: o["walkin_text"]  # noqa: E731
    cname = lambda o: cust[o["cust"]]["name"] if o["cust"] else walk(o)  # noqa: E731
    files: dict[str, list[dict]] = {}
    extra = []
    if family == "joined":
        files["items.csv"] = [{"item": it["item"], "name": it["name"], "group": it["group"], "base_unit": it["base"], "price": it["price"]} for it in rec["items"]]
        files["units.csv"] = [{"item": u["item"], "unit": u["unit"], "factor": u["factor"]} for u in rec["units"]]
        files["customers.csv"] = [{"customer": "KL", "name": "Khách lẻ", "type": "Vãng lai"}] + [
            {"customer": c["cust"], "name": c["name"], "type": c["kind"]} for c in rec["customers"]]
        files["orders.csv"] = [{"order": o["order"], "created_at": o["at"], "status": o["status_text"], "channel": o["channel"],
                                "customer": o["cust"] or "KL", "payment": o["payment"]} for o in rec["orders"]]
        files["order_lines.csv"] = [{"order": ln["order"], "line": ln["line"], "item": ln["item"], "unit": ln["unit"], "qty": ln["qty"],
                                     "price": ln["price"], "amount": ln["amount"]} for ln in rec["lines"]]
        files["refunds.csv"] = [{"refund": r["refund"], "order": r["order"], "refunded_at": r["at"], "item": r["item"], "qty": r["qty"],
                                 "unit": r["unit"], "amount": r["amount"]} for r in rec["refunds"]]
        readme = ["Dữ liệu bán hàng xuất từ phần mềm, nhiều bảng nối với nhau qua mã.",
                  "- items.csv: danh mục hàng (item = mã hàng, name = tên hàng, group = nhóm hàng, base_unit = đơn vị gốc, price = giá bán một đơn vị gốc).",
                  "- units.csv: quy đổi đơn vị; mỗi dòng (item, unit) cho biết 1 unit bằng bao nhiêu đơn vị gốc (factor). Đơn vị gốc có factor = 1.",
                  "- customers.csv: khách hàng; mã KL là khách lẻ (khách vãng lai, không có tên).",
                  "- orders.csv: đơn hàng (order = mã đơn, created_at = thời điểm bán, status = trạng thái ghi tay, channel = kênh bán, customer = mã khách, payment = hình thức thanh toán).",
                  "- order_lines.csv: dòng hàng của đơn (qty tính theo cột unit, price = đơn giá của unit đó, amount = thành tiền = qty × price).",
                  "- refunds.csv: phiếu hoàn tiền khi khách trả hàng (refunded_at = thời điểm hoàn, amount = số tiền hoàn, ghi số dương)."]
        if feats["debt"]:
            files["payments.csv"] = [{"payment": x["payment"], "customer": x["cust"], "paid_at": x["at"], "amount": x["amount"]} for x in rec["payments"]]
            readme.append("- payments.csv: khách trả tiền nợ (customer = mã khách, paid_at = ngày trả, amount = số tiền trả). Đơn có payment 'Ghi nợ' là bán chịu.")
    elif family == "flat":
        files["catalog.csv"] = [{"code": it["item"], "product": it["name"], "category": it["group"], "base_unit": it["base"], "unit_price": it["price"]} for it in rec["items"]]
        files["packs.csv"] = [{"code": u["item"], "pack": u["unit"], "units_per_pack": u["factor"]} for u in rec["units"] if u["factor"] > 1]
        orders = {o["order"]: o for o in rec["orders"]}
        files["sales.csv"] = [{"order_no": ln["order"], "sold_at": orders[ln["order"]]["at"], "status": orders[ln["order"]]["status_text"],
                               "channel": orders[ln["order"]]["channel"], "customer_name": cname(orders[ln["order"]]),
                               "payment": orders[ln["order"]]["payment"], "line_no": ln["line"], "code": ln["item"], "pack": ln["unit"],
                               "qty": ln["qty"], "price": ln["price"], "line_total": ln["amount"]} for ln in rec["lines"]]
        files["refunds.csv"] = [{"refund_no": r["refund"], "order_no": r["order"], "refund_time": r["at"], "refund_amount": r["amount"]} for r in rec["refunds"]]
        readme = ["Một file bán hàng phẳng: mỗi dòng là một dòng hàng, thông tin đơn lặp lại ở mọi dòng của đơn.",
                  "- catalog.csv: danh mục (code = mã hàng, product = tên hàng, category = nhóm hàng, base_unit = đơn vị gốc, unit_price = giá một đơn vị gốc).",
                  "- packs.csv: đơn vị lớn (1 pack = units_per_pack đơn vị gốc). Dòng bán có pack là đơn vị gốc thì hệ số là 1 (không có dòng trong packs.csv).",
                  "- sales.csv: dòng bán (order_no = mã đơn, sold_at = thời điểm bán, status = trạng thái đơn, customer_name = tên khách; khách lẻ ghi 'Khách lẻ', 'kl'... hoặc để trống; qty theo cột pack; line_total = qty × price).",
                  "- refunds.csv: hoàn tiền (refund_time = thời điểm hoàn, refund_amount = số tiền hoàn, số dương)."]
        if feats["debt"]:
            files["debt_payments.csv"] = [{"customer_name": cust[x["cust"]]["name"], "paid_on": x["at"], "amount": x["amount"]} for x in rec["payments"]]
            readme.append("- debt_payments.csv: khách trả nợ (customer_name, paid_on, amount). Đơn có payment 'Ghi nợ' là bán chịu.")
    elif family == "vn_cols":
        files["DanhMucHang.csv"] = [{"Mã hàng": it["item"], "Tên hàng": it["name"], "Nhóm hàng": it["group"], "ĐVT gốc": it["base"], "Giá bán": _vnd(it["price"])} for it in rec["items"]]
        q = []
        for it in rec["items"]:
            ups = sorted((u for u in rec["units"] if u["item"] == it["item"] and u["factor"] > 1), key=lambda u: u["factor"])
            prev = (it["base"], 1)
            for u in ups:  # quy đổi theo bậc: đơn vị lớn = k đơn vị bậc ngay dưới
                if u["factor"] % prev[1] == 0:
                    q.append({"Mã hàng": it["item"], "ĐVT": u["unit"], "Quy đổi": u["factor"] // prev[1], "Theo ĐVT": prev[0]})
                    prev = (u["unit"], u["factor"])
                else:
                    q.append({"Mã hàng": it["item"], "ĐVT": u["unit"], "Quy đổi": u["factor"], "Theo ĐVT": it["base"]})
        files["QuyDoiDVT.csv"] = q
        files["KhachHang.csv"] = [{"Mã KH": c["cust"], "Tên khách hàng": c["name"], "Loại khách": c["kind"]} for c in rec["customers"]]
        files["HoaDon.csv"] = [{"Số HĐ": o["order"], "Ngày bán": _dmy(o["at"]), "Trạng thái": o["status_text"], "Kênh bán": o["channel"],
                                "Khách hàng": cname(o), "Thanh toán": o["payment"]} for o in rec["orders"]]
        files["ChiTietHoaDon.csv"] = [{"Số HĐ": ln["order"], "STT": ln["line"], "Mã hàng": ln["item"], "ĐVT": ln["unit"], "SL": ln["qty"],
                                       "Đơn giá": _vnd(ln["price"]), "Thành tiền": _vnd(ln["amount"])} for ln in rec["lines"]]
        files["PhieuTraHang.csv"] = [{"Số phiếu": r["refund"], "Số HĐ gốc": r["order"], "Ngày trả": _dmy(r["at"]), "Mã hàng": r["item"],
                                      "SL trả": r["qty"], "ĐVT": r["unit"], "Tiền hoàn": _vnd(r["amount"])} for r in rec["refunds"]]
        readme = ["Sổ bán hàng tiếng Việt, ngày ghi dd/mm/yyyy, tiền ghi dấu chấm ngăn nghìn.",
                  "- DanhMucHang.csv: mã hàng, tên hàng, nhóm hàng, đơn vị tính gốc, giá bán một đơn vị gốc.",
                  "- QuyDoiDVT.csv: quy đổi theo bậc: 1 'ĐVT' = 'Quy đổi' × 'Theo ĐVT' (có thể phải nhân nhiều bậc mới về đơn vị gốc). Đơn vị gốc không có dòng.",
                  "- KhachHang.csv: khách hàng có tên. Trong hóa đơn, khách lẻ ghi 'Khách lẻ', 'kl'… hoặc để trống.",
                  "- HoaDon.csv: hóa đơn (Số HĐ, Ngày bán, Trạng thái, Kênh bán, Khách hàng = tên khách, Thanh toán).",
                  "- ChiTietHoaDon.csv: dòng hàng (SL theo cột ĐVT; Thành tiền = SL × Đơn giá).",
                  "- PhieuTraHang.csv: khách trả hàng, cửa hàng hoàn tiền (Ngày trả, Tiền hoàn là số dương)."]
        if feats["debt"]:
            files["ThuNo.csv"] = [{"Ngày thu": _dmy(x["at"]), "Khách hàng": cust[x["cust"]]["name"], "Số tiền": _vnd(x["amount"])} for x in rec["payments"]]
            readme.append("- ThuNo.csv: khách trả nợ. Hóa đơn có Thanh toán 'Ghi nợ' là bán chịu.")
    elif family == "neg_lines":
        files["items.csv"] = [{"sku": it["item"], "item_name": it["name"], "group": it["group"], "base_unit": it["base"], "price": it["price"]} for it in rec["items"]]
        files["unit_conversion.csv"] = [{"sku": u["item"], "unit": u["unit"], "base_qty": u["factor"]} for u in rec["units"]]
        orders = {o["order"]: o for o in rec["orders"]}
        rows = [{"bill": ln["order"], "time": orders[ln["order"]]["at"], "type": "Bán", "status": orders[ln["order"]]["status_text"],
                 "channel": orders[ln["order"]]["channel"], "customer": cname(orders[ln["order"]]), "payment": orders[ln["order"]]["payment"],
                 "ref_bill": "", "sku": ln["item"], "unit": ln["unit"], "qty": ln["qty"], "amount": ln["amount"]} for ln in rec["lines"]]
        for r in rec["refunds"]:
            o = orders[r["order"]]
            rows.append({"bill": r["refund"], "time": r["at"], "type": "Trả", "status": "Trả hàng", "channel": o["channel"], "customer": cname(o),
                         "payment": o["payment"], "ref_bill": r["order"], "sku": r["item"], "unit": r["unit"], "qty": -r["qty"], "amount": -r["amount"]})
        rows.sort(key=lambda x: (x["time"], x["bill"]))
        files["transactions.csv"] = rows
        readme = ["Một sổ giao dịch chung cho bán và trả hàng.",
                  "- items.csv: danh mục (sku = mã hàng, item_name, group, base_unit, price = giá một đơn vị gốc).",
                  "- unit_conversion.csv: 1 unit = base_qty đơn vị gốc (đơn vị gốc có base_qty = 1).",
                  "- transactions.csv: mỗi dòng là một dòng hàng. type = 'Bán' (dòng của đơn bán, bill = mã đơn) hoặc 'Trả' (khách trả hàng, bill = mã phiếu trả, ref_bill = mã đơn gốc).",
                  "  Dòng 'Trả' ghi qty và amount là SỐ ÂM, time là thời điểm hoàn tiền, status ghi 'Trả hàng'. Dòng 'Bán': amount = thành tiền, status = trạng thái đơn."]
        if feats["debt"]:
            files["debt_payments.csv"] = [{"customer": cust[x["cust"]]["name"], "time": x["at"], "amount": x["amount"]} for x in rec["payments"]]
            readme.append("- debt_payments.csv: khách trả nợ. Dòng bán có payment 'Ghi nợ' là bán chịu.")
    elif family == "events":
        files["products.csv"] = [{"sku": it["item"], "product": it["name"], "category": it["group"], "base_unit": it["base"], "price": it["price"]} for it in rec["items"]]
        files["conversion.csv"] = [{"sku": u["item"], "unit": u["unit"], "to_base": u["factor"]} for u in rec["units"]]
        files["headers.csv"] = [{"order": o["order"], "ordered_at": o["at"], "state": o["status_text"], "channel": o["channel"],
                                 "buyer": cname(o), "pay": o["payment"]} for o in rec["orders"]]
        orders = {o["order"]: o for o in rec["orders"]}
        ev = [{"order": ln["order"], "line": ln["line"], "sku": ln["item"], "unit": ln["unit"], "qty": ln["qty"], "amount": ln["amount"],
               "event": "sale", "event_at": orders[ln["order"]]["at"]} for ln in rec["lines"]]
        ev += [{"order": r["order"], "line": 0, "sku": r["item"], "unit": r["unit"], "qty": -r["qty"], "amount": -r["amount"], "event": "refund",
                "event_at": r["at"]} for r in rec["refunds"]]
        ev.sort(key=lambda x: (x["event_at"], x["order"], x["line"]))
        files["events.csv"] = ev
        readme = ["Sổ sự kiện: tiêu đề đơn tách riêng, dòng bán và hoàn tiền chung một bảng events.",
                  "- products.csv: sku, product (tên hàng), category (nhóm), base_unit, price (giá một đơn vị gốc).",
                  "- conversion.csv: 1 unit = to_base đơn vị gốc.",
                  "- headers.csv: một dòng mỗi đơn (order, ordered_at = thời điểm bán, state = trạng thái, channel, buyer = tên khách hoặc khách lẻ, pay).",
                  "- events.csv: event = 'sale' là dòng hàng của đơn (event_at = thời điểm bán, amount = thành tiền);",
                  "  event = 'refund' là hoàn tiền: amount và qty là SỐ ÂM, event_at = thời điểm hoàn (có thể khác tháng với đơn gốc), line = 0."]
        if feats["debt"]:
            files["repayments.csv"] = [{"buyer": cust[x["cust"]]["name"], "paid_at": x["at"], "amount": x["amount"]} for x in rec["payments"]]
            readme.append("- repayments.csv: khách trả nợ. Đơn có pay 'Ghi nợ' là bán chịu.")
    elif family == "monthly":
        files["danh_muc.csv"] = [{"ma_hang": it["item"], "ten_hang": it["name"], "nhom": it["group"], "dvt_goc": it["base"], "gia": it["price"]} for it in rec["items"]]
        files["quy_doi.csv"] = [{"ma_hang": u["item"], "dvt": u["unit"], "he_so": u["factor"]} for u in rec["units"] if u["factor"] > 1]
        orders = {o["order"]: o for o in rec["orders"]}
        months = {}
        for ln in rec["lines"]:
            o = orders[ln["order"]]
            t = _ts(o["at"])
            months.setdefault(f"ban_hang_{t.month:02d}_{t.year}.csv", []).append(
                {"ma_don": o["order"], "ngay_gio": _dmy(o["at"]), "trang_thai": o["status_text"], "kenh": o["channel"], "khach": cname(o),
                 "thanh_toan": o["payment"], "ma_hang": ln["item"], "dvt": ln["unit"], "so_luong": ln["qty"], "don_gia": ln["price"],
                 "thanh_tien": ln["amount"]})
        files.update(months)
        files["tra_hang.csv"] = [{"ma_phieu": r["refund"], "ma_don_goc": r["order"], "ngay_tra": _dmy(r["at"]), "tien_tra": r["amount"]} for r in rec["refunds"]]
        readme = ["Sổ bán hàng tách mỗi tháng một file ban_hang_MM_YYYY.csv (cùng cột), ngày giờ ghi dd/mm/yyyy hh:mm:ss.",
                  "- danh_muc.csv: ma_hang, ten_hang, nhom, dvt_goc (đơn vị gốc), gia (giá một đơn vị gốc).",
                  "- quy_doi.csv: 1 dvt = he_so đơn vị gốc (chỉ ghi đơn vị lớn; đơn vị gốc có hệ số 1).",
                  "- ban_hang_*.csv: mỗi dòng là một dòng hàng (ma_don, ngay_gio, trang_thai, kenh, khach = tên khách hoặc khách lẻ, so_luong theo dvt, thanh_tien = so_luong × don_gia).",
                  "- tra_hang.csv: hoàn tiền khi khách trả hàng (ngay_tra = ngày hoàn, tien_tra là số dương)."]
        if feats["debt"]:
            files["thu_no.csv"] = [{"khach": cust[x["cust"]]["name"], "ngay": _dmy(x["at"]), "so_tien": x["amount"]} for x in rec["payments"]]
            readme.append("- thu_no.csv: khách trả nợ; dòng bán có thanh_toan 'Ghi nợ' là bán chịu.")
    else:
        raise ValueError(family)
    if feats["purchases"]:
        name = {"vn_cols": "NhapHang.csv", "monthly": "nhap_hang.csv"}.get(family, "purchases.csv")
        if family == "vn_cols":
            files[name] = [{"Số phiếu": x["receipt"], "Ngày nhập": _dmy(x["at"]), "Mã hàng": x["item"], "ĐVT": x["unit"], "SL": x["qty"], "Tiền nhập": _vnd(x["amount"])} for x in rec["purchases"]]
        elif family == "monthly":
            files[name] = [{"so_phieu": x["receipt"], "ngay_nhap": _dmy(x["at"]), "ma_hang": x["item"], "dvt": x["unit"], "so_luong": x["qty"], "tien_nhap": x["amount"]} for x in rec["purchases"]]
        else:
            files[name] = [{"receipt": x["receipt"], "received_at": x["at"], "item": x["item"], "unit": x["unit"], "qty": x["qty"], "cost_amount": x["amount"]} for x in rec["purchases"]]
        extra.append(f"- {name}: phiếu nhập hàng từ nhà phân phối (ngày nhập, mã hàng, đơn vị, số lượng, tổng tiền nhập của phiếu dòng đó).")
    if feats["lots"]:
        name = {"vn_cols": "LoHang.csv", "monthly": "lo_hang.csv"}.get(family, "lots.csv")
        if family == "vn_cols":
            files[name] = [{"Số lô": x["lot"], "Mã hàng": x["item"], "Hạn dùng": _dmy(x["expiry"] + " 00:00:00", False), "Tồn còn": x["qty_left"]} for x in rec["lots"]]
        elif family == "monthly":
            files[name] = [{"so_lo": x["lot"], "ma_hang": x["item"], "han_dung": _dmy(x["expiry"] + " 00:00:00", False), "ton_con": x["qty_left"]} for x in rec["lots"]]
        else:
            files[name] = [{"lot": x["lot"], "item": x["item"], "expiry": x["expiry"], "qty_left": x["qty_left"]} for x in rec["lots"]]
        extra.append(f"- {name}: lô hàng (số lô, mã hàng, hạn dùng, tồn còn theo đơn vị gốc tại ngày {_d(rec['data_end']).day}/{_d(rec['data_end']).month}/{_d(rec['data_end']).year}).")
    _augment(rec, family, files, extra)
    end = _d(rec["data_end"])
    readme = [f"# {rec['shop_name']} ({INDUSTRIES[rec['industry']]['label']}) — dữ liệu giả lập tự tạo", ""] + readme + extra + [
        "", f"Dữ liệu từ {_d(rec['start']).day}/{_d(rec['start']).month}/{_d(rec['start']).year} đến hết ngày {end.day}/{end.month}/{end.year}. Quy ước tính: xem rulebook của cửa hàng."]
    return files, "\n".join(readme) + "\n"


FAMILY_KEYS = {  # bảng danh mục, cột mã hàng, bảng/cột đơn hàng (để thêm cột nhân viên) theo họ schema
    "joined": ("items.csv", "item", [("orders.csv", "order")], "price_history.csv", "opening_stock.csv"),
    "flat": ("catalog.csv", "code", [("sales.csv", "order_no")], "price_history.csv", "opening_stock.csv"),
    "vn_cols": ("DanhMucHang.csv", "Mã hàng", [("HoaDon.csv", "Số HĐ")], "LichSuGiaBan.csv", "TonDauKy.csv"),
    "neg_lines": ("items.csv", "sku", [("transactions.csv", "bill")], "price_history.csv", "opening_stock.csv"),
    "events": ("products.csv", "sku", [("headers.csv", "order")], "price_history.csv", "opening_stock.csv"),
    "monthly": ("danh_muc.csv", "ma_hang", [], "lich_su_gia.csv", "ton_dau_ky.csv")}


def _augment(rec: dict, family: str, files: dict, extra: list):
    """Cột/bảng của shop phức tạp: giá vốn, kê đơn, nhân viên bán, lịch sử giá, tồn đầu kỳ (chỉ khi bản ghi có tính năng)."""
    f = rec["features"]
    cat, key, order_tabs, ph_name, st_name = FAMILY_KEYS[family]
    vn = family == "vn_cols"
    mo = family == "monthly"
    items = {it["item"]: it for it in rec["items"]}
    code_col = {"joined": "item", "flat": "code", "vn_cols": "Mã hàng", "neg_lines": "sku", "events": "sku", "monthly": "ma_hang"}[family]
    if f.get("cost") or f.get("rx"):
        for row in files[cat]:
            it = items[row[code_col]]
            if f.get("cost"):
                row["Giá vốn" if vn else "gia_von" if mo else "unit_cost" if family == "flat" else "cost"] = _vnd(it["cost"]) if vn else it["cost"]
            if f.get("rx"):
                row["Kê đơn" if vn else "ke_don" if mo else "rx"] = "Có" if it.get("rx") else "Không"
        extra.append(f"- {cat}: " + ("cột giá vốn = giá vốn một đơn vị gốc (cố định trong kỳ dữ liệu); " if f.get("cost") else "")
                     + ("cột kê đơn = 'Có' với thuốc phải có đơn của bác sĩ." if f.get("rx") else ""))
    if f.get("staff"):
        orders = {o["order"]: o for o in rec["orders"]}
        refs = {r["refund"]: r["order"] for r in rec["refunds"]}
        col = "Nhân viên" if vn else "nhan_vien" if mo else "staff"
        tabs = order_tabs + ([(k, "ma_don") for k in files if k.startswith("ban_hang_")] if mo else [])
        for tab, oc in tabs:
            for row in files[tab]:
                oid = refs.get(row[oc], row[oc])
                row[col] = orders[oid].get("staff") or ""
        extra.append(f"- cột {col}: nhân viên lập đơn bán (ô trống nếu không ghi).")
    if f.get("price_history"):
        if vn:
            files[ph_name] = [{"Mã hàng": x["item"], "Áp dụng từ ngày": _dmy(x["from"] + " 00:00:00", False), "Giá bán": _vnd(x["price"])} for x in rec["price_history"]]
        elif mo:
            files[ph_name] = [{"ma_hang": x["item"], "ap_dung_tu": _dmy(x["from"] + " 00:00:00", False), "gia_moi": x["price"]} for x in rec["price_history"]]
        else:
            files[ph_name] = [{key: x["item"], "effective_from": x["from"], "new_price": x["price"]} for x in rec["price_history"]]
        extra.append(f"- {ph_name}: lần đổi giá bán (giá một đơn vị gốc mới, áp dụng từ đầu ngày ghi). Giá trong {cat} là giá trước lần đổi đầu tiên.")
    if f.get("stock"):
        d0 = _d(rec["start"])
        if vn:
            files[st_name] = [{"Mã hàng": x["item"], "Tồn đầu kỳ (ĐVT gốc)": x["qty"]} for x in rec["opening_stock"]]
        elif mo:
            files[st_name] = [{"ma_hang": x["item"], "ton_dau_ky": x["qty"]} for x in rec["opening_stock"]]
        else:
            files[st_name] = [{key: x["item"], "opening_qty": x["qty"]} for x in rec["opening_stock"]]
        extra.append(f"- {st_name}: tồn kho đầu kỳ theo đơn vị gốc, đếm lúc mở sổ ngày {d0.day}/{d0.month}/{d0.year} (trước mọi giao dịch).")


def rules_text(rec: dict, family: str) -> str:
    """Quy ước tính của cửa hàng (tự soạn), dùng làm rulebook của shop. Không chép rulebook BTC."""
    st, walk = rec["status_text"], [w or "(để trống)" for w in rec["walkin_text"]]
    q = lambda xs: ", ".join(f"'{x}'" for x in xs)  # noqa: E731
    sign = {"events": "Trong events, dòng refund đã ghi số âm: cộng thẳng các dòng refund của kỳ vào doanh thu bán là ra doanh thu thuần.",
            "neg_lines": "Dòng 'Trả' đã ghi số âm: doanh thu thuần = tổng amount dòng 'Bán' của đơn hoàn thành trong kỳ + tổng amount dòng 'Trả' có time trong kỳ."}.get(
        family, "Tiền hoàn ghi số dương ở bảng hoàn tiền: doanh thu thuần = doanh thu bán − tổng tiền hoàn của kỳ.")
    feats = rec["features"]
    out = [f"QUY ƯỚC TÍNH CỦA CỬA HÀNG {rec['shop_name'].upper()} (dữ liệu giả lập)",
           f"1. Trạng thái đơn ghi tay nhiều kiểu. Hoàn thành: {q(st['done'])}. Hủy: {q(st['cancel'])}. "
           f"Chưa hoàn thành (không tính bán): {q(st['pending'])}." + (" Phiếu trả hàng ghi 'Trả hàng' không phải đơn bán." if family == "neg_lines" else ""),
           "2. Doanh thu bán của một kỳ = tổng thành tiền mọi dòng hàng của các đơn hoàn thành có thời điểm bán trong kỳ. Doanh thu bán chưa trừ tiền hoàn.",
           "3. Tiền hoàn trả thuộc kỳ theo THỜI ĐIỂM HOÀN, kể cả khi đơn gốc bán ở kỳ khác. Doanh thu thuần = doanh thu bán của kỳ − tiền hoàn của kỳ. " + sign,
           "4. Số lượng bán tính theo đơn vị gốc: số lượng của dòng × hệ số quy đổi của đúng mặt hàng và đơn vị đó. Không trừ hàng trả.",
           "5. 'Số đơn' hay 'số đơn bán' là số mã đơn hoàn thành (mỗi mã tính một lần). Số đơn hủy đếm mã đơn nhóm hủy. "
           "Tỷ lệ hủy = số đơn hủy / số mọi đơn đặt trong kỳ (mọi trạng thái) × 100.",
           "6. Giá trị trung bình mỗi đơn = doanh thu bán / số đơn hoàn thành của kỳ.",
           f"7. Khách lẻ (ghi {q(walk)} hoặc mã KL) không phải một khách hàng: không xếp hạng, không đếm khi hỏi về khách hàng. "
           "Xếp hạng khách theo tổng tiền mua của đơn hoàn thành, bằng nhau thì tên theo thứ tự chữ cái.",
           "8. Xếp hạng mặt hàng theo số lượng gốc (hoặc theo doanh thu bán khi câu hỏi nói doanh thu), giảm dần; bằng nhau thì mã hàng nhỏ hơn đứng trước. "
           "Ngày hoặc tháng có doanh thu cao nhất: bằng nhau lấy ngày/tháng sớm hơn. Mặt hàng ế của kỳ là mặt hàng trong danh mục không bán được đơn vị nào trong kỳ.",
           "9. Kỳ thời gian tính trọn ngày cuối (đến 23:59:59). Hôm nay là TODAY của câu hỏi; bản ghi sau hết ngày TODAY coi như chưa xảy ra. "
           "'Tháng này' = ngày 1 đến hết hôm nay; 'tháng trước' = cả tháng liền trước; 'tuần này' = thứ Hai đến hôm nay; 'tuần trước' = thứ Hai đến Chủ nhật của tuần liền trước; "
           "'N ngày qua' = N ngày tính cả hôm nay; 'năm nay' = 1/1 đến hôm nay; quý theo năm dương lịch. "
           "So sánh 'tháng này' với tháng trước là so với cùng số ngày đầu tháng trước.",
           "10. Tiền làm tròn đến đồng (nửa đồng làm tròn lên); phần trăm làm tròn 2 chữ số thập phân; phần trăm tăng giảm = (kỳ này − kỳ so) / kỳ so × 100, giảm thì âm."]
    if feats["debt"]:
        out.append("Đơn hoàn thành có thanh toán 'Ghi nợ' là bán chịu. Nợ còn lại của khách tính đến một ngày = tổng tiền các đơn bán chịu đến hết ngày đó − tổng tiền khách đã trả nợ đến hết ngày đó.")
    if feats["purchases"]:
        out.append("Tiền nhập hàng của kỳ = tổng tiền các dòng phiếu nhập có ngày nhập trong kỳ.")
    if feats["lots"]:
        out.append("Lô còn tồn là lô có tồn còn > 0. Lô hết hạn trong N ngày tới: hạn dùng từ hôm nay đến hết ngày TODAY + N.")
    if feats.get("staff"):
        sh = "; ".join(f"{x['shift']} từ {x['from']}:00 đến trước {x['to']}:00" if x["to"] < 24 else f"{x['shift']} từ {x['from']}:00 đến hết ngày" for x in rec["shifts"])
        out.append(f"Ca bán theo giờ lập đơn: {sh}. Doanh thu, số đơn của một ca hay một nhân viên tính như doanh thu bán, số đơn bán (đơn hoàn thành).")
    if feats.get("cost"):
        out.append("Giá vốn hàng bán của kỳ = tổng (số lượng gốc bán ra × giá vốn một đơn vị gốc) của đơn hoàn thành trong kỳ. Lãi gộp = doanh thu bán − giá vốn hàng bán (không trừ hoàn tiền).")
    if feats.get("price_history"):
        out.append("Giá bán của một mặt hàng tại một ngày là giá của lần đổi giá gần nhất có ngày áp dụng không sau ngày đó; chưa đổi lần nào thì là giá trong danh mục. "
                   "Phần trăm đổi giá = (giá sau − giá trước) / giá trước × 100.")
    if feats.get("stock"):
        out.append("Tồn kho của mặt hàng đến hết một ngày (đơn vị gốc) = tồn đầu kỳ + số lượng gốc nhập đến hết ngày đó − số lượng gốc bán ra của đơn hoàn thành đến hết ngày đó (hàng khách trả không nhập lại kho). "
                   "Giá nhập bình quân của kỳ = tổng tiền nhập / tổng số lượng gốc nhập trong kỳ.")
    if feats.get("rx"):
        out.append("Thuốc kê đơn là mặt hàng có cột kê đơn = 'Có'; doanh thu thuốc kê đơn tính như doanh thu bán của các mặt hàng đó.")
    if feats.get("staff"):  # quy ước cho slot đếm có điều kiện / nhiều kỳ (chỉ shop phức tạp; shop cũ giữ nguyên rulebook)
        out.append("Đơn có giá trị từ một mức trở lên: so tổng thành tiền của cả đơn hoàn thành. Doanh thu trung bình mỗi ngày của kỳ = doanh thu bán / số ngày của kỳ (tính cả ngày không bán).")
    missing = ", lợi nhuận ròng, chi phí vận hành" if feats.get("cost") else ", lợi nhuận, giá vốn, chi phí vận hành"
    out.append(f"Cửa hàng không ghi các thông tin: {', '.join(INDUSTRIES[rec['industry']]['missing'])}{missing}. Câu hỏi về các số này không trả lời được từ dữ liệu.")
    head, rest = out[0], [re.sub(r"^\d+\.\s*", "", x) for x in out[1:]]
    return "\n".join([head] + [f"{i}. {x}" for i, x in enumerate(rest, 1)]) + "\n"


def shop_brief(rec: dict) -> dict:
    """Thông tin cho teacher viết câu: tên hàng, nhóm, kênh, khách — không có mã nội bộ."""
    return {"items": [{"ten": it["name"], "nhom": it["group"], "don_vi_goc": it["base"],
                       "don_vi_lon": [u["unit"] for u in rec["units"] if u["item"] == it["item"] and u["factor"] > 1]} for it in rec["items"]],
            "nhom_hang": sorted({it["group"] for it in rec["items"]}), "kenh_ban": sorted({o["channel"] for o in rec["orders"]})}


# ---------------------------------------------------------------- dựng fixture qua loader thật
def write_fixture(root, name: str, rec: dict, family: str) -> dict:
    """Ghi data/<name>/ (CSV + README), rulebook.md, calendar.csv; nạp bằng datamate.loader (pandas của môi trường chạy);
    trả {root, tables_pkl, biz}. Không đọc gì ngoài bản ghi tự tạo."""
    import csv
    import pickle
    from pathlib import Path
    from datamate import loader
    root = Path(root)
    shop = root / "data" / name
    shop.mkdir(parents=True, exist_ok=True)
    files, readme = render(rec, family)
    for fn, rows in files.items():
        with open(shop / fn, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
            w.writeheader()
            w.writerows(rows)
    (shop / "README.md").write_text(readme, encoding="utf-8")
    (root / "rulebook.md").write_text(rules_text(rec, family), encoding="utf-8")
    (root / "calendar.csv").write_text("date,lunar_year,lunar_month,lunar_day,lunar_leap\n" + f"{rec['today']},2025,1,1,false\n", encoding="utf-8")
    biz = loader.load_cached(shop, root / "cache")
    pkl = root / "tables.pkl"
    with pkl.open("wb") as fh:
        pickle.dump(biz.tables, fh)
    return {"root": str(root), "name": name, "tables_pkl": str(pkl), "biz": biz, "files": {k: len(v) for k, v in files.items()}}


def integrity(rec: dict, family: str, tables: dict) -> list[str]:
    """Đọc lại sau loader phải giữ đủ dòng và tổng tiền dòng bán, tiền hoàn (cổng G3 rút gọn). Trả danh sách lỗi."""
    import pandas as pd
    errs = []
    sale_total = sum(ln["amount"] for ln in rec["lines"])
    ref_total = sum(r["amount"] for r in rec["refunds"])
    col = {"joined": ("order_lines", "amount", "refunds", "amount"), "flat": ("sales", "line_total", "refunds", "refund_amount"),
           "vn_cols": ("ChiTietHoaDon", "Thành tiền", "PhieuTraHang", "Tiền hoàn"), "neg_lines": ("transactions", "amount", None, None),
           "events": ("events", "amount", None, None), "monthly": (None, "thanh_tien", "tra_hang", "tien_tra")}[family]
    if family == "monthly":
        t = next((k for k in tables if k.startswith("ban_hang")), None)
        sale = pd.to_numeric(tables[t]["thanh_tien"]).sum() if t else -1
    elif family in ("neg_lines", "events"):
        df = tables[col[0]]
        sale = pd.to_numeric(df[col[1]]).clip(lower=0).sum()
        ref = -pd.to_numeric(df[col[1]]).clip(upper=0).sum()
        if int(ref) != ref_total:
            errs.append(f"refund total {ref} != {ref_total}")
    else:
        sale = pd.to_numeric(tables[col[0]][col[1]]).sum()
    if family not in ("neg_lines", "events"):
        ref = pd.to_numeric(tables[col[2]][col[3]]).sum() if rec["refunds"] else 0
        if int(ref) != ref_total:
            errs.append(f"refund total {ref} != {ref_total}")
    if int(sale) != sale_total:
        errs.append(f"sale total {sale} != {sale_total}")
    return errs


# ---------------------------------------------------------------- đối chứng nhắm đúng phạm vi một đặc tả
def _target_ext(d, rec, spec, v, s, e, day, add, big, named, items, fac, early):
    """Phép sửa nhắm đặc tả cho slot shop phức tạp (gọi từ target_mutation)."""
    slot, p = spec["slot"], spec.get("params", {})
    top_item = max(d["items"], key=lambda x: x["price"])["item"]
    if slot in ("shift_rev", "shift_share"):
        x = next(x for x in d["shifts"] if x["shift"] == p["shift"])
        add(day, qty=3, hour=(x["from"] + x["to"] - 1) // 2)
    elif slot == "staff_rev":
        add(day, qty=3, staff=p["staff"])
    elif slot == "staff_best":
        by = collections.Counter()
        for o, ln in v.done_lines(s, min(e, v.today)):
            if o.get("staff"):
                by[o["staff"]] += ln["amount"]
        best = min(by, key=lambda n: (-by[n], n)) if by else None
        other = next(n for n in d["staff"] if n != best)
        add(day, item=top_item, qty=big(max(by.values() or [0]) * 1.5 + 1, top_item), staff=other)
    elif slot in ("price_on_day", "price_change_pct"):
        at = s if slot == "price_on_day" else e
        cur = price_on(rec, p["item"], at)
        same = [x for x in d["price_history"] if x["item"] == p["item"] and x["from"] == at.isoformat()]
        if same:
            same[0]["price"] = int(round(cur * 1.2, -2))
        else:
            d["price_history"].append({"item": p["item"], "from": at.isoformat(), "price": int(round(cur * 1.2, -2))})
        d["price_history"].sort(key=lambda x: (x["item"], x["from"]))
    elif slot == "avg_cost":
        it = items[p["item"]]
        u = max((x for x in d["units"] if x["item"] == p["item"]), key=lambda x: x["factor"])
        d["purchases"].append({"receipt": "PT0009", "at": dt.datetime.combine(day, dt.time(7, 30)).isoformat(sep=" "), "item": p["item"],
                               "unit": u["unit"], "qty": 3, "amount": int(round(3 * u["factor"] * it["price"] * 1.4, -2))})
    elif slot in ("profit_gross", "avg_daily_rev", "rev_list"):
        add(day, item=top_item, qty=2)
    elif slot == "min_month":
        mm = collections.Counter()
        for o, ln in v.done_lines(s, e):
            mm[(_d(o["at"]).year, _d(o["at"]).month)] += ln["amount"]
        y, m = min(mm, key=lambda k: (mm[k], k))
        add(dt.date(y, m, 10), item=top_item, qty=big(max(mm.values()) * 1.3, top_item))
    elif slot == "stock_qty":
        add(min(e, v.today), item=p["item"], qty=2)
    elif slot == "debt_collected":
        c = d["payments"][0]["cust"] if d["payments"] else named[0]
        d["payments"].append({"payment": "TT0009", "cust": c, "at": dt.datetime.combine(day, dt.time(9, 0)).isoformat(sep=" "), "amount": 77000})
    elif slot in ("debt_n_cust", "debt_cust"):
        owe = collections.Counter()
        for o in v.orders:
            if o["status"] == "done" and o["payment"] == "Ghi nợ" and _d(o["at"]) <= e:
                owe[o["cust"]] += sum(ln["amount"] for ln in v.lines[o["order"]])
        for x in rec["payments"]:
            if _d(x["at"]) <= e:
                owe[x["cust"]] -= x["amount"]
        if slot == "debt_cust":
            c = next(c for c in named if v.cust[c]["name"] == p["customer"])
        else:
            c = next((c for c in named if owe[c] <= 0), None)
            if c is None:  # mọi khách đều đang nợ: một khách trả hết nợ đúng ngày cuối → bớt một khách
                c = min((c for c in named if owe[c] > 0), key=lambda c: (owe[c], c))
                d["payments"].append({"payment": "TT0008", "cust": c, "at": dt.datetime.combine(min(e, v.today), dt.time(10, 0)).isoformat(sep=" "),
                                      "amount": owe[c]})
                return
        add(early, cust=c, qty=2, payment="Ghi nợ")  # trước ngày đầu dữ liệu: bắt mốc đầu cố định
    elif slot == "g_rev_change":
        it = max((x for x in d["items"] if x["group"] == p["group"]), key=lambda x: x["price"])["item"]
        add(day, item=it, qty=4)
    elif slot == "orders_over":
        add(day, item=top_item, qty=big(p["threshold"] * 1.1, top_item))
    elif slot == "items_ge":
        q = collections.Counter()
        for o, ln in v.done_lines(s, min(e, v.today)):
            q[ln["item"]] += v.base_qty(ln)
        cand = [c for c in items if q[c] < p["n"] and ("group" not in p or items[c]["group"] == p["group"])]
        if cand:
            add(day, item=cand[0], qty=p["n"] - q[cand[0]])
        else:  # mọi mặt hàng đã đạt mức: huỷ mọi đơn trong kỳ có một mặt hàng → mặt hàng đó rớt khỏi danh sách
            it = min((c for c in items if "group" not in p or items[c]["group"] == p["group"]), key=lambda c: (q[c], c))
            for o in d["orders"]:
                if o["status"] == "done" and s <= _d(o["at"]) <= min(e, v.today) and any(ln["item"] == it for ln in v.lines.get(o["order"], [])):
                    o["status"], o["status_text"] = "cancel", d["status_text"]["cancel"][0]
    elif slot == "cust_ge_n":
        cnt = collections.Counter(o["cust"] for o in v.orders_in(s, min(e, v.today), "done") if o["cust"])
        c = next((c for c in named if cnt[c] == p["n"] - 1), None) or next((c for c in named if cnt[c] < p["n"]), None)
        if c:
            for _ in range(p["n"] - cnt[c]):
                add(day, cust=c)
        else:  # mọi khách đã đạt: huỷ các đơn trong kỳ của một khách → bớt một khách
            c = min(named, key=lambda c: (cnt[c], c))
            for o in d["orders"]:
                if o["status"] == "done" and o["cust"] == c and s <= _d(o["at"]) <= min(e, v.today):
                    o["status"], o["status_text"] = "cancel", d["status_text"]["cancel"][0]
    elif slot in ("rx_rev", "rx_share"):
        rx = next(x["item"] for x in d["items"] if x.get("rx"))
        add(day, item=rx, qty=5)


def target_mutation(rec: dict, spec: dict):
    """Sửa dữ liệu ngay trong kỳ/đối tượng của đặc tả để đáp án chắc chắn đổi (bắt code đúng do may: đáp án 0, có/không, xếp hạng).
    Trả (bản ghi mới, mô tả) hoặc None khi slot không có phép sửa phù hợp (câu không trả lời được)."""
    slot, p = spec["slot"], spec.get("params", {})
    if slot == "unanswerable":
        return None
    base = gold(rec, spec)
    today = _d(spec["as_of"])
    s, e = resolve(spec["period"], spec["as_of"]) if spec.get("period") else (today, today)
    day = min(e, today)
    d = copy.deepcopy(rec)
    d["mutation"] = "m_target"
    items = {it["item"]: it for it in d["items"]}
    fac = {(u["item"], u["unit"]): u["factor"] for u in d["units"]}
    v = View(rec, spec["as_of"])
    k = [0]

    def add(at_day, status="done", cust=None, item=None, qty=1, unit=None, channel=None, payment="Tiền mặt", hour=12, staff=None):
        k[0] += 1
        oid = f"DT{k[0]:04d}{len(d['orders']):05d}"
        item = item or max(d["items"], key=lambda it: it["price"])["item"]
        u = unit or items[item]["base"]
        price = items[item]["price"] * fac[(item, u)]
        d["orders"].append({"order": oid, "at": dt.datetime.combine(at_day, dt.time(hour, 7, 0)).isoformat(sep=" "), "status": status,
                            "status_text": d["status_text"][status][0], "channel": channel or d["orders"][0]["channel"], "cust": cust,
                            "walkin_text": d["walkin_text"][0] if cust is None else None, "payment": payment})
        if d.get("staff"):
            d["orders"][-1]["staff"] = staff or d["staff"][0]
        d["lines"].append({"order": oid, "line": 1, "item": item, "unit": u, "qty": qty, "price": price, "amount": qty * price})
        return oid

    def done_in(lo, hi):
        return [o for o in v.orders_in(lo, hi, "done")]
    named = [c["cust"] for c in d["customers"]]
    early = min(_d(o["at"]) for o in rec["orders"]) - dt.timedelta(days=23)
    big = lambda target_amount, item: max(1, int(target_amount // (items[item]["price"])) + 1)  # noqa: E731
    desc = slot
    if slot in ("revenue", "revenue_net", "avg_order", "rev_change_pct", "days_over"):
        it = max(d["items"], key=lambda x: x["price"])["item"]
        if slot == "days_over":
            daily = collections.Counter()
            for o, ln in v.done_lines(s, min(e, today)):
                daily[_d(o["at"])] += ln["amount"]
            under = [x for x in (s + dt.timedelta(days=i) for i in range((min(e, today) - s).days + 1)) if daily[x] < p["threshold"]]
            if not under:
                return None
            day = under[0]
            add(day, item=it, qty=big(p["threshold"] - daily[day], it))
        else:
            add(day, item=it, qty=3)
    elif slot in ("revenue_item", "qty_base", "exists_item_day"):
        it = p["item"]
        if slot == "exists_item_day" and base["answer"]:
            for o in d["orders"]:  # huỷ mọi đơn hoàn thành trong ngày có mặt hàng này
                if o["status"] == "done" and s <= _d(o["at"]) <= e and any(ln["item"] == it for ln in v.lines.get(o["order"], [])):
                    o["status"], o["status_text"] = "cancel", d["status_text"]["cancel"][0]
            desc = "huy_don_co_mat_hang"
        else:
            big_u = max((u for u in d["units"] if u["item"] == it), key=lambda u: u["factor"])["unit"]
            add(day, item=it, qty=2, unit=big_u if slot == "qty_base" else None)
    elif slot in ("revenue_group", "share_group", "group_table"):
        grp = p.get("group") or sorted({x["group"] for x in d["items"]})[0]
        it = max((x for x in d["items"] if x["group"] == grp), key=lambda x: x["price"])["item"]
        add(day, item=it, qty=5)
    elif slot == "revenue_channel":
        add(day, channel=p["channel"], qty=2)
    elif slot in ("n_orders",):
        add(day, channel=p.get("channel"))
    elif slot in ("n_cancelled", "cancel_rate"):
        add(day, status="cancel")
        if slot == "cancel_rate":
            add(day, status="cancel")
    elif slot == "refund_total":
        olds = [o for o in rec["orders"] if o["status"] == "done" and _d(o["at"]) < s and o["payment"] != "Ghi nợ"] or \
               [o for o in rec["orders"] if o["status"] == "done" and _d(o["at"]) <= day and o["payment"] != "Ghi nợ"]
        o = olds[-1]
        ln = v.lines[o["order"]][0]
        d["refunds"].append({"refund": "TT0001", "order": o["order"], "at": dt.datetime.combine(day, dt.time(13, 0)).isoformat(sep=" "),
                             "item": ln["item"], "unit": ln["unit"], "qty": 1, "amount": ln["price"]})
    elif slot in ("best_day", "best_month", "best_item_rev", "top_items"):
        q_or_rev = collections.Counter()
        for o, ln in v.done_lines(s, min(e, today)):
            key = _d(o["at"]) if slot == "best_day" else (_d(o["at"]).year, _d(o["at"]).month) if slot == "best_month" else ln["item"]
            q_or_rev[key] += v.base_qty(ln) if slot == "top_items" else ln["amount"]
        if slot in ("best_day", "best_month"):
            if slot == "best_day":
                cand = [x for x in (s + dt.timedelta(days=i) for i in range((min(e, today) - s).days + 1)) if x != _d(base["answer"])]
                at = cand[0]
            else:
                by, bm = map(int, base["answer"].split("-"))
                ms = sorted({(x[0], x[1]) for x in q_or_rev} - {(by, bm)})
                if not ms:
                    return None
                at = max(s, dt.date(ms[0][0], ms[0][1], 1))
            top = max(q_or_rev.values())
            it = max(d["items"], key=lambda x: x["price"])["item"]
            add(at, item=it, qty=big(top * 1.2, it))
        else:
            ranked = sorted(q_or_rev, key=lambda c: (-q_or_rev[c], c))
            others = [c for c in items if c not in ranked[:max(1, p.get("k", 1))]]
            if not others or not ranked:
                return None
            cand = others[0]
            need = q_or_rev[ranked[0]] * 1.3
            if slot == "top_items":
                add(day, item=cand, qty=int(need) + 1)
            else:
                add(day, item=cand, qty=big(need, cand))
    elif slot in ("top_customer", "top_customers", "n_customers"):
        spend = collections.Counter()
        for o, ln in v.done_lines(s, min(e, today)):
            if o["cust"]:
                spend[o["cust"]] += ln["amount"]
        if slot == "n_customers":
            cand = [c for c in named if c not in spend]
            if not cand:
                return None
            add(day, cust=cand[0])
        else:
            kk = 1 if slot == "top_customer" else p["k"]
            ranked = sorted(spend, key=lambda c: (-spend[c], v.cust[c]["name"]))
            cand = [c for c in named if c not in ranked[:kk]]
            it = max(d["items"], key=lambda x: x["price"])["item"]
            add(day, cust=cand[0], item=it, qty=big(max(spend.values() or [0]) * 1.5 + 1, it))
    elif slot in ("debt_total", "top_debtor"):
        # đơn ghi nợ đặt TRƯỚC ngày đầu dữ liệu: bắt code lọc công nợ từ một ngày cố định (mốc đầu tự đặt)
        cust = named[-1] if slot == "debt_total" else sorted(c for c in named if v.cust[c]["name"] != base.get("answer"))[-1]
        it = max(d["items"], key=lambda x: x["price"])["item"]
        add(early, cust=cust, item=it, qty=big(5e7, it) if slot == "top_debtor" else 2, payment="Ghi nợ")
        desc = slot + "_no_truoc_du_lieu"
    elif slot == "purchase_money":
        it = d["items"][0]
        d["purchases"].append({"receipt": "PT0001", "at": dt.datetime.combine(day, dt.time(8, 0)).isoformat(sep=" "), "item": it["item"],
                               "unit": it["base"], "qty": 7, "amount": 7 * it["price"]})
    elif slot == "lots_expiring":
        d["lots"].append({"lot": "LT0001", "item": d["items"][0]["item"], "expiry": (today + dt.timedelta(days=1)).isoformat(), "qty_left": 5})
    elif slot == "unsold_items":
        q = collections.Counter()
        for o, ln in v.done_lines(s, min(e, today)):
            q[ln["item"]] += v.base_qty(ln)
        unsold = [c for c in items if q[c] == 0]
        if unsold:
            add(day, item=unsold[0])
        else:  # huỷ mọi đơn có một mặt hàng → thêm một mặt hàng ế
            it = min(q, key=lambda c: (q[c], c))
            for o in d["orders"]:
                if o["status"] == "done" and s <= _d(o["at"]) <= min(e, today) and any(ln["item"] == it for ln in v.lines.get(o["order"], [])):
                    o["status"], o["status_text"] = "cancel", d["status_text"]["cancel"][0]
    if slot in COMPLEX_SLOTS:
        _target_ext(d, rec, spec, v, s, e, day, add, big, named, items, fac, early)
    if slot not in ("revenue_net", "refund_total", "debt_total", "top_debtor", "purchase_money", "lots_expiring", "debt_collected"):
        # số đo không tính hoàn tiền: thêm một phiếu trả trong kỳ (đáp án đúng không đổi vì khoản này) — bắt code trừ hoàn tiền
        # vào doanh thu bán hoặc đếm phiếu trả như đơn (họ neg_lines/events)
        olds = [o for o in rec["orders"] if o["status"] == "done" and _d(o["at"]) <= day and o["payment"] != "Ghi nợ"]
        if olds:
            ln = v.lines[olds[-1]["order"]][0]
            d["refunds"].append({"refund": "TT0003", "order": olds[-1]["order"], "at": dt.datetime.combine(day, dt.time(15, 0)).isoformat(sep=" "),
                                 "item": ln["item"], "unit": ln["unit"], "qty": 1, "amount": ln["price"]})
    if slot == "revenue_net":  # thêm một khoản hoàn trong kỳ cho đơn bán trước kỳ (chéo kỳ), ngoài phần bán thêm
        olds = [o for o in rec["orders"] if o["status"] == "done" and _d(o["at"]) < s and o["payment"] != "Ghi nợ"]
        if olds:
            ln = v.lines[olds[-1]["order"]][0]
            d["refunds"].append({"refund": "TT0002", "order": olds[-1]["order"], "at": dt.datetime.combine(day, dt.time(14, 0)).isoformat(sep=" "),
                                 "item": ln["item"], "unit": ln["unit"], "qty": 1, "amount": ln["price"]})
    d["orders"].sort(key=lambda o: (o["at"], o["order"]))
    d["refunds"].sort(key=lambda r: (r["at"], r["refund"]))
    try:
        new = gold(d, spec)
    except Degenerate:
        return None
    return (d, desc) if new != base else None
