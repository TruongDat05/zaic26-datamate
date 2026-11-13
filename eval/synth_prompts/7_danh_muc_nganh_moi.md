<!-- phiên bản: v2 (9/11: ghi rõ kiểu dữ liệu, có ví dụ đúng kiểu, luật ngành là định nghĩa tính được) -->
Bạn thiết kế danh mục hàng cho một cửa hàng nhỏ ở Việt Nam thuộc ngành **{ten_nganh}**, để chương trình mô phỏng dữ liệu bán hàng 12–18 tháng.
Mọi con số phải thực tế với thị trường Việt Nam năm 2025–2026.

Yêu cầu:
1. {n_items} mặt hàng (kể cả 1–3 dòng dịch vụ hoặc phí nếu ngành có), tên đầy đủ như trên phần mềm bán hàng (thương hiệu, quy cách, dung tích…),
   không trùng nhau, không trùng tên giữa hai mặt hàng chỉ khác dấu.
2. 5–12 nhóm hàng. Mỗi mặt hàng có đơn vị cơ bản; mặt hàng bán theo đơn vị lớn thì ghi hệ số quy ra đơn vị cơ bản
   (nhiều cấp thì nhân thẳng: thùng = 4 lốc = 24 lon → thùng 24).
3. Giá một đơn vị cơ bản (VND), tỉ lệ giá vốn, độ phổ biến 1–5 (lệch: ít mặt hàng rất chạy, nhiều mặt hàng ít bán).
4. Mùa vụ: hệ số số đơn dịp Tết so với trước Tết, các dịp riêng của ngành (ngày dương lịch MM-DD, hệ số), hệ số 7 ngày từ thứ Hai đến Chủ nhật;
   giờ mở cửa.
5. Shop có ghi những gì: khách có mã, nhân viên, công nợ, nhập hàng, tồn kho, lô/hạn dùng, giao nhiều đợt, chi phí.
6. 2–4 luật riêng của ngành, mỗi luật một dòng bảng hai cột `| khái niệm | cách tính |`:
   - cột 1: cụm từ chủ tiệm ngành này hay dùng khi hỏi số liệu;
   - cột 2: định nghĩa tính được từ dữ liệu: lọc mặt hàng/đơn nào (theo nhóm, attrs, trạng thái…), cộng hay đếm gì, quy đổi gì;
   - thuộc tính mà luật dùng phải có trong attrs của mặt hàng; không viết chính sách bán hàng (giảm giá, cọc, đổi trả…); không thêm dòng tiêu đề.

Ngành đã có trong dữ liệu, KHÔNG làm lại: tạp hoá, cà phê, vật liệu xây dựng, nhà thuốc, quán nhậu.

<!-- Lọc trùng với khái niệm/ngành ngoài train làm CỤC BỘ bằng synth_spec.LOCAL_BLOCKLIST sau khi nhận JSON; không đưa danh sách đó vào prompt gửi API. -->

Kiểu dữ liệu: số ghi dạng số JSON (không đặt trong ngoặc kép); khoảng ghi [thấp, cao]; giờ là số 0–24; `weekday` là danh sách đúng 7 số;
các trường trong `has` là true/false. Ý nghĩa từng trường:
```json
{catalog_schema}
```

Ví dụ đúng kiểu (ngành khác, chỉ để xem cách ghi, KHÔNG chép mặt hàng):
```json
{catalog_example}
```

Trả về đúng một JSON, không giải thích thêm.
