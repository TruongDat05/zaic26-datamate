# Rubric audit bởi LLM độc lập (Bước 7) — syn_sales_sql_v1

Người chấm theo rubric này là subagent Claude độc lập (không phải người). Các file `manual_verdicts*.jsonl` mang tên cũ, nội dung là kết quả audit bởi LLM độc lập.

Mỗi mục trong `outputs/synth_sql_v1/audit/sample.jsonl` là một dòng SFT ứng viên: câu hỏi của chủ shop (`question`, `as_of_date`),
và lời giải (`completion`: Notes, `TYPE:`, khối ```sql```). Quy ước tính: `eval/sqlv1/rules_spec.md`. Dữ liệu shop: `ctx` trong
`outputs/synth_sql_v1/shops/<business_id>/shop.json` (README + SCHEMA + CANONICAL VIEWS + DEBTS, đúng như model thấy).
`gold` chỉ để tham khảo (lời giải đã khớp gold khi chạy).

Một mục ĐẠT khi cả ba đều đúng:
1. **nghia** — SQL tính đúng điều câu hỏi hỏi: số đo, đối tượng (tên hàng/nhóm/kênh/khách), kỳ (cửa sổ ngày khớp DATE ANCHORS của
   `as_of_date`: tháng M/Y, tuần bắt đầu thứ Hai, hôm qua = 1 ngày, "tháng này so với tháng trước" = cùng số ngày, N ngày gồm hôm nay),
   kiểu `TYPE` đúng (tháng nào → date `YYYY-MM`, ngày nào → date, top k → list, một đối tượng → string, có/không → boolean).
   Câu mơ hồ (hiểu được hai cách cho hai đáp án khác) là KHÔNG ĐẠT.
2. **quy_uoc** — đúng rules_spec: chỉ đơn hợp lệ (đang giao tính; hủy/trả bỏ cả đơn) cho doanh thu/số đơn/số lượng/xếp hạng; tỷ lệ hủy
   chia mọi đơn; đếm theo mã đơn; khách lẻ không là khách; số lượng theo đơn vị cơ bản; tiền làm tròn đồng, % 2 số lẻ, % thay đổi âm khi giảm;
   đồng hạng theo mã; công nợ qua macro debt_*; ế = có bán trước kỳ mà trong kỳ không bán; kê đơn theo cờ kê đơn trong danh mục;
   phí phục vụ là doanh thu (không lọc line_kind = 'item' khi tính doanh thu). Bảng (TYPE table): thứ tự DÒNG không ảnh hưởng chấm.
3. **sql_sach** — một câu SELECT/WITH chỉ đọc, không ghi cứng đáp án hay ngày tự chế, chỉ dùng bảng/cột có trong SCHEMA/VIEWS, Notes ≤ 6 dòng,
   ngắn, đúng với SQL (không mâu thuẫn, không trích mã mục rulebook).

Ghi mỗi mục một dòng vào `outputs/synth_sql_v1/audit/manual_verdicts.jsonl`:
`{"row": ..., "id": ..., "nghia": true/false, "quy_uoc": true/false, "sql_sach": true/false, "pass": <cả ba>, "ly_do": "<ngắn, khi không đạt>"}`.
