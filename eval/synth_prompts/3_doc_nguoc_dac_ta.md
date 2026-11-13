Đọc câu hỏi của chủ cửa hàng và ghi lại ý định. Không tính số. Câu có thể viết tắt, không dấu, sai chính tả.
Hôm nay: {as_of_date}. Mặt hàng có thể liên quan: {items}. Nhóm hàng: {groups}.

Câu hỏi: {cau_hoi}

Nếu câu có hơn một cách hiểu hợp lý theo rulebook (kỳ nào, mặt hàng nào, đếm gì, tính tiền hay số lượng…), đặt `mo_ho = true` và ghi
các cách hiểu. Không đoán thay người hỏi.

Trả về đúng một JSON:
```json
{"mo_ho": false, "cach_hieu": [], "dac_ta": {"so_do": "...", "ky": "...", "mat_hang": null, "nhom": null, "khach": null,
 "kenh_ca_thanh_toan": null, "nguong": null, "top_k": null, "kieu_dap_an": "..."}}
```
