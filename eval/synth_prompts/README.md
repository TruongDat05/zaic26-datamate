# Prompt cho teacher

Mỗi file là một prompt có chỗ điền `{...}`. Điền bằng dữ liệu trong `data/synth_source` (qua `eval/synth_tools.py`).
Trả lời của teacher luôn là JSON hoặc một khối code, để kiểm bằng máy. Dòng đầu `<!-- phiên bản: vN … -->` ghi phiên bản prompt
(bị bỏ trước khi gửi; kết quả đo ghi kèm phiên bản). Gọi model qua `eval/llm.py`; model nào làm việc nào: `data_spec/model_routing.json`.

| File | Bước | Đầu vào | Đầu ra |
|---|---|---|---|
| `7_danh_muc_nganh_moi.md` | **Giai đoạn 1**: danh mục ngành mới cho bộ mô phỏng dữ liệu shop | tên ngành, số mặt hàng, khuôn `data_spec/catalog_schema.json` | JSON danh mục (lọc cục bộ bằng `synth_spec.LOCAL_BLOCKLIST`) |
| `8_generate_samples.md` | Sinh mẫu dữ liệu train: câu hỏi gốc (đủ dấu, kỳ rõ) + đặc tả + code đáp án; đo bằng `eval/model_bakeoff.py --tasks T4`, chạy chính thức bằng `eval/gen_data.py` (văn phong chủ tiệm thêm sau bằng `stress_noise.VanPhong`) | mô tả bảng, quy ước, số mẫu, danh sách kỹ năng | JSON danh sách mẫu |
| `1_de_bai_tu_seed.md` | Composer: đề bài mới từ seed (OSS-Instruct + Evol) | câu seed, hồ sơ shop, phép biến đổi | câu chuẩn + đặc tả |
| `2_viet_lai_cau_hoi.md` | Composer: viết lại theo phong cách chủ tiệm | câu chuẩn, đặc tả, phong cách | các cách hỏi |
| `3_doc_nguoc_dac_ta.md` | Reviewer: đọc ngược để bắt câu mơ hồ | câu hỏi | đặc tả hoặc báo mơ hồ |
| `4_cau_hoi_tu_bien_the_luat.md` | Composer: câu chạm dòng luật đã đổi + câu đối chứng | biến thể luật, hồ sơ shop | câu + đặc tả |
| `5_chan_doan_loi.md` | Chẩn đoán lỗi từ log | câu, luật, code sai, đáp án sai, đáp án chuẩn | nhãn K, nguyên nhân, code sửa |
| `6_viet_code.md` | Executor: teacher viết code | `Kit.prompt(q)` (đúng prompt lúc chạy) | một khối code |

Quy tắc chung: chỉ dùng shop `mining`; mọi đáp án phải đến từ oracle hoặc từ code chạy ra khớp; không đưa tên bảng, cột không có trong
mô tả dữ liệu; không chép định nghĩa từ rulebook public.
