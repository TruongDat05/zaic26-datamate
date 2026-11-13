# DataMate Synthetic

Pipeline sinh dữ liệu tổng hợp cho bài toán trả lời câu hỏi tiếng Việt trên sổ bán hàng. Mỗi mẫu kết hợp dữ liệu shop, câu hỏi, đáp án chuẩn và lời giải SQL đã kiểm chứng.

Repo giữ **hai nhánh riêng biệt**, mỗi nhánh có code và môi trường Python riêng:

| Nhánh | Mục đích |
| --- | --- |
| [`main`](https://github.com/TruongDat05/zaic26-datamate/tree/main) | Inference, đánh giá, train LoRA và Docker nộp bài. |
| [`synthetic_data`](https://github.com/TruongDat05/zaic26-datamate/tree/synthetic_data) | Sinh shop, câu hỏi, lời giải và kiểm chứng dữ liệu tổng hợp. |

Hai nhánh cùng có package tên `datamate` nhưng giao diện khác nhau. Dùng checkout/worktree riêng; PR sửa pipeline sinh data phải có base là `synthetic_data`. Xem [cách làm việc với hai nhánh](docs/branches.md).

![Workflow sinh synthetic data cho DataMate](docs/assets/synthetic-data-workflow.png)

## Quy trình

| Bước | Xử lý |
| --- | --- |
| **1. Chuẩn bị nguồn** | Trích schema, kỹ năng, quy ước và phân bố từ phần mining của train; tách dữ liệu đánh giá. |
| **2. Mô phỏng shop** | Sinh danh mục, giao dịch, thanh toán, công nợ và tình huống biên; xuất bảng rồi nạp lại bằng runtime. |
| **3. Sinh câu hỏi** | Viết câu hỏi từ đặc tả; model độc lập đọc ngược để kiểm số đo, đối tượng và kỳ thời gian. |
| **4. Sinh lời giải SQL** | Giải từ câu hỏi và ngữ cảnh shop, không cung cấp đáp án chuẩn cho model. |
| **5. Kiểm chứng và audit** | Kiểm tĩnh, thực thi lặp, so gold, thử trên dữ liệu đột biến và audit mẫu phân tầng; loại hoặc sửa mẫu lỗi rồi kiểm lại. |
| **6. Đóng gói dữ liệu** | Chọn theo phân bố mục tiêu, tách train/dev/final; xuất gói nguồn và chuyển qua exporter SFT với manifest, bằng chứng truy vết. |

**Gold được tính độc lập bằng oracle Python trên máy**, kể cả khi dữ liệu bị đột biến. Final được giữ riêng làm benchmark, không dùng để chọn mẫu train.

## Model

| Model | Vai trò |
| --- | --- |
| **DeepSeek 4 Flash** | Sinh câu hỏi từ đặc tả và bổ sung lời giải. |
| **Qwen 3.8 27B** | Kiểm nghĩa câu hỏi bằng đọc ngược và sinh SQL. |
| **Gemini** | Hỗ trợ diễn đạt, sinh thêm câu hỏi và audit độc lập. |

## Cài đặt

Cần Python 3.12 và `uv`. Các kiểm thử sandbox dùng Linux có Landlock ABI ≥ 4 và `libseccomp.so.2`. Bắt đầu bằng môi trường CPU của nhánh sinh dữ liệu:

```bash
git clone --branch synthetic_data https://github.com/TruongDat05/zaic26-datamate.git zaic26-datamate-synth
cd zaic26-datamate-synth

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export SYNTH_PY="$PWD/.venv/bin/python"
"$SYNTH_PY" eval/check_setup.py
"$SYNTH_PY" -m pytest -q -rs
```

Luồng SQL cần runtime ngoài nhánh này. Có thể dùng checkout `main` cho bước dựng shop/kiểm SQL; bước đóng gói và export còn cần controller/snapshot tương thích `offline_candidates/1` (xem bảng bên dưới).

```bash
git fetch origin
git worktree add --detach ../datamate-inference origin/main
export DATAMATE_UPSTREAM="$(cd ../datamate-inference && pwd)"

uv venv --python 3.12 .venv-sql
uv pip install --python .venv-sql/bin/python \
  -r requirements.txt -e "$DATAMATE_UPSTREAM"

export SQL_PY="$PWD/.venv-sql/bin/python"
test -f .env || cp .env.example .env
```

Điền các biến API trong [`.env.example`](.env.example): `API_*` cho DeepSeek 4 Flash, `MK_*` cho Qwen 3.8 27B và `GEMINI_*` cho Gemini. Chỉ các lệnh sinh qua model/audit LLM mới cần khóa API; cài đặt, `check_setup.py` và kiểm thử CPU không gọi API.

Đặt dữ liệu nguồn trong `data/train/`, gồm `data/<business_id>/`, `questions.jsonl`, `answers.jsonl`, `calendar.csv` và `rulebook.md`. Bước dựng nguồn giữ `pandas==2.2.3` theo `requirements.txt`.

| Bước | Điều kiện bổ sung |
| --- | --- |
| Dựng nguồn (`--stage source`) | Bộ train cục bộ trong checkout `synthetic_data`. |
| Dựng shop/kiểm SQL (`--stage sql`) | `DATAMATE_UPSTREAM/datamate/` có runtime SQL; DuckDB 2.0.0 trong `SQL_PY`. |
| Đóng gói/export (`--stage export`) | `DATAMATE_CONTROLLER` có `harness_control/{evaluate,exec_guard,scorer}.py`, `training/{sft_format,make_synthetic_fixture,offline_candidates,build_sft}.py`; runtime có `datamate/snapshot.py` và `training/make_synthetic_fixture.py`. |

**Controller/exporter chưa nằm trong hai nhánh này.** `main/training/build_sft.py` dựng SFT từ trace và không hỗ trợ `--offline-candidates`; không dùng nó thay exporter SFT v2. Đặt `DATAMATE_CONTROLLER` tới checkout tương thích, chọn `DATAMATE_UPSTREAM` tương ứng **trước khi bắt đầu lượt sinh**, và giữ nguyên revision đến khi export. Nếu chưa đặt biến, code giữ mặc định controller ở thư mục anh em `../datamate_foxel`.

```bash
"$SYNTH_PY" eval/check_setup.py --stage source
"$SQL_PY" eval/check_setup.py --stage sql
# Sau khi cấu hình controller và runtime đầy đủ:
"$SQL_PY" eval/check_setup.py --stage export
```

Các lệnh trên chỉ kiểm phiên bản/phụ thuộc và tệp cần có; không thay thế kiểm chứng pipeline thực tế.

## Generate dữ liệu

Ví dụ một lượt sinh trên nhóm shop train thuộc các ngành lõi:

```bash
# Dựng nguồn và mô phỏng shop
"$SYNTH_PY" eval/build_synth_source.py
"$SYNTH_PY" eval/build_data_spec.py
"$SQL_PY" eval/sqlv1/shops.py build --procs 4

# Sinh câu hỏi và kiểm nghĩa độc lập
mkdir -p outputs/synth_sql_v1/questions
"$SQL_PY" eval/sqlv1/questions.py gen \
  --shops core_train --n-per-shop 24 --seed initial \
  --teacher deepseek --verifier qwen --workers 4 --out gen_train
"$SQL_PY" eval/sqlv1/pool.py

# Sinh SQL và kiểm chứng
"$SQL_PY" eval/sqlv1/solve.py run \
  --pool outputs/synth_sql_v1/questions/pool_train.jsonl \
  --provider qwen --procs 4
```

Xem [hướng dẫn đầy đủ](eval/README.md) để sinh đủ train/dev/final, bổ sung lời giải, audit và xuất SFT. Số mẫu được giữ phụ thuộc kết quả kiểm chứng; gói nguồn cần qua exporter trước khi sử dụng để huấn luyện.

## Cấu trúc

```text
datamate/           Loader, tính toán và sandbox phục vụ dựng nguồn
eval/sqlv1/         Pipeline SQL: mô phỏng, câu hỏi, oracle, solver, kiểm và đóng gói
eval/synth_*.py     Mô phỏng và công cụ corpus dùng chung
eval/synth_prompts/ Prompt sinh dữ liệu
tests/              Kiểm contract và regression
docs/assets/        Ảnh workflow
.github/workflows/  Kiểm thử CPU cho PR vào synthetic_data
```

Dữ liệu và artifact chạy nằm trong `data/`, `outputs/` và thư mục `build/` của controller. `.gitignore` loại dữ liệu, output, cache, log, database, môi trường ảo, khóa API và model weights khỏi các commit mới.

## Kiểm thử

`"$SYNTH_PY" -m pytest -q -rs` chạy kiểm thử CPU với dữ liệu tự tạo; các test cần dữ liệu thi hoặc hồ sơ mining báo `skip` khi thiếu đầu vào. Test mang marker `sql_runtime` được tách khỏi lệnh mặc định. CI dùng cùng lệnh, thêm DuckDB 2.0.0 để kiểm biểu thức SQL cục bộ. Sandbox luôn giữ giới hạn filesystem/network; môi trường thiếu hỗ trợ isolation sẽ báo lỗi.

Chạy kiểm thử quy ước SQL với runtime bằng **tiến trình riêng**, tránh trộn hai package `datamate`:

```bash
"$SQL_PY" -m pytest -q -o addopts= tests/test_sqlv1_rules.py
```

Kết quả CPU/SQL offline không xác nhận API provider, chất lượng dữ liệu sinh thực tế hay export SFT khi chưa có controller.
