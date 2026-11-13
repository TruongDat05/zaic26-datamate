# DataMate · Zalo AI Challenge 2026

**Đội Zaic26-Foxel** — Hệ thống trả lời câu hỏi tiếng Việt trên dữ liệu bán hàng từ CSV, Excel và SQLite. Mô hình sinh SQL DuckDB, thực thi trên dữ liệu từng cửa hàng và bỏ phiếu theo kết quả để trả lời hoặc từ chối (`abstain`).

| Thành phần | Cấu hình mặc định |
| --- | --- |
| Mô hình | Qwen3.5-9B + LoRA [`sql_rft_v3`](adapters/sql_rft_v3/), khoảng 9,70B tham số tổng cộng |
| Inference | vLLM 0.32.0, lượng tử hóa FP8, DuckDB 2.0.0 |
| Đầu ra | `submission.jsonl` với đáp án có kiểu, kèm log và trace |

## Cách hoạt động

![Luồng chuẩn bị adapter và inference của DataMate](figures/datamate_workflow.png)

1. **Nạp và chuẩn hóa dữ liệu:** đọc dữ liệu, README của từng shop và rulebook của split vào ngữ cảnh; chuẩn hóa bảng trong DuckDB. Model gốc xây dựng các view bán hàng và macro công nợ, sau đó kiểm tra trên dữ liệu.
2. **Sinh SQL:** dùng hai prompt, một prompt có thêm ví dụ truy xuất từ train. Cấu hình trong [`scripts/flags.sh`](scripts/flags.sh) sinh 8 mẫu SQL, bổ sung tối đa 4 mẫu thinking khi độ đồng thuận thấp hoặc cần xử lý công nợ.
3. **Thực thi và sửa lỗi:** chỉ chấp nhận một câu truy vấn `SELECT`, chạy trong DuckDB với hạn chế truy cập và timeout 20 giây; sửa mẫu lỗi tối đa 2 lượt.
4. **Chọn đáp án:** gom các kết quả tương đương và trả lời khi nhóm thắng đạt ít nhất 50% số mẫu trong nhóm bỏ phiếu. Các quy tắc công nợ, kết quả rỗng và kiểm tra miền thời gian được áp dụng trước khi xuất đáp án.

Các kiểu đáp án gồm `number`, `percent`, `date`, `string`, `boolean`, `list`, `table` và `unanswerable`. `unanswerable` biểu thị dữ liệu không đủ để trả lời; `abstain` là quyết định không đưa ra đáp án.

## Cài đặt

Yêu cầu **Python ≥ 3.12**, Git LFS và môi trường CUDA tương thích với vLLM. Cấu hình inference đã được ghi nhận trên GPU Blackwell 24 GB; công thức train BF16 hướng tới GPU hỗ trợ BF16 với VRAM từ 40 GB. Đọc dữ liệu, chấm điểm và các bài kiểm tra offline có thể chạy bằng CPU.

```bash
git clone https://github.com/TruongDat05/zaic26-datamate.git
cd zaic26-datamate
git lfs install
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[llm,dev]"

hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir models/Qwen3.5-9B
```

Nếu chỉ làm việc trên CPU, cài `pip install -e ".[dev]"` và bỏ qua bước tải model.

**Dữ liệu và tài nguyên cần chuẩn bị riêng:** repo không phân phối dữ liệu thi, dữ liệu SFT hay các tệp sinh từ dữ liệu thi.

| Đường dẫn | Nội dung và mục đích |
| --- | --- |
| `data/train/`, `data/public_test/` | Mỗi split gồm `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<shop>/`; train có thêm `answers.jsonl` để đánh giá |
| `assets/fewshot_bank.jsonl` | Ngân hàng ví dụ train cho các script inference và Docker; tạo từ trace chạy trên train bằng [`training/build_fewshot_bank.py`](training/build_fewshot_bank.py) |
| `models/Qwen3.5-9B/` | Trọng số và tokenizer của model gốc |
| `data/data/` | `train.jsonl`, `dev.jsonl`, `sft_manifest.json` của bộ SFT tổng hợp, cần cho train BF16 |
| `configs/public_v9/` | View và macro công nợ đã lưu; chỉ cần khi chạy public với `FRESH=0` |

## Inference và đánh giá

Các lệnh dưới đây chạy từ thư mục gốc repo, sau khi chuẩn bị model, adapter và ngân hàng ví dụ. Mỗi GPU xử lý một nhóm shop; script gộp kết quả thành `submission.jsonl`.

```bash
# Dựng mới view và lớp công nợ, cùng cách khởi tạo dữ liệu của Docker
FRESH=1 bash scripts/run_public.sh runs/public_fresh

# Dùng lại cấu hình shop đã lưu trong configs/public_v9/
bash scripts/run_public.sh runs/public_cached

# Chạy và chấm 632 câu holdout của train
bash scripts/run_holdout.sh runs/holdout

# Kiểm tra định dạng và độ đầy đủ của file nộp
python -m datamate.submission check runs/public_fresh/submission.jsonl \
  --questions data/public_test/questions.jsonl
```

Có thể ghi đè `NUM_GPUS`, `MODEL_DIR`, `LORA_DIR`, `GPU_MEM`, `ENGINE`, `QUANT` và `PYTHON` qua biến môi trường. Xem tùy chọn CLI bằng `python -m datamate.predict --help`; các script dùng cấu hình chung trong `scripts/flags.sh`, có thể khác mặc định của CLI.

Để chấm một lượt chạy trên train và phân tích lỗi:

```bash
python -m datamate.scoring --data data/train \
  --pred runs/train/submission.jsonl --output runs/train/score.json
python analysis/err.py runs/train
```

Penalized Accuracy được tính bằng `(số đúng − số sai) / tổng số câu`; bỏ trống được 0 điểm. Scorer trong repo là bản tái hiện quy tắc công bố, không phải mã chấm chính thức. Inference không đọc đáp án train.

**Giới hạn đánh giá:** LoRA v3 đã học một phần câu trong `splits/holdout600.txt`, nên holdout chỉ phù hợp để kiểm tra hồi quy, không phải đánh giá độc lập. Điểm public lịch sử dùng các lớp dữ liệu đã lưu và không đại diện cho một lượt `FRESH=1`. Xem [lịch sử điểm và giới hạn](docs/leaderboard.md).

## Huấn luyện LoRA

Repo có hai hướng huấn luyện: [`train_lora.py`](training/train_lora.py) dùng QLoRA 4-bit cho chuỗi adapter v1–v3; [`train_lora_bf16.py`](training/train_lora_bf16.py) dùng bộ SFT tổng hợp để huấn luyện adapter mới trên model gốc BF16. Adapter mặc định của inference vẫn là `sql_rft_v3`.

Chạy công thức BF16 trong môi trường riêng:

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install -r training/requirements_bf16.txt
.venv-train/bin/pip install --no-deps fla-core==0.6.0

# Kiểm tra dữ liệu bằng tokenizer, không cần GPU
ONLY=data bash training/train_a6000.sh

# Kiểm tra kernel, bộ nhớ và huấn luyện; tự tiếp tục nếu có checkpoint phù hợp
bash training/train_a6000.sh
```

Cấu hình mặc định: LoRA rank 16, alpha 32, độ dài tối đa 16.384 token, learning rate `1e-4`, batch hiệu dụng 16 trên một GPU và tối đa 2 epoch. Checkpoint được chọn theo dev loss. Kết quả nằm trong `runs/lora/qwen35_sft_a6000/`, gồm `adapter/`, cấu hình, manifest và checkpoint để tiếp tục train. Các biến tùy chỉnh được ghi ở đầu [`training/train_a6000.sh`](training/train_a6000.sh).

Đánh giá adapter mới bằng `LORA_DIR=<đường_dẫn_adapter>` trước khi thay mặc định. [`scripts/use_adapter.py`](scripts/use_adapter.py) hỗ trợ `--dry-run` để xem trước việc cập nhật các script chạy, Dockerfile và manifest.

Các [notebook](notebooks/) bao quát EDA, inference, đánh giá, dựng dữ liệu SFT và train LoRA; `00_full_pipeline.ipynb` ghép toàn bộ quy trình.

## Docker

Docker dùng base image vLLM 0.32.0 và đóng gói model, adapter, ngân hàng ví dụ cùng mã và bằng chứng huấn luyện. Trước khi build, cần chuẩn bị các dữ liệu SFT, checkpoint và log mà [`scripts/pack_training_code.py`](scripts/pack_training_code.py) yêu cầu; xem [tài liệu đóng gói huấn luyện](docker/training_code_README.md).

```bash
python3 scripts/pack_training_code.py
docker build --platform linux/amd64 -f docker/Dockerfile -t datamate:final .

mkdir -p result
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
  --entrypoint bash \
  -v /absolute/path/to/split:/data:ro \
  -v "$PWD/result:/result" \
  datamate:final /code/predict.sh
```

Container đọc `/data`, ghi `/result/submission.jsonl`, log từng câu tại `/result/logs/` và tệp làm việc tại `/result/work/`. File nộp được cập nhật sau mỗi shop; mỗi tiến trình có giới hạn thời gian 5 giờ 50 phút. Docker luôn dựng mới view và macro công nợ.

Theo [`docker/model_manifest.json`](docker/model_manifest.json), image và thời gian chạy trên RTX 5090 chưa được xác nhận. Giới hạn thời gian trong script không bảo đảm hoàn thành toàn bộ tập private. Chi tiết giao diện và cách chạy: [docker/README.md](docker/README.md).

## Cấu trúc và kiểm tra

```text
datamate/     Pipeline, CLI, đọc dữ liệu, SQL, bỏ phiếu và chấm điểm
training/     Dựng dữ liệu SFT và huấn luyện LoRA
scripts/      Chạy inference, đổi adapter, đóng gói và kiểm tra tĩnh
docker/       Image, entrypoint và tài liệu bản nộp
adapters/     LoRA mặc định, lưu trọng số bằng Git LFS
analysis/     Phân tích lỗi và phát lại trace offline
notebooks/    Hướng dẫn quy trình từ EDA đến huấn luyện
tests/        Kiểm thử tự động
splits/       Danh sách câu hỏi cho các tập con
docs/         Đề bài, ghi chú dữ liệu và lịch sử điểm
figures/      Hình minh họa workflow
```

```bash
python -m pytest -q
python scripts/rule_check.py
```

`rule_check.py` kiểm tra tĩnh mã và cấu hình; kết quả không thay thế kiểm chứng runtime hoặc chấm thi. Các thư mục `data/`, `models/`, `runs/` và tài nguyên sinh từ dữ liệu được loại khỏi Git.

## License và nguồn gốc

Mã nguồn và adapter của dự án theo [MIT License](LICENSE). Model Qwen3.5-9B được tải riêng và theo Apache-2.0; dữ liệu của ban tổ chức không thuộc giấy phép repo. Dự án phát triển từ [baseline AIO](https://github.com/hugebenevolence/zalo-ai-challenge-datamate-26) (MIT).

Tài liệu bổ sung: [Đề bài](docs/challenge.md) · [Ghi chú dữ liệu](docs/data-notes.md) · [Lịch sử điểm](docs/leaderboard.md).
