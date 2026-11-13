# Hướng dẫn generate synthetic data

Thực hiện [cài đặt và chuẩn bị nguồn](../README.md#cài-đặt) trước khi chạy. Các lệnh dưới đây dùng thư mục làm việc chuẩn `outputs/synth_sql_v1/` và cùng runtime `DATAMATE_UPSTREAM` trong cả lượt sinh.

Chạy từ thư mục gốc checkout **`synthetic_data`**. Các script không sửa code nhánh `main`; chỉ nạp runtime đã chọn ở checkout khác. Để hoàn thành toàn bộ quy trình bên dưới cần controller/exporter tương thích `offline_candidates/1`, hiện chưa được phân phối trong hai nhánh.

## 1. Nguồn và shop tổng hợp

```bash
"$SYNTH_PY" eval/check_setup.py --stage source
"$SQL_PY" eval/check_setup.py --stage sql
"$SYNTH_PY" eval/build_synth_source.py
"$SYNTH_PY" eval/build_data_spec.py
"$SQL_PY" eval/sqlv1/shops.py build --procs 4
```

`data/synth_source/` chứa hồ sơ shop, nhãn kỹ năng, quy ước và split. Shop được mô phỏng từ seed, xuất file và nạp lại bằng runtime; kiểm `outputs/synth_sql_v1/build_report.json` trước khi sinh câu hỏi.

Nếu có log mining, chạy `ingest_real_logs.py` sau khi dựng nguồn và trước `build_data_spec.py`. Kho code mẫu dựng bằng `build_code_bank.py --runs <thư_mục_log>` được ghi vào `data/code_bank.jsonl`; chạy lại `build_synth_source.py` sau khi cập nhật kho.

## 2. Câu hỏi cho train/dev/final

DeepSeek 4 Flash viết câu hỏi từ đặc tả; Qwen 3.8 27B đọc ngược để kiểm nghĩa. Gold được oracle tính độc lập và không gửi cho hai vai trò này. Có thể dùng `--teacher gemini` khi cần Gemini hỗ trợ.

```bash
mkdir -p outputs/synth_sql_v1/questions

for split in train dev final; do
  shop_ids=$("$SQL_PY" -c '
import json, sys
from pathlib import Path
plan = json.loads(Path("outputs/synth_sql_v1/plan.json").read_text())
print(",".join(s["business_id"] for s in plan if s["split"] == sys.argv[1]))
' "$split")

  "$SQL_PY" eval/sqlv1/questions.py gen \
    --shops "$shop_ids" --n-per-shop 24 --seed initial \
    --teacher deepseek --verifier qwen --workers 4 --out "gen_$split"
done

"$SQL_PY" eval/sqlv1/pool.py
```

Pool loại mẫu không qua kiểm nghĩa và câu trùng sau chuẩn hóa, tạo `pool_train.jsonl`, `pool_dev.jsonl`, `final_set.jsonl`. `24` là số đặc tả yêu cầu mỗi shop, không phải số mẫu cam kết được giữ.

## 3. SQL và kiểm chứng

```bash
for split in train dev; do
  pool="outputs/synth_sql_v1/questions/pool_$split.jsonl"

  "$SQL_PY" eval/sqlv1/solve.py run \
    --pool "$pool" --provider qwen --procs 4
  "$SQL_PY" eval/sqlv1/solve.py run \
    --pool "$pool" --provider deepseek --only-unsolved --procs 4
  "$SQL_PY" eval/sqlv1/solve.py template \
    --pool "$pool" --only-unsolved --procs 4
done
```

Qwen 3.8 27B giải trước; DeepSeek 4 Flash bổ sung câu chưa đạt. SQL mẫu tất định hỗ trợ các slot còn thiếu và chịu cùng bộ kiểm. Mỗi ứng viên được kiểm tĩnh, chạy lặp, so gold và chạy trên dữ liệu đột biến có gold tính lại. Luồng API cho phép một lượt sửa lỗi. Final chỉ giữ câu hỏi + gold cho benchmark.

## 4. Chọn mẫu và đóng gói nguồn

Từ bước này cần controller/snapshot đầy đủ đã chọn từ đầu lượt sinh. `package.py` dùng `training/make_synthetic_fixture.py` của runtime để tính content hash; chỉ có runtime inference của `main` là chưa đủ. Không thay runtime giữa một lượt sinh và export; nếu cần revision khác, tạo lượt mới.

```bash
"$SQL_PY" eval/check_setup.py --stage export
"$SQL_PY" eval/sqlv1/package.py \
  --target-rows 6600 --out "$DATAMATE_CONTROLLER/build"
```

Bước chọn cân bằng ngành, kỹ năng, kiểu đáp án, kỳ thời gian và độ dài theo phân bố mining. Một câu có thể giữ tối đa hai lời giải đạt với cấu trúc khác nhau; `6600` là mục tiêu số dòng.

| Đầu ra trong `build/` của controller | Nội dung |
| --- | --- |
| `synthetic_sql_v1/` | `data_root/`, `source_manifest.json`, ứng viên SQL và lý do loại của train/dev. |
| `synthetic_sql_v1_final/` | Benchmark final, có vai trò riêng và không dùng làm dữ liệu train. |

## 5. Replay và export SFT

Runtime/snapshot và exporter upstream cần tương thích `offline_candidates/1`.

`training/build_sft.py` trên nhánh `main` có giao diện khác và không dùng cho lệnh export này. Xem [bảng phụ thuộc](../README.md#cài-đặt) và chỉ chạy tiếp khi `check_setup.py --stage export` đạt; cần kiểm chứng thực thi sau kiểm cấu trúc.

```bash
export SYNTH_PACKAGE="$DATAMATE_CONTROLLER/build/synthetic_sql_v1"

"$SQL_PY" eval/sqlv1/mutation_replay.py \
  --snapshot "$DATAMATE_UPSTREAM" --pkg "$SYNTH_PACKAGE" --procs 4
"$SQL_PY" eval/sqlv1/bundle.py \
  --snapshot "$DATAMATE_UPSTREAM" --pkg "$SYNTH_PACKAGE"

"$SQL_PY" "$DATAMATE_CONTROLLER/training/build_sft.py" \
  --data-root "$SYNTH_PACKAGE/data_root" \
  --manifest "$SYNTH_PACKAGE/source_manifest.json" \
  --offline-candidates "$SYNTH_PACKAGE/offline_bundle" \
  --source "$DATAMATE_UPSTREAM" --out "$SYNTH_PACKAGE/sft"
```

Bundle giữ bằng chứng từ response/cache, ledger hoặc sự kiện sinh SQL mẫu. Exporter kiểm lại nguồn, lineage và lời giải trước khi ghi `sft/train.jsonl`, `sft/dev.jsonl`, `sft/rejected.jsonl` và `sft_manifest.json`.

## 6. Audit và EDA

```bash
"$SQL_PY" eval/sqlv1/audit.py sample --n 160
# Chạy khi đã cấu hình Gemini:
"$SQL_PY" eval/sqlv1/audit.py llm --auditor gemini
"$SQL_PY" eval/sqlv1/audit.py summarize
"$SQL_PY" eval/sqlv1/eda.py stats
```

Audit dùng mẫu phân tầng, auditor khác họ solver và các lời giải cố ý sai đã kiểm cục bộ. EDA so phân bố synthetic với mining bằng chỉ số gộp. `sft_gates.py` tổng hợp cổng chất lượng trên các dòng exporter thực sự nhận; cần báo cáo tokenizer/template thực tế, replay, audit và đối chiếu quy ước tương ứng. `eda_notebook.py` tạo notebook khi đã có đủ các artifact này.

## Công cụ chính

| Module | Trách nhiệm |
| --- | --- |
| `sqlv1/paths.py`, `sqlv1/up.py` | Chọn controller/runtime và kiểm xuất xứ package. |
| `sqlv1/shops.py`, `sqlv1/render.py` | Kế hoạch shop, xuất bảng, nạp lại và dựng view. |
| `sqlv1/world.py`, `sqlv1/specs.py` | Oracle, đặc tả và đột biến dữ liệu. |
| `sqlv1/questions.py`, `sqlv1/pool.py` | Sinh câu hỏi, kiểm nghĩa và lọc trùng. |
| `sqlv1/solve.py`, `sqlv1/sqlgen.py`, `sqlv1/verify.py` | Lời giải model/SQL mẫu và kiểm chứng. |
| `sqlv1/package.py`, `sqlv1/bundle.py` | Chọn mẫu và gói nguồn/bằng chứng. |
| `llm.py`, `sqlv1/api.py` | API, guard, ledger và cache. |
| `synth_world.py`, `synth_corpus.py`, `synth_tools.py` | Mô phỏng và công cụ pandas còn được sử dụng. |

Quy ước nghiệp vụ: [`rules_spec.md`](sqlv1/rules_spec.md). Tiêu chí audit: [`AUDIT_RUBRIC.md`](sqlv1/AUDIT_RUBRIC.md).

Cache và ledger cho phép tiếp tục lượt chạy. Giữ nguyên `outputs/synth_sql_v1/requests.sqlite`, seed và snapshot khi chạy tiếp; không tạo ledger mới để bỏ qua ngân sách hiện có. Chỉ nội dung tự tạo được gửi API; train, log và dữ liệu đánh giá được xử lý cục bộ.
