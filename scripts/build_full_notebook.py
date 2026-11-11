"""Build one notebook that runs the whole workflow (01 EDA -> 02 inference -> 03 evaluation -> 04 training data -> 05 LoRA training)
from the five notebooks, so it never drifts from them.

    .venv/bin/python scripts/build_full_notebook.py                                   # notebooks/00_full_pipeline.ipynb (no outputs)
    .venv/bin/python scripts/build_full_notebook.py --from-dir ~/projects/datamate_notebooks_outputs \
        --out ~/projects/datamate_notebooks_outputs/00_full_pipeline.ipynb            # the same cells with each part's executed outputs

One setup and one import cell at the top. Each part keeps its parameters cell and its cells, the per-part package install
(`install("llm")`, `install("train")`, a no-op in a local checkout) becomes the part's first code cell, and headings move down one level.
The full notebook is the teaching version: the markdown in LESSON replaces the parts' engineering notes, and EXTRA adds a few cells
(an example per answer type, one question through the pipeline)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import re
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
PARTS = ["01_eda", "02_inference", "03_evaluation", "04_training_data", "05_train_lora"]
INTRO = """# DataMate · Baseline text-to-SQL (Zalo AI Challenge 2026, Track 2)

**Bài toán:** trả lời câu hỏi tiếng Việt về dữ liệu kinh doanh của một cửa hàng (các bảng csv / xlsx / sqlite và một README). Đáp án có kiểu
cố định (số, phần trăm, chuỗi, danh sách, ngày, bảng…), hoặc bỏ trống khi không chắc.

**Ý tưởng của baseline:**
- Mô hình ngôn ngữ (Qwen3.5-9B + LoRA) **viết SQL**, còn DuckDB **thực hiện phép tính** trên dữ liệu thật.
- Mỗi câu hỏi lấy nhiều mẫu SQL, chạy từng mẫu rồi **bỏ phiếu** theo đáp án. Khi các mẫu không đồng thuận, hệ thống **bỏ trống**
  thay vì đoán, vì câu trả lời sai bị trừ điểm.
- LoRA được fine-tune bằng **SFT LoRA BF16**. Dữ liệu mặc định là bộ SFT tổng hợp trong `data/data/` (shop và câu hỏi được sinh ra, đã chia
  train / dev theo shop). Phần 04 là cách thứ hai: dựng dữ liệu từ chính các lời giải đúng mà mô hình tìm ra trên tập train (rejection sampling),
  giữ 6 shop holdout ngoài dữ liệu train để đo sạch.

| Phần | Nội dung | Phần cứng |
|---|---|---|
| 01 · EDA | dữ liệu gồm những gì: shop, tệp, kiểu đáp án, dạng câu hỏi, rulebook | CPU |
| 02 · Suy luận | chạy pipeline và ghi file nộp `submission.jsonl` | GPU bf16 từ 24 GB |
| 03 · Đánh giá | điểm PA, nhóm lỗi, ngưỡng bỏ phiếu, so hai adapter trên cùng câu hỏi | CPU |
| 04 · Dữ liệu SFT | dựng dữ liệu SFT từ các lời giải đúng, chia train / dev theo shop (không bắt buộc khi dùng bộ SFT `data/data`) | CPU, GPU cho bước ghi chú |
| 05 · Fine-tune | SFT LoRA BF16 trên Qwen3.5-9B | GPU bf16 từ 40 GB |

Chạy từ trên xuống. Mỗi phần có một ô tham số ở đầu phần. Phần 01-04 cần dữ liệu BTC (`data/train`, `data/public_test` hoặc `DATAMATE_DATA`),
phần 05 chỉ cần bộ SFT `data/data` (chỉ train thì mở `05_train_lora` riêng). Khi không có GPU đủ lớn (ví dụ T4 trên Kaggle), các bước cần mô hình được bỏ qua
hoặc dùng kết quả đã chạy sẵn nếu checkout có (`runs/saved`)."""

# Text for learners: replaces the markdown cell whose first line matches (after headings move down one level).
LESSON = {
    "## 01 · Khám phá dữ liệu (EDA)": """## 01 · Khám phá dữ liệu (EDA)

Trước khi xây pipeline, ta xem dữ liệu trông như thế nào:
- mỗi cửa hàng (shop) có bao nhiêu câu hỏi và những tệp dữ liệu gì
- đáp án có những kiểu nào, câu hỏi thường hỏi gì
- rulebook quy định cách tính (doanh thu, hàng bán chạy, công nợ…) ra sao
- một shop sau khi nạp vào DuckDB trông thế nào: đây chính là phần schema mà mô hình nhìn thấy""",
    "### Dạng câu hỏi": """### Dạng câu hỏi

Tỷ lệ câu hỏi chứa một số dạng khó, ví dụ khung thời gian ("tuần rồi", "quý 2"), công nợ, số tiền viết tắt ("3tr7") hay từ địa phương.
Pipeline tính sẵn gợi ý cho các dạng này (`datamate/hints.py`) để mô hình không phải tự suy đoán.""",
    "## 02 · Suy luận (inference)": """## 02 · Suy luận (inference)

Pipeline xử lý từng shop qua năm bước:
1. **Nạp dữ liệu** vào DuckDB và dựng sẵn view chuẩn (`v_sales`, `v_orders`) cùng macro công nợ, để câu SQL ngắn và ít sai.
2. **Tạo prompt** gồm rulebook, schema, khung thời gian tính sẵn, vài ví dụ tương tự và câu hỏi.
3. **Lấy mẫu:** mô hình viết 8 câu SQL. Câu khó (các mẫu không đồng thuận) được thêm 4 mẫu ở chế độ thinking.
4. **Thực thi và sửa lỗi:** DuckDB chạy từng câu SQL. Câu bị lỗi được gửi lại cho mô hình kèm thông báo lỗi để sửa.
5. **Bỏ phiếu:** gom các đáp án giống nhau. Hệ thống chỉ trả lời khi nhóm lớn nhất đủ đồng thuận, ngược lại thì bỏ trống.

Kết quả là file `submission.jsonl` đúng định dạng nộp bài. `LIMIT = 10` chạy thử 10 câu, còn cả 1000 câu public mất khoảng 3,5 giờ
trên một GPU 24 GB.

| Mục đích | `SPLIT` | `IDS` | `REUSE_LAYERS` | `LORA` | `SAVE_PROMPTS` |
|---|---|---|---|---|---|
| Bản nộp public | `"public_test"` | `None` | `True` | adapter đã chọn | tùy |
| Lượt có nhãn để dựng dữ liệu SFT (phần 04) | `"train"` | `"splits/rft_train24.txt"` | `False` | `"sql_rft_v3"` | `True` |
| Đo adapter mới trên holdout (phần 05 → 03) | `"train"` | `"splits/holdout600.txt"` | `False` | `"runs/lora/qwen35_sft_a6000/adapter"` | tùy |
| Mốc so sánh trên holdout | `"train"` | `"splits/holdout600.txt"` | `False` | `"sql_rft_v3"` | tùy |""",
    "### Kết quả": """### Kết quả

Số câu đã trả lời theo shop, vài câu mẫu cùng đáp án và tỷ lệ đồng thuận (`conf`). Với `SPLIT = "train"` có thêm điểm theo đáp án chuẩn.""",
    "## 03 · Đánh giá": """## 03 · Đánh giá

Điểm của cuộc thi là **PA (penalized accuracy)**: câu đúng +1, câu sai −1, câu bỏ trống 0, rồi chia cho số câu. Vì câu sai bị trừ điểm,
bỏ trống những câu không chắc thường có lợi hơn là đoán.

Phần này gồm:
1. Chấm một lượt chạy trên **train / holdout** (có đáp án): điểm tổng, theo shop, theo dạng câu, nhóm lỗi và ngưỡng bỏ phiếu.
2. So **hai adapter trên cùng câu hỏi** (adapter BF16 mới với `sql_rft_v3`): chênh lệch PA và khoảng tin cậy bootstrap theo shop.
3. So các **bản nộp public** với nhau, không cần nhãn.""",
}

# A few cells for learners, inserted after the code cell whose source starts with the anchor.
EXTRA = {
    "types = Counter(": [
        ("markdown", "Mỗi kiểu đáp án một ví dụ:"),
        ("code", """example = {}
for q in train_q:
    a = train_a[q["id"]]
    example.setdefault(a["type"], (q["business_id"], q["question"], a.get("answer")))
pd.DataFrame([{"type": t, "shop": shop, "question": text[:100], "answer": json.dumps(ans, ensure_ascii=False)[:70]}
              for t, (shop, text, ans) in example.items()])"""),
    ],
    "trace = [r for p in sorted(out.glob": [
        ("markdown", """### Một câu hỏi đi qua pipeline

Câu hỏi có tỷ lệ đồng thuận thấp nhất (thường là câu khó, được thêm mẫu thinking): chế độ, câu SQL và đáp án của từng mẫu, rồi kết quả bỏ phiếu.
- Mẫu không viết ra được SQL hợp lệ (ví dụ ghi chú lặp mãi không dừng) không được tính phiếu.
- Tỷ lệ đồng thuận = số phiếu của đáp án thắng / tổng số mẫu."""),
        ("code", """def last_sql(text):
    return " ".join(text.split("```sql")[-1].split("```")[0].split()) if "```sql" in text else ""


if trace:
    q = min(trace, key=lambda r: (r["conf"], -len(r["samples"])))
    print(q["question"])
    samples = pd.DataFrame([{"mẫu": k + 1,
                             "chế độ": "thinking" if "</think>" in s["text"] else "thường",
                             "đáp án": json.dumps(s["pred"]["answer"], ensure_ascii=False)[:50] if s.get("pred") else "",
                             "SQL": last_sql(s["text"])[:110] or "(lỗi: " + (s.get("error") or "").split(" | ")[0] + ")"}
                            for k, s in enumerate(q["samples"])])
    display(samples)
    votes = Counter(a for a in samples["đáp án"] if a)
    print("bỏ phiếu:", dict(votes.most_common()), f"| {len(samples)} mẫu")
    print("đáp án cuối:", json.dumps(q["final"], ensure_ascii=False), "| tỷ lệ đồng thuận:", round(q["conf"], 2))"""),
    ],
}


def demote(md: str) -> str:
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", md, flags=re.M)


def install_kind(setup_source: str) -> str:
    return re.search(r'install\("(\w+)"\)', setup_source).group(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, default=REPO / "notebooks", help="where the five notebooks are read (executed copies keep outputs)")
    ap.add_argument("--out", type=Path, default=REPO / "notebooks/00_full_pipeline.ipynb")
    a = ap.parse_args()
    books = {name: nbformat.read(a.from_dir / f"{name}.ipynb", as_version=4) for name in PARTS}
    first = books[PARTS[0]]
    full = nbformat.v4.new_notebook(metadata=copy.deepcopy(first.metadata))
    full.cells.append(nbformat.v4.new_markdown_cell(INTRO))
    setup, imports = copy.deepcopy(first.cells[1]), copy.deepcopy(first.cells[2])
    setup.source = setup.source.replace('install("base")', 'install("base")   # mỗi phần bên dưới cài thêm thứ nó cần (chỉ Kaggle / Colab)')
    full.cells += [setup, imports]
    for name in PARTS:
        nb = books[name]
        intro = copy.deepcopy(nb.cells[0])
        intro.source = demote(intro.source)
        intro.source = LESSON.get(intro.source.split("\n")[0], intro.source)
        full.cells.append(intro)
        kind = install_kind(nb.cells[1].source)
        if kind != "base":
            note = "  # Kaggle / Colab: chạy phần này trong một phiên riêng (stack train BF16 khác vLLM)" if kind == "train" else ""
            full.cells.append(nbformat.v4.new_code_cell(f'install("{kind}"){note}'))
        for cell in nb.cells[3:]:
            cell = copy.deepcopy(cell)
            if cell.cell_type == "markdown":
                cell.source = demote(cell.source)
                cell.source = LESSON.get(cell.source.split("\n")[0], cell.source)
            full.cells.append(cell)
            if cell.cell_type == "code":
                for anchor, extra in EXTRA.items():
                    if cell.source.startswith(anchor):
                        full.cells += [nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src) for kind, src in extra]
    count = 0
    for k, cell in enumerate(full.cells):   # stable ids: a rebuild changes only the cells whose text changed
        cell.id = hashlib.sha1(f"00_full_pipeline:{k}".encode()).hexdigest()[:8]
        if cell.cell_type == "code":
            if cell.get("outputs"):
                count += 1
                cell.execution_count = count
                for out in cell.outputs:
                    if "execution_count" in out:
                        out["execution_count"] = count
            else:
                cell.execution_count = None
    nbformat.validate(full)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(full, a.out)
    print(f"{len(full.cells)} cells -> {a.out}")


if __name__ == "__main__":
    main()
