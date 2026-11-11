"""Check the pipeline against the track rules by reading its code and configuration, without running the model.

    python scripts/rule_check.py [--data data] [--repos Qwen/Qwen3.5-9B ...] [--json OUT.json]

Prints three groups and exits 1 when there is any LỖI, so it can gate a submission:
  LỖI      a rule is broken for sure
  CẦN XEM  the script cannot decide; a person reads the item
  ĐẠT      checked, nothing found

Rules (docs/challenge.md): 2 model size, release date, licence; 3 offline; 4 no hard-coding on question or shop ids;
5 writes only to /result or /tmp; 6 train data only, never the test sets (here: no wording learnt from the public test);
7 the 6 h budget, answers kept on a stop; 8 reproducibility and logs.
The inference code is every module datamate/predict.py imports, followed through the package. With --repos the script asks the
Hugging Face API for release dates (network); everything else is offline. data/ is read when present (rule 6 vocabulary check)."""
import argparse
import ast
import json
import re
import struct
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "datamate"
LIMIT = 15_000_000_000
CUTOFF = "2026-10-01"  # docs/challenge.md rule 1 (the TA slides say 2026-10-26): the earlier date is the safe one
LICENCES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}
SHOP_ID = re.compile(r"\b(?:biz|dev)_\d{3}\b")
# English function words: prompt prose, not wording a question uses
ENGLISH = set("a an and are as at be by for from if in is it its not of on or than that the this to with without".split())
QUESTION_ID = re.compile(r"\bq_[0-9a-f]{12}\b")
out = {"LỖI": [], "CẦN XEM": [], "ĐẠT": []}


def report(kind, rule, msg):
    out[kind].append(f"[{rule}] {msg}")


def read(p):
    p = ROOT / p
    return p.read_text(encoding="utf-8") if p.exists() else ""


def jsonl(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] if p.exists() else []


# ---- code model ------------------------------------------------------------------------------------------------------------------
def inference_modules():
    """datamate/predict.py and every package module it imports, transitively."""
    seen, todo = set(), ["predict"]
    while todo:
        m = todo.pop()
        if m in seen or not (PKG / f"{m}.py").exists():
            continue
        seen.add(m)
        for node in ast.walk(ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.level == 1 or (node.module or "").startswith("datamate")):
                mod = (node.module or "").removeprefix("datamate").lstrip(".")
                todo += [mod.split(".")[0]] if mod else [a.name for a in node.names]
    return sorted(PKG / f"{m}.py" for m in seen)


def docstring_nodes(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def cli_nodes(tree):
    """String constants inside argparse add_argument(...) calls: command-line help and defaults, not pipeline logic."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            ids.update(id(n) for n in ast.walk(node) if isinstance(n, ast.Constant))
    return ids


def strings(path):
    """(line, text, kind) of every string constant in a Python file; kind is 'doc' (docstring), 'cli' (argparse) or 'code'."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs, cli = docstring_nodes(tree), cli_nodes(tree)
    return [(n.lineno, n.value, "doc" if id(n) in docs else "cli" if id(n) in cli else "code")
            for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def comments(path):
    return [(i, line.split("#", 1)[1]) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if "#" in line]


def identifier_words():
    """Words used as names anywhere in the repo's Python (date, type, mode...): English/technical vocabulary, not wording."""
    words = set()
    for f in list(PKG.glob("*.py")) + list((ROOT / "analysis").glob("*.py")) + list((ROOT / "training").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            name = getattr(n, "id", None) or getattr(n, "attr", None) or getattr(n, "arg", None) or (getattr(n, "name", None) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else None)
            if isinstance(name, str):
                words.update(w.lower() for w in re.split(r"_|(?<=[a-z])(?=[A-Z])", name) if w)
    return words


def fold(s):
    """Lowercase ASCII words with diacritics stripped, đ as d (the folding the pipeline matches questions with)."""
    s = unicodedata.normalize("NFKD", s.replace("đ", "d").replace("Đ", "D")).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z]+", s)


# ---- rule 2: models --------------------------------------------------------------------------------------------------------------
def safetensors_params(folder):
    total = 0
    for f in sorted(Path(folder).glob("*.safetensors")):
        with f.open("rb") as fh:
            raw = fh.read(8)
            if len(raw) < 8 or raw.startswith(b"versi"):  # a Git LFS pointer, not weights
                return None
            head = json.loads(fh.read(struct.unpack("<Q", raw)[0]))
        for k, v in head.items():
            if k != "__metadata__":
                n = 1
                for d in v["shape"]:
                    n *= d
                total += n
    return total or None


def manifest_adapter(m):
    """The adapters/ folder the manifest declares as the submitted LoRA ("adapters/NAME (in the image: ...)"), or None."""
    hit = re.match(r"\s*(adapters/[\w.-]+)", m.get("lora_adapter") or "")
    return hit.group(1) if hit else None


def rule2(repos):
    m = json.loads(read("docker/model_manifest.json") or "{}")
    if not m:
        report("LỖI", "2", "thiếu docker/model_manifest.json: không khai báo được model dùng lúc thi")
        return
    total = m.get("total_parameters_with_adapter") or 0
    report("ĐẠT" if 0 < total <= LIMIT else "LỖI", "2", f"tổng tham số khai báo {total / 1e9:.3f}B (giới hạn 15B)")
    extra = (m.get("inference_models", 1) - 1) + m.get("embeddings_or_rerankers", 0)
    report("LỖI" if extra else "ĐẠT", "2", f"số model lúc thi: {m.get('inference_models')} LLM, {m.get('embeddings_or_rerankers')} embedding/reranker")
    day = (m.get("revision_date_utc") or "")[:10]
    report("ĐẠT" if day and day < CUTOFF else "LỖI", "2", f"{m.get('repo_id')}@{(m.get('revision') or '')[:8]}: revision {day or '?'} (trước {CUTOFF})")
    lic = (m.get("license") or "").lower()
    report("ĐẠT" if lic in LICENCES else "CẦN XEM", "2", f"giấy phép model gốc: {m.get('license')}")
    for name, folder, declared in (("model gốc", ROOT / "models/Qwen3.5-9B", m.get("safetensors_total_parameters")),
                                   ("LoRA", ROOT / (manifest_adapter(m) or "adapters/?"), m.get("lora_adapter_parameters"))):
        n = safetensors_params(folder) if folder.exists() else None
        if n is None:
            report("CẦN XEM", "2", f"{name}: chưa có trọng số thật ở {folder.relative_to(ROOT)} (chưa tải hoặc chưa git lfs pull), chưa đếm lại được {declared:,} tham số khai báo")
        else:
            report("ĐẠT" if n == declared else "LỖI", "2", f"{name}: đếm được {n:,} tham số, khai báo {declared:,}")
    docker = read("docker/Dockerfile")
    for t in m.get("training_only_models", []):
        created = (t.get("created") or "")[:10]
        report("ĐẠT" if created and created < CUTOFF else "LỖI", "2", f"model chỉ dùng sinh dữ liệu {t['repo_id']}: tạo {created}, giấy phép {t.get('license')}")
        name = t["repo_id"].split("/")[-1]
        report("LỖI" if name in docker else "ĐẠT", "2", f"{name} không nằm trong image")
    pred = read("docker/predict.sh")
    models = set(re.findall(r"--model\s+(\S+)", pred))
    report("ĐẠT" if len(models) == 1 else "LỖI", "2", f"predict.sh nạp đúng một model: {sorted(models)}")
    for r in repos:
        import urllib.request
        try:
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{r}", timeout=20) as fh:
                d = json.load(fh)
            created, lic = (d.get("createdAt") or "")[:10], (d.get("cardData") or {}).get("license")
            report("ĐẠT" if created and created < CUTOFF else "LỖI", "2", f"{r} trên Hugging Face: tạo {created}, giấy phép {lic}")
        except Exception as e:  # noqa: BLE001
            report("CẦN XEM", "2", f"{r}: không đọc được thông tin từ Hugging Face ({e})")


# ---- rule 3: offline -------------------------------------------------------------------------------------------------------------
def rule3(code):
    hits = []
    for f in code:
        text = f.read_text(encoding="utf-8")
        for mt in re.finditer(r"https?://[\w.-]+|^\s*(?:import|from)\s+(?:requests|httpx|aiohttp|socket|urllib\.request)\b", text, re.M):
            hits.append(f"{f.name}:{text[:mt.start()].count(chr(10)) + 1} {mt.group(0).strip()}")
    report("LỖI" if hits else "ĐẠT", "3", f"code inference gọi mạng: {hits or 'không có'}")
    for name, text in (("docker/Dockerfile", read("docker/Dockerfile")), ("scripts/flags.sh", read("scripts/flags.sh"))):
        miss = [v for v in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1") if v not in text]
        report("LỖI" if miss else "ĐẠT", "3", f"{name} đặt chế độ offline" + (f": thiếu {miss}" if miss else ""))
    pipe = read("datamate/pipeline.py")
    report("ĐẠT" if "enable_external_access = false" in pipe else "LỖI", "3", "SQL của model không đọc được tệp ngoài hay mạng (DuckDB enable_external_access = false)")


# ---- rule 4: no hard-coding ------------------------------------------------------------------------------------------------------
def rule4(code):
    in_code, in_notes, qids = [], [], []
    for f in code:
        for line, s, kind in strings(f):
            for m in SHOP_ID.findall(s):
                (in_code if kind == "code" else in_notes).append(f"{f.name}:{line} {m}")
            qids += [f"{f.name}:{line} {m}" for m in QUESTION_ID.findall(s)]
        in_notes += [f"{f.name}:{i} {m}" for i, c in comments(f) for m in SHOP_ID.findall(c)]
    report("LỖI" if in_code else "ĐẠT", "4", f"mã shop trong chuỗi code inference: {in_code or 'không có'}")
    report("LỖI" if qids else "ĐẠT", "4", f"mã câu hỏi trong code inference: {qids or 'không có'}")
    report("CẦN XEM" if in_notes else "ĐẠT", "4", f"mã shop chỉ trong comment/docstring (ghi lý do, logic phải chung): {len(in_notes)} chỗ" + (f", ví dụ {in_notes[:4]}" if in_notes else ""))
    pred = read("docker/predict.sh") + read("scripts/flags.sh")
    bad = [f for f in ("--views-from", "--ids", "--limit") if f in pred]
    report("LỖI" if bad else "ĐẠT", "4", f"bản chạy private không dùng lớp dữ liệu lưu sẵn hay danh sách id: {bad or 'đúng'}")
    report("LỖI" if re.search(r"COPY\s+\S*configs", read("docker/Dockerfile")) else "ĐẠT", "4", "image không chứa configs/ (view và lớp công nợ soạn trên public)")


# ---- rule 5: writes ------------------------------------------------------------------------------------------------------------
def rule5():
    pred = read("docker/predict.sh")
    outs = re.findall(r"--(?:output|log-dir|merge-into)\s+\"?(\S+?)\"?\s", pred)
    ok = outs and all(o.startswith("$OUT") for o in outs) and 'RESULT_DIR:-/result' in pred
    report("ĐẠT" if ok else "LỖI", "5", f"predict.sh chỉ ghi vào /result: {outs}")
    report("LỖI" if re.search(r">\s*\"?\$DATA", pred) else "ĐẠT", "5", "predict.sh không ghi vào /data")


# ---- rule 6: training data only, nothing learnt from the public test -------------------------------------------------------------
def corpus(split):
    """Word and bigram counts of a split's questions, and of its rulebook + shop READMEs."""
    q, docs = Counter(), Counter()
    for r in jsonl(split / "questions.jsonl"):
        t = fold(r["question"])
        q.update(set(t) | {f"{a} {b}" for a, b in zip(t, t[1:])})
    for p in [split / "rulebook.md"] + sorted((split / "data").glob("*/*.md")):
        if p.exists():
            t = fold(p.read_text(encoding="utf-8", errors="ignore"))
            docs.update(set(t) | {f"{a} {b}" for a, b in zip(t, t[1:])})
    return q, docs


def rule6(code, data):
    uses, cli = [], []
    for f in code:
        for line, s, kind in strings(f):
            if "public_test" in s:
                {"code": uses, "cli": cli}.get(kind, []).append(f"{f.name}:{line}")
    report("LỖI" if uses else "ĐẠT", "6", f"code inference dựng đường dẫn tới public_test: {uses or 'không'}")
    if cli:
        report("CẦN XEM", "6", f"public_test chỉ là giá trị mặc định của dòng lệnh khi chạy tay (Docker phải truyền đường dẫn /data): {cli}")
    train_code = [p.relative_to(ROOT).as_posix() for p in (ROOT / "training").glob("*.py") if "public_test" in p.read_text(encoding="utf-8")]
    report("LỖI" if train_code else "ĐẠT", "6", f"code train đọc public_test: {train_code or 'không'}")
    docker = read("docker/Dockerfile")
    report("LỖI" if re.search(r"COPY\s+\S*(?:data|public_test)\b", docker) else "ĐẠT", "6", "image không chứa dữ liệu thi")
    report("CẦN XEM" if "scoring" in {f.stem for f in code} else "ĐẠT", "6", "code inference không import scoring.py (hàm chấm chỉ dùng khi đánh giá)")

    train, pub = data / "train", data / "public_test"
    if not (train / "questions.jsonl").exists() or not (pub / "questions.jsonl").exists():
        report("CẦN XEM", "6", f"không có {train} và {pub}: chưa kiểm tra ngân hàng ví dụ và từ ngữ học từ public")
        return
    tq = {r["id"]: r["business_id"] for r in jsonl(train / "questions.jsonl")}
    bank = jsonl(ROOT / "assets/fewshot_bank.jsonl")
    outside = [r["id"] for r in bank if r.get("id") not in tq]
    report("LỖI" if outside else "ĐẠT", "6", f"ngân hàng ví dụ {len(bank)} câu, câu không thuộc train: {len(outside)}")
    hold = set(read("splits/holdout_shops.txt").split())
    leak = sum(tq.get(r.get("id")) in hold for r in bank)
    report("CẦN XEM" if leak else "ĐẠT", "6", f"ngân hàng ví dụ có {leak} câu của shop holdout (điểm holdout lạc quan, không phải vi phạm)")

    pub_norm = {" ".join(fold(r["question"])) for r in jsonl(pub / "questions.jsonl")}
    whole = [f"{f.name}:{line}" for f in code for line, s, kind in strings(f) if kind == "code" and len(fold(s)) >= 3 and " ".join(fold(s)) in pub_norm]
    report("LỖI" if whole else "ĐẠT", "6", f"chép nguyên câu hỏi public vào code: {whole or 'không có'}")

    tq_c, td_c = corpus(train)
    pq_c, pd_c = corpus(pub)
    names = identifier_words()
    hits = defaultdict(set)
    for f in code:
        for line, s, kind in strings(f):
            if kind != "code":
                continue
            for part in re.split(r"[|()\[\]{}?*+^$\\\n.,;:!\"'`<>=/]+", s):  # regex alternatives and prompt phrases
                t = fold(part)
                for w in t:
                    if len(w) >= 2 and w not in names and w not in ENGLISH and not tq_c[w] and not td_c[w] and (pq_c[w] or pd_c[w]):
                        hits[w].add(f"{f.name}:{line}")
                for a, b in zip(t, t[1:]):
                    bg = f"{a} {b}"
                    if not tq_c[bg] and not td_c[bg] and (pq_c[bg] or pd_c[bg]) and not ({a, b} & ENGLISH) and not (a in names and b in names):
                        hits[bg].add(f"{f.name}:{line}")
    if not hits:
        report("ĐẠT", "6", "từ ngữ trong code chỉ có ở public (không có ở câu hỏi, rulebook, README của train): không có")
        return
    from_q = sorted((k for k in hits if pq_c[k]), key=lambda k: -pq_c[k])
    from_docs = sorted(k for k in hits if not pq_c[k])
    if from_q:
        report("CẦN XEM", "6", f"{len(from_q)} từ/cụm trong code chỉ thấy ở CÂU HỎI public, không có ở train (dễ bị coi là chỉnh theo tập test, nên gỡ hoặc giải trình): "
               + "; ".join(f"'{k}' (public {pq_c[k]} câu) {sorted(hits[k])[:3]}" for k in from_q))
    if from_docs:
        report("CẦN XEM", "6", f"{len(from_docs)} từ/cụm trong code chỉ thấy ở RULEBOOK/README public, không có ở train (luật được đọc lúc chạy, nhưng viết cứng vào code là lấy từ tập test): "
               + "; ".join(f"'{k}' {sorted(hits[k])[:2]}" for k in from_docs))


# ---- rules 7 and 8: time, reproducibility, logs ---------------------------------------------------------------------------------
def rule7_8():
    pred, flags = read("docker/predict.sh"), read("scripts/flags.sh")
    t = re.search(r"TIMEOUT_SECONDS:-(\d+)", pred)
    t = int(t.group(1)) if t else None
    report("ĐẠT" if t and t <= 6 * 3600 - 300 else "LỖI", "7", f"mỗi tiến trình tự dừng sau {t}s (giới hạn 6 giờ gồm cả nạp model)")
    report("ĐẠT" if "--merge-into" in pred else "LỖI", "7", "submission.jsonl được ghi lại sau mỗi shop (dừng giữa chừng vẫn giữ câu đã trả lời)")
    report("ĐẠT" if "--deadline" in pred else "CẦN XEM", "7", "có --deadline: chậm tiến độ thì bỏ mẫu thinking")
    m = json.loads(read("docker/model_manifest.json") or "{}")
    if not m.get("gpu_validated"):
        report("CẦN XEM", "7", "chưa đo thời gian trên 2 × RTX 5090 (manifest gpu_validated=false): chạy image thật trước hạn nộp")
    report("ĐẠT" if "PYTHONHASHSEED=0" in flags else "LỖI", "8", "PYTHONHASHSEED=0")
    seeded = "seed=seed" in read("datamate/llm.py") and "seed=a.seed" in read("datamate/pipeline.py")
    report("ĐẠT" if seeded else "LỖI", "8", "engine vLLM và mọi lượt lấy mẫu đều có seed")
    temp = re.search(r'"--temperature",\s*type=float,\s*default=([\d.]+)', read("datamate/predict.py"))
    if temp and float(temp.group(1)) > 0:
        report("CẦN XEM", "8", f"lấy mẫu ở nhiệt độ {temp.group(1)}: seed không bảo đảm giống hệt trên GPU khác; chạy image 2 lần, độ lệch PA phải dưới 0.03")
    nondet = [f"{p}:{i}" for p in ("datamate/views.py", "datamate/debts.py") for i, line in enumerate(read(p).splitlines(), 1) if "any_value(" in line]
    report("CẦN XEM" if nondet else "ĐẠT", "8", f"lớp dữ liệu dùng any_value() (giá trị tùy ý khi một đơn ghi khác nhau, có thể đổi giữa hai lần chạy): {nondet or 'không có'}")
    report("ĐẠT" if "--log-dir" in pred and "fill-logs" in pred else "LỖI", "8", "ghi log từng câu vào /result/logs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="thư mục chứa train/ và public_test/ (cho kiểm tra luật 6)")
    ap.add_argument("--repos", nargs="*", default=[], help="hỏi Hugging Face ngày phát hành và giấy phép (cần mạng)")
    ap.add_argument("--json", type=Path, help="ghi kết quả ra tệp JSON")
    a = ap.parse_args()
    code = inference_modules()
    report("ĐẠT", "-", f"code inference ({len(code)} tệp): {', '.join(f.name for f in code)}")
    rule2(a.repos)
    rule3(code)
    rule4(code)
    rule5()
    rule6(code, a.data)
    rule7_8()
    for kind in ("LỖI", "CẦN XEM", "ĐẠT"):
        print(f"\n## {kind}: {len(out[kind])}")
        for line in out[kind]:
            print("-", line)
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if out["LỖI"] else 0


if __name__ == "__main__":
    sys.exit(main())
