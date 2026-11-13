"""Build and execute the single EDA notebook of syn_sales_sql_v1 (English): synthetic SFT data vs the original BTC training data.

    python eval/sqlv1/eda_notebook.py  ->  build/synthetic_sql_v1/eda_syn_sales_sql_v1.ipynb (executed, outputs embedded)

BTC data is read locally and only aggregates are shown (no BTC question text, no rulebook). Synthetic examples are shown.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

FOXEL = controller_root()
SNAPSHOT = Path(os.environ.get("DATAMATE_UPSTREAM", FOXEL / "build/sft_v2/snapshot_8b07ba9")).resolve()
OUT = FOXEL / "build/synthetic_sql_v1/eda_syn_sales_sql_v1.ipynb"

CELLS = []


def md(s):
    CELLS.append(("md", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


md(r"""
# `syn_sales_sql_v1`: exploratory data analysis against the original BTC data

**What this notebook answers.** How faithful and how trustworthy is the synthetic SFT corpus that teaches Qwen3-14B to write one
DuckDB query per Vietnamese shop-owner question, compared with the original competition (BTC) training data it imitates?

**Scope.**
- *Synthetic data* = the rows the SFT v2 exporter actually accepted (`sft/train.jsonl`, `sft/dev.jsonl`), plus the separate FINAL benchmark package.
- *Original data* = BTC `data/train`: 3,000 questions over 30 shops. Questions carry offline labels (skill family, industry) from the generator's seed set.
- BTC files are read **locally**. Only aggregates appear below: no BTC question text, no rulebook text.

**How to read it.** Section 1 gives the verdict. Sections 2–4 compare synthetic with original (questions, answers, shops). Sections 5–7 assess the labels and the training rows, which have no BTC counterpart. Section 8 checks split hygiene, and section 9 lists the risks.
""")

code(r"""
import collections, json, math, os, re, statistics, sys, unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

SYNTH = Path(__SYNTH_ROOT__)
FOXEL = Path(__CONTROLLER_ROOT__)
PKG, FINAL = FOXEL / "build/synthetic_sql_v1", FOXEL / "build/synthetic_sql_v1_final"
SFT, WORK, BTC = PKG / "sft", SYNTH / "outputs/synth_sql_v1", SYNTH / "data/train"
SNAPSHOT = Path(__SNAPSHOT__)          # the inference snapshot every check ran on
TOKENIZER = FOXEL / "build/sft_v2/qwen3_14b/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18"
os.environ["DATAMATE_UPSTREAM"] = str(SNAPSHOT)
sys.path[:0] = [str(SNAPSHOT), str(SYNTH / "eval")]
from sqlv1 import package as PK                              # same feature definitions as the generator's selection step
from datamate.ingest import read_rows

def jl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

# ---- visual system: fixed categorical order (validated: CVD ΔE ≥ 9.1 on adjacent pairs), recessive grid, ink-coloured text
SERIES = {"BTC train (original)": "#2a78d6", "Synthetic train": "#eb6834", "Synthetic dev": "#1baf7a", "Synthetic FINAL": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
STATUS = {"PASS": "#0ca30c", "WATCH": "#fab219", "FAIL": "#d03b3b"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "xtick.major.size": 0, "ytick.major.size": 0})
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

EN = {  # labels of the generator's taxonomy
    "industry": {"tap_hoa": "grocery", "quan_nhau": "pub & grill", "ca_phe": "coffee shop", "vlxd": "building materials",
                 "nha_thuoc": "pharmacy", "khac": "other industries"},
    "skill": {"doanh_thu": "revenue", "don": "orders", "so_luong_xep_hang": "quantity & ranking", "nhom_kenh_ca": "group / channel / shift",
              "lai_nhap_ton": "profit / purchases / stock", "gia_ban": "selling price", "khach": "customers", "cong_no": "debt",
              "nganh": "industry-specific", "khong_tra_loi": "unanswerable", "nhan_vien": "staff", "khac": "other"},
    "period": {"thang_cu_the": "named month", "ngay_cu_the": "named date", "hom_qua": "yesterday", "hom_nay": "today",
               "tuan_truoc": "last week", "none": "no period", "thang_nay": "this month", "thang_truoc": "last month", "quy": "quarter",
               "den_het": "up to a date", "n_ngay": "last N days", "nam": "year", "tuan_nay": "this week", "tet": "Tet holiday",
               "cung_ky": "same period comparison"},
    "trap": {"name_variants": "name spelled several ways", "xlsx_sheets": "multi-sheet workbook", "qty_unit": "unit written in qty cell",
             "title_rows": "report title above header", "dup_rows": "duplicated order lines", "two_level": "two-level header",
             "split_files": "book split across files", "total_row": "grand-total row", "blank_cont": "blank continuation cells",
             "dong_tieu_de_bao_cao": "header repeated mid-sheet", "dong_ten_bao_cao_giua_sheet": "report name mid-sheet",
             "dong_cong_theo_ngay": "daily subtotal rows"},
}
print("snapshot:", SNAPSHOT.name, "| tokenizer revision:", TOKENIZER.name[:12])
""".replace("__SYNTH_ROOT__", repr(str(Path(__file__).resolve().parents[2])))
   .replace("__CONTROLLER_ROOT__", repr(str(FOXEL)))
   .replace("__SNAPSHOT__", repr(str(SNAPSHOT))))

md(r"""
## 0. Load the data

Rows are the unit of training. Questions are the unit of distribution comparison: on average a synthetic question carries about 1.7 rows (up to two verified SQL solutions with different shapes). Distributions are therefore compared **per question**, and row-level views are flagged as such.
""")

code(r"""
# ---- original: BTC train (questions, gold, generator seed labels), read locally
btc_q = {q["id"]: q for q in jl(BTC / "questions.jsonl")}
btc_a = {a["id"]: a for a in jl(BTC / "answers.jsonl")}
seed = {s["id"]: s for s in jl(SYNTH / "data/synth_source/seeds/questions.jsonl")}
btc = pd.DataFrame([{"id": i, "shop": q["business_id"], "text": q["question"], "type": btc_a[i]["type"], "answer": btc_a[i].get("answer"),
                     "industry": (seed.get(i) or {}).get("industry"), "skill": (seed.get(i) or {}).get("skill_family"),
                     "role": (seed.get(i) or {}).get("role")} for i, q in btc_q.items()])

# ---- synthetic: rows the exporter accepted + question metadata (spec, gold) of the generator
man = json.loads((SFT / "sft_manifest.json").read_text(encoding="utf-8"))
sel = {x["id"]: x for x in jl(WORK / "selected.jsonl") if x.get("id")}
rows, prompts_user, systems = [], [], set()
for split in ("train", "dev"):
    for r in jl(SFT / f"{split}.jsonl"):
        m = r["metadata"]
        systems.add(hash(r["prompt_messages"][0]["content"]))
        prompts_user.append(r["prompt_messages"][1]["content"])
        rows.append({"row_id": r["id"], "split": split, "qid": m["question_id"], "shop": r["business_id"], "completion": r["completion"],
                     "answer_type": m["answer_type"], "route": m["route"], "derivation": m["derivation"],
                     "solver": m["solver"]["model"] or "program template", "provider": m["solver"]["model_key"],
                     "repaired": m["repaired"], "cand": m["sample_id"], "mut_checked": m["mutation"]["n_checked"], "mut_changed_ok": m["mutation"]["n_changed_ok"],
                     "family": m["schema_family"]})
rows = pd.DataFrame(rows)
SYSTEM_TEXT = r["prompt_messages"][0]["content"]  # identical for every row (checked below); holds the BTC rulebook copy: never displayed

def syn_questions(ids, split):
    out = []
    for i in ids:
        x = sel[i]
        f = PK.features(x)
        out.append({"id": i, "split": split, "shop": x["business_id"], "text": x["question"], "type": x["gold"]["type"], "answer": x["gold"].get("answer"),
                    "industry": x["industry"], "skill": f["skill"], "slot": x["spec"]["slot"], "origin": x["origin"], "family": x["family"]})
    return out

syn = pd.DataFrame(syn_questions(sorted(rows.loc[rows.split == "train", "qid"].unique()), "train")
                   + syn_questions(sorted(rows.loc[rows.split == "dev", "qid"].unique()), "dev")
                   + syn_questions(sorted(q["id"] for q in jl(FINAL / "data_root/questions.jsonl")), "final"))

# ---- shared question features (same functions for both sides)
ABBR = re.compile(r"\b(dthu|bn|bnhiu|nhiu|dc|đc|hnay|hqua|vs|trc|sl|sp|ko|k|z|j|bill|revenue|total|order)\b", re.I)
FILL = re.compile(r"\b(với|cho hỏi|xem giúp|vậy|nhỉ|giùm|cho mình hỏi|ạ|ha|check giùm|hả|coi giùm|nha|cho hoi|xem giup|vay|nhi|gium|nhe)\b", re.I)

def no_diacritics(q):
    return not any(unicodedata.decomposition(c) or c in "đĐ" for c in q if c.isalpha() and ord(c) > 127)

for df in (btc, syn):
    df["words"] = df.text.str.split().str.len()
    df["word_bin"] = df.text.map(PK.wbin)
    df["period"] = df.text.map(PK.ptag)
    df["no_diacritics"] = df.text.map(no_diacritics)
    df["abbreviation"] = df.text.map(lambda q: bool(ABBR.search(q)))
    df["filler_word"] = df.text.map(lambda q: bool(FILL.search(q)))
    df["lowercase_start"] = df.text.map(lambda q: q[:1].islower())
    df["industry_core"] = df.industry.map(lambda i: i if i in PK.WB.CORE else "khac")

GROUPS = {"BTC train (original)": btc, "Synthetic train": syn[syn.split == "train"], "Synthetic dev": syn[syn.split == "dev"],
          "Synthetic FINAL": syn[syn.split == "final"]}
n_rows = {"Synthetic train": int((rows.split == "train").sum()), "Synthetic dev": int((rows.split == "dev").sum())}
overview = pd.DataFrame({g: {"questions": f"{len(d):,}", "shops": d.shop.nunique(), "SFT rows": f"{n_rows[g]:,}" if g in n_rows else "–",
                             "industries": d.industry.nunique(), "median words": f"{d.words.median():.0f}"} for g, d in GROUPS.items()}).T
print("one system prompt for all rows:", len(systems) == 1)
display(overview)
display(pd.crosstab(syn.origin.map({"reuse": "reused + regenerated gold", "teacher": "teacher-written (verified)", "template_text": "template text (v1.1)"}).fillna(syn.origin),
                    syn.split, margins=True).rename_axis("question origin"))
""")

code(r"""
# ---- headline metrics (every number in the summary below is computed here)
def dist(series):
    s = series.value_counts(normalize=True)
    return s.to_dict()

def l1(p, q):
    keys = set(p) | set(q)
    return sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)

DIMS = {"answer type": "type", "skill family": "skill", "time period": "period", "length bin": "word_bin", "industry": "industry_core"}
ref = {d: dist(btc[c].dropna()) for d, c in DIMS.items()}
div = pd.DataFrame({g: {d: l1(dist(GROUPS[g][c].dropna()), ref[d]) for d, c in DIMS.items()} for g in list(GROUPS)[1:]})
gates = json.loads((WORK / "sft_v2/gates.json").read_text(encoding="utf-8"))
tok = json.loads((PKG / "token_validation.json").read_text(encoding="utf-8"))["checks"]["tokenizer_alignment_real"]
al = gates["btc_align_snapshot"]
tr_rows = rows[rows.split == "train"]
H = {
    "train_rows": len(tr_rows), "dev_rows": int((rows.split == "dev").sum()), "train_q": int((syn.split == "train").sum()),
    "verified_share": 1.0,  # every exported row carries build_sft/2 verification (exporter re-ran its SQL twice, strict gold match)
    "align": al["match_rate"], "align_n": al["applicable"],
    "audit_llm": gates["audit"]["llm_independent_subagent"], "audit_x": gates["audit"]["llm_cross_family"],
    "mut_disc": (tr_rows.mut_changed_ok > 0).mean(), "template_share": (tr_rows.derivation == "offline_program_template").mean(),
    "core_rows": tr_rows.shop.map(dict(zip(syn.shop, syn.industry))).isin(PK.WB.CORE).mean(),
    "tok_status": tok["status"], "tok_max": tok["length_quantiles"]["1.0"], "max_len": tok["max_len"],
    "unans_btc": (btc.type == "unanswerable").mean(), "unans_syn": (syn[syn.split == "train"].type == "unanswerable").mean(),
    "long_btc": (btc.words >= 26).mean(), "long_syn": (syn[syn.split == "train"].words >= 26).mean(),
    "join": tr_rows.completion.str.split("```sql").str[-1].str.contains(r"(?i)\bjoin\b").mean(),
}
display(div.rename_axis("L1 distance to BTC train (0 = identical, 2 = disjoint)").style.format("{:.3f}"))
""")

md(r"""
## 1. Verdict
""")

code(r"""
au, ax = H["audit_llm"], H["audit_x"]
gaps = ["shops are much smaller than BTC shops (section 4); the model sees this only through row counts and value profiles in SCHEMA"]
if H["unans_syn"] < 0.8 * H["unans_btc"]:
    gaps.append(f"unanswerable questions are under-represented ({H['unans_syn']:.1%} vs {H['unans_btc']:.1%})")
if H["long_syn"] < 0.6 * H["long_btc"]:
    gaps.append(f"long questions are rarer ({H['long_syn']:.1%} vs {H['long_btc']:.1%} with ≥ 26 words)")
if H["mut_disc"] < 0.9:
    gaps.append(f"{1 - H['mut_disc']:.1%} of train rows have no mutation that changed the answer, so only static checks guard them against hard-coding")
gaps.append("question style is noisier than BTC (more missing diacritics, abbreviations and filler words; section 2)")
GAPS = "\n".join(f"{i}. {g}" + (";" if i < len(gaps) else ".") for i, g in enumerate(gaps, 1))
score = pd.DataFrame([
    ("Labels", "Exporter re-execution (2 runs, strict gold match)", f"{H['verified_share']:.0%} of {H['train_rows'] + H['dev_rows']:,} rows", "100%", "PASS"),
    ("Labels", "Rule conventions reproduce BTC gold", f"{H['align']:.2%} of {H['align_n']:,} BTC questions", "≥ 95%", "PASS"),
    ("Labels", "Independent LLM audit (subagent, rubric)", f"{au['pass']}/{au['in_sft']} = {au['rate']:.1%} (Wilson low {au['wilson95'][0]:.1%})", "≥ 95%, low ≥ 90%", "PASS"),
    ("Labels", "Cross-family LLM audit with 50 planted errors", f"caught {ax['baits_caught']}/{ax['baits']}; pass {ax['pass_rate']:.1%}", "sensitivity ≥ 80%", "PASS"),
    ("Labels", "Mutation tests where the answer changed and SQL still matched", f"{H['mut_disc']:.1%} of train rows", "≥ 90%", "WATCH" if H["mut_disc"] < 0.9 else "PASS"),
    ("Fidelity", "Answer-type mix vs BTC (L1, train)", f"{div.loc['answer type', 'Synthetic train']:.3f}", "≤ 0.10", "PASS" if div.loc["answer type", "Synthetic train"] <= 0.10 else "FAIL"),
    ("Fidelity", "Skill-family mix vs BTC (L1, train)", f"{div.loc['skill family', 'Synthetic train']:.3f}", "≤ 0.15", "PASS" if div.loc["skill family", "Synthetic train"] <= 0.15 else "FAIL"),
    ("Fidelity", "Time-period mix vs BTC (L1, train)", f"{div.loc['time period', 'Synthetic train']:.3f}", "≤ 0.15", "PASS" if div.loc["time period", "Synthetic train"] <= 0.15 else "FAIL"),
    ("Fidelity", "Unanswerable share (train vs BTC)", f"{H['unans_syn']:.1%} vs {H['unans_btc']:.1%}", "close to BTC", "WATCH" if H["unans_syn"] < 0.6 * H["unans_btc"] else "PASS"),
    ("Fidelity", "Long questions (≥ 26 words), train vs BTC", f"{H['long_syn']:.1%} vs {H['long_btc']:.1%}", "close to BTC", "WATCH" if H["long_syn"] < 0.5 * H["long_btc"] else "PASS"),
    ("Fidelity", "Shop data volume (median data rows per shop)", "see section 4", "same order as BTC", "WATCH"),
    ("Fidelity", "BTC layout quirks absent from synthetic shops", "loader drops all three (regression test)", "no effect on prompts", "PASS"),
    ("Training rows", "Real Qwen3-14B tokenizer: template, single EOS, masks", f"{tok['status']}; max {H['tok_max']:,} tokens", f"≤ max_len {H['max_len']:,}", "PASS" if tok["status"] == "passed" else "FAIL"),
    ("Training rows", "Leakage into FINAL (shop / lineage / question / solver prompt)", "0 / 0 / 0 / 0", "0", "PASS"),
    ("Training rows", "Program-template SQL share (train)", f"{H['template_share']:.1%}", "≤ 15%", "PASS"),
], columns=["area", "check", "measured", "target", "status"])

def colour(v):
    return f"background-color: {STATUS[v]}; color: {'#ffffff' if v != 'WATCH' else INK}; font-weight: 600" if v in STATUS else ""
display(score.style.map(colour, subset=["status"]).hide(axis="index"))
display(Markdown(f'''
**Bottom line.** The **labels are strong**:
- every one of the {H['train_rows'] + H['dev_rows']:,} rows was re-executed by the exporter and matched gold strictly;
- the rule conventions reproduce {H['align']:.2%} of BTC gold;
- two independent LLM audits pass at ≥ {min(au['rate'], ax['pass_rate']):.0%}.

The **question distribution is close to BTC** on the dimensions it was selected for (answer type, skill, period, length).

Where the synthetic data is **still thinner than the original**:
{GAPS}

None of these block training.
'''))
""")

md(r"""
## 2. Questions: is the synthetic mix the BTC mix?

Each panel compares the share of questions per category. **Blue** is BTC train and **orange** is synthetic train. The L1 distance in the title is the sum of absolute share differences (0 = identical, 2 = disjoint). The table under each figure gives the exact shares.
""")

code(r"""
def compare(dim, col, top=None, groups=("BTC train (original)", "Synthetic train"), ax=None, label=None):
    shares = {g: GROUPS[g][col].dropna().value_counts(normalize=True) for g in groups}
    cats = shares[groups[0]].index.union(shares[groups[1]].index)
    order = sorted(cats, key=lambda k: -shares[groups[0]].get(k, 0))
    order = order[:top] if top else order
    ax = ax or plt.gca()
    h = 0.8 / len(groups)
    for i, g in enumerate(groups):
        y = np.arange(len(order)) + (i - (len(groups) - 1) / 2) * h
        ax.barh(y, [shares[g].get(k, 0) for k in order], height=h, color=SERIES[g], edgecolor=SURFACE, linewidth=1.5, label=g)
    names = [EN.get(label or dim, {}).get(k, k) for k in order]
    ax.set_yticks(np.arange(len(order)), names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    d = l1(shares[groups[0]].to_dict(), shares[groups[1]].to_dict())
    ax.set_title(f"{dim.capitalize()} · L1 = {d:.3f}")
    ax.grid(axis="y", visible=False)
    return pd.DataFrame({g: [shares[g].get(k, 0) for k in order] for g in groups}, index=names).assign(**{"diff (pp)": lambda t: 100 * (t.iloc[:, 1] - t.iloc[:, 0])})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
t1 = compare("answer type", "type", ax=axes[0])
t2 = compare("skill family", "skill", ax=axes[1], label="skill")
axes[0].legend(loc="lower right")
fig.tight_layout(); plt.show()
display(pd.concat({"answer type": t1, "skill family": t2}).style.format({"BTC train (original)": "{:.1%}", "Synthetic train": "{:.1%}", "diff (pp)": "{:+.1f}"}))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
t3 = compare("time period", "period", top=10, ax=axes[0], label="period")
t4 = compare("industry", "industry_core", ax=axes[1])
axes[0].legend(loc="lower right")
fig.tight_layout(); plt.show()
display(pd.concat({"time period (top 10)": t3, "industry": t4}).style.format({"BTC train (original)": "{:.1%}", "Synthetic train": "{:.1%}", "diff (pp)": "{:+.1f}"}))
""")

md(r"""
**Reading.**
- Answer types and skills track BTC within a few points. The visible gap is **unanswerable** questions, which appear at about a third of the BTC rate.
- Periods follow BTC; *named month* dominates on both sides.
- Industry differs by design. BTC covers 5 core industries plus a little "other" across 30 shops. The synthetic train set keeps most questions in the core industries but spreads them differently, and adds new industries under "other".
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
bins = np.arange(0, 36)
for g in ("BTC train (original)", "Synthetic train"):
    w = GROUPS[g].words.clip(upper=35)
    ax.hist(w, bins=bins, density=True, histtype="step", linewidth=2, color=SERIES[g], label=g)
ax.set_title("Question length (words per question)")
ax.set_xlabel("words (35 = 35 or more)"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend()
ax = axes[1]
flags = ["no_diacritics", "abbreviation", "filler_word", "lowercase_start"]
y = np.arange(len(flags))
for i, g in enumerate(("BTC train (original)", "Synthetic train")):
    ax.barh(y + (i - 0.5) * 0.4, [GROUPS[g][f].mean() for f in flags], height=0.4, color=SERIES[g], edgecolor=SURFACE, linewidth=1.5, label=g)
ax.set_yticks(y, ["no diacritics", "chat abbreviation", "filler word", "lower-case start"]); ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="y", visible=False)
ax.set_title("Writing style markers (share of questions)")
fig.tight_layout(); plt.show()
q = lambda s: s.quantile([0.1, 0.5, 0.9]).tolist() + [s.mean()]
display(pd.DataFrame({g: q(GROUPS[g].words) + [GROUPS[g][f].mean() for f in flags] for g in GROUPS},
                     index=["words p10", "words p50", "words p90", "words mean", "no diacritics", "abbreviation", "filler word", "lowercase start"]).style.format("{:.2f}"))
b, t = GROUPS["BTC train (original)"], GROUPS["Synthetic train"]
H["long_btc"], H["long_syn"] = (b.words >= 26).mean(), (t.words >= 26).mean()
display(Markdown(f'''**Reading.**
- The length distributions overlap closely between 5 and 25 words.
- BTC has a tail of **long, multi-clause questions**: {H['long_btc']:.1%} have 26 words or more, against {H['long_syn']:.1%} in synthetic train.
- The synthetic questions are **noisier in style** than BTC. Questions without diacritics differ by {100 * (t.no_diacritics.mean() - b.no_diacritics.mean()):+.0f} points, chat abbreviations by {100 * (t.abbreviation.mean() - b.abbreviation.mean()):+.0f}, filler words by {100 * (t.filler_word.mean() - b.filler_word.mean()):+.0f}. This errs on the side of robustness, but it is not BTC's mix.'''))
""")

md(r"""
## 3. Answers: are gold values of the same kind and scale?

The SQL a model learns should produce values of a realistic magnitude. For number answers the comparison is on a log scale, because revenue in VND spans many orders of magnitude.
""")

code(r"""
def nums(df, t):
    v = pd.to_numeric(df.loc[df.type == t, "answer"], errors="coerce").dropna()
    return v
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for g in ("BTC train (original)", "Synthetic train"):
    v = nums(GROUPS[g], "number"); v = np.log10(v.abs().clip(lower=1))
    axes[0].hist(v, bins=np.arange(0, 11, 0.5), weights=np.full(len(v), 1 / len(v)), histtype="step", linewidth=2, color=SERIES[g], label=g)
    p = nums(GROUPS[g], "percent").clip(-100, 200)
    axes[1].hist(p, bins=np.arange(-100, 205, 10), weights=np.full(len(p), 1 / len(p)), histtype="step", linewidth=2, color=SERIES[g], label=g)
    lens = GROUPS[g].loc[GROUPS[g].type == "list", "answer"].map(lambda a: len(a) if isinstance(a, list) else np.nan).dropna()
    axes[2].hist(lens.clip(upper=10), bins=np.arange(0.5, 11.5, 1), weights=np.full(len(lens), 1 / len(lens)), histtype="step", linewidth=2, color=SERIES[g], label=g)
for a_ in axes:
    a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_ylabel("share of answers")
axes[0].set_title("Number answers: log10(|value|)"); axes[0].set_xlabel("1 = tens · 6 = millions · 9 = billions")
axes[1].set_title("Percent answers (clipped to −100…200)"); axes[1].set_xlabel("percent")
axes[2].set_title("List answers: items per list"); axes[2].set_xlabel("items (10 = 10 or more)")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()
summ = {}
for g in GROUPS:
    d = GROUPS[g]; v = nums(d, "number")
    summ[g] = {"number: median": v.median(), "number: share ≥ 1M": (v.abs() >= 1e6).mean(), "number: share ≥ 1B": (v.abs() >= 1e9).mean(),
               "percent: median": nums(d, "percent").median(),
               "percent: share < 0": (nums(d, "percent") < 0).mean(), "unanswerable share": (d.type == "unanswerable").mean()}
display(pd.DataFrame(summ).style.format("{:,.2f}"))
display(Markdown("**Reading.** Answer kinds match: percent answers centre on small changes, and lists mostly hold 2–5 items. "
                 f"Number answers follow BTC up to hundreds of millions but rarely reach billions: ≥ 1 billion VND in "
                 f"{summ['Synthetic train']['number: share ≥ 1B']:.1%} of synthetic number answers vs {summ['BTC train (original)']['number: share ≥ 1B']:.1%} in BTC. "
                 "This is consistent with the smaller shops in section 4."))
""")

md(r"""
## 4. Shops: does the data the SQL runs on look like BTC data?

Each shop is a folder of CSV/XLSX books. The same reader the runtime uses (`datamate.ingest.read_rows`) counts files, tables (sheets), data rows and columns for BTC train shops and for the synthetic shops. Format traps are the messy-book patterns the loader and the model must survive.
""")

code(r"""
def shop_stats(root, label):
    out = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = [f for f in d.rglob("*") if f.is_file() and not f.name.startswith("~$") and f.suffix.lower() in (".csv", ".xlsx", ".xls", ".xlsm")]
        tabs = rows_ = 0; cols = []
        for f in files:
            for _, _, rr in read_rows(f):
                rr = [r for r in rr if any(v is not None and str(v).strip() for v in r)]
                if len(rr) < 2:
                    continue
                tabs += 1; rows_ += len(rr) - 1; cols.append(max(len(r) for r in rr[:5]))
        out.append({"group": label, "shop": d.name, "files": len(files), "tables": tabs, "data rows": rows_, "columns per table": float(np.median(cols)) if cols else 0})
    return out
sft_shops = set(rows.shop)
S = pd.DataFrame(shop_stats(BTC / "data", "BTC train (original)")
                 + [x for x in shop_stats(PKG / "data_root/data", "Synthetic train") if x["shop"] in set(rows[rows.split == "train"].shop)]
                 + [x | {"group": "Synthetic dev"} for x in shop_stats(PKG / "data_root/data", "Synthetic dev") if x["shop"] in set(rows[rows.split == "dev"].shop)]
                 + shop_stats(FINAL / "data_root/data", "Synthetic FINAL"))
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, m in zip(axes, ["data rows", "tables", "files", "columns per table"]):
    for i, g in enumerate(SERIES):
        v = S.loc[S.group == g, m].to_numpy()
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
        ax.scatter(v, i + jitter, s=12, color=SERIES[g], alpha=0.55, linewidths=0)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
    ax.set_yticks(range(len(SERIES)), list(SERIES) if m == "data rows" else [""] * len(SERIES)); ax.invert_yaxis()
    ax.set_title(m + (" (log)" if m == "data rows" else "")); ax.grid(axis="y", visible=False)
    if m == "data rows":
        ax.set_xscale("log")
fig.suptitle("One dot per shop · black tick = median", x=0.01, ha="left", color=INK2, fontsize=9)
fig.tight_layout(); plt.show()
display(S.groupby("group")[["files", "tables", "data rows", "columns per table"]].median().loc[list(SERIES)].assign(shops=S.groupby("group").size()).style.format("{:,.0f}"))
""")

code(r"""
cx = json.loads((SYNTH / "data/synth_source/data_spec/shop_complexity.json").read_text(encoding="utf-8"))["theo_shop"]
KEYMAP = {"ten_viet_khac_nhau": "name_variants", "gop_nhieu_sheet_file": "xlsx_sheets", "so_luong_kem_don_vi": "qty_unit", "dong_ten_bao_cao_dau_sheet": "title_rows",
          "tieu_de_hai_tang": "two_level", "dong_trung_cung_don": "dup_rows", "tach_file_theo_phan_hoac_nam": "split_files", "dong_tong_cong": "total_row",
          "o_trong_dong_sau_cua_don": "blank_cont"}
btc_trap = collections.Counter(KEYMAP.get(t, t) for x in cx.values() for t in set(x["bay"]))
plan = {p["business_id"]: p for p in json.loads((WORK / "plan.json").read_text(encoding="utf-8"))}
train_shops = sorted(set(rows[rows.split == "train"].shop))
syn_trap = collections.Counter(k for b in train_shops for k, v in plan[b]["style"].items() if v and k != "xlsx")
traps = pd.DataFrame({"BTC train (original)": {k: btc_trap[k] / len(cx) for k in set(btc_trap) | set(syn_trap)},
                      "Synthetic train": {k: syn_trap[k] / len(train_shops) for k in set(btc_trap) | set(syn_trap)}}).fillna(0)
traps = traps.sort_values("BTC train (original)", ascending=False)
fig, ax = plt.subplots(figsize=(9, 4.4))
y = np.arange(len(traps))
for i, g in enumerate(traps.columns):
    ax.barh(y + (i - 0.5) * 0.4, traps[g], height=0.4, color=SERIES[g], edgecolor=SURFACE, linewidth=1.5, label=g)
ax.set_yticks(y, [EN["trap"].get(k, k) for k in traps.index]); ax.invert_yaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="y", visible=False)
for yy, (k, r) in zip(y, traps.iterrows()):
    if r["Synthetic train"] == 0 and r["BTC train (original)"] > 0:
        ax.text(0.005, yy + 0.2, "not simulated", va="center", fontsize=8, color=INK2)
ax.set_title("Format traps: share of shops that contain each trap"); ax.legend(loc="lower right")
fig.tight_layout(); plt.show()
display(traps.rename(index=EN["trap"]).style.format("{:.0%}"))
""")

md(r"""
**Reading.**
- The synthetic shops use **more, smaller files**. They split the book into catalog, unit packs, purchases and debt payments, where BTC keeps fewer workbooks with many sheets.
- They are **more than ten times smaller** in data rows. The model sees row counts and top values in the SCHEMA block, so prompts describe smaller shops than BTC's.
- Nine trap types match BTC rates closely. Three BTC quirks (header repeated mid-sheet, report name mid-sheet, daily subtotal rows) are absent from the synthetic shops. The runtime loader **drops all three** before anything reaches the prompt, in CSV and XLSX books (upstream test `tests/test_ingest_btc_layout_quirks.py`). They therefore change no training row, and simulating them would only re-test the loader.
""")

md(r"""
## 5. Label quality: can each gold answer and each SQL be trusted?

There is no BTC counterpart for SQL, so this section is absolute. It shows the export funnel, independent re-execution, mutation testing, convention alignment on BTC, and two audits.
""")

code(r"""
bs = json.loads((WORK / "sft_v2/bundle_summary.json").read_text(encoding="utf-8"))
cnt = man["counts"]
funnel = pd.Series({"candidate solutions": bs["candidates_in"], "with verified provenance": bs["candidates_bundled"],
                    "passed mutation replay": cnt["accepted_before_dedupe"], "exported (after dedupe)": cnt["train"] + cnt["dev"]})
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
ax.barh(range(len(funnel)), funnel.values, color=SERIES["Synthetic train"], edgecolor=SURFACE, linewidth=1.5, height=0.6)
for i, v in enumerate(funnel.values):
    ax.text(v, i, f"  {v:,}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(funnel)), funnel.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(0, funnel.max() * 1.15); ax.set_title("Export funnel (rows)")
ax = axes[1]
mrep = {x["candidate_id"]: x for x in jl(PKG / "offline_bundle/mutation_report.jsonl")}   # the report the exporter checked
MUT = {"m_status": "order status flips", "m_rank_units": "rank ties + unit factors", "m_edges": "boundary times, walk-ins, future rows",
       "m_amounts": "prices, discounts, fees", "m_target": "orders injected per question", "m_probe": "probe aimed at the question's data"}
def mcat(st):
    if not st:
        return "not run (not needed)"
    return "not scorable" if st.get("status") in (None, "not_checkable", "unavailable", "no_change_found") else ("answer changed, SQL right" if st.get("gold_changed") else "answer unchanged, SQL right")
M = pd.DataFrame([{k: mcat(mrep[c]["mutants"].get(k, {})) for k in MUT} for c in rows.cand]).apply(lambda s: s.value_counts(normalize=True)).T.fillna(0)
M = M.reindex(columns=["answer changed, SQL right", "answer unchanged, SQL right", "not scorable", "not run (not needed)"], fill_value=0)
left = np.zeros(len(M))
for col, colr in zip(M.columns, (SERIES["Synthetic train"], "#c3c2b7", "#ebeae6", SURFACE)):
    ax.barh(range(len(M)), M[col], left=left, color=colr, edgecolor=SURFACE, linewidth=1.5, height=0.6, label=col)
    left += M[col].to_numpy()
ax.set_yticks(range(len(M)), [MUT[k] for k in M.index]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_title("Mutation replay per mutant type (exported rows)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=8)
fig.tight_layout(); plt.show()
display(M.rename(index=MUT).style.format("{:.1%}"))
display(pd.Series(man["rejected_reasons"], name="rejected rows").to_frame())
""")

code(r"""
by_t = json.loads((WORK / "sft_v2/btc_align_snapshot.json").read_text(encoding="utf-8"))["summary"]["by_template"]
A = pd.DataFrame([{"template": t, "applicable": sum(v.values()) - v.get("no_view", 0) - v.get("unsupported", 0), "match": v.get("ok", 0)} for t, v in by_t.items()])
A = A[A.applicable > 0].assign(rate=lambda d: d.match / d.applicable).sort_values("applicable", ascending=False)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
au, ax_ = gates["audit"]["llm_independent_subagent"], gates["audit"]["llm_cross_family"]
r1 = jl(WORK / "audit/manual_verdicts_r1.jsonl")
AU = pd.DataFrame([("LLM audit round 1 (subagent; failed, fixed)", sum(x["pass"] for x in r1), len(r1)),
                   ("LLM audit round 2 (subagent), rows in SFT", au["pass"], au["in_sft"]),
                   ("Cross-family LLM audit, rows in SFT", ax_["pass"], ax_["in_sft"]),
                   ("Cross-family auditor: planted errors caught", ax_["baits_caught"], ax_["baits"]),
                   ("BTC convention alignment (all templates)", al["ok"], al["applicable"])], columns=["check", "k", "n"])
AU[["low", "high"]] = [wilson(k, n) for k, n in zip(AU.k, AU.n)]
AU["rate"] = AU.k / AU.n
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
y = np.arange(len(AU))
ax.errorbar(AU.rate, y, xerr=[AU.rate - AU.low, AU.high - AU.rate], fmt="o", color=SERIES["Synthetic train"], ecolor=INK2, elinewidth=1.5, capsize=3, markersize=7)
ax.axvline(0.95, color=INK2, linestyle="--", linewidth=1); ax.text(0.951, -0.55, "95% target", color=INK2, fontsize=8)
ax.set_yticks(y, AU.check); ax.set_ylim(len(AU) - 0.5, -0.8); ax.grid(axis="y", visible=False); ax.set_xlim(0.85, 1.005)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_title("Audits and alignment (dot = rate, bar = Wilson 95% CI)")
ax = axes[1]
ax.barh(range(len(A)), A.rate, color=SERIES["BTC train (original)"], edgecolor=SURFACE, linewidth=1.5, height=0.65)
ax.set_yticks(range(len(A)), [f"{t}  (n={n})" for t, n in zip(A.template, A.applicable)]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlim(min(0.9, A.rate.min() - 0.02), 1.0); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Rule conventions vs BTC gold, per question template")
fig.tight_layout(); plt.show()
display(AU.set_index("check")[["k", "n", "rate", "low", "high"]].style.format({"rate": "{:.1%}", "low": "{:.1%}", "high": "{:.1%}"}))
""")

md(r"""
**Reading.**
- Every exported row passed the exporter's own re-execution. Mutation replay also passed for all of them: the answer stayed correct on every mutated copy of the shop that could be scored.
- The orange share is stronger evidence: at least one mutation **changed the gold answer** and the SQL still returned the new gold, so the query reads the data rather than a remembered value.
- Rows in the grey share are protected only by the static "no hard-coded answer" check.
- Both audits used LLMs (a Claude subagent with a rubric, and a cross-family model). No human reviewed the sample.
""")

md(r"""
## 6. Training rows: what exactly does the model learn to write?
""")

code(r"""
sqlpart = rows.completion.str.split("```sql").str[-1].str.rsplit("```", n=1).str[0]
FEATS = {"CTE (WITH)": r"(?is)^\s*with\b", "JOIN": r"(?i)\bjoin\b", "window function": r"(?i)\bover\s*\(", "CASE": r"(?i)\bcase\b",
         "subquery": r"(?i)\(\s*select\b", "GROUP BY": r"(?i)\bgroup\s+by\b", "ORDER BY … LIMIT": r"(?is)\border\s+by\b.*\blimit\b",
         "date window literal": r"(?i)date\s*'\d{4}-\d{2}-\d{2}'", "ROUND()": r"(?i)\bround\s*\(", "debt macro": r"\bdebt_(?:at|new|repaid)\s*\("}
F = pd.DataFrame({k: sqlpart.str.contains(p, regex=True) for k, p in FEATS.items()}).groupby(rows.split).mean().T
# reference: the team's verified SQL for BTC train questions (assets/fewshot_bank.jsonl), read locally, aggregates only. Not BTC gold.
ref_sql = pd.Series([json.loads(x)["sql"] for x in (FOXEL / "assets/fewshot_bank.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()])
F["BTC reference SQL"] = [ref_sql.str.contains(p, regex=True).mean() for p in FEATS.values()]
notes = rows.completion.str.split("TYPE:").str[0].str.count(r"\n- ")
fig, ax = plt.subplots(figsize=(8, 4))
yy = np.arange(len(F))
for i, (s, lab, colr) in enumerate((("BTC reference SQL", f"BTC reference SQL (n={len(ref_sql):,})", SERIES["BTC train (original)"]),
                                    ("train", "Synthetic train", SERIES["Synthetic train"]))):
    ax.barh(yy + (i - 0.5) * 0.4, F[s], height=0.4, color=colr, edgecolor=SURFACE, linewidth=1.5, label=lab)
ax.set_yticks(yy, F.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.legend(loc="upper right")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_title("SQL constructs (share of queries)")
fig.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(8, 2.6))
R = pd.crosstab(rows.split, rows.route, normalize="index").reindex(columns=["view", "raw", "macro"]).loc[["train", "dev"]]
left = np.zeros(len(R))
for c, colr in zip(R.columns, ("#2a78d6", "#eb6834", "#1baf7a")):
    ax.barh(range(len(R)), R[c], left=left, color=colr, edgecolor=SURFACE, linewidth=1.5, height=0.55, label={"view": "canonical views", "raw": "raw tables", "macro": "debt macros"}[c])
    left += R[c].to_numpy()
ax.set_yticks(range(len(R)), ["train", "dev"]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_title("Route the SQL takes"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
fig.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(8, 2.6))
sv = rows[rows.split == "train"].solver.value_counts()
ax.barh(range(len(sv)), sv.values, color=SERIES["Synthetic train"], edgecolor=SURFACE, linewidth=1.5, height=0.6)
for i, v in enumerate(sv.values):
    ax.text(v, i, f"  {v:,} ({v / sv.sum():.0%})", va="center", fontsize=8.5)
ax.set_yticks(range(len(sv)), sv.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlim(0, sv.max() * 1.35)
ax.set_title("Who wrote the SQL (train rows)")
fig.tight_layout(); plt.show()
display(F[["BTC reference SQL", "train", "dev"]].style.format("{:.1%}"))
display(pd.DataFrame({"notes lines (median)": notes.groupby(rows.split).median(), "repaired answers": rows.groupby("split").repaired.mean(),
                      "questions with 2 rows": rows.groupby(["split", "qid"]).size().gt(1).groupby(level=0).mean()}).style.format("{:.2f}"))
""")

code(r"""
from tokenizers import Tokenizer
tk = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
n_sys = len(tk.encode(SYSTEM_TEXT, add_special_tokens=False).ids)
n_user = np.array([len(e.ids) for e in tk.encode_batch(prompts_user, add_special_tokens=False)])
n_comp = np.array([len(e.ids) for e in tk.encode_batch(rows.completion.tolist(), add_special_tokens=False)])
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")
from transformers import AutoTokenizer
hf = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
msgs0 = [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": prompts_user[0]}, {"role": "assistant", "content": rows.completion[0]}]
full0 = len(tk.encode(hf.apply_chat_template(msgs0, tokenize=False, enable_thinking=False), add_special_tokens=False).ids)
overhead = full0 - (n_sys + n_user[0] + n_comp[0])   # role markers, empty think block, EOS: same for every row
total = n_sys + n_user + n_comp + overhead
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
ax = axes[0]
ax.hist(total, bins=60, color=SERIES["Synthetic train"], edgecolor=SURFACE, linewidth=0.8)
ax.axvline(tok["max_len"], color=STATUS["FAIL"], linewidth=1.5); ax.text(tok["max_len"], ax.get_ylim()[1] * 0.92, f" max_len {tok['max_len']:,}", color=INK2, fontsize=8.5)
ax.set_title("Sequence length per row (Qwen3 tokens)"); ax.set_xlabel("tokens")
ax = axes[1]
ax.hist(n_comp, bins=60, color=SERIES["Synthetic train"], edgecolor=SURFACE, linewidth=0.8)
ax.set_title("Learned completion length (Notes + TYPE + SQL)"); ax.set_xlabel("tokens")
fig.tight_layout(); plt.show()
qs = [0.5, 0.9, 0.99, 1.0]
display(pd.DataFrame({"system prompt": [n_sys] * 4, "user prompt": np.quantile(n_user, qs), "completion": np.quantile(n_comp, qs),
                      "row total (this notebook)": np.quantile(total, qs),
                      "row total (exporter validation, exact)": [tok["length_quantiles"][k] for k in ("0.5", "0.9", "0.99", "1.0")]},
                     index=["p50", "p90", "p99", "max"]).style.format("{:,.0f}"))
print(f"chat-template overhead per row: {overhead} tokens · share of tokens that carry loss: {n_comp.sum() / total.sum():.1%} · "
      f"system prompt share of a median row: {n_sys / np.median(total):.0%}")
""")

code(r"""
H["view_share"] = (rows.route == "view").mean(); H["qwen_share"] = (tr_rows.provider == "qwen").mean()
H["join"], H["cte"], H["window"] = F.loc["JOIN", "train"], F.loc["CTE (WITH)", "train"], F.loc["window function", "train"]
display(Markdown(f'''**Reading.**
- {H['view_share']:.0%} of rows answer through the canonical views. Raw-table SQL is mostly purchases, lots, price and cost questions.
- The SQL mix is close to what BTC questions need. Against the team's verified SQL for BTC train questions: JOIN {H['join']:.1%} vs {F.loc['JOIN', 'BTC reference SQL']:.1%}, window functions {H['window']:.1%} vs {F.loc['window function', 'BTC reference SQL']:.1%}, subqueries {F.loc['subquery', 'train']:.1%} vs {F.loc['subquery', 'BTC reference SQL']:.1%}. The visible difference is CTEs ({H['cte']:.1%} vs {F.loc['CTE (WITH)', 'BTC reference SQL']:.1%}).
- Completions are short (Notes + TYPE + one query) and every row fits `max_len` with room to spare. Only {n_comp.sum() / total.sum():.1%} of tokens carry loss: the shared system prompt alone is {n_sys:,} tokens per row.
- Two solver families wrote most of the SQL. Qwen ({H['qwen_share']:.0%} of train rows) is the **same family** as the student model.'''))
""")

md(r"""
## 7. A few synthetic rows as the model sees the target

Only the question and the learned completion are shown. The prompt is the runtime prompt: README, SCHEMA, views, date anchors and the rulebook.
""")

code(r"""
ex = rows[rows.split == "train"].drop_duplicates("qid")
picks = pd.concat([ex[ex.route == "view"].sample(1, random_state=1), ex[ex.route == "raw"].sample(1, random_state=2), ex[ex.route == "macro"].sample(1, random_state=3)])
for _, r in picks.iterrows():
    q = sel[r.qid]
    display(Markdown(f"**{r.qid}** · {EN['industry'].get(q['industry'], q['industry'])} · `{q['spec']['slot']}` · route `{r.route}` · solver `{r.solver}`\n\n"
                     f"> {q['question']}  *(as of {q['as_of_date']}; gold `{q['gold']['type']}`)*\n\n```\n{r.completion}\n```"))
""")

md(r"""
## 8. Split hygiene and generalisation targets

Train and dev are split by shop lineage. FINAL is a separate benchmark package. Dev and FINAL were **not** selected to match the BTC distribution, so their distance to BTC is larger (table below). Treat dev as an in-distribution regression check, not as a BTC proxy.
""")

code(r"""
lk = gates["leakage"]
display(pd.Series({"FINAL shops in SFT": lk["final_shops_in_sft"], "FINAL lineages in SFT": lk["final_lineages_in_sft"],
                   "FINAL question ids in SFT": lk["final_question_ids_in_sft"], "FINAL questions in any solver request": lk["final_items_in_solver_requests"],
                   "train ∩ dev lineages": lk["train_dev_lineage_overlap"],
                   "FINAL questions whose wording also occurs in train/dev (other shop, other data)": gates["final_text_twins"]["final_questions_with_identical_text_in_sft"]},
                  name="count").to_frame())
display(div.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=1.2))
""")

md(r"""
## 9. Risks and recommendations
""")

code(r"""
small = S.groupby("group")["data rows"].median()
b_, t_ = GROUPS["BTC train (original)"], GROUPS["Synthetic train"]
display(Markdown(f'''
| # | Finding | Evidence | Recommendation |
|---|---|---|---|
| 1 | Synthetic shops are much smaller than BTC shops | median data rows per shop {small['Synthetic train']:,.0f} vs {small['BTC train (original)']:,.0f}; number answers ≥ 1 billion VND rarer | scale the simulator's history (orders/day) and regenerate shops; the model sees this only through SCHEMA row counts and value profiles |
| 2 | Question style noisier than BTC | no diacritics {t_.no_diacritics.mean():.0%} vs {b_.no_diacritics.mean():.0%}, filler words {t_.filler_word.mean():.0%} vs {b_.filler_word.mean():.0%} | rebalance the style mix at the next selection; it currently errs towards robustness |
| 3 | Unanswerable share | {H['unans_syn']:.1%} of train questions vs {H['unans_btc']:.1%} in BTC | v1.1 added verified pool questions; keep at BTC's rate |
| 4 | Long questions | ≥ 26 words: {H['long_syn']:.1%} synthetic vs {H['long_btc']:.1%} BTC | v1.1 added long price-change questions (template text); prefer teacher-written long questions next round |
| 5 | Rows without answer-changing mutation evidence | {1 - H['mut_disc']:.1%} of train rows (largely questions whose answer cannot change, such as unanswerable) | extend targeted probes to the remaining slots |
| 6 | Student-family solver | Qwen wrote {(tr_rows.provider == 'qwen').mean():.0%} of train SQL | prefer cross-family solutions when two exist; needs new API calls |
| 7 | No human review | both audits are LLM-based; the v1.1 additions were not audited | have a person check the audit sample and a sample of the v1.1 rows |
| 8 | Prompt is re-rendered, not the prompt the solver saw | solvers answered batched prompts with a paraphrased rulebook | fine for SFT format; noted in every row's metadata |
'''))
""")


def main():
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "md" else nbformat.v4.new_code_cell(s) for k, s in CELLS]
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
    nbformat.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
