"""Đổi tên cột, tên bảng của shop train rồi đo lớp tính toán, xem có bỏ trống thay vì trả lời sai không: python eval/stress_schema.py"""
import collections, copy, json, re, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
NEW = {"order":"Số CT bán","date":"Ngày ghi sổ","status":"Tình hình","item":"Diễn giải hàng","item_code":"Mã mục","amount":"Giá trị dòng","qty":"Lượng","qty_base":"Lượng lẻ","cust":"Người mua","cust_code":"Mã người mua","price":"Giá mỗi đv","group":"Ngành hàng","cost":"Vốn 1 đv","unit":"Đv ghi","staff":"Người lập","pay":"Trả theo","channel":"Đường bán","time":"Lúc tạo","cust_group":"Hạng mua","note":"Lưu ý thêm","big_unit":"Đv to","factor":"Đổi ra","rx":"Cần toa","returned_qty":"Lượng hoàn"}
def camel(c):
    from datamate.clean import norm_key
    return "".join(w.capitalize() for w in norm_key(c).split())
def rename_all(biz, mode):
    from datamate import shop as sm
    from datamate.clean import norm_col
    hints = {}
    for role, hs in sm._NAME_HINTS.items():
        for h in hs: hints.setdefault(h, role)
    readme = biz.readme
    used = {}
    for t in list(biz.tables):
        df = biz.tables[t]; ren = {}
        for c in df.columns:
            if c.startswith("_"): continue
            base, _, suf = c.partition("__")
            if mode == "style":
                nb = camel(base)
            else:
                role = hints.get(norm_col(base))
                nb = NEW.get(role, base) if role else base
            ren[c] = nb + ("__"+suf if suf else "")
            if not suf and nb != base: readme = readme.replace(f"**{base}**", f"**{nb}**")
        biz.tables[t] = df.rename(columns=ren)
    if mode == "syn_table":
        for i, t in enumerate(list(biz.tables)):
            readme = readme.replace(t, f"Bang{i}")
            biz.tables[f"Bang{i}"] = biz.tables.pop(t)
    biz.readme = readme
    return biz
MODES = ["goc","style","syn","syn_table"]
def work(arg):
    from datamate import dm, loader, shop as sm, templates, templates_ext
    from datamate.timeutil import LunarCalendar
    from score import match
    b, qs, gold = arg
    dm.cal = LunarCalendar(ROOT/"data/train/calendar.csv")
    templates_ext.RULEBOOK = (ROOT/"data/train/rulebook.md").read_text(encoding="utf-8")
    out = []
    for mode in MODES:
        biz = loader.load_cached(ROOT/"data/train/data"/b, ROOT/"outputs/cache")
        if mode != "goc": biz = rename_all(biz, mode)
        try: s = sm.build(biz)
        except Exception as e: s = None
        for q in qs:
            try: a = templates.answer(q["question"],q["as_of_date"],s) if s is not None else None
            except Exception: a = None
            ok = None if a is None else match({"id":q["id"],**a},gold[q["id"]])>0
            out.append((b,mode,q["question"],ok,a.get("template") if a else None,str(a.get("answer"))[:30] if a else None,str(gold[q["id"]].get("answer"))[:30]))
    return out
if __name__=="__main__":
    gold={json.loads(l)["id"]:json.loads(l) for l in open(ROOT/"data/train/answers.jsonl",encoding="utf-8")}
    by=collections.defaultdict(list)
    for l in open(ROOT/"data/train/questions.jsonl",encoding="utf-8"):
        q=json.loads(l); by[q["business_id"]].append(q)
    with ProcessPoolExecutor(6) as ex:
        rows=[r for rs in ex.map(work,[(b,v,gold) for b,v in by.items()]) for r in rs]
    hold=set((ROOT/"eval/holdout_businesses.txt").read_text().split())
    for name,f in (("train trừ holdout",lambda b:b not in hold),("shop holdout",lambda b:b in hold)):
        print(name)
        for m in MODES:
            x=[r for r in rows if r[1]==m and f(r[0])]
            a=[r for r in x if r[3] is not None]; ok=sum(r[3] for r in a)
            print(f"  {m:10} câu {len(x)} trả lời {len(a)} ({len(a)/len(x):.0%}) đúng {ok} sai {len(a)-ok} tỉ lệ đúng {ok/max(len(a),1):.1%} PA {(2*ok-len(a))/len(x):+.3f}")
    bad=[r for r in rows if r[3] is False and r[1]!="goc" and r[0] not in hold]
    print(collections.Counter((r[1],r[4]) for r in bad).most_common(25))
    print(collections.Counter((r[1],r[0]) for r in bad).most_common(15))
    import random; random.seed(1); random.shuffle(bad)
    for r in bad[:20]: print(r[1],r[0][-3:],r[4],'|',r[2][:70],'|',r[5],'| gold',r[6])
