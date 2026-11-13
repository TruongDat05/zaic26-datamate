"""Làm nhiễu câu train (bỏ dấu, viết tắt, telex, chữ đệm, đảo kỳ) rồi đo lớp tính toán: python eval/stress_noise.py"""
import collections, json, random, re, sys, threading, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
TELEX = {"à":"af","á":"as","ả":"ar","ã":"ax","ạ":"aj","è":"ef","é":"es","ì":"if","í":"is","ò":"of","ó":"os","ù":"uf","ú":"us","ơ":"ow","ư":"uw","ă":"aw","â":"aa","ê":"ee","ô":"oo","đ":"dd"}
SWAP = [("bao nhiêu","bnhiu"),("doanh thu","dthu"),("tháng này","th này"),("được","đc"),("tuần trước","tuần trc"),("hôm qua","hqua"),("hôm nay","hnay"),("cửa hàng","tiệm"),("khách hàng","khách"),("đơn hàng","đh"),("là","la")]
TAIL = [" vậy"," nhỉ"," z"," ạ"," giúp mình"," nha"," hen"," rứa"," thế"," ta"]
HEAD = ["cho mình hỏi ","e ơi ","xem giúp ","anh ơi cho hỏi ","check giùm "]
def noise(q, kind, rnd):
    if kind == "bo_dau":
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", q) if not unicodedata.combining(c)).replace("đ","d").replace("Đ","D")
    if kind == "viet_tat":
        for a,b in SWAP:
            if a in q.lower() and rnd.random()<0.8: q = re.sub(a,b,q,flags=re.I)
        return q
    if kind == "telex":
        w = q.split()
        idx = [i for i,x in enumerate(w) if any(c in TELEX for c in x.lower())]
        for i in rnd.sample(idx, min(2,len(idx))):
            for c,t in TELEX.items():
                if c in w[i].lower(): w[i] = w[i].lower().replace(c,t); break
        return " ".join(w)
    if kind == "dem":
        return rnd.choice(HEAD) + q.rstrip("?. ") + rnd.choice(TAIL)
    if kind == "dao_ky":  # đưa kỳ lên đầu câu
        m = re.search(r"(?:trong )?(tháng \d{1,2}(?:/| năm )\d{4}|tuần trước|tháng này|hôm qua|hôm nay|quý \d(?:/| năm )\d{4})", q, re.I)
        if not m: return q
        rest = (q[:m.start()] + q[m.end():]).strip(" ,?")
        return f"{m[1]}, {rest}?"
    return q
KINDS = ["goc","bo_dau","viet_tat","telex","dem","dao_ky"]


# ---------------------------------------------------------------- lớp văn phong cho câu sinh (model_bakeoff T4, gen_data)
# Câu gốc (đủ dấu, kỳ rõ) → câu kiểu chủ tiệm. Chạy cục bộ: không gửi câu train đi đâu, chỉ chép vector style của câu mining.
VIET_TAT = {"dthu": "doanh thu", "revenue": "doanh thu", "bnhiu": "bao nhiêu", "bn": "bao nhiêu", "nhiu": "bao nhiêu", "đc": "được", "dc": "được",
            "hnay": "hôm nay", "hqua": "hôm qua", "bữa": "hôm", "vs": "với", "trc": "trước", "z": "vậy", "ko": "không", "k": "không", "hok": "không",
            "j": "gì", "total": "tổng", "bill": "hoá đơn|hóa đơn", "order": "đơn hàng", "sp": "sản phẩm|mặt hàng", "best seller": "bán chạy nhất"}
DEM_DAU = {"cho hỏi", "xem giúp", "cho mình hỏi", "check giùm", "coi giùm", "xem giùm", "check giúp", "tính giúp", "anh ơi", "chị ơi", "em ơi"}
_TONE = {"̀": "f", "́": "s", "̉": "r", "̃": "x", "̣": "j"}
_HAT = {"̂": "", "̆": "w", "̛": "w"}  # â ê ô → aa ee oo; ă ơ ư → aw ow uw


def telex(word):
    """Gõ telex quên chuyển: 'tiền' → 'tieenf', 'được' → 'dduwowcj' (phím dấu thanh ở cuối từ, như câu train)."""
    import unicodedata
    out, tone = [], ""
    for ch in unicodedata.normalize("NFD", word.replace("đ", "dd").replace("Đ", "Dd")):
        if ch in _TONE: tone = _TONE[ch]
        elif ch in _HAT and out: out[-1] += _HAT[ch] or out[-1]
        else: out.append(ch)
    return "".join(out) + tone


def _giu(w, nxt):
    """Không đụng số, ngày, kỳ: từ có chữ số và từ đứng ngay trước nó (tháng 3/2025, quý 2/2025, ngày 5/3/2025)."""
    return any(c.isdigit() for c in w + nxt)


def style_question(q, vec, rnd, w_tat, w_dem):
    """Viết lại câu gốc theo vector style của một câu mining: viết tắt/từ tiếng Anh, từ đệm, 1–2 lỗi telex, bỏ dấu, chữ thường đầu câu,
    bỏ '?'. w_tat, w_dem: tần suất từ viết tắt, từ đệm trong style_profile.json (thay khi từ của câu mining không hợp câu này)."""
    s = q.strip().rstrip("?.! ")

    def swap(a):
        nonlocal s
        m = re.search(rf"(?<!\w)(?:{VIET_TAT[a]})(?!\w)", s, re.I)
        if m:
            s = s[:m.start()] + a + s[m.end():]
        return bool(m)
    if sum(swap(a) for a in vec["abbreviations"] if a in VIET_TAT) < len(vec["abbreviations"]):  # từ của câu mining không hợp câu này
        opts = [a for a in w_tat if a in VIET_TAT and re.search(rf"(?<!\w)(?:{VIET_TAT[a]})(?!\w)", s, re.I)]
        if opts:
            swap(rnd.choices(opts, [w_tat[a] for a in opts])[0])
    used = []
    for f in vec["fillers"]:
        if len(used) == 2 or any(f in u for u in used):  # 'giùm' đã nằm trong 'check giùm'
            continue
        f = f if f in w_dem else rnd.choices(list(w_dem), list(w_dem.values()))[0]
        if f in ("với", "giùm") and s.lower().endswith(("không", "ko", "chưa")):  # 'có … không với' nghe lạ
            f = "vậy"
        s = f"{f} {s[:1].lower()}{s[1:]}" if f in DEM_DAU else f"{s} {f}"
        used.append(f)
    w = s.split()
    idx = [i for i, x in enumerate(w) if x.isalpha() and x[:1].islower() and telex(x) != x and not _giu(x, w[i + 1] if i + 1 < len(w) else "")]
    for i in rnd.sample(idx, min(len(vec["telex"]), len(idx))):
        w[i] = telex(w[i])
    s = " ".join(w)
    if vec["no_diacritic"]:
        s = noise(s, "bo_dau", rnd)
    s = (s[:1].lower() if vec["lower_start"] else s[:1].upper()) + s[1:]
    return s if vec["no_qmark"] else s + "?"
def work(arg):
    from datamate import dm, loader, shop as sm, templates, templates_ext
    from datamate.timeutil import LunarCalendar
    from score import match
    b, qs, gold = arg
    dm.cal = LunarCalendar(ROOT/"data/train/calendar.csv")
    templates_ext.RULEBOOK = (ROOT/"data/train/rulebook.md").read_text(encoding="utf-8")
    s = sm.build(loader.load_cached(ROOT/"data/train/data"/b, ROOT/"outputs/cache"))
    out = []
    for q in qs:
        for k in KINDS:
            rnd = random.Random(q["id"] + k)
            t = q["question"] if k=="goc" else noise(q["question"],k,rnd)
            try: a = templates.answer(t,q["as_of_date"],s) if s is not None else None
            except Exception: a = None
            ok = None if a is None else match({"id":q["id"],**a},gold[q["id"]])>0
            out.append((b,k,t,ok,a.get("template") if a else None,str(a.get("answer"))[:30] if a else None,str(gold[q["id"]].get("answer"))[:30]))
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
        for k in KINDS:
            x=[r for r in rows if r[1]==k and f(r[0])]
            a=[r for r in x if r[3] is not None]; ok=sum(r[3] for r in a)
            print(f"  {k:9} câu {len(x)} trả lời {len(a)} ({len(a)/len(x):.0%}) đúng {ok} sai {len(a)-ok} tỉ lệ đúng {ok/max(len(a),1):.1%} PA {(2*ok-len(a))/len(x):+.3f}")
    bad=[r for r in rows if r[3] is False and r[1]!="goc" and r[0] not in hold]
    print(collections.Counter((r[1],r[4]) for r in bad).most_common(14))
    random.seed(1); random.shuffle(bad)
    for r in bad[:14]: print(r[1],r[4],'|',r[2][:90],'|',r[5],'| gold',r[6])



class VanPhong:
    """Mỗi câu chép vector style của một câu mining ngẫu nhiên (giữ phân bố đồng thời các nét, không rút từng nét riêng); trong k câu
    mining rút ngẫu nhiên, lấy câu làm tỉ lệ cộng dồn (đo bằng build_synth_source.style_of) sát style_profile.json nhất."""
    NET = {"khong_dau": "no_diacritic", "co_viet_tat": "abbreviations", "co_tu_dem": "fillers", "go_telex": "telex",
           "chu_thuong_dau": "lower_start", "khong_dau_hoi": "no_qmark"}

    def __init__(self, seed=0, k=4, known=frozenset(), src=ROOT / "data/synth_source"):
        self.prof = json.loads((src / "style/style_profile.json").read_text(encoding="utf-8"))
        with open(src / "seeds/questions.jsonl", encoding="utf-8") as f:
            self.vecs = [r["style"] for r in map(json.loads, f) if r["role"] == "mining"]  # chỉ lấy vector style, không lấy câu
        self.rnd, self.k, self.known, self.lock = random.Random(seed), k, known, threading.Lock()
        self.dem, self.n = collections.Counter(), 0

    @classmethod
    def net(cls, st):
        return {t: bool(st[k]) for t, k in cls.NET.items()}

    def render(self, q):
        from build_synth_source import style_of
        with self.lock:
            best = None
            for v in self.rnd.sample(self.vecs, self.k):
                s = style_question(q, v, self.rnd, self.prof["viet_tat"], self.prof["tu_dem"])
                net = self.net(style_of(s, self.known))
                d = sum((self.dem[t] + net[t] - self.prof["ti_le"][t] * (self.n + 1)) ** 2 for t in self.NET)
                if best is None or d < best[0]:
                    best = (d, s, net)
            self.n += 1
            self.dem.update({t: int(x) for t, x in best[2].items()})
            return best[1]
