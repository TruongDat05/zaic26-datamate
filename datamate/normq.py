"""Chuẩn hoá câu hỏi trước khi nhận dạng: lỗi gõ telex ('tieenf', 'cuwar', 'dduwowcj') và viết tắt ('bnhiu', 'hqua').

Từ điển chỉ gồm cách viết có trong câu hỏi train (luật 6: không chỉnh theo câu hỏi public). Cách viết khác để model đọc.

Chỉ sửa từ không có trong từ điển (và không nằm trong tên hàng, tên khách của shop), và chỉ khi sửa xong ra đúng một từ đã biết.
"""
from __future__ import annotations

import re

from .clean import norm_key

VOCAB = set("""
a ai an anh ba bac ban bang bao bat ben bi bieu biet binh bo bot bua ca cac cai cao cap cay chai chan chat che chenh chi chiem chieu chiu cho chot chua chung
chuyen co coi con cong cu cua cuoi cung da dang dat dau day den di dia dien dinh dip do doanh doi don dong dot du dua dung duoc duong gan ghi gi gia giam gian giao
gio giong giup goc gom gop gia gium ha hai han hang hay het hien ho hoa hoat hoi hom hon hop huy it ke kem khach khac khi kho khoan khoang khong kien ky la lai lam
lan le len lien liet lo loai loc loi lon luc luon luong luot ly ma mang mat may mo moi mon mot mua muc muoi nam nao nay ne nen ngay ngoai nguoi nha nhan nhap nhat nhau
nhe nhi nhieu nhom nhu nhung no noi nua o oi phai phan pham phi phieu phuc qua quan quay quy ra rieng roi sang sau san sinh so sua suat ta tai tang tat te tet tham than
thang thanh thap the them theo thi thiet thieu tho thoi thu thuoc thuong ti tien tiem tieu tinh to toan toi ton tong tra tram tren tri tro trong truoc trung tu tuan tuoi tung
tuong va van vao vay ve vi vien voi vs xem xep xet xong xuat goi thung tuyp bich binh cuon cuc mieng tam kg gach son thep bia banh keo sua tra mi com nuoc thit ca chien nuong xao hop lon ly dia to
    thoai ngay gio cung ky bua nhat nhom caonhat gioi tiem shop quan nhan size top khong toi hoi may hang don dem ghi hom """.split())

_SLANG_PHRASE = [
    ('\\b(?:bua|hom) qua\\b', 'hom qua'),
    ('\\bhqua\\b', 'hom qua'),
    ('\\bhnay\\b|\\bhum nay\\b|\\bhom ni\\b|\\bbua nay\\b', 'hom nay'),
    ('\\bb(?:ao)? ?nh(?:e?iu|ieu|iu|ui|eiu)\\b|\\bbnh\\b|\\bbn\\b', 'bao nhieu'),
    ('\\b(?:la|het|duoc|thu ve) nhieu\\b(?= *(?:z|vay|a|ha|nhi|$))', lambda m: m.group(0).rsplit(" ", 1)[0] + " bao nhieu"),
    ('\\b(?:thang|th|thg|t) ?(?:nay|ni)\\b', 'thang nay'),
    ('\\b(?:thang|th|thg) (?:truoc|trc|roi)\\b', 'thang truoc'),
    ('\\btuan (?:trc|rui)\\b', 'tuan truoc'),
    ('\\bnam nay (?:tinh )?(?:toi|den)(?: het)? hom nay\\b', 'tu dau nam den hom nay'),
    ('\\b(thang nay|tuan nay) (?:tinh )?(?:toi|den)(?: het)? hom nay\\b', '\\1'),
    ('\\bco nhieu (don|khach)\\b', 'co bao nhieu \\1'),
    ('\\bhang (?:nao )?chay nhat\\b', 'hang ban chay nhat'),
    ('\\bmat hang chay nhat\\b', 'mat hang ban chay nhat'),
    ('\\bbi e\\b|\\be han\\b', 'khong ban duoc'),
    ('\\bdoanh thu bua\\b', 'doanh thu hom'),
    ('\\bngay hom (nay|qua)\\b', 'hom \\1'),
    ('\\btrong\\b(?= tuan| thang| quy| nam)', ' '),
    ('\\bchen\\b', 'tren'),
    ('\\bcac order\\b|\\border\\b', 'don'),
    ('\\bnhiu\\b', 'nhieu'),
    ('\\bnhut\\b|\\bnhứt\\b', 'nhat'),
]
_SLANG_WORD = {'hqua': 'hom qua', 'hnay': 'hom nay', 'bnhiu': 'bao nhieu', 'dthu': 'doanh thu', 'revenue': 'doanh thu', 'dc': 'duoc', 'trc': 'truoc', 'mi': 'may', 'bill': 'don', 'total': 'tong', 'sp': 'san pham', 'ko': 'khong', 'hok': 'khong', 'z': 'vay', 'chong': 'trong'}
_MONTH_WORD = {"mot": 1, "gieng": 1, "hai": 2, "ba": 3, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10, "muoi mot": 11, "muoi hai": 12, "chap": 12}


def _telex(w: str) -> set[str]:
    """Các dạng có thể sau khi bỏ phím telex; thử cả giữ và bỏ phím dấu cuối."""
    a = re.sub(r"([aou])w", r"\1", w)
    a = re.sub(r"w([aou])", r"\1", a)
    a = re.sub(r"aa", "a", a)
    a = re.sub(r"ee", "e", a)
    a = re.sub(r"oo", "o", a)
    a = re.sub(r"dd", "d", a)
    out = {a}
    if len(a) > 2 and a[-1] in "sfrxj":
        out.add(a[:-1])
    if len(w) > 2 and w[-1] in "sfrxj":
        out.add(w[:-1])
    # phím dấu gõ giữa từ: 'nafy' → 'nay', 'thasng' → 'thang'
    for i in range(1, len(a)):
        if a[i] in "sfrxj" and a[i - 1] in "aeiouy":
            out.add(a[:i] + a[i + 1:])
    return out


def _swaps(w: str) -> set[str]:
    return {w[:i] + w[i + 1] + w[i] + w[i + 2:] for i in range(len(w) - 1)}


def _one_edit(w: str) -> set[str]:
    """Thiếu hoặc thừa đúng một chữ: 'bih' → 'binh', 'hangg' → 'hang'."""
    out = {w[:i] + w[i + 1:] for i in range(len(w))}
    for i in range(len(w) + 1):
        out |= {w[:i] + c + w[i:] for c in "abcdeghilmnopqrstuvy"}
    return out


def fix_word(w: str, keep: set[str]) -> str:
    if w in VOCAB or w in keep or not w.isalpha() or len(w) < 3:
        return w
    for cand in (_telex, _swaps, _one_edit):
        if cand is _one_edit and len(w) < 5:
            break
        c = cand(w)
        hits = c & VOCAB or (c & keep if cand is _telex else set())  # tên riêng của shop chỉ sửa theo telex
        if len(hits) == 1:
            return hits.pop()
        if hits:
            break
    return w


_TONE_MID = re.compile(r"[aeiouy][fxj][a-z]|aa|oo|dd|[aou]w")


def has_unread_typo(q: str, keep: set[str]) -> bool:
    """Còn từ gõ lỗi không đọc được: đoán tiếp dễ sai kỳ, sai đối tượng, nên để bỏ trống."""
    return any(w.isalpha() and len(w) <= 8 and w not in VOCAB and w not in keep and _TONE_MID.search(w) for w in q.split())


def normalize(question: str, keep: frozenset | set = frozenset()) -> str:
    """Câu không dấu, chữ thường, đã sửa telex và viết tắt. keep: từ trong tên riêng của shop, không sửa."""
    q = norm_key(question)
    # viết tắt đã biết sửa trước: 'dthu', 'hqua' cách 'thu', 'qua' đúng một chữ
    q = " ".join(_SLANG_WORD.get(w, w) if w not in keep else w for w in q.split())
    q = " ".join(fix_word(w, keep) for w in q.split())
    for pat, rep in _SLANG_PHRASE:
        q = re.sub(pat, rep, q)
    q = " ".join(_SLANG_WORD.get(w, w) if w not in keep else w for w in q.split())
    return q


def shop_words(*name_lists) -> set[str]:
    return {w for names in name_lists for n in names for w in norm_key(n).split()}
