"""Agent: câu hỏi → LLM viết code pandas → chạy trong sandbox → sửa lỗi → bỏ phiếu giữa nhiều lần sinh.

Luật chấm phạt câu sai (−1) nặng hơn bỏ trống (0), nên chỉ nộp đáp án khi đủ số phiếu đồng ý;
còn lại abstain.

Các lượt bỏ phiếu đều decode greedy (temperature 0) để BTC chạy lại ra cùng kết quả (luật 8);
độ đa dạng đến từ biến thể prompt (bộ ví dụ few-shot khác, cách dặn khác), không từ sampling.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from . import dm, formatter, plan, repair, rules, templates, templates_ext, verify
from .llm import extract_code
from .normq import normalize
from .sandbox import run_code
from .timeutil import LunarCalendar

PROMPT_DIR = Path(__file__).parent / "prompts"


def system_prompt(rulebook: str) -> str:
    return (PROMPT_DIR / "system.md").read_text(encoding="utf-8").replace("{rulebook}", rulebook)


def business_block(biz) -> str:
    return (f"# DỮ LIỆU CỬA HÀNG `{biz.business_id}`\n\n## README\n{biz.readme.strip()}\n\n"
            f"## Các bảng trong T\n{biz.profile()}")


def examples_block(examples, code_example=None) -> str:
    out = ""
    if examples:
        lines = ["## Ví dụ câu hỏi tương tự ở cửa hàng khác (chỉ để tham khảo kiểu và format đáp án; số liệu khác)"]
        for q, a in examples:
            shown = {k: a[k] for k in ("type", "answer") if k in a}
            lines.append(f"- \"{q['question']}\" → {json.dumps(shown, ensure_ascii=False)[:300]}")
        out = "\n".join(lines) + "\n\n"
    if code_example:
        q, code = code_example
        out += ("## Code đã chấm đúng cho một câu tương tự ở cửa hàng KHÁC (bảng, cột khác: chỉ học cách áp luật, "
                f"không chép tên bảng/cột)\nCâu hỏi: \"{q['question']}\"\n```python\n{code.strip()}\n```\n\n")
    return out


def question_block(q: dict, hint: str = "") -> str:
    fixed = normalize(q["question"])
    # chữ thường không dấu, đã sửa lỗi gõ và viết tắt: giúp đọc đúng câu viết tắt; tên riêng vẫn lấy ở câu gốc
    return (f"# CÂU HỎI\nHôm nay (TODAY): {q['as_of_date']}\nCâu hỏi: {q['question']}\n"
            f"Câu đã sửa lỗi gõ (không dấu): {fixed}\n\n"
            + (hint.strip() + "\n" if hint else "")
            + "Viết code Python trong một khối ```python```, gán ANSWER.")


# Mỗi biến thể là một lượt bỏ phiếu. fewshot_offset: bỏ qua bao nhiêu ví dụ giống nhất (lượt sau thấy bộ ví dụ khác).
DEFAULT_VARIANTS = [
    {"fewshot_offset": 0, "code_rank": 0, "hint": ""},
    {"fewshot_offset": 4, "code_rank": 1,
     "hint": "Trước khi viết code, liệt kê bằng comment: số đo cần tính, kỳ thời gian (start–end), "
             "đơn nào được tính, bảng và cột sẽ dùng, kiểu đáp án."},
    {"fewshot_offset": 8, "code_rank": 2,
     "hint": "Kiểm tra lại bằng comment các luật mặc định của rulebook áp dụng cho câu này "
             "(lọc trạng thái đơn, quy đổi đơn vị, làm tròn, thứ tự khi bằng nhau) rồi mới viết code."},
]

# Đường bổ sung khi hai phiếu đầu chưa đồng ý, mỗi đường khác nhau ở model, cách dặn hoặc ngôn ngữ để lỗi bớt trùng nhau.
ESCALATE_PATHS = {
    "p3": DEFAULT_VARIANTS[2],
    "quote": {"fewshot_offset": 0, "code_rank": 1,
              "hint": "Trước khi viết code, chép nguyên văn vào comment các dòng rulebook áp dụng cho câu này (kể cả định nghĩa khái niệm "
                      "trong câu hỏi), rồi viết code làm đúng từng dòng đã chép."},
    "sql": {"fewshot_offset": 4, "code_rank": 0,
            "hint": "Câu này tính bằng SQL: `df = sql(\"SELECT ... FROM \\\"tên bảng\\\" ...\")` trả DataFrame, tên bảng và cột như trong T, đặt trong ngoặc kép; "
                    "ngày lưu dạng chuỗi 'YYYY-MM-DD HH:MM:SS'. Sau đó gán ANSWER bằng hàm của dm."},
    "think": {"fewshot_offset": 0, "code_rank": 0, "hint": "", "thinking": True},
    "m2_a": {"fewshot_offset": 0, "code_rank": 0, "hint": "", "model": 1},
    "m2_b": {**DEFAULT_VARIANTS[1], "model": 1},
}


def budget_paths(n_questions: int, cfg: dict) -> dict:
    """Số đường bổ sung theo số câu, không theo đồng hồ, để BTC chạy lại vẫn cùng kế hoạch (luật 8)."""
    b = cfg.get("budget") or {}
    names = list(cfg.get("escalate_paths") or [])
    calls = b.get("target_s", 14400) * b.get("calls_per_s", 0.5) / max(n_questions, 1)
    spare = calls - b.get("base_calls", 1.3)
    guard = bool(cfg.get("template_guard")) and spare >= b.get("guard_calls", 2.5)
    if guard:
        spare -= b.get("guard_calls", 2.5)
    # đường bổ sung chỉ chạy ở câu chưa đồng ý, mỗi đường tốn khoảng path_calls lượt gọi trên một câu trung bình
    keep = max(0, min(len(names), int(spare / b.get("path_calls", 0.6))))
    return {"paths": names[:keep], "template_guard": guard}


class Agent:
    def __init__(self, llm, cfg: dict, rulebook: str, calendar_csv: Path, fewshot=None, vote_llms=None):
        self.llm = llm
        self.vote_llms = vote_llms or [llm]  # mỗi lượt bỏ phiếu một model khác họ thì các phiếu bớt sai giống nhau
        self.cfg = cfg
        self.system = system_prompt(rulebook)
        templates_ext.RULEBOOK = rulebook
        self.calendar_csv = calendar_csv
        if dm.cal is None and Path(calendar_csv).exists():
            dm.cal = LunarCalendar(calendar_csv)  # lớp tính toán cần lịch âm cho câu về Tết
        self.fewshot = fewshot
        self.rules = rules.parse(rulebook) if cfg.get("rule_hints", True) else []

    def _attempt(self, messages: list[dict], biz_pickle: Path, q: dict, temperature: float, seed: int, log: dict, llm=None, biz=None, thinking=False, path=""):
        msgs = list(messages)
        llm = llm or self.llm
        extra = {"thinking": True} if thinking else {}
        for turn in range(1 + self.cfg.get("max_fixes", 2)):
            text = llm.chat(msgs, temperature=temperature, seed=seed + turn, **extra)
            log["llm_calls"] += 1
            code = extract_code(text)
            step = {"path": path, "seed": seed, "turn": turn, "response": text}  # path: phiếu nào, để eval/show_trace.py dựng lại luồng
            log["steps"].append(step)
            if code is None:
                step["error_code"] = "NO_CODE_BLOCK"
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Chưa thấy khối ```python``` hoàn chỉnh (có thể câu trả lời bị cắt vì quá dài). "
                                                     "Viết lại đúng một khối code ngắn, ít comment, gán ANSWER."}]
                continue
            res = run_code(code, biz_pickle, self.calendar_csv, q["as_of_date"], self.cfg.get("exec_timeout_s", 60))
            step["result"] = res
            if res.get("ok"):
                ans = res["answer"]
                final = verify.round_money(q["question"], formatter.finalize(q["id"], ans))
                if final["type"] == "abstain" and ans.get("type") != "abstain":
                    step["error_code"] = "FORMAT_INVALID"
                    err = f"Đáp án sai format: {json.dumps(ans, ensure_ascii=False)[:300]}"
                else:
                    issues = verify.check(q["question"], final)
                    if not issues and not verify.zero_suspect(q["question"], final):
                        return final
                    if issues:
                        step["error_code"] = issues[0].code
                        step["verify"] = [i.code for i in issues]
                        err = "Đáp án chạy được nhưng chưa đúng yêu cầu câu hỏi:\n" + "\n".join(f"- {i.message}" for i in issues)
                    else:
                        step["error_code"] = "ZERO_RESULT"
                        err = ("Kết quả bằng 0, gần như chắc chắn do điều kiện lọc không khớp dòng nào. Kiểm tra: tên trong "
                               "câu hỏi so với giá trị thật của cột (dm.match_one, các cột __ten, __ho_ten, __loai), kỳ "
                               "thời gian, cột đúng chưa. Nếu shop không ghi thông tin này thì dùng dm.unanswerable(...).")
            else:
                step["error_code"] = "TIMEOUT" if res.get("timeout") else "EXECUTION_FAILED"
                err = res.get("error", "lỗi không rõ")
                if self.cfg.get("typed_repair", True):
                    kind, tip = repair.classify(err, biz)
                    step["repair_type"] = kind
                    err = f"{err[:2200]}\n\nLoại lỗi: {kind}. Cần sửa: {tip}"
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Code lỗi:\n```\n{err[:2500]}\n```\nSửa code (vẫn một khối ```python```, gán ANSWER)."}]
        return None

    def _plan(self, q: dict, shop, log: dict):
        try:
            slots = plan.parse(self.llm, q["question"], q["as_of_date"], shop, self.cfg.get("seed", 42))
            log["llm_calls"] += 1
            sentence = plan.render(slots, shop)
            log["plan"] = {"slots": slots, "sentence": sentence}
            ans = templates.answer(sentence, q["as_of_date"], shop) if sentence else None
        except Exception as e:  # noqa: BLE001 — đường này lỗi thì còn đường pandas
            log["plan_error"] = repr(e)[:300]
            return None
        if ans is None:
            return None
        log["plan"]["template"] = ans.get("template")
        return formatter.finalize(q["id"], ans)

    def _judge(self, q: dict, biz, ans: dict, log: dict) -> bool:
        """Một lượt đọc lại độc lập trước khi nộp đáp án chỉ do model viết. Không đọc được phán quyết thì không chặn."""
        code = next((st.get("response") for st in reversed(log["steps"]) if st.get("result", {}).get("ok")), "")
        hint = rules.block(self.rules, q["question"], self.cfg.get("rule_hints_k", 5)) if self.rules else ""
        user = (f"Hôm nay: {q['as_of_date']}\nCâu hỏi: {q['question']}\n{hint}\nCode đã chạy:\n{(code or '')[:5000]}\n"
                f"Đáp án: {json.dumps(ans, ensure_ascii=False)[:800]}\n"
                "Code có làm đúng câu hỏi và đúng luật không: đúng kỳ, đúng đối tượng, đúng lọc đơn, đúng số đo? Trả JSON.")
        schema = {"type": "object", "properties": {"dung": {"type": "boolean"}, "ly_do": {"type": "string"}}, "required": ["dung"]}
        try:
            v = self.llm.chat_json([{"role": "system", "content": "Bạn soát lỗi code phân tích dữ liệu. Chỉ kết luận sai khi chỉ ra được lỗi cụ thể."},
                                    {"role": "user", "content": user}], schema, seed=self.cfg.get("seed", 42))
            log["llm_calls"] += 1
        except Exception as e:  # noqa: BLE001
            log["judge_error"] = repr(e)[:300]
            return True
        log["judge"] = v
        return not (isinstance(v, dict) and v.get("dung") is False)

    def _messages(self, q: dict, biz, variant: dict) -> list[dict]:
        examples, code_example = [], None
        if self.fewshot:
            k, off = self.cfg.get("fewshot_k", 4), variant.get("fewshot_offset", 0)
            examples = self.fewshot.similar(q["question"], q["business_id"], k + off)[off:off + k]
            if self.cfg.get("code_examples", True):
                # mỗi lượt bỏ phiếu thấy một code mẫu khác (giống nhất, nhì, ba…) để các phiếu bớt phụ thuộc nhau
                coded = self.fewshot.similar_code(q["question"], q["business_id"], variant.get("code_rank", 0) + 1)
                code_example = coded[-1] if coded else None
        hints = ""
        if self.rules:
            hints = rules.block(self.rules, q["question"], self.cfg.get("rule_hints_k", 5)) + rules.concept_block(q["question"])
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": business_block(biz) + "\n\n" + examples_block(examples, code_example)
             + hints + question_block(q, variant.get("hint", ""))},
        ]

    def _vote_path(self, name: str, order: int, q: dict, biz, biz_pickle: Path, log: dict):
        """Một phiếu của đường bổ sung; model thứ hai chưa bật thì bỏ qua đường đó."""
        path = ESCALATE_PATHS[name]
        if path.get("model", 0) >= len(self.vote_llms):
            return None
        ans = self._attempt(self._messages(q, biz, path), biz_pickle, q, self.cfg.get("temperature", 0.0),
                            self.cfg.get("seed", 42) + 1000 * (order + 10), log, self.vote_llms[path.get("model", 0)], biz, path.get("thinking", False), name)
        log.setdefault("votes", []).append({"path": name, "answer": ans})
        return ans

    def _guard(self, key: str, q: dict, biz, biz_pickle: Path, log: dict) -> bool:
        """Đáp án đã có còn đứng vững không: bị bác khi đủ số phiếu độc lập cùng ra một đáp án khác."""
        against: Counter = Counter()
        for i, name in enumerate(self.cfg.get("guard_paths") or []):
            ans = self._vote_path(name, i, q, biz, biz_pickle, log)
            if ans is None or ans.get("type") == "abstain":
                continue
            other = formatter.canonical(ans)
            if other != key:
                against[other] += 1
                if against[other] >= self.cfg.get("guard_against", 2):
                    return False
        return True

    def _escalate(self, names, counter, cands, q, biz, biz_pickle, log, final):
        """Hai phiếu đầu chưa đồng ý: thêm đường cho tới khi một đáp án đủ phiếu và dẫn đủ xa."""
        accept = self.cfg.get("accept") or {}
        for j, name in enumerate(names):
            ans = self._vote_path(name, j, q, biz, biz_pickle, log)
            if ans is None:
                continue
            key = formatter.canonical(ans)
            cands.setdefault(key, ans)
            counter[key] += 1
            top = counter.most_common(2)
            lead = top[0][1] - (top[1][1] if len(top) > 1 else 0)
            if top[0][1] >= accept.get("min_votes", 3) and lead >= accept.get("min_lead", 2):
                winner = cands[top[0][0]]
                if winner["type"] in (self.cfg.get("drop_types") or []) or winner["type"] == "abstain":
                    return final
                log["decision"] = "escalated"
                return winner
        return final

    def _need(self, ans: dict) -> int:
        """Số phiếu cần để nộp; có thể đặt riêng theo kiểu đáp án (agent.min_agree_by_type)."""
        return (self.cfg.get("min_agree_by_type") or {}).get(ans["type"], self.cfg.get("min_agree", 2))

    def answer(self, q: dict, biz, biz_pickle: Path, shop=None) -> tuple[dict, dict]:
        t0 = time.time()
        log = {"id": q["id"], "question": q["question"], "as_of_date": q["as_of_date"],
               "business_id": q["business_id"], "steps": [], "candidates": [], "llm_calls": 0}
        # dạng câu cơ bản: lớp tính toán, không gọi model; độ chính xác đo bằng eval/template_eval.py
        if shop is not None and self.cfg.get("templates", True):
            try:
                ans = templates.answer(q["question"], q["as_of_date"], shop)
            except ZeroDivisionError:
                ans = None
            except Exception as e:  # lỗi của lớp tính toán không được làm hỏng câu: chuyển sang model
                log["template_error"] = repr(e)[:300]
                ans = None
            if ans is not None:
                final = formatter.finalize(q["id"], ans)
                if self.cfg.get("template_guard") and not self._guard(formatter.canonical(final), q, biz, biz_pickle, log):
                    log.update(decision="template_rejected", template=ans["template"], candidates=[final], final={"id": q["id"], "type": "abstain"}, elapsed_s=round(time.time() - t0, 2))
                    return log["final"], log
                log.update(decision="template", template=ans["template"], final=final, elapsed_s=round(time.time() - t0, 2))
                return final, log
        # model đọc câu hỏi thành JSON, code tính: đường độc lập với code model viết
        plan_ans = None
        if shop is not None and self.cfg.get("plan", False):
            plan_ans = self._plan(q, shop, log)
            if plan_ans is not None and not self.cfg.get("dual_path", True):
                log.update(decision="plan", final=plan_ans, elapsed_s=round(time.time() - t0, 2))
                return plan_ans, log
        variants = self.cfg.get("variants") or DEFAULT_VARIANTS
        votes = self.cfg.get("votes", len(variants))
        min_need = min([self.cfg.get("min_agree", 2), *(self.cfg.get("min_agree_by_type") or {}).values()])
        temperature = self.cfg.get("temperature", 0.0)
        base_seed = self.cfg.get("seed", 42)
        cands: dict[str, dict] = {}
        counter: Counter = Counter()
        for i in range(votes):
            variant = variants[i % len(variants)]
            messages = self._messages(q, biz, variant)
            ans = self._attempt(messages, biz_pickle, q, temperature, base_seed + 1000 * i, log,
                                self.vote_llms[i % len(self.vote_llms)], biz, path=f"base{i}")
            log["candidates"].append(ans)
            log.setdefault("votes", []).append({"path": f"base{i}", "answer": ans})
            # đường JSON và code pandas độc lập mà ra cùng đáp án: đủ để nộp
            if ans is not None and plan_ans is not None and formatter.canonical(ans) == formatter.canonical(plan_ans):
                log.update(decision="plan_agree", final=plan_ans, elapsed_s=round(time.time() - t0, 2))
                return plan_ans, log
            if ans is not None:
                key = formatter.canonical(ans)
                cands.setdefault(key, ans)
                counter[key] += 1
                if counter[key] >= self._need(ans) and self.cfg.get("early_stop", True):
                    break
            # không còn đủ lượt để một đáp án đạt ngưỡng → dừng sớm
            remaining = votes - i - 1
            if (max(counter.values()) if counter else 0) + remaining < min_need:
                break
        final = {"id": q["id"], "type": "abstain"}
        if not counter:
            log["decision"] = "no_candidate"
        else:
            key, n = counter.most_common(1)[0]
            if n >= self._need(cands[key]):
                # kiểu đáp án đã đo ra sai nhiều hơn đúng thì bỏ trống (agent.drop_types); vẫn giữ phiếu trong log để đo lại
                if cands[key]["type"] in (self.cfg.get("drop_types") or []):
                    log["decision"] = "dropped_type"
                elif plan_ans is not None:  # hai phiếu pandas trùng nhau nhưng khác đường JSON: một trong hai sai
                    log["decision"] = "plan_conflict"
                elif self.cfg.get("judge", False) and not self._judge(q, biz, cands[key], log):
                    log["decision"] = "judge_rejected"
                else:
                    final = cands[key]
                    log["decision"] = "consensus"
            else:
                log["decision"] = "no_consensus"
        names = self.cfg.get("escalate_paths") or []
        if self.cfg.get("collect_all"):  # thu mọi phiếu để eval/replay_votes.py thử ngưỡng mà không chạy lại model; quyết định giữ theo hai phiếu đầu
            for j, name in enumerate(names):
                self._vote_path(name, j, q, biz, biz_pickle, log)
        elif final["type"] == "abstain" and log["decision"] in ("no_candidate", "no_consensus"):
            final = self._escalate(names, counter, cands, q, biz, biz_pickle, log, final)
        elif final["type"] != "abstain" and self.cfg.get("consensus_guard") and not self._guard(formatter.canonical(final), q, biz, biz_pickle, log):
            log["decision"], final = "consensus_rejected", {"id": q["id"], "type": "abstain"}
        log["final"] = final
        log["elapsed_s"] = round(time.time() - t0, 2)
        return final, log
