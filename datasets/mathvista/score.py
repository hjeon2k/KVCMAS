"""MathVista answer extraction + scoring. """
import json
import re
from typing import Optional

# Capture to end of LINE (a "." terminator would truncate decimals: "1.24" -> "1").
_TAIL = re.compile(r"the answer is\s*:?\s*([^\n]+)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _tail(text: str) -> str:
    m = None
    for m in _TAIL.finditer(text or ""):
        pass
    tail = (m.group(1) if m else (text or "")).strip()
    return tail[:-1].rstrip() if tail.endswith(".") else tail   # one sentence-final dot


def extract_answer(text: str, rec: dict) -> Optional[str]:
    tail = _tail(text)
    qt, at = rec.get("question_type"), rec.get("answer_type")
    choices = rec.get("choices") or []
    if qt == "multi_choice" and choices:
        m = re.search(r"\(?([A-H])\)?", tail)
        if m:
            idx = ord(m.group(1)) - 65
            if 0 <= idx < len(choices):
                return str(choices[idx])
        low = tail.lower()
        for c in choices:                      # fuzzy: the choice text itself
            if str(c).lower() in low:
                return str(c)
        return None
    if at == "integer":
        m = _NUM.search(tail.replace(",", ""))
        try:
            return str(int(float(m.group(0)))) if m else None
        except (ValueError, OverflowError):
            return None
    if at == "float":
        m = _NUM.search(tail.replace(",", ""))
        if not m:
            return None
        try:
            prec = int(rec.get("precision") or 0)
            return f"{round(float(m.group(0)), prec):.{prec}f}"
        except (ValueError, OverflowError):
            return None
    return tail.strip().strip(".").strip()


def is_correct(text: str, rec: dict) -> bool:
    pred = extract_answer(text, rec)
    if pred is None:
        return False
    gt = str(rec["answer"]).strip()
    if rec.get("answer_type") == "float":
        try:
            prec = int(rec.get("precision") or 0)
            return abs(float(pred) - float(gt)) < 10 ** (-prec) / 2 + 1e-9
        except (ValueError, OverflowError):
            return False
    if rec.get("answer_type") == "integer":
        try:
            return int(float(pred)) == int(float(gt))
        except (ValueError, OverflowError):
            return False
    return pred.strip().lower() == gt.lower()


if __name__ == "__main__":     # self-test on synthetic outputs
    mc = {"question_type": "multi_choice", "answer_type": "text",
          "choices": ["135°", "140°", "145°", "150°"], "answer": "145°"}
    fl = {"question_type": "free_form", "answer_type": "float", "precision": 1.0,
          "answer": "1.2"}
    it = {"question_type": "free_form", "answer_type": "integer", "answer": "694"}
    cases = [
        ("Let's see... the answer is (C)", mc, True),
        ("I think the answer is C.", mc, True),
        ("the answer is 145°", mc, True),
        ("the answer is (B)", mc, False),
        ("W = ... so the answer is 1.24 J", fl, True),   # rounds to 1.2
        ("the answer is 1.3", fl, False),
        ("total = 204+160+330. the answer is 694", it, True),
        ("the answer is 694 dollars", it, True),
        ("no idea", it, False),
    ]
    for text, rec, want in cases:
        got = is_correct(text, rec)
        status = "ok " if got == want else "FAIL"
        print(f"{status} [{got}] {text!r} -> {extract_answer(text, rec)!r}")
    assert all(is_correct(t, r) == w for t, r, w in cases), "self-test failed"
    print("self-test PASS")
