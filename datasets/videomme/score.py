"""Video-MME scoring: pure MCQ letter match. """
import re

_TAIL = re.compile(r"the answer is\s*:?\s*([^\n]+)", re.IGNORECASE)
_PAREN = re.compile(r"\(([A-Da-d])\)")
_LEAD = re.compile(r"^\s*([A-Da-d])\b")
_STANDALONE_UPPER = re.compile(r"\b([A-D])\b")
_ONLY_LETTER = re.compile(r"\(?([A-Da-d])\)?[.\s]*")
_OPENS_WITH = re.compile(r"\s*\(?([A-Da-d])\)?\s*[.):\-]")


def _from_tail(tail: str):
    for rx in (_PAREN, _LEAD, _STANDALONE_UPPER):
        m = rx.search(tail)
        if m:
            return m.group(1).upper()
    return None


def extract_answer(text: str, rec: dict):
    text = text or ""
    tails = list(_TAIL.finditer(text))
    if tails:
        return _from_tail(tails[-1].group(1).strip())
    stripped = text.strip()
    m = _ONLY_LETTER.fullmatch(stripped)
    if m:
        return m.group(1).upper()
    m = _OPENS_WITH.match(stripped)
    if m:
        return m.group(1).upper()
    return None


def is_correct(text: str, rec: dict) -> bool:
    p = extract_answer(text, rec)
    return p is not None and p == rec["answer"].strip().upper()


if __name__ == "__main__":
    r = {"answer": "C"}
    cases = [("the answer is (C)", True), ("The answer is C.", True),
             ("the answer is B", False), ("no idea", False),
             ("reasoning...\nthe answer is (c)", True),
             # marked answers with no tail phrase
             ("C", True), ("(C).", True), ("C. 3 apples", True),
             # prose that merely CONTAINS a/b/c/d letters must not score
             ("The shield on the mountain appears in Central Europe.", False),
             ("The output of other agents is as follows:", False),
             ("The key entities of the problem are explained.", False),
             ("", False),
             # a lone lowercase letter inside a sentence is prose, not a choice
             ("the answer is a tree in the corner", False)]
    for t, w in cases:
        assert is_correct(t, r) == w, (t, extract_answer(t, r))
    print("self-test PASS")
