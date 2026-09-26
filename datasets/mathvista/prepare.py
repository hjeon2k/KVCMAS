"""MathVista testmini -> our jsonl + image files. """
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pyarrow.parquet as pq
except ImportError:
    # Self-heal on fresh venvs: uv-managed venvs often lack the pip module, so try
    # `python -m pip` first, then `uv pip install` against this interpreter.
    import shutil
    import subprocess
    print("[mathvista] pyarrow missing -> installing into this venv", flush=True)
    _cmds = [[sys.executable, "-m", "pip", "install", "-q", "pyarrow"]]
    if shutil.which("uv"):
        _cmds.append(["uv", "pip", "install", "-q", "--python", sys.executable, "pyarrow"])
    for _c in _cmds:
        if subprocess.run(_c).returncode == 0:
            break
    try:
        import pyarrow.parquet as pq
    except ImportError as _e:
        raise SystemExit("could not auto-install pyarrow; run: uv pip install pyarrow") from _e
from huggingface_hub import hf_hub_download
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)


def render_task(q, qtype, choices, unit, precision, answer_type):
    parts = [q.strip()]
    if qtype == "multi_choice" and choices:
        letters = "ABCDEFGH"
        parts.append("Choices:")
        parts.extend(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        # EXAMPLE form, never placeholder form: 'End with: the answer is X' made the
        # 7B model literally output X (P4 smoke).
        parts.append("Answer with the letter of the correct choice. "
                     "End with the final line, for example: 'the answer is (B)'")
    else:
        hint = {"integer": "an integer", "float": "a number", "list": "a list",
                "text": "a short answer"}.get(answer_type or "text", "a short answer")
        if answer_type == "float" and precision is not None:
            hint += f" rounded to {int(precision)} decimal place(s)"
        if unit:
            hint += f", in {unit}"
        parts.append(f"Answer with {hint}. "
                     "End with the final line, for example: 'the answer is 42'")
    return "\n".join(parts)


def main():
    path = hf_hub_download("AI4Math/MathVista", "default/testmini/0000.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    t = pq.read_table(path)
    cols = {c: t.column(c).to_pylist() for c in
            ["pid", "question", "question_type", "answer_type", "choices",
             "precision", "answer", "unit", "decoded_image"]}
    n = len(cols["pid"])
    out = os.path.join(OUT_DIR, "mathvista.jsonl")
    n_img = 0
    with open(out, "w") as f:
        for i in range(n):
            pid = str(cols["pid"][i])
            img_rec = cols["decoded_image"][i]          # {'bytes': ..., 'path': ...}
            img_path = os.path.join(IMG_DIR, f"{pid}.png")
            if not os.path.exists(img_path):
                Image.open(io.BytesIO(img_rec["bytes"])).convert("RGB").save(img_path)
                n_img += 1
            rec = {
                "pid": pid,
                "question": cols["question"][i],
                "image_path": os.path.relpath(img_path, REPO),
                "question_type": cols["question_type"][i],
                "answer_type": cols["answer_type"][i],
                "choices": cols["choices"][i],
                "precision": cols["precision"][i],
                "answer": str(cols["answer"][i]),
                "unit": cols["unit"][i],
                "task_text": render_task(cols["question"][i], cols["question_type"][i],
                                         cols["choices"][i], cols["unit"][i],
                                         cols["precision"][i], cols["answer_type"][i]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {n} records -> {out}   ({n_img} new images in {IMG_DIR})")
    from collections import Counter
    print("question_type:", dict(Counter(cols["question_type"])))
    print("answer_type:  ", dict(Counter(cols["answer_type"])))


if __name__ == "__main__":
    main()
