"""Video-MME -> our jsonl. """
import json
import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    import shutil, subprocess
    print("[videomme] pyarrow missing -> installing", flush=True)
    cmds = [[sys.executable, "-m", "pip", "install", "-q", "pyarrow"]]
    if shutil.which("uv"):
        cmds.append(["uv", "pip", "install", "-q", "--python", sys.executable, "pyarrow"])
    for c in cmds:
        if subprocess.run(c).returncode == 0:
            break
    import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT = os.path.dirname(os.path.abspath(__file__))
KEEP_PER_CLASS = 150


def main():
    path = hf_hub_download("lmms-eval/Video-MME", "videomme/test/0000.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    t = pq.read_table(path).to_pylist()
    by_class = {}
    for r in t:
        by_class.setdefault(r["duration"], {}).setdefault(r["video_id"], []).append(r)
    n_out = 0
    with open(os.path.join(OUT, "videomme.jsonl"), "w") as f:
        for dur in sorted(by_class):
            vids = sorted(by_class[dur])[::2][:KEEP_PER_CLASS]  # EVERY OTHER video: immune to any monotone ordering (length, domain) hidden in id order
            for v in vids:
                for r in sorted(by_class[dur][v], key=lambda x: x["question_id"]):
                    opts = list(r["options"])
                    task = (r["question"].strip() + "\n" + "\n".join(opts) +
                            "\nAnswer with the letter of the correct choice. End with the final line, "
                            "for example: 'the answer is (B)'")
                    f.write(json.dumps({
                        "question_id": r["question_id"], "video_id": r["video_id"],
                        "videoID": r["videoID"], "url": r["url"],
                        "video_path": os.path.relpath(
                            os.path.join(OUT, "videos", f"{r['videoID']}.mp4"),
                            os.path.dirname(os.path.dirname(OUT))),
                        "duration_class": dur, "task_text": task,
                        "options": opts, "answer": str(r["answer"]).strip(),
                    }, ensure_ascii=False) + "\n")
                    n_out += 1
    print(f"wrote {n_out} QAs ({KEEP_PER_CLASS} videos x 3 x {len(by_class)} classes)")
    from collections import Counter
    print("classes:", dict(Counter(json.loads(l)["duration_class"]
                                   for l in open(os.path.join(OUT, "videomme.jsonl")))))


if __name__ == "__main__":
    main()
