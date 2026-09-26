"""yt-dlp downloader for the halved Video-MME split (450 videos). Skips existing files."""
import json, os, subprocess, sys
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)
seen, missing = set(), []
for l in open(os.path.join(OUT, "videomme.jsonl")):
    r = json.loads(l)
    if r["videoID"] in seen:
        continue
    seen.add(r["videoID"])
    dst = os.path.join(OUT, "videos", f"{r['videoID']}.mp4")
    if os.path.exists(dst):
        continue
    rc = subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "mp4[height<=480]/best[height<=480]/best",
                         "-o", dst, r["url"]]).returncode
    if rc != 0:
        missing.append(r["videoID"])
print(f"{len(seen)-len(missing)}/{len(seen)} videos present; {len(missing)} failed:", missing[:10])
