"""Queue an army-of-loras workflow on a running ComfyUI and wait for the output.

    python scripts/run_prompt.py "isometric diorama of a cozy wizard library"
    python scripts/run_prompt.py --family h3 --seed 7 "16-bit pixel art video of a fox running"
    python scripts/run_prompt.py --batch demo/prompts.txt

Prints the routing report and the saved file paths. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_workflows import build  # noqa: E402


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def run(server: str, prompt: str, family: str, seed: int, prefix: str, timeout: float = 3600) -> dict:
    wf = build(family, prompt, seed=seed, filename_prefix=prefix)
    t0 = time.time()
    pid = post(f"{server}/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
    while time.time() - t0 < timeout:
        try:
            hist = get(f"{server}/history/{pid}")
        except OSError:  # server busy (e.g. loading weights); keep waiting
            time.sleep(5)
            continue
        if pid in hist:
            h = hist[pid]
            status = h.get("status", {})
            if status.get("status_str") == "error":
                msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(json.dumps(msgs[-1][1] if msgs else status)[:2000])
            files, report = [], ""
            for out in h["outputs"].values():
                for item in out.get("images", []):
                    files.append(str(Path(item.get("subfolder", "")) / item["filename"]))
                report = "\n".join(out.get("text", [])) or report
            return {"prompt_id": pid, "seconds": round(time.time() - t0, 1), "files": files, "report": report}
        time.sleep(1.0)
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--server", default="http://127.0.0.1:8199")
    ap.add_argument("--family", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="army-of-loras")
    ap.add_argument("--batch", help="file with one prompt per line (optionally 'family|prompt')")
    ap.add_argument("--log", help="append JSON lines of results here")
    args = ap.parse_args()
    jobs = []
    if args.batch:
        for line in Path(args.batch).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                fam, _, p = line.partition("|") if "|" in line else (args.family, "", line)
                jobs.append((fam.strip(), p.strip()))
    else:
        jobs.append((args.family, args.prompt))
    for fam, p in jobs:
        print(f"\n=== [{fam}] {p}", flush=True)
        try:
            res = run(args.server, p, fam, args.seed, args.prefix)
        except Exception as e:  # keep going through a batch
            print("FAILED:", e, flush=True)
            res = {"error": str(e)}
        else:
            print(res["report"])
            print(f"-> {res['files']}  ({res['seconds']} s)", flush=True)
        if args.log:
            with open(args.log, "a") as f:
                f.write(json.dumps({"family": fam, "prompt": p, **res}) + "\n")


if __name__ == "__main__":
    main()
