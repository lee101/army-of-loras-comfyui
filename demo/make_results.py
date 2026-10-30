"""Collect demo outputs from ComfyUI's output dir into demo/results/ and write demo/results.md.

    python demo/make_results.py RUNS_JSONL COMFY_OUTPUT_DIR

RUNS_JSONL is what `scripts/run_prompt.py --log` wrote. Each output keeps its .plan.json sidecar
(the exact routing decision), so the gallery and the video show the real plans.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "results"
LABEL = {"flux2_klein9b": "FLUX.2 [klein] 9B", "qwen_image21": "Qwen-Image 2.1", "moss": "MOSS-Image",
         "ltx23": "LTX-2.3", "h3": "MiniMax H3"}


def main():
    runs_path, comfy_out = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    latest: dict[str, dict] = {}
    for line in runs_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("files"):
            latest[r["prompt"]] = r  # keep the newest successful run per prompt
    rows, kept = [], []
    for i, r in enumerate(latest.values()):
        src = comfy_out / r["files"][0]
        plan_src = src.with_name(src.name.rsplit(".", 1)[0] + ".plan.json")
        plan = json.loads(plan_src.read_text())
        stem = f"{i:02d}_{plan['family']}"
        media = OUT / (stem + src.suffix)
        shutil.copy2(src, media)
        shutil.copy2(plan_src, OUT / (stem + ".plan.json"))
        r = {**r, "media_file": media.name, "plan_file": stem + ".plan.json"}
        kept.append(r)
        loras = ", ".join(f"{l['name']} x{l['strength']:.2f}" for l in plan["loras"]) or "(none, plain base model)"
        rows.append(f"| [{media.name}](results/{media.name}) | {plan['original_prompt']} | {plan['mode']} | "
                    f"{LABEL[plan['family']]} | {loras} | {plan['elapsed_ms']:.1f} | {r['seconds']:.0f} |")
    (OUT / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    md = ["# Demo results", "",
          "Every prompt in `demo/prompts.txt` run through `workflows/army_auto.json` with the router on `auto`,",
          "seed 42, on one RTX 3090 Ti (24 GB). Wall time includes model loading from disk the first time a",
          "family is used, so the first run per model is much slower than the rest.", "",
          "| output | prompt | mode | model | drafted LoRAs | route ms | wall s |", "|---|---|---|---|---|---|---|", *rows, ""]
    (ROOT / "demo" / "results.md").write_text("\n".join(md))
    print(f"{len(kept)} results -> {OUT}")


if __name__ == "__main__":
    main()
