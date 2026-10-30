"""Write docs/CREDITS.md: every routed LoRA with its author repo, licence and trigger."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["flux2_klein9b", "qwen_image21", "moss", "ltx23", "h3"]
lines = ["# Credits: the army", "",
         "Every LoRA the router can draft, fetched from its author's Hugging Face repo on first use.",
         "All are text-to-image/video LoRAs, picked by hand from `catalogs/raw/` (see `tools/scrape_hf_loras.py`):",
         "open or base-model community licences only, no adult, likeness or franchise LoRAs.", ""]
total = 0
for fam in ORDER:
    c = json.loads((ROOT / "catalogs" / f"{fam}.json").read_text())
    lines += [f"## {c['label']}", "", f"Base model: [{c['base']}](https://huggingface.co/{c['base']}) ({c['base_license']})", "",
              "| LoRA | group | licence | trigger | repo |", "|---|---|---|---|---|"]
    for l in c["loras"]:
        total += 1
        trig = f"`{l['trigger']}`" if l.get("trigger") else "-"
        lines.append(f"| {l['name']} | {l['group']} | {l['license']} | {trig} | [{l['repo']}](https://huggingface.co/{l['repo']}) |")
    lines.append("")
lines.insert(2, f"**{total} community LoRAs across {len(ORDER)} model families.**\n")
(ROOT / "docs" / "CREDITS.md").write_text("\n".join(lines) + "\n")
print(total)
