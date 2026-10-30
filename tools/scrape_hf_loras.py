"""Scrape community LoRAs for each supported base model from the Hugging Face Hub.

Writes catalogs/raw/<family>.json with every candidate adapter and the reason it was
kept or dropped. `tools/build_catalog.py` turns the kept ones into routing catalogs.

    uv run tools/scrape_hf_loras.py            # all families
    uv run tools/scrape_hf_loras.py ltx23      # one family
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "catalogs" / "raw"

FAMILIES = {
    "flux2_klein9b": ["black-forest-labs/FLUX.2-klein-9B", "black-forest-labs/FLUX.2-klein-base-9B"],
    "qwen_image21": ["Qwen/Qwen-Image-2.1"],
    "moss": ["OpenMOSS-Team/MOSS-Image-SFT", "OpenMOSS-Team/MOSS-Image-Base", "Comfy-Org/MOSS-Image"],
    "ltx23": ["Lightricks/LTX-2.3"],
    "h3": ["MiniMaxAI/MiniMax-H3"],
}

OPEN_LICENSES = {
    "apache-2.0", "mit", "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "openrail", "openrail++",
    "creativeml-openrail-m", "bsd-3-clause", "unlicense", "cc-by-nc-4.0", "cc-by-nc-sa-4.0",
}

# Anything sexual, gory or about real people / franchises is dropped. Matched on whole
# words against the repo id, tags and the model card.
NSFW_WORDS = r"""nsfw nude nudity naked porn porno hentai lewd erotic erotica sex sexy sexual uncensored
uncensor nipple nipples breast breasts boob boobs topless lingerie bikini fetish bdsm penis pussy vagina
genital genitals ahegao onlyfans xxx r18 18\+ explicit seductive sensual luxe_sensual undress undressing
furry yiff gore gory blood bloody corpse eros 10eros heretic abliterated jiggle twerk""".split()
IP_WORDS = r"""disney pixar ghibli marvel dc batman superman spiderman pokemon pikachu simpsons star.?wars
jedi harry.?potter hogwarts nintendo mario zelda sonic naruto one.?piece dragon.?ball genshin hololive
celebrity celeb likeness lookalike deepfake faceswap face.?swap actor actress trump musk swift
mickey barbie lego minecraft fortnite""".split()
BAD = re.compile(r"(?<![a-z0-9])(" + "|".join(NSFW_WORDS + IP_WORDS) + r")(?![a-z0-9])", re.I)


def get(url: str, raw: bool = False, tries: int = 4):
    for attempt in range(tries):
        try:
            return _get(url, raw)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _get(url: str, raw: bool):
    req = Request(url, headers={"User-Agent": "army-of-loras-comfyui/0.1"})
    tok = Path.home() / ".cache/huggingface/token"
    if tok.exists():
        req.add_header("Authorization", "Bearer " + tok.read_text().strip())
    with urlopen(req, timeout=120) as r:
        data = r.read()
    return data.decode("utf-8", "replace") if raw else json.loads(data)


def card_text(repo: str) -> str:
    try:
        return get(f"https://huggingface.co/{repo}/raw/main/README.md", raw=True)
    except Exception:
        return ""


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            text = text[end + 4:]
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def candidates(base: str) -> list[dict]:
    url = (f"https://huggingface.co/api/models?filter={quote('base_model:adapter:' + base)}"
           "&sort=downloads&limit=300&full=true&cardData=true")
    return get(url)


def review(m: dict) -> dict:
    repo = m["id"]
    card = m.get("cardData") or {}
    tags = m.get("tags") or []
    lic = card.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    files = [s["rfilename"] for s in m.get("siblings") or [] if s["rfilename"].endswith(".safetensors")]
    text = card_text(repo)
    body = strip_front_matter(text)
    out = {
        "repo": repo,
        "downloads": m.get("downloads", 0),
        "likes": m.get("likes", 0),
        "license": lic,
        "tags": tags,
        "files": files,
        "instance_prompt": card.get("instance_prompt"),
        "widget": [w.get("text") for w in card.get("widget") or [] if isinstance(w, dict) and w.get("text")][:4],
        "card": body[:1500],
        "keep": True,
        "why": "",
    }
    all_files = [s["rfilename"] for s in m.get("siblings") or []]
    # File names matter: some repos with an innocent card ship explicit LoRAs alongside the SFW ones.
    hay = " ".join([repo, " ".join(tags), body[:4000], str(out["instance_prompt"] or ""), " ".join(out["widget"]),
                    " ".join(f.replace("_", " ").replace("-", " ") for f in all_files)])
    if "not-for-all-audiences" in tags:
        out.update(keep=False, why="tag:not-for-all-audiences")
    elif (hit := BAD.search(hay)):
        out.update(keep=False, why=f"blocklist:{hit.group(0).lower()}")
    elif not files:
        out.update(keep=False, why="no safetensors")
    elif (lic or "").lower() not in OPEN_LICENSES and not str(lic or "").startswith("flux"):
        out.update(keep=False, why=f"license:{lic}")
    return out


def scrape(family: str) -> list[dict]:
    seen, models = set(), []
    for base in FAMILIES[family]:
        for m in candidates(base):
            if m["id"] not in seen:
                seen.add(m["id"])
                models.append(m)
    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(review, models))
    rows.sort(key=lambda r: -r["downloads"])
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"{family}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    kept = sum(r["keep"] for r in rows)
    print(f"{family}: {len(rows)} adapters, {kept} kept")
    return rows


if __name__ == "__main__":
    for fam in sys.argv[1:] or FAMILIES:
        scrape(fam)
