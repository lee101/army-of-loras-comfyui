"""Prompt -> model family + a small squad of community LoRAs, decided with pybed embeddings.

Scoring is the omniserve-native `olora_route` rule, applied to every LoRA in the army:

    score = max cos(prompt, positive) - avoid_weight * max(0, max cos(prompt, avoid))
            + KEYWORD_BONUS if a keyword matches          (evidence, not a verdict)
    skipped entirely if an `exclude` keyword matches       (e.g. no style LoRA on lettering)

LoRAs at or above their `min_score` are eligible; the best one per `group` is drafted
(one style, one look, one motion...) up to `max_loras`. Strength scales with confidence.

Family selection ("auto"): first image vs video, then the family whose own description and
best-matching LoRAs fit the prompt best - the army decides which model gets deployed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from . import embed, safety

CATALOG_DIR = Path(__file__).resolve().parents[1] / "catalogs"
KEYWORD_BONUS = 0.25
DEFAULT_MIN_SCORE = 0.40
VIDEO_MARGIN = 0.15  # without an explicit video cue, video must win semantically by this much

IMAGE_FAMILIES = ("flux2_klein9b", "qwen_image21", "moss")
VIDEO_FAMILIES = ("ltx23", "h3")
FAMILIES = IMAGE_FAMILIES + VIDEO_FAMILIES

_VIDEO_WORDS = ["video", "videos", "clip", "footage", "animate", "animated sequence", "animation of",
                "timelapse", "time-lapse", "slow motion", "slow-motion", "camera pans", "camera pan",
                "panning", "dolly", "tracking shot", "orbit", "orbits", "zooms", "seconds", "scene where",
                "moving", "walks", "walking", "running", "dancing", "flying through", "flows", "waves crash",
                "explodes", "talking", "sings", "music video", "trailer", "drone shot",
                "turns into", "transforms into", "morphs into", "blasts off", "takes off"]
_IMAGE_WORDS = ["photo", "photograph", "poster", "illustration", "logo", "sticker", "icon", "painting",
                "wallpaper", "portrait", "drawing", "sketch", "render", "still", "album cover", "book cover",
                "t-shirt design", "emoji", "pixel art", "isometric", "coloring book", "mockup", "infographic"]
_VIDEO_PROTOS = ["a short video clip with motion over time", "cinematic footage with a moving camera",
                 "an animated scene where characters move and talk", "timelapse of clouds rolling",
                 "slow motion shot of water splashing", "a drone shot flying over mountains"]
_IMAGE_PROTOS = ["a single still image", "a detailed illustration", "a photograph of a subject",
                 "a poster design with a title", "a painting on canvas", "a logo or icon design"]


def _pattern(keyword: str) -> re.Pattern:
    """Whole-word match; a trailing '*' makes it a prefix match (olora_route semantics)."""
    kw = keyword.lower().strip()
    if kw.endswith("*"):
        return re.compile(r"(?<![a-z0-9])" + re.escape(kw[:-1]))
    return re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])")


def _any_keyword(text: str, keywords: list[str]) -> str | None:
    for kw in keywords or []:
        if _pattern(kw).search(text):
            return kw
    return None


@lru_cache(maxsize=None)
def load_catalog(family: str) -> dict:
    data = json.loads((CATALOG_DIR / f"{family}.json").read_text())
    for lora in data["loras"]:
        lora.setdefault("group", "style")
        lora.setdefault("strength", 1.0)
        lora.setdefault("min_score", data.get("min_score", DEFAULT_MIN_SCORE))
        # Precompute (and cache) every description embedding once.
        embed.embed_many(lora.get("positive", []) + lora.get("avoid", []))
    return data


def all_catalogs() -> dict[str, dict]:
    return {f: load_catalog(f) for f in FAMILIES if (CATALOG_DIR / f"{f}.json").exists()}


@dataclass
class Pick:
    id: str
    name: str
    repo: str
    file: str
    license: str
    group: str
    strength: float
    score: float
    matched: str
    keyword: str | None = None
    trigger: str = ""


@dataclass
class Plan:
    prompt: str
    original_prompt: str
    negative: str
    family: str
    family_label: str
    mode: str
    loras: list[Pick] = field(default_factory=list)
    considered: list[dict] = field(default_factory=list)
    family_scores: dict = field(default_factory=dict)
    mode_scores: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def __post_init__(self):
        self.loras = [p if isinstance(p, Pick) else Pick(**p) for p in self.loras]

    @property
    def is_video(self) -> bool:
        return self.mode == "video"

    def to_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        lines = [f"army-of-loras plan  ({self.elapsed_ms:.2f} ms, pybed static int8 embeddings)",
                 f"mode   : {self.mode}   " + "  ".join(f"{k}={v:.3f}" for k, v in self.mode_scores.items()),
                 f"family : {self.family_label}  [{self.family}]"]
        if self.family_scores:
            lines.append("         " + "  ".join(f"{k}={v:.3f}" for k, v in
                                                sorted(self.family_scores.items(), key=lambda kv: -kv[1])))
        lines.append("drafted:")
        if not self.loras:
            lines.append("  (none - prompt renders on the plain base model)")
        for p in self.loras:
            why = f'~ "{p.matched}"' + (f"  +kw '{p.keyword}'" if p.keyword else "")
            lines.append(f"  {p.strength:.2f} x {p.name:<28} score {p.score:.3f}  {why}")
            lines.append(f"         {p.repo}  ({p.license})")
        bench = [c for c in self.considered if c["id"] not in {p.id for p in self.loras}][:4]
        if bench:
            lines.append("bench (not drafted): " + ", ".join(f"{c['name']} {c['score']:.2f}" for c in bench))
        lines.append(f"prompt : {self.prompt}")
        if self.negative:
            lines.append(f"negative: {self.negative}")
        return "\n".join(lines)


def score_lora(lora: dict, q, lowered: str, avoid_weight: float) -> tuple[float, str, str | None] | None:
    if _any_keyword(lowered, lora.get("exclude", [])):
        return None
    pos, matched = embed.best_cos(q, lora.get("positive", []))
    if lora.get("avoid"):
        a, _ = embed.best_cos(q, lora["avoid"])
        if a > 0:
            pos -= avoid_weight * a
    kw = _any_keyword(lowered, lora.get("keywords", []))
    if kw:
        pos += KEYWORD_BONUS
    return pos, matched, kw


def rank_loras(catalog: dict, prompt: str) -> list[dict]:
    q = embed.embed(prompt)
    lowered = prompt.lower()
    weight = float(catalog.get("avoid_weight", 0.5))
    ranked = []
    for lora in catalog["loras"]:
        s = score_lora(lora, q, lowered, weight)
        if s is None:
            continue
        score, matched, kw = s
        ranked.append({"id": lora["id"], "name": lora["name"], "score": score, "matched": matched,
                       "keyword": kw, "eligible": score >= lora["min_score"], "lora": lora})
    ranked.sort(key=lambda r: -r["score"])
    return ranked


def decide_mode(prompt: str) -> tuple[str, dict]:
    q = embed.embed(prompt)
    lowered = prompt.lower()
    v, _ = embed.best_cos(q, _VIDEO_PROTOS)
    i, _ = embed.best_cos(q, _IMAGE_PROTOS)
    cue = _any_keyword(lowered, _VIDEO_WORDS)
    if cue:
        v += 0.2
    if _any_keyword(lowered, _IMAGE_WORDS):
        i += 0.2
    video = v > i if cue else v > i + VIDEO_MARGIN
    return ("video" if video else "image"), {"video": v, "image": i}


def family_fit(catalog: dict, prompt: str, ranked: list[dict]) -> float:
    q = embed.embed(prompt)
    describe, _ = embed.best_cos(q, catalog.get("describes", []))
    kw = KEYWORD_BONUS if _any_keyword(prompt.lower(), catalog.get("keywords", [])) else 0.0
    eligible = [r["score"] for r in ranked if r["eligible"]][:2]
    army = sum(eligible) / len(eligible) if eligible else 0.0
    return 0.5 * describe + 0.5 * army + kw + float(catalog.get("prior", 0.0))


def draft(catalog: dict, ranked: list[dict], max_loras: int, strength_scale: float,
          min_score_offset: float) -> list[Pick]:
    picks: list[Pick] = []
    groups: set[str] = set()
    budget = float(catalog.get("max_total_strength", 1.6)) * strength_scale
    for r in ranked:
        lora = r["lora"]
        if len(picks) >= max_loras:
            break
        if r["score"] < lora["min_score"] + min_score_offset or lora["group"] in groups:
            continue
        confidence = min(1.0, max(0.0, (r["score"] - lora["min_score"]) / 0.2))
        strength = round(lora["strength"] * strength_scale * (0.75 + 0.25 * confidence), 3)
        strength = round(min(strength, budget), 3)
        if strength <= 0.05:
            break
        budget -= strength
        groups.add(lora["group"])
        picks.append(Pick(id=lora["id"], name=lora["name"], repo=lora["repo"], file=lora["file"],
                          license=lora.get("license", "?"), group=lora["group"], strength=strength,
                          score=round(r["score"], 4), matched=r["matched"], keyword=r["keyword"],
                          trigger=lora.get("trigger", "")))
    return picks


def compose_prompt(prompt: str, picks: list[Pick], catalog: dict) -> tuple[str, str]:
    triggers = [p.trigger for p in picks if p.trigger and p.trigger.lower() not in prompt.lower()]
    suffixes = [l.get("suffix", "") for p in picks for l in catalog["loras"] if l["id"] == p.id]
    out = ", ".join(triggers + [prompt]) if triggers else prompt
    out += "".join(s for s in suffixes if s)
    negatives = [catalog.get("negative", "")] + [l.get("negative", "") for p in picks
                                                 for l in catalog["loras"] if l["id"] == p.id]
    seen, neg = set(), []
    for part in ", ".join(n for n in negatives if n).split(","):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            neg.append(part)
    return out, ", ".join(neg)


def route(prompt: str, family: str = "auto", mode: str = "auto", max_loras: int = 3,
          strength_scale: float = 1.0, min_score_offset: float = 0.0) -> Plan:
    t0 = time.perf_counter()
    prompt = (prompt or "").strip()
    safety.require_sfw(prompt)
    catalogs = all_catalogs()

    if family in catalogs:
        mode_used = catalogs[family]["mode"]
        mode_scores: dict = {}
        candidates = [family]
    else:
        mode_used, mode_scores = decide_mode(prompt) if mode == "auto" else (mode, {})
        pool = IMAGE_FAMILIES if mode_used == "image" else VIDEO_FAMILIES
        candidates = [f for f in pool if f in catalogs]

    ranked_by = {f: rank_loras(catalogs[f], prompt) for f in candidates}
    fits = {f: family_fit(catalogs[f], prompt, ranked_by[f]) for f in candidates}
    chosen = max(fits, key=fits.get)
    catalog = catalogs[chosen]
    picks = draft(catalog, ranked_by[chosen], max_loras, strength_scale, min_score_offset)
    final, negative = compose_prompt(prompt, picks, catalog)
    considered = [{k: v for k, v in r.items() if k != "lora"} for r in ranked_by[chosen][:8]]
    return Plan(prompt=final, original_prompt=prompt, negative=negative, family=chosen,
                family_label=catalog.get("label", chosen), mode=catalog["mode"], loras=picks,
                considered=considered, family_scores=fits if len(fits) > 1 else {},
                mode_scores=mode_scores, settings=catalog.get("settings", {}),
                elapsed_ms=(time.perf_counter() - t0) * 1000)
