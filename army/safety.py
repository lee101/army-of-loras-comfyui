"""SFW gate shared by the HF scraper, the catalogs and the prompt router.

Two layers: a whole-word blocklist (cheap, catches the obvious) and a pybed semantic check
against descriptions of unsafe requests (catches paraphrases the list misses). A prompt that
fails either is refused before any model runs; a LoRA that fails is never catalogued.
"""

from __future__ import annotations

import re

from . import embed

_NSFW = r"""nsfw nude nudes nudity naked porn porno pornographic hentai lewd erotic erotica sex sexy sexual
sexually uncensored nipple nipples breast breasts boob boobs topless bottomless lingerie fetish bdsm
penis pussy vagina genital genitals ahegao onlyfans xxx r18 explicit seductive sensual undress undressing
undressed furry yiff gore gory dismember dismembered decapitated corpse cumshot orgasm masturbate
masturbation stripper striptease twerk jiggle panties thong""".split()
_LIKENESS = r"""celebrity celeb likeness lookalike deepfake faceswap face.?swap""".split()
_IP = r"""disney pixar ghibli marvel batman superman spiderman spider-man pokemon pikachu simpsons
star.?wars jedi harry.?potter hogwarts nintendo mario zelda sonic naruto genshin hololive mickey barbie
lego minecraft fortnite""".split()

BLOCK = re.compile(r"(?<![a-z0-9])(" + "|".join(_NSFW + _LIKENESS + _IP) + r")(?![a-z0-9])", re.I)

# Semantic guard: prompt is refused if it is closer to these than to the safe anchors.
_UNSAFE = [
    "explicit sexual content with naked people",
    "a nude woman posing",
    "graphic gore and mutilated bodies",
    "a photo of a famous celebrity or real politician",
    "violent attack on a real person",
]
_SAFE = [
    "a portrait of a person in everyday clothes",
    "a landscape painting",
    "a product photo",
    "an animated character in a cartoon",
    "a cinematic shot of a city",
    "a fantasy illustration of a castle",
]


def block_reason(text: str) -> str | None:
    """Return why `text` is not SFW, or None if it is fine."""
    if not text:
        return None
    hit = BLOCK.search(text)
    if hit:
        return f"blocked word '{hit.group(0).lower()}'"
    q = embed.embed(text)
    unsafe, phrase = embed.best_cos(q, _UNSAFE)
    safe, _ = embed.best_cos(q, _SAFE)
    if unsafe > 0.62 and unsafe > safe + 0.08:
        return f"semantically close to unsafe request '{phrase}' ({unsafe:.2f})"
    return None


class UnsafePromptError(ValueError):
    pass


def require_sfw(text: str) -> None:
    why = block_reason(text)
    if why:
        raise UnsafePromptError(f"army-of-loras is SFW only; prompt refused: {why}")
