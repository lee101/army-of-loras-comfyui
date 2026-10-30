"""Router, catalog and SFW-gate tests. Pure Python: no ComfyUI or GPU needed.

    uv run --with pytest --with "pybed @ git+https://github.com/lee101/pybed" pytest -q
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from army import router, safety  # noqa: E402

CATALOGS = sorted((ROOT / "catalogs").glob("*.json"))


def test_catalogs_are_complete_and_sfw():
    assert {p.stem for p in CATALOGS} == set(router.FAMILIES)
    seen = set()
    for path in CATALOGS:
        cat = json.loads(path.read_text())
        assert cat["mode"] in ("image", "video") and cat["describes"] and cat["base_license"]
        for lora in cat["loras"]:
            assert lora["id"] not in seen, lora["id"]
            seen.add(lora["id"])
            assert lora["repo"] and lora["file"].endswith(".safetensors") and lora["license"]
            assert lora["positive"], lora["id"]
            text = " ".join([lora["name"], lora["repo"], lora["file"], lora.get("trigger", "")] + lora["positive"])
            assert safety.block_reason(text) is None, (lora["id"], safety.block_reason(text))
    assert len(seen) >= 60


@pytest.mark.parametrize("prompt,family,lora", [
    ("isometric diorama of a cozy wizard library", "flux2_klein9b", "redmond_isometric"),
    ("a crochet amigurumi octopus on a wooden table", "flux2_klein9b", "redmond_amigurumi"),
    ("a monet style garden with a footbridge over a lily pond", "moss", "moss_impressionism"),
    ("shot on kodachrome, a 1950s diner at dusk", "moss", "moss_kodachrome"),
    ("synthwave outrun sunset over a chrome city", "qwen_image21", "qwen_outrun"),
    ("16-bit pixel art video of a fox running through a forest", "h3", "h3_pixel16"),
    ("a paper boat transforms into a koi fish, cinematic video", "ltx23", "ltx_transition"),
])
def test_routes(prompt, family, lora):
    plan = router.route(prompt)
    assert plan.family == family, plan.report()
    assert lora in [p.id for p in plan.loras], plan.report()


def test_text_prompts_go_to_qwen_without_style_loras():
    plan = router.route('a vintage travel poster that says "VISIT THE MOON" in bold letters')
    assert plan.family == "qwen_image21" and plan.mode == "image"


def test_one_lora_per_group_and_budget():
    plan = router.route("a cute sticker of a red panda eating ramen, pixel art", max_loras=6)
    groups = [p.group for p in plan.loras]
    assert len(groups) == len(set(groups))
    assert sum(p.strength for p in plan.loras) <= router.load_catalog(plan.family)["max_total_strength"] + 1e-6


def test_triggers_are_prepended_once():
    plan = router.route("a cute sticker of a red panda, pixel art")
    assert plan.prompt.endswith("a cute sticker of a red panda, pixel art")
    assert plan.prompt.count("PixArFK") == 1


def test_fixed_family_and_mode():
    assert router.route("a castle", family="moss").family == "moss"
    assert router.route("a castle", mode="video").mode == "video"


@pytest.mark.parametrize("prompt", ["a nude woman on a beach", "nsfw anime girl", "deepfake of a celebrity"])
def test_unsafe_prompts_are_refused(prompt):
    with pytest.raises(safety.UnsafePromptError):
        router.route(prompt)


def test_plan_roundtrips_through_dict():
    plan = router.route("isometric diorama of a cozy wizard library")
    again = router.Plan(**plan.to_dict())
    assert again.loras[0].id == plan.loras[0].id and again.report() == plan.report()


def test_routing_is_fast():
    router.route("warm up")
    plan = router.route("a psychedelic poster of a glowing mushroom forest")
    assert plan.elapsed_ms < 50
