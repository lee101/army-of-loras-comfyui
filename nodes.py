"""ComfyUI nodes for army-of-loras.

ArmyLoraRouter   prompt -> ARMY_PLAN (family, drafted LoRAs, rewritten prompt, report)
ArmyLoraApply    MODEL (+CLIP) + ARMY_PLAN -> model with the drafted LoRAs, fetched from HF on demand
ArmyAutoOutput   output node with one lazy input per family: only the routed branch executes
"""

from __future__ import annotations

import json
import logging
import os
import re

import comfy.sd
import comfy.utils
import folder_paths

from .army import router, safety

log = logging.getLogger("army-of-loras")

FAMILY_CHOICES = ["auto", "auto_image", "auto_video", *router.FAMILIES]
IMAGE_OUT = {f: "IMAGE" for f in router.IMAGE_FAMILIES}
VIDEO_OUT = {f: "VIDEO" for f in router.VIDEO_FAMILIES}


def _plan_from(plan):
    return plan if isinstance(plan, router.Plan) else router.Plan(**plan)


def lora_path(repo: str, file: str) -> str:
    """Local path of a catalog LoRA, downloading it from the Hugging Face Hub if needed."""
    sub = os.path.join("army", re.sub(r"[^A-Za-z0-9_.-]+", "__", repo))
    rel = os.path.join(sub, file)
    found = folder_paths.get_full_path("loras", rel)
    if found:
        return found
    from huggingface_hub import hf_hub_download

    root = os.path.join(folder_paths.get_folder_paths("loras")[0], sub)
    log.info("army-of-loras: fetching %s/%s", repo, file)
    return hf_hub_download(repo_id=repo, filename=file, local_dir=root)


class ArmyLoraRouter:
    CATEGORY = "army-of-loras"
    FUNCTION = "route"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ARMY_PLAN", "STRING", "STRING", "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("plan", "prompt", "negative", "family", "is_video", "report")
    DESCRIPTION = ("Scores the prompt against every community LoRA in the SFW catalogs with pybed static "
                   "embeddings, picks the model family (image or video) and drafts the best LoRA squad.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": "a cute sticker of a red panda eating ramen, pixel art"}),
                "family": (FAMILY_CHOICES, {"default": "auto",
                                            "tooltip": "auto picks image vs video and the model; auto_image/auto_video fix the mode."}),
                "max_loras": ("INT", {"default": 3, "min": 0, "max": 6}),
                "strength_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "min_score_offset": ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.5, "step": 0.01,
                                               "tooltip": "Raise to be pickier about which LoRAs are drafted."}),
            }
        }

    def route(self, prompt, family, max_loras, strength_scale, min_score_offset):
        mode = "auto"
        if family in ("auto_image", "auto_video"):
            mode, family = family.split("_", 1)[1], "auto"
        plan = router.route(prompt, family=family, mode=mode, max_loras=max_loras,
                            strength_scale=strength_scale, min_score_offset=min_score_offset)
        report = plan.report()
        log.info("\n%s", report)
        return {"ui": {"text": [report], "plan": [json.dumps(plan.to_dict())]},
                "result": (plan.to_dict(), plan.prompt, plan.negative, plan.family, plan.is_video, report)}


class ArmyLoraApply:
    CATEGORY = "army-of-loras"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "CLIP")
    DESCRIPTION = "Loads the LoRAs drafted in the plan (downloading from Hugging Face on first use)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "plan": ("ARMY_PLAN",),
                "family": (["any", *router.FAMILIES], {"default": "any",
                                                       "tooltip": "Only apply when the plan routed to this family."}),
            },
            "optional": {"clip": ("CLIP",)},
        }

    def apply(self, model, plan, family, clip=None):
        plan = _plan_from(plan)
        if family != "any" and plan.family != family:
            return (model, clip)
        for pick in plan.loras:
            pick = pick if isinstance(pick, router.Pick) else router.Pick(**pick)
            path = lora_path(pick.repo, pick.file)
            lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            model, clip_out = comfy.sd.load_lora_for_models(model, clip, lora, pick.strength, 0.0,
                                                            lora_metadata=meta)
            clip = clip_out if clip is not None else None
            log.info("army-of-loras: %s x %s", pick.strength, pick.name)
        return (model, clip)


class ArmyAutoOutput:
    """One lazy input per family; ComfyUI only executes the branch the plan routed to."""

    CATEGORY = "army-of-loras"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        lazy = {f: (t, {"lazy": True}) for f, t in {**IMAGE_OUT, **VIDEO_OUT}.items()}
        return {
            "required": {"plan": ("ARMY_PLAN",),
                         "filename_prefix": ("STRING", {"default": "army-of-loras"})},
            "optional": lazy,
        }

    def check_lazy_status(self, plan, filename_prefix, **branches):
        family = _plan_from(plan).family
        return [family] if branches.get(family) is None else []

    def save(self, plan, filename_prefix, **branches):
        plan = _plan_from(plan)
        result = branches.get(plan.family)
        if result is None:
            raise RuntimeError(f"army-of-loras: no {plan.family} branch is connected to ArmyAutoOutput")
        prefix = f"{filename_prefix}/{plan.family}"
        if plan.family in IMAGE_OUT:
            return self._save_images(result, prefix, plan)
        return self._save_video(result, prefix, plan)

    @staticmethod
    def _sidecar(folder, stem, plan):
        with open(os.path.join(folder, stem + ".plan.json"), "w") as f:
            json.dump(plan.to_dict(), f, indent=1)

    def _save_images(self, images, prefix, plan):
        import numpy as np
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        folder, name, counter, sub, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0])
        out = []
        for img in images:
            arr = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
            meta = PngInfo()
            meta.add_text("army_plan", json.dumps(plan.to_dict()))
            stem = f"{name}_{counter:05}_"
            Image.fromarray(arr).save(os.path.join(folder, stem + ".png"), pnginfo=meta, compress_level=4)
            self._sidecar(folder, stem, plan)
            out.append({"filename": stem + ".png", "subfolder": sub, "type": "output"})
            counter += 1
        return {"ui": {"images": out, "text": [plan.report()]}}

    def _save_video(self, video, prefix, plan):
        from comfy_api.latest import Types

        w, h = video.get_dimensions()
        folder, name, counter, sub, _ = folder_paths.get_save_image_path(
            prefix, folder_paths.get_output_directory(), w, h)
        stem = f"{name}_{counter:05}_"
        video.save_to(os.path.join(folder, stem + ".mp4"), format=Types.VideoContainer("mp4"),
                      codec=Types.VideoCodec("h264"))
        self._sidecar(folder, stem, plan)
        return {"ui": {"images": [{"filename": stem + ".mp4", "subfolder": sub, "type": "output"}],
                       "animated": (True,), "text": [plan.report()]}}


class ArmySfwGate:
    CATEGORY = "army-of-loras"
    FUNCTION = "check"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    DESCRIPTION = "Passes the prompt through unchanged, or stops the run if it is not SFW."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"multiline": True})}}

    def check(self, prompt):
        safety.require_sfw(prompt)
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "ArmyLoraRouter": ArmyLoraRouter,
    "ArmyLoraApply": ArmyLoraApply,
    "ArmyAutoOutput": ArmyAutoOutput,
    "ArmySfwGate": ArmySfwGate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ArmyLoraRouter": "Army of LoRAs: Router (pybed)",
    "ArmyLoraApply": "Army of LoRAs: Apply Drafted LoRAs",
    "ArmyAutoOutput": "Army of LoRAs: Auto Output (image or video)",
    "ArmySfwGate": "Army of LoRAs: SFW Gate",
}
