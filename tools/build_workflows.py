"""Build the army-of-loras ComfyUI workflows (API format) from one description per model family.

    python tools/build_workflows.py            # writes workflows/api/*.json
    python tools/build_workflows.py --ui URL   # also writes UI-format workflows/*.json via a running ComfyUI

Each family branch is: loaders -> speed LoRA -> ArmyLoraApply(plan) -> sampler -> decode.
`army_auto.json` wires all five branches into ArmyAutoOutput, whose inputs are lazy, so
ComfyUI only loads and runs the model the router picked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows"

# Model files, as named in each official Comfy-Org repo (see README "Models").
FILES = {
    "flux_unet": "flux-2-klein-base-9b-fp8.safetensors",
    "flux_te": "qwen_3_8b.safetensors",
    "flux_vae": "flux2-vae.safetensors",
    "qwen_unet": "qwen_image_2.1_int8_convrot.safetensors",
    "qwen_turbo_lora": "qwen-image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
    "qwen_te": "qwen3vl_8b_w4a8.safetensors",
    "qwen_vae": "qwen_image_2.1_vae_bf16.safetensors",
    "moss_unet": "moss_image_sft_int8_convrot.safetensors",
    "moss_te": "qwen3.5_9b_int8_convrot.safetensors",
    "moss_vae": "flux2-vae.safetensors",
    "moss_adapter": "rapid-moss-4step",
    "ltx_ckpt": "ltx-2.3-22b-distilled-fp8.safetensors",
    "ltx_te": "gemma_3_12B_it_fp4_mixed.safetensors",
    "ltx_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "h3_unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "h3_turbo_lora": "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
    "h3_te": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "h3_video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "h3_audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
}
# Optional per-machine filename overrides (gitignored), e.g. {"qwen_unet": "my_quant.safetensors"}.
OFFICIAL_FILES = dict(FILES)
_LOCAL = ROOT / "models.local.json"
if _LOCAL.exists():
    FILES.update(json.loads(_LOCAL.read_text()))


class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.lane = "router"  # which family branch new nodes belong to (used for UI layout)
        self.lanes: dict[str, str] = {}

    def add(self, cls: str, title: str | None = None, **inputs) -> str:
        nid = str(len(self.nodes) + 1)
        self.nodes[nid] = {"class_type": cls, "inputs": inputs}
        if title:
            self.nodes[nid]["_meta"] = {"title": title}
        self.lanes[nid] = self.lane
        return nid

    @staticmethod
    def out(nid: str, slot: int = 0) -> list:
        return [nid, slot]


def router(g: Graph, prompt: str, family: str = "auto") -> str:
    return g.add("ArmyLoraRouter", "Army of LoRAs: Router (pybed)", prompt=prompt, family=family,
                 max_loras=3, strength_scale=1.0, min_score_offset=0.0)


def flux_branch(g: Graph, r: str, seed: int) -> list:
    unet = g.add("UNETLoader", "FLUX.2 klein 9B base fp8", unet_name=FILES["flux_unet"], weight_dtype="default")
    m = g.add("ArmyLoraApply", "Army: apply FLUX LoRAs", model=[unet, 0], plan=[r, 0], family="flux2_klein9b")
    clip = g.add("CLIPLoader", clip_name=FILES["flux_te"], type="flux2", device="default")
    pos = g.add("CLIPTextEncode", text=[r, 1], clip=[clip, 0])
    neg = g.add("ConditioningZeroOut", conditioning=[pos, 0])
    guider = g.add("CFGGuider", model=[m, 0], positive=[pos, 0], negative=[neg, 0], cfg=3.5)
    sigmas = g.add("Flux2Scheduler", steps=20, width=1024, height=1024)
    latent = g.add("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)
    s = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=seed), 0], guider=[guider, 0],
              sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0], sigmas=[sigmas, 0],
              latent_image=[latent, 0])
    vae = g.add("VAELoader", vae_name=FILES["flux_vae"])
    return [g.add("VAEDecode", samples=[s, 0], vae=[vae, 0]), 0]


def qwen_branch(g: Graph, r: str, seed: int) -> list:
    unet = g.add("UNETLoader", "Qwen-Image 2.1", unet_name=FILES["qwen_unet"], weight_dtype="default")
    m = g.add("LoraLoaderModelOnly", "Qwen 2.1 turbo (6-step)", model=[unet, 0],
              lora_name=FILES["qwen_turbo_lora"], strength_model=1.0)
    m = g.add("ArmyLoraApply", "Army: apply Qwen LoRAs", model=[m, 0], plan=[r, 0], family="qwen_image21")
    clip = g.add("CLIPLoader", clip_name=FILES["qwen_te"], type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=FILES["qwen_vae"])
    enc = g.add("TextEncodeQwenImage21", clip=[clip, 0], prompt=[r, 1], negative_prompt=[r, 2], resolution=1024)
    guider = g.add("CFGGuider", model=[m, 0], positive=[enc, 0], negative=[enc, 1], cfg=1.0)
    sigmas = g.add("BasicScheduler", model=[m, 0], scheduler="simple", steps=6, denoise=1.0)
    latent = g.add("EmptyLatentImage", width=1024, height=1024, batch_size=1)
    s = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=seed), 0], guider=[guider, 0],
              sampler=[g.add("KSamplerSelect", sampler_name="euler"), 0], sigmas=[sigmas, 0],
              latent_image=[latent, 0])
    return [g.add("VAEDecode", samples=[s, 0], vae=[vae, 0]), 0]


def moss_branch(g: Graph, r: str, seed: int) -> list:
    unet = g.add("UNETLoader", "MOSS-Image SFT int8", unet_name=FILES["moss_unet"], weight_dtype="default")
    m = g.add("ArmyLoraApply", "Army: apply MOSS LoRAs", model=[unet, 0], plan=[r, 0], family="moss")
    fast = g.add("RapidMOSSAdapter", "rapid-moss 4-step adapter", model=[m, 0], adapter=FILES["moss_adapter"],
                 steps=4, lora_strength=1.0)
    clip = g.add("CLIPLoader", clip_name=FILES["moss_te"], type="moss_image", device="default")
    pos = g.add("CLIPTextEncode", text=[r, 1], clip=[clip, 0])
    guider = g.add("BasicGuider", model=[fast, 0], conditioning=[pos, 0])
    latent = g.add("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1)
    s = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=seed), 0], guider=[guider, 0],
              sampler=[fast, 1], sigmas=[fast, 2], latent_image=[latent, 0])
    vae = g.add("VAELoader", vae_name=FILES["moss_vae"])
    return [g.add("VAEDecode", samples=[s, 0], vae=[vae, 0]), 0]


def ltx_branch(g: Graph, r: str, seed: int, width=1024, height=576, frames=97, fps=24) -> list:
    """Official LTX-2.3 two-stage T2V recipe on the distilled checkpoint: half-res pass, x2 latent upsample, refine."""
    ck = g.add("CheckpointLoaderSimple", "LTX-2.3 22B distilled fp8", ckpt_name=FILES["ltx_ckpt"])
    m = g.add("ArmyLoraApply", "Army: apply LTX LoRAs", model=[ck, 0], plan=[r, 0], family="ltx23")
    clip = g.add("LTXAVTextEncoderLoader", text_encoder=FILES["ltx_te"], ckpt_name=FILES["ltx_ckpt"], device="default")
    avae = g.add("LTXVAudioVAELoader", ckpt_name=FILES["ltx_ckpt"])
    pos = g.add("CLIPTextEncode", text=[r, 1], clip=[clip, 0])
    neg = g.add("CLIPTextEncode", text=[r, 2], clip=[clip, 0])
    cond = g.add("LTXVConditioning", positive=[pos, 0], negative=[neg, 0], frame_rate=float(fps))
    vlat = g.add("EmptyLTXVLatentVideo", width=width // 2, height=height // 2, length=frames, batch_size=1)
    alat = g.add("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=[avae, 0])
    av = g.add("LTXVConcatAVLatent", video_latent=[vlat, 0], audio_latent=[alat, 0])
    g1 = g.add("CFGGuider", model=[m, 0], positive=[cond, 0], negative=[cond, 1], cfg=1.0)
    s1 = g.add("SamplerCustomAdvanced", "stage 1 (half res)", noise=[g.add("RandomNoise", noise_seed=seed), 0],
               guider=[g1, 0], sampler=[g.add("KSamplerSelect", sampler_name="euler_ancestral_cfg_pp"), 0],
               sigmas=[g.add("ManualSigmas", sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"), 0],
               latent_image=[av, 0])
    sep1 = g.add("LTXVSeparateAVLatent", av_latent=[s1, 0])
    up_model = g.add("LatentUpscaleModelLoader", model_name=FILES["ltx_upscaler"])
    up = g.add("LTXVLatentUpsampler", samples=[sep1, 0], upscale_model=[up_model, 0], vae=[ck, 2])
    crop = g.add("LTXVCropGuides", positive=[cond, 0], negative=[cond, 1], latent=[sep1, 0])
    av2 = g.add("LTXVConcatAVLatent", video_latent=[up, 0], audio_latent=[sep1, 1])
    g2 = g.add("CFGGuider", model=[m, 0], positive=[crop, 0], negative=[crop, 1], cfg=1.0)
    s2 = g.add("SamplerCustomAdvanced", "stage 2 (x2 refine)", noise=[g.add("RandomNoise", noise_seed=seed + 1), 0],
               guider=[g2, 0], sampler=[g.add("KSamplerSelect", sampler_name="euler_cfg_pp"), 0],
               sigmas=[g.add("ManualSigmas", sigmas="0.85, 0.7250, 0.4219, 0.0"), 0], latent_image=[av2, 0])
    sep2 = g.add("LTXVSeparateAVLatent", av_latent=[s2, 0])
    frames_img = g.add("VAEDecodeTiled", samples=[sep2, 0], vae=[ck, 2], tile_size=768, overlap=64,
                       temporal_size=4096, temporal_overlap=4)
    audio = g.add("LTXVAudioVAEDecode", samples=[sep2, 1], audio_vae=[avae, 0])
    return [g.add("CreateVideo", images=[frames_img, 0], audio=[audio, 0], fps=float(fps)), 0]


def h3_branch(g: Graph, r: str, seed: int, width=832, height=480, frames=73, fps=24) -> list:
    unet = g.add("UNETLoader", "MiniMax H3 int8", unet_name=FILES["h3_unet"], weight_dtype="default")
    m = g.add("LoraLoaderModelOnly", "H3 lightx2v turbo (4-step)", model=[unet, 0],
              lora_name=FILES["h3_turbo_lora"], strength_model=1.0)
    m = g.add("ArmyLoraApply", "Army: apply H3 LoRAs", model=[m, 0], plan=[r, 0], family="h3")
    clip = g.add("CLIPLoader", clip_name=FILES["h3_te"], type="minimax", device="default")
    vvae = g.add("VAELoader", vae_name=FILES["h3_video_vae"])
    avae = g.add("VAELoader", vae_name=FILES["h3_audio_vae"])
    cond = g.add("MiniMaxH3ImageToVideo", clip=[clip, 0], vae=[vvae, 0], prompt=[r, 1], width=width,
                 height=height, length=frames)
    guider = g.add("BasicGuider", model=[m, 0], conditioning=[cond, 0])
    sigmas = g.add("BasicScheduler", model=[m, 0], scheduler="simple", steps=4, denoise=1.0)
    s = g.add("SamplerCustomAdvanced", noise=[g.add("RandomNoise", noise_seed=seed), 0], guider=[guider, 0],
              sampler=[g.add("KSamplerSelect", sampler_name="res_multistep"), 0], sigmas=[sigmas, 0],
              latent_image=[cond, 1])
    frames_img = g.add("VAEDecode", samples=[s, 0], vae=[vvae, 0])
    audio = g.add("VAEDecodeAudio", samples=[s, 0], vae=[avae, 0])
    return [g.add("CreateVideo", images=[frames_img, 0], audio=[audio, 0], fps=float(fps)), 0]


BRANCHES = {"flux2_klein9b": flux_branch, "qwen_image21": qwen_branch, "moss": moss_branch,
            "ltx23": ltx_branch, "h3": h3_branch}
DEFAULT_PROMPTS = {
    "auto": "a cute sticker of a red panda eating ramen, pixel art",
    "flux2_klein9b": "isometric diorama of a cozy wizard library with floating books",
    "qwen_image21": "35mm film still of a neon noodle bar at night, a sign that says \"ARMY OF LORAS\"",
    "moss": "a monet style garden with a wooden footbridge over a lily pond",
    "ltx23": "a paper boat transforms into a glowing koi fish as it drifts down a rainy street, cinematic",
    "h3": "16-bit pixel art video of a fox running through a forest at dusk, chiptune music",
}


def build(family: str, prompt: str | None = None, seed: int = 42, filename_prefix: str = "army-of-loras",
          with_lanes: bool = False):
    g = Graph()
    r = router(g, prompt or DEFAULT_PROMPTS[family], "auto" if family == "auto" else family)
    fams = BRANCHES if family == "auto" else {family: BRANCHES[family]}
    outs = {}
    for f, fn in fams.items():
        g.lane = f
        outs[f] = fn(g, r, seed)
    g.lane = "router"
    g.add("ArmyAutoOutput", "Army of LoRAs: Auto Output", plan=[r, 0], filename_prefix=filename_prefix, **outs)
    if with_lanes:
        return g.nodes, g.lanes
    return g.nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", help="ComfyUI base URL for object_info, to also write UI-format workflows")
    args = ap.parse_args()
    FILES.clear()
    FILES.update(OFFICIAL_FILES)  # committed workflows always use the official file names
    (OUT / "api").mkdir(parents=True, exist_ok=True)
    for fam in ["auto", *BRANCHES]:
        name = f"army_{fam}.json"
        wf, lanes = build(fam, with_lanes=True)
        (OUT / "api" / name).write_text(json.dumps(wf, indent=1) + "\n")
        if args.ui:
            from api_to_ui import convert

            (OUT / name).write_text(json.dumps(convert(wf, lanes, args.ui), indent=1) + "\n")
        print("wrote", name, len(wf), "nodes")


if __name__ == "__main__":
    main()
