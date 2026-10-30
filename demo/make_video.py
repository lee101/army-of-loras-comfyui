"""Assemble the 2-minute demo video from real outputs + a Kokoro TTS voiceover.

    python demo/make_video.py demo/results/runs.jsonl demo/army-of-loras-demo.mp4

Needs: pillow, numpy, soundfile, kokoro-onnx (+ phonemizer-fork), ffmpeg on PATH, and the Kokoro
v1.0 onnx model + voices (https://github.com/thewh1teagle/kokoro-onnx/releases). Each segment's
length is its narration length, so the video always matches the voice.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W, H = 1920, 1080
BG, PANEL, LINE = (15, 17, 21), (23, 26, 33), (42, 47, 58)
TEXT, DIM, ACCENT, OK = (230, 232, 238), (154, 163, 178), (124, 196, 255), (126, 224, 161)
FAMILY_LABEL = {"flux2_klein9b": "FLUX.2 [klein] 9B", "qwen_image21": "Qwen-Image 2.1", "moss": "MOSS-Image",
                "ltx23": "LTX-2.3", "h3": "MiniMax H3"}
KOKORO = Path(os.environ.get("KOKORO_DIR", Path.home() / ".cache/kokoro"))
VOICE = os.environ.get("KOKORO_VOICE", "af_heart")


def font(size: int, bold: bool = False, mono: bool = False):
    names = (["DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"] if mono else
             ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"])
    for d in ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/usr/share/fonts/dejavu"]:
        p = Path(d) / names[0]
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def wrap(d, xy, text, f, fill, width_px, line_gap=8):
    x, y = xy
    avg = f.getlength("abcdefghijklmnopqrstuvwxyz") / 26
    for para in text.split("\n"):
        for line in textwrap.wrap(para, max(10, int(width_px / avg))) or [""]:
            d.text((x, y), line, font=f, fill=fill)
            y += f.size + line_gap
    return y


def footer(d, text="army-of-loras-comfyui  ·  ComfyUI + pybed"):
    d.text((60, H - 50), text, font=font(22), fill=DIM)


# ---------------------------------------------------------------- slides
def slide_title() -> Image.Image:
    img, d = canvas()
    d.text((120, 330), "Army of LoRAs", font=font(120, True), fill=TEXT)
    d.text((126, 480), "one prompt -> the right model + a squad of community LoRAs", font=font(44), fill=ACCENT)
    d.text((126, 560), "FLUX.2 [klein]  ·  Qwen-Image 2.1  ·  MOSS-Image  ·  LTX-2.3  ·  MiniMax H3", font=font(34), fill=DIM)
    d.text((126, 620), "routed in ~2 ms with pybed static embeddings, inside ComfyUI", font=font(34), fill=DIM)
    footer(d)
    return img


def slide_problem(catalogs: dict) -> Image.Image:
    img, d = canvas()
    d.text((120, 120), "Thousands of LoRAs. Which one?", font=font(64, True), fill=TEXT)
    qs = ["which LoRA fits this prompt?", "trained on which base model?", "what strength?", "which trigger word?",
          "image or video?"]
    for i, q in enumerate(qs):
        d.text((140, 260 + i * 78), "?  " + q, font=font(42), fill=DIM if i % 2 else TEXT)
    n = sum(len(c["loras"]) for c in catalogs.values())
    d.text((140, 700), f"755 adapters scraped from Hugging Face -> 227 SFW + licensed -> {n} hand-picked",
           font=font(34), fill=ACCENT)
    footer(d)
    return img


def slide_how() -> Image.Image:
    img, d = canvas()
    d.text((100, 70), "How the router decides", font=font(60, True), fill=TEXT)
    boxes = [("prompt", "SFW gate"), ("pybed embed", "int8 static, 512-d"), ("image or video?", "motion cues"),
             ("score the army", "cos(positive) - avoid + keyword"), ("pick the model", "best army fit"),
             ("draft LoRAs", "1 per group, by confidence"), ("ComfyUI graph", "only that branch runs")]
    bw, bh, gap, y0 = 236, 150, 26, 330
    x = 70
    for i, (a, b) in enumerate(boxes):
        d.rounded_rectangle((x, y0, x + bw, y0 + bh), 18, fill=PANEL, outline=ACCENT if i in (1, 3, 4) else LINE, width=3)
        d.text((x + 14, y0 + 26), a, font=font(23, True), fill=TEXT)
        wrap(d, (x + 16, y0 + 72), b, font(21), DIM, bw - 30, 4)
        if i < len(boxes) - 1:
            ax = x + bw + 4
            d.line((ax, y0 + bh / 2, ax + gap - 8, y0 + bh / 2), fill=ACCENT, width=4)
            d.polygon([(ax + gap - 6, y0 + bh / 2 - 9), (ax + gap - 6, y0 + bh / 2 + 9), (ax + gap + 4, y0 + bh / 2)], fill=ACCENT)
        x += bw + gap
    f = font(30, mono=True)
    d.text((120, 600), "score  = max cos(prompt, positive) - 0.5 * max cos(prompt, avoid) + 0.25 * keyword", font=f, fill=OK)
    d.text((120, 650), "family = 0.5 * cos(prompt, family) + 0.5 * mean(top-2 LoRA scores)", font=f, fill=OK)
    d.text((120, 740), "no transformer forward pass: token lookup + mean pool, ~0.01 ms per phrase on CPU", font=font(32), fill=DIM)
    d.text((120, 790), "lazy inputs: the other four models are never loaded", font=font(32), fill=DIM)
    footer(d)
    return img


def slide_army(catalogs: dict) -> Image.Image:
    img, d = canvas()
    d.text((80, 50), "The army: 65 community LoRAs, 5 base models", font=font(52, True), fill=TEXT)
    colw = (W - 120) // 5
    for i, (fam, c) in enumerate(catalogs.items()):
        x = 60 + i * colw
        d.text((x, 150), FAMILY_LABEL[fam], font=font(28, True), fill=ACCENT)
        d.text((x, 188), "video + audio" if c["mode"] == "video" else "image", font=font(20), fill=DIM)
        y = 228
        size = 17 if len(c["loras"]) > 14 else 21
        for l in c["loras"]:
            name = l["name"].split(" (")[0]
            d.text((x, y), name, font=font(size), fill=TEXT)
            y += size + 9
    footer(d, "all SFW · open or base-model community licences · credits for every author in docs/CREDITS.md")
    return img


def fit(im: Image.Image, box: tuple[int, int]) -> Image.Image:
    im = im.convert("RGB")
    s = min(box[0] / im.width, box[1] / im.height)
    return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)


def slide_result(run: dict, media_box=(1040, 1000)) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Right-hand plan panel; returns the frame and the media rectangle (x, y, w, h) on the left."""
    img, d = canvas()
    plan = run["plan"]
    px = 40 + media_box[0] + 40
    pw = W - px - 50
    d.rounded_rectangle((px - 20, 40, W - 30, H - 40), 18, fill=PANEL, outline=LINE, width=2)
    y = wrap(d, (px, 70), "“" + plan["original_prompt"] + "”", font(30, True), TEXT, pw)
    y += 18
    chips = [plan["mode"], FAMILY_LABEL[plan["family"]], f"routed in {plan['elapsed_ms']:.1f} ms"]
    x = px
    for i, c in enumerate(chips):
        f = font(24, True)
        w = f.getlength(c) + 28
        d.rounded_rectangle((x, y, x + w, y + 40), 20, fill=OK if i < 2 else BG, outline=OK)
        d.text((x + 14, y + 7), c, font=f, fill=BG if i < 2 else OK)
        x += w + 12
    y += 70
    fams = sorted((plan.get("family_scores") or {}).items(), key=lambda kv: -kv[1])
    if fams:
        d.text((px, y), "MODEL FIT", font=font(20, True), fill=DIM)
        y += 32
        top = max(0.01, fams[0][1])
        for f_, v in fams:
            d.text((px, y), FAMILY_LABEL[f_], font=font(22), fill=TEXT if f_ == plan["family"] else DIM)
            bx = px + 250
            d.rounded_rectangle((bx, y + 6, bx + 360, y + 22), 6, fill=BG)
            d.rounded_rectangle((bx, y + 6, bx + max(8, int(360 * max(v, 0) / top)), y + 22), 6,
                                fill=ACCENT if f_ == plan["family"] else LINE)
            d.text((bx + 375, y), f"{v:.2f}", font=font(22, mono=True), fill=DIM)
            y += 36
        y += 14
    d.text((px, y), "DRAFTED LORAS", font=font(20, True), fill=DIM)
    y += 34
    if not plan["loras"]:
        y = wrap(d, (px, y), "none: nothing in this army fits, so the base model renders it plain "
                 "(lettering stays clean)", font(24), DIM, pw)
    for l in plan["loras"]:
        d.text((px, y), f"{l['strength']:.2f}x", font=font(28, True, mono=True), fill=OK)
        d.text((px + 120, y), l["name"], font=font(28, True), fill=TEXT)
        y += 38
        y = wrap(d, (px + 120, y), f"score {l['score']:.2f}  ~ \"{l['matched']}\"" +
                 (f"  + keyword '{l['keyword']}'" if l.get("keyword") else ""), font(21), DIM, pw - 120, 4)
        d.text((px + 120, y), f"{l['repo']}  ·  {l['license']}", font=font(19), fill=ACCENT)
        y += 40
    if run.get("seconds"):
        d.text((px, H - 100), f"generated in {run['seconds']:.0f} s on one RTX 3090 Ti", font=font(22), fill=DIM)
    return img, (40, (H - media_box[1]) // 2, media_box[0], media_box[1])


def slide_end() -> Image.Image:
    img, d = canvas()
    d.text((120, 300), "Army of LoRAs", font=font(96, True), fill=TEXT)
    lines = ["open source (MIT): nodes, router, catalogs, workflows, this video's script",
             "github.com/lee101/army-of-loras-comfyui",
             "pybed: github.com/lee101/pybed",
             "voice: Kokoro-82M · assembled with ffmpeg · every LoRA author credited in docs/CREDITS.md"]
    for i, t in enumerate(lines):
        d.text((126, 450 + i * 60), t, font=font(34), fill=ACCENT if i == 1 else DIM)
    footer(d)
    return img


# ---------------------------------------------------------------- audio
def tts_segments(texts: list[str], out_dir: Path) -> list[Path]:
    from kokoro_onnx import EspeakConfig, Kokoro

    cfg = None
    sys_lib = Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
    if sys_lib.exists():  # the wheel-bundled espeak data path is broken on some installs
        cfg = EspeakConfig(lib_path=str(sys_lib), data_path=str(sys_lib.parent / "espeak-ng-data"))
    k = Kokoro(str(KOKORO / "kokoro-v1.0.onnx"), str(KOKORO / "voices-v1.0.bin"), espeak_config=cfg)
    paths = []
    for i, t in enumerate(texts):
        p = out_dir / f"vo_{i:02d}.wav"
        samples, sr = k.create(t, voice=VOICE, speed=1.08, lang="en-us")
        pad = np.zeros(int(sr * 0.35), dtype=samples.dtype)
        sf.write(p, np.concatenate([pad[: len(pad) // 2], samples, pad]), sr)
        paths.append(p)
    return paths


def duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def render_segment(i: int, seg: dict, vo: Path, out: Path, work: Path) -> Path:
    dur = duration(vo)
    frame = work / f"frame_{i:02d}.png"
    seg["frame"].save(frame)
    dst = out / f"seg_{i:02d}.mp4"
    common = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", f"{dur:.2f}"]
    media = seg.get("media")
    if media and media.suffix == ".mp4":
        x, y, w, h = seg["rect"]
        has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                         "stream=index", "-of", "csv=p=0", str(media)], capture_output=True,
                                        text=True).stdout.strip())
        audio = ("[2:a]volume=0.30[va];[1:a][va]amix=inputs=2:duration=first:normalize=0[a]" if has_audio
                 else "[1:a]anull[a]")
        ff("-loop", "1", "-i", str(frame), "-i", str(vo), "-stream_loop", "-1", "-i", str(media),
           "-filter_complex",
           f"[2:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0f1115[v];"
           f"[0:v][v]overlay={x}:{y}[o];{audio}", "-map", "[o]", "-map", "[a]", *common, str(dst))
    else:
        if media:
            img = seg["frame"].copy()
            x, y, w, h = seg["rect"]
            im = fit(Image.open(media), (w, h))
            img.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))
            img.save(frame)
        ff("-loop", "1", "-i", str(frame), "-i", str(vo), "-map", "0:v", "-map", "1:a", *common, str(dst))
    return dst


def main():
    runs_path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "demo/results/runs.jsonl")
    out_path = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "demo/army-of-loras-demo.mp4")
    script = json.loads((ROOT / "demo/script.json").read_text())
    from army import router

    catalogs = router.all_catalogs()
    results_dir = runs_path.parent
    runs = {}
    for line in runs_path.read_text().splitlines():
        r = json.loads(line)
        if r.get("files"):
            r["plan"] = json.loads((results_dir / r["plan_file"]).read_text())
            runs[r["prompt"]] = r

    segs = []
    for s in script:
        kind = s["kind"]
        if kind == "result":
            r = runs.get(s["prompt"])
            if not r:
                print("skip (no result):", s["prompt"])
                continue
            frame, rect = slide_result(r)
            segs.append({"text": s["text"], "frame": frame, "rect": rect, "media": results_dir / r["media_file"]})
        elif kind == "image":
            img, _ = canvas()
            im = fit(Image.open(ROOT / s["path"]), (W - 80, H - 80))
            img.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
            segs.append({"text": s["text"], "frame": img})
        else:
            frame = {"title": slide_title, "how": slide_how, "end": slide_end}.get(kind)
            frame = frame() if frame else {"problem": slide_problem, "army": slide_army}[kind](catalogs)
            segs.append({"text": s["text"], "frame": frame})

    work = ROOT / "demo/build"
    work.mkdir(parents=True, exist_ok=True)
    vos = tts_segments([s["text"] for s in segs], work)
    total = sum(duration(v) for v in vos)
    print(f"{len(segs)} segments, {total:.1f} s of narration")
    parts = [render_segment(i, s, v, work, work) for i, (s, v) in enumerate(zip(segs, vos))]
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out_path))
    (ROOT / "demo/voiceover.txt").write_text("\n\n".join(s["text"] for s in segs) + "\n")
    print("wrote", out_path, f"{total:.1f} s")


if __name__ == "__main__":
    main()
