# Demo results

Every prompt in `demo/prompts.txt` run through `workflows/army_auto.json` with the router on `auto`,
seed 42, on one RTX 3090 Ti (24 GB). Wall time includes model loading from disk the first time a
family is used, so the first run per model is much slower than the rest.

| output | prompt | mode | model | drafted LoRAs | route ms | wall s |
|---|---|---|---|---|---|---|
| [00_flux2_klein9b.png](results/00_flux2_klein9b.png) | a cute sticker of a red panda eating ramen, pixel art | image | FLUX.2 [klein] 9B | Stickers (Redmond) x0.90, Pixel Art (Redmond) x0.80 | 3.0 | 73 |
| [01_flux2_klein9b.png](results/01_flux2_klein9b.png) | isometric diorama of a cozy wizard library with floating books | image | FLUX.2 [klein] 9B | Isometric (Redmond) x0.90 | 13.2 | 1180 |
| [02_flux2_klein9b.png](results/02_flux2_klein9b.png) | a crochet amigurumi octopus reading a newspaper on a wooden table | image | FLUX.2 [klein] 9B | Crochet Amigurumi (Redmond) x0.90 | 42.7 | 82 |
| [03_flux2_klein9b.png](results/03_flux2_klein9b.png) | claymation snail astronaut planting a flag on a mushroom planet | image | FLUX.2 [klein] 9B | Clay Animation (Redmond) x0.85 | 7.3 | 70 |
| [04_flux2_klein9b.png](results/04_flux2_klein9b.png) | technical blueprint wireframe of a vintage steam locomotive | image | FLUX.2 [klein] 9B | Blueprint Wireframe (DeverStyle) x0.90 | 6.8 | 72 |
