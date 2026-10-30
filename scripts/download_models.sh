#!/usr/bin/env bash
# Download the base models for each family into a ComfyUI models dir.
#   scripts/download_models.sh /path/to/ComfyUI/models [flux qwen moss ltx h3]
# Community LoRAs are NOT downloaded here: ArmyLoraApply fetches each one from Hugging Face
# the first time the router drafts it (into models/loras/army/<repo>/).
# Some repos are gated: accept the licence on huggingface.co and `hf auth login` first.
set -euo pipefail
M=${1:?usage: download_models.sh COMFYUI_MODELS_DIR [families...]}
shift || true
FAMS=${*:-flux qwen moss ltx h3}
HF=https://huggingface.co
get() {  # get <subdir> <url> [outname]
  mkdir -p "$M/$1"
  local out=${3:-$(basename "$2")}
  [ -s "$M/$1/$out" ] && { echo "have $1/$out"; return; }
  echo "get  $1/$out"
  curl -fL --retry 5 -C - ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} -o "$M/$1/$out" "$2"
}
for f in $FAMS; do case $f in
  flux)  # FLUX.2 [klein] 9B base fp8 (20 steps, CFG 3.5)
    get diffusion_models $HF/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors
    get text_encoders $HF/Comfy-Org/flux2-klein-9B/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors
    get vae $HF/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors ;;
  qwen)  # Qwen-Image 2.1 + Viggle 6-step turbo LoRA
    get diffusion_models $HF/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors
    get text_encoders $HF/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_w4a8.safetensors
    get vae $HF/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors
    get loras $HF/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors qwen-image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors ;;
  moss)  # MOSS-Image SFT int8 + rapid-moss 4-step adapter (needs the rapid-moss custom node)
    get diffusion_models $HF/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_sft_int8_convrot.safetensors
    get text_encoders $HF/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors
    get vae $HF/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae.safetensors
    for x in lora.safetensors time_embedding.safetensors dataspace.safetensors; do
      get rapid_moss/rapid-moss-4step $HF/darask0/MOSS-Image_4-step_adapter_lora/resolve/main/$x; done ;;
  ltx)   # LTX-2.3 22B distilled fp8 (video+audio) + Gemma 3 12B text encoder + x2 latent upscaler
    get checkpoints $HF/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors
    get text_encoders $HF/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
    get latent_upscale_models $HF/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors ;;
  h3)    # MiniMax H3 int8 (video+audio) + lightx2v 4-step turbo LoRA
    get diffusion_models $HF/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
    get text_encoders $HF/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    get vae $HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors
    get vae $HF/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors
    get loras $HF/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors ;;
  *) echo "unknown family $f"; exit 1 ;;
esac; done
