"""Image upscaler: Flux2 VAE encoder + PiD pixel decoder.

Pipeline: input image → center-crop + bicubic resize to --input_resolution →
Flux2 VAE encode → optional noise injection (--degrade_sigmas) →
PiD pixel decoder at --scale × input_resolution.

Model weights are downloaded automatically from HuggingFace on first run.

Usage:
    python upscale_PiD_flux2vae.py \
        --input_path /path/to/image.jpg \
        --input_resolution 512 \
        --degrade_sigmas 0.0 \
        --cfg_scale 2.75 --pid_inference_steps 25 --scale 4
"""

import argparse
import logging
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from models import (
    Flux2VAEInterface,
    PidModel,
    PidModelConfig,
    PidNet,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.enable_grad(False)

FLUX2_VAE_REPO = "black-forest-labs/FLUX.2-dev"
FLUX2_VAE      = "ae.safetensors"

FLUX2_PID_REPO = "nvidia/PiD"
FLUX2_CHECKPOINTS = {
    "2k":     "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
    # "2kto4k": "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth",
    # June update to checkpoint
    "2kto4k": "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step_2606/model_ema_bf16.pth",
}


def weights_path(repo_id: str, hf_path: str) -> str:
    return os.path.join("weights", repo_id.split("/")[-1], hf_path)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] in [-1, 1] to PIL Image."""
    tensor = (tensor.float().clamp(-1, 1) + 1) * 127.5
    arr = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)


def color_match(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Channel-wise mean+std match: pull output color distribution toward reference.

    Both tensors are [C, H, W] float32 in [-1, 1]. Output and reference may differ in resolution.
    """
    out = output.clone()
    for c in range(output.shape[0]):
        ref_mean = reference[c].mean()
        ref_std = reference[c].std().clamp(min=1e-6)
        out_mean = output[c].mean()
        out_std = output[c].std().clamp(min=1e-6)
        out[c] = (output[c] - out_mean) / out_std * ref_std + ref_mean
    return out.clamp(-1, 1)


def save_image(sample: torch.Tensor, save_path: str, quality: int = 95) -> str:
    """Save [C, H, W] or [C, 1, H, W] tensor in [-1, 1]. Format inferred from extension."""
    if sample.dim() == 4:
        sample = sample.squeeze(1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img = tensor_to_pil(sample)
    if save_path.lower().endswith((".jpg", ".jpeg")):
        img.save(save_path, quality=quality)
    else:
        img.save(save_path)
    return save_path


def load_input_image(
    path: str,
    resolution: int,
    keep_input_size: bool = False,
    pad_to_multiple: int = 16,
) -> torch.Tensor:
    """Load image and return [1, 3, H, W] float32 in [-1, 1] on CPU."""
    img = Image.open(path).convert("RGB")
    if keep_input_size:
        w, h = img.size
        new_w = (w // pad_to_multiple) * pad_to_multiple
        new_h = (h // pad_to_multiple) * pad_to_multiple
        if new_w == 0 or new_h == 0:
            raise ValueError(
                f"Image {path} size {w}x{h} is smaller than pad_to_multiple={pad_to_multiple}."
            )
        if (new_w, new_h) != (w, h):
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))
    else:
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((resolution, resolution), Image.BICUBIC)

    arr = np.asarray(img, np.uint8).astype("float32")
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return t


def vae_decode(model, latent_4d: torch.Tensor) -> torch.Tensor:
    """Wrap model.vae_encoder.decode; handles 5D <-> 4D shape contract.

    Input  latent_4d: [B, C, zH, zW]
    Output recon:     [B, 3, H, W] in [-1, 1]
    """
    z5 = latent_4d.unsqueeze(2)          # [B, C, 1, zH, zW]
    recon5 = model.vae_encoder.decode(z5)  # [B, 3, 1, H, W]
    if recon5.ndim == 5:
        recon5 = recon5[:, :, 0]          # [B, 3, H, W]
    return recon5


def add_noise(
    clean_latent: torch.Tensor, sigma: float, generator: torch.Generator
) -> torch.Tensor:
    """x_t = (1 - sigma) * x_0 + sigma * eps."""
    if sigma <= 0.0:
        return clean_latent
    noise = torch.randn(
        clean_latent.shape,
        generator=generator,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    return (1.0 - sigma) * clean_latent + sigma * noise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="From-clean demo: image -> Flux2 VAE encode -> optional noise -> PiD pixel decoder"
    )

    parser.add_argument(
        "--pid_ckpt_type",
        type=str,
        choices=["2k", "2kto4k"],
        default="2k",
        help="'2k' = original 2048px decoders; '2kto4k' = multi-res-trained (1024→4K).",
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="high quality photo")
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=512,
        help="Square resolution for center-crop + bicubic resize before VAE encode. "
             "Ignored when --keep_input_size is set.",
    )
    parser.add_argument("--keep_input_size", action="store_true")
    parser.add_argument(
        "--degrade_sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.1],
        help="Sigma value(s) in [0,1] to inject into the clean latent. "
             "0.0 = clean round-trip.",
    )

    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--pid_inference_steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument(
        "--save_format", type=str, choices=["png", "jpg"], default="jpg"
    )
    return parser

def download_models(ckpt_type: str):
    for repo_id, hf_path in [
        (FLUX2_VAE_REPO, FLUX2_VAE),
        (FLUX2_PID_REPO, FLUX2_CHECKPOINTS[ckpt_type]),
    ]:
        local_dir = os.path.join("weights", repo_id.split("/")[-1])
        dst = os.path.join(local_dir, hf_path)
        if not os.path.exists(dst):
            logger.info(f"Downloading {hf_path} from {repo_id} ...")
            hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"  -> {dst}")


def build_flux2_model(ckpt_type: str) -> PidModel:
    config = PidModelConfig(
        precision="bfloat16",
        input_caption_key="caption",
        text_encoder_name="gemma-2-2b-it",
        caption_channels=2304,
        model_max_length=300,
        chi_prompt=[
            'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
        fm_timescale=1000.0,
        image_size=2048,
        negative_prompt=(
            "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, "
            "cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, "
            "watermark"
        ),
        lq_condition_type="latent",
        state_ch=128,
        sample_steps=4,
        t_schedule=[0.999, 0.866, 0.634, 0.342, 0.0],
        dynamic_shift=(
            {"base_shift": 4.0, "base_image_size_for_shift_calc": 1024}
            if ckpt_type == "2kto4k" else None
        ),
        net=PidNet(
            num_groups=24,
            hidden_size=1536,
            pixel_hidden_size=16,
            pixel_attn_hidden_size=1152,
            pixel_num_groups=16,
            patch_depth=14,
            patch_size=16,
            txt_embed_dim=2304,
            txt_max_length=300,
            lq_in_channels=0,
            lq_latent_channels=128,
            lq_interval=2,
            latent_spatial_down_factor=16,
        ),
        tokenizer=Flux2VAEInterface(
            vae_pth=weights_path(FLUX2_VAE_REPO, FLUX2_VAE),
        ),
    )
    return PidModel(config).cuda()


def main():
    parser = build_parser()
    args = parser.parse_args()

    download_models(args.pid_ckpt_type)

    logger.info(
        f"Backbone(VAE): flux2  input_resolution: {args.input_resolution}  "
        f"sigmas: {sorted(args.degrade_sigmas)}  scale: {args.scale}  "
        f"pid_steps: {args.pid_inference_steps}"
    )

    checkpoint_path = weights_path(FLUX2_PID_REPO, FLUX2_CHECKPOINTS[args.pid_ckpt_type])

    logger.info(f"Loading pixel decoder from {checkpoint_path} ...")
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    model = build_flux2_model(args.pid_ckpt_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False), strict=False)
    model.eval()
    torch.cuda.empty_cache()

    image_path = args.input_path
    caption = args.prompt
    bn = os.path.splitext(os.path.basename(image_path))[0]
    input_dir = os.path.dirname(os.path.abspath(image_path))
    debug_dir = os.path.join(input_dir, f"{bn}_debug")

    input_tensor = load_input_image(
        image_path, args.input_resolution, keep_input_size=args.keep_input_size
    ).to(dtype=torch.bfloat16, device="cuda")

    with torch.no_grad():
        clean_latent = model.encode_lq_latent(input_tensor)  # [1, C, zH, zW]

    vae_compression = int(model.vae_encoder.spatial_compression_factor)
    vae_h = int(clean_latent.shape[-2]) * vae_compression
    vae_w = int(clean_latent.shape[-1]) * vae_compression
    target_hw = (vae_h * args.scale, vae_w * args.scale)

    logger.info(
        f"Clean latent shape={tuple(clean_latent.shape)}  "
        f"vae_native=({vae_h}x{vae_w})  target_hw={target_hw}  "
        f"caption={caption[:60]!r}"
    )

    input_save = input_tensor.float().cpu().squeeze(0).clamp(-1, 1)
    save_image(input_save, os.path.join(debug_dir, "input", f"{bn}.{args.save_format}"))

    for sigma in sorted(args.degrade_sigmas):
        sigma_label = f"sigma_{sigma:.3f}"

        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        latent = add_noise(clean_latent.float(), float(sigma), gen).to(dtype=torch.bfloat16)

        with torch.no_grad():
            vae_img = vae_decode(model, latent)  # [1, 3, R, R] in [-1, 1]

        lq_placeholder = torch.zeros_like(vae_img, dtype=torch.bfloat16, device="cuda")
        data_batch = {
            model.config.input_caption_key: [caption],
            "LQ_video_or_image": lq_placeholder,
            "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
            "degrade_sigma": torch.tensor([float(sigma)], device="cuda", dtype=torch.float32),
        }
        with torch.no_grad():
            samples_out = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=args.cfg_scale,
                num_steps=args.pid_inference_steps,
                seed=args.seed,
                shift=args.shift,
                image_size=target_hw,
            )
        img_upscaled = color_match(samples_out[0].float().cpu().clamp(-1, 1), input_save)

        sigma_suffix = f"_{sigma_label}" if len(args.degrade_sigmas) > 1 else ""
        img_upscaled_path = os.path.join(input_dir, f"{bn}_upscaled{sigma_suffix}.{args.save_format}")
        save_image(img_upscaled, img_upscaled_path)

        logger.info(f"sigma={sigma:.3f} -> upscaled={img_upscaled_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
