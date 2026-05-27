# NVIDIA PiD + FLUX2 VAE Upscaler

4x image Upscaler based on NVIDIA's PiD (note: [license](https://huggingface.co/nvidia/PiD/commit/b87dba45e5a2b2a18bac9515fca883f52b957558) [is](https://huggingface.co/nvidia/PiD/commit/40e08ea3b2c1d89f0e00f92a775bb310b9a1c497) [volatile](https://huggingface.co/nvidia/PiD/commit/1c6eee3132182c408b8b0dc52b2a2151d8bc07d9)) and FLUX2 VAE (Apache 2.0)

## Usage

Get a GPU with at least 24 GB VRAM, better 32 GB, and:

```bash
pip install -r requirements.txt

./upscale.sh input.jpg
```

### Full Options

```
python upscale_PiD_flux2vae.py (options)
```


```
--input_path       Input image path (required)
--input_resolution Square resolution for center-crop before VAE encode (default: 512)
--keep_input_size  Keep original image size (no square crop)
--prompt           Caption for conditioning (default: "high quality photo")
--scale            Upscale factor (default: 4)
--pid_ckpt_type    Model variant: "2k" or "2kto4k" (default: "2k")
--degrade_sigmas   Noise sigma value(s) to inject [0,1] (default: [0.0])
--cfg_scale        CFG guidance scale (default: 1.0)
--pid_inference_steps  Inference steps for PiD (default: 4)
--seed             Random seed (default: 5)
--save_format      Output format: "png" or "jpg" (default: "jpg")
```

## Model Weights

Downloaded automatically to `weights/` directory:

- FLUX.2 VAE: `black-forest-labs/FLUX.2-dev` 
- PiD 2k: `nvidia/PiD` 
- PiD 2kto4k: `nvidia/PiD` 

The pipeline also downloads gemma-2-2b-it. That's the text encoder for PiD.

## Output

- Upscaled image saved as `*_upscaled` 

## Changelog 
- Cloned https://github.com/nv-tlabs/PiD, then Claude Code got rid of all training related stuff, result is models.py (1.5k lines) and upscale_PiD_flux2vae.py. No other dependencies than the ones in requirements. It's basically just torch, numpy and transformers
