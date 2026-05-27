#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: upscale.sh <input_file>}"

python3 upscale_PiD_flux2vae.py --input_path "$INPUT" --keep_input_size --pid_ckpt_type 2kto4k
