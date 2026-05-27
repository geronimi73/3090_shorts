"""Self-contained model definitions for the Flux2 upscaling pipeline.

Layers mirror pid/_src/tokenizers/flux2_vae.py but with all imaginaire /
pid._src infrastructure removed:
  - all distributed / context-parallel code dropped (single-GPU inference only)
  - checkpoint loading replaced with safetensors.torch.load_file directly
  - VideoTokenizerInterface ABC replaced with a plain base class
  - LazyCall / LazyDict config layer omitted
  - PidNet, PidModel and all dependencies inlined (no pid._src imports)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import attrs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# Raw Flux 2 AutoEncoder (adapted from official Flux 2 repo)
# =============================================================================


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        q = self.q(h).reshape(B, 1, C, H * W).transpose(2, 3)
        k = self.k(h).reshape(B, 1, C, H * W).transpose(2, 3)
        v = self.v(h).reshape(B, 1, C, H * W).transpose(2, 3)
        h = F.scaled_dot_product_attention(q, k, v)
        return h.transpose(2, 3).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(self.norm(x)))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        return self.nin_shortcut(x) + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 before interpolate for bfloat16 safety
        return self.conv(F.interpolate(x.float(), scale_factor=2.0, mode="nearest").type_as(x))


class Encoder(nn.Module):
    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        return self.quant_conv(self.conv_out(swish(self.norm_out(h))))


class Decoder(nn.Module):
    def __init__(self, ch, out_ch, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        upscale_dtype = next(self.up.parameters()).dtype
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))


class AutoEncoder(nn.Module):
    """Flux 2 AutoEncoder: 32 z-channels, 8× spatial encoder + 2×2 patchify → 128 ch at 16×."""

    def __init__(self, params: AutoEncoderParams = None):
        super().__init__()
        if params is None:
            params = AutoEncoderParams()
        self.encoder = Encoder(
            resolution=params.resolution, in_channels=params.in_channels, ch=params.ch,
            ch_mult=params.ch_mult, num_res_blocks=params.num_res_blocks, z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            ch=params.ch, out_ch=params.out_ch, ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks, z_channels=params.z_channels,
        )
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps, momentum=self.bn_momentum, affine=False, track_running_stats=True,
        )

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z: torch.Tensor) -> torch.Tensor:
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            assert x.shape[2] == 1
            x, video_fmt = x.squeeze(2), True
        else:
            video_fmt = False
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=self.ps[0], pj=self.ps[1])
        z = self.normalize(z)
        return z.unsqueeze(2) if video_fmt else z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 5:
            assert z.shape[2] == 1
            z, video_fmt = z.squeeze(2), True
        else:
            video_fmt = False
        z = self.inv_normalize(z)
        z = rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=self.ps[0], pj=self.ps[1])
        dec = self.decoder(z)
        return dec.unsqueeze(2) if video_fmt else dec


# =============================================================================
# Factory function
# =============================================================================


# =============================================================================
# Flux2VAE dtype/AMP wrapper
# =============================================================================


class Flux2VAE:
    def __init__(
        self,
        vae_pth: str = "./checkpoints/flux2_ae.safetensors",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        is_amp: bool = True,
    ):
        self.dtype = dtype
        with torch.device("meta"):
            model = AutoEncoder()
        logger.info(f"Loading Flux 2 VAE from {vae_pth}")
        model.load_state_dict(safetensors_load_file(vae_pth), assign=True)
        self.model = model.to(device).eval().requires_grad_(False)
        if not is_amp:
            self.model = self.model.to(dtype=dtype)
            self.context = nullcontext()
        else:
            self.context = torch.amp.autocast("cuda", dtype=dtype)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        in_dtype = images.dtype
        with self.context:
            latent = self.model.encode(images.to(self.dtype))
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, zs: torch.Tensor) -> torch.Tensor:
        in_dtype = zs.dtype
        with self.context:
            recon = self.model.decode(zs.to(self.dtype))
        return recon.to(in_dtype)


# =============================================================================
# Flux2VAEInterface — pipeline-compatible tokenizer
# =============================================================================


class Flux2VAEInterface:
    """Flux 2 VAE: 128-channel latents at 16× spatial compression."""

    def __init__(self, vae_pth, **kwargs):
        self.model = Flux2VAE(
            vae_pth = vae_pth,
            dtype = torch.bfloat16,
            is_amp = False,
        )

    @property
    def dtype(self):
        return self.model.dtype

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 5:
            assert state.shape[2] == 1
            return self.model.encode(state.squeeze(2)).unsqueeze(2)
        return self.model.encode(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 5:
            assert latent.shape[2] == 1
            return self.model.decode(latent.squeeze(2)).unsqueeze(2)
        return self.model.decode(latent)

    @property
    def spatial_compression_factor(self):
        return 16

    @property
    def latent_ch(self):
        return 128


# =============================================================================
# PixelDiT T2I network (from pid/_src/networks/pixeldit_official.py)
# =============================================================================


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        if t_freq.dtype != mlp_dtype:
            t_freq = t_freq.to(mlp_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        return x


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale=16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def precompute_freqs_cis_2d_ntk(
    dim: int,
    height: int,
    width: int,
    ref_grid_h: int,
    ref_grid_w: int,
    theta: float = 10000.0,
    scale: float = 16.0,
):
    dim_axis = dim // 2
    h_scale = height / ref_grid_h
    w_scale = width / ref_grid_w
    h_ntk = h_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    w_ntk = w_scale ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    h_theta = theta * h_ntk
    w_theta = theta * w_ntk

    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)

    freqs_w = 1.0 / (w_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_h = 1.0 / (h_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    x_freqs = torch.outer(x_pos, freqs_w).float()
    y_freqs = torch.outer(y_pos, freqs_h).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    freqs_cis = freqs_cis.reshape(height * width, -1)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2).contiguous()
        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return x


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 768, norm_layer=None, bias: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_patch(self, patch_size: int, device, dtype):
        key = ("patch", patch_size)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device=device, dtype=dtype)
        pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, patch_size)
        pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
        self._pos_cache[key] = pos
        return pos

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        key = ("image", height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device=device, dtype=dtype)
        if height == width:
            pos_np = get_2d_sincos_pos_embed(self.hidden_size_output, height)
        else:
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)
        pos = torch.from_numpy(pos_np).to(device=device, dtype=dtype)
        self._pos_cache[key] = pos
        return pos

    def forward(self, inputs: torch.Tensor, img_height: int = None, img_width: int = None, patch_size: int = None):
        if inputs.dim() == 3:
            batch_tokens, p2, _ = inputs.shape
            patch_sz = int(p2**0.5)
            pos = self._fetch_pixel_pos_patch(patch_sz, inputs.device, inputs.dtype)
            x = self.proj(inputs)
            x = x + pos.unsqueeze(0)
            return x
        elif inputs.dim() == 4:
            assert img_height is not None and img_width is not None and patch_size is not None
            B, C, H, W = inputs.shape
            assert H == img_height and W == img_width
            assert (H % patch_size == 0) and (W % patch_size == 0)
            Hs, Ws = H // patch_size, W // patch_size
            P2 = patch_size * patch_size
            x = inputs.permute(0, 2, 3, 1).contiguous()
            x = self.proj(x)
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)
            pos_full = pos_full.view(H, W, self.hidden_size_output)
            x = x + pos_full.unsqueeze(0)
            x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.view(B * Hs * Ws, P2, self.hidden_size_output)
            return x
        else:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B*L,P2,C] or [B,C,H,W]")


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: Optional[int] = None,
        attn_num_heads: Optional[int] = None,
        rope_mode: str = "original",
        rope_ref_grid_h: int = 32,
        rope_ref_grid_w: int = 32,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_grid_h
        self.rope_ref_grid_w = rope_ref_grid_w
        assert self.attn_dim % self.num_heads == 0
        p2 = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, 6 * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        head_dim = self.attn_dim // self.num_heads
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(device)
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(self, x: torch.Tensor, s_cond: torch.Tensor, image_height: int, image_width: int, patch_size: int, mask=None) -> torch.Tensor:
        BL, P2, C = x.shape
        if C != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {C}")
        assert patch_size == self.patch_size
        assert P2 == patch_size * patch_size
        assert (image_height % patch_size == 0) and (image_width % patch_size == 0)
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        cp_size = 1
        assert L % cp_size == 0
        L_local = L // cp_size
        assert s_cond.shape[0] == BL
        assert BL % L_local == 0
        B = BL // L_local
        cond_params = self.adaLN_modulation(s_cond)
        cond_params = cond_params.view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L_local, self.attn_dim)
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        attn_out = self.attn(x_comp, pos_comp, mask)
        attn_flat = self.expand_from_attn(attn_out.view(B * L_local, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        x = x + gate_msa * attn_exp
        mlp_out = self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        return x


class MMDiTJointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.proj_drop_y = nn.Dropout(proj_drop)

    def forward(self, x, y, pos_img, pos_txt=None, attn_mask=None):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape
        assert B == By and C == Cy

        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = qkv_y[0], qkv_y[1], qkv_y[2]
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        qx = qx.transpose(1, 2)
        kx = kx.transpose(1, 2)
        vx = vx.transpose(1, 2)
        qy = qy.transpose(1, 2)
        ky = ky.transpose(1, 2)
        vy = vy.transpose(1, 2)

        q_joint = torch.cat([qy, qx], dim=2)
        k_joint = torch.cat([ky, kx], dim=2)
        v_joint = torch.cat([vy, vx], dim=2)

        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        out_y = out_joint[:, :, :Ny, :]
        out_x = out_joint[:, :, Ny:, :]

        out_y = out_y.transpose(1, 2).reshape(B, Ny, C)
        out_x = out_x.transpose(1, 2).reshape(B, Nx, C)

        out_x = self.proj_drop_x(self.proj_x(out_x))
        out_y = self.proj_drop_y(self.proj_y(out_y))
        return out_x, out_y


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size, groups, mlp_ratio=4.0, adaLN_modulation_img=None, adaLN_modulation_txt=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.groups = groups
        self.head_dim = hidden_size // groups
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = (
            adaLN_modulation_img if adaLN_modulation_img is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )
        self.adaLN_modulation_txt = (
            adaLN_modulation_txt if adaLN_modulation_txt is not None
            else nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        )

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)
        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class PixDiT_T2I(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        rope_mode: str = "original",
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.rope_mode = rope_mode
        self.rope_ref_grid_h = rope_ref_h // self.patch_size
        self.rope_ref_grid_w = rope_ref_w // self.patch_size
        if self.pixel_depth <= 0:
            raise ValueError("PixDiT_T2I expects pixel_depth > 0")

        self.pixel_embedder = PixelTokenEmbedder(in_channels, self.pixel_hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, hidden_size))

        self._shared_cond_adaln = None
        self._shared_cond_adaln_img = None
        self._shared_cond_adaln_txt = None
        self.patch_blocks = nn.ModuleList([
            MMDiTBlockT2I(self.hidden_size, self.num_groups,
                          adaLN_modulation_img=self._shared_cond_adaln_img,
                          adaLN_modulation_txt=self._shared_cond_adaln_txt)
            for _ in range(self.patch_depth)
        ])
        self.pixel_attn_hidden_size = int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(self.pixel_hidden_size, self.hidden_size, patch_size=self.patch_size,
                     num_heads=self.num_groups, mlp_ratio=4.0,
                     attn_hidden_size=self.pixel_attn_hidden_size,
                     attn_num_heads=self.pixel_num_groups,
                     rope_mode=self.rope_mode,
                     rope_ref_grid_h=self.rope_ref_grid_h,
                     rope_ref_grid_w=self.rope_ref_grid_w)
            for _ in range(self.pixel_depth)
        ])
        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)

        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()
    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        head_dim = self.hidden_size // self.num_groups
        if self.rope_mode == "ntk_aware":
            pos = precompute_freqs_cis_2d_ntk(head_dim, height, width, self.rope_ref_grid_h, self.rope_ref_grid_w).to(device)
        else:
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
        self.precompute_pos[(height, width)] = pos
        return pos

    def fetch_pos_text(self, length, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis



# =============================================================================
# LQ Projection 2D (from pid/_src/networks/lq_projection_2d.py — verbatim)
# =============================================================================


class SigmaAwareGatePerTokenPerDim(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.content_proj = nn.Linear(dim * 2, dim)
        nn.init.trunc_normal_(self.content_proj.weight, std=0.01)
        nn.init.constant_(self.content_proj.bias, 2.0)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(5.0)))

    def compute_gate_scalar(self, x, lq, sigma=None):
        assert sigma is not None
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))
        sigma_offset = -self.log_alpha.exp() * sigma.float().view(-1, 1, 1)
        return torch.sigmoid(content_logit + sigma_offset)

    def forward(self, x, lq, sigma=None):
        return x + self.compute_gate_scalar(x, lq, sigma) * lq


_SUPPORTED_GATE_TYPE = "sigma_aware_per_token_per_dim"


def _build_gate(gate_type: str, dim: int, zero_init: bool = True) -> nn.Module:
    if gate_type != _SUPPORTED_GATE_TYPE:
        raise ValueError(f"Unknown gate_type: {gate_type!r}")
    return SigmaAwareGatePerTokenPerDim(dim)


class ResBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 4):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class LQProjection2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 0,
        hidden_dim: int = 512,
        out_dim: int = 1536,
        patch_size: int = 16,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        num_res_blocks: int = 4,
        num_outputs: int = 1,
        gate_type: str = _SUPPORTED_GATE_TYPE,
        interval: int = 1,
        zero_init: bool = True,
    ):
        super().__init__()
        assert in_channels > 0 or latent_channels > 0
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.sr_scale = sr_scale
        self.latent_spatial_down_factor = latent_spatial_down_factor
        self.num_outputs = num_outputs
        self.interval = interval
        self.zero_init = zero_init

        if in_channels > 0:
            assert patch_size >= sr_scale and patch_size % sr_scale == 0
            self.image_unshuffle_factor = patch_size // sr_scale
            unshuffle_ch = in_channels * self.image_unshuffle_factor**2
            layers = [
                nn.Conv2d(unshuffle_ch, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim))
            self.image_conv = nn.Sequential(*layers)
        else:
            self.image_conv = None
            self.image_unshuffle_factor = 0

        if latent_channels > 0:
            z_to_patch_ratio = (sr_scale * latent_spatial_down_factor) / patch_size
            self.z_to_patch_ratio = z_to_patch_ratio
            if z_to_patch_ratio > 1:
                self.latent_upsampler = None
                self.latent_upsample_ratio = int(z_to_patch_ratio)
                latent_proj_in_ch = latent_channels
            elif z_to_patch_ratio == 1:
                self.latent_upsampler = None
                latent_proj_in_ch = latent_channels
            else:
                fold_factor = int(1 / z_to_patch_ratio)
                self.latent_upsampler = None
                self.latent_fold_factor = fold_factor
                latent_proj_in_ch = latent_channels * fold_factor**2
            layers = [
                nn.Conv2d(latent_proj_in_ch, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim))
            self.latent_proj = nn.Sequential(*layers)
        else:
            self.latent_proj = None
            self.z_to_patch_ratio = 0
            self.latent_upsampler = None

        if in_channels > 0 and latent_channels > 0:
            layers = [nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1), nn.SiLU()]
            for _ in range(num_res_blocks):
                layers.append(ResBlock(hidden_dim))
            self.merge = nn.Sequential(*layers)
        else:
            self.merge = None

        self.output_heads = nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(num_outputs)])
        self.gate_modules = nn.ModuleList([_build_gate(gate_type, out_dim, zero_init=zero_init) for _ in range(num_outputs)])

    def is_gate_active(self, block_idx: int) -> bool:
        if self.interval > 1:
            return block_idx % self.interval == 0
        return True

    def _get_output_index(self, block_idx: int) -> int:
        if self.interval > 1:
            return block_idx // self.interval
        return block_idx

    def gate(self, x, lq, sigma=None, out_idx=0):
        return self.gate_modules[out_idx](x, lq, sigma=sigma)

    def _align_image_to_patch_grid(self, lq_video_or_image, target_pH, target_pW):
        f = self.image_unshuffle_factor
        B, C, H_lq, W_lq = lq_video_or_image.shape
        target_H_lq = target_pH * f
        target_W_lq = target_pW * f
        if H_lq != target_H_lq or W_lq != target_W_lq:
            lq_video_or_image = F.interpolate(lq_video_or_image, size=(target_H_lq, target_W_lq), mode="bilinear", align_corners=False)
        x = F.pixel_unshuffle(lq_video_or_image, f)
        return self.image_conv(x)

    def _align_latent_to_patch_grid(self, lq_latent, pH, pW):
        B, z_dim = lq_latent.shape[:2]
        if self.z_to_patch_ratio > 1:
            z_aligned = F.interpolate(lq_latent, size=(pH, pW), mode="nearest")
        elif self.z_to_patch_ratio == 1:
            z_aligned = lq_latent
            if z_aligned.shape[2] != pH or z_aligned.shape[3] != pW:
                z_aligned = F.interpolate(z_aligned, size=(pH, pW), mode="nearest")
        else:
            f = self.latent_fold_factor
            zH_expected, zW_expected = pH * f, pW * f
            if lq_latent.shape[2] != zH_expected or lq_latent.shape[3] != zW_expected:
                lq_latent = F.interpolate(lq_latent, size=(zH_expected, zW_expected), mode="nearest")
            z_aligned = lq_latent.reshape(B, z_dim, pH, f, pW, f)
            z_aligned = z_aligned.permute(0, 1, 3, 5, 2, 4)
            z_aligned = z_aligned.reshape(B, z_dim * f * f, pH, pW)
        return self.latent_proj(z_aligned)

    def forward(self, lq_video_or_image=None, lq_latent=None, target_pH=0, target_pW=0):
        assert target_pH > 0 and target_pW > 0
        features = []
        if self.image_conv is not None and lq_video_or_image is not None:
            features.append(self._align_image_to_patch_grid(lq_video_or_image, target_pH, target_pW))
        if self.latent_proj is not None and lq_latent is not None:
            features.append(self._align_latent_to_patch_grid(lq_latent, target_pH, target_pW))
        if len(features) == 2 and self.merge is not None:
            merged = self.merge(torch.cat(features, dim=1))
        elif len(features) == 1:
            merged = features[0]
        else:
            ref = lq_video_or_image if lq_video_or_image is not None else lq_latent
            B, device, dtype = ref.shape[0], ref.device, ref.dtype
            N = target_pH * target_pW
            return [torch.zeros(B, N, self.out_dim, device=device, dtype=dtype) for _ in range(self.num_outputs)]
        tokens = merged.flatten(2).transpose(1, 2)
        return [head(tokens) for head in self.output_heads]


# =============================================================================
# PidNet (from pid/_src/networks/pid_net.py)
# Changes: log.info → logger.info; pid imports replaced by local definitions
# =============================================================================


class PidNet(PixDiT_T2I):
    def __init__(
        self,
        in_channels=3,
        num_groups=16,
        hidden_size=1152,
        pixel_hidden_size=64,
        pixel_attn_hidden_size=None,
        pixel_num_groups=None,
        patch_depth=26,
        pixel_depth=2,
        num_text_blocks=4,
        patch_size=16,
        txt_embed_dim=4096,
        txt_max_length=1024,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        rope_mode: str = "ntk_aware",
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
        lq_in_channels: int = 3,
        lq_latent_channels: int = 0,
        lq_hidden_dim: int = 512,
        lq_num_res_blocks: int = 4,
        lq_gate_type: str = "sigma_aware_per_token_per_dim",
        lq_interval: int = 1,
        zero_init_lq: bool = True,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
    ):
        super().__init__(
            in_channels=in_channels, num_groups=num_groups, hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size, pixel_attn_hidden_size=pixel_attn_hidden_size,
            pixel_num_groups=pixel_num_groups, patch_depth=patch_depth, pixel_depth=pixel_depth,
            num_text_blocks=num_text_blocks, patch_size=patch_size, txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length, use_text_rope=use_text_rope, text_rope_theta=text_rope_theta,
            rope_mode=rope_mode, rope_ref_h=rope_ref_h, rope_ref_w=rope_ref_w,
        )
        self.sr_scale = sr_scale
        num_lq_outputs = (patch_depth + lq_interval - 1) // lq_interval
        self.lq_proj = LQProjection2D(
            in_channels=lq_in_channels, latent_channels=lq_latent_channels,
            hidden_dim=lq_hidden_dim, out_dim=hidden_size, patch_size=patch_size,
            sr_scale=sr_scale, latent_spatial_down_factor=latent_spatial_down_factor,
            num_res_blocks=lq_num_res_blocks, num_outputs=num_lq_outputs,
            gate_type=lq_gate_type, interval=lq_interval, zero_init=zero_init_lq,
        )

    def _compute_lq_features(self, lq_video_or_image, lq_latent, lq_mask, Hs, Ws):
        lq_features = self.lq_proj(
            lq_video_or_image=lq_video_or_image, lq_latent=lq_latent,
            target_pH=Hs, target_pW=Ws,
        )
        if lq_mask is not None:
            lq_features = [f * lq_mask.view(-1, 1, 1) for f in lq_features]
        return lq_features

    def _run_patch_blocks(self, s_main, y_emb, condition, pos, pos_txt, attn_mask_joint, lq_features, degrade_sigma=None):
        has_lq = lq_features is not None
        for i in range(self.patch_depth):
            if has_lq and self.lq_proj.is_gate_active(i):
                out_idx = self.lq_proj._get_output_index(i)
                if out_idx < len(lq_features):
                    s_main = self.lq_proj.gate(s_main, lq_features[out_idx], sigma=degrade_sigma, out_idx=out_idx)
            s_main, y_emb = self.patch_blocks[i](s_main, y_emb, condition, pos, pos_txt, attn_mask_joint)
        return s_main, y_emb

    def forward(
        self, x, t, y, mask=None,
        lq_video_or_image=None, lq_latent=None, lq_mask=None, degrade_sigma=None,
    ):
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        has_lq = lq_video_or_image is not None or lq_latent is not None
        lq_features = self._compute_lq_features(lq_video_or_image, lq_latent, lq_mask, Hs, Ws) if has_lq else None

        pos = self.fetch_pos(Hs, Ws, x.device)
        x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(y_emb.dtype)
        condition = torch.nn.functional.silu(t_emb)

        pad = None
        pos_txt = self.fetch_pos_text(Ltxt, x.device) if self.use_text_rope else None
        if mask is not None and isinstance(mask, torch.Tensor):
            m = mask
            while m.dim() > 2 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 3 and m.size(1) == 1:
                m = m.squeeze(1)
            if m.dim() == 2:
                pad = m == 0

        s_main = self.s_embedder(x_patches)
        attn_mask_joint = None
        if pad is not None:
            pad_img = torch.zeros((B, L), dtype=torch.bool, device=x.device)
            pad_txt = (
                pad[:, :Ltxt] if pad.size(1) >= Ltxt
                else torch.nn.functional.pad(pad, (0, Ltxt - pad.size(1)), value=True)
            )
            attn_mask_joint = torch.cat([pad_txt, pad_img], dim=1).view(B, 1, 1, Ltxt + L)

        s_main, y_emb = self._run_patch_blocks(
            s_main, y_emb, condition, pos, pos_txt, attn_mask_joint, lq_features,
            degrade_sigma=degrade_sigma,
        )
        s = torch.nn.functional.silu(t_emb + s_main)

        s_cond = s.reshape(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        return torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)


# =============================================================================
# PixelDiTModel (from pid/_src/models/pixeldit_model.py)
# =============================================================================


@attrs.define(slots=False)
class PixelDiTModelConfig:
    net: Any = None
    precision: str = "bfloat16"
    input_caption_key: str = "caption"
    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    model_max_length: int = 300
    chi_prompt: list = attrs.Factory(list)
    fm_timescale: float = 1000.0
    prediction_type: str = "velocity"
    image_size: int = 1024
    negative_prompt: str = "low quality, worst quality, over-saturated, three legs, six fingers, cartoon, anime, cgi, low res, blurry, deformed, distortion, duplicated limbs, plastic skin, jpeg artifacts, watermark"
    dynamic_shift: Optional[dict] = None


_TEXT_ENCODER_DICT = {
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
}


_EMBED_CACHE_DIR = "text-embed_cache"


def _embed_cache_lookup(full_prompts: list) -> tuple | None:
    index_path = os.path.join(_EMBED_CACHE_DIR, "index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        index = json.load(f)
    if not all(p in index for p in full_prompts):
        return None
    embs_list, masks_list = [], []
    for p in full_prompts:
        cached = torch.load(os.path.join(_EMBED_CACHE_DIR, index[p]), map_location="cuda", weights_only=True)
        embs_list.append(cached["embs"])
        masks_list.append(cached["masks"])
    return torch.stack(embs_list), torch.stack(masks_list)


def _embed_cache_save(full_prompts: list, embs: torch.Tensor, masks: torch.Tensor):
    os.makedirs(_EMBED_CACHE_DIR, exist_ok=True)
    index_path = os.path.join(_EMBED_CACHE_DIR, "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
    for i, p in enumerate(full_prompts):
        if p not in index:
            fname = hashlib.md5(p.encode()).hexdigest() + ".pth"
            torch.save({"embs": embs[i].cpu(), "masks": masks[i].cpu()}, os.path.join(_EMBED_CACHE_DIR, fname))
            index[p] = fname
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def _load_text_encoder(name: str, device: str = "cuda"):
    assert name in _TEXT_ENCODER_DICT, f"Unsupported text encoder: {name}"
    model_id = _TEXT_ENCODER_DICT[name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    text_encoder = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).get_decoder().to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return tokenizer, text_encoder


class PixelDiTModel(nn.Module):
    def __init__(self, config: PixelDiTModelConfig):
        super().__init__()
        self.config = config

        if config.dynamic_shift is not None:
            _ds = config.dynamic_shift
            logger.info(f"PixelDiT dynamic shift: base_shift={_ds['base_shift']} base_image_size={_ds['base_image_size_for_shift_calc']}")

        _dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        requested_dtype = _dtype_map[config.precision]
        if requested_dtype != torch.float32:
            self.autocast_dtype = requested_dtype
            self.precision = torch.float32
        else:
            self.autocast_dtype = None
            self.precision = torch.float32
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        self.net = config.net
        self.net = self.net.to(device="cuda", dtype=torch.float32)
        logger.info(f"PixDiT_T2I params: {sum(p.numel() for p in self.net.parameters()):,}")

        object.__setattr__(self, "tokenizer", None)
        object.__setattr__(self, "text_encoder", None)
        self._chi_prompt_str = "\n".join(config.chi_prompt) if config.chi_prompt else ""
        self._num_chi_tokens = 0
        null_caption = config.negative_prompt if config.negative_prompt else ""
        null_full = (self._chi_prompt_str + null_caption) if self._chi_prompt_str else null_caption
        cached = _embed_cache_lookup([null_full])
        if cached is not None:
            logger.info("Null caption embedding: cache hit")
            self._null_caption_embs = cached[0]
        else:
            logger.info("Null caption embedding: cache miss — loading text encoder and encoding")
            _tok, _enc = _load_text_encoder(config.text_encoder_name, device="cuda")
            object.__setattr__(self, "tokenizer", _tok)
            object.__setattr__(self, "text_encoder", _enc)
            self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str)) if self._chi_prompt_str else 0
            null_embs, null_masks = self._encode_text_raw([null_caption])
            self._null_caption_embs = null_embs
            _embed_cache_save([null_full], null_embs, null_masks)
            logger.info("Null caption embedding saved to cache")


    @torch.no_grad()
    def _encode_text_raw(self, captions):
        if self._chi_prompt_str:
            prompts_all = [self._chi_prompt_str + cap for cap in captions]
            max_length_all = self._num_chi_tokens + self.config.model_max_length - 2
        else:
            prompts_all = captions
            max_length_all = self.config.model_max_length
        caption_token = self.tokenizer(
            prompts_all, max_length=max_length_all, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to("cuda")
        caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0]
        select_index = [0] + list(range(-self.config.model_max_length + 1, 0))
        caption_embs = caption_embs[:, select_index]
        emb_masks = caption_token.attention_mask[:, select_index]
        return caption_embs, emb_masks


# =============================================================================
# PidModel (from pid/_src/models/pid_model.py + pid_distill_model.py)
# =============================================================================


@attrs.define(slots=False)
class PidModelConfig(PixelDiTModelConfig):
    lq_condition_type: str = "latent"
    tokenizer: Any = None
    state_ch: int = 16
    sample_steps: int = 1
    t_schedule: Optional[list] = None


class PidModel(PixelDiTModel):
    def __init__(self, config: PidModelConfig):
        super().__init__(config)
        if config.tokenizer is not None:
            self.vae_encoder: Any = config.tokenizer
            if config.state_ch > 0:
                assert self.vae_encoder.latent_ch == config.state_ch, (
                    f"latent_ch {self.vae_encoder.latent_ch} != state_ch {config.state_ch}"
                )
        else:
            self.vae_encoder = None
            logger.warning("No VAE configured — LQ latent encoding disabled.")

    @torch.no_grad()
    def encode_lq_latent(self, lq_image: Tensor) -> Tensor:
        if lq_image.ndim == 4:
            lq_image = lq_image.unsqueeze(2)
        latent = self.vae_encoder.encode(lq_image)
        if latent.ndim == 5:
            latent = latent[:, :, 0, :, :]
        return latent

    def _net_output_to_x0(self, x_t, net_output, t, prediction_type):
        if prediction_type == "x0":
            return net_output.to(x_t.dtype)
        if prediction_type == "velocity":
            original_dtype = x_t.dtype
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s)
            return (x_t.double() - t_shaped * net_output.double()).to(original_dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _velocity_to_x0(self, x_t, net_output, t):
        return self._net_output_to_x0(x_t, net_output, t, self.config.prediction_type)

    def _get_t_list(self, device, num_steps=None):
        target_steps = num_steps if num_steps is not None else self.config.sample_steps
        if self.config.t_schedule is not None:
            full_t = torch.tensor(self.config.t_schedule, device=device, dtype=torch.float32)
            if target_steps != self.config.sample_steps:
                indices = torch.linspace(0, len(full_t) - 1, target_steps + 1).round().long()
                t_list = full_t[indices]
            else:
                t_list = full_t
        else:
            t_list = torch.linspace(1.0, 0.0, target_steps + 1, device=device, dtype=torch.float32)
        assert abs(t_list[-1].item()) < 1e-6
        if num_steps is not None:
            logger.info(f"num_steps={num_steps}, t_list={t_list.tolist()}")
        return t_list

    def _sample_loop(self, noise, t_list, caption_embs, lq_video_or_image, lq_latent, degrade_sigma_tensor, generator=None):
        B = noise.shape[0]
        timescale = self.config.fm_timescale
        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        x = noise
        net = self.net
        with autocast_ctx:
            for t_cur, t_next in zip(t_list[:-1], t_list[1:]):
                t_cur_batch = t_cur.expand(B)
                t_cur_scaled = t_cur_batch * timescale
                v_pred = net(x, t_cur_scaled, caption_embs, lq_video_or_image=lq_video_or_image, lq_latent=lq_latent, degrade_sigma=degrade_sigma_tensor)
                if t_next.item() > 0:
                    x0_pred = self._velocity_to_x0(x, v_pred, t_cur_batch)
                    eps_infer = torch.randn(x0_pred.shape, device=x0_pred.device, dtype=x0_pred.dtype, generator=generator)
                    s = [B] + [1] * (x.ndim - 1)
                    t_next_bcast = t_next.reshape(1).expand(s)
                    x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
                else:
                    x = self._velocity_to_x0(x, v_pred, t_cur_batch)
        return x

    @torch.no_grad()
    def generate_samples_from_batch(self, data_batch, cfg_scale=None, num_steps=None, seed=0, image_size=None, shift=None, **kwargs):
        if "LQ_latent" not in data_batch and "LQ_video_or_image" in data_batch and self.vae_encoder is not None:
            data_batch["LQ_latent"] = self.encode_lq_latent(data_batch["LQ_video_or_image"]).contiguous().to(**self.tensor_kwargs)
        if "degrade_sigma" not in data_batch and "LQ_latent" in data_batch:
            B = data_batch["LQ_latent"].shape[0]
            data_batch["degrade_sigma"] = torch.zeros(B, device=data_batch["LQ_latent"].device, dtype=torch.float32)

        image_size = image_size or self.config.image_size
        if isinstance(image_size, (list, tuple)):
            img_h, img_w = int(image_size[0]), int(image_size[1])
        else:
            img_h = img_w = int(image_size)

        if shift is None and self.config.dynamic_shift is not None:
            _ds = self.config.dynamic_shift
            shift = _ds["base_shift"] * math.sqrt(max(img_h, img_w) / _ds["base_image_size_for_shift_calc"])

        captions = data_batch[self.config.input_caption_key]
        if isinstance(captions, str):
            captions = [captions]
        B = len(captions)
        full_prompts = [(self._chi_prompt_str + c) if self._chi_prompt_str else c for c in captions]
        cached = _embed_cache_lookup(full_prompts)
        if cached is not None:
            logger.info("Caption embedding: cache hit")
            caption_embs = cached[0].to(**self.tensor_kwargs)
        else:
            if self.text_encoder is None:
                logger.info("Caption embedding: cache miss — loading text encoder and encoding")
                _tok, _enc = _load_text_encoder(self.config.text_encoder_name, device="cuda")
                object.__setattr__(self, "tokenizer", _tok)
                object.__setattr__(self, "text_encoder", _enc)
                self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str)) if self._chi_prompt_str else 0
            else:
                logger.info("Caption embedding: cache miss — encoding")
            caption_embs, caption_masks = self._encode_text_raw(captions)
            _embed_cache_save(full_prompts, caption_embs, caption_masks)
            logger.info("Caption embedding saved to cache")
            caption_embs = caption_embs.to(**self.tensor_kwargs)

        lq_video_or_image = None
        lq_latent = None
        if self.config.lq_condition_type in ("image", "image_latent"):
            lq_video_or_image = data_batch.get("LQ_video_or_image")
            if lq_video_or_image is not None:
                lq_video_or_image = lq_video_or_image.to(**self.tensor_kwargs)
        if self.config.lq_condition_type in ("latent", "image_latent"):
            lq_latent = data_batch.get("LQ_latent")
            if lq_latent is not None:
                lq_latent = lq_latent.to(**self.tensor_kwargs)

        sigma_val = data_batch.get("degrade_sigma", 0.0)
        if isinstance(sigma_val, torch.Tensor):
            degrade_sigma_tensor = sigma_val.to(device="cuda", dtype=torch.float32).reshape(-1)
            if degrade_sigma_tensor.numel() == 1:
                degrade_sigma_tensor = degrade_sigma_tensor.expand(B).contiguous()
            assert degrade_sigma_tensor.shape == (B,)
        elif isinstance(sigma_val, (list, tuple)):
            degrade_sigma_tensor = torch.tensor(sigma_val, device="cuda", dtype=torch.float32)
            assert degrade_sigma_tensor.shape == (B,)
        else:
            degrade_sigma_tensor = torch.full((B,), float(sigma_val), device="cuda", dtype=torch.float32)

        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        noise = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

        autocast_ctx = torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        net = self.net
        net.eval()
        effective_steps = num_steps if num_steps is not None else self.config.sample_steps

        if effective_steps == 1:
            t_start = torch.full((B,), 1.0, device="cuda", dtype=torch.float32)
            t_start_scaled = t_start * self.config.fm_timescale
            with autocast_ctx:
                v = net(noise, t_start_scaled, caption_embs, lq_video_or_image=lq_video_or_image, lq_latent=lq_latent, degrade_sigma=degrade_sigma_tensor)
                x0 = self._velocity_to_x0(noise, v, t_start)
        else:
            t_list = self._get_t_list(device=torch.device("cuda"), num_steps=num_steps)
            x0 = self._sample_loop(noise, t_list, caption_embs, lq_video_or_image, lq_latent, degrade_sigma_tensor, generator=gen)

        return x0.clamp(-1, 1).unsqueeze(2)

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs):
        _net_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net.") and not k.startswith("net_ema."):
                _net_sd[k[len("net."):]] = v
            elif not k.startswith("net_ema."):
                _net_sd[k] = v
        missing, unexpected = self.net.load_state_dict(_net_sd, strict=False, assign=assign)
        if missing:
            lq_missing = [k for k in missing if "lq_proj" in k]
            other_missing = [k for k in missing if "lq_proj" not in k]
            if lq_missing:
                logger.info(f"Expected missing LQ keys ({len(lq_missing)} keys)")
            if other_missing and strict:
                logger.warning(f"Missing keys in net: {other_missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in net: {unexpected}")
