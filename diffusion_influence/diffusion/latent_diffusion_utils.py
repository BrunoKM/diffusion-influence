from typing import Union

import torch
from diffusers import AutoencoderKL
from torch import Tensor


def vae_encode_to_latent(image: Tensor, vae: AutoencoderKL):
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        return latents


def vae_decode_from_latent(latents: Tensor, vae: AutoencoderKL):
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        return vae.decode(latents).sample


def vae_encode_batch(
    batch,
    vae: AutoencoderKL,
    device: torch.device,
    batch_image_idx: Union[str, int] = "input",
):
    # Assumes batch to be a dict (Huggingface datasets) or indexable object
    return {
        key: vae_encode_to_latent(batch[batch_image_idx].to(device), vae=vae)
        if key == batch_image_idx
        else value
        for key, value in batch.items()
    }
