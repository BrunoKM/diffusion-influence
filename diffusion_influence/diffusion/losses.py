import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn


def ddpm_loss_for_timestep(
    denoise_model: nn.Module,
    timestep: int,
    batch: Tensor,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
) -> Tensor:
    """
    Compute the DDPM loss for a given diffusion timestep. Note, the time indexing
    is 0-indexed, so that `timestep=0` corresponds to the first timestep in the
    DDPM paper.
    """
    # --- Get the noised up samples
    batch_size = batch.shape[0]
    timesteps = torch.ones(size=(batch_size,), device=device, dtype=torch.long) * (
        timestep - 1
    )
    noise = torch.randn_like(batch)
    xt = noise_scheduler.add_noise(batch, noise, timesteps)
    # --- Compute the loss
    model_output = denoise_model(xt, timesteps).sample

    loss = torch.nn.functional.mse_loss(model_output, noise)
    return loss
