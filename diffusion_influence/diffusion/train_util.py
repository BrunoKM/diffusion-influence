from typing import Optional

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from torch import Tensor, nn


def ddpm_sample_inputs_and_targets(
    x0: Tensor,  # [batch_size, *state_shape]
    max_time: int,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # Sample time
    batch_size = x0.shape[0]
    timesteps = torch.randint(
        0, max_time, size=(batch_size,), device=device, dtype=torch.long
    )
    noise = torch.randn_like(x0)
    xt = noise_scheduler.add_noise(x0, noise, timesteps)
    return x0, xt, timesteps, noise


def ddpm_train_step(
    denoise_model: nn.Module,
    batch: Tensor,
    optimizer: torch.optim.Optimizer,
    noise_scheduler: DDPMScheduler,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    max_time: int,
    device: torch.device,
    clip_grad: float,
    accelerator: Accelerator,
) -> Tensor:
    # Get the noised up samples
    _, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
        x0=batch,
        max_time=max_time,
        noise_scheduler=noise_scheduler,
        device=device,
    )

    # Compute the loss
    with accelerator.accumulate(denoise_model):
        # Predict the noise residual
        model_output = denoise_model(xt, timesteps).sample

        loss = torch.nn.functional.mse_loss(model_output, noise)

        accelerator.backward(loss)

        if clip_grad < float("inf"):
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoise_model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return loss


def weighted_ddpm_train_step(
    denoise_model: nn.Module,
    batch: Tensor,
    example_weights: Tensor,
    optimizer: torch.optim.Optimizer,
    noise_scheduler: DDPMScheduler,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    max_time: int,
    device: torch.device,
    clip_grad: float,
    accelerator: Accelerator,
) -> Tensor:
    # Get the noised up samples
    _, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
        x0=batch,
        max_time=max_time,
        noise_scheduler=noise_scheduler,
        device=device,
    )

    # Compute the loss
    with accelerator.accumulate(denoise_model):
        # Predict the noise residual
        model_output = denoise_model(xt, timesteps).sample

        losses = torch.nn.functional.mse_loss(model_output, noise, reduction="none")
        losses = losses.mean(dim=tuple(range(1, losses.ndim)))
        loss = (example_weights * losses).mean()

        accelerator.backward(loss)

        if clip_grad < float("inf"):
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(denoise_model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return loss
