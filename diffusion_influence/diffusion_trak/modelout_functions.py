from functools import partial
from typing import Callable, Iterable, Protocol

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.models.unets.unet_2d import UNet2DOutput
from torch import Tensor
from trak.modelout_functions import AbstractModelOutput

from diffusion_influence.diffusion.elbo import stochastic_elbo_sample
from diffusion_influence.diffusion.latent_diffusion_utils import vae_encode_to_latent
from diffusion_influence.diffusion.train_util import ddpm_sample_inputs_and_targets


class ModelOutputFunction(Protocol):
    def __call__(
        self,
        model: Callable,
        weights: Iterable[Tensor],
        buffers: Iterable[Tensor],
        image: Tensor,
        noise_scheduler: DDPMScheduler,
    ) -> Tensor: ...


def diffusion_loss_output_function(
    model: Callable,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    image: Tensor,
    noise_scheduler: DDPMScheduler,
) -> Tensor:
    clean_image = image.unsqueeze(0)

    x0, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
        x0=clean_image,
        max_time=noise_scheduler.config.num_train_timesteps,
        noise_scheduler=noise_scheduler,
        device=clean_image.device,
    )

    kwargs = {"return_dict": False}

    noise_pred = torch.func.functional_call(
        model, (weights, buffers), args=(xt, timesteps), kwargs=kwargs
    )[0]
    return F.mse_loss(noise_pred, noise)


def simplified_elbo_output_function(
    model: Callable,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    image: Tensor,
    noise_scheduler: DDPMScheduler,
) -> Tensor:
    clean_image = image.unsqueeze(0)

    x0, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
        x0=clean_image,
        max_time=noise_scheduler.config.num_train_timesteps,
        noise_scheduler=noise_scheduler,
        device=clean_image.device,
    )

    kwargs = {"return_dict": False}

    noise_pred = torch.func.functional_call(
        model, (weights, buffers), args=(xt, timesteps), kwargs=kwargs
    )[0]
    losses = F.mse_loss(noise_pred, noise, reduction="none")
    loss_weight_terms = (
        noise_scheduler.betas.to(losses.device)
        / (
            2
            * noise_scheduler.alphas.to(losses.device)
            * (1 - noise_scheduler.alphas_cumprod.to(losses.device))
        )
    ).to(losses.device)[timesteps]
    losses = losses.mean(dim=tuple(range(loss_weight_terms.ndim, losses.ndim)))
    return (loss_weight_terms * losses).mean()


def elbo_output_function(
    model: Callable,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    image: Tensor,
    noise_scheduler: DDPMScheduler,
) -> Tensor:
    clean_image = image.unsqueeze(0)

    kwargs = {"return_dict": False}

    def model_call_func(sample, timestep):
        output = torch.func.functional_call(
            model, (weights, buffers), args=(sample, timestep), kwargs=kwargs
        )[0]
        return UNet2DOutput(sample=output)

    return stochastic_elbo_sample(
        denoise_model=model_call_func,
        noise_scheduler=noise_scheduler,
        original_samples=clean_image,
        data_range=(-1.0, 1.0),
    ).squeeze(dim=0)


def square_norm_output_function(
    model: Callable,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    image: Tensor,
    noise_scheduler: DDPMScheduler,
) -> Tensor:
    clean_image = image.unsqueeze(0)

    x0, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
        x0=clean_image,
        max_time=noise_scheduler.config.num_train_timesteps,
        noise_scheduler=noise_scheduler,
        device=clean_image.device,
    )

    kwargs = {"return_dict": False}

    noise_pred: Tensor = torch.func.functional_call(
        model, (weights, buffers), args=(xt, timesteps), kwargs=kwargs
    )[0]

    # Mean squared L2 norm (normalised by dimensionality of the image)
    # of the predicted noise (as a vector, no matrix norm)

    # Note that this is different from what's describe in the D-TRAK paper:
    # https://github.com/sail-sg/D-TRAK
    # to be a proper norm, we'd have to use `sum()` instead of `mean()`.
    # However, this is what they do in the code, so here we go
    return noise_pred.square().mean(dim=(-1, -2, -3)).mean()


def latent_diffusion_loss_output_function(
    model: Callable,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    image: Tensor,
    noise_scheduler: DDPMScheduler,
    vae: AutoencoderKL,
) -> Tensor:
    clean_image = image.unsqueeze(0)
    z0 = vae_encode_to_latent(clean_image, vae)
    return diffusion_loss_output_function(
        model, weights, buffers, z0.squeeze(0), noise_scheduler
    )


def latent_square_norm_output_function(
    model: Callable,
    weights: Iterable[Tensor],
    buffers: Iterable[Tensor],
    image: Tensor,
    noise_scheduler: DDPMScheduler,
    vae: AutoencoderKL,
) -> Tensor:
    clean_image = image.unsqueeze(0)
    z0 = vae_encode_to_latent(clean_image, vae)
    return square_norm_output_function(
        model, weights, buffers, z0.squeeze(0), noise_scheduler
    )


class UnconditionalDiffusionModelOutput(AbstractModelOutput):
    """Model output function for diffusion models."""

    def __init__(
        self,
        measure_func: ModelOutputFunction,
        featurize_output_func: ModelOutputFunction,
        noise_scheduler: DDPMScheduler,
    ):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.max_timestep = self.noise_scheduler.config.num_train_timesteps

        self._are_we_featurizing: bool = False

        self.get_output_featurizing = partial(
            featurize_output_func, noise_scheduler=noise_scheduler
        )
        self.get_output_scoring = partial(measure_func, noise_scheduler=noise_scheduler)

    def get_output(self, *args, **kwargs) -> Tensor:
        if self._are_we_featurizing:
            return self.get_output_featurizing(*args, **kwargs)
        else:
            return self.get_output_scoring(*args, **kwargs)

    def get_out_to_loss_grad(self, model, weights, buffers, batch: Tensor):
        # latents, _, __ = batch
        latents = batch["input"]
        return torch.ones(latents.shape[0]).to(latents.device).unsqueeze(-1)
