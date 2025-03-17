import abc
from typing import Callable, Literal, Optional, Sequence, Union

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from einops import rearrange
from torch import Tensor, nn

from diffusion_influence.diffusion.elbo import stochastic_elbo_sample
from diffusion_influence.diffusion.latent_diffusion_utils import vae_encode_to_latent
from diffusion_influence.diffusion.train_util import ddpm_sample_inputs_and_targets
from diffusion_influence.measurements import (
    probability_of_sampling_trajectory_stochastic_sample,
)


class DiffusionTask(metaclass=abc.ABCMeta):
    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        num_augment_samples_loss: int,
        num_augment_samples_measurement: int,
        train_augmentations: Callable[[Tensor], Tensor],
        batch_image_idx: Union[str, int] = "input",
        device: Optional[torch.device] = None,
    ) -> None:
        self.noise_scheduler = noise_scheduler
        self.num_augment_samples_loss = num_augment_samples_loss
        self.num_augment_samples_measurement = num_augment_samples_measurement
        self.batch_image_idx = batch_image_idx
        self.train_augmentations = train_augmentations
        self.device = device

    def compute_train_loss(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
        reduction: Literal["mean", "sum"] = "mean",
        num_augment_samples: Optional[int] = None,
        apply_train_augmentations: bool = True,
    ) -> Tensor:
        """Computes training loss for a given batch and model.

        Args:
            batch (tuple[tuple[Tensor, Tensor], Tensor]):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for loss computation.

        Returns:
            Tensor:
                The computed loss as a tensor.
        """
        x0 = batch[self.batch_image_idx]
        # Duplicate the input tensor to match the number of augment samples
        num_augment_samples = num_augment_samples or self.num_augment_samples_loss

        assert x0.shape[0] == 1, "Batch size must be 1."
        x0_repeated = (
            torch.stack(
                [
                    # Assumes train augmentations can only be (correctly) applied per-example
                    # (e.g. RandomHorizontalFlip)
                    self.train_augmentations(
                        x0.squeeze(0)
                    )  #  Squeeze the batch dimension
                    for _ in range(num_augment_samples)
                ],
                dim=0,
            )
            if apply_train_augmentations
            else x0.repeat(num_augment_samples, 1, 1, 1)
        )
        _, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
            x0=x0_repeated,
            max_time=self.noise_scheduler.config.num_train_timesteps,  #  type: ignore
            noise_scheduler=self.noise_scheduler,
            device=x0.device,
        )
        device: torch.device = self.device or next(iter(model.parameters())).device
        xt, timesteps, noise = xt.to(device), timesteps.to(device), noise.to(device)
        # Predict the noise residual
        model_output: Tensor = model(xt, timesteps).sample
        # Compute the loss
        return nn.functional.mse_loss(model_output, noise, reduction=reduction)

    @abc.abstractmethod
    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        """Computes a measurable quantity (e.g., loss, logit, log probability) for a
        given batch and model. This is defined as f(θ) from
        https://arxiv.org/pdf/2308.03296.pdf.

        Args:
            batch (Any):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for measurement computation.

        Returns:
            torch.Tensor:
                The measurable quantity as a tensor.
        """
        raise NotImplementedError("Measurement computation is not implemented.")


class DiffusionLossTask(DiffusionTask):
    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        return self.compute_train_loss(
            batch,
            model,
            reduction="mean",
            num_augment_samples=self.num_augment_samples_measurement,
            apply_train_augmentations=False,  # We don't want to augment the measurement
        )


class DiffusionELBOTask(DiffusionTask):
    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        """Computes ELBO (with the reconstruction log-likelihood) for a given batch and model.

        Args:
            batch (tuple[tuple[Tensor, Tensor], Tensor]):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for loss computation.

        Returns:
            Tensor:
                The computed loss as a tensor.
        """
        device: torch.device = self.device or next(iter(model.parameters())).device
        x0 = batch[self.batch_image_idx].to(device)
        # Duplicate the input tensor to match the number of augment samples
        num_augment_samples = self.num_augment_samples_measurement

        assert x0.shape[0] == 1, "Batch size must be 1."
        x0_repeated = x0.repeat(num_augment_samples, 1, 1, 1)

        # Compute the ELBO estimate stochastically:
        elbo_samples = stochastic_elbo_sample(
            denoise_model=model,
            noise_scheduler=self.noise_scheduler,
            original_samples=x0_repeated,
            data_range=(-1.0, 1.0),
        )
        return elbo_samples.mean()


class DiffusionSquareNormLossTask(DiffusionTask):
    """The square norm loss measurement that allegedly works well in D-TRAK."""

    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        device: torch.device = self.device or next(iter(model.parameters())).device
        x0 = batch[self.batch_image_idx].to(device)

        # Duplicate the input tensor to match the number of augment samples
        num_augment_samples = self.num_augment_samples_measurement

        assert x0.shape[0] == 1, "Batch size must be 1."
        x0_repeated = x0.repeat(num_augment_samples, 1, 1, 1)

        _, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
            x0=x0_repeated,
            max_time=self.noise_scheduler.config.num_train_timesteps,  #  type: ignore
            noise_scheduler=self.noise_scheduler,
            device=x0.device,
        )
        device: torch.device = self.device or next(iter(model.parameters())).device
        xt, timesteps, noise = xt.to(device), timesteps.to(device), noise.to(device)
        # Predict the noise residual
        model_output: Tensor = model(xt, timesteps).sample

        return model_output.square().mean()


class DiffusionWeightedLossTask(DiffusionTask):
    def __init__(
        self,
        weighting: Sequence[float],
        noise_scheduler: DDPMScheduler,
        num_augment_samples_loss: int,
        num_augment_samples_measurement: int,
        train_augmentations: Callable[[Tensor], Tensor],
        batch_image_idx: str | int = "input",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            noise_scheduler,
            num_augment_samples_loss,
            num_augment_samples_measurement,
            train_augmentations,
            batch_image_idx,
            device,
        )
        self.loss_weight_terms = torch.tensor(weighting, dtype=torch.float32).to(
            self.device
        )

    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        """
        Computes the reweighted loss objective for a given batch and model.
        Still uses uniform sampling of the terms, but weights them according to the
        `loss_weight_terms` parameter.

        Args:
            batch (tuple[tuple[Tensor, Tensor], Tensor]):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for loss computation.

        Returns:
            Tensor:
                The computed loss as a tensor.
        """
        device: torch.device = self.device or next(iter(model.parameters())).device
        x0 = batch[self.batch_image_idx].to(device)
        state_ndim = x0.ndim - 1
        # Duplicate the input tensor to match the number of augment samples
        num_augment_samples = self.num_augment_samples_measurement

        assert x0.shape[0] == 1, "Batch size must be 1."
        x0_repeated = x0.repeat(num_augment_samples, 1, 1, 1)

        _, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
            x0=x0_repeated,
            max_time=self.noise_scheduler.config.num_train_timesteps,  #  type: ignore
            noise_scheduler=self.noise_scheduler,
            device=x0.device,
        )
        device: torch.device = self.device or next(iter(model.parameters())).device
        xt, timesteps, noise = xt.to(device), timesteps.to(device), noise.to(device)
        # Predict the noise residual
        model_output: Tensor = model(xt, timesteps).sample

        loss_weight_terms_for_timesteps = self.loss_weight_terms[timesteps]

        # Compute the loss
        unweighted_loss = nn.functional.mse_loss(model_output, noise, reduction="none")
        unweighted_loss = unweighted_loss.mean(dim=tuple(range(1, state_ndim + 1)))
        loss = loss_weight_terms_for_timesteps * unweighted_loss
        return loss.mean()


class DiffusionSimplifiedELBOTask(DiffusionWeightedLossTask):
    """
    Computes ELBO “loss” --- as the reweighted loss objective without the
    log reconstruction (likelihood) term --- for a given batch and model.
    """

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        num_augment_samples_loss: int,
        num_augment_samples_measurement: int,
        train_augmentations: Callable[[Tensor], Tensor],
        batch_image_idx: str | int = "input",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            weighting=(
                noise_scheduler.betas
                / (2 * noise_scheduler.alphas * (1 - noise_scheduler.alphas_cumprod))
            ).numpy(),  # type: ignore
            noise_scheduler=noise_scheduler,
            num_augment_samples_loss=num_augment_samples_loss,
            num_augment_samples_measurement=num_augment_samples_measurement,
            train_augmentations=train_augmentations,
            batch_image_idx=batch_image_idx,
            device=device,
        )


class DiffusionLossAtTimeTask(DiffusionTask):
    def __init__(
        self,
        time: int,
        noise_scheduler: DDPMScheduler,
        num_augment_samples_loss: int,
        num_augment_samples_measurement: int,
        train_augmentations: Callable[[Tensor], Tensor],
        batch_image_idx: str | int = "input",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            noise_scheduler,
            num_augment_samples_loss,
            num_augment_samples_measurement,
            train_augmentations,
            batch_image_idx,
            device,
        )
        assert 0 < time < noise_scheduler.config.num_train_timesteps  #  type: ignore
        self.time = time

    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        """Computes ELBO “loss” --- as the reweighted loss objective without the
        log reconstruction (likelihood) term --- for a given batch and model.

        Args:
            batch (tuple[tuple[Tensor, Tensor], Tensor]):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for loss computation.

        Returns:
            Tensor:
                The computed loss as a tensor.
        """
        device: torch.device = self.device or next(iter(model.parameters())).device
        x0 = batch[self.batch_image_idx].to(device)

        # Duplicate the input tensor to match the number of augment samples
        num_augment_samples = self.num_augment_samples_measurement

        assert x0.shape[0] == 1, "Batch size must be 1."
        x0_repeated = x0.repeat(num_augment_samples, 1, 1, 1)

        noise = torch.randn_like(x0_repeated)
        xt = self.noise_scheduler.add_noise(
            x0_repeated, noise, torch.tensor(self.time, device=device)
        )

        device: torch.device = self.device or next(iter(model.parameters())).device
        xt, noise = xt.to(device), noise.to(device)

        # Predict the noise residual
        model_output: Tensor = model(xt, self.time).sample

        # Compute the loss
        loss = nn.functional.mse_loss(model_output, noise, reduction="none")
        return loss.mean()


class DiffusionSamplingTrajectoryLogprobTask(DiffusionTask):
    def compute_measurement(
        self,
        batch: dict[str, Tensor],
        model: nn.Module,
    ) -> Tensor:
        device: torch.device = self.device or next(iter(model.parameters())).device
        trajectory = batch["trajectory"].to(
            device
        )  # [1, num_timesteps + 1, *state_shape]
        log_prob = probability_of_sampling_trajectory_stochastic_sample(
            denoise_model=model,
            noise_scheduler=self.noise_scheduler,
            trajectory=trajectory.squeeze(0),  # [num_timesteps + 1, *state_shape]
            batch_size=self.num_augment_samples_measurement,
        )  # []

        return log_prob


class LatentDiffusionTask(DiffusionTask, metaclass=abc.ABCMeta):
    """
    Same as Diffusion Task, but uses a VAE for encoding to the latent space.

    """

    def __init__(
        self,
        vae: AutoencoderKL,
        noise_scheduler: DDPMScheduler,
        num_augment_samples_loss: int,
        num_augment_samples_measurement: int,
        train_augmentations: Callable[[Tensor], Tensor],
        batch_image_idx: Union[str, int] = "input",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            noise_scheduler=noise_scheduler,
            num_augment_samples_loss=num_augment_samples_loss,
            num_augment_samples_measurement=num_augment_samples_measurement,
            train_augmentations=train_augmentations,
            batch_image_idx=batch_image_idx,
            device=device,
        )
        self.vae = vae

    def compute_train_loss(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
        reduction: Literal["mean", "sum"] = "mean",
        num_augment_samples: Optional[int] = None,
        apply_train_augmentations: bool = True,
    ) -> Tensor:
        """Computes training loss for a given batch and model.

        Args:
            batch (tuple[tuple[Tensor, Tensor], Tensor]):
                Batch of data sourced from the DataLoader.
            model (nn.Module):
                The PyTorch model for loss computation.
            sample (bool):
                Indicates whether to sample from the model's outputs or to use the
                actual targets from the batch. Defaults to False. The case where
                `sample` is set to True must be implemented to approximate the true
                Fisher.

        Returns:
            Tensor:
                The computed loss as a tensor.
        """
        device: torch.device = self.device or next(iter(model.parameters())).device

        x0 = batch[self.batch_image_idx].to(device)
        # Duplicate the input tensor to match the number of augment samples
        num_augment_samples = num_augment_samples or self.num_augment_samples_loss

        assert x0.shape[0] == 1, "Batch size must be 1."
        x0_repeated = (
            torch.stack(
                [
                    # Assumes train augmentations can only be (correctly) applied per-example
                    # (e.g. RandomHorizontalFlip)
                    self.train_augmentations(
                        x0.squeeze(0)
                    )  #  Squeeze the batch dimension
                    for _ in range(num_augment_samples)
                ],
                dim=0,
            )
            if apply_train_augmentations
            else x0.repeat(num_augment_samples, 1, 1, 1)
        )
        # Encode to the latent space:
        # Note: might be a bit wasteful if we we don't use different train. augmentations
        # for every example, but oh well..
        z0 = vae_encode_to_latent(x0_repeated, self.vae)

        _, xt, timesteps, noise = ddpm_sample_inputs_and_targets(
            x0=z0,
            max_time=self.noise_scheduler.config.num_train_timesteps,  #  type: ignore
            noise_scheduler=self.noise_scheduler,
            device=z0.device,
        )
        xt, timesteps, noise = xt.to(device), timesteps.to(device), noise.to(device)
        # Predict the noise residual
        model_output: Tensor = model(xt, timesteps).sample
        # Compute the loss
        return nn.functional.mse_loss(model_output, noise, reduction=reduction)


class LatentDiffusionLossTask(LatentDiffusionTask):
    def compute_measurement(
        self,
        batch: Union[tuple[Tensor, ...], dict[str, Tensor]],
        model: nn.Module,
    ) -> Tensor:
        return self.compute_train_loss(
            batch,
            model,
            reduction="mean",
            num_augment_samples=self.num_augment_samples_measurement,
            apply_train_augmentations=False,  # We don't want to augment the measurement
        )


def data_batch_to_diffusion_regression_batch(
    batch,
    noise_scheduler: DDPMScheduler,
    batch_image_idx: Union[str, int] = "input",
) -> tuple[tuple[Tensor, Tensor], Tensor]:
    """
    Only used for the data loader passed to `KFACLinearOperator`.

    This creates the “targets” for the regression task in the `KFACLinearOperator`
    to compute a GGN approximation. It also reshapes the targets to match the expected
    format of flat targets.

    Args:
        base_dataset (Dataset):
            The base image dataset to train on.
        noise_scheduler (DDPMScheduler):
            The noise scheduler for the diffusion process.
        batch_image_idx (Union[str, int]):
            The index to access the image data in a batch. Defaults to "input".
    """
    # Assumes batch to be a dict (Huggingface datasets) or indexable object
    x0: Tensor = batch[batch_image_idx]
    batch_shape = x0.shape[:-3]
    device = x0.device
    # Get the noised up samples
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,  #  type: ignore
        size=batch_shape,
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn_like(x0)
    xt = noise_scheduler.add_noise(x0, noise, timesteps)  # type: ignore
    # Return noisy samples and corresponding timesteps as inputs, noise as target
    inputs = (xt, timesteps)
    targets = noise
    # Rearrange the target to have the channel dimension last, which is expected by
    # `KFACLinearOperator` for the `MSELoss`
    return inputs, rearrange(targets, "... c h w -> ... (h w c)")


class KFACCompatibleImageRegressionModelWrapper(nn.Module):
    """
    Wrapper to make the UNet (or any image to image) model compatible with the `KFACLinearOperator`.
    KFACLinearOperator expects a 2D input (batch-size, target size) for regression, or
    for the output in regression to have the channel dimension last. In this case,
    the right thing to do to approximate the GGN as defined in:
        https://arxiv.org/abs/2410.13850
    requires treating the height, width and channel size dimensions as the
    output dimension, as we do by flattening them below.
    """

    def __init__(self, model, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, inps):
        outputs = self.model(inps[0].to(self.device), inps[1].to(self.device)).sample
        # Rearrange the output output to have the channel dimension last, which is
        # expected by `KFACLinearOperator` for the `MSELoss`
        return rearrange(outputs, "b c h w -> b (h w c)")
