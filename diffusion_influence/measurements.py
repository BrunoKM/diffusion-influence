import math

import torch
import tqdm
from diffusers import DDPMScheduler, UNet2DModel
from torch import FloatTensor, Tensor
from torch.distributions import Normal

from diffusion_influence.diffusion.elbo import predict_prev_sample


def probability_of_sampling_trajectory(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    trajectory: FloatTensor,  # [batch_size, num_inference_steps + 1, *state_shape]
) -> Tensor:  # [batch_size,]
    """
    The (dimension normalized) probability of sampling the given trajectory from the diffusion model.

    Note: This measurement has the issue of not being reparameterisation invariant.
    We could rescale the latent space by an arbitrary factor at different timesteps,
    rescaling the diffusion model output correspondingly, changing the probability of the trajectory, and
    how much each timestep contributes to the probability.
    """
    n_statedim = len(trajectory.shape) - 2
    max_diffusion_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    if len(noise_scheduler.timesteps) != max_diffusion_time:
        raise NotImplementedError(
            "The prob. of sampling trajectory measurement is only implemented for standard"
            "DDPM sampling with no skipping"
        )

    log_prob = (
        torch.distributions.Normal(loc=0.0, scale=1.0)
        .log_prob(trajectory[:, 0])
        .mean(dim=[i for i in range(1, n_statedim + 1)])
    )  # [batch_size,]
    scheduler_timesteps = noise_scheduler.timesteps.to(trajectory.device)
    for i in tqdm.trange(
        0, max_diffusion_time, desc="Trajectory timestep", leave=False
    ):
        trajectory_inputs: FloatTensor = trajectory[:, i]  #  type: ignore
        trajectory_outputs = trajectory[:, i + 1]
        # noise_scheduler.timesteps should be already reversed
        timestep = scheduler_timesteps[i]
        model_output = denoise_model(
            trajectory_inputs,
            timestep,
        ).sample
        # Turn into a prediction distribution
        p_xprev_given_x_mean, p_xprev_given_x_scale = predict_prev_sample(
            xt=trajectory_inputs,
            model_output=model_output,
            timestep=timestep,  #  type: ignore
            noise_scheduler=noise_scheduler,
            # Use the same variance-type as was used to sample the trajectory
            variance_type=noise_scheduler.config.variance_type,
        )
        p_xprev_given_x = Normal(
            loc=p_xprev_given_x_mean,
            scale=p_xprev_given_x_scale,
            validate_args=False,
        )
        log_prob += p_xprev_given_x.log_prob(trajectory_outputs).mean(
            dim=[i for i in range(1, n_statedim + 1)]
        )  # [batch_size,]
    return log_prob  # [batch_size,]


def probability_of_sampling_trajectory_batch_fold(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    trajectory: FloatTensor,  # [num_inference_steps + 1, *state_shape]
    batch_size: int,
) -> Tensor:  # [,]
    """
    The (dimension normalized) probability of sampling the given trajectory from the diffusion model.

    It takes the trajectory and folds it into a batch of images to be processed by the denoising model to
    parallelise.

    However, it's not very useful for taking gradients, because you still to backward pass
    through the whole trajectory.

    Note: This measurement has the issue of not being reparameterisation invariant.
    We could rescale the latent space by an arbitrary factor at different timesteps,
    rescaling the diffusion model output correspondingly, changing the probability of the trajectory, and
    how much each timestep contributes to the probability.
    """
    n_statedim = len(trajectory.shape) - 1
    max_diffusion_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    if len(noise_scheduler.timesteps) != max_diffusion_time:
        raise NotImplementedError(
            "The prob. of sampling trajectory measurement is only implemented for standard"
            "DDPM sampling with no skipping"
        )

    log_prob = (
        torch.distributions.Normal(loc=0.0, scale=1.0).log_prob(trajectory[:, 0]).mean()
    )  # [,]
    scheduler_timesteps = noise_scheduler.timesteps.to(trajectory.device)

    # Partition timesteps into batches:
    num_batches = math.ceil(max_diffusion_time / batch_size)
    for i in range(num_batches):
        batch_time_idxs = torch.arange(
            i * batch_size,
            min((i + 1) * batch_size, max_diffusion_time),
            device=trajectory.device,
        )
        trajectory_inputs: FloatTensor = trajectory[batch_time_idxs]  #  type: ignore
        trajectory_outputs = trajectory[batch_time_idxs + 1]
        # noise_scheduler.timesteps should be already reversed
        timestep = scheduler_timesteps[batch_time_idxs]
        model_output = denoise_model(
            trajectory_inputs,
            timestep,
        ).sample
        # Turn into a prediction distribution
        p_xprev_given_x_mean, p_xprev_given_x_scale = predict_prev_sample(
            xt=trajectory_inputs,
            model_output=model_output,
            timestep=timestep,  #  type: ignore
            noise_scheduler=noise_scheduler,
            # Use the same variance-type as was used to sample the trajectory
            variance_type=noise_scheduler.config.variance_type,
        )
        p_xprev_given_x = Normal(
            loc=p_xprev_given_x_mean,
            scale=p_xprev_given_x_scale,
            validate_args=False,
        )
        log_prob += (
            p_xprev_given_x.log_prob(trajectory_outputs)
            .mean(dim=[i for i in range(1, n_statedim + 1)])
            .sum(dim=0)
        )  # [,]
    return log_prob  # [,]


def probability_of_sampling_trajectory_stochastic_sample(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    trajectory: FloatTensor,  # [num_inference_steps + 1, *state_shape]
    batch_size: int,
) -> Tensor:  # [,]
    """
    The (dimension normalized) probability of sampling the given trajectory from the diffusion model.

    It takes the trajectory and folds it into a batch of images to be processed by the denoising model to
    parallelise.

    Note: This measurement has the issue of not being reparameterisation invariant.
    We could rescale the latent space by an arbitrary factor at different timesteps,
    rescaling the diffusion model output correspondingly, changing the probability of the trajectory, and
    how much each timestep contributes to the probability.
    """
    n_statedim = len(trajectory.shape) - 1
    max_diffusion_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    if len(noise_scheduler.timesteps) != max_diffusion_time:
        raise NotImplementedError(
            "The prob. of sampling trajectory measurement is only implemented for standard"
            "DDPM sampling with no skipping"
        )

    log_prob = (
        torch.distributions.Normal(loc=0.0, scale=1.0).log_prob(trajectory[:, 0]).mean()
    )  # [,]
    scheduler_timesteps = noise_scheduler.timesteps.to(trajectory.device)

    # Partition timesteps into batches:
    # num_batches = math.ceil(max_diffusion_time / batch_size)
    # for i in range(num_batches):
    batch_time_idxs = torch.randint(
        low=0,
        high=max_diffusion_time,
        size=(batch_size,),
        device=trajectory.device,
    )
    trajectory_inputs: FloatTensor = trajectory[batch_time_idxs]  #  type: ignore
    trajectory_outputs = trajectory[batch_time_idxs + 1]
    # noise_scheduler.timesteps should be already reversed
    timestep = scheduler_timesteps[batch_time_idxs]
    model_output = denoise_model(
        trajectory_inputs,
        timestep,
    ).sample
    # Turn into a prediction distribution
    p_xprev_given_x_mean, p_xprev_given_x_scale = predict_prev_sample(
        xt=trajectory_inputs,
        model_output=model_output,
        timestep=timestep,  #  type: ignore
        noise_scheduler=noise_scheduler,
        # Use the same variance-type as was used to sample the trajectory
        variance_type=noise_scheduler.config.variance_type,
    )
    p_xprev_given_x = Normal(
        loc=p_xprev_given_x_mean,
        scale=p_xprev_given_x_scale,
        validate_args=False,
    )
    log_prob += (
        p_xprev_given_x.log_prob(trajectory_outputs)
        .mean(dim=[i for i in range(1, n_statedim + 1)])
        .sum(dim=0)
    )  # [,]
    return log_prob  # [,]
