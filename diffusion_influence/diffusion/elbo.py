import math
from typing import Optional, Sequence

import torch
import tqdm
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from torch import FloatTensor, Tensor, sqrt
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


# --- Functions to extend DDPMScheduler beyond just sampling (but actually returning mean and variance) ---
def predict_prev_sample(
    xt: torch.FloatTensor,
    model_output: torch.FloatTensor,
    timestep: torch.IntTensor | int,
    noise_scheduler: DDPMScheduler,
    variance_type: Optional[str] = None,
) -> tuple[Tensor, Tensor]:
    t: torch.IntTensor = (
        torch.tensor(timestep, device=xt.device, dtype=torch.long)
        if isinstance(timestep, int)
        else timestep
    )
    state_ndim = xt.ndim - 1
    if isinstance(noise_scheduler, DDPMScheduler):
        # diffusers.DDPMScheduler.step() doesn't support a timestep that's a tensor, and
        # it doesn't return the variance/std, so we need to
        # implement prediction for previous sample manually

        num_inference_steps = (
            noise_scheduler.num_inference_steps
            if noise_scheduler.num_inference_steps
            else noise_scheduler.config.num_train_timesteps
        )  #  type: ignore
        if noise_scheduler.custom_timesteps:
            raise NotImplementedError
        prev_t = t - noise_scheduler.config.num_train_timesteps // num_inference_steps  #  type: ignore

        # 1. compute alphas, betas
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0,
            noise_scheduler.alphas_cumprod[torch.clip(prev_t, min=0, max=None)],
            noise_scheduler.one,
        )  #  if isinstance(prev_t, Tensor) else (
        # noise_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else noise_scheduler.one
        # )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        # 1.5 Make all alphas, betas broadcastable with the state shape:
        (
            alpha_prod_t,
            alpha_prod_t_prev,
            beta_prod_t,
            beta_prod_t_prev,
            current_alpha_t,
            current_beta_t,
        ) = map(
            lambda coeff: coeff.view(-1, *[1] * state_ndim),
            (
                alpha_prod_t,
                alpha_prod_t_prev,
                beta_prod_t,
                beta_prod_t_prev,
                current_alpha_t,
                current_beta_t,
            ),
        )

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if noise_scheduler.config.prediction_type == "epsilon":  # type: ignore
            pred_original_sample = (
                xt - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif noise_scheduler.config.prediction_type == "sample":  #  type: ignore
            pred_original_sample = model_output
        elif noise_scheduler.config.prediction_type == "v_prediction":  #  type: ignore
            pred_original_sample = (alpha_prod_t**0.5) * xt - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"  #  type: ignore
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if noise_scheduler.config.thresholding:  #  type: ignore
            pred_original_sample = noise_scheduler._threshold_sample(
                pred_original_sample
            )  #  type: ignore
        elif noise_scheduler.config.clip_sample:  #  type: ignore
            pred_original_sample = pred_original_sample.clamp(
                -noise_scheduler.config.clip_sample_range,
                noise_scheduler.config.clip_sample_range,  #  type: ignore
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample_mean = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * xt
        )

        # 6. Get the variance
        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance: Tensor = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = noise_scheduler.config.variance_type  #  type: ignore

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            pass
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif noise_scheduler.config.variance_type in {
            "learned",
            "learned_range",
            "fixed_large_log",
            "fixed_small_log",
        }:  #  type: ignore
            raise NotImplementedError

        std = torch.sqrt(variance)

        return pred_prev_sample_mean, std
    else:
        raise NotImplementedError


def noise_scheduler_predict_prev_sample_given_original(
    x0: torch.FloatTensor,
    xt: torch.FloatTensor,
    timestep: torch.LongTensor
    | int,  # Note, timestep here is different than t in the DDPM paper!
    noise_scheduler: DDPMScheduler,
    validate_args: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Note, timestep here is different than `t` in the DDPM paper! It's shifted by `1`,
    so `timestep=0` corresponds to `t=1` in the DDPM paper. (this is because all the schedule values
    are 1-indexed in the paper, but stored with 0-indexing in the `diffusers` codebase)
    """
    t = timestep
    prev_t = timestep - 1
    state_ndim = x0.ndim - 1
    if validate_args:
        assert torch.all(prev_t >= 0) if isinstance(prev_t, Tensor) else prev_t >= 0, (
            "Previous timestep should be >= 0"
        )
    if isinstance(noise_scheduler, DDPMScheduler):
        # Compute alphas, betas
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = noise_scheduler.alphas_cumprod[prev_t]
        alpha_t = alpha_prod_t / alpha_prod_t_prev
        beta_t = 1 - alpha_t

        # Compute "interpolation" coefficient between x_0 and x_t
        x0_coeff = sqrt(alpha_prod_t_prev) * beta_t / (1 - alpha_prod_t)
        xt_coeff = sqrt(alpha_t) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        # Expand x0_coeff and xt_coeff to shape broadcastable with x0 and xt
        x0_coeff = x0_coeff.view(-1, *[1] * state_ndim)
        xt_coeff = xt_coeff.view(-1, *[1] * state_ndim)

        loc = x0_coeff * x0 + xt_coeff * xt
        variance = (1 - alpha_prod_t_prev) * beta_t / (1 - alpha_prod_t)
        scale = sqrt(variance)
        # Expand scale to shape broadcastable with loc
        scale = scale.view(-1, *[1] * state_ndim)
        return loc, scale
    else:
        raise NotImplementedError


def noise_scheduler_get_noise_mean_scale(
    noise_scheduler: DDPMScheduler,
    original_samples: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Note, timestep here is different than `t` in the DDPM paper! It's shifted by `1`,
    so `timestep=0` corresponds to `t=1` in the DDPM paper. (this is because all the schedule values
    are 1-indexed in the paper, but stored with 0-indexing in the `diffusers` codebase)
    """
    if isinstance(noise_scheduler, DDPMScheduler):
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)  # type: ignore

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        squeeze_factor = sqrt_alpha_prod
        noise_mean = squeeze_factor * original_samples
        noise_scale = sqrt_one_minus_alpha_prod
        return noise_mean, noise_scale
    else:
        raise NotImplementedError


# --- Functions to compute the ELBO ---
def stochastic_elbo_sample(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    data_range: tuple[
        Tensor | float, Tensor | float
    ],  # Each broadcastable with state_shape
) -> Tensor:  # [batch_size,]
    """
    For "stochastic" ELBO computation, we sample the time-index for which term
    in the ELBO sum in eq. (5) in the DDPM paper to compute.
    Terms 1 < t <= T correspond to the KL divergence terms, while t=1 corresponds
    to the reconstruction log-likelihood term. When sampling times for the
    diffusers library scheduler, all the time indices are shifted by 1 ofc.
    """
    batch_size = original_samples.shape[0]
    max_diffusion_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    # Sample a random timestep for each image
    timesteps: torch.IntTensor = torch.randint(
        low=0,
        high=max_diffusion_time,
        size=(batch_size,),
        device=original_samples.device,
        dtype=torch.long,
    )  # type: ignore
    # Sample noise for each image
    xt = noise_scheduler.add_noise(
        original_samples=original_samples,
        noise=torch.randn_like(original_samples),
        timesteps=timesteps,
    )  # [batch_size, *state_shape]
    model_output = denoise_model(xt, timesteps).sample

    # Compute the KL-intermediate term (L_t) or reconstruction term (L_0) depending
    # on the timestep
    kl_terms = kl_between_intermediate_states_from_model_output(
        model_output=model_output,
        x0=original_samples,
        xt=xt,
        # Clip to [1, max_time], otherwise the KL is undefined. We're going to
        # ignore the KL for the first timesteps==0 anyway, so the value doesn't matter
        timesteps=torch.clip(timesteps, min=1, max=None),
        noise_scheduler=noise_scheduler,
    )
    reconstruction_log_likelihood_terms = (
        reconstruction_log_likelihood_from_model_output(
            model_output=model_output,
            x0=original_samples,
            x1=xt,
            noise_scheduler=noise_scheduler,
            data_range=data_range,
        )
    )
    # At timestep 0, the ELBO term is the reconstruction log-likelihood, for all other timesteps
    # it's the KL term
    terms = torch.where(
        timesteps == 0,
        reconstruction_log_likelihood_terms,
        -kl_terms,
    )
    return terms * max_diffusion_time - kl_to_prior(
        original_samples=original_samples, noise_scheduler=noise_scheduler
    )


def stochastic_elbo_sample_dequantized_data(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    data_range: tuple[
        Tensor | float, Tensor | float
    ],  # Each broadcastable with state_shape
) -> Tensor:  # [batch_size,]
    """
    This assumes a diffusion model expecting dequantized data. Dequantization is
    treated as part of the ELBO bound, and sampled during the ELBO computation.

    For "stochastic" ELBO computation, we sample the time-index for which term
    in the ELBO sum in eq. (5) in the DDPM paper to compute.
    Terms 0 < t <= T correspond to the KL divergence terms, while t=1 corresponds
    to the p(x_0 |x_1) log-likelihood term. When sampling times for the
    diffusers library scheduler, all the time indices are shifted by 1 ofc.

    Note: This function ignore the dequantization entropy term H[q(x_0|x_quant)].
    For state dimensions for typical images, this is pretty much always so close to 0
    that it's outside the numerical accuracy of float64.
    """
    batch_ndim = 1
    state_ndim = original_samples.ndim - batch_ndim
    batch_size = original_samples.shape[0]

    data_range_min, data_range_max = data_range
    data_range_delta = data_range_max - data_range_min

    quantized_samples = torch.floor(
        ((original_samples + data_range_min) / data_range_delta)  # to [0, 1] range
        * 256.0,  # back to [0, 256] range
    )  # integers in [0, ..., 255]
    # Resample the dequantized samples
    dequantized_samples: FloatTensor = (
        (
            (
                quantized_samples + torch.rand_like(quantized_samples)
            )  # add uniform noise
            / 256.0  # back to [0, 1] range
        )
        * data_range_delta
        + data_range_min  # back to original range
    )  #  type: ignore

    max_diffusion_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    # Sample a random timestep for each image
    timesteps: torch.IntTensor = torch.randint(
        low=0,
        high=max_diffusion_time,
        size=(batch_size,),
        device=original_samples.device,
        dtype=torch.long,
    )  # type: ignore
    # Sample noise for each image
    xt = noise_scheduler.add_noise(
        original_samples=dequantized_samples,
        noise=torch.randn_like(original_samples),
        timesteps=timesteps,
    )  # [batch_size, *state_shape]
    model_output = denoise_model(xt, timesteps).sample

    # Compute the KL-intermediate term (L_t) or reconstruction term (L_0) depending
    # on the timestep
    kl_terms = kl_between_intermediate_states_from_model_output(
        model_output=model_output,
        x0=dequantized_samples,
        xt=xt,
        # Clip to [1, max_time], otherwise the KL is undefined. We're going to
        # ignore the KL for the first timesteps==0 anyway, so the value doesn't matter
        timesteps=torch.clip(timesteps, min=1, max=None),
        noise_scheduler=noise_scheduler,
    )

    p_x0_given_x1_mean, p_x0_given_x1_scale = predict_prev_sample(
        xt=xt,
        model_output=model_output,
        timestep=torch.zeros_like(timesteps),  #  type: ignore
        noise_scheduler=noise_scheduler,
        # We hard-code the variance type to "fixed_large" for the reverse process
        # as "fixed_small" works terribly for ELBO computation
        variance_type="fixed_large",
    )
    p_x0_given_x1 = Normal(
        loc=p_x0_given_x1_mean,
        scale=p_x0_given_x1_scale,
        validate_args=False,
    )
    time_0_term = p_x0_given_x1.log_prob(dequantized_samples).sum(
        dim=list(range(batch_ndim, batch_ndim + state_ndim))
    )  # Reduce over the state dimensions only

    # At timestep 0, the ELBO term is the reconstruction log-likelihood, for all other timesteps
    # it's the KL term
    terms = torch.where(
        timesteps == 0,
        time_0_term,
        -kl_terms,
    )

    return terms * max_diffusion_time - kl_to_prior(
        original_samples=dequantized_samples, noise_scheduler=noise_scheduler
    )


def ddpm_elbo(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    # data_range: tuple[Tensor | float, Tensor | float],  # Each broadcastable with state_shape
    num_samples_per_timestep: int = 1,
    subsampled_timesteps: Optional[Sequence[int]] = None,
):
    """
    Note! `subsampled_timesteps` are assumed to be 0-indexed as is convention in the diffusers codebase, i.e. a timestep
    value of 0 corresponds to `t=1` in the DDPM paper notation.
    """
    if noise_scheduler.config.thresholding:  # type: ignore
        raise ValueError(
            "The noise scheduler should not have clip_sample or thresholding enabled "
            "for computing the ELBO."
        )
    if noise_scheduler.config.variance_type not in ["fixed_small", "fixed_large"]:  # type: ignore
        raise NotImplementedError
    if (
        noise_scheduler.num_inference_steps is not None
        and noise_scheduler.config.num_train_timesteps
        != noise_scheduler.num_inference_steps  # type: ignore
    ):
        raise ValueError(
            "For ELBO computation, the number of training timesteps should be equal to "
            "the number of inference steps in the scheduler."
        )
    # Note: we might want `noise_scheduler.config.clip_sample` to be False as well,
    # althought technically original DDPM also uses clipping for their ELBO log-likelihood
    # evaluation.

    # Compute KL[q(x_T|x_0)||p(x_T)]: the KL to the "final" latent at time T (prior):

    kl_T = kl_to_prior(
        original_samples=original_samples, noise_scheduler=noise_scheduler
    )  # [batch_size,]

    # Compute KL[q(x_t-1|x_0, x_t)||p(x_t-1, x_t)]:
    # the KL terms L_t for 2 <= t < max_time (eq. (5) from DDPM paper https://arxiv.org/pdf/2006.11239.pdf)
    max_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    if subsampled_timesteps is not None:
        assert set(subsampled_timesteps) <= set(range(1, max_time)), (
            "Subsampled timesteps should be in [1, max_time)"
        )
        intermediate_terms_timesteps = subsampled_timesteps
    else:
        intermediate_terms_timesteps = range(1, max_time)

    kl_ts = 0
    for t in tqdm.tqdm(intermediate_terms_timesteps, leave=False):
        t = torch.tensor(t, device=original_samples.device)  # [1]
        kl_t: Tensor = (
            sum(
                kl_sample_between_intermediate_states(
                    model=denoise_model,
                    original_samples=original_samples,
                    timesteps=t,
                    noise_scheduler=noise_scheduler,
                )
                for _ in range(num_samples_per_timestep)
            )  #  type: ignore
            / num_samples_per_timestep
        )  #  type: ignore
        # kl_t_list.append(kl_t)
        kl_ts += kl_t
    # Rescale if we're using subsampled timesteps:
    kl_ts *= (max_time - 1) / len(intermediate_terms_timesteps)

    # Compute E_[q(x_1|x0][log p(x_0|x1)] - the reconstruction term:
    likelihood_term = reconstruction_log_likelihood(
        denoise_model=denoise_model,
        original_samples=original_samples,
        noise_scheduler=noise_scheduler,
        data_range=(-1.0, 1.0),
    )
    elbo = -kl_T - kl_ts + likelihood_term  # [batch_size]
    return elbo

    # return -kl_T - sum(kl_t_list) + likelihood_term


def kl_to_prior(
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    noise_scheduler: DDPMScheduler,
    validate_args: bool = False,
) -> Tensor:  # [batch_size]
    max_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    state_shape = original_samples.shape[1:]

    q_xT_given_x0_mean, q_xT_given_x0_scale = noise_scheduler_get_noise_mean_scale(
        noise_scheduler=noise_scheduler,
        original_samples=original_samples,
        # We're setting the timesteps to the last timestep, i.e. T - 1 in the diffusers 0-indexed notation
        # (i.e. T in the DDPM paper notation)
        timesteps=torch.tensor(
            max_time - 1, device=original_samples.device, dtype=torch.long
        ),  # type: ignore
    )  # [batch_size, *state_shape]
    q_xT_given_x0 = Normal(
        loc=q_xT_given_x0_mean, scale=q_xT_given_x0_scale, validate_args=validate_args
    )  # [batch_size, *state_shape]
    kl = kl_divergence(q_xT_given_x0, Normal(0, scale=1))
    return kl.sum(
        dim=list(range(1, len(state_shape) + 1))
    )  # Sum over the state dimensions


def kl_sample_between_intermediate_states(
    model: UNet2DModel,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    timesteps: torch.IntTensor,  # [batch_size] or [,]
    noise_scheduler: DDPMScheduler,
    noise_sample: Optional[torch.FloatTensor] = None,
) -> Tensor:  # [batch_size]
    """
    Note, timestep here is different than `t` in the DDPM paper! It's shifted by `1`,
    so `timestep=0` corresponds to `t=1` in the DDPM paper. (this is because all the schedule values
    are 1-indexed in the paper, but stored with 0-indexing in the `diffusers` codebase)
    """
    xt = noise_scheduler.add_noise(
        original_samples=original_samples,
        noise=noise_sample
        if noise_sample is not None
        else torch.randn_like(original_samples),  # type: ignore
        timesteps=timesteps,
    )  # [batch_size, *state_shape]
    model_output = model(xt, timesteps).sample
    return kl_between_intermediate_states_from_model_output(
        model_output=model_output,
        x0=original_samples,
        xt=xt,
        timesteps=timesteps,
        noise_scheduler=noise_scheduler,
    )


def kl_between_intermediate_states_from_model_output(
    model_output: torch.FloatTensor,  # [batch_size, *state_shape]
    x0: torch.FloatTensor,  # [batch_size, *state_shape]
    xt: torch.FloatTensor,  # [batch_size, *state_shape]
    timesteps: torch.IntTensor,  # [batch_size] or [,]
    noise_scheduler: DDPMScheduler,
    validate_args: bool = False,
) -> Tensor:  # [batch_size]
    """
    Computes the KL divergence KL[q(x_t-1|x_0, x_t)||p(x_t-1| x_t)] for a single timestep `t`
    for a single sample of x_t. When x_t is sampled from q(x_t|x_0), this function
    corresponds to an MC sample of the L_t term in eq. (5) from DDPM paper:
    https://arxiv.org/pdf/2006.11239.pdf

    Args:
        model_output: The output of the denoising model for the sample (xt, timesteps)
        x0: The original sample x_0
        xt: The noised up version of the sample x_0, assumed sampled from x_t ~ q(x_t|x_0)
        timesteps: The timestep `t` (0-indexed, i.e. an index of 0 corresponds to `t=1` in the DDPM paper notation)
        noise_scheduler: The noise scheduler object
    Returns:
        A tensor of shape [batch_size,], with each entry corresponding to the
        KL divergence KL[q(x_t-1|x_0, x_t)||p(x_t-1, x_t)] for the samples
    """
    batch_ndim = 1
    state_ndim = x0.ndim - batch_ndim

    (
        q_xt_prev_given_x0_xt_mean,
        q_xt_prev_given_x0_xt_scale,
    ) = noise_scheduler_predict_prev_sample_given_original(
        x0=x0,
        xt=xt,
        timestep=timesteps,
        noise_scheduler=noise_scheduler,
    )  # [batch_size, *state_shape] and [batch_size]

    q_xt_prev_given_x0_xt = Normal(
        loc=q_xt_prev_given_x0_xt_mean,
        scale=q_xt_prev_given_x0_xt_scale,
        validate_args=validate_args,
    )
    p_xt_prev_given_xt_mean, p_xt_prev_given_xt_scale = predict_prev_sample(
        xt=xt,
        model_output=model_output,
        timestep=timesteps,
        noise_scheduler=noise_scheduler,
        # We hard-code the variance type to "fixed_large" for the reverse process
        # as "fixed_small" works terribly for ELBO computation
        variance_type="fixed_large",
    )
    p_xt_prev_given_xt = Normal(
        loc=p_xt_prev_given_xt_mean,
        scale=p_xt_prev_given_xt_scale,
        validate_args=validate_args,
    )
    return kl_divergence(
        q_xt_prev_given_x0_xt,
        p_xt_prev_given_xt,
    ).sum(
        dim=list(range(batch_ndim, batch_ndim + state_ndim))
    )  # Reduce over the state dimensions only


def reconstruction_log_likelihood(
    denoise_model: UNet2DModel,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    noise_scheduler: DDPMScheduler,
    data_range: tuple[
        Tensor | float, Tensor | float
    ],  # Each broadcastable with state_shape
    noise_sample: Optional[Tensor] = None,
) -> Tensor:  # [batch_size]
    timesteps = torch.tensor(0, dtype=torch.long, device=original_samples.device)

    x1 = noise_scheduler.add_noise(
        original_samples=original_samples,
        noise=noise_sample
        if noise_sample is not None
        else torch.randn_like(original_samples),  # type: ignore
        timesteps=timesteps,  # All zeros # type: ignore
    )  # [batch_size, *state_shape]
    model_output = denoise_model(x1, timesteps).sample
    return reconstruction_log_likelihood_from_model_output(
        model_output=model_output,
        x0=original_samples,
        x1=x1,
        noise_scheduler=noise_scheduler,
        data_range=data_range,
    )


def reconstruction_log_likelihood_from_model_output(
    model_output: torch.FloatTensor,  # [batch_size, *state_shape]
    x0: torch.FloatTensor,  # [batch_size, *state_shape]
    x1: torch.FloatTensor,  # [batch_size, *state_shape]
    noise_scheduler: DDPMScheduler,
    data_range: tuple[
        Tensor | float, Tensor | float
    ],  # Each broadcastable with state_shape
    validate_args: bool = False,
) -> Tensor:  # [batch_size]
    stateshape_ndim = x0.ndim - 1

    timesteps = torch.tensor(0, dtype=torch.long, device=x0.device)

    p_x0_given_x1_continuous_mean, p_x0_given_x1_continuous_scale = predict_prev_sample(
        xt=x1,
        model_output=model_output,
        timestep=timesteps,  # type: ignore
        noise_scheduler=noise_scheduler,
        # We hard-code the variance type to "fixed_large" for the reverse process
        # as "fixed_small" works terribly for ELBO computation
        variance_type="fixed_large",
    )  # [batch_size, *state_shape] and broadcastable to [batch_size, *state_shape]
    p_x0_given_x1_continuous = Normal(
        loc=p_x0_given_x1_continuous_mean,
        scale=p_x0_given_x1_continuous_scale,
        validate_args=validate_args,
    )
    # The actual likelihood is the discretization of the Gaussian
    # (see eq. (13) in DDPM paper: https://arxiv.org/pdf/2006.11239.pdf)
    bin_width = (data_range[1] - data_range[0]) / 255
    cdf_upper = p_x0_given_x1_continuous.cdf(
        torch.where(
            x0 >= data_range[1] - bin_width / 2,
            torch.inf,
            x0 + bin_width / 2,
        )
    )
    cdf_lower = p_x0_given_x1_continuous.cdf(
        torch.where(
            x0 <= data_range[0] + bin_width / 2,
            -torch.inf,
            x0 - bin_width / 2,
        )
    )
    cdf_difference = cdf_upper - cdf_lower
    # Same as in the original DDPM code, clip the log_probs to log(1e-12) to avoid -inf
    log_prob = torch.log(torch.clip(cdf_difference, min=1e-12, max=None))
    return log_prob.sum(
        dim=list(range(1, stateshape_ndim + 1))
    )  # Sum over the state dimensions


def ddpm_elbo_dequantized_data(
    denoise_model: UNet2DModel,
    noise_scheduler: DDPMScheduler,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    num_samples_per_timestep: int = 1,
    subsampled_timesteps: Optional[Sequence[int]] = None,
    data_range: tuple[Tensor | float, Tensor | float] = (
        -1.0,
        1.0,
    ),  # Each broadcastable with state_shape
):
    """
    Like `ddpm_elbo`, but assumes that the models are trained on dequantized data.
    We assume that the data is dequentized uniformly within the range of each pixel
    and transformed to [-1, 1) range. The ELBO is computed by taking into the account
    the dequantization (encoder) and requantization (decoder) steps in the log-likelihood,
    and just applying the “regular” ELBO for continuous densities in the latent
    dequantised space.

    Note! `subsampled_timesteps` are assumed to be 0-indexed as is convention in the
    diffusers codebase, i.e. a timestep value of 0 corresponds to `t=1` in the DDPM paper notation.
    """
    if noise_scheduler.config.thresholding:  # type: ignore
        raise ValueError(
            "The noise scheduler should not have clip_sample or thresholding enabled "
            "for computing the ELBO."
        )
    if noise_scheduler.config.variance_type not in ["fixed_small", "fixed_large"]:  # type: ignore
        raise NotImplementedError
    if (
        noise_scheduler.num_inference_steps is not None
        and noise_scheduler.config.num_train_timesteps
        != noise_scheduler.num_inference_steps  # type: ignore
    ):
        raise ValueError(
            "For ELBO computation, the number of training timesteps should be equal to "
            "the number of inference steps in the scheduler."
        )
    # Note: we might want `noise_scheduler.config.clip_sample` to be False as well,
    # althought technically original DDPM also uses clipping for their ELBO log-likelihood
    # evaluation.

    data_range_min, data_range_max = data_range
    data_range_delta = data_range_max - data_range_min

    quantized_samples = torch.floor(
        ((original_samples + data_range_min) / data_range_delta)  # to [0, 1] range
        * 256.0,  # back to [0, 256] range
    )  # integers in [0, ..., 255]

    # Resample the dequantized samples
    def get_dequantized_samples() -> FloatTensor:
        dequantized_samples: FloatTensor = (
            (
                (
                    quantized_samples + torch.rand_like(quantized_samples)
                )  # add uniform noise
                / 256.0  # back to [0, 1] range
            )
            * data_range_delta
            + data_range_min  # back to original range
        )  #  type: ignore
        return dequantized_samples

    # Compute KL[q(x_T|x_0)||p(x_T)]: the KL to the "final" latent at time T (prior):

    kl_T = kl_to_prior(
        original_samples=get_dequantized_samples(), noise_scheduler=noise_scheduler
    )  # [batch_size,]

    # Compute KL[q(x_t-1|x_0, x_t)||p(x_t-1, x_t)]:
    # the KL terms L_t for 2 <= t < max_time (eq. (5) from DDPM paper https://arxiv.org/pdf/2006.11239.pdf)
    max_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    if subsampled_timesteps is not None:
        assert set(subsampled_timesteps) <= set(range(1, max_time)), (
            "Subsampled timesteps should be in [1, max_time)"
        )
        intermediate_terms_timesteps = subsampled_timesteps
    else:
        intermediate_terms_timesteps = range(1, max_time)

    kl_ts = 0
    for t in tqdm.tqdm(intermediate_terms_timesteps, leave=False):
        t = torch.tensor(t, device=original_samples.device)  # [1]
        kl_t: Tensor = (
            sum(
                kl_sample_between_intermediate_states(
                    model=denoise_model,
                    original_samples=get_dequantized_samples(),
                    timesteps=t,
                    noise_scheduler=noise_scheduler,
                )
                for _ in range(num_samples_per_timestep)
            )  #  type: ignore
            / num_samples_per_timestep
        )  #  type: ignore
        # kl_t_list.append(kl_t)
        kl_ts += kl_t
    # Rescale if we're using subsampled timesteps:
    kl_ts *= (max_time - 1) / len(intermediate_terms_timesteps)

    # Compute E_[q(x_1|x0][log p(x_0|x1)] - the reconstruction term:
    t = torch.tensor(0, device=original_samples.device, dtype=torch.long)
    x0 = get_dequantized_samples()
    x1 = noise_scheduler.add_noise(
        original_samples=x0,
        noise=torch.randn_like(original_samples),
        timesteps=t,  # type: ignore
    )  # [batch_size, *state_shape]
    model_output = denoise_model(x1, t).sample
    p_x0_given_x1_mean, p_x0_given_x1_scale = predict_prev_sample(
        xt=x1,
        model_output=model_output,
        timestep=t,  #  type: ignore
        noise_scheduler=noise_scheduler,
        # We hard-code the variance type to "fixed_large" for the reverse process
        # as "fixed_small" works terribly for ELBO computation
        variance_type="fixed_large",
    )
    p_x0_given_x1 = Normal(
        loc=p_x0_given_x1_mean,
        scale=p_x0_given_x1_scale,
        validate_args=False,
    )
    batch_ndim = 1
    state_ndim = original_samples.ndim - batch_ndim
    time_0_term = p_x0_given_x1.log_prob(x0).sum(
        dim=list(range(batch_ndim, batch_ndim + state_ndim))
    )  # Reduce over the state dimensions only
    likelihood_term = time_0_term

    elbo = -kl_T - kl_ts + likelihood_term  # [batch_size]
    return elbo


def ldm_elbo(
    denoise_model: UNet2DModel,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    original_samples: torch.FloatTensor,  # [batch_size, *state_shape]
    num_samples_per_timestep: int = 1,
    subsampled_timesteps: Optional[Sequence[int]] = None,
    data_range: tuple[Tensor | float, Tensor | float] = (
        -1.0,
        1.0,
    ),  # Each broadcastable with state_shape
    # In principle, we should optimise the VAE decoder output
    # variance as well (as it is not given as part of the Hugging Face models)
    # But for the case of evaluating ELBO with a fixed VAE, this will only affect the ELBO by a constant additive value through
    # the VAE reconstruction term.
    # Hence, for most cases it can be set to 1
    vae_log_var: float = 0,
):
    """
    Like `ddpm_elbo`, but computes the ELBO for the whole LDM model, including the VAE encoder and decoder. If the original image space is x, the latent space to which the VAE encodes is z0, and diffusion goes over z0, z1, ..., zT, then an ELBO can be written as:
        E_q(z0|x)[ddpm_elbo(z0) ] + E_q(z0|x)[log p(x|z0)] + H[q(z0|x)]
    where:
    - q(z0|x) is the VAE encoder
    - p(x|z0) is the VAE decoder
    - ddpm_elbo(z0) is the "usual" ELBO for the log-marginal likelihood log p(z0) for a diffusion model.

    Note! `subsampled_timesteps` are assumed to be 0-indexed as is convention in the
    diffusers codebase, i.e. a timestep value of 0 corresponds to `t=1` in the DDPM paper notation.
    """
    if noise_scheduler.config.thresholding:  # type: ignore
        raise ValueError(
            "The noise scheduler should not have clip_sample or thresholding enabled "
            "for computing the ELBO."
        )
    if noise_scheduler.config.variance_type not in ["fixed_small", "fixed_large"]:  # type: ignore
        raise NotImplementedError
    if (
        noise_scheduler.num_inference_steps is not None
        and noise_scheduler.config.num_train_timesteps
        != noise_scheduler.num_inference_steps  # type: ignore
    ):
        raise ValueError(
            "For ELBO computation, the number of training timesteps should be equal to "
            "the number of inference steps in the scheduler."
        )
    latent_z0_dist = vae.encode(original_samples).latent_dist

    def latent_z0_sample() -> FloatTensor:
        return latent_z0_dist.sample() * vae.config.scaling_factor

    # Compute MC sample of the ELBO for log p(z0) (in expectation over samples from q(z0|x0))
    kl_T = kl_to_prior(
        original_samples=latent_z0_sample(), noise_scheduler=noise_scheduler
    )  # [batch_size,]

    # Compute KL[q(z_t-1|z_0, z_t)||p(z_t-1, z_t)]:
    # the KL terms L_t for 2 <= t < max_time (eq. (5) from DDPM paper https://arxiv.org/pdf/2006.11239.pdf)
    max_time: int = noise_scheduler.config.num_train_timesteps  # type: ignore
    if subsampled_timesteps is not None:
        assert set(subsampled_timesteps) <= set(range(1, max_time)), (
            "Subsampled timesteps should be in [1, max_time)"
        )
        intermediate_terms_timesteps = subsampled_timesteps
    else:
        intermediate_terms_timesteps = range(1, max_time)

    kl_ts = 0
    for t in tqdm.tqdm(intermediate_terms_timesteps, leave=False):
        t = torch.tensor(t, device=original_samples.device)  # [1]
        kl_t: Tensor = (
            sum(
                kl_sample_between_intermediate_states(
                    model=denoise_model,
                    original_samples=latent_z0_sample(),
                    timesteps=t,
                    noise_scheduler=noise_scheduler,
                )
                for _ in range(num_samples_per_timestep)
            )  #  type: ignore
            / num_samples_per_timestep
        )  #  type: ignore
        kl_ts += kl_t
    # Rescale if we're using subsampled timesteps:
    kl_ts *= (max_time - 1) / len(intermediate_terms_timesteps)

    # Compute E_[q(z_1|z0][log p(z_0|z1)] - the reconstruction term:
    t = torch.tensor(0, device=original_samples.device, dtype=torch.long)
    z0 = latent_z0_sample()
    z1 = noise_scheduler.add_noise(
        original_samples=z0,
        noise=torch.randn_like(z0),
        timesteps=t,  # type: ignore
    )  # [batch_size, *state_shape]
    model_output = denoise_model(z1, t).sample
    p_z0_given_z1_mean, p_z0_given_z1_scale = predict_prev_sample(
        xt=z1,
        model_output=model_output,
        timestep=t,  #  type: ignore
        noise_scheduler=noise_scheduler,
        # We hard-code the variance type to "fixed_large" for the reverse process
        # as "fixed_small" works terribly for ELBO computation
        variance_type="fixed_large",
    )
    p_z0_given_z1 = Normal(
        loc=p_z0_given_z1_mean,
        scale=p_z0_given_z1_scale,
        validate_args=False,
    )
    batch_ndim = 1
    state_ndim = original_samples.ndim - batch_ndim
    time_0_term = p_z0_given_z1.log_prob(z0).sum(
        dim=list(range(batch_ndim, batch_ndim + state_ndim))
    )  # Reduce over the state dimensions only
    likelihood_term = time_0_term

    elbo_latent = -kl_T - kl_ts + likelihood_term  # [batch_size]

    # The ELBO for LDMs can be deconstructed into the ELBO for the latent space in expectation over q(z0|x) and the terms:
    # 1) Reconstruction term E_q(z0|x)[log p(x|z0)]
    z0 = latent_z0_sample()
    reconstruction_term_mean = vae.decode(z0).sample
    # The above is called `sample``, but it is the decoder mean. huggingface
    # just has interesting naming conventions.
    reconstruction_term = (
        -0.5
        * (original_samples - reconstruction_term_mean) ** 2
        / math.exp(vae_log_var)
        - 0.5 * math.log(2 * math.pi)
        - vae_log_var / 2
    ).sum(dim=(1, 2, 3))  # Sum over the state dimensions
    # 2) Entropy term H[q(z0|x)]
    encoder_entropy = latent_z0_dist.nll(latent_z0_sample())

    elbo = elbo_latent + reconstruction_term + encoder_entropy

    return elbo
