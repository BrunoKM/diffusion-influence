from typing import Literal, Optional, Protocol

import numpy as np
import torch
from scipy import integrate

from diffusion_influence.config_schemas import DataRangeType
from diffusion_influence.likelihood.sde_interface import (
    SDE,
    VPSDE,
)


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.

    Returns:
        A model function.
    """

    def model_fn(x, labels):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(x, labels).sample  # .sample necessary for diffusers models
        else:
            model.train()
            return model(x, labels).sample  # .sample necessary for diffusers models

    return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Note: Assumes the data passed to score_fn is already normalized to the input range the model expects it in.

    Args:
        sde: An `SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to
            directly take continuous time steps. To plug a DDPM model into
            this function, set `continuous=False`.

    Returns:
        A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, VPSDE):

        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None, None]
            return score

        return score_fn
    else:
        raise NotImplementedError


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_func(f):
    """Create the divergence function of `f` using the Hutchinson trace estimator, noting that:
    div(f) = tr(∇f) = E[eps
    """

    def div_func(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(f(x, t) * eps)
            grad_func_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_func_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_func


class DataScalerFunc(Protocol):
    """Data-scaler that operates on inputs element-wise"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def get_likelihood_fn(
    sde: SDE,
    # inverse_scaler: DataScalerFunc,
    data_range_for_correction: Optional[DataRangeType] = DataRangeType.UNIT_BOUNDED,
    hutchinson_type: Literal["rademacher", "gaussian"] = "rademacher",
    rtol: float = 1e-5,
    atol: float = 1e-5,
    method="RK45",
    eps: float = 1e-5,
    unit: Literal["NPD", "BPD"] = "BPD",  # nats per dim or bits per dim
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Actually compute the likelihood (not negative likelihood) and return in bits/dim or nats/dim.

    Explicitly assumes the data only has one dimension!

    Args:
      sde: An `SDE` object that represents the forward SDE.
      data_range_for_correction: The data range to correct the log-likelihood with
        the change of variables formula. This is because we're usually interested in
        the log-likelihood in “the original data space”, not the one in which the
        diffusion model SDE is defined. If `None`, no correction is applied.
      hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(model, x, t, noise):
        return get_div_func(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim or nats/dim.

        Args:
          model: A score model.
          data: A PyTorch tensor. Already normalized to the data range the SDE expects (usually [-1, 1)).

        Note! Assumes the data passed to score_fn is already normalized to the data range the SDE (i.e. usually the [-1, 1) range for all image experiments). The likelihood, however, is compute for the original data range (in this case [0, 1))!

        Returns:
          bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
          z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == "gaussian":
                epsilon = torch.randn_like(data)
            elif hutchinson_type == "rademacher":
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = (
                    from_flattened_numpy(x[: -shape[0]], shape)
                    .to(data.device)
                    .type(torch.float32)
                )
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(model, sample, vec_t))
                logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate(
                [to_flattened_numpy(data), np.zeros((shape[0],))], axis=0
            )
            solution = integrate.solve_ivp(
                ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method
            )
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = (
                from_flattened_numpy(zp[: -shape[0]], shape)
                .to(data.device)
                .type(torch.float32)
            )
            delta_logp = (
                from_flattened_numpy(zp[-shape[0] :], (shape[0],))
                .to(data.device)
                .type(torch.float32)
            )
            prior_logp = sde.prior_logp(z)
            # Return
            N = np.prod(shape[-sde.state_ndims :])
            # No - sign compared to Song et al. Actually compute the likelihood (not negative likelihood)
            nats = prior_logp + delta_logp
            nats_per_dim = nats / N
            # Convert from SDE marginal to data space log-likelihood:
            if data_range_for_correction == DataRangeType.UNIT_BOUNDED:
                # The elementwise jacobian of the scaler is 2:
                # Note, this assumes we are interested in log-likelihood where the pixel values are in the range [0, 1)!!!
                # This is different to those reported by Song et al. (2020) where the pixel values are [0, 256). To convert, need to subtract ln(256) per dimension
                nats_per_dim += np.log(2.0)
            elif data_range_for_correction is None:
                pass
            else:
                raise NotImplementedError(
                    f"Data range for correction {data_range_for_correction} not implemented."
                )

            if unit == "NPD":
                return nats_per_dim, z, nfe
            elif unit == "BPD":
                bpd = nats_per_dim / np.log(2)
                return bpd, z, nfe
            else:
                raise NotImplementedError
            # elif unit == "BPD":
            #     bpd = nats / np.log(2)
            #     N = np.prod(shape[1:])
            #     bpd = bpd / N
            #     # A hack to convert log-likelihoods to bits/dim
            #     # The 7 comes from log2(2) - log2(256). inverse_scaler(-1) should be 0. I don't know wtf it's meant to represent
            #     offset = 7.0 - inverse_scaler(-1.0)
            #     bpd = bpd + offset
            #     return bpd, z, nfe

    return likelihood_fn
