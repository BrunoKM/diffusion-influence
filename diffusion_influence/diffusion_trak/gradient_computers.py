import logging
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from trak.gradient_computers import AbstractGradientComputer

from diffusion_influence.diffusion_trak.modelout_functions import (
    UnconditionalDiffusionModelOutput,
)
from diffusion_influence.diffusion_trak.utils import (
    _accumulate_vectorize,
    get_num_params,
)


class DiffusionGradientComputer(AbstractGradientComputer):
    modelout_fn: UnconditionalDiffusionModelOutput

    def __init__(
        self,
        model: torch.nn.Module,
        task: UnconditionalDiffusionModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        train_transforms: Callable[[Tensor], Tensor],
        grad_wrt: Optional[Iterable[str]] = None,
        num_samples_for_scoring: int = 1,
        num_samples_for_featurizing: int = 1,
    ) -> None:
        super().__init__(model, task, grad_dim, dtype, device)
        if grad_wrt is not None:
            raise NotImplementedError(
                "grad_wrt is not implemented for DiffusionGradientComputer"
            )
        self.model = model
        self.num_params = get_num_params(self.model)
        self.load_model_params(model)
        self._are_we_featurizing = False
        self.num_samples_for_scoring = num_samples_for_scoring
        self.num_samples_for_featurizing = num_samples_for_featurizing
        self.train_transforms = train_transforms

    def load_model_params(self, model) -> None:
        """Given a a torch.nn.Module model, inits/updates the (functional)
        weights and buffers. See https://pytorch.org/docs/stable/func.html
        for more details on :code:`torch.func`'s functional models.

        Args:
            model (torch.nn.Module):
                model to load

        """
        self.func_weights = dict(model.named_parameters())
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(
        self,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """Uses functorch's :code:`vmap` (see
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html#functorch.vmap
        for more details) to vectorize the computations of per-sample gradients.
        """
        if self._are_we_featurizing:
            return self._compute_per_sample_grad_featurizing(batch)
        else:
            return self._compute_per_sample_grad_scoring(batch)

    def _compute_per_sample_grad_featurizing(
        self,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """
        Args:
            batch (Iterable[Tensor]):
                contains [data, labels, timesteps]
                each of those arrays should have the same first dimension (=batch size)

        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.

        """
        images = batch["input"]
        # Taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        grads_loss = torch.func.grad(
            self.modelout_fn.get_output_featurizing, has_aux=False, argnums=1
        )

        batch_size = images.shape[0]
        grads = torch.zeros(
            size=(batch_size, self.num_params),
            dtype=images.dtype,
            device=images.device,
        )

        for _ in range(self.num_samples_for_featurizing):
            # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
            augmented_images = torch.stack(
                [self.train_transforms(image) for image in images],
                dim=0,
            )
            _accumulate_vectorize(
                g=torch.func.vmap(
                    grads_loss,
                    in_dims=(None, None, None, 0),
                    randomness="different",
                )(
                    self.model,
                    self.func_weights,
                    self.func_buffers,
                    augmented_images,
                ),
                arr=grads,
            )
        if torch.isnan(grads).any():
            logging.warning("NaNs in gradients")
            raise ValueError("NaNs in gradients")
        return grads

    def _compute_per_sample_grad_scoring(
        self,
        batch: dict[str, Tensor],
    ) -> Tensor:
        """
        Args:
            batch (Iterable[Tensor]):
            [images, labels, timesteps]
            each of the three should have the same first dimension (batch size)
            for CIFAR:
            images.shape = [batch size, 1000, 3, 32, 32]
            labels.shape = [batch size, 1]
            timesteps.shape = [batch size, num_timesteps, 1] or [batch size, num_timesteps, 2]
            2 should be passed if using trajectory noise (rather than fresh
            noise); in that case, it will use the first one to index into the
            x_0_hats hat and the second one to index into x_ts and  pass to the
            U-net.


        Returns:
            Tensor:
                gradients of the model output function of each sample in the
                batch with respect to the model's parameters.

        """
        # taking the gradient wrt weights (second argument of get_output, hence argnums=1)
        measurement_grad_fn = torch.func.grad(
            self.modelout_fn.get_output_scoring, has_aux=False, argnums=1
        )

        images = batch["input"]

        batch_size = images.shape[0]
        grads = torch.zeros(
            size=(batch_size, self.num_params), dtype=images[0].dtype, device="cuda"
        )

        for _ in range(self.num_samples_for_scoring):
            # map over batch dimensions (hence 0 for each batch dimension, and None for model params)
            _accumulate_vectorize(
                g=torch.func.vmap(
                    measurement_grad_fn,
                    in_dims=(None, None, None, 0),
                    randomness="different",
                )(
                    self.model,
                    self.func_weights,
                    self.func_buffers,
                    images,
                ),
                arr=grads,
            )
        if torch.isnan(grads).any():
            logging.warning("NaNs in gradients")
            raise ValueError("NaNs in gradients")
        return grads

    def compute_loss_grad(self, batch: dict[str, Tensor]) -> Tensor:
        """Computes the gradient of the loss with respect to the model output

        .. math::

            \\partial \\ell / \\partial \\text{(model output)}

        Note: For all applications we considered, we analytically derived the
        out-to-loss gradient, thus avoiding the need to do any backward passes
        (let alone per-sample grads). If for your application this is not feasible,
        you'll need to subclass this and modify this method to have a structure
        similar to the one of :meth:`FunctionalGradientComputer:.get_output`,
        i.e. something like:

        .. code-block:: python

            grad_out_to_loss = grad(self.model_out_to_loss_grad, ...)
            grads = vmap(grad_out_to_loss, ...)
            ...

        Args:
            batch (Iterable[Tensor]):
                batch of data

        """
        return self.modelout_fn.get_out_to_loss_grad(
            self.model, self.func_weights, self.func_buffers, batch
        )
