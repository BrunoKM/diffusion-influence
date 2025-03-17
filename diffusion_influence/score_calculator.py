from operator import itemgetter
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.utils import data
from tqdm import tqdm

from diffusion_influence.influence import DiffusionTask
from diffusion_influence.iter_utils import batched, func_on_enum, reiterable_map
from diffusion_influence.linear_operators import (
    Identity,
    ParameterPreconditioningLinearOperator,
)


class GradientIterable(Iterable[tuple[int, Sequence[Tensor]]]):
    def __init__(
        self,
        model: nn.Module,
        params: Sequence[nn.Parameter],
        func: Callable[[tuple[Tensor] | dict[str, Tensor], nn.Module], Tensor],
        preconditioner: ParameterPreconditioningLinearOperator,
        num_batch_aggregations: int,
        dataset: data.Dataset,
        set_model_to_train: bool,
        dataloader_kwargs: dict[str, Any] | None,
    ):
        self.model = model
        self.params = params
        self.func = func
        self.num_batch_aggregations = num_batch_aggregations
        self.preconditioner = preconditioner
        self.dataset = dataset
        self.dataloader_kwargs = dataloader_kwargs
        self.set_model_to_train = set_model_to_train

    def __iter__(
        self,
    ) -> Iterator[tuple[int, Sequence[Tensor]]]:
        # Setup train and query dataloaders
        dataloader_params = {
            "shuffle": False,
            "drop_last": False,
        } | (self.dataloader_kwargs if self.dataloader_kwargs is not None else dict())
        # Only one example per batch (allow for expanding in the loss/measurement)
        dataloader = data.DataLoader(
            dataset=self.dataset, batch_size=1, **dataloader_params
        )

        # Compute the preconditioned query gradients and cache them
        for idx, train_input in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.set_model_to_train:
                self.model.train()
            else:
                self.model.eval()

            gradient = self.batch_aggregated_gradient(
                train_input,
                func=self.func,
                num_batch_aggregations=self.num_batch_aggregations,
            )

            maybe_preconditioned_gradient: list[Tensor] = self.preconditioner.matvec(
                gradient
            )  #  type: ignore

            # Return the (possibly preconditioned) train gradient
            yield idx, maybe_preconditioned_gradient

    def batch_aggregated_gradient(
        self,
        inp,
        func: Callable[[tuple[Tensor] | dict[str, Tensor], nn.Module], Tensor],
        num_batch_aggregations: int,
    ) -> list[torch.Tensor]:
        gradient: list[torch.Tensor] = [torch.zeros_like(p) for p in self.params]
        for _ in range(num_batch_aggregations):
            # Compute the gradient of the measurement w.r.t. model parameters
            loss = func(inp, self.model)
            for i, grad in enumerate(torch.autograd.grad(loss, self.params)):
                gradient[i] += grad

        gradient = [param_grad / num_batch_aggregations for param_grad in gradient]

        return gradient


class ScoreCalculator:
    def __init__(
        self,
        model: nn.Module,
        task: DiffusionTask,
        preconditioner: ParameterPreconditioningLinearOperator,
        num_loss_batch_aggregations: int = 1,
        num_measurement_batch_aggregations: int = 1,
        precondition_query_gradients: bool = True,
    ):
        """
        Args:
            model: The model for which to compute the influence scores.
            task: The task for which to compute the influence scores (defines measurement).
            preconditioner: The preconditioner — a linear operator that approximates the
                the inverse Hessian of the loss w.r.t. the model parameters.
            num_loss_batch_aggregations: The number of batches to aggregate the gradients
                of the loss w.r.t. the model parameters. (equivalent to increasing
                the number of samples for computing the loss function)
            num_measurement_batch_aggregations: The number of batches to aggregate the
                gradients of the measurement w.r.t. the model parameters. (equivalent to
                increasing the number of samples for computing the measurement function)
            precondition_query_gradients: Whether to precondition the query gradients
                or the train gradients. If True, the query gradients will be preconditioned,
                if False, the train gradients will be preconditioned. This is useful
                if compressing either the train or query gradients, and deciding which
                is more efficient, and which is more compressible.
        """

        self.model = model
        self.task = task
        self.preconditioner = preconditioner
        self.num_loss_batch_aggregations = num_loss_batch_aggregations
        self.num_measurement_batch_aggregations = num_measurement_batch_aggregations
        self.precondition_query_gradients = precondition_query_gradients

        # Set the device and move the model, params, and preconditioner to the device
        self.device = next(model.parameters()).device

        # Parameters to compute the influence scores for; will be determined by the
        # preconditioner. This is necessary since the preconditioner might use
        # only a subset of the model's parameters (e.g. only Linear and Conv2d layers)
        self.params: Sequence[nn.Parameter] = preconditioner.params
        # Make sure that the preconditioner params are the same pointers as the model
        # parameters:
        assert all(
            any(p is param for param in model.parameters()) for p in self.params
        ), "Preconditioner parameters must be the same pointers as the model parameters"

        self.num_params = sum(p.numel() for p in self.params)

    def batch_aggregated_gradient(
        self,
        inp,
        func: Callable[[tuple[Tensor] | dict[str, Tensor], nn.Module], Tensor],
        num_batch_aggregations: int,
    ):
        gradient: list[torch.Tensor] = [torch.zeros_like(p) for p in self.params]
        for _ in range(num_batch_aggregations):
            # Compute the gradient of the measurement w.r.t. model parameters
            loss = func(inp, self.model)
            for i, grad in enumerate(torch.autograd.grad(loss, self.params)):
                gradient[i] += grad

        gradient = [param_grad / num_batch_aggregations for param_grad in gradient]
        return gradient

    def get_train_gradients_iterable(
        self,
        train_dataset: data.Dataset,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> Iterable[tuple[int, Sequence[Tensor]]]:
        return GradientIterable(
            model=self.model,
            params=self.params,
            func=self.task.compute_train_loss,
            preconditioner=Identity(params=self.params)
            if self.precondition_query_gradients
            else self.preconditioner,
            num_batch_aggregations=self.num_loss_batch_aggregations,
            dataset=train_dataset,
            set_model_to_train=True,
            dataloader_kwargs=dataloader_kwargs,
        )

    def get_query_gradients_iterable(
        self,
        query_dataset: data.Dataset,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> Iterable[tuple[int, Sequence[Tensor]]]:
        return GradientIterable(
            model=self.model,
            params=self.params,
            func=self.task.compute_measurement,
            preconditioner=self.preconditioner
            if self.precondition_query_gradients
            else Identity(params=self.params),
            num_batch_aggregations=self.num_measurement_batch_aggregations,
            dataset=query_dataset,
            set_model_to_train=False,
            dataloader_kwargs=dataloader_kwargs,
        )

    def compute_pairwise_scores_from_gradients(
        self,
        query_gradients: Iterable[tuple[int, Sequence[Tensor]]],
        train_gradients: Iterable[tuple[int, Sequence[Tensor]]],
        num_query_examples: int,
        num_train_examples: int,
        query_batch_size: int,
        train_batch_size: int,
        outer_is_query: bool,
    ) -> Tensor:
        """
        Allow for specifying whether the outer loop is over the query or train examples.

        If the outer loop is over N_o examples, and the inner loop is over N_i examples,
        and the outer batch-size is B_o, then the number of “recomputations” (i.e.
        the number of times `next()` is called on `train_gradients`/`query_gradients`)
        is:
            For outer gradients: N_o
            For inner gradients: N_o / B_o * N_i

        Hence, generally we want N_o < N_i, which implies that the outer loop should be
        over the query examples. However, if query_gradients or train_gradients are
        easier to recompute (e.g. due to caching), then it might be more efficient to
        move the more efficient computation to the inner loop.

        Args:
            query_gradients: An iterable of tuples, where each tuple contains the index
                of the query example and the query gradient.
            train_gradients: An iterable of tuples, where each tuple contains the index
                of the train example and the train gradient.
            num_query_examples: The total number of query examples.
            num_train_examples: The total number of train examples.
            query_batch_size, train_batch_size: The batch sizes for the query/train
                gradients. This will allow for slightly more efficient computation of
                the scores by batching the inner product, at the cost of more memory.
                In general, the gain in efficiency is pretty small, so this should be
                relatively small.
            outer_is_query: Whether the outer loop is over the query examples (explained
                above in more detail).
        """
        if outer_is_query:
            outer_gradients = query_gradients
            outer_batch_size = query_batch_size
            outer_num_examples = num_query_examples
            inner_gradients = train_gradients
            inner_batch_size = train_batch_size
            inner_num_examples = num_train_examples
        else:
            outer_gradients = train_gradients
            outer_batch_size = train_batch_size
            outer_num_examples = num_train_examples
            inner_gradients = query_gradients
            inner_batch_size = query_batch_size
            inner_num_examples = num_query_examples

        # Flatten the outer and inner gradients:
        outer_gradients = reiterable_map(
            func_on_enum(lambda gs: torch.cat([g.flatten() for g in gs])),
            outer_gradients,
        )
        inner_gradients = reiterable_map(
            func_on_enum(lambda gs: torch.cat([g.flatten() for g in gs])),
            inner_gradients,
        )

        # Initialize the scores tensor
        scores: Tensor = torch.zeros(
            outer_num_examples, inner_num_examples, device=self.device
        )

        # The query/train gradients to do not have to be in order, so keep track of
        # which ones were seen
        seen_outer_examples: set[int] = set()

        # --- Iterate over the outer gradients ---
        for outer_idxs_and_grads_batch in batched(outer_gradients, outer_batch_size):
            outer_idxs = list(map(itemgetter(0), outer_idxs_and_grads_batch))

            assert seen_outer_examples.isdisjoint(outer_idxs), (
                "outer examples must be unique"
            )
            assert len(set(outer_idxs)) == len(outer_idxs), (
                "outer examples must be unique"
            )

            seen_outer_examples.update(outer_idxs)
            # Concatenate the outer gradients:
            outer_gradients_batch = torch.stack(
                tuple(map(itemgetter(1), outer_idxs_and_grads_batch)),
                dim=0,
            )  # [num_outer_batch, num_params]

            # --- Iterate over the inner gradients ---
            seen_inner_examples: set[int] = set()
            for inner_idxs_and_grads_batch in batched(
                inner_gradients, inner_batch_size
            ):
                inner_idxs = list(map(itemgetter(0), inner_idxs_and_grads_batch))

                assert seen_inner_examples.isdisjoint(inner_idxs), (
                    "inner examples must be unique"
                )
                assert len(set(inner_idxs)) == len(inner_idxs), (
                    "inner examples must be unique"
                )

                seen_inner_examples.update(inner_idxs)
                # Concatenate the inner gradients:
                inner_gradients_batch = torch.stack(
                    tuple(map(itemgetter(1), inner_idxs_and_grads_batch)),
                    dim=0,
                )  # [num_inner_batch, num_params]

                # Compute the (batched) inner product: the below is equivalent to setting
                # `scores[outer_idxs[i], inner_idxs[j]] = outer_gradients_batch[i].dot(inner_gradients_batch[j])`
                # for every `i` and `j`
                idx_meshgrid_outer, idx_meshgrid_inner = torch.meshgrid(
                    torch.tensor(outer_idxs), torch.tensor(inner_idxs)
                )
                scores[idx_meshgrid_outer, idx_meshgrid_inner] = (
                    outer_gradients_batch @ inner_gradients_batch.T
                )

            assert len(seen_inner_examples) == inner_num_examples, (
                "inner examples must be unique"
            )
        assert len(seen_outer_examples) == outer_num_examples, (
            "outer examples must be unique"
        )

        # Make sure to return scores of shape [num_query_examples, num_train_examples]
        if outer_is_query:
            return scores
        else:
            return scores.T
