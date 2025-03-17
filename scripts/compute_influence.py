import functools
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence, Type

import diffusers
import hydra
import numpy as np
import omegaconf
import torch
import yaml
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from torch import Tensor, cuda, device, manual_seed, nn
from torchvision.transforms import Compose
from tqdm import tqdm

from diffusion_influence.compressor import WeightCompressor
from diffusion_influence.config_schemas import (
    InfluenceConfig,
    MeasurementType,
    PreconditionerType,
)
from diffusion_influence.constructors import (
    construct_compressor,
    construct_datasets_with_transforms,
    construct_inverse_kfac_preconditioner,
    construct_model_and_vae,
    construct_model_training_config,
    construct_scheduler_from_config,
    get_train_only_transforms_for_dataset,
    get_transforms_for_dataset,
)
from diffusion_influence.data_utils import (
    load_samples_and_trajectories_dataset_from_dir,
    load_samples_dataset_from_dir,
)
from diffusion_influence.influence import (
    DiffusionELBOTask,
    DiffusionLossAtTimeTask,
    DiffusionLossTask,
    DiffusionSamplingTrajectoryLogprobTask,
    DiffusionSimplifiedELBOTask,
    DiffusionSquareNormLossTask,
    DiffusionTask,
    DiffusionWeightedLossTask,
    LatentDiffusionLossTask,
)
from diffusion_influence.iter_utils import func_on_enum, reiterable_map
from diffusion_influence.linear_operators import Identity
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers
from diffusion_influence.score_calculator import ScoreCalculator
from diffusion_influence.torch_fileloader import TorchFileLoader

# Check diffusers version sufficiently high
diffusers.utils.check_min_version("0.16.0")


DEVICE = device("cuda" if cuda.is_available() else "cpu")


# Register custom resolvers for the config parser:
register_custom_resolvers()
# Register the structured dataclass config schema with Hydra
# (for type-checking, autocomplete, and validation)
cs = ConfigStore.instance()
cs.store(name="base_schema", node=InfluenceConfig)


@hydra.main(version_base=None, config_path="configs")
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: InfluenceConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


def main(config: InfluenceConfig):
    # Set seed
    manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load the training config.
    model_training_config = construct_model_training_config(
        config.pretrained_model_config_path
    )
    data_config = model_training_config.data

    # Prepare the trained model.
    model, vae = construct_model_and_vae(
        config.pretrained_model_dir_path, model_training_config, DEVICE
    )

    # Initialize the scheduler.
    noise_scheduler = construct_scheduler_from_config(
        scheduler_name_or_path=model_training_config.scheduler_name_or_path,
        diffusion_config=model_training_config.diffusion,
    )
    # Prepare the dataset.
    train_dataset, eval_datasets = construct_datasets_with_transforms(data_config)

    # Define task
    # Get the train transforms for computing the per-example loss. These will
    # be applied AFTER the base transforms (e.g. normalization), which
    # is different from the training pipeline, but for datasets of interest
    # (e.g. CIFAR-10 with random horizontal flip) this shouldn't change the result

    train_transforms = get_train_only_transforms_for_dataset(data_config.dataset_name)

    task_class: Type[DiffusionTask]
    if vae is None:
        match config.measurement:
            case MeasurementType.ELBO:
                task_class = DiffusionELBOTask
            case MeasurementType.SIMPLIFIED_ELBO:
                task_class = DiffusionSimplifiedELBOTask
            case MeasurementType.SQUARE_NORM:
                task_class = DiffusionSquareNormLossTask
            case MeasurementType.LOSS:
                task_class = DiffusionLossTask
            case MeasurementType.LOSS_AT_TIME:
                task_class = functools.partial(
                    DiffusionLossAtTimeTask, **config.measurement_kwargs
                )  #  type: ignore
            case MeasurementType.WEIGHTED_LOSS:
                task_class = functools.partial(
                    DiffusionWeightedLossTask, **config.measurement_kwargs
                )  #  type: ignore
            case MeasurementType.SAMPLING_TRAJECTORY_LOGPROB:
                task_class = DiffusionSamplingTrajectoryLogprobTask
            case _:
                raise ValueError(f"Unsupported measurement type: {config.measurement}")
    else:
        match config.measurement:
            case MeasurementType.LOSS:
                task_class = functools.partial(
                    LatentDiffusionLossTask,
                    vae=vae,
                )  #  type: ignore

            case _:
                raise ValueError(
                    f"Unsupported measurement type for LDM: {config.measurement}"
                )

    task = task_class(
        noise_scheduler=noise_scheduler,
        num_augment_samples_loss=config.num_samples_for_loss_scoring,
        num_augment_samples_measurement=config.num_samples_for_measurement_scoring,
        batch_image_idx="input",
        train_augmentations=Compose(train_transforms),
        device=DEVICE,
    )

    #################################### (E)K-FAC ######################################

    # Only consider layer types supported by the KFAC implementation.
    params = [
        p
        for mod in model.modules()
        for p in mod.parameters()
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and p.requires_grad
    ]
    # Construct the preconditioning matrix.
    match config.preconditioner:
        case PreconditionerType.identity:
            preconditioner = Identity(params=params)
        case PreconditionerType.kfac:
            preconditioner = construct_inverse_kfac_preconditioner(
                kfac_config=config.kfac,
                dataloader_config=config.kfac_dataloader,
                model=model,
                params=params,
                vae=vae,
                noise_scheduler=noise_scheduler,
                train_dataset=train_dataset,
                cache_inverse_kfac=config.cache_inverse_kfac,
                cached_inverse_kfac_path=config.cached_inverse_kfac_path,
                device=DEVICE,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported preconditioner: {config.preconditioner}"
            )

    ############################## Influence Computation ##############################

    # Move the noise_scheduler to device
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(DEVICE)

    # Construct query and train datasets
    _, eval_transforms = get_transforms_for_dataset(
        model_training_config.data.dataset_name,
    )

    if config.measurement in {MeasurementType.SAMPLING_TRAJECTORY_LOGPROB}:
        query_dataset = load_samples_and_trajectories_dataset_from_dir(
            config.samples_dir_path, Compose(eval_transforms)
        )
    else:
        query_dataset = load_samples_dataset_from_dir(
            config.samples_dir_path, Compose(eval_transforms)
        )
    train_dataset = eval_datasets["train_no_augment"]

    train_gradient_compressor = construct_compressor(
        config.train_gradient_compressor,
        compressor_kwargs=config.train_compressor_kwargs,
    )
    query_gradient_compressor = construct_compressor(
        config.query_gradient_compressor,
        compressor_kwargs=config.query_compressor_kwargs,
    )

    # --- Calculate the influence scores ----
    score_calculator = ScoreCalculator(
        model,
        task,
        preconditioner=preconditioner,
        num_loss_batch_aggregations=config.num_loss_batch_aggregations,
        num_measurement_batch_aggregations=config.num_measurement_batch_aggregations,
    )
    query_gradient_iterable = get_and_possibly_cache_query_gradient_iterable(
        config.cache_query_gradients,
        query_dataset,
        query_gradient_compressor,
        score_calculator,
    )

    # --- Either compute the train gradients or load them from disk ---
    train_gradient_iterable = get_and_possibly_cache_train_gradient_iterable(
        config, train_dataset, train_gradient_compressor, score_calculator
    )

    # Compute the scores
    scores = score_calculator.compute_pairwise_scores_from_gradients(
        query_gradients=query_gradient_iterable,
        train_gradients=train_gradient_iterable,  # type: ignore
        num_query_examples=len(query_dataset),
        num_train_examples=len(train_dataset),
        query_batch_size=config.query_batch_size,
        train_batch_size=config.train_batch_size,
        outer_is_query=config.outer_is_query_in_score_loop,
    )

    logging.info(f"Scores shape: {scores.shape}")
    # --- Save the scores to a file
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    scores_path = output_dir / "influence_scores.npy"
    np.save(file=scores_path, arr=scores.cpu().numpy())


def get_and_possibly_cache_train_gradient_iterable(
    config: InfluenceConfig,
    train_dataset,
    train_gradient_compressor: WeightCompressor,
    score_calculator: ScoreCalculator,
) -> Iterable[tuple[int, Sequence[Tensor]]]:
    # Iterable to compute the the train gradients directly
    train_gradient_iterable = score_calculator.get_train_gradients_iterable(
        train_dataset=train_dataset,
    )

    if config.cached_train_gradients_path is not None:
        # Compress the train gradients
        train_gradient_iterable = map(
            func_on_enum(train_gradient_compressor.compress), train_gradient_iterable
        )

        train_grad_filenames = [f"train_grad_{i}.pt" for i in range(len(train_dataset))]
        if config.cache_train_gradients:
            # --- Compute the train gradients and cache them to disk ---
            config.cached_train_gradients_path.mkdir(parents=True, exist_ok=True)
            # Save a file that describes which gradient files to expect:
            with open(
                config.cached_train_gradients_path / "gradient_files.txt", "w"
            ) as f:
                f.write("\n".join(train_grad_filenames))
            # Also save a note about 1) whether these gradients are preconditioned and 2) the compressor used
            # to a yaml file:
            with open(config.cached_train_gradients_path / "metadata.yaml", "w") as f:
                metadata = {
                    "preconditioned": not config.precondition_query_gradients,
                    "compressor": str(config.train_gradient_compressor),
                    "compressor_kwargs": config.train_compressor_kwargs,
                }
                yaml.dump(metadata, f)

            # Save the individual compressed train gradients:
            for i, compressed_train_grad in tqdm(train_gradient_iterable):
                # Save the preconditioned train gradients
                torch.save(
                    compressed_train_grad,
                    config.cached_train_gradients_path / train_grad_filenames[i],
                )

        # --- Load the cached gradients from disk ---
        # Validate the metadata:
        metadata = yaml.load(
            (config.cached_train_gradients_path / "metadata.yaml").open("r"),
            Loader=yaml.FullLoader,
        )
        for key, value in metadata.items():
            match key:
                case "preconditioned":
                    assert value == (not config.precondition_query_gradients), (
                        "Expected the same preconditioning setting"
                    )
                case "compressor":
                    assert value == str(config.train_gradient_compressor), (
                        "Expected the same compressor"
                    )

        gradient_file_list_file = (
            config.cached_train_gradients_path / "gradient_files.txt"
        )
        compressed_gradient_filenames = list(
            gradient_file_list_file.read_text().split("\n")
        )
        compressed_gradient_files = [
            config.cached_train_gradients_path / filename
            for filename in compressed_gradient_filenames
        ]
        train_gradient_iterable: TorchFileLoader[Any, Sequence[Tensor]] = (
            TorchFileLoader(
                files=compressed_gradient_files,
                num_workers=config.cached_train_gradient_queue_num_workers,
                max_queue_size=config.cached_train_gradient_queue_size,
                data_map=train_gradient_compressor.decompress,
                device=DEVICE,
            )
        )

    return train_gradient_iterable


def get_and_possibly_cache_query_gradient_iterable(
    cache_query_gradients: bool,
    query_dataset,
    query_gradient_compressor: WeightCompressor,
    score_calculator: ScoreCalculator,
) -> Iterable[tuple[int, Sequence[Tensor]]]:
    query_gradient_iterable = score_calculator.get_query_gradients_iterable(
        query_dataset=query_dataset,
    )
    # If `cache_query_gradients` is enabled, precompute and store the query gradients on GPU.
    if cache_query_gradients:
        # Compress query gradients
        query_gradient_iterable = map(
            func_on_enum(query_gradient_compressor.compress), query_gradient_iterable
        )
        # Manifest/cache the iterable:
        query_gradient_iterable = list(query_gradient_iterable)
        # Note: ideally we would allow for caching in batches, but figuring out the
        # interface for that is tricky, so it's future work for now.

        # Uncompress the query gradients
        query_gradient_iterable = reiterable_map(
            func_on_enum(query_gradient_compressor.decompress), query_gradient_iterable
        )
    return query_gradient_iterable


if __name__ == "__main__":
    hydra_main()
