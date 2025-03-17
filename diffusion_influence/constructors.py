import functools
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import omegaconf
import torch
from curvlinops import (
    EKFACLinearOperator,
    KFACInverseLinearOperator,
    KFACLinearOperator,
)
from datasets import Dataset, load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from torch import (
    equal,
    load,
    nn,
    ones,
    save,
)
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from diffusion_influence.config_schemas import (
    COMPRESSOR_CLASS_MAPPING,
    CompressorType,
    DataConfig,
    DataLoaderConfig,
    DataRangeType,
    DatasetType,
    DiffusionConfig,
    DiffusionTrainConfig,
    KFACConfig,
    OptimiserType,
)
from diffusion_influence.data import artbench, cifar
from diffusion_influence.data.cifar import (
    TransformType,
)
from diffusion_influence.diffusion.latent_diffusion_utils import (
    vae_encode_batch,
)
from diffusion_influence.influence import (
    KFACCompatibleImageRegressionModelWrapper,
    data_batch_to_diffusion_regression_batch,
)
from diffusion_influence.iter_utils import (
    ChainedIterable,
    SizedIterable,
    reiterable_map,
)
from diffusion_influence.linear_operators import KFACInverseLinearOperatorWrapper
from diffusion_influence.optim_utils import get_warmup_scheduler


def construct_model_training_config(
    pretrained_model_config_path: Path,
) -> DiffusionTrainConfig:
    # --- Load in the data config from the path for the pretrained model config.
    model_training_config_omegadict = omegaconf.OmegaConf.load(
        pretrained_model_config_path
    )
    # Register the structured model default
    model_training_config_struct = omegaconf.OmegaConf.structured(DiffusionTrainConfig)
    # Merge the loaded config into the structured schema (to ensure type conversion)
    model_training_config_omegadict_merged = omegaconf.OmegaConf.merge(
        model_training_config_struct, model_training_config_omegadict
    )
    model_training_config: DiffusionTrainConfig = omegaconf.OmegaConf.to_object(
        model_training_config_omegadict_merged
    )  # type: ignore  # insufficient typing of to_object()
    return model_training_config


def construct_datasets_with_transforms(config: DataConfig):
    if config.data_range != DataRangeType.UNIT_BOUNDED:
        raise NotImplementedError(
            "Only unit-bounded data normalisation is supported in HuggingFace's diffusers"
        )
    # --- Load the dataset ---
    train_dataset, eval_datasets = construct_and_subsample_datasets(
        dataset_name=config.dataset_name,
        cache_dir=config.cache_dir,
        examples_idxs_path=config.examples_idxs_path,
    )
    # --- Apply transforms ---
    train_transforms, eval_transforms = get_transforms_for_dataset(
        dataset_name=config.dataset_name,
    )

    def transform_images(examples, transforms):
        # images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        images = [transforms(image.convert("RGB")) for image in examples["img"]]
        return {kw: val for kw, val in examples.items() if kw != "img"} | {
            "input": images
        }

    # Set transforms for the dataset
    train_dataset.set_transform(
        functools.partial(transform_images, transforms=Compose(train_transforms))
    )
    for eval_dataset in eval_datasets.values():
        eval_dataset.set_transform(
            functools.partial(transform_images, transforms=Compose(eval_transforms))
        )

    return train_dataset, eval_datasets


def add_weights_to_dataset(dataset: Dataset, example_weights: np.ndarray):
    dataset = dataset.map(
        lambda example, idx: example | {"example_weight": float(example_weights[idx])},
        with_indices=True,
    )
    return dataset


def weight_examples_in_dataset(
    dataset: Dataset,
    example_idxs: np.ndarray,
    # examples_idxs_path: Optional[str | Path],
    weighted_examples_idxs: np.ndarray,
    invert_weighted_examples_idxs: bool,
    weighted_examples_weight: float,
):
    assert len(example_idxs) == len(dataset), (
        "The passed dataset should have been subsampled using the same idxs, but have"
        f"len(example_idxs)={len(example_idxs)} and len(dataset)={len(dataset)}"
    )
    weighted_examples_idxs_set = set(weighted_examples_idxs)
    assert weighted_examples_idxs_set.issubset(set(example_idxs)), (
        "The weighted examples should be a subset of the examples."
    )
    # Assert no duplicates:
    assert len(weighted_examples_idxs_set) == len(weighted_examples_idxs), "Duplicates"
    # Retrieve the indices into the subsampled dataset that the indices
    # `weighted_examples_idxs` (into the non-subsampled dataset) correspond to.
    should_example_be_weighted = np.isin(example_idxs, weighted_examples_idxs)
    assert should_example_be_weighted.sum() == len(weighted_examples_idxs), (
        "Something has gone wrong."
    )
    # Possibly invert the weighted examples
    if invert_weighted_examples_idxs:
        should_example_be_weighted = np.logical_not(should_example_be_weighted)
    example_weights = np.where(
        should_example_be_weighted,
        torch.tensor(weighted_examples_weight),
        torch.tensor(1.0),
    )
    return add_weights_to_dataset(dataset, example_weights)


def construct_and_subsample_datasets(
    dataset_name: DatasetType,
    cache_dir: str,
    examples_idxs_path: Optional[str | Path],
) -> tuple[Dataset, dict[str, Dataset]]:
    # --- Load the dataset ---
    train_dataset, eval_datasets = construct_datasets(
        dataset_name=dataset_name, cache_dir=cache_dir
    )
    # --- Subselect examples if requested ---
    if examples_idxs_path is not None:
        idxs = np.loadtxt(examples_idxs_path, dtype=int)
        if idxs.max() > len(train_dataset):
            raise ValueError(
                f"Requested number of training examples "
                f"exceeds the number of examples in the dataset ({len(train_dataset)})"
            )
        train_dataset = train_dataset.select(idxs)
        eval_datasets = {
            dataset_name: eval_dataset.select(idxs)
            if "train" in dataset_name
            else eval_dataset
            for dataset_name, eval_dataset in eval_datasets.items()
        }
    return train_dataset, eval_datasets


def construct_datasets(
    dataset_name: DatasetType,
    cache_dir: str,
) -> tuple[Dataset, dict[str, Dataset]]:
    train_dataset: Dataset
    eval_datasets: dict[str, Dataset]
    match dataset_name:
        case DatasetType.artbench:
            artbench.validate_integrity(cache_dir)
            logging.info("Integrity of ArtBench verified.")
            artbench_dir = Path(cache_dir) / artbench.DATA_DIR_NAME
            datasets = load_dataset(
                "imagefolder",
                data_dir=str(artbench_dir),
            )
            # Map the column names from "image" to "img" to be consistent with CIFAR
            datasets = datasets.rename_column("image", "img")
            train_dataset = datasets["train"]  #  type: ignore
            eval_datasets = {
                "test": datasets["test"],  #  type: ignore
                # This should copy the dataset:
                "train_no_augment": train_dataset.map(lambda x: x),
            }
        case DatasetType.coco:
            raise NotImplementedError
        case DatasetType.cifar10 | DatasetType.cifar10deq:
            datasets = load_dataset(
                "cifar10",
                cache_dir=cache_dir,
            )
            train_dataset = datasets["train"]  #  type: ignore
            eval_datasets = {
                "test": datasets["test"],  #  type: ignore
                # This should copy the dataset:
                "train_no_augment": train_dataset.map(lambda x: x),
            }
        case DatasetType.cifar2 | DatasetType.cifar2deq:
            datasets = load_dataset(
                "cifar10",
                cache_dir=cache_dir,
            )
            # Filter out all classes except horses and automobiles (labels 7 and 1)
            HORSE_LABEL = 7
            AUTOMOBILE_LABEL = 1
            train_labels = np.array(datasets["train"]["label"])  #  type: ignore
            train_idxs = np.argwhere(
                np.logical_or(
                    train_labels == AUTOMOBILE_LABEL, train_labels == HORSE_LABEL
                )
            )
            test_labels = np.array(datasets["test"]["label"])  #  type: ignore
            test_idxs = np.argwhere(
                np.logical_or(
                    test_labels == AUTOMOBILE_LABEL, test_labels == HORSE_LABEL
                )
            )
            train_dataset = datasets["train"].select(train_idxs)  #  type: ignore
            eval_datasets = {
                "test": datasets["test"].select(test_idxs),  # type: ignore
                # This should copy the dataset:
                "train_no_augment": train_dataset.map(lambda x: x),
            }
        case DatasetType.dummy1 | DatasetType.dummy2:
            from PIL import Image

            base_example = np.zeros([32, 32, 3], dtype=np.uint8)
            example1 = base_example.copy()
            example1[0, 0, 0] = 255
            example2 = base_example.copy()
            example2[-1, 0, -1] = 255
            # Convert to PIL image
            example1 = Image.fromarray(example1)
            example2 = Image.fromarray(example2)

            # from torch.utils.data import TensorDataset
            # train_dataset = TensorDataset(torch.stack([example1] * 128))
            examples_list = [{"img": example1}] * 128 + (
                [{"img": example2}] * 128
                if dataset_name == DatasetType.dummy2
                else [{"img": example1}] * 128
            )
            from datasets import Features, Image

            # features = Features({"img": Array3D(shape=(3, 32, 32), dtype='int8')})
            features = Features({"img": Image()})
            train_dataset = Dataset.from_list(examples_list, features=features)
            # train_dataset = train_dataset.with_format("torch")
            eval_datasets = {
                "test": train_dataset,
            }
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, eval_datasets


def get_transforms_for_dataset(
    dataset_name: DatasetType,
) -> tuple[tuple[TransformType, ...], tuple[TransformType, ...]]:
    match dataset_name:
        case DatasetType.artbench:
            return artbench.get_transforms()
        case DatasetType.cifar10 | DatasetType.cifar2:
            return cifar.get_cifar10_transforms()
        case DatasetType.cifar2deq | DatasetType.cifar10deq:
            return cifar.get_cifar10deq_transforms()
        case DatasetType.dummy1 | DatasetType.dummy2:
            from torchvision.transforms import Normalize, ToTensor

            return (
                ToTensor(),
                Normalize(0.5, 0.5),
            ), (
                ToTensor(),
                Normalize(0.5, 0.5),
            )
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")


def get_train_only_transforms_for_dataset(
    dataset_name: DatasetType,
) -> tuple[TransformType, ...]:
    """
    Get only the additional transforms that are applied to the training set.

    Note: this doesn't include the “base” transforms (such as, typically, converting to tensor
    and range normalization).
    """
    match dataset_name:
        case DatasetType.artbench:
            return artbench.get_train_only_transforms()
        case (
            DatasetType.cifar10
            | DatasetType.cifar2
            | DatasetType.cifar2deq
            | DatasetType.cifar10deq
        ):
            return cifar.get_cifar10_train_only_transforms()
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")


def construct_dataloader(
    config: DataLoaderConfig, train_dataset: Dataset, eval_datasets: dict[str, Dataset]
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
    )
    eval_loaders = {
        eval_dataset_name: DataLoader(
            eval_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        for eval_dataset_name, eval_dataset in eval_datasets.items()
    }
    return train_loader, eval_loaders


def construct_model_and_vae(
    pretrained_model_dir_path: Path,
    model_training_config: DiffusionTrainConfig,
    device: torch.device,
) -> tuple[UNet2DModel, Optional[AutoencoderKL]]:
    model: UNet2DModel = UNet2DModel.from_pretrained(str(pretrained_model_dir_path))  #  type: ignore
    model.to(device)
    model.eval()

    # Load the VAE encoder (for latent diffusion) if it is specified in the config
    if model_training_config.vae_name_or_path is not None:
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            model_training_config.vae_name_or_path, subfolder="vae"
        )
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device)
    else:
        vae = None

    return model, vae


def construct_optimizer(
    config: DiffusionTrainConfig,
    model: nn.Module,
) -> tuple[Optimizer, Optional[LRScheduler]]:
    match config.optimisation.optimiser_type:
        case OptimiserType.SGD:
            optim_constructor = SGD
        case OptimiserType.ADAM:
            optim_constructor = Adam
        case OptimiserType.ADAMW:
            optim_constructor = AdamW
        case _:
            raise ValueError(
                f"Unknown optimizer type: {config.optimisation.optimiser_type}"
            )

    optim = optim_constructor(
        params=model.parameters(),
        lr=config.optimisation.lr,
        **config.optimisation.optimiser_kwargs,
    )

    # Construct the scheduler
    if config.optimisation.warmup_steps > 0:
        warmup_scheduler = get_warmup_scheduler(
            optimizer=optim,
            warmup_steps=config.optimisation.warmup_steps,
        )
    else:
        warmup_scheduler = None

    if config.optimisation.cosine_lr_schedule:
        scheduler = CosineAnnealingLR(
            optim,
            T_max=config.num_training_iter - config.optimisation.warmup_steps,
        )
        if warmup_scheduler is not None:
            scheduler = SequentialLR(
                optimizer=optim,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[config.optimisation.warmup_steps],
            )
    else:
        scheduler = warmup_scheduler

    return optim, scheduler


def construct_compressor(
    compressor_type: CompressorType, compressor_kwargs: dict[str, Any]
):
    compressor_class = COMPRESSOR_CLASS_MAPPING[compressor_type]
    return compressor_class(**compressor_kwargs)


def construct_scheduler_from_config(
    scheduler_name_or_path: Optional[Path | str], diffusion_config: DiffusionConfig
) -> DDPMScheduler:
    # Initialize the scheduler
    if scheduler_name_or_path is None:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_config.max_timestep,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
            beta_schedule=diffusion_config.schedule_type,
            variance_type=diffusion_config.variance_type,
        )
    else:
        # Use the scheduler for the same pipeline as the VAE. It's important that the scheduler
        # doesn't, for example, clip the samples to the image range for latent diffusion.
        noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            scheduler_name_or_path,
            subfolder="scheduler",
        )
        # The below is necessary if using `num_inference_steps == num_training_steps`
        noise_scheduler.config.steps_offset = 0
        noise_scheduler.config["steps_offset"] = 0
        # Assert that the config scheduler is the same as the loaded scheduler
        assert (
            noise_scheduler.config.num_train_timesteps == diffusion_config.max_timestep
        )
        assert noise_scheduler.config.beta_start == diffusion_config.beta_start, (
            f"{noise_scheduler.config.beta_start} != {diffusion_config.beta_start}"
        )
        assert noise_scheduler.config.beta_end == diffusion_config.beta_end, (
            f"{noise_scheduler.config.beta_end} != {diffusion_config.beta_end}"
        )
        assert (
            noise_scheduler.config.beta_schedule == diffusion_config.schedule_type.name
        ), f"{noise_scheduler.config.beta_schedule} != {diffusion_config.schedule_type}"
        assert (
            noise_scheduler.config.variance_type == diffusion_config.variance_type.name
        ), f"{noise_scheduler.config.variance_type} != {diffusion_config.variance_type}"
    return noise_scheduler


def construct_inverse_kfac_preconditioner(
    kfac_config: KFACConfig,
    dataloader_config: DataLoaderConfig,
    model: UNet2DModel,
    params: list[nn.Parameter],
    vae: Optional[AutoencoderKL],
    noise_scheduler: DDPMScheduler,
    train_dataset: Dataset,
    cache_inverse_kfac: bool,
    cached_inverse_kfac_path: Optional[Path],
    device: torch.device,
) -> KFACInverseLinearOperatorWrapper:
    data_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=dataloader_config.train_batch_size,
        num_workers=dataloader_config.num_workers,
        pin_memory=dataloader_config.pin_memory,
        persistent_workers=dataloader_config.persistent_workers,
        drop_last=False,
    )
    # Define the "augmented" diffusion dataset iterable for KFAC computation
    # Repeat the dataloader:
    data_loader = ChainedIterable(
        *(data_loader for _ in range(kfac_config.num_samples_for_loss))
    )
    # If VAE is used, the sample will be transformed to the latent space first:
    if vae is not None:
        data_loader = reiterable_map(
            functools.partial(vae_encode_batch, vae=vae, device=device),
            data_loader,
        )
    data_loader = reiterable_map(
        functools.partial(
            data_batch_to_diffusion_regression_batch,
            noise_scheduler=noise_scheduler,
        ),
        data_loader,
    )
    # Add a size to the iterable for tqdm
    data_loader = SizedIterable(
        iterable=data_loader,
        size=(
            math.ceil(len(train_dataset) / dataloader_config.train_batch_size)
            * kfac_config.num_samples_for_loss
        ),
    )
    # Define the loss function with appropriate reduction
    loss_fn = nn.MSELoss(reduction="mean")
    # Only consider the parameters of layers supported by the KFAC implementation
    kfac_wrapped_model = KFACCompatibleImageRegressionModelWrapper(model, device=device)

    if kfac_config.correct_eigenvalues:
        kfac_class = EKFACLinearOperator
    else:
        kfac_class = KFACLinearOperator
    # KFAC is not computed here (since `check_deterministic=False`)
    kfac_ggn = kfac_class(
        kfac_wrapped_model,
        loss_fn,
        params,
        data_loader,
        progressbar=kfac_config.use_progressbar,
        fisher_type=kfac_config.fisher_type,
        mc_samples=kfac_config.mc_samples,
        kfac_approx=kfac_config.kfac_approx,
        num_data=kfac_config.num_samples_for_loss * len(train_dataset),
        # Setting `num_per_example_loss_terms` to 1 makes the scaling of KFAC correct
        # w.r.t. to the GGN we're trying to approximate, which includes a mean-reduced
        # loss.
        # This assumes the model output has shape (batch_size, height*width*channels).
        # curvlinops is correctly accounting for the mean reduction over the output
        # dimension (of size `height*width*channels`). If `num_per_example_loss_terms`
        # is not equal to `1`, curvlinops will assume that the gradient is additionally
        # scaled by `1 / num_per_example_loss_terms`, which is not what we want here.
        num_per_example_loss_terms=1,
        check_deterministic=kfac_config.check_deterministic,
    )
    # KFAC and its inverse are still not computed here, only in next step if it is
    # cached to disk or in `ScoreCalculator.compute_pairwise_scores` for the first
    # matrix-vector product
    kfac_ggn_inv = KFACInverseLinearOperator(
        kfac_ggn,
        damping=kfac_config.damping,
        use_heuristic_damping=kfac_config.use_heuristic_damping,
        min_damping=kfac_config.min_damping,
        use_exact_damping=kfac_config.use_exact_damping,
    )

    if cache_inverse_kfac:
        if cached_inverse_kfac_path is None:
            raise ValueError("Path to save inverse KFAC must be provided, got None")
        # Make the directory if it doesn't exist
        cached_inverse_kfac_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info("Computing KFAC GGN approximation and its inverse")
        # Trigger KFAC computation and inversion
        _ = kfac_ggn_inv @ ones(kfac_ggn_inv.shape[1], device=device)
        logging.info(f"Saving inverse KFAC to {cached_inverse_kfac_path}")
        save(kfac_ggn_inv.state_dict(), cached_inverse_kfac_path)
    elif cached_inverse_kfac_path is not None:
        logging.info(f"Loading cached inverse KFAC from {cached_inverse_kfac_path}")
        state_dict = load(cached_inverse_kfac_path, map_location=device)
        # Verify that the model weights match the ones used to compute the inverse KFAC
        loaded_model_state_dict: dict = state_dict["A"]["model_func_state_dict"]
        for val, val_loaded in zip(
            model.state_dict().values(), loaded_model_state_dict.values()
        ):
            if not equal(val, val_loaded):
                raise ValueError(
                    "The model weights do not match the ones used to compute the "
                    "inverse KFAC approximation."
                )
        # Check if the hyperparameters match for the inverse. If not, recompute
        # Note: We're not checking if the hyperparameters for the KFAC (not the inverse)
        # match!!!
        if (
            state_dict["damping"] != kfac_config.damping
            or state_dict["use_heuristic_damping"] != kfac_config.use_heuristic_damping
            or state_dict["min_damping"] != kfac_config.min_damping
            or state_dict["use_exact_damping"] != kfac_config.use_exact_damping
        ):
            logging.warning(
                "The hyperparameters for the cached inverse KFAC do not match the "
                + "current configuration:\n"
                + "\tCached:\n"
                + "\n".join(
                    [
                        f"\t\t{k}: {state_dict[k]}"
                        for k in [
                            "damping",
                            "use_heuristic_damping",
                            "min_damping",
                            "use_exact_damping",
                        ]
                    ]
                )
                + "\n\tConfig:\n"
                + "\n".join(
                    [
                        f"\t\t{k}: {v}"
                        for k, v in zip(
                            [
                                "damping",
                                "use_heuristic_damping",
                                "min_damping",
                                "use_exact_damping",
                            ],
                            [
                                kfac_config.damping,
                                kfac_config.use_heuristic_damping,
                                kfac_config.min_damping,
                                kfac_config.use_exact_damping,
                            ],
                        )
                    ]
                )
                + "\nRecomputing the inverse KFAC."
            )
            # Load the state dict for non-inverted KFAC
            # Note! This will overwrite the state_dict of kfac_ggn._model_func,
            # i.e. the parameters of model will be replaced
            kfac_ggn.load_state_dict(state_dict["A"])
            # Invert again:
            kfac_ggn_inv = KFACInverseLinearOperator(
                kfac_ggn,
                damping=kfac_config.damping,
                use_heuristic_damping=kfac_config.use_heuristic_damping,
                min_damping=kfac_config.min_damping,
                use_exact_damping=kfac_config.use_exact_damping,
            )
        else:
            # Just load the state dict
            kfac_ggn_inv.load_state_dict(state_dict)

    return KFACInverseLinearOperatorWrapper(kfac_ggn_inv)
