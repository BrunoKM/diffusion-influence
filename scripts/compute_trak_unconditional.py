import functools
import logging
import os
import random
import shutil
from dataclasses import field
from pathlib import Path
from typing import Any, Optional

import diffusers
import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from diffusers.schedulers import DDPMScheduler
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from numpy.lib.format import open_memmap
from pydantic import dataclasses
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from trak import TRAKer
from trak.savers import MmapSaver

from diffusion_influence.config_schemas import (
    DataLoaderConfig,
    MeasurementType,
)
from diffusion_influence.constructors import (
    construct_datasets_with_transforms,
    construct_model_and_vae,
    construct_model_training_config,
    get_train_only_transforms_for_dataset,
    get_transforms_for_dataset,
)
from diffusion_influence.data_utils import load_samples_dataset_from_dir
from diffusion_influence.diffusion_trak.gradient_computers import (
    DiffusionGradientComputer,
)
from diffusion_influence.diffusion_trak.modelout_functions import (
    UnconditionalDiffusionModelOutput,
    diffusion_loss_output_function,
    elbo_output_function,
    latent_diffusion_loss_output_function,
    latent_square_norm_output_function,
    simplified_elbo_output_function,
    square_norm_output_function,
)
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers


@dataclasses.dataclass
class TrakConfig:
    projection_dim: int
    latent_diffusion: bool
    lambda_reg: float
    experiment_name: str
    use_half_precision: bool
    save_dir: Optional[str] = None
    """By default save to the output directory / 'save_dir'"""


@dataclasses.dataclass
class TRAKFeaturiseConfig:
    seed: int
    pretrained_model_dir_path: Path
    """Path to the directory containing the 1) pretrained model 2) corresponding config.json"""

    pretrained_model_config_path: Path
    """Used for loading in information about the training data. This will
    be the Hydra config (rather than the Hugging Face model one), usually
    located in .hydra/config.yaml"""
    model_id: int

    num_samples_for_loss: int
    num_samples_for_score: int
    dataloader: DataLoaderConfig

    trak: TrakConfig
    samples_dir_path: Path

    training_loss: MeasurementType
    measurement: MeasurementType
    measurement_kwargs: dict[str, Any] = field(default_factory=dict)

    cached_trak_gradients_path: Optional[Path] = None
    """If provided, load the gradients from this path instead of computing them from scratch."""
    delete_grads_after_use: bool = True
    delete_features_after_use: bool = True
    delete_savedir_after_use: bool = False

    def __post_init__(self):
        if self.measurement == MeasurementType.LOSS_AT_TIME:
            assert "time" in self.measurement_kwargs


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cached_trak_grads(
    source_save_dir: Path, saver: MmapSaver, model_id: Optional[int] = None
) -> None:
    """
    Copy the gradients from the save_dir of (possibly another) TRAK run
    to the current save_dir. This is useful to save on compute and storage
    when e.g. using the same model for multiple TRAK runs (e.g. with different
    damping factors/measurement functions).

    This is a somewhat hacky way to do this, but it works
    """
    if not isinstance(saver, MmapSaver):
        raise ValueError(
            f"Expected `saver` to be an instance of `MmapSaver`, got {type(saver)}"
        )
    if saver.current_model_id != model_id:
        raise ValueError(
            f"Model ID mismatch for saver: Expected {model_id}, got {saver.current_model_id}"
        )
    if not source_save_dir.exists():
        raise FileNotFoundError(f"Cache directory {source_save_dir} does not exist.")

    source_model_subfolder = source_save_dir / str(model_id)
    source_grads_file = source_model_subfolder / "grads.mmap"
    source_is_featurized_file = source_model_subfolder / "_is_featurized.mmap"
    source_out_to_loss_file = source_model_subfolder / "out_to_loss.mmap"
    if not source_model_subfolder.is_dir():
        raise FileNotFoundError(
            f"Model subfolder {source_model_subfolder} does not exist."
        )
    if not source_grads_file.exists():
        raise FileNotFoundError(f"File {source_grads_file} does not exist.")
    if not source_is_featurized_file.exists():
        raise FileNotFoundError(f"File {source_is_featurized_file} does not exist.")
    if not source_out_to_loss_file.exists():
        raise FileNotFoundError(f"File {source_out_to_loss_file} does not exist.")
    # Open the mmap files:
    source_grads = open_memmap(source_grads_file, mode="r")
    source_is_featurized = open_memmap(source_is_featurized_file, mode="r")
    source_out_to_loss = open_memmap(source_out_to_loss_file, mode="r")
    # Validate dtype and shape match the current saver
    if source_grads.dtype != saver.current_store["grads"].dtype:
        raise ValueError(
            f"Expected matching dtype {saver.current_store['grads'].dtype} for cached grads, got {source_grads.dtype}"
        )
    if source_grads.shape != saver.current_store["grads"].shape:
        raise ValueError(
            f"Expected matching shape {saver.current_store['grads'].shape} for cached grads, got {source_grads.shape}"
        )
    if source_is_featurized.shape != saver.current_store["is_featurized"].shape:
        raise ValueError(
            f"Expected matching shape {saver.current_store['is_featurized'].shape} for cached is_featurized, got {source_is_featurized.shape}"
        )
    if source_out_to_loss.shape != saver.current_store["out_to_loss"].shape:
        raise ValueError(
            f"Expected matching shape {saver.current_store['out_to_loss'].shape} for cached out_to_loss, got {source_out_to_loss.shape}"
        )
    if source_out_to_loss.dtype != saver.current_store["out_to_loss"].dtype:
        raise ValueError(
            f"Expected matching dtype {saver.current_store['out_to_loss'].dtype} for cached out_to_loss, got {source_out_to_loss.dtype}"
        )
    # Copy the data over:
    saver.current_store["grads"][:] = source_grads[:]
    saver.current_store["is_featurized"][:] = source_is_featurized[:]
    saver.current_store["out_to_loss"][:] = source_out_to_loss[:]
    # Flush the changes to disk:
    saver.current_store["grads"].flush()
    saver.current_store["is_featurized"].flush()
    saver.current_store["out_to_loss"].flush()
    # “Close” the mmap files (currently no proper API for this in numpy)
    # https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    del source_grads
    del source_is_featurized
    del source_out_to_loss
    # Make sure to update the metadata in the saver, just as the end of featurize.
    # This is needed because saver.seralize_current_model_id_metadata() is not called
    # in traker.featurize() if all indices are already done..
    saver.serialize_current_model_id_metadata()


# Check diffusers version sufficiently high
diffusers.utils.check_min_version("0.16.0")

logger = logging.Logger(__name__)

# Register custom resolvers for the config parser:
register_custom_resolvers()

# Register the structured dataclass config schema (for type-checking, autocomplete, and validation) with Hydra:
cs = ConfigStore.instance()
cs.store(name="base_schema", node=TRAKFeaturiseConfig)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "configs"),
    config_name="base_schema",
)
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: TRAKFeaturiseConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore  # insufficient typing of to_object()
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


def main(config: TRAKFeaturiseConfig):
    # Load the training config.
    model_training_config = construct_model_training_config(
        config.pretrained_model_config_path
    )
    data_config = model_training_config.data

    # Set device to cuda if available.
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # Get the output directory created by Hydra:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seeds(config.seed)

    # Prepare the trained model.
    model, vae = construct_model_and_vae(
        config.pretrained_model_dir_path, model_training_config, DEVICE
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_training_config.diffusion.max_timestep,
        beta_start=model_training_config.diffusion.beta_start,
        beta_end=model_training_config.diffusion.beta_end,
        beta_schedule=model_training_config.diffusion.schedule_type,
        variance_type=model_training_config.diffusion.variance_type,
    )

    if noise_scheduler.config.prediction_type != "epsilon":  #  type: ignore
        raise ValueError(
            f"Only epsilon prediction type is supported. Got {noise_scheduler.config.prediction_type}."  #  type: ignore
        )

    _, eval_datasets = construct_datasets_with_transforms(data_config)
    train_clean = eval_datasets["train_no_augment"]
    # Get the train transforms for computing the per-example loss. These will
    # be applied AFTER the base transforms (e.g. normalization), which
    # is different from the training pipeline, but for datasets of interest
    # (e.g. CIFAR-10 with random horizontal flip) this shouldn't change the result
    train_transforms = get_train_only_transforms_for_dataset(data_config.dataset_name)

    train_loader = DataLoader(
        train_clean,  #  type: ignore
        # train_dataset,  # type: ignore
        batch_size=config.dataloader.train_batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=config.dataloader.persistent_workers,
    )

    if vae is None:
        match config.measurement:
            case MeasurementType.LOSS:
                measure_func = diffusion_loss_output_function
            case MeasurementType.ELBO:
                measure_func = elbo_output_function
            case MeasurementType.SIMPLIFIED_ELBO:
                measure_func = simplified_elbo_output_function
            case MeasurementType.SQUARE_NORM:
                measure_func = square_norm_output_function
            case _:
                raise NotImplementedError(
                    f"Unrecognized measurement type: {config.measurement}"
                )
        match config.training_loss:
            case MeasurementType.LOSS:
                featurize_output_func = diffusion_loss_output_function
            case MeasurementType.ELBO:
                featurize_output_func = elbo_output_function
            case MeasurementType.SIMPLIFIED_ELBO:
                featurize_output_func = simplified_elbo_output_function
            case MeasurementType.SQUARE_NORM:
                featurize_output_func = square_norm_output_function
            case _:
                raise NotImplementedError(
                    f"Unrecognized measurement type: {config.training_loss}"
                )
    else:
        match config.measurement:
            case MeasurementType.LOSS:
                measure_func = functools.partial(
                    latent_diffusion_loss_output_function,
                    vae=vae,
                )
            case MeasurementType.SQUARE_NORM:
                measure_func = functools.partial(
                    latent_square_norm_output_function,
                    vae=vae,
                )
            case _:
                raise NotImplementedError(
                    f"Unrecognized measurement type: {config.measurement}"
                )
        match config.training_loss:
            case MeasurementType.LOSS:
                featurize_output_func = functools.partial(
                    latent_diffusion_loss_output_function,
                    vae=vae,
                )
            case MeasurementType.SQUARE_NORM:
                featurize_output_func = functools.partial(
                    latent_square_norm_output_function,
                    vae=vae,
                )
            case _:
                raise NotImplementedError(
                    f"Unrecognized measurement type: {config.training_loss}"
                )

    trak_save_dir = (
        config.trak.save_dir
        if config.trak.save_dir is not None
        else str(output_dir / "save_dir")
    )
    task = UnconditionalDiffusionModelOutput(
        measure_func=measure_func,
        featurize_output_func=featurize_output_func,
        noise_scheduler=noise_scheduler,
    )
    # Since TRAK doesn't by default distinguish between loss function
    # and measurement, it is necessary to "hack" around the implementation a little
    # bit to get the desired functionality. (code mostly taken from Journey-TRAK
    # https://github.com/MadryLab/journey-TRAK)
    traker = TRAKer(
        model=model,
        task=task,
        gradient_computer=functools.partial(
            DiffusionGradientComputer,
            train_transforms=Compose(train_transforms),
            num_samples_for_scoring=config.num_samples_for_score,
            num_samples_for_featurizing=config.num_samples_for_loss,
        ),  #  type: ignore
        proj_dim=config.trak.projection_dim,
        lambda_reg=config.trak.lambda_reg,
        save_dir=trak_save_dir,
        use_half_precision=config.trak.use_half_precision,
        train_set_size=len(train_loader.dataset),  #  type: ignore
        load_from_save_dir=True,
        device="cuda",
    )

    # This is a hacky way to use TRAK with different measurement and loss functions,
    # as it doesn't support that in the API. We set the flag to True to tell
    # the gradient computer to use the loss function (rather than the measurement) to
    # compute the gradients
    traker.gradient_computer._are_we_featurizing = True
    traker.task._are_we_featurizing = True  #  type: ignore

    traker.load_checkpoint(
        # Get the state-dict directly from the model (which has been loaded)
        model.state_dict(),
        model_id=config.model_id,
    )

    # Possibly load pre-cached gradients into the gradient computer
    # (has to be done after the `load_checkpoint` call to make sure saver is initialized)
    if config.cached_trak_gradients_path is not None:
        logging.info(
            f"Loading precomputed TRAK gradients from {config.cached_trak_gradients_path}"
        )
        load_cached_trak_grads(
            source_save_dir=config.cached_trak_gradients_path,
            saver=traker.saver,
            model_id=config.model_id,
        )
        logging.info("Precomputed TRAK gradients loaded.")

    for batch in tqdm.tqdm(train_loader, desc="Computing TRAK embeddings"):
        # batch will consist of [image, label, timestep]
        current_bs = batch["input"].shape[0]
        batch = {k: x.to(DEVICE) for k, x in batch.items()}

        traker.featurize(batch=batch, num_samples=current_bs)

    traker.finalize_features(model_ids=[config.model_id])

    # Load query dataset.
    _, eval_transforms = get_transforms_for_dataset(
        model_training_config.data.dataset_name,
    )
    query_dataset = load_samples_dataset_from_dir(
        config.samples_dir_path, Compose(eval_transforms)
    )

    query_loader = DataLoader(
        query_dataset,  #  type: ignore
        batch_size=config.dataloader.train_batch_size,  # Simply use the same batch size - it does not really matter.
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        persistent_workers=config.dataloader.persistent_workers,
    )

    traker.gradient_computer._are_we_featurizing = False
    traker.task._are_we_featurizing = False

    traker.start_scoring_checkpoint(
        exp_name=config.trak.experiment_name,
        num_targets=len(query_loader.dataset),
        model_id=config.model_id,
        checkpoint=model.state_dict(),
    )
    for batch in tqdm.tqdm(query_loader, desc="Computing TRAK scores"):
        current_bs = batch["input"].shape[0]
        batch = {k: x.to(DEVICE) for k, x in batch.items()}
        traker.score(batch=batch, num_samples=current_bs)

    scores = traker.finalize_scores(exp_name=config.trak.experiment_name)
    scores_path = output_dir / "scores.npy"
    np.save(file=scores_path, arr=torch.from_numpy(scores).T.numpy())

    # Delete the save_dir if requested:
    if config.delete_grads_after_use:
        grads_path = Path(trak_save_dir) / str(config.model_id) / "grads.mmap"
        logging.info(f"Deleting gradients file {grads_path}")
        # Delete the file:
        grads_path.unlink()
    if config.delete_features_after_use:
        features_path = Path(trak_save_dir) / str(config.model_id) / "features.mmap"
        logging.info(f"Deleting features file {features_path}")
        # Delete the file:
        features_path.unlink()
    if config.delete_savedir_after_use:
        logging.info(f"Deleting save directory {trak_save_dir}")
        shutil.rmtree(Path(trak_save_dir))


if __name__ == "__main__":
    hydra_main()
