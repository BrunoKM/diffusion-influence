"""
Sample from a trained DDPM/DDIM model using the DDIM pipeline.

Changes to D-TRAK:
"""

import logging
import random
from dataclasses import field
from pathlib import Path
from typing import Any

import diffusers
import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from pydantic import dataclasses
from torch import FloatTensor, IntTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from diffusion_influence.config_schemas import (
    DatasetType,
    MeasurementType,
)
from diffusion_influence.constructors import (
    construct_model_and_vae,
    construct_model_training_config,
    construct_scheduler_from_config,
    get_transforms_for_dataset,
)
from diffusion_influence.data_utils import (
    load_samples_and_trajectories_dataset_from_dir,
    load_samples_dataset_from_dir,
)
from diffusion_influence.diffusion.elbo import (
    stochastic_elbo_sample,
)
from diffusion_influence.likelihood.likelihood import get_likelihood_fn
from diffusion_influence.likelihood.sde_interface import ddpm_scheduler_to_vpsde
from diffusion_influence.measurements import probability_of_sampling_trajectory
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers


@dataclasses.dataclass
class ComputeMeasureFunctionConfig:
    seed: int
    pretrained_model_dir_path: Path
    """Path to the directory containing the 1) pretrained model 2) corresponding config.json"""

    pretrained_model_config_path: Path
    """Used for loading in information about the training data. This will
    be the Hydra config (rather than the Hugging Face model one), usually
    located in .hydra/config.yaml"""

    samples_dir_path: Path
    batch_size: int
    dataset_name: DatasetType
    """Dataset name used to determine the correct transforms for the dataset."""

    # Number of samples to compute the measurement:
    num_samples_for_measurement: int
    measurement: MeasurementType
    measurement_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.measurement == MeasurementType.LOSS_AT_TIME:
            assert "time" in self.measurement_kwargs


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Check diffusers version sufficiently high
diffusers.utils.check_min_version("0.16.0")

logger = logging.Logger(__name__)

# Register custom resolvers for the config parser:
register_custom_resolvers()

# Register the structured dataclass config schema (for type-checking, autocomplete, and validation) with Hydra:
cs = ConfigStore.instance()
cs.store(name="base_schema", node=ComputeMeasureFunctionConfig)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "configs"),
    config_name="base_schema",
)
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: ComputeMeasureFunctionConfig = omegaconf.OmegaConf.to_object(
        omegaconf_config
    )  # type: ignore  # insufficient typing of to_object()
    main(config)


def main(config: ComputeMeasureFunctionConfig):
    # Load the training config.
    model_training_config = construct_model_training_config(
        config.pretrained_model_config_path
    )

    # Set device to cuda or mps if available.
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

    # Initialize the scheduler
    noise_scheduler = construct_scheduler_from_config(
        scheduler_name_or_path=model_training_config.scheduler_name_or_path,
        diffusion_config=model_training_config.diffusion,
    )
    max_diffusion_time = noise_scheduler.config.num_train_timesteps  # type: ignore
    # Cast noise_scheduler to device (there is no explicit method in Hugging Face diffusers
    # , they just expect you to rely on the side-effects of the methods...)
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(DEVICE)

    if noise_scheduler.config.prediction_type != "epsilon":  # type: ignore
        raise ValueError(
            f"Only epsilon prediction type is supported. Got {noise_scheduler.config.prediction_type}."  # type: ignore
        )

    # --- Load the image samples ---
    _, eval_transforms = get_transforms_for_dataset(
        config.dataset_name,
    )

    if config.measurement in {MeasurementType.SAMPLING_TRAJECTORY_LOGPROB}:
        dataset = load_samples_and_trajectories_dataset_from_dir(
            config.samples_dir_path, Compose(eval_transforms)
        )
    else:
        dataset = load_samples_dataset_from_dir(
            config.samples_dir_path, Compose(eval_transforms)
        )

    # --- Create the dataloader ---
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=1,  # Don't need many workers since we'll be sampling noise many times
        drop_last=False,
    )

    if config.measurement == MeasurementType.LOG_LIKELIHOOD:
        sde = ddpm_scheduler_to_vpsde(noise_scheduler)
        likelihood_fn = get_likelihood_fn(
            sde=sde,
            data_range_for_correction=(
                model_training_config.data.data_range if vae is None else None
            ),
            unit="BPD",
        )

    # List of average losses for each example
    measurements_list = np.zeros(shape=(len(dataset),), dtype=float)
    with torch.no_grad():
        for _ in tqdm.tqdm(range(config.num_samples_for_measurement)):
            examples_processed = 0
            for batch in tqdm.tqdm(dataloader, leave=False):
                clean_images = batch["input"].to(DEVICE)
                # If using latent diffusion, depending on the measurement type, simply add
                # the VAE encoding
                if vae is not None:
                    match config.measurement:
                        case (
                            MeasurementType.LOSS
                            | MeasurementType.LOSS_AT_TIME
                            | MeasurementType.SIMPLIFIED_ELBO
                            | MeasurementType.LOG_LIKELIHOOD
                        ):
                            latent_images_dist = vae.encode(clean_images).latent_dist
                        case _:
                            raise NotImplementedError
                batch_size = clean_images.shape[0]

                batch_measurements: list[float]
                match config.measurement:
                    case MeasurementType.LOSS:
                        images = (
                            latent_images_dist.sample() * vae.config.scaling_factor
                            if vae is not None
                            else clean_images
                        )
                        # Sample a random timestep for each image
                        timesteps: IntTensor = torch.randint(
                            low=0,
                            high=max_diffusion_time,
                            size=(batch_size,),
                            device=DEVICE,
                            dtype=torch.long,
                        )  # type: ignore

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (forward diffusion process)
                        noise: FloatTensor = torch.randn_like(images, dtype=torch.float)  #  type: ignore
                        noisy_image = noise_scheduler.add_noise(
                            images, noise, timesteps
                        )

                        # Predict the noise residual and compute loss
                        with torch.no_grad():
                            model_pred = model(noisy_image, timesteps).sample
                        loss = torch.nn.functional.mse_loss(
                            model_pred, noise, reduction="none"
                        )
                        loss = loss.mean(dim=(1, 2, 3))  # [batch_size, ]
                        batch_measurements = loss.detach().cpu().numpy().tolist()
                    case MeasurementType.LOSS_AT_TIME:
                        assert "time" in config.measurement_kwargs
                        time = int(config.measurement_kwargs["time"])
                        time = torch.tensor(time, dtype=torch.long, device=DEVICE)
                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (forward diffusion process)
                        noise: FloatTensor = torch.randn_like(
                            clean_images, dtype=torch.float
                        )  #  type: ignore
                        noisy_image = noise_scheduler.add_noise(
                            clean_images,
                            noise,
                            time,
                        )

                        # Predict the noise residual and compute loss
                        with torch.no_grad():
                            model_pred = model(noisy_image, time).sample
                        loss = torch.nn.functional.mse_loss(
                            model_pred, noise, reduction="none"
                        )
                        loss = loss.mean(dim=(1, 2, 3))  # [batch_size, ]
                        batch_measurements = loss.detach().cpu().numpy().tolist()

                    case MeasurementType.SIMPLIFIED_ELBO:  # TODO: Merge with MeasurementType.LOSS and only add the weighting
                        # Sample a random timestep for each image
                        timesteps: IntTensor = torch.randint(
                            low=0,
                            high=max_diffusion_time,
                            size=(batch_size,),
                            device=DEVICE,
                            dtype=torch.long,
                        )  # type: ignore

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (forward diffusion process)
                        noise: FloatTensor = torch.randn_like(
                            clean_images, dtype=torch.float
                        )  #  type: ignore
                        noisy_image = noise_scheduler.add_noise(
                            clean_images, noise, timesteps
                        )

                        # Predict the noise residual and compute loss
                        with torch.no_grad():
                            model_pred = model(noisy_image, timesteps).sample
                            loss_weight_terms = (
                                noise_scheduler.betas.to(DEVICE)
                                / (
                                    2
                                    * noise_scheduler.alphas.to(DEVICE)
                                    * (1 - noise_scheduler.alphas_cumprod.to(DEVICE))
                                )
                            ).to(DEVICE)[timesteps]
                        loss = torch.nn.functional.mse_loss(
                            model_pred, noise, reduction="none"
                        )
                        loss = loss.mean(dim=(1, 2, 3))  # [batch_size, ]
                        loss = loss * loss_weight_terms
                        batch_measurements = loss.detach().cpu().numpy().tolist()
                    case MeasurementType.ELBO:
                        elbo_sample = stochastic_elbo_sample(
                            denoise_model=model,
                            noise_scheduler=noise_scheduler,
                            original_samples=clean_images,
                            data_range=(-1.0, 1.0),
                        )
                        batch_measurements = elbo_sample.detach().cpu().numpy().tolist()
                    case MeasurementType.LOG_LIKELIHOOD:
                        images = (
                            latent_images_dist.sample() * vae.config.scaling_factor
                            if vae is not None
                            else clean_images
                        )

                        ll_sample, z, nfe = likelihood_fn(  # type: ignore  # defined in measurement is LOG_LIKELIHOOD
                            model=model,
                            data=images,
                        )
                        batch_measurements = ll_sample.cpu().numpy().tolist()
                    case MeasurementType.SAMPLING_TRAJECTORY_LOGPROB:
                        trajectory = batch["trajectory"].to(DEVICE)
                        log_probs = probability_of_sampling_trajectory(
                            denoise_model=model,
                            noise_scheduler=noise_scheduler,
                            trajectory=trajectory,
                        )  # [batch_size]
                        batch_measurements = log_probs.cpu().numpy().tolist()
                    case _:
                        raise ValueError(
                            f"Measurement type {config.measurement} not recognised."
                        )
                # Extend loss_list by `batch_size` entries
                measurements_list[
                    examples_processed : examples_processed + batch_size
                ] += batch_measurements
                examples_processed += batch_size

    measurements_list /= config.num_samples_for_measurement

    # Save the loss list to a binary file with numpy
    np.save(output_dir / "measurements.npy", measurements_list)
    # Save to a csv file:
    np.savetxt(output_dir / "measurements.csv", measurements_list, delimiter=",")


if __name__ == "__main__":
    hydra_main()
