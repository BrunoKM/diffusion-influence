"""
Sample from a trained DDPM/DDIM model using the DDIM pipeline.
"""

import enum
import logging
import random
from pathlib import Path
from typing import Optional

import diffusers
import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from diffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDPMPipeline,
    DDPMScheduler,
    UNet2DModel,
)
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from PIL import Image
from pydantic import dataclasses

from diffusion_influence.diffusion.latent_diffusion_utils import vae_decode_from_latent
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers
from diffusion_influence.pipelines import DDPMPipelineRaw


class SamplingScheduleType(str, enum.Enum):
    DDIM = "DDIM"
    """DDIM with "eta" set to 0.0."""
    DDPM = "DDPM"


@dataclasses.dataclass
class DiffusionSampleConfig:
    seed: int
    pretrained_pipeline_path: Path
    num_samples_to_generate: int
    batch_size: int

    sampling_method: SamplingScheduleType
    num_inference_steps: int
    """Into how many diffusion timesteps to discretise diffusion for each generated sample."""

    dequantized_model_output: bool = False
    """Whether the model was trained on dequantized data. If True, the samples need to
    quantized differently."""

    vae_name_or_path: Optional[str] = None
    """Whether to use a VAE to decode from the latent space. If None, the model is assumed to be a DDPM/DDIM model."""


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
cs.store(name="base_schema", node=DiffusionSampleConfig)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "configs"),
    config_name="base_schema",
)
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: DiffusionSampleConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore  # insufficient typing of to_object()
    main(config)


def main(config: DiffusionSampleConfig):
    # Set device to cuda or mps if available
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Get the output directory created by Hydra:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seeds(config.seed)
    # Load the VAE encoder (for latent diffusion) if it is specified in the config
    if config.vae_name_or_path is not None:
        vae = AutoencoderKL.from_pretrained(config.vae_name_or_path, subfolder="vae")
        vae.eval()
        vae.to(DEVICE)
    else:
        vae = None

    # Loading a DDPM/DDIM pipeline from pretrained into a DDIMPipeline object
    # should result in overriding the scheduler with the DDIM scheduler
    pipeline: DDIMPipeline | DDPMPipeline
    match config.sampling_method:
        case SamplingScheduleType.DDIM:
            if vae is not None:
                raise NotImplementedError

            pipeline = DDIMPipeline.from_pretrained(
                str(config.pretrained_pipeline_path)
            )  # type: ignore
        case SamplingScheduleType.DDPM:
            if vae is not None:
                pipeline = DDPMPipelineRaw(
                    unet=UNet2DModel.from_pretrained(
                        config.pretrained_pipeline_path / "unet"
                    ),
                    scheduler=DDPMScheduler.from_pretrained(
                        config.pretrained_pipeline_path / "scheduler"
                    ),
                )
            else:
                pipeline = DDPMPipelineRaw.from_pretrained(
                    str(config.pretrained_pipeline_path),
                )  #  type: ignore
    pipeline.to(DEVICE)
    pipeline.unet.eval()  # Should be set to eval already, but for good measure

    # Create a numpy random seed generator for the latents
    np_rng = np.random.default_rng(config.seed)  # Initial seed for the numpy generator

    # Generate seeds for each generated example
    torch_generator_seeds = np_rng.integers(
        low=np.iinfo(np.int64).min,
        high=np.iinfo(np.int64).max,
        size=config.num_samples_to_generate,
    )

    for i in tqdm.tqdm(range(0, config.num_samples_to_generate, config.batch_size)):
        this_batch_size = min(config.batch_size, config.num_samples_to_generate - i)

        # Create and seed PyTorch generators
        generators = [
            torch.Generator().manual_seed(int(seed))
            for seed in torch_generator_seeds[i : i + this_batch_size]
        ]

        images = pipeline(
            generator=generators,
            batch_size=this_batch_size,
            num_inference_steps=config.num_inference_steps,
            output_type="numpy",
            postprocess=(vae is None),
        )  # type: ignore
        # Decode from the latent space if LDM VAE was given
        if vae is not None:
            images = vae_decode_from_latent(latents=images, vae=vae)
            # Post-process the images
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.permute(0, 2, 3, 1)  # Channel last
            # To numpy
            images = images.cpu().numpy()
            images = pipeline.numpy_to_pil(images)

        else:
            images = images.images
            if config.dequantized_model_output:
                images = np.floor(images * 256).astype("uint8")
                if images.shape[-1] == 1:
                    # special case for grayscale (single channel) images
                    images = [
                        Image.fromarray(image.squeeze(), mode="L") for image in images
                    ]
                else:
                    images = [Image.fromarray(image) for image in images]
            else:
                images = pipeline.numpy_to_pil(images)

        for j, image in enumerate(images):
            image.save(output_dir / f"{i + j}.png")


if __name__ == "__main__":
    hydra_main()
