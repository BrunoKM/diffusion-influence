"""
Sample from a trained DDPM/DDIM model using the DDIM pipeline.
"""

import enum
import logging
import random
from pathlib import Path

import diffusers
import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from pydantic import dataclasses

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
    # Get the output directory created by Hydra:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seeds(config.seed)

    # Loading a DDPM/DDIM pipeline from pretrained into a DDIMPipeline object
    # should result in overriding the scheduler with the DDIM scheduler
    pipeline: DDPMPipelineRaw
    match config.sampling_method:
        case SamplingScheduleType.DDIM:
            raise NotImplementedError
        case SamplingScheduleType.DDPM:
            pipeline = DDPMPipelineRaw.from_pretrained(
                str(config.pretrained_pipeline_path),
            )  #  type: ignore
    pipeline.to("cuda")
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

        trajectory: list[np.ndarray] = pipeline(
            generator=generators,
            batch_size=this_batch_size,
            num_inference_steps=config.num_inference_steps,
            output_type="numpy",
            return_trajectory=True,
        )  # type: ignore
        # Convert to numpy array:
        trajectory = np.stack(trajectory, axis=0)  #  type: ignore # [num_inference_steps, batch_size, 3, H, W]

        final_image: np.ndarray = trajectory[-1]  # [batch_size, 3, H, W]
        # Post-process the final image
        if config.dequantized_model_output:
            raise NotImplementedError
        else:
            final_image = np.clip(final_image / 2 + 0.5, a_min=0, a_max=1)
            final_image = final_image.transpose((0, 2, 3, 1))

            images = pipeline.numpy_to_pil(final_image)

        for j, image in enumerate(images):
            image.save(output_dir / f"{i + j}.png")
            # Save the trajectory as well:
            np.save(output_dir / f"{i + j}_trajectory.npy", trajectory[:, j])


if __name__ == "__main__":
    hydra_main()
