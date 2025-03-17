"""
Changes to D-TRAK:
- Remove the (0.95, 0.999) beta hyperparameter values for AdamW

"""

import logging
import math
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import datasets
import diffusers
import hydra
import numpy as np
import omegaconf
import torch
import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMPipeline,
    ImagePipelineOutput,
    UNet2DModel,
)
from diffusers.training_utils import EMAModel
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import wandb
from diffusion_influence.config_schemas import DatasetType, DiffusionTrainConfig
from diffusion_influence.constructors import (
    construct_dataloader,
    construct_datasets_with_transforms,
    construct_optimizer,
    construct_scheduler_from_config,
    weight_examples_in_dataset,
)
from diffusion_influence.data_utils import ad_infinitum, take_n
from diffusion_influence.diffusion import latent_diffusion_utils
from diffusion_influence.diffusion.elbo import (
    ddpm_elbo,
    ddpm_elbo_dequantized_data,
    ldm_elbo,
)
from diffusion_influence.diffusion.losses import ddpm_loss_for_timestep
from diffusion_influence.diffusion.train_util import (
    ddpm_train_step,
    weighted_ddpm_train_step,
)
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers
from diffusion_influence.pipelines import DDPMPipelineRaw


def set_seeds(seed):
    set_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Check diffusers version sufficiently high
diffusers.utils.check_min_version("0.16.0")

logger = get_logger(__name__, log_level="INFO")

# Register custom resolvers for the config parser:
register_custom_resolvers()

# Register the structured dataclass config schema (for type-checking, autocomplete, and validation) with Hydra:
cs = ConfigStore.instance()
cs.store(name="base_schema", node=DiffusionTrainConfig)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "configs"),
    config_name="base_schema",
)
def main_with_wandb(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: DiffusionTrainConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore  # insufficient typing of to_object()
    config_dict: dict[str, Any] = omegaconf.OmegaConf.to_container(
        omegaconf_config, enum_to_str=True, resolve=True
    )  #  type: ignore
    accelerator = Accelerator(
        gradient_accumulation_steps=config.accelerate.gradient_accumulation_steps,
        mixed_precision=config.accelerate.mixed_precision.value,
        log_with="wandb",
        project_config=ProjectConfiguration(
            project_dir=HydraConfig.get().runtime.output_dir,
            automatic_checkpoint_naming=True,
            # `total_limit` only enforced if `automatic_checkpoint_naming=True`
            total_limit=config.accelerate.checkpoints_total_limit,
        ),
    )
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            config.wandb.project_name,
            config=config_dict,
            init_kwargs={
                "wandb": {
                    "entity": config.wandb.entity,
                    # "project": config.wandb.project_name,
                    "group": config.wandb.group_name,
                    # This is needed to make WandB and Hydra play nicely:
                    "settings": wandb.Settings(start_method="thread"),
                    # Log the config to WandB
                    # Allow for disabling upload when testing code
                    "mode": config.wandb.mode,
                    "name": config.wandb.name,
                }
            },
        )

    # --- Runtime setup (logging directories, etc.)
    wandb.summary["output_dir"] = HydraConfig.get().runtime.output_dir
    main(config, accelerator=accelerator)


def main(config: DiffusionTrainConfig, accelerator: Accelerator):
    # Get the output directory created by Hydra:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, checkpoint_output_dir):
        # Save the step: (for resuming training)
        with (Path(checkpoint_output_dir) / "step.txt").open("w") as f:
            f.write(str(accelerator.step))
        if ema_model is not None:
            ema_model.save_pretrained(Path(checkpoint_output_dir) / "unet_ema")

        for i, model in enumerate(models):
            model.save_pretrained(Path(checkpoint_output_dir) / "unet")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        if ema_model is not None:
            loaded_model = EMAModel.from_pretrained(
                Path(input_dir) / "unet_ema", UNet2DModel
            )
            ema_model.load_state_dict(loaded_model.state_dict())
            ema_model.to(accelerator.device)
            del loaded_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            loaded_model: UNet2DModel = UNet2DModel.from_pretrained(
                input_dir, subfolder="unet"
            )  #  type: ignore
            model.register_to_config(**loaded_model.config)

            model.load_state_dict(loaded_model.state_dict())
            del loaded_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seeds(config.seed)

    # Initialize the model
    model_config = UNet2DModel.load_config(config.model_config_path)
    model: UNet2DModel = UNet2DModel.from_config(model_config)  #  type: ignore

    # Create EMA for the model.
    if config.optimisation.ema_decay is not None:
        ema_model = EMAModel(
            model.parameters(),
            decay=config.optimisation.ema_decay,
            use_ema_warmup=config.optimisation.use_ema_warmup,
            power=config.optimisation.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
    else:
        ema_model = None

    # Load the VAE encoder (for latent diffusion) if it is specified in the config
    if config.vae_name_or_path is not None:
        vae = AutoencoderKL.from_pretrained(config.vae_name_or_path, subfolder="vae")
        vae.requires_grad_(False)
        vae.eval()
    else:
        vae = None

    vae_encode_to_latent = partial(latent_diffusion_utils.vae_encode_to_latent, vae=vae)
    vae_decode_from_latent = partial(
        latent_diffusion_utils.vae_decode_from_latent, vae=vae
    )

    # Initialize the scheduler
    noise_scheduler = construct_scheduler_from_config(
        scheduler_name_or_path=config.scheduler_name_or_path,
        diffusion_config=config.diffusion,
    )

    # Initialize the optimizer
    optimizer, scheduler = construct_optimizer(config=config, model=model)

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    train_dataset, eval_datasets = construct_datasets_with_transforms(config.data)
    if config.weighted_examples_idxs_path is not None:
        train_dataset = weight_examples_in_dataset(
            dataset=train_dataset,
            example_idxs=(
                np.loadtxt(config.data.examples_idxs_path, dtype=int)
                if config.data.examples_idxs_path is not None
                else np.arange(len(train_dataset))
            ),
            weighted_examples_idxs=np.loadtxt(
                config.weighted_examples_idxs_path, dtype=int
            ),
            invert_weighted_examples_idxs=config.invert_weighted_examples_idxs,
            weighted_examples_weight=config.weighted_examples_weight,
        )
        # This is a pretty sketchy way of doing things:
        # The HuggingFace interface doesn't easily allow of adding another transform to
        # previous transforms in the dataset to cast example_weights to torch
        train_dataset.set_format(type="torch", columns=["input", "example_weight"])

    train_dataloader, eval_dataloaders = construct_dataloader(
        config.dataloader,
        train_dataset,
        eval_datasets
        | {
            "train_eval": train_dataset
        },  # Create additional dataloader for the training set for evaluation
    )

    # Prepare everything with `accelerator`.
    # Do not prepare EMA/VAE encoder, as they do not have “trainable” parameters
    (
        model,
        optimizer,
        train_dataloader,
        scheduler,
        *eval_dataloaders_values,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler, *eval_dataloaders.values()
    )
    eval_dataloaders = dict(zip(eval_dataloaders.keys(), eval_dataloaders_values))

    if ema_model is not None:
        ema_model.to(accelerator.device)
    if vae is not None:
        vae.to(accelerator.device)

    total_batch_size = (
        config.dataloader.train_batch_size
        * accelerator.num_processes
        * config.accelerate.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.accelerate.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(
        f"Num Epochs = {math.ceil(config.num_training_iter / num_update_steps_per_epoch)}"
    )
    logger.info(f"Total optimization steps = {config.num_training_iter}")
    logger.info(f"Warmup steps = {config.optimisation.warmup_steps}")
    logger.info(
        f"Instantaneous batch size per device = {config.dataloader.train_batch_size}"
    )
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"Gradient Accumulation steps = {config.accelerate.gradient_accumulation_steps}"
    )

    step = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint is not None:
        if config.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = dirs[-1] if len(dirs) > 0 else None
        else:
            checkpoint_path = os.path.basename(config.resume_from_checkpoint)

        if checkpoint_path is None:
            raise ValueError(
                f"Could not find a checkpoint to resume from in {output_dir}"
            )

        logger.info(f"Resuming from checkpoint {checkpoint_path}")
        with (Path(checkpoint_path) / "step.txt").open("r") as f:
            step_str = f.read()
            step = int(step_str)
        accelerator.load_state(os.path.join(output_dir, checkpoint_path))

        start_step = step * config.accelerate.gradient_accumulation_steps
        start_step = start_step % (
            num_update_steps_per_epoch * config.accelerate.gradient_accumulation_steps
        )

        # Skip the first 'start_step' batches of the dataloader
        # TODO: verify that 'start_step is the correct number of batches to skip
        train_dataloader = accelerator.skip_first_batches(train_dataloader, start_step)
    else:
        start_step = 0

    # --- Training and evaluation loop
    train_iter = ad_infinitum(train_dataloader)
    for step, batch in (
        progress_bar := tqdm.tqdm(
            zip(range(start_step, config.num_training_iter), train_iter),
            total=config.num_training_iter - start_step,
            disable=not accelerator.is_local_main_process,
        )
    ):
        model.train()

        clean_images = batch["input"]
        if vae is not None:
            clean_images = vae_encode_to_latent(clean_images)
        if config.weighted_examples_idxs_path is not None:
            example_weights = batch["example_weight"]
            # TODO: maybe unify with ddpm_train_step
            loss = weighted_ddpm_train_step(
                denoise_model=model,
                batch=clean_images,
                example_weights=example_weights,
                optimizer=optimizer,
                scheduler=scheduler,
                noise_scheduler=noise_scheduler,
                max_time=config.diffusion.max_timestep,
                device=accelerator.device,
                clip_grad=config.optimisation.clip_grad,
                accelerator=accelerator,
            )
        else:
            loss = ddpm_train_step(
                denoise_model=model,
                batch=clean_images,
                optimizer=optimizer,
                scheduler=scheduler,
                noise_scheduler=noise_scheduler,
                max_time=config.diffusion.max_timestep,
                device=accelerator.device,
                clip_grad=config.optimisation.clip_grad,
                accelerator=accelerator,
            )

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if ema_model is not None:
                ema_model.step(model.parameters())

            if (
                config.checkpoint_frequency is not None
                and step % config.checkpoint_frequency == 0
                and step > 0
            ):
                if accelerator.is_main_process:
                    save_path = output_dir / f"checkpoint-{step}"
                    accelerator.save_state(str(save_path))
                    logger.info(f"Saved state to {save_path}")

        logs = {
            "loss": loss.detach().item(),
            "lr": scheduler.get_last_lr()[0],
            "step": step,
        }
        if config.optimisation.ema_decay is not None:
            logs["ema_decay"] = ema_model.cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)

        # ------------------------------- Evaluation -----------------------------------
        if step % config.eval_frequency == 0 or step == config.num_training_iter - 1:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unet = accelerator.unwrap_model(model)
                unet.eval()

                # --- Set the EMA parameters for the model ---
                if ema_model is not None:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                # --- Log test losses ---
                with torch.no_grad():
                    losses_by_dataset_by_timestep = {}
                    for dataset_name, dataloader in eval_dataloaders.items():
                        losses_by_timestep: dict[int, float] = defaultdict(lambda: 0)
                        for eval_batch in take_n(dataloader, n=5):
                            clean_images = eval_batch["input"]
                            if vae is not None:
                                clean_images = vae_encode_to_latent(clean_images)
                            for timestep in config.log_loss_at_timesteps:
                                loss = ddpm_loss_for_timestep(
                                    denoise_model=unet,
                                    timestep=timestep,
                                    batch=clean_images,
                                    noise_scheduler=noise_scheduler,
                                    device=accelerator.device,
                                )
                                losses_by_timestep[timestep] += loss.item()
                        for timestep in config.log_loss_at_timesteps:
                            # TODO: make below constant (5) explicit
                            losses_by_dataset_by_timestep[
                                f"{dataset_name}.timestep_{timestep}"
                            ] = losses_by_timestep[timestep] / 5
                # --- Log ELBO ---
                match config.data.dataset_name:
                    case DatasetType.cifar10deq | DatasetType.cifar2deq:
                        if vae is not None:
                            raise ValueError(
                                "ELBO not implemented for dequantized data with VAE"
                            )
                        elbo_function = ddpm_elbo_dequantized_data
                    case (
                        DatasetType.cifar10 | DatasetType.cifar2 | DatasetType.artbench
                    ):
                        if vae is None:
                            elbo_function = ddpm_elbo
                        else:
                            elbo_function = partial(ldm_elbo, vae=vae)
                    case _:
                        raise ValueError(
                            f"ELBO not implemented for {config.data.dataset_name}"
                        )
                with torch.no_grad():
                    elbo_by_dataset: dict[str, float] = defaultdict(lambda: 0)
                    for dataset_name, dataloader in eval_dataloaders.items():
                        for eval_batch in take_n(dataloader, n=1):
                            samples = eval_batch["input"]

                            elbo = elbo_function(
                                denoise_model=unet,
                                noise_scheduler=noise_scheduler,
                                original_samples=samples,
                            )
                            elbo_by_dataset[f"{dataset_name}.elbo"] += (
                                elbo.mean().item()
                            )
                # Log to wandb:
                accelerator.get_tracker("wandb").log(
                    losses_by_dataset_by_timestep | elbo_by_dataset, step=step
                )

                # --- Generate sample images for visual inspection ---
                if vae is not None:
                    pipeline = DDPMPipelineRaw(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )
                else:
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )

                generator = torch.Generator(device=pipeline.device).manual_seed(42)
                # run pipeline in inference (sample random noise and denoise)
                if vae is not None:
                    pipeline: DDPMPipelineRaw
                    latents = pipeline(
                        generator=generator,
                        batch_size=config.dataloader.eval_batch_size,
                        num_inference_steps=config.diffusion.num_inference_steps,
                        postprocess=False,
                    )
                    images = vae_decode_from_latent(latents)
                    # Post-process the images
                    images = (images / 2 + 0.5).clamp(0, 1)
                    images = images.permute(0, 2, 3, 1)  # Channel last
                    # To numpy
                    images = images.cpu().numpy()
                else:
                    pipeline: DDPMPipeline
                    pipeline_output: ImagePipelineOutput = pipeline(
                        generator=generator,
                        batch_size=config.dataloader.eval_batch_size,
                        num_inference_steps=config.diffusion.num_inference_steps,
                        output_type="numpy",
                    )
                    images = pipeline_output.images

                # Denormalize the images and save
                images_processed = (images * 255).round().astype("uint8")

                # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                accelerator.get_tracker("wandb").log(
                    {"test_samples": [wandb.Image(img) for img in images_processed]},
                    step=step,
                )

                # --- Restore training parameters to model at end of testing ---
                if ema_model is not None:
                    ema_model.restore(unet.parameters())

    # --- Save the final diffusion pipeline to use for testing ---
    unet = accelerator.unwrap_model(model)
    unet.eval()
    # Load the EMA weights for saving the final model
    if ema_model is not None:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

    pipeline = DDPMPipelineRaw(
        unet=unet,
        scheduler=noise_scheduler,
    )

    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(exist_ok=True)
    # Pipeline doesn't include the VAE, so we don't have to worry about needlessly saving it
    pipeline.save_pretrained(pipeline_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main_with_wandb()
