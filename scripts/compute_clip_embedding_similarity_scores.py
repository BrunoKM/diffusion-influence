import logging
import os
from pathlib import Path
from typing import Optional

import diffusers
import hydra
import numpy as np
import omegaconf
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from pydantic.dataclasses import dataclass
from torch import cuda, device
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

from diffusion_influence.config_schemas import (
    DataConfig,
    DataLoaderConfig,
)
from diffusion_influence.constructors import (
    construct_and_subsample_datasets,
)
from diffusion_influence.data_utils import load_samples_dataset_from_dir
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers


@dataclass
class ScoreWithCLIPEmbeddingConfig:
    data: DataConfig
    # pretrained_model_config_path: Path
    """Used for loading in information about the training data. This is the path
    to the config.yaml file with training settings for the model, usually saved
    to the .hydra directory in the output directory of the training script."""

    num_train_augment_samples: Optional[int]
    """How many samples to use for estimating data augmentation."""

    samples_dir_path: Path
    """Path to the directory containing the samples to score the influence for."""
    dataloader: DataLoaderConfig


# Check diffusers version sufficiently high
diffusers.utils.check_min_version("0.16.0")


DEVICE = device("cuda" if cuda.is_available() else "cpu")


# Register custom resolvers for the config parser:
register_custom_resolvers()
# Register the structured dataclass config schema with Hydra
# (for type-checking, autocomplete, and validation)
cs = ConfigStore.instance()
cs.store(name="base_schema", node=ScoreWithCLIPEmbeddingConfig)


@hydra.main(version_base=None, config_path="configs", config_name="base_schema")
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: ScoreWithCLIPEmbeddingConfig = omegaconf.OmegaConf.to_object(
        omegaconf_config
    )  # type: ignore
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


def main(config: ScoreWithCLIPEmbeddingConfig):
    # --- Construct the CLIP model
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Prepare the dataset.
    train_dataset, eval_datasets = construct_and_subsample_datasets(
        dataset_name=config.data.dataset_name,
        cache_dir=config.data.cache_dir,
        examples_idxs_path=config.data.examples_idxs_path,
    )
    if config.num_train_augment_samples is None:
        train_dataset = eval_datasets["train_no_augment"]

    # Construct query and train datasets
    query_dataset = load_samples_dataset_from_dir(
        config.samples_dir_path, transforms=lambda x: x
    )

    # Print info on the model:
    logging.info(
        f"CLIP Model parameters: {sum([int(np.prod(p.shape)) for p in model.parameters()])}"
    )

    # --- Compute the scores
    # Compute the query embeddings
    query_embeddings = []
    for i in tqdm(range(0, len(query_dataset), config.dataloader.eval_batch_size)):
        query_batch = clip_processor(
            images=query_dataset[i : i + config.dataloader.eval_batch_size]["input"],
            return_tensors="pt",
        )
        with torch.no_grad():
            query_embedding = model(**query_batch).pooler_output
            # Normalise the embeddings
            query_embedding /= query_embedding.norm(p=2, dim=-1, keepdim=True)
        query_embeddings += [query_embedding]
    # Compute the train embeddings
    num_samples_per_example = config.num_train_augment_samples or 1
    train_embeddings = []
    for i in tqdm(range(0, len(train_dataset), config.dataloader.eval_batch_size)):
        train_embedding_samples = 0
        for sample in range(num_samples_per_example):
            train_batch = clip_processor(
                images=train_dataset[i : i + config.dataloader.eval_batch_size]["img"],
                return_tensors="pt",
            )
            with torch.no_grad():
                train_embedding_sample = model(**train_batch).pooler_output
                # Normalise the embeddings
                train_embedding_sample /= train_embedding_sample.norm(
                    p=2, dim=-1, keepdim=True
                )
                train_embedding_samples += train_embedding_sample
        train_embedding = train_embedding_samples / num_samples_per_example
        train_embeddings += [train_embedding]
    query_embeddings = torch.cat(query_embeddings, dim=0)
    train_embeddings = torch.cat(train_embeddings, dim=0)

    scores = torch.einsum(
        "qd,td->qt", query_embeddings, train_embeddings
    )  # [num_validation_samples, num_train_data]
    logging.info(f"Scores shape: {scores.shape}")
    # --- Save the scores to a file
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    scores_path = output_dir / "scores.npy"
    np.save(file=scores_path, arr=scores.cpu().numpy())


if __name__ == "__main__":
    hydra_main()
