import logging
import os
from pathlib import Path

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
    DataLoaderConfig,
)
from diffusion_influence.data_utils import load_samples_dataset_from_dir
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers


@dataclass
class ScoreWithCLIPEmbeddingConfig:
    samples_dir_path1: Path
    """Path to the directory containing the samples to compute CLIP similarity for."""
    samples_dir_path2: Path
    """Path to the directory containing the samples to compute CLIP similarity to."""

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
    # train_dataset, eval_datasets = construct_and_subsample_datasets(
    #     dataset_name=config.data.dataset_name,
    #     cache_dir=config.data.cache_dir,
    #     examples_idxs_path=config.data.examples_idxs_path,
    # )
    # if config.num_train_augment_samples is None:
    #     train_dataset = eval_datasets["train_no_augment"]

    # Construct the two datasets to CLIP compare
    query_dataset1 = load_samples_dataset_from_dir(
        config.samples_dir_path1, transforms=lambda x: x
    )
    query_dataset2 = load_samples_dataset_from_dir(
        config.samples_dir_path2, transforms=lambda x: x
    )

    # Print info on the model:
    logging.info(
        f"CLIP Model parameters: {sum([int(np.prod(p.shape)) for p in model.parameters()])}"
    )

    # --- Compute the scores
    # Compute the query embeddings
    query_embeddings1 = []
    query_embeddings2 = []
    for query_embeddings, query_dataset in zip(
        [query_embeddings1, query_embeddings2], [query_dataset1, query_dataset2]
    ):
        for i in tqdm(range(0, len(query_dataset), config.dataloader.eval_batch_size)):
            query_batch = clip_processor(
                images=query_dataset[i : i + config.dataloader.eval_batch_size][
                    "input"
                ],
                return_tensors="pt",
            )
            with torch.no_grad():
                query_embedding = model(**query_batch).pooler_output
                # Normalise the embeddings
                query_embedding /= query_embedding.norm(p=2, dim=-1, keepdim=True)
            query_embeddings += [query_embedding]

    query_embeddings1 = torch.cat(query_embeddings1, dim=0)
    query_embeddings2 = torch.cat(query_embeddings2, dim=0)

    scores = torch.einsum(
        "qd,td->qt", query_embeddings1, query_embeddings2
    )  # [num_query_samples1, num_query_samples2]
    logging.info(f"Scores shape: {scores.shape}")
    # --- Save the scores to a file
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    scores_path = output_dir / "scores.npy"
    np.save(file=scores_path, arr=(scores_arr := scores.cpu().numpy()))
    # Save to .csv as well
    np.savetxt(output_dir / "scores.csv", scores_arr, delimiter=",")


if __name__ == "__main__":
    hydra_main()
