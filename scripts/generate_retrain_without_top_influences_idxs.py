import logging
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import omegaconf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from pydantic import dataclasses

from diffusion_influence.config_schemas import (
    DatasetType,
)
from diffusion_influence.constructors import (
    construct_datasets,
)
from diffusion_influence.data_utils import load_samples_dataset_from_dir
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers


@dataclasses.dataclass
class GenerateRetrainWithoutTopIdxsConfig:
    num_influences_to_remove: int

    scores_path: Path
    """Path to the scores `.npy` file of shape `(N, M)` where `N` is the number of training samples
    and `M` is the number of samples for which the scores were computed."""

    sample_idx: int
    """Sample index to compute top influences for. If the number of samples for which the scores
    were computed is `M`, this should be in the range `[0, M)`."""

    samples_dir_path: Path
    """Path to the directory containing the samples for which the scores/influence was computed.
    Not strictly necessary for computing the top influences.
    This will be used for saving which sample `sample_idx` corresponds to, so that we avoid
    file order surprises."""

    maximise_measurement: bool
    """Set to False for ELBO, True for loss if you want top influences that act to decrease 
    the ELBO and increase the loss."""

    dataset_name: DatasetType
    """Dataset name used to determine the dataset length."""

    examples_idxs_path: Optional[Path]
    """Path to the file containing the sub-indices used for training the main model
    that we're interested in doing counterfactual retraining for."""

    dataset_cache_dir: str = "data"
    save_top_influece_images: bool = False


# Register custom resolvers for the config parser:
register_custom_resolvers()

# Register the structured dataclass config schema (for type-checking, autocomplete, and validation) with Hydra:
cs = ConfigStore.instance()
cs.store(name="base_schema", node=GenerateRetrainWithoutTopIdxsConfig)


@hydra.main(version_base=None, config_name="base_schema")
def main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: GenerateRetrainWithoutTopIdxsConfig = omegaconf.OmegaConf.to_object(
        omegaconf_config
    )  # type: ignore  # insufficient typing of to_object()

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logging.info(f"Output directory: {output_dir}")

    # --- Load the idxs: ---
    train_dataset, eval_datasets = construct_datasets(
        config.dataset_name, cache_dir=config.dataset_cache_dir
    )
    train_dataset_length = len(train_dataset)
    if config.examples_idxs_path is not None:
        train_subset_idxs = np.loadtxt(config.examples_idxs_path, dtype=int)
        if train_subset_idxs.max() > len(train_dataset):
            raise ValueError(
                f"Requested number of training examples "
                f"exceeds the number of examples in the dataset ({len(train_dataset)})"
            )
    else:
        train_subset_idxs = np.arange(train_dataset_length)

    # --- Load the scores & get top influences ---
    scores = np.load(config.scores_path)  # Shape [M (gen. samples), N (train. samples)]
    assert scores.shape[1] == len(train_subset_idxs), (
        f"Scores shape mismatch {scores.shape} at idx `1` != {len(train_subset_idxs)}"
    )

    scores_for_sample = scores[config.sample_idx, :]  # Shape [N]
    # Get the top influences
    top_influences_idxs = np.argsort(scores_for_sample)
    top_influences_idxs = (
        top_influences_idxs[::-1]
        if config.maximise_measurement
        else top_influences_idxs
    )
    top_influences_to_remove = top_influences_idxs[: config.num_influences_to_remove]
    top_influences_original_dataset_to_remove = train_subset_idxs[
        top_influences_to_remove
    ]

    assert set(top_influences_original_dataset_to_remove).issubset(
        set(train_subset_idxs)
    ), "Top influences to remove are not in the training subset"
    # --- Save the retrain indices without the top influences ---
    retrain_idxs_without_top_influences = np.setdiff1d(
        train_subset_idxs, top_influences_original_dataset_to_remove
    )
    assert (
        len(retrain_idxs_without_top_influences)
        == len(train_subset_idxs) - config.num_influences_to_remove
    ), "Length mismatch"

    filename = output_dir / "idx_train.csv"
    np.savetxt(filename, retrain_idxs_without_top_influences, fmt="%d")

    # --- Save the top influences for visualisation purposes ---
    if config.save_top_influece_images:
        top_influences_dir = output_dir / "top_influences"
        top_influences_dir.mkdir(exist_ok=True)
        for i, dataset_idx in enumerate(top_influences_original_dataset_to_remove):
            dataset_idx = int(dataset_idx)  # int() necessary to convert from np.int64
            sample_to_remove_influences_for = train_dataset[dataset_idx]
            sample_to_remove_influences_for_path = (
                top_influences_dir / f"influence{i:05d}_{dataset_idx}.png"
            )
            sample_to_remove_influences_for["img"].save(
                sample_to_remove_influences_for_path
            )
            # logging.info(f"Saved influence {i} for dataset index {dataset_idx} to {sample_to_remove_influences_for_path}")

        # --- Save the sample to remove influences for for visualisation purposes ---
        dataset = load_samples_dataset_from_dir(config.samples_dir_path, lambda x: x)
        sample_to_remove_influences_for = dataset[config.sample_idx]
        sample_to_remove_influences_for_path = output_dir / "sample.png"
        sample_to_remove_influences_for["input"].save(
            sample_to_remove_influences_for_path
        )


if __name__ == "__main__":
    main()
