from pathlib import Path

import hydra
import numpy as np
import omegaconf
from hydra.core.config_store import ConfigStore
from pydantic import dataclasses

from diffusion_influence.config_schemas import (
    DatasetType,
)
from diffusion_influence.constructors import (
    construct_datasets,
)
from diffusion_influence.omegaconf_resolvers import register_custom_resolvers


@dataclasses.dataclass
class GenerateRetrainIdxsConfig:
    seed: int
    core_subsample_size: int
    """
    The number of examples to use for the `core` training set with "all" training samples.
    If smaller than the corresponding dataset, allows for subsampling the dataset to reduce evaluation cost.
    """
    retrain_subsample_size: int
    """How many examples in the dataset size used for retraining for linear-datamodelling-score evaluation."""
    num_validation_subsamples: int
    """How many subsamples to use for validation evaluation."""
    num_subsampled_datasets: int

    dataset_name: DatasetType
    idxs_save_path: Path
    data_cache_dir: str = "data"


# Register custom resolvers for the config parser:
register_custom_resolvers()

# Register the structured dataclass config schema (for type-checking, autocomplete, and validation) with Hydra:
cs = ConfigStore.instance()
cs.store(name="base_schema", node=GenerateRetrainIdxsConfig)


@hydra.main(version_base=None, config_name="base_schema")
def main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: GenerateRetrainIdxsConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore  # insufficient typing of to_object()

    assert config.retrain_subsample_size < config.core_subsample_size, (
        "The retrain subsample size must be smaller than the core subsample size."
    )

    train_dataset, eval_datasets = construct_datasets(
        dataset_name=config.dataset_name, cache_dir=config.data_cache_dir
    )

    assert len(train_dataset) >= config.core_subsample_size, (
        "The core subsample size must be smaller than the dataset size."
    )

    # Make the necessary directories
    config.idxs_save_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(config.seed)

    # Subsample the evaluation datasets
    valid_idxs = np.random.choice(
        len(eval_datasets["test"]), size=config.num_validation_subsamples, replace=False
    )
    valid_idxs = np.sort(valid_idxs)
    filename = config.idxs_save_path / "idx_val.csv"
    np.savetxt(filename, valid_idxs, fmt="%d")

    # Subsample the training dataset
    train_idxs = np.random.choice(
        len(train_dataset), size=config.core_subsample_size, replace=False
    )
    train_idxs = np.sort(train_idxs)
    filename = config.idxs_save_path / "idx_train.csv"
    np.savetxt(filename, train_idxs, fmt="%d")

    for k in range(config.num_subsampled_datasets):
        # Subsample the original `train_idxs`
        subsampled_train_idxs = np.random.choice(
            train_idxs, size=config.retrain_subsample_size, replace=False
        )
        subsampled_train_idxs = np.sort(subsampled_train_idxs)
        retrain_dir = config.idxs_save_path / "retrain"
        retrain_dir.mkdir(exist_ok=True)
        filename = retrain_dir / f"sub_idx_{k}.csv"
        np.savetxt(filename, subsampled_train_idxs, fmt="%d")


if __name__ == "__main__":
    main()
