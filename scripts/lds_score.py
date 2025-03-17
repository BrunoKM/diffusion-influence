import logging
import os

import hydra
import omegaconf
from hydra.core.config_store import ConfigStore
from scipy.stats import sem

from diffusion_influence.config_schemas import LDSScoreConfig
from diffusion_influence.lds import (
    compute_lds_score_from_config,
)

# Register the structured dataclass config schema with Hydra
# (for type-checking, autocomplete, and validation)
cs = ConfigStore.instance()
cs.store(name="base_schema", node=LDSScoreConfig)


@hydra.main(version_base=None, config_path="configs")
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: LDSScoreConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


def main(config: LDSScoreConfig):
    (
        rank_correlations,
        rank_correlations_across_seeds_array,
        rank_correlations_across_averaged_seeds_array,
    ) = compute_lds_score_from_config(config)

    logging.info(f"Rank correlation shape: {rank_correlations.shape}")
    logging.info(f"Rank correlation mean: {rank_correlations.mean()}")
    logging.info(f"Rank correlation stde: {sem(rank_correlations)}")

    logging.info(
        f"Accross seeds rank correlation mean: {rank_correlations_across_seeds_array.mean()}"
    )
    logging.info(
        f"Accross seeds rank correlation stde: {sem(rank_correlations_across_seeds_array)}"
    )

    logging.info(
        f"Accross seeds (to other averaged) rank correlation mean: {rank_correlations_across_averaged_seeds_array.mean()}"
    )
    logging.info(
        f"Accross seeds (to other averaged) rank correlation stde: {sem(rank_correlations_across_averaged_seeds_array)}"
    )


if __name__ == "__main__":
    hydra_main()
