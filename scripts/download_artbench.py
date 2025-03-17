import dataclasses
import logging
import os
import pathlib

import hydra
import omegaconf
from hydra.core.config_store import ConfigStore
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
)

from diffusion_influence.data.artbench import validate_integrity

TAR_URL = "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar"
TAR_FILENAME = "artbench-10-imagefolder-split.tar"
TAR_MD5 = "64f4d66dcf34b3c77e8c7d68fed2ec8c"
METADATA_URL = "https://artbench.eecs.berkeley.edu/files/ArtBench-10.csv"
METADATA_FILENAME = "ArtBench-10.csv"
METADATA_MD5 = "cc8e5df471b294f19f3fab19ad9f5100"


@dataclasses.dataclass
class DownloadConfig:
    root: pathlib.Path
    force_redownload: bool = False
    remove_tar_after_extract: bool = True


cs = ConfigStore.instance()
cs.store(name="base_schema", node=DownloadConfig)


@hydra.main(version_base=None, config_name="base_schema")
def hydra_main(omegaconf_config: omegaconf.DictConfig):
    # Parse the config into pydantic dataclasses (for validation)
    config: DownloadConfig = omegaconf.OmegaConf.to_object(omegaconf_config)  # type: ignore
    logging.info(f"Current working directory: {os.getcwd()}")
    main(config)


def main(config: DownloadConfig):
    logging.info(f"Downloading to: {config.root}")
    download_and_extract_archive(
        TAR_URL,
        download_root=config.root,
        extract_root=config.root,
        filename=TAR_FILENAME,
        md5=TAR_MD5,
        remove_finished=config.remove_tar_after_extract,
    )
    # Download the .csv metadata file
    download_url(
        url=METADATA_URL,
        root=config.root,
        filename=METADATA_FILENAME,
        md5=METADATA_MD5,
    )
    logging.info("Downloaded ArtBench dataset.")
    logging.info("Verifying integrity.")
    validate_integrity(config.root)
    logging.info("Integrity of ArtBench verified.")


if __name__ == "__main__":
    hydra_main()
