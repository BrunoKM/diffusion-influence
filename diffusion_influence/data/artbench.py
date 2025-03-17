from pathlib import Path

import pandas as pd
from torchvision import transforms

from diffusion_influence.data.cifar import TransformType

METADATA_FILENAME = "ArtBench-10.csv"
DATA_DIR_NAME = "artbench-10-imagefolder-split"
DATASET_SUGGEST_DOWNLOAD_MSG = (
    "Please download the dataset with `scripts/download_artbench.py`"
)
DATASET_SHAPE: tuple[int, ...] = (3, 256, 256)


def get_base_transforms() -> tuple[TransformType, TransformType]:
    return (
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    )


def get_train_only_transforms() -> tuple[transforms.RandomHorizontalFlip, ...]:
    return (transforms.RandomHorizontalFlip(),)


def get_transforms() -> tuple[tuple[TransformType, ...], tuple[TransformType, ...]]:
    base_transforms = get_base_transforms()
    train_transforms = (
        *get_train_only_transforms(),
        *base_transforms,
    )
    eval_transforms = base_transforms
    return train_transforms, eval_transforms


def validate_integrity(root: str | Path):
    metadata_file = Path(root) / METADATA_FILENAME
    if not metadata_file.exists():
        return FileNotFoundError(
            f"Metadata file {metadata_file} not found. {DATASET_SUGGEST_DOWNLOAD_MSG}"
        )
    data_dir = Path(root) / DATA_DIR_NAME
    if not data_dir.exists() or not data_dir.is_dir():
        return FileNotFoundError(
            f"Data directory {data_dir} not found. {DATASET_SUGGEST_DOWNLOAD_MSG}"
        )
    # Load the metadata file:
    metadata = pd.read_csv(metadata_file)
    # Verify that every image in the metadata file exists at expected location:
    for _, row in metadata.iterrows():
        filename = row["name"]
        split = row["split"]
        label = row["label"]
        image_path = data_dir / split / label / filename
        if not image_path.exists():
            return FileNotFoundError(
                f"Image file {image_path} is missing. {DATASET_SUGGEST_DOWNLOAD_MSG}"
            )
