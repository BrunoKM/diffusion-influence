from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TypeVar

import numpy as np
import torch
from datasets import Array4D, Dataset, Features, Image

T = TypeVar("T")


def ad_infinitum(i: Iterable[T]) -> Iterator[T]:
    while True:
        yield from i


def take_n(i: Iterable[T], n: Optional[int]) -> Iterator[T]:
    """
    Take the first `n` elements from the iterable `i`. If n is None, iterate over all elements.

    The resulting iterator is always finite.
    """
    assert n is None or n > 0
    if n is None:
        try:
            # If n is None, then we want to iterate over the entire iterable.
            n = len(i)
        except TypeError as e:
            # If the iterable does not have a length, then `n` must be specified (not None)
            raise ValueError(
                "If the iterable does not have a length, then `n` must be specified (not None), otherwise "
                "resulting iterator might be infinite"
            ) from e

    yield from (x for _, x in zip(range(n), i))


def load_samples_dataset_from_dir(samples_dir: Path, transforms: Callable) -> Dataset:
    if not samples_dir.exists():
        raise ValueError(f"Directory {samples_dir} does not exist")
    elif not samples_dir.is_dir():
        raise ValueError(f"Directory {samples_dir} is not a directory")

    # Get all the file paths in the samples directory
    samples_paths: list[str] = sorted([str(path) for path in samples_dir.glob("*.png")])

    dataset = Dataset.from_dict({"img": samples_paths}).cast_column("img", Image())

    def transform_images_fn(examples):
        images = [transforms(image.convert("RGB")) for image in examples["img"]]
        return {"input": images}

    dataset.set_transform(transform_images_fn)
    return dataset


def load_samples_and_trajectories_dataset_from_dir(
    samples_dir: Path, transforms: Callable
) -> Dataset:
    if not samples_dir.exists():
        raise ValueError(f"Directory {samples_dir} does not exist")
    elif not samples_dir.is_dir():
        raise ValueError(f"Directory {samples_dir} is not a directory")

    # Get all the file paths in the samples directory
    samples_paths: list[str] = sorted([str(path) for path in samples_dir.glob("*.png")])
    trajectories_paths: list[str] = [
        samples_path.replace(".png", "_trajectory.npy")
        for samples_path in samples_paths
    ]
    for i in range(len(samples_paths)):
        assert Path(trajectories_paths[i]).exists(), (
            f"Path {trajectories_paths[i]} does not exist."
        )
        assert Path(samples_paths[i]).exists(), (
            f"Path {samples_paths[i]} does not exist."
        )
    # Load all the trajectories
    trajectories = [np.load(trajectory_path) for trajectory_path in trajectories_paths]
    trajectory_shape = trajectories[0].shape

    features = Features(
        img=Image(),
        trajectory=Array4D(shape=trajectory_shape, dtype="float32"),
    )
    dataset = Dataset.from_dict(
        {"img": samples_paths, "trajectory": trajectories},
        features=features,
    )

    def transform_images_fn(examples):
        images = [transforms(image.convert("RGB")) for image in examples["img"]]
        # Map to tensor
        trajectories = [
            torch.tensor(trajectory) for trajectory in examples["trajectory"]
        ]
        return {"input": images, "trajectory": trajectories}

    dataset.set_transform(transform_images_fn)
    return dataset
