from typing import Callable

import torch
from torchvision import transforms

from diffusion_influence.transforms import UniformDequantize

DATASET_SHAPE: tuple[int, ...] = (3, 32, 32)


TransformType = Callable[[torch.Tensor], torch.Tensor]


def get_cifar10_base_transforms() -> tuple[TransformType, TransformType]:
    return (
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    )


def get_cifar10_train_only_transforms() -> tuple[transforms.RandomHorizontalFlip, ...]:
    return (transforms.RandomHorizontalFlip(),)


def get_cifar10_transforms() -> tuple[
    tuple[TransformType, ...], tuple[TransformType, ...]
]:
    base_transforms = get_cifar10_base_transforms()
    train_transforms = (
        *get_cifar10_train_only_transforms(),
        *base_transforms,
    )
    eval_transforms = base_transforms
    return train_transforms, eval_transforms


def get_cifar10deq_base_transforms() -> tuple[TransformType, ...]:
    return (
        transforms.ToTensor(),
        UniformDequantize(),
        transforms.Normalize((0.5,), (0.5,)),
    )


def get_cifar10deq_transforms() -> tuple[
    tuple[TransformType, ...], tuple[TransformType, ...]
]:
    """Get the transforms for the CIFAR10 dataset with uniformdequantization."""
    base_transforms = get_cifar10deq_base_transforms()
    train_transforms = (
        *get_cifar10_train_only_transforms(),
        *base_transforms,
    )
    eval_transforms = base_transforms
    return train_transforms, eval_transforms
