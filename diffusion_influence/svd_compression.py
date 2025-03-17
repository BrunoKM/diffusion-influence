"""Quantization utilities for PyTorch tensors."""

from typing import NamedTuple, Optional

import torch
from torch import Size, Tensor

from diffusion_influence.quantization import QuantizedTensor


class SVDCompressedTensor(NamedTuple):
    US: Tensor
    Vh: Optional[Tensor]  # None implies Vh is identity (not required, e.g. for vectors)
    original_shape: Optional[Size]


class QSVDCompressedTensor(NamedTuple):
    US: QuantizedTensor
    Vh: Optional[QuantizedTensor]
    original_shape: Optional[Size]


def svd_compress_matrix(matrix: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    # Get the top `rank` singular values and vectors
    # `.clone()` is needed to avoid memory leaks
    # Note: `.contiguous()` will not work, because the slice might
    # already be a contiguous view, and the `torch` garbage collector
    # isn't clever enough to detect only a slice of a huge array is needed
    U = U[:, :rank]
    S = S[:rank]
    US = U @ torch.diag_embed(S)
    Vh = Vh[:rank, :].clone()
    return US, Vh


def svd_compress_weight(
    tensor: Tensor,
    rank: int,
) -> SVDCompressedTensor:
    """

    Args:
        tensor (Tensor):
        dtype (torch.dtype):

    Returns:
        tuple[Tensor, float, float]: quantized tensor, scale, zero_point
    """
    if not isinstance(tensor, Tensor):
        raise ValueError("Input is not a Tensor.")
    if not torch.isfinite(tensor).all():
        raise ValueError("Input tensor contains NaN or Inf values.")

    match tensor.shape:
        case [dim]:  # noqa: F841
            # For biases etc. we don't compress
            return SVDCompressedTensor(US=tensor, Vh=None, original_shape=None)
        case [dim_out, dim_in]:
            US, Vh = svd_compress_matrix(tensor, rank)
            return SVDCompressedTensor(US=US, Vh=Vh, original_shape=None)
        case [dim_out, dim_in, kernel_size_1, kernel_size_2]:
            # Assume that the tensor is a convolutional weight tensor
            linear_weight_tensor_view = tensor.view(
                dim_out, dim_in * kernel_size_1 * kernel_size_2
            )
            US, Vh = svd_compress_matrix(linear_weight_tensor_view, rank)
            return SVDCompressedTensor(US=US, Vh=Vh, original_shape=tensor.size())
        case _:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def svd_decompress_weight(
    tensor: SVDCompressedTensor,
) -> Tensor:
    US, Vh = tensor.US, tensor.Vh
    # Compute the full matrix from the SVD
    USVh = (US @ Vh) if Vh is not None else US
    # Reshape the matrix if original shape is provided
    USVh = (
        USVh.view(tensor.original_shape) if tensor.original_shape is not None else USVh
    )
    return USVh
