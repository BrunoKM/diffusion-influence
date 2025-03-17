"""Quantization utilities for PyTorch tensors."""

import math
from typing import NamedTuple

import torch
from torch import Size, Tensor


class QuantizedTensor(NamedTuple):
    tensor: Tensor
    shape: Size
    scale: float
    zero_point: float
    bits: int


def _uint8_to_uint4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8
    assert x.shape[0] % 2 == 0, f"{x.shape}"  # could use any dimension of even size

    x1 = (x[1::2] & 0xF) << 4
    x2 = x[::2] & 0xF

    return x1 | x2


def _uint4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.uint8

    x1 = x & 0xF
    x2 = (x >> 4) & 0xF

    x = torch.empty(x.shape[0] * 2, *x.shape[1:], dtype=torch.uint8)
    x[::2], x[1::2] = x1, x2

    return x


def quantize(tensor: Tensor, bits: int) -> QuantizedTensor:
    """Quantize a tensor using affine quantization

    Args:
        tensor (Tensor): tensor to quantize
        bits (int): Number of bits after quantization (4 or 8)

    Returns:
        tuple[Tensor, float, float, int]: quantized tensor, scale, zero_point, bits
    """

    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Expected tensor to be a torch.Tensor, but got {type(tensor)}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(
            f"Expected tensor to have dtype torch.float32, but got {tensor.dtype}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Tensor contains NaN or Inf values.")

    shape = tensor.shape

    tensor = tensor.ravel()

    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    q_min = 0
    q_max = 2**bits - 1

    scale = (max_val - min_val) / (q_max - q_min)
    zero_point = q_min - min_val / scale

    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, q_min, q_max).to(torch.uint8)

    # uint4 is implemented by packing two uint8 values into a single uint8 value
    UINT4 = 4
    if bits == UINT4:
        if q_tensor.shape[0] % 2 != 0:
            q_tensor = torch.cat([q_tensor, torch.zeros(1, dtype=torch.uint8)])
        q_tensor = _uint8_to_uint4(q_tensor)

    return QuantizedTensor(q_tensor, shape, scale, zero_point, bits)


def dequantize(q_tensor: QuantizedTensor) -> Tensor:
    """Dequantize a tensor using affine quantization

    Args:
       q_tensor (QuantizedTensor): Quantized tensor

    Returns:
       dequantized float32 Tensor
    """

    tensor = q_tensor.tensor
    shape = q_tensor.shape
    scale = q_tensor.scale
    zero_point = q_tensor.zero_point
    bits = q_tensor.bits

    UINT4 = 4
    if bits == UINT4:
        tensor = _uint4_to_uint8(tensor)
        if tensor.numel() != math.prod(shape):
            tensor = tensor[:-1]

    tensor = (tensor.to(torch.float32) - zero_point) * scale
    return tensor.view(shape)
