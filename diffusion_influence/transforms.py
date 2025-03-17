import torch
from torch import Tensor


class UniformDequantize(torch.nn.Module):
    """Dequantize a tensor image in [0, 1] with uniform noise.
    This transform does not support PIL Image.

    Given an image with pixel values in [0, 1], this transform adds uniform noise
    such that the output is in [0, 1).

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be dequantized. Assumes input is within
                range [0, 1], where 0 corresponds to pixel intensity 0, and 1 corresponds
                to pixel intensity 255.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return (torch.rand_like(tensor) + tensor * 255.0) / 256.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
