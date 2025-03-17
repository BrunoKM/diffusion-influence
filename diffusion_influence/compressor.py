import abc
from typing import Generic, Sequence, TypeVar

from torch import Tensor

from diffusion_influence.quantization import QuantizedTensor, dequantize, quantize
from diffusion_influence.svd_compression import (
    QSVDCompressedTensor,
    SVDCompressedTensor,
    svd_compress_weight,
    svd_decompress_weight,
)

T = TypeVar("T")


class WeightCompressor(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def compress(self, parameters: Sequence[Tensor]) -> Sequence[T]: ...

    @abc.abstractmethod
    def decompress(self, compressed_parameters: Sequence[T]) -> Sequence[Tensor]: ...


class IdentityCompressor(WeightCompressor[Tensor]):
    """A compressor that improves storage by as much as I've improved my productivity over the past year."""

    def compress(self, parameters: Sequence[Tensor]) -> Sequence[Tensor]:
        return parameters

    def decompress(self, compressed_parameters: Sequence[Tensor]) -> Sequence[Tensor]:
        return compressed_parameters


class SVDCompressor(WeightCompressor[SVDCompressedTensor]):
    def __init__(self, rank: int) -> None:
        """
        Args:
            rank (int): Maximum rank for the low-rank approximation to weight-matrices
        """
        super().__init__()
        self.rank = rank

    def compress(self, parameters: Sequence[Tensor]) -> list[SVDCompressedTensor]:
        return [svd_compress_weight(weight, self.rank) for weight in parameters]

    def decompress(
        self, compressed_parameters: Sequence[SVDCompressedTensor]
    ) -> list[Tensor]:
        return [
            svd_decompress_weight(compressed_weight)
            for compressed_weight in compressed_parameters
        ]


class QuantizationCompressor(WeightCompressor[QuantizedTensor]):
    def __init__(self, bits: int = 8) -> None:
        """
        Args:
            bits (int): Number of bits after quantization (4 or 8)
        """
        super().__init__()
        if bits not in [4, 8]:
            raise ValueError(f"bits must be 4 or 8, but got: {bits}")
        self.bits = bits

    def compress(self, parameters: Sequence[Tensor]) -> list[QuantizedTensor]:
        return [quantize(weight, self.bits) for weight in parameters]

    def decompress(
        self, compressed_parameters: Sequence[QuantizedTensor]
    ) -> list[Tensor]:
        return [
            dequantize(compressed_weight) for compressed_weight in compressed_parameters
        ]


class QSVDCompressor(WeightCompressor[QSVDCompressedTensor]):
    def __init__(self, rank: int = 100, bits: int = 8) -> None:
        """
        Args:
            rank (int): Maximum rank for the low-rank approximation to weight-matrices
            bits (int): Number of bits after quantization (4 or 8)
        """
        super().__init__()
        if bits not in [4, 8]:
            raise ValueError(f"bits must be 4 or 8, but got: {bits}")

        self.rank = rank
        self.bits = bits

    def _compress_tensor(self, tensor: Tensor) -> QSVDCompressedTensor:
        svd_tensor = svd_compress_weight(tensor, self.rank)
        qsvd_tensor = QSVDCompressedTensor(
            quantize(svd_tensor.US, self.bits),
            quantize(svd_tensor.Vh, self.bits) if svd_tensor.Vh is not None else None,
            svd_tensor.original_shape,
        )
        return qsvd_tensor

    def compress(self, parameters: Sequence[Tensor]) -> list[QSVDCompressedTensor]:
        return [self._compress_tensor(weight) for weight in parameters]

    def _decompress_tensor(self, compressed_tensor: QSVDCompressedTensor) -> Tensor:
        svd_tensor = SVDCompressedTensor(
            dequantize(compressed_tensor.US),
            dequantize(compressed_tensor.Vh)
            if compressed_tensor.Vh is not None
            else None,
            compressed_tensor.original_shape,
        )
        return svd_decompress_weight(svd_tensor)

    def decompress(
        self, compressed_parameters: Sequence[QSVDCompressedTensor]
    ) -> list[Tensor]:
        return [
            self._decompress_tensor(compressed_weight)
            for compressed_weight in compressed_parameters
        ]
