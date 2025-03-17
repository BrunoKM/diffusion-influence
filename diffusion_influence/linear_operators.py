from typing import Protocol

from curvlinops import KFACInverseLinearOperator
from torch import Tensor

ParameterType = list[Tensor]


class ParameterPreconditioningLinearOperator(Protocol):
    def matvec(self, X: ParameterType) -> ParameterType: ...

    @property
    def params(self) -> ParameterType: ...


class KFACInverseLinearOperatorWrapper(ParameterPreconditioningLinearOperator):
    def __init__(self, inverse_kfac: KFACInverseLinearOperator):
        self.inverse_kfac = inverse_kfac

    def matvec(self, X: ParameterType) -> ParameterType:
        return self.inverse_kfac @ X

    @property
    def params(self) -> ParameterType:
        return self.inverse_kfac._A._params


class ConstantPreconditioner(ParameterPreconditioningLinearOperator):
    def __init__(self, const: float, params: ParameterType):
        self.const = const
        self._params = params

    @property
    def params(self):
        return self._params

    def matvec(self, X):
        """
        Apply the identity operator to a matrix.
        """
        return [self.const * x for x in X]


class Identity(ConstantPreconditioner):
    def __init__(self, params: ParameterType):
        super().__init__(const=1.0, params=params)

    def matvec(self, X):
        """Apply the identity operator."""
        return X
