from typing import Literal, Union

from sklearn.gaussian_process.kernels import RBF, Matern


class KernelUtils:
    @staticmethod
    def create(kernel_type: Literal["rbf", "matern"]) -> Union[RBF, Matern]:
        if kernel_type == "rbf":
            return RBF()

        elif kernel_type == "matern":
            return Matern()
        
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")