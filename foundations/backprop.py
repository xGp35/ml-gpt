import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array - [1,2]
        # w: 1D weight array - [1, 2]
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        z = np.dot(x,w) + b  # [1,2], b is broadcasted
        y_hat = sigmoid(z) # [1,2]
        dL_dz_unmerged = (y_hat-y_true)*y_hat*(1-y_hat) # common term - [1, 2]
        dL_dw = dL_dz_unmerged*x
        dL_db = np.sum(dL_dz_unmerged)

        return (np.round(dL_dw,5), np.round(dL_db, 5))
