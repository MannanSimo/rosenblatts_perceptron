"""Provide abstraction for Frank Rosenblatt's Perceptron."""

import loguru
import numpy as np
from typing import Optional, Tuple


class Perceptron:
    """Provide representation of Perceptron entity."""

    def __init__(
            self,
            eta: float=0.01,
            n_iter: int=10,
            shuffle: bool=True,
            random_state: Optional[int]=None,
            logger: loguru._logger.Logger=None
        ) -> None:
        """Return an Perceptron object.

        Args:
            eta (float, optional): learning rate. Defaults to 0.01.
            n_iter (int, optional):  iterations over training. Defaults to 10.
            shuffle (bool, optional): indicate if shuffle should be done. Defaults to True.
            random_state ([int], optional): seed for random generator. Defaults to None.

        Returns:
            self: Perceptron object
        """
        self._eta: float = eta
        self._n_iter: int = n_iter
        self.__shuffle: bool = shuffle
        self._weights: np.ndarray = None
        self._errors: Optional[list] = None
        self._logger = logger
        if random_state:
            np.random.seed(random_state)

        self._logger.info(f"Perceptron object {id(self)} is created.")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Perform model training.

        Args:
            x (ndarray): array of traning vectors
            y (ndarray): array of target values

        Returns:
            self: Perceptron object
        """
        self._weights = np.zeros(1 + x.shape[1])
        self._errors = []

        self._logger.info(f"Perceptron {id(self)} training is started.")
        for i in range(self._n_iter):
            if self.__shuffle:
                x, y = self._shuffle(x, y)
            errors = 0
            for xi, target in zip(x, y):
                update = self._eta * (target - self.predict(xi))
                self._weights[1:] += update * xi
                self._weights[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
            self._logger.info(f"Iteration {i+1} is done.")
        self._logger.info(f"Perceptron {id(self)} training is finished.")
        return self

    def _shuffle(
            self,
            x: np.ndarray,
            y: np.ndarray
            )-> Tuple[np.ndarray, np.ndarray]:
        """[summary]

        Args:
            x (np.ndarray): training vectors
            y (np.ndarray): target vectors

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def net_input(self, x: np.ndarray) -> np.ndarray:
        """Calculate net input.

        Args:
            x ([np.ndarray]): training vectors

        Returns:
            ndarray: input
        """
        return np.dot(x, self._weights[1:]) + self._weights[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return class label after unit step.

        Args:
            x (np.ndarray): training vectors

        Returns:
            ndarray: prediction vector
        """
        return np.where(self.net_input(x) >= 0.0, 1, -1)

    @property
    def errors(self) -> np.ndarray:
        """Return error vector.

        Returns:
            np.ndarray: errors vector
        """
        return self._errors

    @property
    def eta(self) -> float:
        """Return learning rate.

        Returns:
            float: learning rate.
        """
        return self._eta

    @property
    def n_iter(self) -> int:
        """Return number of iteration should be performed over training data.

        Returns:
            int: iteration over training data set
        """
        return self._n_iter
