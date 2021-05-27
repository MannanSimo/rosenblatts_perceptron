from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Tuple

from perceptron import Perceptron


class CliApplication:
    """Representation of command line application."""

    markers: Tuple[str] = ('s', 'x', 'o', '^', 'v')
    colors: Tuple[str] = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    def __init__(self, show: bool=False, logger=logger) -> None:
        """Return an object of command line application entity.

        Args:
            show (bool, optional): shows plots. Defaults to False.
            logger ([type], optional): logger entity. Defaults to logger.
        """
        self._iris_df: pd.DataFrame = None
        self._x: np.ndarray = None
        self._y: np.ndarray = None
        self._show: bool = show
        self._classifier: Perceptron = None
        self._logger = logger

        self._logger.info(f"CliApplication object {id(self)} is created.")

    def _get_iris_data_set(self) -> pd.DataFrame:
        """Return Iris data frame.

        Returns:
            pd.DataFrame: date frame with Iris information.
        """
        return pd.read_csv(
            'https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data',
            header=None
        )

    def _select_training_data(self) -> None:
        """Select Setosa and Versicolor information."""
        # select setosa and versicolor
        self._y: np.ndarray = self._iris_df.iloc[0:100, 4].values
        self._y: np.ndarray = np.where(self._y == 'Iris-setosa', -1, 1)
        # extract sepal length and petal length
        self._x: np.ndarray = self._iris_df.iloc[0:100, [0, 2]].values

        plt.scatter(
            self._x[:50, 0],
            self._x[:50, 1],
            color='red',
            marker='o',
            label='setosa'
        )

        plt.scatter(
            self._x[50:100, 0],
            self._x[50:100, 1],
            color='blue',
            marker='x',
            label='versicolor'
        )

        plt.title(label="Data", fontsize=16)
        plt.xlabel('sepal length in cm')
        plt.ylabel('petal length in cm')
        plt.legend(loc='upper left')
        plt.savefig('./1_data.png', dpi=300)
        if self._show:
            plt.show()

    def _perform_training(self) -> None:
        """Execute training process or Rosenblatt's perceptron."""
        self._classifier = Perceptron(eta=0.1, n_iter=10, logger=self._logger)

        self._classifier.fit(self._x, self._y)

        plt.plot(
            range(1, len(self._classifier.errors) + 1),
            self._classifier.errors, marker='o'
        )
        plt.title(label="Training info", fontsize=16)
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.savefig('./2_perceptron_training.png', dpi=300)
        if self._show:
            plt.show()

    def _identify_decision_making_region(self, resolution: float=0.02):
        """Perform identification of decision making region of perceptron.

        Args:
            resolution (float, optional): resolution. Defaults to 0.02.
        """
        self.markers = ('s', 'x', 'o', '^', 'v')
        self.colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(self.colors[:len(np.unique(self._y))])

        x1_min, x1_max = self._x[:, 0].min() - 1, self._x[:, 0].max() + 1
        x2_min, x2_max = self._x[:, 1].min() - 1, self._x[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
        )
        Z = self._classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(self._y)):
            plt.scatter(
                x=self._x[self._y == cl, 0],
                y=self._x[self._y == cl, 1],
                alpha=0.8, c=cmap(idx),
                edgecolor='black',
                marker=self.markers[idx],
                label=cl
            )

        plt.title(label="Classification", fontsize=16)
        plt.xlabel('sepal length in cm')
        plt.ylabel('petal length in cm')
        plt.legend(loc='upper left')

        plt.savefig('./3_perceptron_decision_refions.png', dpi=300)
        if self._show:
            plt.show()

    def run(self):
        """Encapsulate application's main workflow."""
        self._iris_df = self._get_iris_data_set()
        self._logger.info(f"Iris data set is initialized.")
        self._select_training_data()
        self._logger.info(f"Traning data is selected.")
        self._perform_training()
        self._logger.info(f"Model is trained.")
        self._identify_decision_making_region()
        self._logger.info(f"Decision making area is identified.")
        self._logger.info(f"Application {id(self)} is finished.")
