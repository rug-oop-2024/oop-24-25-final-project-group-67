from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(Artifact, ABC):
    """
    Abstract base class for machine learning models.

    Attributes:
        name (str): Name of the model.
        type (Literal["classification", "regression"]): Type of the model.
        parameters (dict): Model's hyperparameters.
    """
    def __init__(self, name: str, model_type: Literal["classification", "regression"], parameters: dict = None):
        self.name = name
        self.type = model_type
        self.parameters = parameters if parameters else {}
        self.fitted_model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the provided training data.

        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Target labels.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the fitted model.

        Args:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Predictions.
        """
        pass

    def save(self) -> dict:
        """
        Save the model parameters and fitted state.

        Returns:
            dict: A dictionary containing model state.
        """
        return deepcopy({
            "name": self.name,
            "type": self.type,
            "parameters": self.parameters,
            "fitted_model": self.fitted_model,
        })

    def load(self, state: dict) -> None:
        """
        Load the model parameters and fitted state.

        Args:
            state (dict): A dictionary containing model state.
        """
        self.name = state["name"]
        self.type = state["type"]
        self.parameters = state["parameters"]
        self.fitted_model = state["fitted_model"]