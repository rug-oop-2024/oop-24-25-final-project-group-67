
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    """
    Represent a feature in a dataset with name and type.

    Attributes:
        name: Name of the feature.
        type: Type of the feature.
    """
    name: str = Field(..., description="Name of the feature.")
    type: Literal["categorical", "numerical"] = Field(..., description="Type of the feature.")

    def __str__(self):
        """
        Returns a string representation of the feature.
        """
        return f"Feature(name={self.name}, type={self.type})"