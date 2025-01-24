from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from autoop.core.ml.model import Model
import numpy as np


class LogisticRegressionModel(Model):
    """
    Logistic Regression Model.

    Uses scikit-learn's LogisticRegression implementation.

    Attributes:
        parameters (dict): Hyperparameters for LogisticRegression.
    """
    def __init__(self, parameters=None):
        """
        Initialize the LogisticRegressionModel.

        Args:
            parameters (dict, optional): Hyperparameters for the Logistic Regression model. Defaults to None.
        """
        super().__init__(name="LogisticRegression", model_type="classification", parameters=parameters)
        self.fitted_model = LogisticRegression(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Logistic Regression model to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.fitted_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.fitted_model.predict(X)


class RandomForestClassifierModel(Model):
    """
    Random Forest Classifier.

    Uses scikit-learn's RandomForestClassifier implementation.

    Attributes:
        parameters (dict): Hyperparameters for RandomForestClassifier.
    """
    def __init__(self, parameters=None):
        """
        Initialize the RandomForestClassifierModel.

        Args:
            parameters (dict, optional): Hyperparameters for the Random Forest model. Defaults to None.
        """
        super().__init__(name="RandomForestClassifier", model_type="classification", parameters=parameters)
        self.fitted_model = RandomForestClassifier(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest model to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels of shape.
        """
        self.fitted_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given input features.

        Args:
            X (np.ndarray): Input features of shape.

        Returns:
            np.ndarray: Predicted labels of shape.
        """
        return self.fitted_model.predict(X)


class SVCModel(Model):
    """
    Support Vector Classifier (SVC) for classification tasks.

    Uses scikit-learn's SVC implementation.

    Attributes:
        parameters (dict): Hyperparameters for SVC.
    """
    def __init__(self, parameters=None):
        """
        Initialize the SVCModel.

        Args:
            parameters (dict, optional): Hyperparameters for the Support Vector Classifier. Defaults to None.
        """
        super().__init__(name="SVC", model_type="classification", parameters=parameters)
        self.fitted_model = SVC(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVC model to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.fitted_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for the given input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.fitted_model.predict(X)
