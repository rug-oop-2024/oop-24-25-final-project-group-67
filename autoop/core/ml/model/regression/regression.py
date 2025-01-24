from autoop.core.ml.model import Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression (MLR) model.

    Does linear regression using the closed-form solution.
    """

    def __init__(self):
        """
        Initialize the MultipleLinearRegression model.

        Attributes:
            coefficients (np.ndarray): Coefficients of the fitted linear model.
        """
        super().__init__(name="MultipleLinearRegression", model_type="regression")
        self.coefficients = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear model to the data using the normal equation.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coefficients = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted linear model.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If the model has not been fitted before calling `predict`.
        """
        if self.coefficients is None:
            raise ValueError("Model is not fitted yet. Call `fit` before making predictions.")
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_with_intercept @ self.coefficients


class RandomForestRegressorModel(Model):
    """
    Random Forest Regressor model.

    Uses scikit-learn's RandomForestRegressor implementation.
    """

    def __init__(self, parameters=None):
        """
        Initialize the RandomForestRegressorModel.

        Args:
            parameters (dict, optional): Hyperparameters for the Random Forest model. Defaults to None.
        """
        super().__init__(name="RandomForestRegressor", model_type="regression", parameters=parameters)
        self.fitted_model = RandomForestRegressor(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest model to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training target values.
        """
        self.fitted_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self.fitted_model.predict(X)


class SVRModel(Model):
    """
    Support Vector Regressor (SVR) model.

    Uses scikit-learn's SVR implementation.
    """

    def __init__(self, parameters=None):
        """
        Initialize the SVRModel.

        Args:
            parameters (dict, optional): Hyperparameters for the SVR model. Defaults to None.
        """
        super().__init__(name="SVR", model_type="regression", parameters=parameters)
        self.fitted_model = SVR(**self.parameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVR model to the training data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training target values.
        """
        self.fitted_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input features.

        Args:
            X (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        return self.fitted_model.predict(X)
