
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data = dataset.read()
    features = []
    for column_name in data.columns:
        column_data = data[column_name]
        if column_data.dtypes in ['object', 'category']:
            feature_type = "categorical"
        else:
            feature_type = "numerical"
        features.append(Feature(name=column_name, type=feature_type))
    return features