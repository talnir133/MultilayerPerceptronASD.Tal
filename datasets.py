import numpy as np
import itertools
import torch


def classification_rule(shape_name, n_types, deciding_feature):
    """
    Determines the label based on a specific feature's value.
    :param shape_name: Tuple of indices representing the category of each feature.
    :param n_types: List of integers representing the number of categories per feature.
    :param deciding_feature: Index of the feature used for classification.
    :return: 1 if the feature value is below threshold (half of n_types), else 0.
    """
    types = list(range(n_types[deciding_feature]))
    threshold = len(types) // 2
    if shape_name[deciding_feature] < threshold:
        return 1
    else:
        return 0


class SummerfieldTask:
    """
    Generates synthetic categorical data using One-Hot encoded features.
    :param features_types: List of integers; each represents the number of types for a feature.
    :param odd_dim: Dimension of an additional 'odd' feature (used for context/noise).
    :param sd: Standard deviation for potential noise (default 0).
    """

    def __init__(self, features_types, odd_dim, sd=0):
        self.n_features = len(features_types) + 1 if odd_dim != 0 else len(features_types)
        self.n_types = features_types + [odd_dim]
        self.sd = sd
        self.odd_dim = odd_dim
        self.shapes = None
        self.names = None

    def create_shapes(self):
        """
        Creates all possible feature combinations and their corresponding names.
        """
        features = []
        names = []
        for i in range(self.n_features):
            samples = np.eye(self.n_types[i])
            feature_labels = list(range(self.n_types[i]))
            features.append(samples)
            names.append(feature_labels)
        self.shapes = torch.tensor(np.array([np.concatenate(combo) for combo in itertools.product(*features)])).float()
        self.names = torch.tensor(list(itertools.product(*names))).float()

    def get_data(self, deciding_feature=0, odd=False):
        """
        Generates the final dataset and labels.
        :param deciding_feature: The feature index used to determine the label.
        :param odd: Boolean; if False, the 'odd_dim' feature is zeroed out in X.
        :return: X (features tensor), y (labels tensor).
        """
        self.create_shapes()
        labels = [classification_rule(name, self.n_types, deciding_feature) for name in self.names]
        y = torch.tensor(labels)[:, None].float()

        x = self.shapes.clone()
        if not odd and self.odd_dim != 0:
            x[:, -self.odd_dim:] = 0

        return x, y


# st = SummerfieldTask([2, 2], 1)
# x, y = st.get_data(0, odd=False)
# print(x)
# print(y)
