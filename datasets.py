import numpy as np
import itertools
import torch


def classification_rule(shape_name, n_types, deciding_feature):
    """
    Determines the binary label based on whether the specific deciding feature's
    value is below the halfway threshold of its possible categories.
    """
    types = list(range(n_types[deciding_feature]))
    threshold = len(types) // 2
    if shape_name[deciding_feature] < threshold:
        return 1
    else:
        return 0


class Dataset:
    """
    Generates synthetic categorical data using One-Hot encoded features.
    """

    def __init__(self, features_types):
        self.n_features = len(features_types)
        self.n_types = features_types
        self.names = None
        self.X = None
        self.noise = None

    def create_exp_data(self):
        """
        Creates all combinations of One-Hot features and pre-calculates the noise tensor.
        """
        features = []
        names = []
        for i in range(self.n_features):
            samples = np.eye(self.n_types[i])
            feature_labels = list(range(self.n_types[i]))
            features.append(samples)
            names.append(feature_labels)

        self.X = torch.tensor(np.array([np.concatenate(combo) for combo in itertools.product(*features)])).float()
        self.names = torch.tensor(list(itertools.product(*names))).float()

    def get_block_data_and_labels(self, zero_features, deciding_feature, **kwargs):
        """
        Generates block-specific data by zeroing selected features, and removing duplicates.
        """
        block_X = self.X.clone()

        start_idx = 0
        for i, dim in enumerate(self.n_types):
            if i in zero_features:
                block_X[:, start_idx: start_idx + dim] = 0.0
            start_idx += dim

        labels = [classification_rule(name, self.n_types, deciding_feature) for name in self.names]
        y = torch.tensor(labels)[:, None].float()

        combined = torch.cat((block_X, y), dim=1).numpy()
        _, unique_indices = np.unique(combined, axis=0, return_index=True)
        unique_indices.sort()

        block_X = block_X[unique_indices]
        y = y[unique_indices]

        return block_X, y
