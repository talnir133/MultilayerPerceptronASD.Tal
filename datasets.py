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

    def __init__(self, features_types, sd, seed):
        self.n_features = len(features_types)
        self.n_types = features_types
        self.sd = sd
        self.seed = seed
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

        self.noise = torch.zeros_like(self.X)
        if self.sd > 0:
            torch.manual_seed(self.seed)
            self.noise = torch.normal(mean=0.0, std=self.sd, size=self.X.shape)

    def get_stage_data_and_labels(self, zero_features, deciding_feature, **kwargs):
        """
        Generates stage-specific data by zeroing selected features, removing duplicates,
        and applying the pre-calculated noise to the unique samples.
        """
        stage_X = self.X.clone()
        stage_noise = self.noise.clone()

        start_idx = 0
        for i, dim in enumerate(self.n_types):
            if i in zero_features:
                stage_X[:, start_idx: start_idx + dim] = 0.0
                stage_noise[:, start_idx: start_idx + dim] = 0.0
            start_idx += dim

        labels = [classification_rule(name, self.n_types, deciding_feature) for name in self.names]
        y = torch.tensor(labels)[:, None].float()

        combined = torch.cat((stage_X, y), dim=1).numpy()
        _, unique_indices = np.unique(combined, axis=0, return_index=True)
        unique_indices.sort()

        stage_X = stage_X[unique_indices]
        y = y[unique_indices]
        stage_noise = stage_noise[unique_indices]

        if self.sd > 0:
            stage_X = stage_X + stage_noise

        return stage_X, y