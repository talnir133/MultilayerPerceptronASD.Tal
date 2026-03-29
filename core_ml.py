import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Dataset:
    def __init__(self, features_types):
        self.n_features = len(features_types)
        self.features_types = features_types
        self.names = None
        self.exp_X = None

    def create_exp_data(self):
        features, names = [], []
        for i in range(self.n_features):
            samples = np.eye(self.features_types[i])
            feature_labels = list(range(self.features_types[i]))
            features.append(samples)
            names.append(feature_labels)

        self.exp_X = torch.tensor(np.array([np.concatenate(combo) for combo in itertools.product(*features)])).float()
        self.names = torch.tensor(list(itertools.product(*names))).float()

    def classification_rule(self, shape_name, deciding_feature):
        types = list(range(self.features_types[deciding_feature]))
        threshold = len(types) // 2
        return 1 if shape_name[deciding_feature] < threshold else 0

    def get_block_data(self, zero_features, deciding_feature):
        X = self.exp_X.clone()
        start_idx = 0
        for i, dim in enumerate(self.features_types):
            if i in zero_features:
                X[:, start_idx: start_idx + dim] = 0.0
            start_idx += dim

        labels = [self.classification_rule(name, deciding_feature) for name in self.names]
        y = torch.tensor(labels)[:, None].float()

        combined = torch.cat((X, y), dim=1).numpy()
        _, unique_indices = np.unique(combined, axis=0, return_index=True)
        unique_indices.sort()
        block_X, block_y = X[unique_indices], y[unique_indices]

        return block_X, block_y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type, **kwargs):
        super(MLP, self).__init__()
        self.w_scale = w_scale
        self.b_scale = b_scale
        self._layers = nn.Sequential()

        self._layers.add_module('fc1', nn.Linear(input_size, hidden_size, bias=bool(b_scale)))
        self._layers.add_module('activation_func1', activation_type)

        for i in range(n_hidden):
            self._layers.add_module(f'fc{i + 2}', nn.Linear(hidden_size, hidden_size, bias=bool(b_scale)))
            self._layers.add_module(f'activation_func{i + 2}', activation_type)

        self._layers.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=bool(b_scale)))

    def forward(self, x):
        """Standard forward pass through the network."""
        return self._layers(x)

    def reinitialize(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if bool(self.b_scale):
                    nn.init.normal_(m.bias, 0, self.b_scale)
                if 'fc1' in name:
                    nn.init.xavier_normal_(m.weight, gain=self.w_scale)
                else:
                    nn.init.xavier_normal_(m.weight)


def train_model(model, optimizer, X_base, y, epochs, batch_size, noise_sd, noise_mask, metric_callback):
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs), desc="Training"):
        # 1. Addition of noise for this epoch, if needed
        if noise_sd > 0:
            X_noisy = X_base + (torch.randn_like(X_base) * noise_sd * noise_mask)
            X = X_noisy
        else:
            X = X_base

        # 2. Evaluation & External Metric Collection
        model.eval()
        with torch.no_grad():
            metric_callback(model, X, criterion)

        # 3. Training Step
        model.train()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    return model, optimizer
