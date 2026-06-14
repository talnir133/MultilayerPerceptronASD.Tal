import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==========================================
# Data Generation & Manipulation
# ==========================================

class Dataset:
    def __init__(self, features_types):
        self.features_types = features_types
        self.exp_X = None

    def create_exp_data(self):
        feature_options = [torch.eye(dim) for dim in self.features_types]
        all_combos = list(itertools.product(*feature_options))
        self.exp_X = torch.stack([torch.cat(combo) for combo in all_combos])

    def get_block_data(self, zero_features):
        X = self.exp_X.clone()
        offsets = np.cumsum([0] + self.features_types)

        for i in zero_features:
            X[:, offsets[i]: offsets[i + 1]] = 0.0

        _, idx = np.unique(X.cpu().numpy(), axis=0, return_index=True)
        return X[np.sort(idx)]



# ==========================================
# Classifier Model
# ==========================================

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, w_scale, b_scale, activation_type, hidden_size=None, n_hidden=1, **kwargs):
        super(Classifier, self).__init__()
        self.w_scale = w_scale
        self.b_scale = b_scale
        self._layers = nn.Sequential()
        hidden_size= input_size if hidden_size is None else hidden_size

        if n_hidden <= 0:
            self._layers.add_module('fc_last', nn.Linear(input_size, output_size, bias=bool(b_scale)))
        else:
            self._layers.add_module('fc1', nn.Linear(input_size, hidden_size, bias=bool(b_scale)))
            self._layers.add_module('activation_func1', activation_type)

            for i in range(n_hidden - 1):
                self._layers.add_module(f'fc{i + 2}', nn.Linear(hidden_size, hidden_size, bias=bool(b_scale)))
                self._layers.add_module(f'activation_func{i + 2}', activation_type)

            self._layers.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=bool(b_scale)))

    def forward(self, x):
        """Standard forward pass through the network."""
        return self._layers(x)

    def reinitialize(self):
        for name, m in self.named_modules():
            if 'fc1' in name:
                nn.init.xavier_normal_(m.weight, gain=self.w_scale)
                if bool(self.b_scale):
                    nn.init.normal_(m.bias, 0, self.b_scale)


def train_classifier(model, optimizer, X_base, rule_func, epochs, batch_size, noise_sd, noise_mask, alpha_class=1.0,
                alpha_rec=0.0, metric_callback=None):
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs), desc="Training Classifier"):

        y_class = rule_func(X_base).view(-1, 1)
        y_rec = X_base.clone()
        y = torch.cat((y_class, y_rec), dim=1)

        # 1. Addition of noise for this epoch, if needed
        if noise_sd > 0:
            X_noisy = X_base + (torch.randn_like(X_base) * noise_sd * noise_mask)
            X = X_noisy
        else:
            X = X_base

        # 2. Evaluation & External Metric Collection
        if metric_callback:
            model.eval()
            with torch.no_grad():
                metric_callback(model, criterion)

        # 3. Training Step
        model.train()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # --- DUAL LOSS CALCULATION ---

            # 1. Classification Loss (First coordinate)
            preds_class = outputs[:, 0:1]
            labels_class = labels[:, 0:1]
            loss = alpha_class * criterion(preds_class, labels_class)

            # 2. Reconstruction Loss (Remaining coordinates)
            if alpha_rec > 0:
                preds_rec = outputs[:, 1:]
                labels_rec = labels[:, 1:]
                loss += alpha_rec * criterion(preds_rec, labels_rec)

            # Backpropagation of the combined loss
            loss.backward()
            optimizer.step()

    return model, optimizer



# ==========================================
# Logistic Regression Decoder
# ==========================================

class LogisticDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticDecoder, self).__init__()
        self._layers = nn.Sequential()
        self._layers.add_module('fc_decoder', nn.Linear(input_size, output_size))

    def forward(self, x):
        """Standard forward pass for the logistic decoder."""
        return self._layers(x)


def train_logistic_decoder(activations, clean_targets, epochs=100, lr=0.1, feature_mask=None):
    # 1. Setup Model, Optimizer & Criterion
    input_size = activations.shape[1]
    output_size = clean_targets.shape[1]
    device = activations.device

    decoder = LogisticDecoder(input_size, output_size).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    clean_targets_float = clean_targets.float()

    # 2. Training Step (Iterative Optimization)
    with torch.enable_grad():
        decoder.train()
        prev_loss = float('inf')
        converged = False
        patience = 3
        patience_counter = 0

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = decoder(activations)
            loss = criterion(outputs, clean_targets_float)

            # Early stopping if the model has converged
            loss_diff = abs(prev_loss - loss.item())
            if loss_diff < 1e-4:
                patience_counter += 1
                if patience_counter >= patience:
                    converged = True
                    break
            else:
                patience_counter = 0

            prev_loss = loss.item()

            loss.backward()
            optimizer.step()

    # 3. DR and Accuracy Calculation
    decoder.eval()
    with torch.no_grad():
        logits = decoder(activations)
        probs = torch.sigmoid(logits)

        # Binary Entropy Calculation
        p_c = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        H = -p_c * torch.log2(p_c) - (1.0 - p_c) * torch.log2(1.0 - p_c)

        # Apply feature mask if zero_features are present
        if feature_mask is not None:
            valid_cols = feature_mask[0] > 0
            entropy = H[:, valid_cols].mean().item()
            acc = ((probs > 0.5).float() == clean_targets_float)[:, valid_cols].float().mean().item()
        else:
            entropy = H.mean().item()
            acc = ((probs > 0.5).float() == clean_targets_float).float().mean().item()

    return entropy, acc, converged