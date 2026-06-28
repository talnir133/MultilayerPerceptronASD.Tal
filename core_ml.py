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
    def __init__(self, features_types: list[int]) -> None:
        """
        Initializes the dataset generator with the specified dimensions for each feature.
        """
        self.features_types = features_types
        self.exp_X = None

    def create_exp_data(self) -> None:
        """
        Generates the full dataset by creating all possible combinations
        of one-hot encoded vectors based on the defined features_types.
        """
        feature_options = [torch.eye(dim) for dim in self.features_types]
        all_combos = list(itertools.product(*feature_options))
        self.exp_X = torch.stack([torch.cat(combo) for combo in all_combos])

    def get_block_data(self, zero_features: list[int]) -> torch.Tensor:
        """
        Returns a dataset subset where specified features are zeroed out (ablated).
        Removes duplicate rows that emerge from this ablation.
        """
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
    def __init__(self, input_size: int, output_size: int, w_scale: float, b_scale: float,
                 activation_type: type[nn.Module], hidden_size: int = None, n_hidden: int = 1, **kwargs) -> None:
        """
        Initializes the MLP classifier architecture based on the given configuration.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the network layers.
        """
        return self._layers(x)

    def reinitialize(self) -> None:
        """
        Reinitializes the weights and biases of the first hidden layer (fc1)
        according to the variance scales specified in the configuration.
        """
        for name, m in self.named_modules():
            if 'fc1' in name:
                nn.init.xavier_normal_(m.weight, gain=self.w_scale)
                if bool(self.b_scale):
                    nn.init.normal_(m.bias, 0, self.b_scale)


def train_classifier(model: nn.Module, optimizer: torch.optim.Optimizer, X_base: torch.Tensor,
                     rule_func: callable, epochs: int, batch_size: int, noise_sd: float,
                     noise_mask: torch.Tensor, alpha_class: float = 1.0, alpha_rec: float = 0.0,
                     metric_callback: callable = None) -> tuple[nn.Module, torch.optim.Optimizer]:
    """
    Executes the training loop for the classification model over a specified number of epochs.
    Injects dynamic Gaussian noise per epoch and optionally computes reconstruction loss.
    """
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
    """
    Multinomial Logistic Regression (Softmax Regression) Decoder.
    Evaluates the decisiveness and accuracy of network representations.
    Assumes feature independence, creating a separate model per feature.
    """

    def __init__(self, input_size: int, features_types: list[int]) -> None:
        """
        Initializes the multinomial logistic regression probes for each feature independently.
        """
        super().__init__()
        self.features_types = features_types
        self.num_features = len(features_types)
        self.loss_trajectory = []

        self.feature_probes = nn.ModuleList([
            nn.Linear(input_size, dim) for dim in features_types
        ])

    def forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns a list of raw logits for each feature separately.
        """
        return [probe(h) for probe in self.feature_probes]

    def fit(self, classification_model: nn.Module, X_clean: torch.Tensor, train_noise_sd: float,
            noise_mask: torch.Tensor, target_layer: str = "fc1", epochs: int = 100, lr: float = 0.1) -> None:
        """
        Trains the linear probes to predict the clean source features from noisy network activations.
        The target labels are derived dynamically using an optimal Bayesian nearest-neighbor approach.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        true_labels = []
        start_idx = 0
        for dim in self.features_types:
            labels = torch.argmax(X_clean[:, start_idx: start_idx + dim], dim=1)
            true_labels.append(labels)
            start_idx += dim

        self.loss_trajectory = []
        self.train()

        with torch.enable_grad():
            for _ in range(epochs):
                optimizer.zero_grad()

                X_noisy = X_clean + (torch.randn_like(X_clean) * train_noise_sd * noise_mask)

                classification_model.eval()
                with torch.no_grad():
                    out = X_noisy
                    for name, layer in classification_model._layers.named_children():
                        out = layer(out)
                        if name == target_layer:
                            classification_activations = out.detach()
                            break

                logits_per_feature = self.forward(classification_activations)
                total_loss = sum(criterion(logits_per_feature[k], true_labels[k]) for k in range(self.num_features))

                total_loss.backward()
                optimizer.step()

                # Tracking the loss trajectory using Bayesian optimal labels
                closest_clean_idx = torch.cdist(X_noisy, X_clean).argmin(dim=1)
                bayes_labels = []
                start_idx = 0
                for dim in self.features_types:
                    clean_labels_k = torch.argmax(X_clean[:, start_idx: start_idx + dim], dim=1)
                    bayes_labels.append(clean_labels_k[closest_clean_idx])
                    start_idx += dim
                tracking_loss = [criterion(logits_per_feature[k], bayes_labels[k]).item() for k in range(self.num_features)]
                self.loss_trajectory.append(tracking_loss)

    def measure_decisiveness(self, classification_activations: torch.Tensor) -> list[float]:
        """
        Evaluates the dynamic range (decisiveness) of the activations per feature.
        Returns a list of scores normalized to [0, 1], where 1.0 represents a perfectly confident (one-hot) prediction.
        """
        self.eval()
        decisiveness_scores = []
        import math

        with torch.no_grad():
            logits_per_feature = self.forward(classification_activations)

            for k, dim in enumerate(self.features_types):
                probs = torch.softmax(logits_per_feature[k], dim=1)
                p_c = torch.clamp(probs, 1e-7, 1.0)
                entropy_k = -torch.sum(p_c * torch.log2(p_c), dim=1)

                max_entropy = math.log2(dim) if dim > 1 else 1.0
                normalized_entropy = entropy_k / max_entropy

                decisiveness = 1.0 - normalized_entropy
                decisiveness_scores.append(decisiveness.mean().item())

        return decisiveness_scores

    def measure_decision_boundary_accuracy(self, classification_activations: torch.Tensor,
                                           X_noisy: torch.Tensor, X_clean: torch.Tensor) -> list[float]:
        """
        Measures the alignment between the decoder's decision boundary and the optimal Bayesian classifier.
        Returns the accuracy percentage per feature.
        """
        self.eval()
        accuracies = []

        with torch.no_grad():
            logits_per_feature = self.forward(classification_activations)

            # Bayesian Evaluation (Simplified to exact geometric nearest neighbor)
            closest_clean_idx = torch.cdist(X_noisy, X_clean).argmin(dim=1)

            start_idx = 0
            for k, dim in enumerate(self.features_types):
                # Model assignment
                model_argmax = torch.argmax(logits_per_feature[k], dim=1)

                # Bayes assignment
                clean_labels_k = torch.argmax(X_clean[:, start_idx: start_idx + dim], dim=1)
                bayes_argmax = clean_labels_k[closest_clean_idx]

                # Comparison
                accuracy = (model_argmax == bayes_argmax).float().mean().item()
                accuracies.append(accuracy)

                start_idx += dim

        return accuracies

    def evaluate(self, classification_model: nn.Module, X_clean: torch.Tensor, noise_mask: torch.Tensor,
                 test_noise_sd: float = 0.3, target_layer: str = "fc1", samples_per_point: int = 20) -> dict:
        """
        Consolidates the full probing evaluation: measures decisiveness and accuracy
        both near the original data points (interpolation) and near the center (extrapolation).
        """
        results = {}
        M = samples_per_point

        # 1. Evaluation Near Data (Interpolation)
        X_clean_repeated = X_clean.repeat(M, 1)
        mask_repeated = noise_mask.repeat(M, 1)
        X_noisy_data = X_clean_repeated + (torch.randn_like(X_clean_repeated) * test_noise_sd * mask_repeated)

        classification_model.eval()
        with torch.no_grad():
            out = X_noisy_data
            for name, layer in classification_model._layers.named_children():
                out = layer(out)
                if name == target_layer:
                    classification_activations_data = out.detach()
                    break

        results["near_data_decisiveness"] = self.measure_decisiveness(classification_activations_data)
        results["near_data_accuracy"] = self.measure_decision_boundary_accuracy(classification_activations_data,
                                                                                X_noisy_data, X_clean)

        # 2. Evaluation Near Center (Extrapolation)
        X_center = torch.full_like(X_clean, 0.5)
        X_center_repeated = X_center.repeat(M, 1)
        X_noisy_center = X_center_repeated + (torch.randn_like(X_center_repeated) * test_noise_sd * mask_repeated)

        with torch.no_grad():
            out = X_noisy_center
            for name, layer in classification_model._layers.named_children():
                out = layer(out)
                if name == target_layer:
                    classification_activations_center = out.detach()
                    break

        results["near_center_decisiveness"] = self.measure_decisiveness(classification_activations_center)
        results["near_center_accuracy"] = self.measure_decision_boundary_accuracy(classification_activations_center,
                                                                                  X_noisy_center, X_clean)

        # 3. Training Loss Trajectory
        results["loss_trajectory"] = self.loss_trajectory

        return results