import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
import copy

matplotlib.use('TkAgg')
from sklearn.metrics.pairwise import pairwise_distances
from datasets import Dataset
from models import MLP


def merge_configs(block_config, config):
    config.update(block_config)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["input_size"] = sum(config["features_types"])

    opt_map = {"Adam": optim.Adam, "SGD": optim.SGD}
    config["optimizer_type"] = opt_map[config["optimizer_type"]]

    act_map = {
        "Tanh": nn.Tanh, "RelU": nn.ReLU,
        "Sigmoid": nn.Sigmoid, "Identity": nn.Identity

    }
    act_name = config.get("activation_type", "Identity")
    if isinstance(act_name, str):
        config["activation_type"] = act_map.get(act_name, nn.Identity)()

    return config



def get_bayes_optimal_probabilities(noisy_X, sd, features_types, deciding_feature, **kwargs):

    start_idx = sum(features_types[:deciding_feature])
    n_types = features_types[deciding_feature]
    end_idx = start_idx + n_types

    noisy_deciding_feature = noisy_X[:, start_idx:end_idx]

    if sd == 0.0:
        best_match = torch.argmax(noisy_deciding_feature, dim=1)
        threshold = n_types // 2
        optimal_probs = (best_match < threshold).float().unsqueeze(1)
        return optimal_probs

    templates = torch.eye(n_types, device=noisy_X.device)
    dists = torch.cdist(noisy_deciding_feature, templates) ** 2
    likelihoods = torch.exp(-dists / (2 * (sd ** 2)))

    threshold = n_types // 2
    L_class1 = torch.sum(likelihoods[:, :threshold], dim=1)
    L_class0 = torch.sum(likelihoods[:, threshold:], dim=1)

    optimal_probs = L_class1 / (L_class1 + L_class0)

    return optimal_probs.unsqueeze(1)


def train_mlp_model(model, optimizer, X, y, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                    optimizer_type, activation_type, batch_size, seed, sd, epochs, **current_block_config):
    """
    Initializes and trains an MLP model.
    Tracks clean loss/accuracy, activation distances, raw activations, and robustness against noise.
    """
    # ==========================================
    # 1. Setup & Preparations
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    if not model:
        model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type)
        model = model.to(device)
        model.reinitialize()
        optimizer = optimizer_type(model.parameters(), lr=0.004)

    criterion = nn.BCEWithLogitsLoss()

    # ==========================================
    # 2. Data Tracking
    # ==========================================
    data = {
        "losses": [],
        "accuracies": [],
        "optimal_accuracies": [],
        "noised_data": [],
        "activation_history": [],
        "activation_distances_history": [],
        "X": X,
        "y": y
    }

    #  creating a mask to apply noise only to non-zeroed features
    noise_mask = torch.ones_like(X)
    zero_feats = current_block_config.get("zero_features", [])
    features_types = current_block_config.get("features_types", [])

    start_idx = 0
    for i, dim in enumerate(features_types):
        if i in zero_feats:
            noise_mask[:, start_idx: start_idx + dim] = 0.0
        start_idx += dim

    # ==========================================
    # 3. Training Loop
    # ==========================================
    for epoch in tqdm(range(epochs), desc="Training"):

        # --- Noise Generation ---
        noise = torch.randn_like(X) * sd * noise_mask
        current_X = X + noise
        data["noised_data"].append(current_X.cpu().detach().numpy())

        # --- evaluation ---
        model.eval()
        activations = {}
        model.set_activations_hook(activations)

        with torch.no_grad():
            clean_preds = model(X)
            data["losses"].append(criterion(clean_preds, y).item())
            data["accuracies"].append(1 - torch.abs(torch.sigmoid(clean_preds) - y).mean().item())

            epoch_act_dist = {}
            epoch_raw_act = {}
            for layer_name, layer_activations in activations.items():
                layer_np = layer_activations
                epoch_raw_act[layer_name] = layer_np
                distances = pairwise_distances(layer_np)[np.triu_indices(X.shape[0], k=1)]
                epoch_act_dist[layer_name] = distances
            data["activation_history"].append(epoch_raw_act)
            data["activation_distances_history"].append(epoch_act_dist)
            model.remove_activations_hook()

            optimal_probs = get_bayes_optimal_probabilities(current_X, sd, **current_block_config)
            noisy_logits = model(current_X)
            model_probs = torch.sigmoid(noisy_logits)
            data["optimal_accuracies"].append(1-torch.abs(model_probs - optimal_probs).mean().item())

        # --- Training Step ---
        model.train()
        dataset = TensorDataset(current_X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # ==========================================
    # 4. Finalization
    # ==========================================
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        test_loss = criterion(y_pred, y)
        print(f'Final Loss: {test_loss.item():.4f}')
        data["final predicted y"] = y_pred.cpu().numpy()

    return model, optimizer, data


class Figures():
    def __init__(self, results, config, save):
        self.config, self.save, self.results = config, save, results
        self.path = f"figures/{config.get('exp_name', 'Experiment')}"

    def graph_temp1(self, y, y_axis_name, log=False):
        cfg = self.config
        exp_name = cfg.get("exp_name", "Experiment").capitalize()
        low, high, boundaries, block_names = [], [], [0], []

        for block_name, block_results in self.results:
            low += block_results["data_low"][y]
            high += block_results["data_high"][y]
            boundaries.append(len(low))
            block_names.append(block_name)

        plt.figure(figsize=(14, 6))
        plt.subplots_adjust(right=0.65, bottom=0.15)
        plt.plot(low, label='Low Variance (RichMLP)', color='blue')
        plt.plot(high, label='High Variance (LazyMLP)', color='red')

        ax = plt.gca()
        x_offset = boundaries[-1] * 0.02

        for i in range(1, len(boundaries) - 1):
            line = boundaries[i]
            plt.axvline(x=line, color='gray', linestyle='--', linewidth=1)
            plt.text(line - x_offset, sum(plt.ylim()) / 2, 'Block Shift', color='gray', fontsize=9,
                     rotation=90, va='center', ha='right')

        for i in range(len(block_names)):
            start, end = boundaries[i], boundaries[i + 1]
            ax.text((start + end) / 2, -0.06, block_names[i], transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=10, fontweight='bold', color='darkblue')

        config_text = r"$\mathbf{Simulation's\ Configurations:}$" + "\n\n"
        categories = {
            "Input:": ['features_types', 'seed', 'sd'],
            "Network:": ['hidden_size', 'n_hidden', 'output_size', 'b_scale_low', 'b_scale_high',
                         'w_scale_low', 'w_scale_high', 'optimizer_type', 'activation_type', 'batch_size']
        }

        for title, keys in categories.items():
            config_text += f"{title}\n"
            for k in keys:
                val = cfg.get(k)
                if isinstance(val, list): val = ', '.join(map(str, val))
                config_text += f"   {k}: {val}\n"
            config_text += "\n"

        config_text += "Experiment Blocks:\n"
        for idx, block in enumerate(cfg.get('exp_blocks', []), 1):
            name, eps, feat = block.get('block_name', 'Unnamed'), block.get('epochs', 0), block.get('deciding_feature',
                                                                                                     0)
            zf = block.get('zero_features', [])
            zf_str = "None" if not zf else (",".join(map(str, zf)) if isinstance(zf, (list, tuple)) else str(zf))
            config_text += f"   {idx}. {name}, epochs: {eps}, deciding_feature: {feat}, zero_features: {zf_str}\n"

        plt.gca().text(1.05, 1.0, config_text.strip(), transform=plt.gca().transAxes,
                       fontsize=9, va='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

        plt.title(f"Summerfield's Replication, {y.capitalize()} Comparison in {exp_name}", fontweight='bold')
        plt.xlabel('Epochs', labelpad=20)
        plt.ylabel(y_axis_name)

        if log: plt.yscale('log')

        plt.legend(loc='lower left', bbox_to_anchor=(1.035, 0.0), frameon=True, edgecolor='gray', borderaxespad=0.)
        plt.grid(True, which="both", ls="-", alpha=0.5)

        if self.save:
            base = f"{self.path}/{y}_figure_{exp_name.replace(' ', '_')}"
            path, i = f"{base}.png", 1
            while os.path.exists(path):
                path, i = f"{base}_{i}.png", i + 1
            plt.savefig(path, bbox_inches='tight', dpi=300)

        plt.show()

    def loss_graph(self):
        self.graph_temp1("losses", "BCE loss")

    def accuracy_graph(self):
        self.graph_temp1("accuracies", "Accuracy")
