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
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
                    optimizer_type, activation_type, batch_size, sd, epochs, **current_block_config):
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
        "losses_clean": [],
        "MAE_clean": [],
        "accuracies_clean": [],
        "MAE_noisy": [],
        "accuracies_noisy": [],
        "MAE_noisy_optimal": [],
        "accuracies_noisy_optimal": [],
        "MAE_to_optimal": [],
        "accuracies_to_optimal": [],
        "noised_data": [],
        "activations_clean": [],
        "activation_distances_clean": [],
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
            data["losses_clean"].append(criterion(clean_preds, y).item())
            data["MAE_clean"].append(torch.abs(torch.sigmoid(clean_preds) - y).mean().item())
            data["accuracies_clean"].append(((torch.sigmoid(clean_preds) > 0.5) == y.bool()).float().mean().item())

            epoch_act_dist = {}
            epoch_raw_act = {}
            for layer_name, layer_activations in activations.items():
                layer_np = layer_activations
                epoch_raw_act[layer_name] = layer_np
                distances = pairwise_distances(layer_np)[np.triu_indices(X.shape[0], k=1)]
                epoch_act_dist[layer_name] = distances
            data["activations_clean"].append(epoch_raw_act)
            data["activation_distances_clean"].append(epoch_act_dist)
            model.remove_activations_hook()

            optimal_probs = get_bayes_optimal_probabilities(current_X, sd, **current_block_config)
            noisy_logits = model(current_X)
            model_probs = torch.sigmoid(noisy_logits)
            data["MAE_noisy"].append(torch.abs(model_probs - y).mean().item())
            data["accuracies_noisy"].append(((model_probs > 0.5) == y.bool()).float().mean().item())
            data["MAE_noisy_optimal"].append(torch.abs(optimal_probs - y).mean().item())
            data["accuracies_noisy_optimal"].append(((optimal_probs > 0.5) == y.bool()).float().mean().item())
            data["MAE_to_optimal"].append(torch.abs(model_probs - optimal_probs).mean().item())
            data["accuracies_to_optimal"].append(((model_probs > 0.5) == (optimal_probs > 0.5)).float().mean().item())

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

    def _add_config_info(self, ax, show_config=True):
        if not show_config:
            return

        cfg = self.config
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
            config_text += f"   {idx}. {name}, eps: {eps}, feat: {feat}, zero: {zf_str}\n"

        ax.text(1.05, 0.985, config_text.strip(), transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

    def graph_temp1(self, y, y_axis_name, log=False, show_config=True):
        cfg = self.config
        exp_name = cfg.get("exp_name", "Experiment").capitalize()
        low, high, optimal, boundaries, block_names = [], [], [], [0], []

        for block_name, block_results in self.results:
            low += block_results["data_low"][y]
            high += block_results["data_high"][y]
            if y + "_optimal" in block_results["data_low"]:
                optimal += block_results["data_low"][y + "_optimal"]
            boundaries.append(len(low))
            block_names.append(block_name)

        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.65, bottom=0.15)

        plt.plot(low, label='Low Variance (RichMLP)', color='blue')
        plt.plot(high, label='High Variance (LazyMLP)', color='red')
        if self.config["sd"] > 0.0 and len(optimal) > 0:
            plt.plot(optimal, label='Bayes Optimal', color='green', alpha=0.5)

        ax = plt.gca()
        self._add_config_info(ax, show_config)

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

        plt.title(f"{y_axis_name} Comparison in {exp_name}",
                  fontweight='bold')
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

    def loss_graph(self, sub_type="clean"):
        self.graph_temp1("losses_"+sub_type, sub_type.capitalize()+" BCE loss")

    def accuracy_graph(self, sub_type="clean"):
        self.graph_temp1("accuracies_"+sub_type, sub_type.capitalize()+" Accuracy")

    def MAE_graph(self, sub_type="clean"):
        self.graph_temp1("MAE_"+sub_type, sub_type.capitalize()+" Mean Absolute Error")

    def mds_graph(self, epoch=-1, layer_name='fc_last', model_type='low', show_config=True):
        target_block, target_res, target_config, rel_epoch = None, None, None, 0
        current_ep = 0

        for block_name, block_results in self.results:
            n_epochs = len(block_results[f"data_{model_type}"]["activation_distances_clean"])
            if epoch == -1 or current_ep <= epoch < current_ep + n_epochs:
                target_block = block_name
                target_res = block_results[f"data_{model_type}"]
                target_config = block_results["config"]
                rel_epoch = (n_epochs - 1) if epoch == -1 else (epoch - current_ep)
                break
            current_ep += n_epochs

        if target_res is None:
            return

        dists_condensed = target_res["activation_distances_clean"][rel_epoch][layer_name]
        dist_matrix = squareform(dists_condensed)
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4)
        coords = mds.fit_transform(dist_matrix)

        y = target_res["y"].cpu().numpy().flatten()
        X = target_res["X"].cpu().numpy()

        plt.figure(figsize=(9, 7))
        plt.subplots_adjust(right=0.65)

        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap='coolwarm', s=130, edgecolors='gray', alpha=0.85)
        self._add_config_info(plt.gca(), show_config)

        features_types = self.config["features_types"]
        zero_feats = target_config.get("zero_features", [])

        for i, coord in enumerate(coords):
            x_row = X[i]
            feats_str = []
            start_idx = 0
            for f_idx, dim in enumerate(features_types):
                if f_idx in zero_feats:
                    feats_str.append("-")
                else:
                    val = np.argmax(x_row[start_idx:start_idx + dim])
                    feats_str.append(str(val))
                start_idx += dim

            label_text = f"({','.join(feats_str)})"
            plt.annotate(label_text, (coord[0], coord[1]), xytext=(6, 6), textcoords='offset points',
                         fontsize=9, fontweight='bold', color='#444444')

        plt.margins(0.15)
        plt.title(
            f"MDS of '{layer_name}' Activations ({model_type.capitalize()} Variance)\nBlock: {target_block} | Absolute Epoch: {epoch if epoch != -1 else 'Last'}",
            fontweight='bold')

        handles, _ = scatter.legend_elements()
        plt.legend(handles, ["Label 0", "Label 1"], title="True Categories", loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)

        if self.save:
            base = f"{self.path}/MDS_{layer_name}_{model_type}_ep{epoch}"
            path, i = f"{base}.png", 1
            while os.path.exists(path):
                path, i = f"{base}_{i}.png", i + 1
            plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.show()