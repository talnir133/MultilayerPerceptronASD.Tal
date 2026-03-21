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


def train_mlp_model(model, X, y, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                    optimizer_type, activation_type, batch_size, seed, **kwargs):
    """
    Initializes and trains an MLP model, capturing activation distances and loss history.

    :param model: MLP model
    :param input_size: Integer, number of input features (total dimension of concatenated one-hots).
    :param hidden_size: Integer, number of neurons in each hidden layer.
    :param n_hidden: Integer, number of hidden layers (excluding input and output).
    :param output_size: Integer, dimension of the output layer (usually 1 for binary classification).
    :param w_scale: List of 2 floats, standard deviation used for normal initialization of weights (low and high).
    :param b_scale: Float, standard deviation used for normal initialization of biases.
    :param X: Tensor, the full feature set used for evaluation and distance calculation.
    :param y: Tensor, the full label set used for final loss evaluation.
    :param optimizer_type: The PyTorch optimizer class to use (e.g., optim.Adam or optim.SGD).
    :param activation_type: String, the type of activation function for hidden layers ('RelU', 'Sigmoid', 'Tanh', etc.).
    """
    # Move data to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Measurement:
    data = {"losses": [], "accuracies": [], "X": X, "y": y}

    if not model:
        # Model setup
        model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type)
        model = model.to(device)
        model.reinitialize(seed=seed)

    activations = {}
    model.set_activations_hook(activations)
    criterion = nn.BCEWithLogitsLoss()  # removed sigmoid - Should be fixed for Oded's experiment
    optimizer = optimizer_type(model.parameters(), lr=0.004)

    # Saving distances data before training
    model.eval()
    data["input distances"] = pairwise_distances(X.cpu().detach().numpy())[np.triu_indices(X.shape[0])]
    with torch.no_grad():
        model(X)  # filling "activations" with data
        data["initial activations distances"] = {k: pairwise_distances(v)[np.triu_indices(X.shape[0])] for k, v in
                                                 activations.items()}
    model.remove_activations_hook()

    # Training loop
    model.train()
    for epoch in tqdm(range(kwargs.get("epoches")), desc="Training"):

        model.eval()
        with torch.no_grad():
            preds = model(X)
            data["losses"].append(criterion(preds, y).item())
            data["accuracies"].append(1 - torch.abs(torch.sigmoid(preds) - y).mean().item())
        model.train()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        test_loss = criterion(y_pred, y)
        print(f'Final Loss: {test_loss.item() / len(dataloader):.4f}')
        data["final predicted y"] = y_pred

    return model, data


class Figures():
    def __init__(self, results, config, save):
        self.config, self.save, self.results = config, save, results
        self.path = f"figures/{config.get('exp_name', 'Experiment')}"

    def graph_temp1(self, y, y_axis_name, log=False):
        cfg = self.config
        exp_name = cfg.get("exp_name", "Experiment").capitalize()
        low, high, boundaries, block_names = [], [], [0], []

        for step_name, step_results in self.results:
            low += step_results["data_low"][y]
            high += step_results["data_high"][y]
            boundaries.append(len(low))
            block_names.append(step_name)

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
            name, eps, feat = block.get('block_name', 'Unnamed'), block.get('epoches', 0), block.get('deciding_feature',
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
