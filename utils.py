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
from datasets import SummerfieldTask
from models import MLP


def train_mlp_model(model, X, y, dataloader, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                    optimizer_type=optim.Adam,
                    num_epochs=150, device=None, activation_type=None, **kwargs):
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
    :param dataloader: PyTorch DataLoader providing shuffled batches for the training loop.
    :param optimizer_type: The PyTorch optimizer class to use (e.g., optim.Adam or optim.SGD).
    :param num_epochs: Integer, the number of complete passes over the dataset.
    :param device: torch.device, specifies whether to train on 'cpu' or 'cuda'.
    :param activation_type: String, the type of activation function for hidden layers ('RelU', 'Sigmoid', 'Tanh', etc.).
    """
    # Move data to device
    X = X.to(device)
    y = y.to(device)

    # Measurement:
    data = {"losses": [], "accuracies": [], "X": X, "y": y}

    if not model:
        # Model setup
        model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type=activation_type)
        model = model.to(device)
        model.reinitialize(seed=42)

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
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Data saving:
            data["losses"].append(loss.item() / len(dataloader))
            data["accuracies"].append(1 - torch.abs(torch.sigmoid(outputs) - labels).mean().item())

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        test_loss = criterion(y_pred, y)
        print(f'Final Loss: {test_loss.item() / len(dataloader):.4f}\n')
        data["final predicted y"] = y_pred

    return model, data


def create_dataset(features_types, odd_dim
                   , deciding_feature=0, odd=False, seed=0, device=None, unique_points_only=False, batch_size=1,
                   **kwargs):
    """
    A wrapper to initialize the task, generate data, and prepare a DataLoader.
    :param features_types: Dimensions for each One-Hot encoded feature.
    :param odd_dim: Dimension of the odd feature.
    :param deciding_feature: Index used for the labeling rule.
    :param odd: Whether to include the 'odd' feature data in X.
    :param seed: Random seed for reproducibility.
    :param device: torch.device (CPU/CUDA).
    :return: X, y, and a DataLoader configured with batch_size=len(X).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    st = SummerfieldTask(features_types, odd_dim)

    X, y = st.get_data(deciding_feature, odd, unique_points_only)
    X = X.to(device)
    y = y.to(device)

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return X, y, dataloader


class Figures():  # continue working on it
    def __init__(self, results, config, save):
        self.config = config
        self.save = save
        folder_name = results["initial"]["config"]["config_name"]
        self.path = f"figures/{folder_name}"
        self.data = results
        self.exps = results["initial"]["config"]["exp_stages"]
        if self.data["initial"]:
            self.variable = "Weights" if self.data["initial"]["config"]["b_scale_low"] == 0 else "Biases"

    def graph_temp1(self, y, y_axis_name, log=False):
        cfg = self.config
        act_type = cfg.get("activation_type", "Unknown")
        n_epochs = cfg.get("num_epochs", 0)

        low, high, lines = [], [], []
        exp_name = self.exps[1].capitalize() if len(self.exps) == 2 else 'Initial condition'

        for name in self.exps:
            exp = self.data[name]
            low += exp["data_low"][y]
            high += exp["data_high"][y]
            lines.append(len(low))

        plt.figure(figsize=(12, 6))
        plt.subplots_adjust(right=0.75)

        plt.plot(low, label='Low Variance (RichMLP)', color='blue')
        plt.plot(high, label='High Variance (LazyMLP)', color='red')

        for line in lines[:-1]:
            plt.axvline(x=line, color='gray', linestyle='--', linewidth=1)
            plt.text(line - 5, sum(plt.ylim()) / 2, 'Shift-point', color='gray', fontsize=9,
                     rotation=90, va='center', ha='right')

        config_text = "Simulation's Configurations:\n" + "-" * 32 + "\n\n"
        categories = {
            "1. Input": ['features_types', 'odd_dim', 'batch_size', 'seed', 'unique_points_only'],
            "2. Network": ['hidden_size', 'n_hidden', 'output_size', 'b_scale_low', 'b_scale_high',
                           'w_scale_low', 'w_scale_high', 'optimizer_type', 'activation_type'],
            "3. Experiment": ['num_epochs', 'exp_stages']
        }

        for title, keys in categories.items():
            config_text += f"{title}\n"
            for k in keys:
                val = cfg.get(k)
                if isinstance(val, list): val = ', '.join(map(str, val))
                config_text += f"   {k}: {val}\n"
            config_text += "\n"

        plt.gca().text(1.05, 1.0, config_text.strip(), transform=plt.gca().transAxes,
                       fontsize=9, va='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

        plt.title(f"Summerfield's Replication, {y.capitalize()} Comparison in {exp_name} Experiment", fontweight='bold')
        plt.xlabel(f'Batches ({n_epochs} epochs per stage in total)')
        plt.ylabel(y_axis_name)

        if log: plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)

        if self.save:
            safe_name = exp_name.replace(" ", "_")
            plt.savefig(f"{self.path}/{y}_comparison_{safe_name}_{self.variable}_{act_type}.png", bbox_inches='tight',
                        dpi=300)

        plt.show()

    def loss_graph(self):
        self.graph_temp1("losses", "BCE loss")

    def accuracy_graph(self):
        self.graph_temp1("accuracies", "Accuracy")


def added_config(stage, config):
    config["input_size"] = sum(config["features_types"]) + config["odd_dim"]
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["optimizer_type"] == "Adam":
        config["optimizer_type"] = optim.Adam
    if config["optimizer_type"] == "SGD":
        config["optimizer_type"] = optim.SGD
    odd, deciding_feature = False, 0
    if stage == "flexibility":
        odd, deciding_feature = False, 1
    if stage == "generalization":
        odd, deciding_feature = True, 0
    config["stage"] = stage
    config["odd"] = odd
    config["deciding_feature"] = deciding_feature

    return config
