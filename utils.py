import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib

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
    def __init__(self, exp_initial, exp_odd=None, exp_flex=None):
        os.makedirs("figures", exist_ok=True)
        self.data = {"initial": exp_initial,
                     "odd": exp_odd,
                     "flex": exp_flex}
        if exp_initial:
            self.variable = "Weights" if exp_initial["config"]["b_scale_low"] == 0 else "Biases"

    def graph_temp1(self, y, y_axis_name, exps, log=False):
        act_type = self.data["initial"]["config"]["activation_type"]
        n_epochs = self.data["initial"]["config"]["num_epochs"]
        low = []
        high = []
        lines = []
        exp_name = 'initial condition'
        if len(exps)==2:
            if exps[1] == 'flex':
                exp_name = 'flexibility'
            if exps[1] == 'odd':
                exp_name = 'generalization'

        for exp_name in exps:
            exp = self.data[exp_name]
            low += exp["data_low"][y]
            high += exp["data_high"][y]
            lines.append(len(low))
        lines = lines[:-1]
        plt.figure(figsize=(12, 6))
        plt.plot(low, label=f'Low Variance (RichMLP)', color='blue')
        plt.plot(high, label=f'High Variance (LazyMLP)', color='red')
        for line in lines:
            plt.axvline(x=line, color='gray', linestyle='--', linewidth=1)
            plt.text(line - 5, sum(plt.ylim()) / 2, 'Shift-point', color='gray', fontsize=9, rotation=90,
                     verticalalignment='center', horizontalalignment='right')
        plt.text(1.02, 0.5, f'activation function: {act_type}',
                 rotation=270, fontsize=10, verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(
            f"Summerfield's Replication, {y.capitalize()} Comparison in {exp_name.capitalize()} Experiment, Different {self.variable} Variances ",
            fontweight='bold')
        plt.xlabel(f'Batches ({n_epochs} epochs per stage in total)')
        plt.ylabel(y_axis_name)
        if log:
            plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig(f"figures/{y}_comparison, {exp_name}, {self.variable}, {act_type}.png", bbox_inches='tight', dpi=300)
        plt.show()

    def loss_graph(self, exps):
        self.graph_temp1("losses", "BCE loss", exps)

    def accuracy_graph(self, exps):
        self.graph_temp1("accuracies", "Accuracy", exps)


def added_config(config, seed, device, odd, deciding_feature, unique_points_only):
    config["seed"] = seed
    config["input_size"] = sum(config["features_types"]) + config["odd_dim"]
    if device is None:
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["optimizer_type"] = optim.Adam if config["optimizer_type"] == "Adam" else optim.SGD
    config["odd"] = odd
    config["deciding_feature"] = deciding_feature
    config["unique_points_only"] = unique_points_only
    return config
