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


def train_mlp_model(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                    X, y, dataloader, optimizer_type=optim.Adam,
                    num_epochs=150, device=None, activation_type=None):
    """
    Initializes and trains an MLP model, capturing activation distances and loss history.

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
    data = {"input distances": None, "initial activations distances": None, "losses": [], "accuracies": [], "X": X, "y": y, "final predicted y": None}

    # Model setup
    activations = {}
    model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type=activation_type)
    model = model.to(device)
    model.reinitialize(seed=42)
    model.set_activations_hook(activations)
    criterion = nn.BCELoss()
    optimizer = optimizer_type(model.parameters(), lr=0.004)

    # Saving distances data before training
    model.eval()
    data["input distances"] = pairwise_distances(X.cpu().detach().numpy())[np.triu_indices(X.shape[0])]
    with torch.no_grad():
        model(X)  # filling "activations" with data
        data["initial activations distances"] = {k: pairwise_distances(v)[np.triu_indices(X.shape[0])] for k, v in activations.items()}
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
            data["accuracies"].append(1-torch.abs(outputs - labels).mean().item())

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        test_loss = criterion(y_pred, y)
        print(f'Final Loss: {test_loss.item() / len(dataloader):.4f}')
        data["final predicted y"] = y_pred

    return model, data


def create_dataset(features_types, odd_dim
                   , deciding_feature=0, odd=False, seed=0, device=None):
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

    X, y = st.get_data(deciding_feature, odd)
    X = X.to(device)
    y = y.to(device)

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return X, y, dataloader


class Figures():  # continue working on it
    def __init__(self, config, data_low, data_high):
        os.makedirs("figures", exist_ok=True)
        plt.figure(figsize=(10, 6))
        self.data_low = data_low
        self.data_high = data_high
        self.config = config

    def loss_graph(self):

        plt.plot(self.data_low["losses"], label=f'Low Variance Scale', color='blue')
        plt.plot(self.data_high["losses"], label=f'High Variance Scale)', color='red')

        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('BCE Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()

    def accuracy_graph(self):
        plt.plot(self.data_low["accuracies"], label=f'Low Variance Scale', color='blue')
        plt.plot(self.data_high["accuracies"], label=f'High Variance Scale)', color='red')

        plt.title('Training Accuracies Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.show()


