from collections import defaultdict
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics.pairwise import pairwise_distances
from datasets import GaussianTask
from models import MLP
ASD_COLOR = "#FF0000"
NT_COLOR = "#00A08A"

def sigmoid(x, thresh, slope):
    """
    Sigmoid function
    :param x: Input
    :param thresh: Threshold
    :param slope: Slope
    """
    return 1 / (1 + np.exp(-slope * (x - thresh)))


# Initialize the model, loss function, and optimizer
def train_mlp_model(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale,
                    X_train, y_train, X_test, y_test, dataloader, dg, grid, optimizer_type=optim.Adam,
                    num_epochs=300, device=None, activation_type=None):
    """
    Create and train the model using the given arguments
    :param input_size: The dimension of the input
    :param hidden_size: The size of the hidden layers
    :param n_hidden: The number of hidden layers
    :param output_size: The size of the output layer
    :param w_scale: The scale of the weights
    :param b_scale: The scale of the biases
    :param X_test: The test data
    :param y_test: The test labels
    :param dataloader: The DataLoader for the training data
    :param dg: The DataGenerator object
    :param grid: The grid for the centers on which the sigmoid will be fitted
    :param optimizer_type: The optimizer type
    :param num_epochs: The number of epochs for training
    :return: The trained model, the responses, the fitted parameters, and the covariance matrices of the parameters
    """

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)

    # Model setup
    resps = []
    activations = {}
    model = MLP(input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type=activation_type)
    model = model.to(device)
    model.reinitialize(seed=42)
    model.set_activations_hook(activations)
    criterion = nn.BCELoss()
    optimizer = optimizer_type(model.parameters(), lr=0.001)

    # Saving distances data before training
    model.eval()
    input_dist = pairwise_distances(X_train.cpu().detach().numpy())[np.triu_indices(X_train.shape[0])]
    with torch.no_grad():
        resps.append(model(grid_tensor).cpu().detach().numpy())
        model(X_train)  # filling "activations" with data
        layer_distances = {k: pairwise_distances(v)[np.triu_indices(X_train.shape[0])] for k, v in activations.items()}
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
        model.eval()
        with torch.no_grad():
            resp = model(grid_tensor).cpu().detach().numpy()
            resps.append(resp)
        model.train()

    # Test evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        print(f'Test Loss: {test_loss.item():.4f}')
    return model, resps, input_dist, layer_distances

def create_gaussian_dataset(input_size, num_samples, loc, scale, n_gaussians=2, seed=0, device=None):
    """
    Create a dataset using the DataGenerator
    :param input_size: The dimension of the input
    :param num_samples: The number of samples
    :param loc: The means of the Gaussians
    :param scale: The scale of the Gaussians
    :param n_gaussians: The number of Gaussians
    :param seed: The seed for the random number generator
    :return: X_train, X_test, y_train, y_test, The DataGenerator object, grid, training data DataLoader
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    dg = GaussianTask(input_size, n_gaussians, [-loc, loc], [scale, scale])

    X, y = dg.create(num_samples)
    X = X.to(device)
    y = y.to(device)

    grid = dg.get_centers_grid(150)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X_train, X_test, y_train, y_test, dg, grid, dataloader