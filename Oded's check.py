import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=False, variance=1.0):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size,bias=bias)
        self.identity = nn.Identity()  # Identity activation function
        self.layer2 = nn.Linear(hidden_size, output_size, bias=bias)

        # Manual Xavier-like initialization for hidden layer (layer1)
        # Weights: normal distribution with mean 0 and std = sqrt(variance)
        std_dev = torch.sqrt(torch.tensor(variance, dtype=torch.float32))
        nn.init.xavier_normal_(self.layer1.weight,gain=variance)

    def forward(self, x):
        x = self.layer1(x)
        x = self.identity(x)
        x = self.layer2(x)
        return x


def generate_dataset(map_choice):
    zero_dataset = np.zeros((16,8),dtype=int)
    dataset = np.array(list(product([*np.eye(4)],[*np.eye(4)]))).reshape((16,8))
    return np.hstack([dataset,zero_dataset] if map_choice=="map1" else [zero_dataset,dataset])

def rule1_targets(dataset):
  return (dataset[:,0]+dataset[:,1]+dataset[:,8]+dataset[:,9])>0

def rule2_targets(dataset):
  return (dataset[:,4]+dataset[:,5]+dataset[:,12]+dataset[:,13])>0

m1_dataset = generate_dataset("map1")
m2_dataset = generate_dataset("map2")
rule1_targets = rule1_targets(m1_dataset)
rule2_targets = rule2_targets(m1_dataset)

print(m1_dataset)
print('\n', rule1_targets)
print('\n', rule2_targets)

def train_one_epoch(model, X, y, criterion, optimizer):
    n_samples = X.shape[0]
    # Evaluate accuracy on the full dataset before the epoch
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        all_outputs = model(X)
        predicted = (torch.sigmoid(all_outputs) > 0.5).float()
        correct = (predicted == y).sum().item()
        accuracy = correct / n_samples
    model.train() # Set model to training mode

    ##
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for input_data, target in dataloader:
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    return accuracy, loss.item()


# Instantiate the MLP model

input_size = 16
hidden_size = 30
output_size = 1

# Convert data to PyTorch tensors
X = torch.tensor(m1_dataset, dtype=torch.float32)

# Training parameters

batch_size = 1 # As requested
n_samples = X.shape[0]

all_accuracies = []
print("Starting training...")
for model_idx, var in enumerate([0.1, 50]):
  seed=42
  torch.manual_seed(seed)
  np.random.seed(seed)
  model =  SimpleMLP(input_size, hidden_size, output_size, variance=var, bias=False)
  epochs = 10
  y = torch.tensor(rule1_targets, dtype=torch.float32).unsqueeze(1) # Unsqueeze for BCEWithLogitsLoss

  # Define loss function and optimizer
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.004)
  accuracies = []

  print(f'Model {model_idx+1} (Variance={0.1 if model_idx == 0 else 50})')

  for epoch in tqdm(range(epochs)):
      accuracy, loss_item = train_one_epoch(model, X, y, criterion, optimizer)
      accuracies.append(accuracy)

      print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_item:.4f}, Accuracy: {accuracy:.4f}')
  all_accuracies.append(accuracies)
  print("Training finished.")
  print(f"Final Accuracy: {accuracies[-1]:.4f}")
  print("shifting rule...")
  y = torch.tensor(rule2_targets, dtype=torch.float32).unsqueeze(1) # Unsqueeze for BCEWithLogitsLoss
  epochs = 10
  print(f'Model {model_idx+1} (Variance={0.1 if model_idx == 0 else 50}): Rule Change Initial Accuracy: {accuracy:.4f}')
  for epoch in tqdm(range(epochs)):
      accuracy, loss_item = train_one_epoch(model, X, y, criterion, optimizer)
      accuracies.append(accuracy)

      print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_item:.4f}, Accuracy: {accuracy:.4f}')

  print("Training finished.")
  print(f"Final Accuracy: {accuracies[-1]:.4f}")

plt.figure(figsize=(10, 6))
epochs_range = range(0, 2*epochs) # Adjusted to include epoch 0

if len(all_accuracies) > 0:
    plt.plot(epochs_range, all_accuracies[0], label='Model with Low Variance (0.1)')
if len(all_accuracies) > 1:
    plt.plot(epochs_range, all_accuracies[1], label='Model with High Variance (50)')

plt.vlines(10,0.5,1,linestyles=":",colors="gray")
plt.title('Training Accuracy over Epochs for Different Initial Variances')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs_range)
plt.grid(True)
plt.legend()
plt.show()
print(all_accuracies)

