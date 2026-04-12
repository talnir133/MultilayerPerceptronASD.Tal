# RichMLP vs LazyMLP Dynamics Simulation

## Overview & The Learning Task
This project simulates and analyzes the learning dynamics of Multilayer Perceptrons (MLPs) by comparing two distinct network initialization regimes:
1. **Rich Regime (Low Variance):** The network initializes with a small variance in its weights and biases, encouraging it to actively learn features and build meaningful internal representations.
2. **Lazy Regime (High Variance):** The network initializes with a large variance. In this state, the network behaves similarly to a kernel method (Neural Tangent Kernel), relying mostly on memorization and local interpolation rather than extracting deep features.

### The Core Task: A Simple Baseline
At its core, the experiment tests how these two regimes learn a simple, deterministic binary classification task. 
* **The Data:** The input consists of synthetic, one-hot encoded categorical features. For example, a data point might represent an object defined by 3 features (like color, shape, and size), each having multiple possible categories.
* **The Rule:** The network must classify the data based on a straightforward, hidden rule. In the simplest scenario, we use an `upper_half` rule: the network looks at one specific "deciding feature" and outputs `1` if the active category of that feature falls in the lower half of its possible values, and `0` otherwise.

In this pure baseline setup, the network trains on clean, noiseless data to minimize a standard classification loss without any secondary objectives.

### Adding Complexity: Noise, Reconstruction, and Dynamic Blocks
Once the baseline is established, the simulation can introduce several layers of complexity to test the robustness and adaptability of the Rich vs. Lazy regimes:
* **Noise Injection:** Gaussian noise is added to the inputs. We then evaluate how well each model handles this uncertainty by comparing its predictions against the theoretical Bayes Optimal predictor.
* **Dual-Loss Optimization:** The network can be forced to multitask. By adjusting the `alpha_rec` parameter, the model is trained to not only classify the data but also *reconstruct* the original, clean input features simultaneously.
* **Dynamic Environments (Blocks):** Training occurs in continuous, sequential stages called **Blocks**. While the models retain their weights, the environment can suddenly change between blocks. A new block might introduce a different classification rule (like a `parity` rule) or mask out (zero) specific features entirely. This allows us to observe how quickly each regime "forgets" obsolete rules and adapts to new constraints.

---

## Project Structure
The codebase is highly modular, completely separating the core PyTorch machine learning logic from the experiment tracking and visualization:

* `core_ml.py`: The pure ML core. Contains the `Dataset` generator, the `MLP` architecture, and the minimal, stateless `train_model` loop.
* `classification_rules.py`: Contains the logic for generating data labels (e.g., `upper_half_rule`, `parity_rule`).
* `simulation.py`: The experiment manager. The `Simulation` class handles config parsing, global seed locking, and safely injects a tracking callback into the core training loop to collect metrics epoch-by-epoch.
* `analysis.py`: Contains the `SimulationAnalyzer` class, responsible for generating all plots and standard metrics based on the collected data.
* `dynamic_ranges.py`: Contains the `IDR_check` module, a specialized mathematical tool for analyzing the sigmoid dynamics and transition sharpness (Dynamic Range) across the continuous input space between classes.
* `gui_app.py`: A PyQt5-based Graphical User Interface for easily building and managing experiment configurations without editing code.
* `main.py`: The main entry point to run simulations (via GUI, JSON, or dictionary) and generate plots.

---

## Configuration & Degrees of Freedom
The entire simulation is controlled via a configuration dictionary (or JSON file). Here are the available degrees of freedom:

**1. Network Architecture & Initialization:**
* `hidden_size`: Number of neurons per hidden layer.
* `n_hidden`: Number of hidden layers (excluding the initial input-to-hidden layer).
* `w_scale_low`, `b_scale_low`: Weight and bias initialization standard deviation for the Rich (Low Variance) model.
* `w_scale_high`, `b_scale_high`: Weight and bias initialization standard deviation for the Lazy (High Variance) model.
* `activation_type`: The activation function to use (`"Identity"`, `"Tanh"`, `"RelU"`, `"Sigmoid"`).
* `optimizer_type`: The optimizer to use (`"Adam"`, `"SGD"`).
* `batch_size`: The batch size for the DataLoader.

**2. Data & Noise:**
* `features_types`: A list defining the dimensions of the categorical one-hot features (e.g., `[2, 2]`).
* `sd`: The standard deviation of the Gaussian noise added to the data during training.
* `seed`: Global random seed for total reproducibility.

**3. Experiment Blocks (Dynamic Shifts):**
A list of dictionaries (`exp_blocks`). Each block defines a specific training phase:
* `block_name`: A string identifier for the block (e.g., `"M1"`).
* `epochs`: Number of epochs to train this specific block.
* `zero_features`: A tuple/list of feature indices to mask (set to 0.0) during this block.
* `alpha_class`: Weight of the classification loss (BCE).
* `alpha_rec`: Weight of the reconstruction loss (BCE).
* `rule`: The classification rule to apply (`"upper_half"`, `"parity"`).
  * *Rule parameters:* If using `"upper_half"`, provide `deciding_feature` (index). If using `"parity"`, provide `feat_idx` (index).

---

## Getting Started

### Installation
This project uses `uv` and `pyproject.toml` for modern, deterministic dependency management. 

1. Ensure Python >= 3.11 and `uv` are installed (`pip install uv`).
2. Clone the repository and navigate to the project root.
3. Sync the environment:
```bash
uv sync --active
