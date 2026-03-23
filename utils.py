import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import warnings
from scipy.optimize import curve_fit
from scipy.stats import norm

warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use('TkAgg')
from sklearn.metrics.pairwise import pairwise_distances
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
        "fc1_weight_sd": [],
        "fc1_bias_sd": [],
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
            data["fc1_weight_sd"].append(model._layers.fc1.weight.std().item())
            if model._layers.fc1.bias is not None:
                data["fc1_bias_sd"].append(model._layers.fc1.bias.std().item())
            else:
                data["fc1_bias_sd"].append(0.0)
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
    def __init__(self, results, config, save=True):
        self.config, self.save, self.results = config, save, results
        self.path = f"figures/{config.get('exp_name', 'Experiment')}"

    def _save_fig(self, base_name):
        if not self.save: return
        p, i = f"{self.path}/{base_name}.png", 1
        while os.path.exists(p):
            p, i = f"{self.path}/{base_name}_{i}.png", i + 1
        plt.savefig(p, bbox_inches='tight', dpi=300)

    def _add_config_info(self, ax, show_config=True):
        if not show_config: return
        cfg = self.config
        txt = r"$\mathbf{Simulation's\ Configurations:}$" + "\n\n"
        cats = {
            "Input:": ['features_types', 'seed', 'sd'],
            "Network:": ['hidden_size', 'n_hidden', 'output_size', 'b_scale_low', 'b_scale_high', 'w_scale_low',
                         'w_scale_high', 'optimizer_type', 'activation_type', 'batch_size']
        }
        for t, keys in cats.items():
            txt += f"{t}\n"
            for k in keys:
                v = cfg.get(k)
                txt += f"   {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}\n"
            txt += "\n"

        txt += "Experiment Blocks:\n"
        for idx, b in enumerate(cfg.get('exp_blocks', []), 1):
            zf = b.get('zero_features', [])
            zfs = "None" if not zf else (",".join(map(str, zf)) if isinstance(zf, (list, tuple)) else str(zf))
            txt += f"   {idx}. {b.get('block_name', 'Unnamed')}, eps: {b.get('epochs', 0)}, feat: {b.get('deciding_feature', 0)}, zero: {zfs}\n"

        ax.text(1.05, 0.985, txt.strip(), transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

    def _plot_metric_on_ax(self, ax, metric_name, ylabel, log=False):
        low, high, opt, bounds, blocks = [], [], [], [0], []
        for name, res in self.results:
            low += res["data_low"].get(metric_name, [])
            high += res["data_high"].get(metric_name, [])
            if f"{metric_name}_optimal" in res["data_low"]:
                opt += res["data_low"][f"{metric_name}_optimal"]
            bounds.append(len(low))
            blocks.append(name)

        ax.plot(low, label='Low Variance (RichMLP)', color='blue')
        ax.plot(high, label='High Variance (LazyMLP)', color='red')
        if self.config.get("sd", 0) > 0.0 and opt:
            ax.plot(opt, label='Bayes Optimal', color='green', alpha=0.5)

        xo = bounds[-1] * 0.02
        for i in range(1, len(bounds) - 1):
            ax.axvline(x=bounds[i], color='gray', linestyle='--', linewidth=1)
            ax.text(bounds[i] - xo, sum(ax.get_ylim()) / 2, 'Block Shift', color='gray', fontsize=9, rotation=90,
                    va='center', ha='right')

        for i in range(len(blocks)):
            ax.text((bounds[i] + bounds[i + 1]) / 2, -0.06, blocks[i], transform=ax.get_xaxis_transform(), ha='center',
                    va='top', fontsize=10, fontweight='bold', color='darkblue')

        ax.set(xlabel='Epochs', ylabel=ylabel)
        ax.xaxis.labelpad = 20
        if log: ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.5)

    def graph_temp1(self, y, y_axis_name, log=False, show_config=True):
        exp = self.config.get("exp_name", "Experiment").capitalize()
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(right=0.65, bottom=0.15)

        self._plot_metric_on_ax(ax, y, y_axis_name, log)
        ax.set_title(f"{y_axis_name} Comparison in {exp}", fontweight='bold')
        ax.legend(loc='lower left', bbox_to_anchor=(1.035, 0.0), frameon=True, edgecolor='gray')
        self._add_config_info(ax, show_config)
        self._save_fig(f"{y}_figure_{exp.replace(' ', '_')}")
        plt.show()

    def loss_graph(self, sub_type="clean"):
        self.graph_temp1(f"losses_{sub_type}", f"{sub_type.capitalize()} BCE loss")

    def accuracy_graph(self, sub_type="clean"):
        self.graph_temp1(f"accuracies_{sub_type}", f"{sub_type.capitalize()} Accuracy")

    def MAE_graph(self, sub_type="clean"):
        self.graph_temp1(f"MAE_{sub_type}", f"{sub_type.capitalize()} Mean Absolute Error")

    def parameters_std_graph(self, layer_name="fc1", show_config=True):
        exp = self.config.get("exp_name", "Experiment").capitalize()
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(right=0.75, bottom=0.15, wspace=0.25)

        self._plot_metric_on_ax(axs[0], f"{layer_name}_weight_sd", "Std Dev")
        axs[0].set_title("Weights Standard Deviation", fontweight='bold')

        self._plot_metric_on_ax(axs[1], f"{layer_name}_bias_sd", "Std Dev")
        axs[1].set_title("Biases Standard Deviation", fontweight='bold')

        fig.suptitle(f"{layer_name.upper()} Parameters Standard Deviation in {exp}", fontweight='bold', fontsize=14)
        axs[1].legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), frameon=True, edgecolor='gray')
        self._add_config_info(axs[1], show_config)
        self._save_fig(f"{layer_name}_std_figure_{exp.replace(' ', '_')}")
        plt.show()

    def mds_graph(self, epoch=-1, layer_name='fc_last', model_type='low', show_config=True):
        tb, t_res, t_cfg, rel_ep, cur_ep = None, None, None, 0, 0
        for b_name, b_res in self.results:
            n_ep = len(b_res[f"data_{model_type}"]["activation_distances_clean"])
            if epoch == -1 or cur_ep <= epoch < cur_ep + n_ep:
                tb, t_res, t_cfg = b_name, b_res[f"data_{model_type}"], b_res["config"]
                rel_ep = (n_ep - 1) if epoch == -1 else (epoch - cur_ep)
                break
            cur_ep += n_ep

        if t_res is None: return

        dist_mat = squareform(t_res["activation_distances_clean"][rel_ep][layer_name])
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4).fit_transform(dist_mat)
        y, X = t_res["y"].cpu().numpy().flatten(), t_res["X"].cpu().numpy()

        plt.figure(figsize=(9, 7))
        plt.subplots_adjust(right=0.65)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap='coolwarm', s=130, edgecolors='gray', alpha=0.85)
        self._add_config_info(plt.gca(), show_config)

        ft, zf = self.config["features_types"], t_cfg.get("zero_features", [])
        for i, coord in enumerate(coords):
            f_str, s_idx = [], 0
            for f_idx, dim in enumerate(ft):
                f_str.append("-" if f_idx in zf else str(np.argmax(X[i][s_idx:s_idx + dim])))
                s_idx += dim
            plt.annotate(f"({','.join(f_str)})", coord, xytext=(6, 6), textcoords='offset points', fontsize=9,
                         fontweight='bold', color='#444444')

        plt.margins(0.15)
        plt.title(
            f"MDS of '{layer_name}' Activations ({model_type.capitalize()} Variance)\nBlock: {tb} | Absolute Epoch: {epoch if epoch != -1 else 'Last'}",
            fontweight='bold')
        plt.legend(*scatter.legend_elements(), title="True Categories", loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        self._save_fig(f"MDS_{layer_name}_{model_type}_ep{epoch}")
        plt.show()


class IDR_check():
    def __init__(self, sd=0, activation_type="Identity", optimization_type="Adam", w_scale_low=0.1, w_scale_high=50,
                 b_scale_low=0, b_scale_high=0, epochs=100, seed=0):
        self.seed = seed
        self.config = {
            "exp_name": "idr_check", "features_types": [2], "hidden_size": 30, "n_hidden": 0, "output_size": 1,
            "b_scale_low": b_scale_low, "b_scale_high": b_scale_high, "w_scale_low": w_scale_low,
            "w_scale_high": w_scale_high,
            "optimizer_type": optimization_type, "activation_type": activation_type, "batch_size": 1, "seed": seed,
            "sd": sd,
            "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": (), "epochs": epochs}]
        }
        self.t = torch.linspace(-0.5, 1.5, steps=100).view(-1, 1)
        self._reset_data()

    def _reset_data(self):
        self.model_low_preds = []
        self.model_high_preds = []
        self.low_params = []
        self.high_params = []
        self.config["seed"] = self.seed

    def _cdf_func(self, x, mu, sigma):
        return norm.cdf(x, loc=mu, scale=sigma)

    def get_data(self, seed):
        from simulations import run_experiment
        self.config["seed"] = seed
        results = run_experiment(self.config)
        model_low, model_high = results[0][1]["model_low"], results[0][1]["model_high"]

        start_point, end_point = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
        line_points = start_point + self.t * (end_point - start_point)

        with torch.no_grad():
            low_preds = torch.sigmoid(model_low(line_points)).numpy().flatten()
            high_preds = torch.sigmoid(model_high(line_points)).numpy().flatten()

        self.model_low_preds.append(low_preds)
        self.model_high_preds.append(high_preds)

        t_vals = self.t.flatten().numpy()

        try:
            popt_low, _ = curve_fit(self._cdf_func, t_vals, low_preds, p0=[0.5, 0.1], bounds=([-2, 1e-5], [3, 5]))
            self.low_params.append(tuple(popt_low))
        except RuntimeError:
            self.low_params.append((np.nan, np.nan))
        try:
            popt_high, _ = curve_fit(self._cdf_func, t_vals, high_preds, p0=[0.5, 0.1], bounds=([-2, 1e-5], [3, 5]))
            self.high_params.append(tuple(popt_high))
        except RuntimeError:
            self.high_params.append((np.nan, np.nan))

        return results

    def plot_sigmoids(self, seed=0):
        self._reset_data()
        results = self.get_data(seed)
        t_vals = self.t.flatten().numpy()

        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.65)

        plt.plot(t_vals, self.model_low_preds[-1], color='blue', label='Low Variance (RichMLP)', linewidth=2)
        plt.plot(t_vals, self.model_high_preds[-1], color='red', label='High Variance (LazyMLP)', linewidth=2)

        mu_low = self.low_params[-1][0]
        mu_high = self.high_params[-1][0]

        plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Actual Midpoint (t=0.5)')
        if not np.isnan(mu_low):
            plt.axvline(x=mu_low, color='blue', linestyle='--', linewidth=1, alpha=0.3,
                        label=f'RichMLP Fit $\\mu$={mu_low:.2f}')
        if not np.isnan(mu_high):
            plt.axvline(x=mu_high, color='red', linestyle='--', linewidth=1, alpha=0.3,
                        label=f'LazyMLP Fit $\\mu$={mu_high:.2f}')

        plt.title("Dynamic Range", fontweight='bold', fontsize=14)
        plt.xlabel("Input space on the line between (0,1) to (1,0)", fontsize=11)
        plt.ylabel("Model Prediction (Sigmoid)", fontsize=11)
        plt.xticks([0, 0.5, 1], ['(0,1)', '(0.5,0.5)', '(1,0)'])

        Figures(results, self.config)._add_config_info(plt.gca())

        plt.legend(loc='lower left', bbox_to_anchor=(1.035, 0.0), frameon=True, edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.ylim(-0.05, 1.05)
        plt.show()

    def plot_histograms(self, num_seeds=50):
        import sys, os
        from tqdm import tqdm
        from matplotlib.ticker import MaxNLocator

        self._reset_data()
        orig_out, orig_err = sys.stdout, sys.stderr

        try:
            with open(os.devnull, 'w') as devnull:
                sys.stdout, sys.stderr = devnull, devnull
                pbar = tqdm(total=num_seeds, file=orig_err, desc="Running Simulations")
                for s in range(num_seeds):
                    self.get_data(seed=s)
                    pbar.update(1)
                pbar.close()
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err

        l_mu = [p[0] for p in self.low_params if not np.isnan(p[0])]
        l_sig = [p[1] for p in self.low_params if not np.isnan(p[1])]
        h_mu = [p[0] for p in self.high_params if not np.isnan(p[0])]
        h_sig = [p[1] for p in self.high_params if not np.isnan(p[1])]

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        a_mu, a_sig = l_mu + h_mu, l_sig + h_sig

        b_mu = np.linspace(min(a_mu), max(a_mu), 41) if a_mu and max(a_mu) > min(a_mu) else 40
        axs[0].hist(l_mu, bins=b_mu, alpha=0.5, color='blue',
                    label=f'Low Variance (Mean={np.mean(l_mu):.2f}, Var={np.var(l_mu):.4f})')
        axs[0].hist(h_mu, bins=b_mu, alpha=0.5, color='red',
                    label=f'High Variance (Mean={np.mean(h_mu):.2f}, Var={np.var(h_mu):.4f})')
        axs[0].set_title(r"Distribution of Means ($\mu$)")
        axs[0].set(xlabel=r"$\mu$ value", ylabel="Frequency")
        axs[0].legend()
        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))

        b_sig = np.linspace(min(a_sig), max(a_sig), 41) if a_sig and max(a_sig) > min(a_sig) else 40
        axs[1].hist(l_sig, bins=b_sig, alpha=0.5, color='blue',
                    label=f'Low Variance (Mean={np.mean(l_sig):.2f}, Var={np.var(l_sig):.4f})')
        axs[1].hist(h_sig, bins=b_sig, alpha=0.5, color='red',
                    label=f'High Variance (Mean={np.mean(h_sig):.2f}, Var={np.var(h_sig):.4f})')
        axs[1].set_title(r"Distribution of Standard Deviations ($\sigma$)")
        axs[1].set(xlabel=r"$\sigma$ value")
        axs[1].legend()
        axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.show()
