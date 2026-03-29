import json
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from simulation import Simulation


class SimulationAnalyzer:
    """
    Analyzes and visualizes simulation results.
    Handles automatic directory creation and configuration saving upon initialization.
    """

    def __init__(self, results, config, save_figures=True):
        self.results = results
        self.config = config
        self.save_figures = save_figures
        self.exp_name = config.get("exp_name", "Experiment")
        self.save_dir = f"figures/{self.exp_name}"

        # Automatic directory creation and config dumping
        if self.save_figures:
            os.makedirs(self.save_dir, exist_ok=True)
            config_path = os.path.join(self.save_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)

    def _save_fig(self, base_name):
        """Saves a figure with an auto-incrementing suffix to avoid overwriting."""
        if not self.save_figures:
            return

        file_path = os.path.join(self.save_dir, f"{base_name}.png")
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(self.save_dir, f"{base_name}_{counter}.png")
            counter += 1

        plt.savefig(file_path, bbox_inches='tight', dpi=300)

    def _add_config_info(self, ax, show_config=True):
        """Overlays the experiment configuration details on the plot."""
        if not show_config:
            return

        cfg = self.config
        txt = r"$\mathbf{Simulation\ Configurations:}$" + "\n\n"

        categories = {
            "Input:": ['features_types', 'seed', 'sd'],
            "Network:": ['hidden_size', 'n_hidden', 'output_size', 'b_scale_low', 'b_scale_high',
                         'w_scale_low', 'w_scale_high', 'optimizer_type', 'activation_type', 'batch_size']
        }

        for title, keys in categories.items():
            txt += f"{title}\n"
            for k in keys:
                val = cfg.get(k)
                val_str = ', '.join(map(str, val)) if isinstance(val, list) else val
                txt += f"   {k}: {val_str}\n"
            txt += "\n"

        txt += "Experiment Blocks:\n"
        for idx, block in enumerate(cfg.get('exp_blocks', []), 1):
            zf = block.get('zero_features', [])
            zf_str = "None" if not zf else (",".join(map(str, zf)) if isinstance(zf, (list, tuple)) else str(zf))
            txt += f"   {idx}. {block.get('block_name', 'Unnamed')}, eps: {block.get('epochs', 0)}, feat: {block.get('deciding_feature', 0)}, zero: {zf_str}\n"

        ax.text(1.05, 0.985, txt.strip(), transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

    def _plot_metric_on_ax(self, ax, metric_name, ylabel, log_scale=False):
        """Core plotting engine for tracking metrics across continuous blocks."""
        low, high, opt, bounds, blocks = [], [], [], [0], []

        for name, res in self.results:
            low.extend(res["data_low"].get(metric_name, []))
            high.extend(res["data_high"].get(metric_name, []))
            if f"{metric_name}_optimal" in res["data_low"]:
                opt.extend(res["data_low"][f"{metric_name}_optimal"])

            bounds.append(len(low))
            blocks.append(name)

        ax.plot(low, label='Low Variance (RichMLP)', color='blue')
        ax.plot(high, label='High Variance (LazyMLP)', color='red')
        if self.config.get("sd", 0) > 0.0 and opt:
            ax.plot(opt, label='Bayes Optimal', color='green', alpha=0.5)

        x_offset = bounds[-1] * 0.02
        for i in range(1, len(bounds) - 1):
            ax.axvline(x=bounds[i], color='gray', linestyle='--', linewidth=1)
            ax.text(bounds[i] - x_offset, sum(ax.get_ylim()) / 2, 'Block Shift',
                    color='gray', fontsize=9, rotation=90, va='center', ha='right')

        for i in range(len(blocks)):
            ax.text((bounds[i] + bounds[i + 1]) / 2, -0.06, blocks[i],
                    transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=10, fontweight='bold', color='darkblue')

        ax.set(xlabel='Epochs', ylabel=ylabel)
        ax.xaxis.labelpad = 20
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.5)

    def _graph_template(self, metric_key, y_axis_name, log_scale=False, show_config=True):
        """Standardizes the layout for single-metric plots."""
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(right=0.65, bottom=0.15)

        self._plot_metric_on_ax(ax, metric_key, y_axis_name, log_scale)
        ax.set_title(y_axis_name, fontweight='bold', fontsize=14, pad=25)
        ax.text(0.5, 1.02, f"Simulation name: {self.exp_name.capitalize()}",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10, fontweight='normal')
        ax.legend(loc='lower left', bbox_to_anchor=(1.035, 0.0), frameon=True, edgecolor='gray')

        self._add_config_info(ax, show_config)
        self._save_fig(f"{metric_key}_figure_{self.exp_name.replace(' ', '_')}")
        plt.show()

    def plot_loss(self, sub_type="clean", log_scale=False, show_config=True):
        self._graph_template(f"losses_{sub_type}", f"{sub_type.capitalize()} BCE Loss", log_scale, show_config)

    def plot_accuracy(self, sub_type="clean", show_config=True):
        self._graph_template(f"accuracies_{sub_type}", f"{sub_type.capitalize()} Accuracy", False, show_config)

    def plot_mae(self, sub_type="clean", show_config=True):
        self._graph_template(f"MAE_{sub_type}", f"{sub_type.capitalize()} Mean Absolute Error", False, show_config)

    def plot_parameters_std(self, layer_name="fc1", show_config=True):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(right=0.75, bottom=0.15, wspace=0.25)

        self._plot_metric_on_ax(axs[0], f"{layer_name}_weight_sd", "Std Dev")
        axs[0].set_title("Weights Standard Deviation", fontweight='bold')

        self._plot_metric_on_ax(axs[1], f"{layer_name}_bias_sd", "Std Dev")
        axs[1].set_title("Biases Standard Deviation", fontweight='bold')

        fig.suptitle(f"{layer_name.upper()} Parameters Standard Deviation in {self.exp_name.capitalize()}",
                     fontweight='bold', fontsize=14)
        axs[1].legend(loc='lower left', bbox_to_anchor=(1.05, 0.0), frameon=True, edgecolor='gray')

        self._add_config_info(axs[1], show_config)
        self._save_fig(f"{layer_name}_std_figure_{self.exp_name.replace(' ', '_')}")
        plt.show()

    def plot_mds(self, epoch=-1, layer_name='fc_last', model_type='low', show_config=True):
        target_block_name, target_res, target_cfg, rel_epoch, current_epoch = None, None, None, 0, 0

        for b_name, b_res in self.results:
            n_epochs = len(b_res[f"data_{model_type}"]["activation_distances_clean"])
            if epoch == -1 or current_epoch <= epoch < current_epoch + n_epochs:
                target_block_name = b_name
                target_res = b_res[f"data_{model_type}"]
                target_cfg = b_res["config"]
                rel_epoch = (n_epochs - 1) if epoch == -1 else (epoch - current_epoch)
                break
            current_epoch += n_epochs

        if target_res is None:
            print(f"Epoch {epoch} not found in results.")
            return

        dist_mat = squareform(target_res["activation_distances_clean"][rel_epoch][layer_name])
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4).fit_transform(dist_mat)

        y = target_res["y"].cpu().numpy().flatten()
        X = target_res["X"].cpu().numpy()

        plt.figure(figsize=(9, 7))
        plt.subplots_adjust(right=0.65)
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, cmap='coolwarm', s=130, edgecolors='gray', alpha=0.85)
        self._add_config_info(plt.gca(), show_config)

        ft = self.config["features_types"]
        zf = target_cfg.get("zero_features", [])

        for i, coord in enumerate(coords):
            f_str, s_idx = [], 0
            for f_idx, dim in enumerate(ft):
                f_str.append("-" if f_idx in zf else str(np.argmax(X[i][s_idx:s_idx + dim])))
                s_idx += dim
            plt.annotate(f"({','.join(f_str)})", coord, xytext=(6, 6), textcoords='offset points',
                         fontsize=9, fontweight='bold', color='#444444')

        plt.margins(0.15)
        epoch_str = 'Last' if epoch == -1 else epoch
        plt.title(
            f"MDS of '{layer_name}' Activations ({model_type.capitalize()} Variance)\nBlock: {target_block_name} | Absolute Epoch: {epoch_str}",
            fontweight='bold')
        plt.legend(*scatter.legend_elements(), title="True Categories", loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)

        self._save_fig(f"MDS_{layer_name}_{model_type}_ep{epoch}")
        plt.show()


class IDR_check:
    def __init__(self, sd=0, activation_type="Identity", optimization_type="Adam", w_scale_low=0.1,
                 w_scale_high=50, b_scale_low=0, b_scale_high=0, epochs=100, seed=0):
        self.seed = seed
        self.config = {
            "exp_name": "idr_check", "features_types": [2], "hidden_size": 30, "n_hidden": 0, "output_size": 1,
            "b_scale_low": b_scale_low, "b_scale_high": b_scale_high, "w_scale_low": w_scale_low,
            "w_scale_high": w_scale_high, "optimizer_type": optimization_type, "activation_type": activation_type,
            "batch_size": 1, "seed": seed, "sd": sd,
            "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": (), "epochs": epochs}]
        }
        self.t = torch.linspace(-0.5, 1.5, steps=100).view(-1, 1)
        self._reset_data()

    def _reset_data(self):
        self.model_low_preds, self.model_high_preds = [], []
        self.low_params, self.high_params = [], []
        self.config["seed"] = self.seed

    def get_data(self, seed):
        self.config["seed"] = seed
        results = Simulation(self.config).run()
        models = [results[0][1]["model_low"], results[0][1]["model_high"]]

        # Calculate spatial interpolation line from coordinate (0,1) to (1,0)
        line_points = torch.tensor([0.0, 1.0]) + self.t * torch.tensor([1.0, -1.0])

        with torch.no_grad():
            preds = [torch.sigmoid(m(line_points)).numpy().flatten() for m in models]

        self.model_low_preds.append(preds[0])
        self.model_high_preds.append(preds[1])

        # Extract shift (mu) and dynamic range (sigma) by fitting a cumulative distribution function
        t_vals = self.t.flatten().numpy()
        for param_list, y_pred in zip((self.low_params, self.high_params), preds):
            try:
                popt, _ = curve_fit(norm.cdf, t_vals, y_pred, p0=[0.5, 0.1], bounds=([-2, 1e-5], [3, 5]))
                param_list.append(tuple(popt))
            except RuntimeError:
                param_list.append((np.nan, np.nan))

        return results

    def plot_sigmoids(self, seed=0):
        self._reset_data()
        results = self.get_data(seed)
        t_vals = self.t.flatten().numpy()

        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(right=0.65)

        plt.plot(t_vals, self.model_low_preds[-1], color='blue', label='Low Variance (RichMLP)', linewidth=2)
        plt.plot(t_vals, self.model_high_preds[-1], color='red', label='High Variance (LazyMLP)', linewidth=2)

        mu_low, mu_high = self.low_params[-1][0], self.high_params[-1][0]
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

        analyzer = SimulationAnalyzer(results, self.config, save_figures=False)
        analyzer._add_config_info(plt.gca())

        plt.legend(loc='lower left', bbox_to_anchor=(1.035, 0.0), frameon=True, edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.ylim(-0.05, 1.05)
        plt.show()

    def plot_histograms(self, num_seeds=50):
        self._reset_data()

        # 1. Run simulations with a progress bar bypassing the "black hole"
        with open(os.devnull, 'w') as devnull:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                for s in tqdm(range(num_seeds), file=old_err, desc="Running Simulations"):
                    self.get_data(s)
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        # 2. Extract valid parameters
        l_mu = [p[0] for p in self.low_params if not np.isnan(p[0])]
        l_sig = [p[1] for p in self.low_params if not np.isnan(p[1])]
        h_mu = [p[0] for p in self.high_params if not np.isnan(p[0])]
        h_sig = [p[1] for p in self.high_params if not np.isnan(p[1])]

        # 3. Create plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Sigmoid Parameters' Distributions", fontweight='bold', fontsize=16)

        def _plot_hist(ax, low_vals, high_vals, title, xlabel):
            all_v = low_vals + high_vals
            bins = np.linspace(min(all_v), max(all_v), 41) if all_v and max(all_v) > min(all_v) else 40
            ax.hist(low_vals, bins=bins, alpha=0.5, color='blue',
                    label=f'Low Var (Mean={np.mean(low_vals):.2f}, Var={np.var(low_vals):.4f})')
            ax.hist(high_vals, bins=bins, alpha=0.5, color='red',
                    label=f'High Var (Mean={np.mean(high_vals):.2f}, Var={np.var(high_vals):.4f})')
            ax.set(title=title, xlabel=xlabel, ylabel="Frequency")
            ax.legend()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        _plot_hist(axs[0], l_mu, h_mu, r"Distribution of Means ($\mu$)", r"$\mu$ value")
        _plot_hist(axs[1], l_sig, h_sig, r"Distribution of Standard Deviations ($\sigma$)", r"$\sigma$ value")

        plt.tight_layout()
        plt.show()