import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from simulation import Simulation


class IDR_check:
    def __init__(self, sd=0, activation_type="Identity", optimization_type="Adam", w_scale_low=0.1,
                 w_scale_high=50, b_scale_low=0, b_scale_high=0, epochs=100, seed=0):
        self.sd = sd
        self.act_type = activation_type
        self.opt_type = optimization_type
        self.w_l, self.w_h = w_scale_low, w_scale_high
        self.b_l, self.b_h = b_scale_low, b_scale_high
        self.epochs = epochs
        self.seed = seed

        self.config = {
            "exp_name": "idr_check", "features_types": [2], "hidden_size": 30, "n_hidden": 0, "output_size": 1,
            "b_scale_low": b_scale_low, "b_scale_high": b_scale_high, "w_scale_low": w_scale_low,
            "w_scale_high": w_scale_high, "optimizer_type": optimization_type, "activation_type": activation_type,
            "batch_size": 1, "seed": seed, "sd": sd,
            "exp_blocks": [{"block_name": "M1", "rule": "upper_half", "deciding_feature": 0, "zero_features": (), "epochs": epochs}]
        }
        self.t = torch.linspace(-0.5, 1.5, steps=100).view(-1, 1)
        self.save_dir = "figures/IDR_check"
        os.makedirs(self.save_dir, exist_ok=True)
        self._reset_data()

    def _reset_data(self):
        self.model_low_preds, self.model_high_preds = [], []
        self.low_params, self.high_params = [], []
        self.config["seed"] = self.seed

    def get_data(self, seed):
        self.config["seed"] = seed
        results = Simulation(self.config).run()
        models = [results[0][1]["model_low"], results[0][1]["model_high"]]

        # Spatial interpolation line from (0,1) to (1,0)
        line_points = torch.tensor([0.0, 1.0]) + self.t * torch.tensor([1.0, -1.0])

        with torch.no_grad():
            preds = [torch.sigmoid(m(line_points)).numpy().flatten() for m in models]

        self.model_low_preds.append(preds[0])
        self.model_high_preds.append(preds[1])

        # Extract shift (mu) and dynamic range (sigma) via CDF curve fitting
        t_vals = self.t.flatten().numpy()
        for param_list, y_pred in zip((self.low_params, self.high_params), preds):
            try:
                popt, _ = curve_fit(norm.cdf, t_vals, y_pred, p0=[0.5, 0.1], bounds=([-2, 1e-5], [3, 5]))
                param_list.append(tuple(popt))
            except RuntimeError:
                param_list.append((np.nan, np.nan))

        return results

    def _save_fig(self, base_name):
        file_path = os.path.join(self.save_dir, f"{base_name}.png")
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(self.save_dir, f"{base_name}_{counter}.png")
            counter += 1
        plt.savefig(file_path, bbox_inches='tight', dpi=300)

    def plot_sigmoids(self, seed=0):
        self._reset_data()
        self.get_data(seed)
        t_vals = self.t.flatten().numpy()

        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(t_vals, self.model_low_preds[-1], color='blue', label='Low Variance (RichMLP)', linewidth=2)
        ax.plot(t_vals, self.model_high_preds[-1], color='red', label='High Variance (LazyMLP)', linewidth=2)

        mu_low, mu_high = self.low_params[-1][0], self.high_params[-1][0]
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, label='Actual Midpoint (t=0.5)')

        if not np.isnan(mu_low):
            ax.axvline(x=mu_low, color='blue', linestyle='--', linewidth=1, alpha=0.3,
                       label=f'Low Variance (RichMLP) Fit $\\mu$={mu_low:.2f}')
        if not np.isnan(mu_high):
            ax.axvline(x=mu_high, color='red', linestyle='--', linewidth=1, alpha=0.3,
                       label=f'High Variance (LazyMLP) Fit $\\mu$={mu_high:.2f}')

        ax.set_title("Dynamic Range", fontweight='bold', fontsize=14)
        ax.set_xlabel("Input space on the line between (0,1) to (1,0)", fontsize=11)
        ax.set_ylabel("Model Prediction (Sigmoid)", fontsize=11)
        ax.set_xticks([0, 0.5, 1], ['(0,1)', '(0.5,0.5)', '(1,0)'])

        # Added spaces at the end of the title to force the bounding box to stretch and match the legend width
        txt_multi = (
                r"$\mathbf{IDR\ Simulation\ Configurations}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ $" + "\n"
                                                                                                f"sd: {self.sd}\n"
                                                                                                f"Activation: '{self.act_type}'\n"
                                                                                                f"Optimizer: '{self.opt_type}'\n"
                                                                                                f"Epochs: {self.epochs}\n"
                                                                                                f"Seed: {seed}\n"
                                                                                                f"w_scale_low: {self.w_l}\n"
                                                                                                f"w_scale_high: {self.w_h}\n"
                                                                                                f"b_scale_low: {self.b_l}\n"
                                                                                                f"b_scale_high: {self.b_h}"
        )

        # Shifted slightly right (1.045) to align the left border perfectly with the legend
        ax.text(1.045, 0.985, txt_multi, transform=ax.transAxes, fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

        # Placed the legend directly underneath the config box (loc='upper left', Y=0.56)
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 0.56), frameon=True, edgecolor='gray')

        fig.subplots_adjust(right=0.60)

        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_ylim(-0.05, 1.05)

        self._save_fig("sigmoids")
        plt.show()

    def plot_histograms(self, num_seeds=50):
        self._reset_data()

        with open(os.devnull, 'w') as devnull:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                for s in tqdm(range(num_seeds), file=old_err, desc="Running Simulations"):
                    self.get_data(s)
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        l_mu = [p[0] for p in self.low_params if not np.isnan(p[0])]
        l_sig = [p[1] for p in self.low_params if not np.isnan(p[1])]
        h_mu = [p[0] for p in self.high_params if not np.isnan(p[0])]
        h_sig = [p[1] for p in self.high_params if not np.isnan(p[1])]

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Sigmoid Parameters' Distributions", fontweight='bold', fontsize=16)

        def _plot_hist(ax, low_vals, high_vals, title, xlabel):
            all_v = low_vals + high_vals
            bins = np.linspace(min(all_v), max(all_v), 41) if all_v and max(all_v) > min(all_v) else 40
            ax.hist(low_vals, bins=bins, alpha=0.5, color='blue', label=f'Low Var (Mean={np.mean(low_vals):.2f})')
            ax.hist(high_vals, bins=bins, alpha=0.5, color='red', label=f'High Var (Mean={np.mean(high_vals):.2f})')
            ax.set(title=title, xlabel=xlabel, ylabel="Frequency")
            ax.legend()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        _plot_hist(axs[0], l_mu, h_mu, r"Distribution of Means ($\mu$)", r"$\mu$ value")
        _plot_hist(axs[1], l_sig, h_sig, r"Distribution of Standard Deviations ($\sigma$)", r"$\sigma$ value")

        txt_single = (
            r"$\mathbf{IDR\ Simulation\ Configurations:}$ "
            f"sd={self.sd}, activation_type='{self.act_type}', optimization_type='{self.opt_type}', "
            f"epochs={self.epochs}, num_seeds={num_seeds}, "
            f"w_scale_low={self.w_l}, w_scale_high={self.w_h}, b_scale_low={self.b_l}, b_scale_high={self.b_h}"
        )

        fig.text(0.5, 0.02, txt_single, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

        fig.subplots_adjust(bottom=0.15)

        self._save_fig("histograms")
        plt.show()