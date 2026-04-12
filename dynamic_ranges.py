import os
import sys
import contextlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import plotly.graph_objects as go
from simulation import Simulation


@contextlib.contextmanager
def _suppress_output():
    old_out, old_err = sys.stdout, sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield old_err
        finally:
            sys.stdout, sys.stderr = old_out, old_err


class IDR_check:
    def __init__(self, sd=0, activation_type="Identity", optimization_type="Adam",
                 w_scale_low=0.1, w_scale_high=50, b_scale_low=0, b_scale_high=0, epochs=50, seed=0):
        self.sd, self.seed, self.epochs = sd, seed, epochs
        self.act_type, self.opt_type = activation_type, optimization_type
        self.w_l, self.w_h = w_scale_low, w_scale_high
        self.b_l, self.b_h = b_scale_low, b_scale_high

        self.config = {
            "exp_name": "idr_check", "features_types": [2], "hidden_size": 30, "n_hidden": 1,
            "b_scale_low": self.b_l, "b_scale_high": self.b_h, "w_scale_low": self.w_l, "w_scale_high": self.w_h,
            "optimizer_type": self.opt_type, "activation_type": self.act_type, "batch_size": 1,
            "seed": self.seed, "sd": self.sd,
            "exp_blocks": [{"block_name": "M1", "rule": "upper_half", "deciding_feature": 0,
                            "zero_features": (), "epochs": self.epochs, "alpha_class": 1, "alpha_rec": 0}]
        }

        self.t = torch.linspace(-0.5, 1.5, steps=100).view(-1, 1)
        self.save_dir = "figures/IDR_check"
        os.makedirs(self.save_dir, exist_ok=True)
        self._reset_data()

    def _reset_data(self):
        self.model_low_preds, self.model_high_preds = [], []
        self.low_params, self.high_params = [], []

    def get_data(self, seed):
        self.config["seed"] = seed
        results = Simulation(self.config).run(track_metrics=False)
        models = [results[0][1]["model_low"], results[0][1]["model_high"]]

        line_points = torch.tensor([0.0, 1.0]) + self.t * torch.tensor([1.0, -1.0])

        with torch.no_grad():
            preds = [torch.sigmoid(m(line_points))[:, 0].numpy().flatten() for m in models]

        self.model_low_preds.append(preds[0])
        self.model_high_preds.append(preds[1])

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

        ax.plot(t_vals, self.model_low_preds[-1], 'b-', label='Low Variance (RichMLP)', lw=2)
        ax.plot(t_vals, self.model_high_preds[-1], 'r-', label='High Variance (LazyMLP)', lw=2)
        ax.axvline(x=0.5, color='gray', linestyle='--', lw=1, label='Actual Midpoint (t=0.5)')

        for mu, color, label in [(self.low_params[-1][0], 'blue', 'Low Variance (RichMLP)'),
                                 (self.high_params[-1][0], 'red', 'High Variance (LazyMLP)')]:
            if not np.isnan(mu):
                ax.axvline(x=mu, color=color, linestyle='--', lw=1, alpha=0.3, label=f'{label} Fit $\\mu$={mu:.2f}')

        ax.set(title="Dynamic Range", xlabel="Input space on the line between (0,1) to (1,0)",
               ylabel="Model Prediction (Sigmoid)", xticks=[0, 0.5, 1], xticklabels=['(0,1)', '(0.5,0.5)', '(1,0)'])
        ax.title.set_weight('bold')
        ax.title.set_size(14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)

        txt_cfg = (f"$\\mathbf{{IDR\\ Simulation\\ Configurations}}$\n"
                   f"sd: {self.sd}\nAct: '{self.act_type}'\nOpt: '{self.opt_type}'\n"
                   f"Epochs: {self.epochs}\nSeed: {seed}\nw_l: {self.w_l}\nw_h: {self.w_h}\nb_l: {self.b_l}\nb_h: {self.b_h}")

        ax.text(1.045, 0.985, txt_cfg, transform=ax.transAxes, fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))
        ax.legend(loc='upper left', bbox_to_anchor=(1.03, 0.56), frameon=True, edgecolor='gray')

        fig.subplots_adjust(right=0.60)
        self._save_fig("sigmoids")
        plt.show()

    def plot_histograms(self, num_seeds=50):
        self._reset_data()

        with _suppress_output() as old_err:
            for s in tqdm(range(num_seeds), file=old_err, desc="Running Simulations"):
                self.get_data(s)

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

        cfg_txt = (f"$\\mathbf{{IDR\\ Configs:}}$ sd={self.sd}, act='{self.act_type}', opt='{self.opt_type}', "
                   f"epochs={self.epochs}, seeds={num_seeds}, w_scale=[{self.w_l}, {self.w_h}], b_scale=[{self.b_l}, {self.b_h}]")
        fig.text(0.5, 0.02, cfg_txt, ha='center', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))
        fig.subplots_adjust(bottom=0.15)

        self._save_fig("histograms")
        plt.show()

    def plot_sigmoid_SDs(self, samples_per_b_w=20, epochs_per_simulation=None,
                                   b_range=(0.1, 2), w_range=(0.1, 5), dots_density=10):
        w_vals, b_vals, sd_vals = [], [], []
        seed_counter = self.seed
        run_epochs = epochs_per_simulation if epochs_per_simulation is not None else self.epochs
        self.config["exp_blocks"][0]["epochs"] = run_epochs

        w_space = np.linspace(w_range[0], w_range[1], num=dots_density)
        b_space = np.linspace(b_range[0], b_range[1], num=dots_density)
        grid = [(w, b) for w in w_space for b in b_space]

        with _suppress_output() as old_err:
            with tqdm(total=len(grid), file=old_err, desc="Mapping Dynamic Ranges") as pbar:
                for i in range(0, len(grid), 2):
                    p1 = grid[i]
                    p2 = grid[i + 1] if i + 1 < len(grid) else p1

                    self.config.update({"w_scale_low": p1[0], "b_scale_low": p1[1], "w_scale_high": p2[0], "b_scale_high": p2[1]})

                    l_sds, h_sds = [], []
                    for _ in range(samples_per_b_w):
                        self._reset_data()
                        self.get_data(seed=seed_counter)
                        seed_counter += 1

                        if not np.isnan(l_sd := self.low_params[-1][1]): l_sds.append(l_sd)
                        if not np.isnan(h_sd := self.high_params[-1][1]): h_sds.append(h_sd)

                    if l_sds:
                        w_vals.append(p1[0]); b_vals.append(p1[1]); sd_vals.append(np.mean(l_sds))
                    pbar.update(1)

                    if h_sds and i + 1 < len(grid):
                        w_vals.append(p2[0]); b_vals.append(p2[1]); sd_vals.append(np.mean(h_sds))
                        pbar.update(1)

        self.config["exp_blocks"][0]["epochs"] = self.epochs

        fig = go.Figure(data=[go.Scatter3d(x=w_vals, y=b_vals, z=sd_vals, mode='markers',
            marker=dict(size=6, color=sd_vals, colorscale='Viridis', opacity=0.8,
                        line=dict(width=1, color='black'), colorbar=dict(title="Mean SD (σ)")))])

        fig.update_layout(title=f"Sigmoid SD vs. w_scale and b_scale<br>Activation: {self.act_type} | Epochs: {run_epochs}",
            scene=dict(xaxis_title="Weight Scale (w)", yaxis_title="Bias Scale (b)", zaxis_title="Mean SD (σ)"),
            margin=dict(l=0, r=0, b=0, t=50))

        file_path = os.path.join(self.save_dir, "sigmoid_sd_regression_3D.html")
        c = 1
        while os.path.exists(file_path):
            file_path = os.path.join(self.save_dir, f"sigmoid_sd_regression_3D_{c}.html"); c += 1
        fig.write_html(file_path)
        fig.show()