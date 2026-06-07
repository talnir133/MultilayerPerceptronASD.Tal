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
                 w_scale_low=0.1, w_scale_high=50, b_scale_low=0, b_scale_high=0, epochs=50, seed=0, n_hidden=1, hidden_size=30, lr=0.04):
        self.sd, self.seed, self.epochs = sd, seed, epochs
        self.act_type, self.opt_type = activation_type, optimization_type
        self.w_l, self.w_h = w_scale_low, w_scale_high
        self.b_l, self.b_h = b_scale_low, b_scale_high

        self.config = {
            "exp_name": "idr_check", "features_types": [2], "hidden_size": hidden_size, "n_hidden": n_hidden,
            "b_scale_low": self.b_l, "b_scale_high": self.b_h, "w_scale_low": self.w_l, "w_scale_high": self.w_h,
            "optimizer_type": self.opt_type, "activation_type": self.act_type, "batch_size": 1,
            "seed": self.seed,
            "exp_blocks": [{"block_name": "M1", "sd": self.sd,"lr": lr, "rule": "upper_half", "deciding_feature": 0,
                            "zero_features": (), "epochs": self.epochs, "alpha_class": 1, "alpha_rec": 0}]
        }

        self.t = torch.linspace(-0.5, 1.5, steps=100).view(-1, 1)
        self.save_dir = "figures/IDR_check"
        os.makedirs(self.save_dir, exist_ok=True)
        self._reset_data()

    def _reset_data(self):
        self.model_low_preds, self.model_high_preds = [], []
        self.low_params, self.high_params = [], []
        self.low_errors, self.high_errors = [], []

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
        for param_list, error_list, y_pred in zip((self.low_params, self.high_params),
                                                  (self.low_errors, self.high_errors),
                                                  preds):
            try:
                popt, _ = curve_fit(norm.cdf, t_vals, y_pred, p0=[0.5, 0.1], bounds=([-2, 1e-5], [3, 5]))
                param_list.append(tuple(popt))

                # --- R-squared Calculation ---
                fitted_y = norm.cdf(t_vals, *popt)
                ss_res = np.sum((y_pred - fitted_y) ** 2)
                ss_tot = np.sum((y_pred - np.mean(y_pred)) ** 2)

                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                error_list.append(max(0, r2))

            except RuntimeError:
                param_list.append((np.nan, np.nan))
                error_list.append(np.nan)

        return results

    def plot_sigmoids(self, num_simulations=1, start_seed=0):
        from matplotlib.lines import Line2D
        from scipy.stats import norm
        self._reset_data()

        if num_simulations > 1:
            with _suppress_output() as old_err:
                for i in tqdm(range(num_simulations), file=old_err, desc="Running Simulations"): self.get_data(
                    seed=start_seed + i)
        else:
            self.get_data(seed=start_seed)

        t_vals = self.t.flatten().numpy()
        fig, ax = plt.subplots(figsize=(13, 6))

        l_preds, h_preds = np.array(self.model_low_preds), np.array(self.model_high_preds)
        l_mean, l_std = np.mean(l_preds, axis=0), np.std(l_preds, axis=0)
        h_mean, h_std = np.mean(h_preds, axis=0), np.std(h_preds, axis=0)

        l_r2, h_r2 = np.nanmean(self.low_errors), np.nanmean(self.high_errors)
        l_min_r2, h_min_r2 = np.nanmin(self.low_errors), np.nanmin(self.high_errors)

        v_l_mu, v_l_sig = [p[0] for p in self.low_params if not np.isnan(p[0])], [p[1] for p in self.low_params if
                                                                                  not np.isnan(p[1])]
        v_h_mu, v_h_sig = [p[0] for p in self.high_params if not np.isnan(p[0])], [p[1] for p in self.high_params if
                                                                                   not np.isnan(p[1])]

        l_fit_curve = norm.cdf(t_vals, np.mean(v_l_mu), np.mean(v_l_sig)) if v_l_mu else np.zeros_like(t_vals)
        h_fit_curve = norm.cdf(t_vals, np.mean(v_h_mu), np.mean(v_h_sig)) if v_h_mu else np.zeros_like(t_vals)

        line_l_emp, = ax.plot(t_vals, l_mean, color='dodgerblue', linestyle='-', lw=2, alpha=0.3)
        ax.fill_between(t_vals, l_mean - l_std, l_mean + l_std, color='dodgerblue', alpha=0.1)
        line_l_fit, = ax.plot(t_vals, l_fit_curve, color='darkblue', linestyle='-', lw=2.5)

        line_h_emp, = ax.plot(t_vals, h_mean, color='salmon', linestyle='-', lw=2, alpha=0.3)
        ax.fill_between(t_vals, h_mean - h_std, h_mean + h_std, color='salmon', alpha=0.1)
        line_h_fit, = ax.plot(t_vals, h_fit_curve, color='darkred', linestyle='-', lw=2.5)

        line_mid = ax.axvline(x=0.5, color='gray', linestyle='--', lw=1, label='Actual Midpoint (t=0.5)')

        if v_l_mu: ax.axvline(x=np.mean(v_l_mu), color='darkblue', linestyle='--', lw=1, alpha=0.5)
        if v_h_mu: ax.axvline(x=np.mean(v_h_mu), color='darkred', linestyle='--', lw=1, alpha=0.5)

        def get_stat_str(name, vals):
            res = f"{name} Avg: {np.mean(vals):.2f}" if vals else f"{name} Avg: N/A"
            return res + (f" (Var: {np.var(vals):.4f})" if num_simulations > 1 and len(vals) > 1 else "")

        handles = [
            Line2D([], [], color='none', label=r'$\bf{Low\ Variance\ Model:}$'),
            Line2D([], [], color='darkblue', lw=2.5, label=r'Parametric Fit (Mean $\mu, \sigma$)'),
            Line2D([], [], color='dodgerblue', lw=2, alpha=0.3, label='Mean Across Sigmoids'),
            Line2D([], [], color='none', label=get_stat_str(r"Fit $\mu$", v_l_mu)),
            Line2D([], [], color='none', label=get_stat_str(r"Fit $\sigma$", v_l_sig)),
            Line2D([], [], color='none', label=""),
            Line2D([], [], color='none', label=r'$\bf{High\ Variance\ Model:}$'),
            Line2D([], [], color='darkred', lw=2.5, label=r'Parametric Fit (Mean $\mu, \sigma$)'),
            Line2D([], [], color='salmon', lw=2, alpha=0.3, label='Mean Across Sigmoids'),
            Line2D([], [], color='none', label=get_stat_str(r"Fit $\mu$", v_h_mu)),
            Line2D([], [], color='none', label=get_stat_str(r"Fit $\sigma$", v_h_sig)),
            Line2D([], [], color='none', label=""), line_mid
        ]

        ax.set(title=f"Dynamic Range ({num_simulations} Runs Averaged)",
               xlabel="Input space on the line between (0,1) to (1,0)", ylabel="Model Prediction (Sigmoid)",
               xticks=[0, 0.5, 1], xticklabels=['(0,1)', '(0.5,0.5)', '(1,0)'])
        ax.title.set_weight('bold');
        ax.title.set_size(14)
        ax.set_ylim(-0.05, 1.05);
        ax.grid(True, linestyle='--', alpha=0.4)

        r2_str = f"Low Var: Avg {l_r2:.2f}" + (
            f" (Worst: {l_min_r2:.2f})" if num_simulations > 1 else "") + f"\nHigh Var: Avg {h_r2:.2f}" + (
                     f" (Worst: {h_min_r2:.2f})" if num_simulations > 1 else "")
        ax.text(0.02, 0.95, f"$\\mathbf{{Fit\\ Reliability\\ (R^2)}}$\n{r2_str}", transform=ax.transAxes, fontsize=10,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

        txt_cfg = (
            f"$\\mathbf{{IDR\\ Simulation\\ Configurations}}$\nRuns: {num_simulations}\nsd: {self.sd}\nAct: '{self.act_type}'\nOpt: '{self.opt_type}'\nEpochs: {self.epochs}\nStart Seed: {start_seed}\nw_l: {self.w_l}\nw_h: {self.w_h}\nb_l: {self.b_l}\nb_h: {self.b_h}")
        ax.text(1.02, 0.985, txt_cfg, transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))

        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=True, edgecolor='gray',
                  fontsize=8)
        fig.subplots_adjust(right=0.60)
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
        plt.show()

    def plot_sigmoid_SDs(self, samples_per_b_w=20, epochs_per_simulation=None,
                         b_range=(0.1, 2), w_range=(0.1, 5), dots_density=10):
        """Maps the Dynamic Range (SD) across a grid of weight and bias initialization scales."""
        w_vals, b_vals, sd_vals, r2_vals = [], [], [], []
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

                    self.config.update({"w_scale_low": p1[0], "b_scale_low": p1[1],
                                        "w_scale_high": p2[0], "b_scale_high": p2[1]})

                    l_sds, h_sds, l_r2s, h_r2s = [], [], [], []
                    for _ in range(samples_per_b_w):
                        self._reset_data()
                        self.get_data(seed=seed_counter)
                        seed_counter += 1

                        if not np.isnan(l_sd := self.low_params[-1][1]):
                            l_sds.append(l_sd)
                            l_r2s.append(self.low_errors[-1])

                        if not np.isnan(h_sd := self.high_params[-1][1]):
                            h_sds.append(h_sd)
                            h_r2s.append(self.high_errors[-1])

                    if l_sds:
                        w_vals.append(p1[0]);
                        b_vals.append(p1[1]);
                        sd_vals.append(np.mean(l_sds))
                        r2_vals.append(np.mean(l_r2s))
                    pbar.update(1)

                    if h_sds and i + 1 < len(grid):
                        w_vals.append(p2[0]);
                        b_vals.append(p2[1]);
                        sd_vals.append(np.mean(h_sds))
                        r2_vals.append(np.mean(h_r2s))
                        pbar.update(1)

        self.config["exp_blocks"][0]["epochs"] = self.epochs

        # Calculate statistics for the display title (converted to percentages)
        avg_r2 = np.nanmean(r2_vals) * 100
        min_r2 = np.nanmin(r2_vals) * 100

        # Create interactive Plotly 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=w_vals, y=b_vals, z=sd_vals, mode='markers',
            marker=dict(size=6, color=sd_vals, colorscale='Viridis', opacity=0.8,
                        line=dict(width=1, color='black'), colorbar=dict(title="Mean SD (σ)"))
        )])

        # Set title with Fit Reliability info
        fit_info = f"<br><br><sup>Activation Function: {self.act_type} | Epochs: {run_epochs}<br>Fit Reliability (R²): Average {avg_r2:.1f}% | Worst Fit: {min_r2:.1f}%</sup>"
        fig.update_layout(
            title=f"Dynamic Range (fitted Gaussian CDF's sd) vs. Weights & Bias Scales{fit_info}",
            scene=dict(xaxis_title="Weight Scale (w)", yaxis_title="Bias Scale (b)", zaxis_title="Mean SD (σ)"),
            margin=dict(l=0, r=0, b=0, t=80)
        )

        file_path = os.path.join(self.save_dir, "sigmoid_sd_regression_3D.html")
        c = 1
        while os.path.exists(file_path):
            file_path = os.path.join(self.save_dir, f"sigmoid_sd_regression_3D_{c}.html");
            c += 1
        fig.write_html(file_path)
        fig.show()