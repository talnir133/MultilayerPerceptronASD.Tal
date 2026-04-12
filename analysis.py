import json
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")



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
            "Network:": ['hidden_size', 'n_hidden', 'b_scale_low', 'b_scale_high',
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
            rule_name = block.get('rule', 'upper_half')
            a_class = block.get('alpha_class', 1)
            a_rec = block.get('alpha_rec', 0)
            rule_params = [f"{k}={v}" for k, v in block.items() if
                           k not in ['block_name', 'epochs', 'zero_features', 'rule', 'alpha_class', 'alpha_rec']]
            params_str = f"({', '.join(rule_params)})" if rule_params else ""
            txt += f"   {idx}. {block.get('block_name', 'Unnamed')}, eps: {block.get('epochs', 0)}, zero: {zf_str}, a_c: {a_class}, a_r: {a_rec}\n"
            txt += f"      Rule: {rule_name} {params_str}\n"

        ax.text(1.05, 0.985, txt.strip(), transform=ax.transAxes, fontsize=7, va='top',
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

    def plot_mds(self, epoch=-1, layer_name='fc1', model_type='low', show_config=True):
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
        coords = MDS(n_components=2, metric='precomputed', random_state=42, n_init=4, init='random').fit_transform(dist_mat)

        y = target_res["y"][:,0].cpu().numpy().flatten()
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