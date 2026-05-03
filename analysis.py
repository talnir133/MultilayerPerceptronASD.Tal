import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

warnings.filterwarnings("ignore")


class SimulationAnalyzer:
    def __init__(self, results, config, save_figures=True):
        self.results = results
        self.config = config
        self.save_figures = save_figures
        self.exp_name = config.get("exp_name", "Experiment")
        self.save_dir = f"figures/{self.exp_name}"

        if self.save_figures:
            os.makedirs(self.save_dir, exist_ok=True)
            config_path = os.path.join(self.save_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)

    def _save_fig(self, base_name):
        if not self.save_figures:
            return

        file_path = os.path.join(self.save_dir, f"{base_name}.png")
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(self.save_dir, f"{base_name}_{counter}.png")
            counter += 1

        plt.savefig(file_path, bbox_inches='tight', dpi=300)

    def _get_signatures(self, b_cfg):
        zf = b_cfg.get("zero_features", [])
        zf_list = list(zf) if isinstance(zf, (list, tuple)) else ([zf] if zf is not None else [])
        domain_sig = f"(Zero-Features:{','.join(map(str, sorted(zf_list))) or 'None'})"
        rule_sig = f"{b_cfg.get('rule')}_{b_cfg.get('deciding_feature', '')}"
        return domain_sig, rule_sig

    def _add_config_info(self, ax, show_config=True):
        if not show_config:
            return

        cfg = self.config
        txt = r"$\mathbf{Simulation\ Configurations:}$" + "\n\n"

        categories = {
            "Input:": ['features_types', 'seed', 'sd'],
            "Network:": ['hidden_size', 'n_hidden', 'b_scale_low', 'b_scale_high',
                         'w_scale_low', 'w_scale_high', 'optimizer_type', 'activation_type', 'batch_size', 'lr']
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
            zf_str = "None" if not zf and zf != 0 else (
                ",".join(map(str, zf)) if isinstance(zf, (list, tuple)) else str(zf))
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
        import matplotlib.colors as mcolors

        sample_metric = self.results[0][1]["data_low"].get(metric_name)
        is_env_based = isinstance(sample_metric, dict)
        bounds, blocks = [0], []

        shift_indices = set()
        active_rules = {}
        for i in range(len(self.results)):
            b_cfg = self.results[i][1]["config"]
            domain_sig, rule_sig = self._get_signatures(b_cfg)

            if domain_sig not in active_rules:
                active_rules[domain_sig] = rule_sig
            elif active_rules[domain_sig] != rule_sig:
                shift_indices.add(i)
                for target_d_sig in active_rules.keys():
                    for j in range(i, len(self.results)):
                        fut_cfg = self.results[j][1]["config"]
                        fut_d_sig, fut_r_sig = self._get_signatures(fut_cfg)
                        if fut_d_sig == target_d_sig:
                            active_rules[target_d_sig] = fut_r_sig
                            break

        if not is_env_based:
            low, high = [], []
            for name, res in self.results:
                low.extend(res["data_low"].get(metric_name, []))
                high.extend(res["data_high"].get(metric_name, []))
                bounds.append(len(low))
                blocks.append(name)
            ax.plot(low, label='Low Variance', color='blue')
            ax.plot(high, label='High Variance', color='red')
        else:
            envs = [env for env in sample_metric.keys() if env != "Combined"]
            if not envs: envs = ["Combined"]
            env_data_low = {env: [] for env in envs}
            env_data_high = {env: [] for env in envs}
            env_data_opt = {env: [] for env in envs}

            color_pairs = [
                ('#0047AB', '#00BFFF'),
                ('#8B0000', '#FF4500'),
                ('#006400', '#32CD32'),
                ('#4B0082', '#DA70D6'),
                ('#8B4513', '#D2691E'),
                ('#2F4F4F', '#20B2AA')
            ]

            for name, res in self.results:
                for env in envs:
                    env_data_low[env].extend(res["data_low"][metric_name][env])
                    env_data_high[env].extend(res["data_high"][metric_name][env])
                    opt_key = f"{metric_name}_optimal"
                    if opt_key in res["data_low"] and env in res["data_low"].get(opt_key, {}):
                        env_data_opt[env].extend(res["data_low"][opt_key][env])
                bounds.append(len(env_data_low[envs[0]]))
                blocks.append(name)

            for idx, env in enumerate(envs):
                c_low, c_high = color_pairs[idx % len(color_pairs)]

                ax.plot(env_data_low[env], label=f'Low Var ({env})', color=c_low, linewidth=2, linestyle='-')
                ax.plot(env_data_high[env], label=f'High Var ({env})', color=c_high, linewidth=2, linestyle='-')

                if self.config.get("sd", 0) > 0.0 and env_data_opt[env]:
                    rgb_low = np.array(mcolors.to_rgb(c_low))
                    rgb_high = np.array(mcolors.to_rgb(c_high))
                    c_avg = tuple((rgb_low + rgb_high) / 2.0)
                    ax.plot(env_data_opt[env], label=f'Bayes Opt ({env})', color=c_avg, alpha=0.6, linestyle='--',
                            linewidth=1.5)

        x_offset = bounds[-1] * 0.015
        for i in range(1, len(bounds) - 1):
            if i in shift_indices:
                ax.axvline(x=bounds[i], color='#888888', linestyle='--', linewidth=1.5)
                ax.text(bounds[i] - x_offset, sum(ax.get_ylim()) / 2, 'Rules Shift',
                        color='#888888', fontsize=10, rotation=90, va='center', ha='right', fontweight='normal')
            else:
                ax.axvline(x=bounds[i], color='gray', linestyle='--', linewidth=1, alpha=0.7)

        for i in range(len(blocks)):
            ax.text((bounds[i] + bounds[i + 1]) / 2, -0.06, blocks[i], transform=ax.get_xaxis_transform(), ha='center',
                    va='top', fontsize=8, fontweight='bold', color='darkblue')

        ax.set(xlabel='Epochs', ylabel=ylabel)
        ax.xaxis.labelpad = 20
        if log_scale: ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.5)

    def _graph_template(self, metric_key, y_axis_name, log_scale=False, show_config=True):
        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(right=0.65, bottom=0.15)

        self._plot_metric_on_ax(ax, metric_key, y_axis_name, log_scale)
        ax.set_title(y_axis_name, fontweight='bold', fontsize=14, pad=25)
        ax.text(0.5, 1.02, f"Simulation name: {self.exp_name.capitalize()}",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10, fontweight='normal')
        ax.legend(loc='lower left', bbox_to_anchor=(1.035, 0.0), frameon=True, edgecolor='gray')

        self._add_config_info(ax, show_config)
        self._save_fig(f"{metric_key}_figure_{self.exp_name.replace(' ', '_')}")
        plt.show()

    def _plot_standard_metric(self, metric_base, title_base, sub_type="clean", log_scale=False, show_config=True):
        sd = self.config.get("sd", 0)
        actual_sub_type = "clean" if sd == 0 else sub_type
        y_axis_name = title_base if sd == 0 else f"{title_base} (on {actual_sub_type} data)"
        metric_key = f"{metric_base}_{actual_sub_type}"

        self._graph_template(metric_key, y_axis_name, log_scale, show_config)

    def plot_loss(self, sub_type="noisy", log_scale=False, show_config=True):
        self._plot_standard_metric("losses", "BCE Loss", sub_type, log_scale, show_config)

    def plot_accuracy(self, sub_type="noisy", show_config=True):
        self._plot_standard_metric("accuracies", "Accuracy", sub_type, False, show_config)

    def plot_mae(self, sub_type="noisy", show_config=True):
        self._plot_standard_metric("MAE", "Mean Absolute Error", sub_type, False, show_config)

    def plot_parameter_distributions(self, param_type="weights", bins=50, show_config=True):
        """
        Creates an animation showing the distribution of weights or biases over all epochs.
        param_type: "weights" or "biases"
        """
        from matplotlib import animation
        from IPython.display import display, HTML

        if param_type not in ["weights", "biases"]:
            raise ValueError("param_type must be 'weights' or 'biases'")

        key = f"fc1_{param_type}"
        low_data, high_data, epoch_labels = [], [], []

        # Extract all epochs sequentially across all blocks
        for b_name, b_res in self.results:
            l_params = b_res["data_low"][key]
            h_params = b_res["data_high"][key]
            low_data.extend(l_params)
            high_data.extend(h_params)
            epoch_labels.extend([f"Block: {b_name} | Epoch: {i}" for i in range(len(l_params))])

        if not low_data:
            print(f"No data found for {param_type}.")
            return

        # Calculate individual min/max for dynamic X-axes to fix the scale issue
        all_low = np.concatenate([d.flatten() for d in low_data])
        all_high = np.concatenate([d.flatten() for d in high_data])

        low_min, low_max = all_low.min(), all_low.max()
        high_min, high_max = all_high.min(), all_high.max()

        # Add a tiny padding to the bounds to prevent clipping on the edges
        low_pad = (low_max - low_min) * 0.05
        high_pad = (high_max - high_min) * 0.05

        low_range = (low_min - low_pad, low_max + low_pad)
        high_range = (high_min - high_pad, high_max + high_pad)

        # Find the maximum bin density for the Y axis (using density=True)
        y_max = 0
        for l, h in zip(low_data, high_data):
            c_l, _ = np.histogram(l.flatten(), bins=bins, range=low_range, density=True)
            c_h, _ = np.histogram(h.flatten(), bins=bins, range=high_range, density=True)
            y_max = max(y_max, c_l.max(), c_h.max())
        y_max = y_max * 1.1

        # Adjust height to give titles more room
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # Hardcoded margins to strictly prevent title overlap
        plt.subplots_adjust(top=0.75, bottom=0.15, wspace=0.3, right=0.75 if show_config else 0.95)

        if show_config:
            p_top = axs[1].get_position()
            g_top = fig.add_axes([p_top.x0 + 0.15, p_top.y0, p_top.width, p_top.height])
            g_top.axis('off')
            self._add_config_info(g_top, show_config)

        def update(frame_idx):
            for ax in axs:
                ax.clear()

            # Low Variance
            axs[0].hist(low_data[frame_idx].flatten(), bins=bins, range=low_range, density=True, color='blue',
                        alpha=0.7, edgecolor='black')
            axs[0].set_title(f"Low Variance (RichMLP)\n{epoch_labels[frame_idx]}", fontweight='bold', pad=15)
            axs[0].set_xlim(low_range)
            axs[0].set_ylim(0, y_max)
            axs[0].grid(True, linestyle='--', alpha=0.5)
            axs[0].set_ylabel("Density")
            axs[0].set_xlabel("Value")

            # High Variance
            axs[1].hist(high_data[frame_idx].flatten(), bins=bins, range=high_range, density=True, color='red',
                        alpha=0.7, edgecolor='black')
            axs[1].set_title(f"High Variance (LazyMLP)\n{epoch_labels[frame_idx]}", fontweight='bold', pad=15)
            axs[1].set_xlim(high_range)
            axs[1].set_ylim(0, y_max)
            axs[1].grid(True, linestyle='--', alpha=0.5)
            axs[1].set_ylabel("Density")
            axs[1].set_xlabel("Value")

        fig.suptitle(
            f"Layer 'fc1' {param_type.capitalize()} Distributions over Time\nSimulation: {self.exp_name.capitalize()}",
            fontweight='bold', fontsize=18, y=0.92)

        anim = animation.FuncAnimation(fig, update, frames=len(low_data), interval=200, repeat=False)
        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    def plot_mds(self, epochs=(-1,), layer_name='fc1', target_env="Combined", show_config=True, mode='static'):
        if isinstance(epochs, int): epochs = (epochs,)
        if target_env not in self.results[0][1]["data_low"]["activations_clean"]:
            target_env = list(self.results[0][1]["data_low"]["activations_clean"].keys())[0]

        if mode == 'static':
            n_cols, n_rows, n_blks = len(epochs), 2, len(self.results)
            model_types, row_titles = ['low', 'high'], ['Low Variance\n(RichMLP)', 'High Variance\n(LazyMLP)']
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 10))
            if n_cols == 1: axs = axs.reshape(n_rows, 1)
            plt.subplots_adjust(right=0.65, wspace=0.3, hspace=0.3)

            for row_idx, (m_type, r_title) in enumerate(zip(model_types, row_titles)):
                for col_idx, epoch in enumerate(epochs):
                    ax = axs[row_idx, col_idx]
                    t_block, t_res, rel_ep, cur_ep = None, None, 0, 0
                    for b_name, b_res in self.results:
                        n_eps = len(b_res[f"data_{m_type}"]["activation_distances_clean"][target_env])
                        if epoch == -1 or cur_ep <= epoch < cur_ep + n_eps:
                            t_block, t_res, rel_ep = b_name, b_res[f"data_{m_type}"], (n_eps - 1) if epoch == -1 else (
                                    epoch - cur_ep)
                            break
                        cur_ep += n_eps
                    if t_res is None:
                        ax.text(0.5, 0.5, f"Epoch {epoch}\nNot Found", ha='center', va='center', color='red',
                                fontsize=12)
                        ax.axis('off')
                        continue

                    dist_mat = squareform(t_res["activation_distances_clean"][target_env][rel_ep][layer_name])
                    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4).fit_transform(
                        dist_mat)
                    env_X = t_res["envs_data"][target_env]["X"]

                    ax.scatter(coords[:, 0], coords[:, 1], color='#85C1E9', s=100, edgecolors='#1B4F72', alpha=0.85)
                    ft = self.config["features_types"]
                    for i, coord in enumerate(coords):
                        f_str, s_idx = [], 0
                        for dim in ft:
                            feat_slice = env_X[i][s_idx:s_idx + dim]
                            f_str.append("-" if np.sum(feat_slice) == 0 else str(np.argmax(feat_slice)))
                            s_idx += dim
                        ax.annotate(f"({','.join(f_str)})", coord, xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, fontweight='bold', color='#444444')
                    ax.margins(0.15)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    if row_idx == 0: ax.set_title(
                        f"Epoch: {'Last' if epoch == -1 else epoch}" + (f"\nBlock: {t_block}" if n_blks > 1 else ""),
                        fontweight='bold', fontsize=12)
                    if col_idx == 0: ax.set_ylabel(r_title, fontweight='bold', fontsize=13, labelpad=15)

            if show_config:
                shift = 0.15 / n_cols
                p_top = axs[0, -1].get_position()
                g_top = fig.add_axes([p_top.x0 + shift, p_top.y0, p_top.width, p_top.height])
                g_top.axis('off')
                self._add_config_info(g_top, show_config)

            fig.suptitle(
                f"MDS Evolution of '{layer_name}' Activations in {self.exp_name.capitalize()}\nEvaluated on Environment: {target_env}",
                fontweight='bold', fontsize=18, y=1.02)
            self._save_fig(f"MDS_grid_{layer_name}_{target_env}_eps_{'_'.join(map(str, epochs))}")
            plt.show()

        elif mode == 'animation':

            from matplotlib import animation
            from IPython.display import display, HTML

            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            model_types, titles = ['low', 'high'], ['Low Variance (RichMLP)', 'High Variance (LazyMLP)']
            plt.subplots_adjust(wspace=0.2)

            def update(frame_idx):
                epoch = epochs[frame_idx]
                for idx, ax in enumerate(axs):
                    ax.clear()
                    m_type = model_types[idx]
                    t_block, t_res, rel_ep, cur_ep = None, None, 0, 0

                    for b_name, b_res in self.results:
                        n_eps = len(b_res[f"data_{m_type}"]["activation_distances_clean"][target_env])
                        if epoch == -1 or cur_ep <= epoch < cur_ep + n_eps:
                            t_block, t_res, rel_ep = b_name, b_res[f"data_{m_type}"], (n_eps - 1) if epoch == -1 else (
                                    epoch - cur_ep)
                            break
                        cur_ep += n_eps

                    if t_res is None:
                        ax.text(0.5, 0.5, f"Epoch {epoch}\nNot Found", ha='center', va='center', color='red',
                                fontsize=12)
                        ax.axis('off')
                        continue

                    dist_mat = squareform(t_res["activation_distances_clean"][target_env][rel_ep][layer_name])
                    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=4).fit_transform(
                        dist_mat)
                    env_X = t_res["envs_data"][target_env]["X"]

                    ax.scatter(coords[:, 0], coords[:, 1], color='#85C1E9', s=130, edgecolors='#1B4F72', alpha=0.85)

                    ft = self.config["features_types"]
                    for i, coord in enumerate(coords):
                        f_str, s_idx = [], 0
                        for dim in ft:
                            feat_slice = env_X[i][s_idx:s_idx + dim]
                            f_str.append("-" if np.sum(feat_slice) == 0 else str(np.argmax(feat_slice)))
                            s_idx += dim
                        ax.annotate(f"({','.join(f_str)})", coord, xytext=(5, 5), textcoords='offset points',
                                    fontsize=9, fontweight='bold', color='#444444')

                    ax.margins(0.15)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_title(f"{titles[idx]}\nEpoch: {'Last' if epoch == -1 else epoch} | Block: {t_block}",
                                 fontweight='bold', fontsize=13)

            self.anim = animation.FuncAnimation(fig, update, frames=len(epochs), interval=800, repeat=True)
            plt.close(fig)
            display(HTML(self.anim.to_jshtml()))