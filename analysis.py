import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import colorsys
from matplotlib import animation
from IPython.display import display, HTML
import matplotlib as mpl

warnings.filterwarnings("ignore")


def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def align_coords(coords, ref_coords):
    if coords.shape[0] == 0: return coords
    mean_c = coords.mean(axis=0)
    mean_ref = ref_coords.mean(axis=0)
    c_c = coords - mean_c
    ref_c = ref_coords - mean_ref
    try:
        U, S, Vt = np.linalg.svd(c_c.T @ ref_c)
        R = U @ Vt

        return c_c @ R + mean_ref
    except Exception:
        return coords


class SimulationAnalyzer:
    def __init__(self, results, config, save_figures=True):
        if isinstance(results[0][0], str):
            self.runs = [results]
            self.is_multi = False
        else:
            self.runs = results
            self.is_multi = len(results) > 1

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

    def _add_config_info(self, ax, show_config=True):
        if not show_config:
            return

        cfg = self.config
        txt = r"$\mathbf{Simulation\ Configurations:}$" + "\n\n"
        categories = {
            "Input:": ['features_types', 'seed' if not getattr(self, 'is_multi', False) else 'seed (multiple)'],
            "Network:": ['hidden_size', 'n_hidden', 'b_scale_low', 'b_scale_high',
                         'w_scale_low', 'w_scale_high', 'optimizer_type', 'activation_type', 'batch_size', 'lr']
        }
        for title, keys in categories.items():
            txt += f"{title}\n"
            for k in keys:
                val = cfg.get(k, 'N/A')
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
            b_sd = block.get('sd', 0.0)
            rule_params = [f"{k}={v}" for k, v in block.items() if
                           k not in ['block_name', 'epochs', 'zero_features', 'rule', 'alpha_class', 'alpha_rec', 'sd']]
            params_str = f"({', '.join(rule_params)})" if rule_params else ""
            txt += f"   {idx}. {block.get('block_name', 'Unnamed')}, eps: {block.get('epochs', 0)}, zero: {zf_str}, a_c: {a_class}, a_r: {a_rec}, sd: {b_sd}\n"
            txt += f"      Rule: {rule_name} {params_str}\n"

        ax.text(0.0, 1.0, txt.strip(), transform=ax.transAxes, fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=0.8, edgecolor='gray'))


    def plot_metric_over_epochs(self, metric_name, envs="current_only", show_std=False, show_config=True):
        is_global_metric = isinstance(self.runs[0][0][1]["data_low"].get(metric_name), list)

        if is_global_metric:
            available_metrics = [k for k, v in self.runs[0][0][1]["data_low"].items() if
                                 isinstance(v, list) and not isinstance(v[0], dict) and not isinstance(v[0],
                                                                                                       np.ndarray)]
        else:
            available_metrics = list(self.runs[0][0][1]["data_low"].keys())

        if metric_name not in available_metrics and not is_global_metric:
            alt_match = next((m for m in available_metrics if m.lower() == metric_name.lower()), None)
            if alt_match:
                print(f"Notice: Metric '{metric_name}' not found. Automatically using '{alt_match}'.")
                metric_name = alt_match
            else:
                raise KeyError(f"Metric '{metric_name}' not found! \nAvailable metrics are: {available_metrics}")

        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(right=0.74 if show_config else 0.85, top=0.88, bottom=0.15)

        bounds = [0]
        blocks = []
        for name, res in self.runs[0]:
            if is_global_metric:
                n_eps = len(res["data_low"][metric_name])
            else:
                n_eps = len(next(iter(res["data_low"]["losses_clean"].values())))
            bounds.append(bounds[-1] + n_eps)
            blocks.append(name)

        has_noise = any(b.get("sd", 0.0) > 0 for b in self.config.get("exp_blocks", []))

        if is_global_metric:
            all_runs_low = []
            all_runs_high = []
            for run_res in self.runs:
                run_low, run_high = [], []
                for b_name, b_res in run_res:
                    run_low.extend(b_res["data_low"][metric_name])
                    run_high.extend(b_res["data_high"][metric_name])
                all_runs_low.append(run_low)
                all_runs_high.append(run_high)

            mean_low = np.mean(all_runs_low, axis=0)
            mean_high = np.mean(all_runs_high, axis=0)
            x_range = np.arange(len(mean_low))

            ax.plot(x_range, mean_low, color='blue', linewidth=2, linestyle='-', label="Low Variance (RichMLP)")
            ax.plot(x_range, mean_high, color='red', linewidth=2, linestyle='-', label="High Variance (LazyMLP)")

            if show_std and self.is_multi:
                std_low = np.std(all_runs_low, axis=0)
                std_high = np.std(all_runs_high, axis=0)
                ax.fill_between(x_range, mean_low - std_low, mean_low + std_low, color='blue', alpha=0.15)
                ax.fill_between(x_range, mean_high - std_high, mean_high + std_high, color='red', alpha=0.15)

        else:
            unique_envs = list(self.runs[0][0][1]["data_low"][metric_name].keys())
            all_runs_low = {env: [] for env in unique_envs}
            all_runs_high = {env: [] for env in unique_envs}
            all_runs_opt = {env: [] for env in unique_envs}

            for run_res in self.runs:
                run_env_low = {env: [] for env in unique_envs}
                run_env_high = {env: [] for env in unique_envs}
                run_env_opt = {env: [] for env in unique_envs}

                for b_name, b_res in run_res:
                    for env in unique_envs:
                        main_low = b_res["data_low"][metric_name].get(env, [])
                        if len(main_low) == 0 and "_noisy" in metric_name:
                            fallback = metric_name.replace("_noisy", "_clean")
                            main_low = b_res["data_low"][fallback][env]
                            main_high = b_res["data_high"][fallback][env]
                        else:
                            main_high = b_res["data_high"][metric_name].get(env, [])

                        run_env_low[env].extend(main_low)
                        run_env_high[env].extend(main_high)

                        if has_noise and "_noisy" in metric_name:
                            opt_key = f"{metric_name}_optimal"
                            b_opt = b_res["data_low"].get(opt_key, {}).get(env, [])
                            if len(b_opt) == 0:
                                clean_metric = metric_name.replace("_noisy", "_clean")
                                b_opt = b_res["data_low"][clean_metric][env]
                            run_env_opt[env].extend(b_opt)

                for env in unique_envs:
                    all_runs_low[env].append(run_env_low[env])
                    all_runs_high[env].append(run_env_high[env])
                    if has_noise and "_noisy" in metric_name:
                        all_runs_opt[env].append(run_env_opt[env])

            color_pairs = [
                ('#0047AB', '#00BFFF'), ('#8B0000', '#FF4500'),
                ('#006400', '#32CD32'), ('#4B0082', '#DA70D6'),
                ('#8B4513', '#D2691E'), ('#2F4F4F', '#20B2AA')
            ]
            color_map = {env: color_pairs[i % len(color_pairs)] for i, env in enumerate(unique_envs)}

            if envs == "all":
                active_envs_per_block = [unique_envs for _ in blocks]
            elif envs == "current_only":
                active_envs_per_block = [[b] for b in blocks]
            else:
                active_envs_per_block = envs

            is_current_only = (envs == "current_only")
            plotted_labels = set()

            for idx, b_name in enumerate(blocks):
                start_ep = bounds[idx]
                end_ep = bounds[idx + 1]
                x_range = np.arange(start_ep, end_ep)

                active_envs = active_envs_per_block[idx] if idx < len(active_envs_per_block) else []

                for env in active_envs:
                    if env not in unique_envs:
                        continue

                    if is_current_only:
                        c_low, c_high = 'blue', 'red'
                        lbl_low = "Low Var"
                        lbl_high = "High Var"
                    else:
                        c_low, c_high = color_map[env]
                        lbl_low = f"Low Var ({env})"
                        lbl_high = f"High Var ({env})"

                    mean_low = np.mean(all_runs_low[env], axis=0)[start_ep:end_ep]
                    mean_high = np.mean(all_runs_high[env], axis=0)[start_ep:end_ep]

                    ax.plot(x_range, mean_low, color=c_low, linewidth=2, linestyle='-',
                            label=lbl_low if lbl_low not in plotted_labels else "")
                    ax.plot(x_range, mean_high, color=c_high, linewidth=2, linestyle='-',
                            label=lbl_high if lbl_high not in plotted_labels else "")

                    if show_std and self.is_multi:
                        std_low = np.std(all_runs_low[env], axis=0)[start_ep:end_ep]
                        std_high = np.std(all_runs_high[env], axis=0)[start_ep:end_ep]
                        ax.fill_between(x_range, mean_low - std_low, mean_low + std_low, color=c_low, alpha=0.15)
                        ax.fill_between(x_range, mean_high - std_high, mean_high + std_high, color=c_high, alpha=0.15)

                    plotted_labels.update([lbl_low, lbl_high])

                    if has_noise and "_noisy" in metric_name and len(all_runs_opt[env][0]) > 0:
                        lbl_opt = "Bayes Opt" if is_current_only else f"Bayes Opt ({env})"

                        if is_current_only:
                            c_avg = 'purple'
                        else:
                            rgb_low = np.array(mcolors.to_rgb(c_low))
                            rgb_high = np.array(mcolors.to_rgb(c_high))
                            c_avg = tuple((rgb_low + rgb_high) / 2.0)

                        mean_opt = np.mean(all_runs_opt[env], axis=0)[start_ep:end_ep]
                        ax.plot(x_range, mean_opt, color=c_avg, alpha=0.6, linestyle='--', linewidth=1.5,
                                label=lbl_opt if lbl_opt not in plotted_labels else "")

                        if show_std and self.is_multi:
                            std_opt = np.std(all_runs_opt[env], axis=0)[start_ep:end_ep]
                            ax.fill_between(x_range, mean_opt - std_opt, mean_opt + std_opt, color=c_avg, alpha=0.1)

                        plotted_labels.add(lbl_opt)

        for i in range(1, len(bounds) - 1):
            ax.axvline(x=bounds[i], color='gray', linestyle='--', linewidth=1, alpha=0.7)

        for i in range(len(blocks)):
            ax.text((bounds[i] + bounds[i + 1]) / 2, -0.06, blocks[i], transform=ax.get_xaxis_transform(), ha='center',
                    va='top', fontsize=8, fontweight='bold', color='darkblue')

        has_any_noise = any(b.get("sd", 0.0) > 0 for b in self.config.get("exp_blocks", []))

        if not has_any_noise:
            clean_metric_name = metric_name.replace('_noisy', '').replace('_clean', '')
        else:
            clean_metric_name = metric_name
        parts = clean_metric_name.split('_')
        y_label = " ".join([p.upper() if p.lower() in ['mae', 'pr'] else p.capitalize() for p in parts])
        subtitle = ""

        if "PR" in metric_name:
            if "weights" in metric_name:
                y_label = "Normalized Weights' PR"
                subtitle = "Mean PR over all incoming weight vectors of the different neurons"
            else:
                y_label = "Normalized Activations' PR"
                subtitle = "Mean PR over all activation vectors of the data samples of current block"
            ax.set_ylim(-0.05, 1.05)

        elif "correlation" in metric_name.lower():
            y_label = "Pearson Correlation (r)"
            subtitle = "Correlation between incoming weights' L2 norm and absolute bias magnitude"
            ax.set_ylim(-1.05, 1.05)

        ax.set(xlabel='Epochs', ylabel=y_label)
        ax.xaxis.labelpad = 20
        ax.grid(True, which="both", ls="-", alpha=0.5)

        title_suffix = f" ({len(self.runs)} Runs Averaged)" if self.is_multi else ""

        ax.set_title(f"{y_label}{title_suffix}", fontweight='bold', fontsize=15, pad=30)

        if subtitle:
            ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=11, color='#555555', style='italic')

        ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=True, edgecolor='gray')

        if show_config:
            cfg_ax = fig.add_axes([0.76, 0.15, 0.22, 0.73])
            cfg_ax.axis('off')
            self._add_config_info(cfg_ax, show_config=show_config)

        self._save_fig(f"{metric_name}_{envs}_figure_{self.exp_name.replace(' ', '_')}")
        plt.show()

    def plot_parameter_distributions(self, layer_name="fc1", param_type="weight", color_mode=None, bins=50,
                                     clip_percentile=2, show_config=True):
        mpl.rcParams['animation.embed_limit'] = 100.0

        param_type = param_type.lower()
        if "weight" in param_type:
            actual_param = "weight"
            if color_mode is None:
                color_mode = "mean_bias"
        elif "bias" in param_type:
            actual_param = "bias"
            if color_mode is None:
                color_mode = "weight_norm"
        else:
            print("Error: param_type must be either 'weight' or 'bias'")
            return

        w_key = f"_layers.{layer_name}.weight"
        b_key = f"_layers.{layer_name}.bias"

        first_b_res = self.runs[0][0][1]
        if w_key not in first_b_res["data_low"]["weights"]:
            print(f"Warning: Weight key '{w_key}' not found for layer '{layer_name}'.")
            return

        has_bias = b_key in first_b_res["data_low"]["biases"]
        if actual_param == "bias" and not has_bias:
            print(f"Warning: Layer '{layer_name}' does not have biases to plot.")
            return

        if not has_bias and color_mode in ["mean_bias", "mean_abs_bias"]:
            color_mode = "input_features" if layer_name == "fc1" else "weight_norm"

        if color_mode == "input_features" and layer_name != "fc1":
            print(f"Notice: 'input_features' color_mode is only valid for 'fc1'. Reverting to 'weight_norm'.")
            color_mode = "weight_norm"

        low_X, high_X = [], []
        low_C, high_C = [], []
        epoch_labels = []

        for b_name, b_res in self.runs[0]:
            n_eps = len(b_res["data_low"]["losses_clean"][list(b_res["data_low"]["losses_clean"].keys())[0]])
            for ep in range(n_eps):
                epoch_labels.append(f"Block: {b_name} | Epoch: {ep}")

        for ep_idx in range(len(epoch_labels)):
            l_x_ep, h_x_ep = [], []
            l_c_ep, h_c_ep = [], []

            cur_ep = ep_idx
            b_idx = 0
            while cur_ep >= len(self.runs[0][b_idx][1]["data_low"]["losses_clean"][
                                    list(self.runs[0][b_idx][1]["data_low"]["losses_clean"].keys())[0]]):
                cur_ep -= len(self.runs[0][b_idx][1]["data_low"]["losses_clean"][
                                  list(self.runs[0][b_idx][1]["data_low"]["losses_clean"].keys())[0]])
                b_idx += 1

            for run_res in self.runs:
                b_res = run_res[b_idx][1]

                w_l = b_res["data_low"]["weights"][w_key][cur_ep]
                w_h = b_res["data_high"]["weights"][w_key][cur_ep]

                b_l = b_res["data_low"]["biases"][b_key][cur_ep] if has_bias else np.zeros(w_l.shape[0])
                b_h = b_res["data_high"]["biases"][b_key][cur_ep] if has_bias else np.zeros(w_h.shape[0])

                if actual_param == "weight":
                    l_x_ep.append(w_l.flatten())
                    h_x_ep.append(w_h.flatten())

                    if color_mode == "input_features":
                        F = np.zeros_like(w_l)
                        s_idx = 0
                        for f_idx, dim in enumerate(self.config["features_types"]):
                            F[:, s_idx:s_idx + dim] = f_idx
                            s_idx += dim
                        l_c_ep.append(F.flatten())
                        h_c_ep.append(F.flatten())
                    elif color_mode in ["mean_bias", "mean_abs_bias"]:
                        b_l_exp = np.repeat(b_l[:, None], w_l.shape[1], axis=1).flatten()
                        b_h_exp = np.repeat(b_h[:, None], w_h.shape[1], axis=1).flatten()
                        if color_mode == "mean_abs_bias":
                            l_c_ep.append(np.abs(b_l_exp))
                            h_c_ep.append(np.abs(b_h_exp))
                        else:
                            l_c_ep.append(b_l_exp)
                            h_c_ep.append(b_h_exp)
                    else:  # weight_norm fallback
                        l_c_ep.append(np.linalg.norm(w_l, axis=1).flatten())
                        h_c_ep.append(np.linalg.norm(w_h, axis=1).flatten())
                else:
                    l_x_ep.append(b_l.flatten())
                    h_x_ep.append(b_h.flatten())

                    l_c_ep.append(np.linalg.norm(w_l, axis=1).flatten())
                    h_c_ep.append(np.linalg.norm(w_h, axis=1).flatten())

            low_X.append(np.concatenate(l_x_ep))
            high_X.append(np.concatenate(h_x_ep))
            low_C.append(np.concatenate(l_c_ep))
            high_C.append(np.concatenate(h_c_ep))

        if not low_X:
            return

        all_low_X = np.concatenate(low_X)
        all_high_X = np.concatenate(high_X)
        all_low_C = np.concatenate(low_C)
        all_high_C = np.concatenate(high_C)

        low_min, low_max = np.percentile(all_low_X, clip_percentile), np.percentile(all_low_X, 100 - clip_percentile)
        high_min, high_max = np.percentile(all_high_X, clip_percentile), np.percentile(all_high_X,
                                                                                       100 - clip_percentile)

        low_pad = (low_max - low_min) * 0.05
        high_pad = (high_max - high_min) * 0.05

        low_range = (low_min - low_pad, low_max + low_pad)
        high_range = (high_min - high_pad, high_max + high_pad)

        c_min_low, c_max_low = np.percentile(all_low_C, clip_percentile), np.percentile(all_low_C,
                                                                                        100 - clip_percentile)
        c_min_high, c_max_high = np.percentile(all_high_C, clip_percentile), np.percentile(all_high_C,
                                                                                           100 - clip_percentile)

        is_continuous = color_mode != "input_features"
        norm_low = None
        norm_high = None

        if color_mode == "mean_bias":
            cmap = plt.cm.coolwarm
            max_abs_low = max(abs(c_min_low), abs(c_max_low))
            max_abs_high = max(abs(c_min_high), abs(c_max_high))
            norm_low = mcolors.Normalize(vmin=-max_abs_low, vmax=max_abs_low)
            norm_high = mcolors.Normalize(vmin=-max_abs_high, vmax=max_abs_high)
            cbar_label = "Mean Target Bias"
        elif color_mode == "mean_abs_bias":
            cmap = plt.cm.plasma
            norm_low = mcolors.Normalize(vmin=0, vmax=c_max_low)
            norm_high = mcolors.Normalize(vmin=0, vmax=c_max_high)
            cbar_label = "Mean Target Absolute Bias"
        elif color_mode == "input_features":
            n_features = len(self.config["features_types"])
            feature_colors = get_distinct_colors(n_features)
        else:
            cmap = plt.cm.plasma
            norm_low = mcolors.Normalize(vmin=0, vmax=c_max_low)
            norm_high = mcolors.Normalize(vmin=0, vmax=c_max_high)
            cbar_label = "Mean L2 Norm of Incoming Weights"

        y_max_low, y_max_high = 0, 0
        for l_x, h_x in zip(low_X, high_X):
            c_l, _ = np.histogram(l_x, bins=bins, range=low_range, density=True)
            c_h, _ = np.histogram(h_x, bins=bins, range=high_range, density=True)
            y_max_low = max(y_max_low, c_l.max())
            y_max_high = max(y_max_high, c_h.max())

        y_max_low *= 1.1
        y_max_high *= 1.1

        fig, axs = plt.subplots(1, 2, figsize=(14, 7.5))

        left_margin = 0.08
        right_margin = 0.74 if show_config else 0.95
        wspace = 0.25
        fig.subplots_adjust(left=left_margin, right=right_margin, top=0.85, bottom=0.20, wspace=wspace)

        if is_continuous:
            total_width = right_margin - left_margin

            plot_width = total_width / (2.0 + wspace)
            actual_wspace = plot_width * wspace

            sm_low = plt.cm.ScalarMappable(cmap=cmap, norm=norm_low)
            sm_low.set_array([])
            cbar_ax_low = fig.add_axes([left_margin, 0.06, plot_width, 0.03])
            cbar_low = fig.colorbar(sm_low, cax=cbar_ax_low, orientation='horizontal')
            cbar_low.set_label(f"{cbar_label} (Low Var)", fontweight='bold')

            sm_high = plt.cm.ScalarMappable(cmap=cmap, norm=norm_high)
            sm_high.set_array([])

            cbar_ax_high = fig.add_axes([left_margin + plot_width + actual_wspace, 0.06, plot_width, 0.03])
            cbar_high = fig.colorbar(sm_high, cax=cbar_ax_high, orientation='horizontal')
            cbar_high.set_label(f"{cbar_label} (High Var)", fontweight='bold')
        else:
            legend_elements = [Line2D([0], [0], color=feature_colors[i], lw=8, label=f'Feature {i}') for i in
                               range(n_features)]
            fig.legend(handles=legend_elements, loc='lower center',
                       bbox_to_anchor=((left_margin + right_margin) / 2, 0.04), ncol=n_features, title="Input Features")

        def update(frame_idx):
            for ax, X_data, C_data, rng, title, y_max, current_norm in zip(
                    axs, [low_X, high_X], [low_C, high_C], [low_range, high_range],
                    ["Low Variance (RichMLP)", "High Variance (LazyMLP)"],
                    [y_max_low, y_max_high], [norm_low, norm_high]):

                ax.clear()
                vals = X_data[frame_idx]
                c_vals = C_data[frame_idx]

                if color_mode == "input_features":
                    data_list = [vals[c_vals == f_idx] for f_idx in range(n_features)]
                    ax.hist(data_list, bins=bins, range=rng, density=True, stacked=True, color=feature_colors,
                            edgecolor='black', linewidth=0.5)
                else:
                    n, bins_edges, patches = ax.hist(vals, bins=bins, range=rng, density=True, edgecolor='black',
                                                     linewidth=0.5)
                    bin_indices = np.digitize(vals, bins_edges) - 1
                    bin_indices = np.clip(bin_indices, 0, bins - 1)

                    for i, patch in enumerate(patches):
                        mask = (bin_indices == i)
                        if np.any(mask):
                            mean_c = np.mean(c_vals[mask])
                            patch.set_facecolor(cmap(current_norm(mean_c)))
                        else:
                            patch.set_facecolor('lightgray')

                ax.set_title(f"{title}\n{epoch_labels[frame_idx]}", fontweight='bold', pad=15)
                ax.set_xlim(rng)
                ax.set_ylim(0, y_max)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylabel("Density")
                ax.set_xlabel("Value")

        title_suffix = f" (Pooled over {len(self.runs)} Runs)" if self.is_multi else ""
        fig.suptitle(f"Layer '{layer_name}' {actual_param.capitalize()} Distributions over Time{title_suffix}",
                     fontweight='bold', fontsize=18, y=0.96)

        if show_config:
            cfg_ax = fig.add_axes([0.76, 0.20, 0.22, 0.65])
            cfg_ax.axis('off')
            self._add_config_info(cfg_ax, show_config=show_config)

        anim = animation.FuncAnimation(fig, update, frames=len(low_X), interval=200, repeat=False)
        plt.close(fig)
        display(HTML(anim.to_jshtml()))


    def plot_PCs(self, epochs=(-1,), layer_name='fc1', show_config=True):
        if isinstance(epochs, int): epochs = (epochs,)

        env_X = self.runs[0][0][1]["data_low"]["X_global"]
        ft = self.config["features_types"]
        f_strs = []
        for i in range(len(env_X)):
            f_str, s_idx = [], 0
            for dim in ft:
                feat_slice = env_X[i][s_idx:s_idx + dim]
                f_str.append("-" if np.sum(feat_slice) == 0 else str(np.argmax(feat_slice)))
                s_idx += dim
            f_strs.append(f"({','.join(f_str)})")

        unique_labels = sorted(list(set(f_strs)))
        use_binary_colors = len(unique_labels) > 4

        n_runs = len(self.runs)
        total_epochs_in_sim = sum(len(b_res["data_low"]["activations"]) for _, b_res in self.runs[0])
        N_samples = len(env_X)

        target_info = []
        for epoch in epochs:
            rel_ep, cur_ep, b_idx, t_block = 0, 0, 0, None
            for b_name, b_res in self.runs[0]:
                n_eps = len(b_res["data_low"]["activations"])
                if epoch == -1 or cur_ep <= epoch < cur_ep + n_eps:
                    t_block = b_name
                    rel_ep = (n_eps - 1) if epoch == -1 else (epoch - cur_ep)
                    break
                cur_ep += n_eps
                b_idx += 1
            abs_ep = cur_ep + rel_ep if epoch != -1 else total_epochs_in_sim - 1
            target_info.append((b_idx, rel_ep, abs_ep, t_block, epoch))

        num_frames = len(target_info)

        raw_coords = {'low': np.zeros((n_runs, num_frames, N_samples, 2)),
                      'high': np.zeros((n_runs, num_frames, N_samples, 2))}
        pca_ev = {'low': np.zeros((n_runs, num_frames, 2)),
                  'high': np.zeros((n_runs, num_frames, 2))}

        for r_idx, run in enumerate(self.runs):
            for m_type in ['low', 'high']:
                for k, (b_idx, rel_ep, _, _, _) in enumerate(target_info):
                    acts = run[b_idx][1][f"data_{m_type}"]["activations"][rel_ep][layer_name]
                    if acts.shape[1] < 2:
                        acts = np.pad(acts, ((0, 0), (0, 2 - acts.shape[1])))

                    pca = PCA(n_components=2)
                    if np.all(acts == acts[0]):
                        c = np.zeros((N_samples, 2))
                        ev = np.zeros(2)
                    else:
                        c = pca.fit_transform(acts)
                        if c.shape[1] < 2: c = np.pad(c, ((0, 0), (0, 2 - c.shape[1])))
                        ev = pca.explained_variance_ratio_
                        if len(ev) < 2: ev = np.pad(ev, (0, 2 - len(ev)))

                    raw_coords[m_type][r_idx, k] = c
                    pca_ev[m_type][r_idx, k] = ev

        pca_coords = {'low': np.zeros_like(raw_coords['low']),
                      'high': np.zeros_like(raw_coords['high'])}

        if num_frames > 0:
            pca_coords['low'][0, 0] = raw_coords['low'][0, 0]
            for k in range(1, num_frames):
                pca_coords['low'][0, k] = align_coords(raw_coords['low'][0, k],
                                                                     pca_coords['low'][0, k - 1])

            for k in range(num_frames):
                pca_coords['high'][0, k] = align_coords(raw_coords['high'][0, k], pca_coords['low'][0, k])

            for r_idx in range(1, n_runs):
                for k in range(num_frames):
                    pca_coords['low'][r_idx, k] = align_coords(raw_coords['low'][r_idx, k],
                                                                             pca_coords['low'][0, k])
                    pca_coords['high'][r_idx, k] = align_coords(raw_coords['high'][r_idx, k],
                                                                              pca_coords['high'][0, k])

        from matplotlib import animation
        from IPython.display import display, HTML

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], wspace=0.2, hspace=0.3)

        ax_low = fig.add_subplot(gs[0, 0])
        ax_high = fig.add_subplot(gs[0, 1])
        ax_ev_low = fig.add_subplot(gs[1, 0])
        ax_ev_high = fig.add_subplot(gs[1, 1])

        if show_config:
            fig.subplots_adjust(right=0.74, top=0.90, wspace=0.2, hspace=0.3)
            cfg_ax = fig.add_axes([0.76, 0.15, 0.22, 0.75])
            cfg_ax.axis('off')
            self._add_config_info(cfg_ax, show_config=show_config)
        else:
            fig.subplots_adjust(right=0.88, top=0.90, wspace=0.2, hspace=0.3)

        x_vals = [info[2] for info in target_info]

        vlines = {}
        for m_type, ax_ev, title in zip(['low', 'high'], [ax_ev_low, ax_ev_high],
                                        ['Low Var Explained Variance', 'High Var Explained Variance']):
            if num_frames > 0:
                pc1_vals = pca_ev[m_type][:, :, 0]
                pc2_vals = pca_ev[m_type][:, :, 1]
                cum_vals = np.sum(pca_ev[m_type][:, :, :2], axis=2)

                for r_idx in range(n_runs):
                    ax_ev.plot(x_vals, cum_vals[r_idx, :], color='purple', alpha=0.15, linestyle=':')

                ax_ev.plot(x_vals, pc1_vals.mean(axis=0), color='blue', label='PC1', linewidth=1.5)
                ax_ev.plot(x_vals, pc2_vals.mean(axis=0), color='orange', label='PC2', linewidth=1.5)
                ax_ev.plot(x_vals, cum_vals.mean(axis=0), color='purple', label='PC1 + PC2', linewidth=2.5)

                vlines[m_type] = ax_ev.axvline(x_vals[0], color='black', linestyle='--', linewidth=2)

            ax_ev.set_title(title, fontsize=10, fontweight='bold')
            ax_ev.set_ylim(-0.05, 1.05)
            ax_ev.grid(True, linestyle=':', alpha=0.6)

        ax_ev_high.legend(loc='lower left', bbox_to_anchor=(1.02, 0.0), fontsize=8)

        titles = ['Low Variance (RichMLP)', 'High Variance (LazyMLP)']

        scatters = {'low': [], 'high': []}
        ax_titles = {}

        for idx, (ax, m_type) in enumerate(zip([ax_low, ax_high], ['low', 'high'])):
            ax.grid(True, linestyle='--', alpha=0.5)
            run_scats = []
            for r_idx in range(n_runs):
                lbl_scats = {}
                for lbl in unique_labels:
                    mask = np.array([s == lbl for s in f_strs])
                    coords = pca_coords[m_type][r_idx, 0][mask]
                    scat = ax.scatter(coords[:, 0], coords[:, 1], color='white', s=100, alpha=0.6, edgecolors='none')
                    lbl_scats[lbl] = (scat, mask)
                run_scats.append(lbl_scats)
            scatters[m_type] = run_scats
            ax_titles[m_type] = ax.set_title("", fontweight='bold', fontsize=13)

        def update(frame_idx):
            b_idx, rel_ep, abs_ep, t_block, epoch_val = target_info[frame_idx]

            if use_binary_colors:
                env_X_test = self.runs[0][b_idx][1]["data_low"]["test_envs"][t_block]["X"]
                env_y_test = self.runs[0][b_idx][1]["data_low"]["test_envs"][t_block]["y"]

                y_global = np.zeros(len(env_X))
                if len(env_X) == len(env_y_test):
                    y_global = env_y_test[:, 0]
                else:
                    for i, x_val in enumerate(env_X):
                        dist = np.sum(np.abs(env_X_test - x_val), axis=1)
                        match_idx = np.argmin(dist)
                        y_global[i] = env_y_test[match_idx, 0]

                colors_for_frame = ['#1f77b4' if y > 0.5 else '#d62728' for y in y_global]

                legend_elements = [
                    Line2D([0], [0], marker='o', color='black', label='Class 0 (Avg)' if self.is_multi else 'Class 0',
                           markerfacecolor='#d62728', markersize=10, linestyle='None', markeredgewidth=1.5),
                    Line2D([0], [0], marker='o', color='black', label='Class 1 (Avg)' if self.is_multi else 'Class 1',
                           markerfacecolor='#1f77b4', markersize=10, linestyle='None', markeredgewidth=1.5)
                ]
            else:
                distinct_rgbs = get_distinct_colors(len(unique_labels))
                color_map = {lbl: distinct_rgbs[i] for i, lbl in enumerate(unique_labels)}
                colors_for_frame = [color_map[f_strs[i]] for i in range(len(env_X))]

                legend_elements = [
                    Line2D([0], [0], marker='o', color='black', label=f'{lbl} (Avg)' if self.is_multi else lbl,
                           markerfacecolor=color_map[lbl], markersize=10, linestyle='None', markeredgewidth=1.5)
                    for lbl in unique_labels
                ]

            for idx, (ax, m_type) in enumerate(zip([ax_low, ax_high], ['low', 'high'])):
                ax.clear()

                if t_block is None:
                    ax.text(0.5, 0.5, f"Epoch {epoch_val}\nNot Found", ha='center', va='center', color='red',
                            fontsize=12)
                    ax.axis('off')
                    continue

                all_x, all_y = [], []

                if self.is_multi:
                    for r_idx in range(n_runs):
                        coords = pca_coords[m_type][r_idx, frame_idx]
                        all_x.extend(coords[:, 0])
                        all_y.extend(coords[:, 1])
                        ax.scatter(coords[:, 0], coords[:, 1], c=colors_for_frame, s=15, alpha=0.2, edgecolors='none')

                    mean_coords = np.mean(pca_coords[m_type][:, frame_idx, :, :], axis=0)
                    ax.scatter(mean_coords[:, 0], mean_coords[:, 1], c=colors_for_frame, s=150, alpha=1.0,
                               edgecolors='black', linewidths=1.5, zorder=5)
                else:
                    coords = pca_coords[m_type][0, frame_idx]
                    all_x.extend(coords[:, 0])
                    all_y.extend(coords[:, 1])
                    ax.scatter(coords[:, 0], coords[:, 1], c=colors_for_frame, s=100, alpha=0.6, edgecolors='none')

                if all_x and all_y:
                    min_val = min(min(all_x), min(all_y))
                    max_val = max(max(all_x), max(all_y))
                    margin = (max_val - min_val) * 0.15 if max_val != min_val else 0.1
                    lim_min = min_val - margin
                    lim_max = max_val + margin
                    ax.set_xlim(lim_min, lim_max)
                    ax.set_ylim(lim_min, lim_max)
                    ax.set_aspect('equal', 'box')

                ax.margins(0.15)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(f"{titles[idx]}\nEpoch: {'Last' if epoch_val == -1 else epoch_val} | Block: {t_block}",
                             fontweight='bold', fontsize=13)
                vlines[m_type].set_xdata([abs_ep, abs_ep])

            ax_high.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.02, 0.0),
                           title="Classes" if use_binary_colors else "Features", fontsize=8)

        title_suffix = f" ({len(self.runs)} Runs)" if self.is_multi else ""
        fig.suptitle(f"PCA Evolution of '{layer_name}' Activations in {self.exp_name.capitalize()}{title_suffix}",
                     fontweight='bold', fontsize=18, y=0.98)

        anim = animation.FuncAnimation(fig, update, frames=len(epochs), interval=200, repeat=False)
        plt.close(fig)
        display(HTML(anim.to_jshtml()))