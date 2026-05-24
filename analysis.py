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


def align_coords_no_reflection(coords, ref_coords):
    if coords.shape[0] == 0: return coords
    mean_c = coords.mean(axis=0)
    mean_ref = ref_coords.mean(axis=0)
    c_c = coords - mean_c
    ref_c = ref_coords - mean_ref
    try:
        U, S, Vt = np.linalg.svd(c_c.T @ ref_c)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
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

    def plot_metric_over_epochs(self, metric_name, envs="current_only", show_std=True, show_config=True):
        available_metrics = list(self.runs[0][0][1]["data_low"].keys())

        if metric_name not in available_metrics:
            alt_match = next((m for m in available_metrics if m.lower() == metric_name.lower()), None)
            if alt_match:
                print(f"Notice: Metric '{metric_name}' not found. Automatically using '{alt_match}'.")
                metric_name = alt_match
            else:
                raise KeyError(f"Metric '{metric_name}' not found! \nAvailable metrics are: {available_metrics}")

        fig, ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(right=0.74 if show_config else 0.85, top=0.88, bottom=0.15)

        unique_envs = list(self.runs[0][0][1]["data_low"][metric_name].keys())
        bounds = [0]
        blocks = []
        for name, res in self.runs[0]:
            n_eps = len(next(iter(res["data_low"]["losses_clean"].values())))
            bounds.append(bounds[-1] + n_eps)
            blocks.append(name)

        has_noise = any(b.get("sd", 0.0) > 0 for b in self.config.get("exp_blocks", []))

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

        parts = metric_name.split('_')
        y_label = " ".join([p if p.isupper() else p.capitalize() for p in parts])

        ax.set(xlabel='Epochs', ylabel=y_label)
        ax.xaxis.labelpad = 20
        ax.grid(True, which="both", ls="-", alpha=0.5)

        title_suffix = f" ({len(self.runs)} Runs Averaged)" if self.is_multi else ""
        ax.set_title(f"{y_label}{title_suffix}", fontweight='bold', fontsize=14, pad=25)

        ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=True, edgecolor='gray', fontsize=8)

        if show_config:
            cfg_ax = fig.add_axes([0.76, 0.15, 0.22, 0.73])
            cfg_ax.axis('off')
            self._add_config_info(cfg_ax, show_config=show_config)

        self._save_fig(f"{metric_name}_{envs}_figure_{self.exp_name.replace(' ', '_')}")
        plt.show()

    def plot_parameter_distributions(self, layer_name="fc1", param_type="weight", bins=50, show_config=True):
        from matplotlib import animation
        from IPython.display import display, HTML

        param_type = param_type.lower()
        if "weight" in param_type:
            actual_param = "weight"
            group = "weights"
        elif "bias" in param_type:
            actual_param = "bias"
            group = "biases"
        else:
            print("Error: param_type must be either 'weight' or 'bias'")
            return

        key = f"_layers.{layer_name}.{actual_param}"

        low_data, high_data, epoch_labels = [], [], []

        for b_name, _ in self.runs[0]:
            n_eps = len(self.runs[0][0][1]["data_low"]["losses_clean"][
                            list(self.runs[0][0][1]["data_low"]["losses_clean"].keys())[0]])
            for ep in range(n_eps):
                epoch_labels.append(f"Block: {b_name} | Epoch: {ep}")

        for ep_idx in range(len(epoch_labels)):
            low_ep_params = []
            high_ep_params = []

            cur_ep = ep_idx
            b_idx = 0
            while cur_ep >= len(self.runs[0][b_idx][1]["data_low"]["losses_clean"][
                                    list(self.runs[0][b_idx][1]["data_low"]["losses_clean"].keys())[0]]):
                cur_ep -= len(self.runs[0][b_idx][1]["data_low"]["losses_clean"][
                                  list(self.runs[0][b_idx][1]["data_low"]["losses_clean"].keys())[0]])
                b_idx += 1

            for run_res in self.runs:
                b_res = run_res[b_idx][1]
                if key not in b_res["data_low"][group]:
                    print(f"Warning: '{key}' not found in tracked {group}.")
                    return
                low_ep_params.append(b_res["data_low"][group][key][cur_ep])
                high_ep_params.append(b_res["data_high"][group][key][cur_ep])

            low_data.append(np.concatenate([p.flatten() for p in low_ep_params]))
            high_data.append(np.concatenate([p.flatten() for p in high_ep_params]))

        if not low_data:
            return

        all_low = np.concatenate(low_data)
        all_high = np.concatenate(high_data)

        low_min, low_max = np.percentile(all_low, 0.5), np.percentile(all_low, 99.5)
        high_min, high_max = np.percentile(all_high, 0.5), np.percentile(all_high, 99.5)

        low_pad = (low_max - low_min) * 0.05
        high_pad = (high_max - high_min) * 0.05

        low_range = (low_min - low_pad, low_max + low_pad)
        high_range = (high_min - high_pad, high_max + high_pad)

        y_max_low, y_max_high = 0, 0
        for l, h in zip(low_data, high_data):
            c_l, _ = np.histogram(l, bins=bins, range=low_range, density=True)
            c_h, _ = np.histogram(h, bins=bins, range=high_range, density=True)
            y_max_low = max(y_max_low, c_l.max())
            y_max_high = max(y_max_high, c_h.max())

        y_max_low *= 1.1
        y_max_high *= 1.1

        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        fig.subplots_adjust(top=0.80, bottom=0.15, wspace=0.3, right=0.74 if show_config else 0.95)

        def update(frame_idx):
            for ax in axs:
                ax.clear()

            axs[0].hist(low_data[frame_idx], bins=bins, range=low_range, density=True, color='blue',
                        alpha=0.7, edgecolor='black')
            axs[0].set_title(f"Low Variance (RichMLP)\n{epoch_labels[frame_idx]}", fontweight='bold', pad=15)
            axs[0].set_xlim(low_range)
            axs[0].set_ylim(0, y_max_low)
            axs[0].grid(True, linestyle='--', alpha=0.5)
            axs[0].set_ylabel("Density")
            axs[0].set_xlabel("Value")

            axs[1].hist(high_data[frame_idx], bins=bins, range=high_range, density=True, color='red',
                        alpha=0.7, edgecolor='black')
            axs[1].set_title(f"High Variance (LazyMLP)\n{epoch_labels[frame_idx]}", fontweight='bold', pad=15)
            axs[1].set_xlim(high_range)
            axs[1].set_ylim(0, y_max_high)
            axs[1].grid(True, linestyle='--', alpha=0.5)
            axs[1].set_ylabel("Density")
            axs[1].set_xlabel("Value")

        title_suffix = f" (Pooled over {len(self.runs)} Runs)" if self.is_multi else ""
        fig.suptitle(f"Layer '{layer_name}' {actual_param.capitalize()} Distributions over Time{title_suffix}",
                     fontweight='bold',
                     fontsize=18, y=0.92)

        if show_config:
            cfg_ax = fig.add_axes([0.76, 0.15, 0.22, 0.65])
            cfg_ax.axis('off')
            self._add_config_info(cfg_ax, show_config=show_config)

        anim = animation.FuncAnimation(fig, update, frames=len(low_data), interval=200, repeat=False)
        plt.close(fig)
        display(HTML(anim.to_jshtml()))

    def plot_PCS(self, epochs=(-1,), layer_name='fc1', show_config=True):
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
        distinct_rgbs = get_distinct_colors(len(unique_labels))
        color_map = {lbl: distinct_rgbs[i] for i, lbl in enumerate(unique_labels)}

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=lbl,
                                  markerfacecolor=color_map[lbl], markersize=10) for lbl in unique_labels]

        n_runs = len(self.runs)
        total_epochs = sum(len(b_res["data_low"]["activations"]) for _, b_res in self.runs[0])
        N_samples = len(env_X)

        raw_coords = {'low': np.zeros((n_runs, total_epochs, N_samples, 2)),
                      'high': np.zeros((n_runs, total_epochs, N_samples, 2))}
        pca_ev = {'low': np.zeros((n_runs, total_epochs, 2)),
                  'high': np.zeros((n_runs, total_epochs, 2))}

        for r_idx, run in enumerate(self.runs):
            for m_type in ['low', 'high']:
                ep_idx = 0
                for b_name, b_res in run:
                    acts_list = b_res[f"data_{m_type}"]["activations"]
                    for layer_acts in acts_list:
                        acts = layer_acts[layer_name]
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

                        raw_coords[m_type][r_idx, ep_idx] = c
                        pca_ev[m_type][r_idx, ep_idx] = ev
                        ep_idx += 1

        pca_coords = {'low': np.zeros_like(raw_coords['low']),
                      'high': np.zeros_like(raw_coords['high'])}

        pca_coords['low'][0, 0] = raw_coords['low'][0, 0]
        for ep in range(1, total_epochs):
            pca_coords['low'][0, ep] = align_coords_no_reflection(raw_coords['low'][0, ep],
                                                                  pca_coords['low'][0, ep - 1])

        for ep in range(total_epochs):
            pca_coords['high'][0, ep] = align_coords_no_reflection(raw_coords['high'][0, ep], pca_coords['low'][0, ep])

        for r_idx in range(1, n_runs):
            for ep in range(total_epochs):
                pca_coords['low'][r_idx, ep] = align_coords_no_reflection(raw_coords['low'][r_idx, ep],
                                                                          pca_coords['low'][0, ep])
                pca_coords['high'][r_idx, ep] = align_coords_no_reflection(raw_coords['high'][r_idx, ep],
                                                                           pca_coords['high'][0, ep])

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

        vlines = {}
        for m_type, ax_ev, title in zip(['low', 'high'], [ax_ev_low, ax_ev_high],
                                        ['Low Var Explained Variance', 'High Var Explained Variance']):
            pc1_vals = pca_ev[m_type][:, :, 0]
            pc2_vals = pca_ev[m_type][:, :, 1]
            cum_vals = np.sum(pca_ev[m_type][:, :, :2], axis=2)

            for r_idx in range(n_runs):
                ax_ev.plot(cum_vals[r_idx, :], color='purple', alpha=0.15, linestyle=':')

            ax_ev.plot(pc1_vals.mean(axis=0), color='blue', label='PC1', linewidth=1.5)
            ax_ev.plot(pc2_vals.mean(axis=0), color='orange', label='PC2', linewidth=1.5)
            ax_ev.plot(cum_vals.mean(axis=0), color='purple', label='PC1 + PC2', linewidth=2.5)

            vlines[m_type] = ax_ev.axvline(0, color='black', linestyle='--', linewidth=2)

            ax_ev.set_title(title, fontsize=10, fontweight='bold')
            ax_ev.set_ylim(-0.05, 1.05)
            ax_ev.set_xlim(0, total_epochs)
            ax_ev.grid(True, linestyle=':', alpha=0.6)

        ax_ev_high.legend(loc='lower left', bbox_to_anchor=(1.02, 0.0), fontsize=8)

        titles = ['Low Variance (RichMLP)', 'High Variance (LazyMLP)']

        def update(frame_idx):
            epoch = epochs[frame_idx]

            rel_ep, cur_ep, b_idx, t_block = 0, 0, 0, None
            for b_name, b_res in self.runs[0]:
                n_eps = len(b_res["data_low"]["activations"])
                if epoch == -1 or cur_ep <= epoch < cur_ep + n_eps:
                    t_block, rel_ep = b_name, (n_eps - 1) if epoch == -1 else (epoch - cur_ep)
                    break
                cur_ep += n_eps;
                b_idx += 1

            target_ep_idx = cur_ep + rel_ep if epoch != -1 else total_epochs - 1

            for idx, (ax, m_type) in enumerate(zip([ax_low, ax_high], ['low', 'high'])):
                ax.clear()
                if t_block is None:
                    ax.text(0.5, 0.5, f"Epoch {epoch}\nNot Found", ha='center', va='center', color='red', fontsize=12)
                    ax.axis('off')
                    continue

                for r_idx in range(n_runs):
                    coords = pca_coords[m_type][r_idx, target_ep_idx]
                    for lbl in unique_labels:
                        mask = [s == lbl for s in f_strs]
                        ax.scatter(coords[mask, 0], coords[mask, 1], color=color_map[lbl], s=100, alpha=0.6,
                                   edgecolors='none')

                ax.margins(0.15);
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_title(f"{titles[idx]}\nEpoch: {'Last' if epoch == -1 else epoch} | Block: {t_block}",
                             fontweight='bold', fontsize=13)

                vlines[m_type].set_xdata([target_ep_idx, target_ep_idx])

            ax_high.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(1.02, 0.0), title="Features")

        title_suffix = f" ({len(self.runs)} Runs)" if self.is_multi else ""
        fig.suptitle(f"PCA Evolution of '{layer_name}' Activations in {self.exp_name.capitalize()}{title_suffix}",
                     fontweight='bold', fontsize=18, y=0.98)

        anim = animation.FuncAnimation(fig, update, frames=len(epochs), interval=200, repeat=False)
        plt.close(fig)
        display(HTML(anim.to_jshtml()))