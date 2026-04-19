import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
from core_ml import Dataset, MLP, train_model
from classification_rules import RULES_REGISTRY


class Simulation:
    def __init__(self, config):
        self.active_rules = None
        self.track_metrics = None
        self.global_config = copy.deepcopy(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = Dataset(config["features_types"])
        self.dataset.create_exp_data()

        # Lock random seed for reproducibility
        torch.manual_seed(self.global_config["seed"])
        np.random.seed(self.global_config["seed"])

    def run(self, track_metrics=True):
        self.track_metrics = track_metrics
        simulation_results = []
        optimizers, models = {"low": None, "high": None}, {"low": None, "high": None}
        test_envs = {}

        for i, block in enumerate(self.global_config["exp_blocks"]):
            block_config = self.get_block_configs(block)
            self._sync_environments(i, block_config, test_envs)
            block_results = self.run_block(models, optimizers, block_config, test_envs)
            
            models["low"], models["high"] = block_results["model_low"], block_results["model_high"]
            optimizers["low"], optimizers["high"] = block_results["optimizer_low"], block_results["optimizer_high"]
            simulation_results.append([block_config["block_name"], block_results])

        return simulation_results

    def run_block(self, models, optimizers, cfg, test_envs):
        print(f"\n--- Running Block: {cfg['block_name']} ---")
        rule_name = cfg.get("rule", "upper_half")
        rule_to_apply = partial(RULES_REGISTRY[rule_name], **cfg)
        X, y = self.dataset.get_block_data(cfg.get("zero_features", []), rule_to_apply)
        X, y = X.to(self.device), y.to(self.device)
        noise_mask = get_noise_mask(cfg, X).to(self.device)

        data_low = create_tracker(test_envs, X, y)
        data_high = create_tracker(test_envs, X, y)

        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()

        # ==========================================
        # LOW VARIANCE
        # ==========================================

        if models["low"] is None:
            models["low"] = MLP(
                cfg["input_size"], cfg["hidden_size"], cfg["n_hidden"],
                cfg["output_size"], cfg["w_scale_low"],
                cfg["b_scale_low"], cfg["activation_type"]
            ).to(self.device)
            models["low"].reinitialize()
            optimizers["low"] = cfg["optimizer_type"](models["low"].parameters(), lr=cfg.get("lr", 0.004))

        model_low, optimizer_low = train_model(
            models["low"], optimizers["low"], X, y,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=get_metric_callback(data_low, cfg, test_envs) if self.track_metrics else lambda m, x,
                                                                                                            c: None
        )

        # ==========================================
        # HIGH VARIANCE
        # ==========================================

        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

        if models["high"] is None:
            models["high"] = MLP(
                cfg["input_size"], cfg["hidden_size"], cfg["n_hidden"],
                cfg["output_size"], cfg["w_scale_high"],
                cfg["b_scale_high"], cfg["activation_type"]
            ).to(self.device)
            models["high"].reinitialize()
            optimizers["high"] = cfg["optimizer_type"](models["high"].parameters(), lr=cfg.get("lr", 0.004))

        model_high, optimizer_high = train_model(
            models["high"], optimizers["high"], X, y,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=get_metric_callback(data_high, cfg, test_envs) if self.track_metrics else lambda m, x,
                                                                                                             c: None
        )

        return {
            "X": X, "y": y, "config": cfg,
            "model_low": model_low, "optimizer_low": optimizer_low, "data_low": data_low,
            "model_high": model_high, "optimizer_high": optimizer_high, "data_high": data_high
        }

    def get_block_configs(self, block):
        block_config = copy.deepcopy(self.global_config)
        block_config.update(block)

        block_config["input_size"] = sum(block_config["features_types"])
        block_config["output_size"] = 1 + block_config["input_size"]
        block_config.setdefault("alpha_class", 1.0)
        block_config.setdefault("alpha_rec", 0.0)
        opt_map = {"Adam": optim.Adam, "SGD": optim.SGD}
        block_config["optimizer_type"] = opt_map[block_config["optimizer_type"]]

        act_map = {
            "Tanh": nn.Tanh, "RelU": nn.ReLU,
            "Sigmoid": nn.Sigmoid, "Identity": nn.Identity
        }
        block_config["activation_type"] = act_map.get(block_config["activation_type"], nn.Identity)()

        return block_config

    def _sync_environments(self, current_idx, current_cfg, test_envs):
        # Phase 1: Initial Setup
        if not test_envs:
            self.active_rules = {}
            for block in self.global_config["exp_blocks"]:
                b_cfg = self.get_block_configs(block)
                domain_sig, rule_sig = self._get_signatures(b_cfg)

                if domain_sig not in test_envs:
                    r_func = partial(RULES_REGISTRY[b_cfg["rule"]], **b_cfg)
                    X_test, y_test = self.dataset.get_block_data(b_cfg["zero_features"], r_func)

                    test_envs[domain_sig] = {
                        "X": X_test.to(self.device),
                        "y": y_test[:, 0:1].to(self.device),
                        "mask": get_noise_mask(b_cfg, X_test).to(self.device)
                    }
                    self.active_rules[domain_sig] = rule_sig

            self._update_combined_env(test_envs)
            return

        # Phase 2: Shift Detection & Update
        domain_sig, rule_sig = self._get_signatures(current_cfg)

        if rule_sig == self.active_rules.get(domain_sig, rule_sig):
            return

        print(f"\n[!!!] GLOBAL SHIFT TRIGGERED AT BLOCK '{current_cfg['block_name']}' [!!!]")
        domains_to_update = [k for k in test_envs.keys() if k != "Combined"]

        for d_sig in domains_to_update:
            for i in range(current_idx, len(self.global_config["exp_blocks"])):
                b_cfg = self.get_block_configs(self.global_config["exp_blocks"][i])
                loop_d_sig, loop_r_sig = self._get_signatures(b_cfg)

                if loop_d_sig == d_sig:
                    r_func = partial(RULES_REGISTRY[b_cfg.get("rule", "upper_half")], **b_cfg)
                    _, y_test = self.dataset.get_block_data(b_cfg.get("zero_features", []), r_func)

                    test_envs[d_sig]["y"] = y_test[:, 0:1].to(self.device)
                    self.active_rules[d_sig] = loop_r_sig
                    break

        self._update_combined_env(test_envs)

    def _update_combined_env(self, test_envs):
        all_X, all_y = [], []
        for k, env in test_envs.items():
            if k != "Combined":
                all_X.append(env["X"])
                all_y.append(env["y"])

        if all_X:
            comb_data = torch.unique(torch.cat((torch.cat(all_X, dim=0), torch.cat(all_y, dim=0)), dim=1), dim=0)
            D = all_X[0].shape[1]
            test_envs["Combined"] = {
                "X": comb_data[:, :D],
                "y": comb_data[:, D:],
                "mask": torch.ones_like(comb_data[:, :D])
            }

    def _get_signatures(self, b_cfg):
        zf = b_cfg.get("zero_features", [])
        domain_sig = f"Domain(Zero:{','.join(map(str, sorted(zf))) or 'None'})"
        rule_sig = f"{b_cfg.get('rule')}_{b_cfg.get('deciding_feature', '')}"
        return domain_sig, rule_sig


# ==========================================
# Simulation's Helper Functions
# ==========================================

def get_noise_mask(block_config, X):
    # creating a mask to apply noise only to non-zeroed features
    noise_mask = torch.ones_like(X)
    zero_feats = block_config.get("zero_features", [])
    features_types = block_config.get("features_types", [])

    start_idx = 0
    for i, dim in enumerate(features_types):
        if i in zero_feats:
            noise_mask[:, start_idx: start_idx + dim] = 0.0
        start_idx += dim

    return noise_mask


def create_tracker(test_envs, X_train, y_train):
    tracker = {
        "noised_data": [], "fc1_weight_sd": [], "fc1_bias_sd": [],
        "X_train": X_train, "y_train": y_train,
        "envs_data": {env: {"X": data["X"].cpu().numpy(), "y": data["y"].cpu().numpy()} for env, data in
                      test_envs.items()}
    }
    metrics = [
        "losses_clean", "MAE_clean", "accuracies_clean",
        "losses_noisy", "MAE_noisy", "accuracies_noisy",
        "losses_noisy_optimal", "MAE_noisy_optimal", "accuracies_noisy_optimal",
        "MAE_to_optimal", "accuracies_to_optimal",
        "activations_clean", "activation_distances_clean"
    ]
    for m in metrics:
        tracker[m] = {env: [] for env in test_envs.keys()}
    return tracker


def get_metric_callback(tracker, cfg, test_envs):
    global_sd = cfg.get("sd", 0)

    def callback(current_model, X_noisy, loss_criterion):
        tracker["noised_data"].append(X_noisy.cpu().detach().numpy())
        tracker["fc1_weight_sd"].append(current_model._layers.fc1.weight.std().item())
        tracker["fc1_bias_sd"].append(
            current_model._layers.fc1.bias.std().item() if current_model._layers.fc1.bias is not None else 0.0)

        for env_name, env_data in test_envs.items():
            X_env = env_data["X"]
            y_env = env_data["y"]

            acts = get_network_activations(current_model, X_env)
            epoch_act_dist = {}
            for name, act_np in acts.items():
                epoch_act_dist[name] = pairwise_distances(act_np)[np.triu_indices(X_env.shape[0], k=1)]
            tracker["activations_clean"][env_name].append(acts)
            tracker["activation_distances_clean"][env_name].append(epoch_act_dist)

            clean_preds = current_model(X_env)[:, 0:1]
            tracker["losses_clean"][env_name].append(loss_criterion(clean_preds, y_env).item())
            tracker["MAE_clean"][env_name].append(torch.abs(torch.sigmoid(clean_preds) - y_env).mean().item())
            tracker["accuracies_clean"][env_name].append(
                ((torch.sigmoid(clean_preds) > 0.5) == y_env.bool()).float().mean().item())

            if global_sd > 0:
                X_noisy_env = X_env + (torch.randn_like(X_env) * global_sd * env_data["mask"])
                noisy_preds = current_model(X_noisy_env)[:, 0:1]
                opt_probs = get_bayes_optimal_probabilities(X_noisy_env, global_sd, X_env, y_env)
                model_probs = torch.sigmoid(noisy_preds)

                tracker["losses_noisy"][env_name].append(loss_criterion(noisy_preds, y_env).item())
                tracker["MAE_noisy"][env_name].append(torch.abs(model_probs - y_env).mean().item())
                tracker["accuracies_noisy"][env_name].append(
                    ((model_probs > 0.5) == y_env.bool()).float().mean().item())
                tracker["MAE_noisy_optimal"][env_name].append(torch.abs(opt_probs - y_env).mean().item())
                tracker["accuracies_noisy_optimal"][env_name].append(
                    ((opt_probs > 0.5) == y_env.bool()).float().mean().item())
                tracker["MAE_to_optimal"][env_name].append(torch.abs(model_probs - opt_probs).mean().item())
                tracker["accuracies_to_optimal"][env_name].append(
                    ((model_probs > 0.5) == (opt_probs > 0.5)).float().mean().item())

    return callback


def get_bayes_optimal_probabilities(noisy_X, noise_sd, clean_X, clean_y):
    """
    Generalized Bayesian optimal predictor.
    Calculates P(y=1 | noisy_X) by marginalizing over all possible clean prototypes.
    """
    if noise_sd <= 1e-7:
        idx = torch.cdist(noisy_X, clean_X).argmin(dim=1)
        return clean_y[idx]

    sq_dists = torch.cdist(noisy_X, clean_X) ** 2
    likelihoods = torch.exp(-sq_dists / (2 * noise_sd ** 2))
    prob_y1 = (likelihoods @ clean_y) / likelihoods.sum(dim=1, keepdim=True)

    return prob_y1


def get_network_activations(model, x):
    """Manually passes data through the network layer by layer to collect activations."""
    activations = {}
    out = x
    for name, layer in model._layers.named_children():
        out = layer(out)
        activations[name] = out.detach().cpu().numpy()
    return activations