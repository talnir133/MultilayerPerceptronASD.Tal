import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from functools import partial
import os

from core_ml import Dataset, MLP, train_model
from classification_rules import RULES_REGISTRY
import sys
import contextlib
from copy import deepcopy
from tqdm import tqdm


class Simulation:
    def __init__(self, config):
        self.track_metrics = None
        self.global_config = copy.deepcopy(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = Dataset(config["features_types"])
        self.dataset.create_exp_data()
        self.X_global = self.dataset.exp_X

        torch.manual_seed(self.global_config["seed"])
        np.random.seed(self.global_config["seed"])

    def run(self, track_metrics=True):
        self.track_metrics = track_metrics
        simulation_results = []
        optimizers, models = {"low": None, "high": None}, {"low": None, "high": None}

        test_envs = self._initialize_test_envs()

        for i, block in enumerate(self.global_config["exp_blocks"]):
            block_config = self.get_block_configs(block)
            block_results = self.run_block(models, optimizers, block_config, test_envs)
            models["low"], models["high"] = block_results["model_low"], block_results["model_high"]
            optimizers["low"], optimizers["high"] = block_results["optimizer_low"], block_results["optimizer_high"]
            simulation_results.append([block_config["block_name"], block_results])

        return simulation_results

    def run_block(self, models, optimizers, cfg, test_envs):
        print(f"\n--- Running Block: {cfg['block_name']} ---")
        rule_name = cfg.get("rule")
        rule_to_apply = partial(RULES_REGISTRY[rule_name], **cfg)

        X = self.dataset.get_block_data(cfg.get("zero_features", []))
        X = X.to(self.device)
        y = rule_to_apply(X).to(self.device)
        noise_mask = get_noise_mask(cfg, X).to(self.device)
        X_global_dev = self.X_global.to(self.device)

        data_low = create_tracker(test_envs, self.X_global)
        data_high = create_tracker(test_envs, self.X_global)

        # ==========================================
        # LOW VARIANCE
        # ==========================================

        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()

        if models["low"] is None:
            models["low"], optimizers["low"] = self._init_model_and_optimizer(cfg, "low")

        model_low, optimizer_low = train_model(
            models["low"], optimizers["low"], X, rule_to_apply,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=get_metric_callback(data_low, cfg, test_envs, X_global_dev) if self.track_metrics else lambda m, x, c: None
        )

        # ==========================================
        # HIGH VARIANCE
        # ==========================================

        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

        if models["high"] is None:
            models["high"], optimizers["high"] = self._init_model_and_optimizer(cfg, "high")

        model_high, optimizer_high = train_model(
            models["high"], optimizers["high"], X, rule_to_apply,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=get_metric_callback(data_high, cfg, test_envs, X_global_dev) if self.track_metrics else lambda m, x, c: None
        )

        return {
            "X": X, "y": y, "config": cfg,
            "model_low": model_low, "optimizer_low": optimizer_low, "data_low": data_low,
            "model_high": model_high, "optimizer_high": optimizer_high, "data_high": data_high
        }

    def _initialize_test_envs(self):
        test_envs = {}
        for block in self.global_config["exp_blocks"]:
            b_name = block["block_name"]
            if b_name not in test_envs:
                b_cfg = self.get_block_configs(block)
                rule_func = partial(RULES_REGISTRY[b_cfg["rule"]], **b_cfg)
                X_test = self.dataset.get_block_data(b_cfg.get("zero_features", []))
                X_test = X_test.to(self.device)
                y_test = rule_func(X_test).to(self.device)
                mask = get_noise_mask(b_cfg, X_test).to(self.device)
                test_envs[b_name] = {
                    "X": X_test,
                    "y": y_test[:, 0:1],
                    "mask": mask,
                    "rule_func": rule_func,
                    "sd": b_cfg["sd"]
                }
        return test_envs

    def _init_model_and_optimizer(self, cfg, variance_type):
        w_scale = cfg[f"w_scale_{variance_type}"]
        b_scale = cfg[f"b_scale_{variance_type}"]

        model = MLP(
            cfg["input_size"], cfg["hidden_size"], cfg["n_hidden"],
            cfg["output_size"], w_scale,
            b_scale, cfg["activation_type"]
        ).to(self.device)
        model.reinitialize()

        optimizer = cfg["optimizer_type"](model.parameters(), lr=cfg.get("lr", 0.004))
        return model, optimizer

    def get_block_configs(self, block):
        block_config = copy.deepcopy(self.global_config)
        block_config.update(block)

        block_config["input_size"] = sum(block_config["features_types"])
        block_config["output_size"] = 1 + block_config["input_size"]
        block_config.setdefault("alpha_class", 1.0)
        block_config.setdefault("alpha_rec", 0.0)
        block_config.setdefault("sd", 0.0)
        opt_map = {"Adam": optim.Adam, "SGD": optim.SGD}
        block_config["optimizer_type"] = opt_map[block_config["optimizer_type"]]
        act_map = {
            "Tanh": nn.Tanh, "RelU": nn.ReLU,
            "Sigmoid": nn.Sigmoid, "Identity": nn.Identity
        }
        block_config["activation_type"] = act_map.get(block_config["activation_type"], nn.Identity)()

        return block_config

# ================================================
# "Simulation" Helper Functions
# ================================================

def get_noise_mask(block_config, X):
    noise_mask = torch.ones_like(X)
    zero_feats = block_config.get("zero_features", [])
    features_types = block_config.get("features_types", [])

    start_idx = 0
    for i, dim in enumerate(features_types):
        if i in zero_feats:
            noise_mask[:, start_idx: start_idx + dim] = 0.0
        start_idx += dim

    return noise_mask


def create_tracker(test_envs, X_global):
    tracker = {
        "weights": {},
        "biases": {},
        "X_global": X_global.cpu().numpy(),
        "test_envs": {env: {"X": data["X"].cpu().numpy(), "y": data["y"].cpu().numpy()} for env, data in
                      test_envs.items()},
        "activations": [],
        "activation_distances": [],
        "PR_weights": [],
        "bias_weight_correlation": []
    }
    metrics = [
        "losses_clean", "MAE_clean", "accuracy_clean",
        "losses_noisy", "MAE_noisy", "accuracy_noisy",
        "losses_noisy_optimal", "MAE_noisy_optimal", "accuracy_noisy_optimal",
        "PR_activations"
    ]
    for m in metrics:
        tracker[m] = {env: [] for env in test_envs.keys()}
    return tracker


def get_metric_callback(tracker, cfg, test_envs, X_global, target_layer="fc1"):

    def callback(current_model, loss_criterion):

        # ==========================================
        # 1. Collection of Raw Parameters
        # ==========================================
        for name, param in current_model.named_parameters():
            if 'weight' in name:
                if name not in tracker["weights"]:
                    tracker["weights"][name] = []
                tracker["weights"][name].append(param.detach().cpu().numpy().copy())
            elif 'bias' in name and param is not None:
                if name not in tracker["biases"]:
                    tracker["biases"][name] = []
                tracker["biases"][name].append(param.detach().cpu().numpy().copy())

        # ==========================================
        # 2. Collection of Global Activations
        # ==========================================
        acts = get_network_activations(current_model, X_global)
        epoch_act_dist = {}
        for name, act_np in acts.items():
            epoch_act_dist[name] = pairwise_distances(act_np)[np.triu_indices(X_global.shape[0], k=1)]
        tracker["activations"].append(acts)
        tracker["activation_distances"].append(epoch_act_dist)

        # ==========================================
        # 3. Global Structural Metrics Calculation
        # ==========================================
        for name, layer in current_model._layers.named_children():
            if name == target_layer:
                w_np = layer.weight.detach().cpu().numpy()
                b_np = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(w_np.shape[0])

                # Calculate PR Weights (Normalized)
                num_w = np.sum(w_np ** 2, axis=1) ** 2
                den_w = np.sum(w_np ** 4, axis=1)
                N_weights = w_np.shape[1]
                pr_w_raw = np.mean(np.divide(num_w, den_w, out=np.zeros_like(num_w), where=den_w != 0))
                tracker["PR_weights"].append(pr_w_raw / N_weights)

                # Calculate Pearson Correlation (Weights L2 Norm vs Abs Bias)
                w_norm = np.linalg.norm(w_np, axis=1)
                abs_b = np.abs(b_np.flatten())
                if np.std(w_norm) == 0 or np.std(abs_b) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(w_norm, abs_b)[0, 1]
                tracker["bias_weight_correlation"].append(corr)
                break

        # ==========================================
        # 4. Environment Specific Metrics
        # ==========================================
        for env_name, env_data in test_envs.items():
            X_env = env_data["X"]
            y_env = env_data["y"]
            env_sd = env_data["sd"]

            # Calculate PR for Activations (Normalized)
            env_acts = get_network_activations(current_model, X_env)[target_layer]
            num_a = np.sum(env_acts ** 2, axis=1) ** 2
            den_a = np.sum(env_acts ** 4, axis=1)
            N_neurons = env_acts.shape[1]
            pr_a_raw = np.mean(np.divide(num_a, den_a, out=np.zeros_like(num_a), where=den_a != 0))
            tracker["PR_activations"][env_name].append(pr_a_raw / N_neurons)

            # Clean Predictions
            clean_preds = current_model(X_env)[:, 0:1]
            clean_probs = torch.sigmoid(clean_preds)
            tracker["losses_clean"][env_name].append(loss_criterion(clean_preds, y_env).item())
            tracker["MAE_clean"][env_name].append(torch.abs(clean_probs - y_env).mean().item())
            tracker["accuracy_clean"][env_name].append(
                ((clean_probs > 0.5) == y_env.bool()).float().mean().item())

            # Noisy Predictions (If applicable for specific environment)
            if env_sd > 0:
                X_noisy_env = X_env + (torch.randn_like(X_env) * env_sd * env_data["mask"])
                noisy_preds = current_model(X_noisy_env)[:, 0:1]
                opt_probs = get_bayes_optimal_probabilities(X_noisy_env, env_sd, X_env, y_env)
                model_probs = torch.sigmoid(noisy_preds)

                tracker["losses_noisy"][env_name].append(loss_criterion(noisy_preds, y_env).item())
                tracker["MAE_noisy"][env_name].append(torch.abs(model_probs - y_env).mean().item())
                tracker["accuracy_noisy"][env_name].append(
                    ((model_probs > 0.5) == y_env.bool()).float().mean().item())
                tracker["MAE_noisy_optimal"][env_name].append(torch.abs(opt_probs - y_env).mean().item())
                tracker["accuracy_noisy_optimal"][env_name].append(
                    ((opt_probs > 0.5) == y_env.bool()).float().mean().item())

    return callback


def get_bayes_optimal_probabilities(noisy_X, noise_sd, clean_X, clean_y):
    if noise_sd <= 1e-7:
        idx = torch.cdist(noisy_X, clean_X).argmin(dim=1)
        return clean_y[idx]

    sq_dists = torch.cdist(noisy_X, clean_X) ** 2
    likelihoods = torch.exp(-sq_dists / (2 * noise_sd ** 2))
    prob_y1 = (likelihoods @ clean_y) / likelihoods.sum(dim=1, keepdim=True)

    return prob_y1

def get_network_activations(model, x):
    activations = {}
    out = x
    for name, layer in model._layers.named_children():
        out = layer(out)
        activations[name] = out.detach().cpu().numpy()
    return activations

# ================================================
# Multi-Simulation
# ================================================

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def averaged_simulation(config, n):
    simulations = []
    for i in tqdm(range(n), desc="Simulations Progress"):
        cfg = deepcopy(config)
        cfg["seed"] = i
        with suppress_output():
            simulations.append(Simulation(cfg).run(track_metrics=True))
    return simulations