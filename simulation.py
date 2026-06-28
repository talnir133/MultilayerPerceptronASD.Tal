import os
import sys
import contextlib
from functools import partial
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from core_ml import Dataset, Classifier, train_classifier
from classification_rules import RULES_REGISTRY
from metrics import create_tracker, get_metric_callback, get_noise_mask

class Simulation:
    def __init__(self, config):
        self.track_metrics = None
        self.global_config = deepcopy(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = Dataset(config["features_types"])
        self.dataset.create_exp_data()
        self.X_global = self.dataset.exp_X

        torch.manual_seed(self.global_config["seed"])
        np.random.seed(self.global_config["seed"])

    def run(self, track_metrics=True):
        self.track_metrics = track_metrics
        simulation_results = []

        models = {"low": None, "high": None}
        optimizers = {"low": None, "high": None}

        test_envs = self._initialize_test_envs()

        for i, block in enumerate(self.global_config["exp_blocks"]):
            block_config = self.get_block_configs(block)

            if i == 0:
                models["low"], optimizers["low"] = self._init_model_and_optimizer(block_config, "low")
                models["high"], optimizers["high"] = self._init_model_and_optimizer(block_config, "high")

            block_results = self.run_block(models, optimizers, block_config, test_envs)
            models["low"], models["high"] = block_results["model_low"], block_results["model_high"]
            optimizers["low"], optimizers["high"] = block_results["optimizer_low"], block_results["optimizer_high"]

            simulation_results.append([block_config["block_name"], block_results])

        return simulation_results

    def run_block(self, models, optimizers, cfg, test_envs):
        print(f"\n--- Running Block: {cfg['block_name']} ---")

        rule_to_apply = partial(RULES_REGISTRY[cfg["rule"]], **cfg)

        X = self.dataset.get_block_data(cfg.get("zero_features", [])).to(self.device)
        y = rule_to_apply(X).to(self.device)
        noise_mask = get_noise_mask(cfg, X).to(self.device)
        X_global_dev = self.X_global.to(self.device)

        data_low = create_tracker(test_envs, self.X_global)
        data_high = create_tracker(test_envs, self.X_global)

        if self.track_metrics:
            callback_low = get_metric_callback(data_low, test_envs, X_global_dev, cfg.get("decoder", False), cfg["features_types"])
            callback_high = get_metric_callback(data_high, test_envs, X_global_dev, cfg.get("decoder", False), cfg["features_types"])
        else:
            callback_low, callback_high = None, None

        # ==========================================
        # LOW VARIANCE
        # ==========================================
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()

        model_low, optimizer_low = train_classifier(
            models["low"], optimizers["low"], X, rule_to_apply,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=callback_low
        )

        # ==========================================
        # HIGH VARIANCE
        # ==========================================
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

        model_high, optimizer_high = train_classifier(
            models["high"], optimizers["high"], X, rule_to_apply,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=callback_high
        )

        return {
            "X": X, "y": y, "config": cfg,
            "model_low": model_low, "optimizer_low": optimizer_low, "data_low": data_low,
            "model_high": model_high, "optimizer_high": optimizer_high, "data_high": data_high
        }

    def get_block_configs(self, block):
        # Base config merging
        block_config = deepcopy(self.global_config)
        block_config.update(block)

        # Dimensions & Object Mapping
        block_config["input_size"] = sum(block_config["features_types"])
        block_config["output_size"] = 1 + block_config["input_size"]
        act_map = {
            "Tanh": nn.Tanh, "RelU": nn.ReLU,
            "Sigmoid": nn.Sigmoid, "Identity": nn.Identity
        }
        block_config["activation_type"] = act_map.get(block_config["activation_type"], nn.Identity)()
        opt_map = {"Adam": optim.Adam, "SGD": optim.SGD}
        block_config["optimizer_type"] = opt_map[block_config["optimizer_type"]]

        # Setting Defaults
        block_config.setdefault("alpha_class", 1.0)
        block_config.setdefault("alpha_rec", 0.0)
        block_config.setdefault("sd", 0.0)

            # Decoder Defaults
        if "decoder" in self.global_config.keys():
            dec_cfg = block_config.setdefault("decoder", {})
            dec_cfg.setdefault("train_sd", 0)
            dec_cfg.setdefault("test_sd", 0.2)
            dec_cfg.setdefault("samples_per_point", 20)
            dec_cfg.setdefault("freq", 5)
            dec_cfg.setdefault("epochs", 100)
            dec_cfg.setdefault("lr", 0.1)

        return block_config

    def _init_model_and_optimizer(self, cfg, variance_type):
        model = Classifier(
            input_size=cfg["input_size"],
            output_size=cfg["output_size"],
            w_scale=cfg[f"w_scale_{variance_type}"],
            b_scale=cfg[f"b_scale_{variance_type}"],
            activation_type=cfg["activation_type"],
            hidden_size=cfg["hidden_size"],
            n_hidden=cfg["n_hidden"]
        ).to(self.device)

        model.reinitialize()
        optimizer = cfg["optimizer_type"](model.parameters(), lr=cfg["lr"])

        return model, optimizer

    def _initialize_test_envs(self):
        test_envs = {}
        for block in self.global_config["exp_blocks"]:
            b_name = block["block_name"]

            if b_name in test_envs:
                continue

            b_cfg = self.get_block_configs(block)
            rule_func = partial(RULES_REGISTRY[b_cfg["rule"]], **b_cfg)

            X_test = self.dataset.get_block_data(b_cfg.get("zero_features", [])).to(self.device)
            y_test = rule_func(X_test).to(self.device)
            mask = get_noise_mask(b_cfg, X_test).to(self.device)

            test_envs[b_name] = {
                "X": X_test,
                "y": y_test[:, 0:1],
                "mask": mask,
                "sd": b_cfg["sd"]
            }

        return test_envs


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