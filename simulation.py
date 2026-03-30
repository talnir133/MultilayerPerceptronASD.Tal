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
        for block in self.global_config["exp_blocks"]:
            block_config = self.get_block_configs(block)
            block_results = self.run_block(models, optimizers, block_config)
            models["low"], models["high"] = block_results["model_low"], block_results["model_high"]
            optimizers["low"], optimizers["high"] = block_results["optimizer_low"], block_results["optimizer_high"]
            simulation_results.append([block_config["block_name"], block_results])
        return simulation_results

    def run_block(self, models, optimizers, cfg):
        print(f"\n--- Running Block: {cfg['block_name']} ---")
        rule_name = cfg.get("rule", "upper_half")
        base_rule = RULES_REGISTRY[rule_name]
        rule_to_apply = partial(base_rule, **cfg)
        X, y = self.dataset.get_block_data(cfg["zero_features"], rule_to_apply)
        X, y = X.to(self.device), y.to(self.device)
        # print("X:", X)
        # print("y:", y)
        noise_mask = get_noise_mask(cfg, X).to(self.device)
        data_low = create_tracker(X, y)
        data_high = create_tracker(X, y)

        # ==========================================
        # LOW VARIANCE
        # ==========================================
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()

        if models["low"] is None:
            models["low"] = MLP(
                cfg["input_size"], cfg["hidden_size"], cfg["n_hidden"],
                cfg["output_size"], cfg["w_scale_low"],
                cfg["b_scale_low"], cfg["activation_type"]
            ).to(self.device)
            models["low"].reinitialize()
            optimizers["low"] = cfg["optimizer_type"](models["low"].parameters(), lr=0.004)

        model_low, optimizer_low = train_model(
            models["low"], optimizers["low"], X, y,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=get_metric_callback(data_low, cfg, X, y) if self.track_metrics else lambda m, x, c: None
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
            optimizers["high"] = cfg["optimizer_type"](models["high"].parameters(), lr=0.004)

        model_high, optimizer_high = train_model(
            models["high"], optimizers["high"], X, y,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            cfg["alpha_class"], cfg["alpha_rec"],
            metric_callback=get_metric_callback(data_low, cfg, X, y) if self.track_metrics else lambda m, x, c: None
        )

        # --- Final loss computation---
        criterion = nn.BCEWithLogitsLoss()
        for condition, mdl, data_dict in [("low", model_low, data_low), ("high", model_high, data_high)]:
            mdl.eval()
            with torch.no_grad():
                y_pred = mdl(X)
                final_class_loss = criterion(y_pred[:, 0:1], y[:, 0:1]).item()
                print(f'{condition.capitalize()} Final Classification Loss: {final_class_loss:.4f}')

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


# ==========================================
# Simulation's Helper Functions
# ==========================================
def get_noise_mask(block_config, X):
    #  creating a mask to apply noise only to non-zeroed features
    noise_mask = torch.ones_like(X)
    zero_feats = block_config.get("zero_features", [])
    features_types = block_config.get("features_types", [])

    start_idx = 0
    for i, dim in enumerate(features_types):
        if i in zero_feats:
            noise_mask[:, start_idx: start_idx + dim] = 0.0
        start_idx += dim

    return noise_mask


def create_tracker(X, y):
    """Creates the empty dictionary for metric tracking before training starts."""
    keys = [
        "losses_clean", "MAE_clean", "accuracies_clean",
        "losses_noisy", "MAE_noisy", "accuracies_noisy","losses_noisy_optimal", "MAE_noisy_optimal", "accuracies_noisy_optimal",
        "MAE_to_optimal", "accuracies_to_optimal", "noised_data",
        "activations_clean", "activation_distances_clean",
        "fc1_weight_sd", "fc1_bias_sd"
    ]
    return {k: [] for k in keys} | {"X": X, "y": y}


def get_metric_callback(tracker, cfg, X_base, y):
    """Factory method: generates the specific callback function for a model."""

    # מבודדים מראש את עמודת ה-Label של הקטלוג מתוך כלל המטריצה Y
    y_class = y[:, 0:1]

    def callback(current_model, X_noisy, loss_criterion):
        tracker["noised_data"].append(X_noisy.cpu().detach().numpy())
        tracker["fc1_weight_sd"].append(current_model._layers.fc1.weight.std().item())
        tracker["fc1_bias_sd"].append(
            current_model._layers.fc1.bias.std().item() if current_model._layers.fc1.bias is not None else 0.0)

        # Clean tracking
        clean_preds = current_model(X_base)
        clean_preds_class = clean_preds[:, 0:1]
        tracker["losses_clean"].append(loss_criterion(clean_preds_class, y_class).item())
        tracker["MAE_clean"].append(torch.abs(torch.sigmoid(clean_preds_class) - y_class).mean().item())
        tracker["accuracies_clean"].append(
            ((torch.sigmoid(clean_preds_class) > 0.5) == y_class.bool()).float().mean().item())

        acts = get_network_activations(current_model, X_base)
        epoch_act_dist = {}
        for name, act_np in acts.items():
            epoch_act_dist[name] = pairwise_distances(act_np)[np.triu_indices(X_base.shape[0], k=1)]
        tracker["activations_clean"].append(acts)
        tracker["activation_distances_clean"].append(epoch_act_dist)

        # Noisy tracking
        if cfg["sd"] > 0:
            noisy_preds = current_model(X_noisy)
            noisy_preds_class = noisy_preds[:, 0:1]
            opt_probs = get_bayes_optimal_probabilities(X_noisy, cfg["sd"], X_base, y_class)
            model_probs = torch.sigmoid(noisy_preds_class)
            tracker["losses_noisy"].append(loss_criterion(noisy_preds_class, y_class).item())
            tracker["MAE_noisy"].append(torch.abs(model_probs - y_class).mean().item())
            tracker["accuracies_noisy"].append(((model_probs > 0.5) == y_class.bool()).float().mean().item())
            tracker["MAE_noisy_optimal"].append(torch.abs(opt_probs - y_class).mean().item())
            tracker["accuracies_noisy_optimal"].append(((opt_probs > 0.5) == y_class.bool()).float().mean().item())
            tracker["MAE_to_optimal"].append(torch.abs(model_probs - opt_probs).mean().item())
            tracker["accuracies_to_optimal"].append(
                ((model_probs > 0.5) == (opt_probs > 0.5)).float().mean().item())

    return callback


def get_bayes_optimal_probabilities(noisy_X, noise_sd, clean_X, clean_y):
    """
    Generalized Bayesian optimal predictor.
    Calculates P(y=1 | noisy_X) by marginalizing over all possible clean prototypes.
    """
    # 1. Handling the zero-noise edge case
    if noise_sd <= 1e-7:
        idx = torch.cdist(noisy_X, clean_X).argmin(dim=1)
        return clean_y[idx]

    # 2. Compute squared Euclidean distances between EVERY noisy sample
    sq_dists = torch.cdist(noisy_X, clean_X) ** 2

    # 3. Calculate Likelihoods P(X_noisy | X_clean)
    likelihoods = torch.exp(-sq_dists / (2 * noise_sd ** 2))

    # 4. Bayesian Inference (Marginalization)
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
