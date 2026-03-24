import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from core_ml import Dataset, MLP, train_model


class Simulation:
    def __init__(self, config):
        self.global_config = copy.deepcopy(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = Dataset(config["features_types"])
        self.dataset.create_exp_data()

        # Lock random seed for reproducibility
        torch.manual_seed(self.global_config["seed"])
        np.random.seed(self.global_config["seed"])

    def run(self):

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

        if models["low"] is None:
            for condition in ["low", "high"]:
                models[condition] = MLP(
                    cfg["input_size"], cfg["hidden_size"], cfg["n_hidden"],
                    cfg["output_size"], cfg[f"w_scale_{condition}"],
                    cfg[f"b_scale_{condition}"], cfg["activation_type"]
                ).to(self.device)
                optimizers[condition] = cfg["optimizer_type"](models[condition].parameters(), lr=0.004)

        X, y = self.dataset.get_block_data(cfg.get("zero_features", []), cfg["deciding_feature"])
        X, y = X.to(self.device), y.to(self.device)
        noise_mask = self.get_noise_mask(cfg, X).to(self.device)
        data_low = self.create_tracker(X, y)
        data_high = self.create_tracker(X, y)
        torch_rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()


        model_low, optimizer_low = train_model(
            models["low"], optimizers["low"], X, y,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            metric_callback=self.get_metric_callback(data_low, cfg, X, y)
        )

        torch.set_rng_state(torch_rng_state)
        np.random.set_state(np_rng_state)

        model_high, optimizer_high = train_model(
            models["high"], optimizers["high"], X, y,
            cfg["epochs"], cfg["batch_size"], cfg["sd"], noise_mask,
            metric_callback=self.get_metric_callback(data_high, cfg, X, y)
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
        opt_map = {"Adam": optim.Adam, "SGD": optim.SGD}
        block_config["optimizer_type"] = opt_map[block_config["optimizer_type"]]

        act_map = {
            "Tanh": nn.Tanh, "RelU": nn.ReLU,
            "Sigmoid": nn.Sigmoid, "Identity": nn.Identity

        }
        act_name = block_config.get("activation_type", "Identity")
        block_config["activation_type"] = act_map.get(act_name, nn.Identity)()

        return block_config

    def get_noise_mask(self, block_config, X):
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

    def create_tracker(self, X, y):
        """Creates the empty dictionary for metric tracking before training starts."""
        keys = [
            "losses_clean", "MAE_clean", "accuracies_clean",
            "MAE_noisy", "accuracies_noisy", "MAE_noisy_optimal", "accuracies_noisy_optimal",
            "MAE_to_optimal", "accuracies_to_optimal", "noised_data",
            "activations_clean", "activation_distances_clean",
            "fc1_weight_sd", "fc1_bias_sd"
        ]
        return {k: [] for k in keys} | {"X": X, "y": y}

    def get_metric_callback(self, tracker, cfg, X_base, y):
        """Factory method: generates the specific callback function for a model."""

        def callback(current_model, X_noisy, criterion):
            tracker["noised_data"].append(X_noisy.cpu().detach().numpy())
            tracker["fc1_weight_sd"].append(current_model._layers.fc1.weight.std().item())
            tracker["fc1_bias_sd"].append(
                current_model._layers.fc1.bias.std().item() if current_model._layers.fc1.bias is not None else 0.0)

            clean_preds = current_model(X_base)
            tracker["losses_clean"].append(criterion(clean_preds, y).item())
            tracker["MAE_clean"].append(torch.abs(torch.sigmoid(clean_preds) - y).mean().item())
            tracker["accuracies_clean"].append(((torch.sigmoid(clean_preds) > 0.5) == y.bool()).float().mean().item())

            acts = self.get_network_activations(current_model, X_base)
            epoch_act_dist = {}
            for name, act_np in acts.items():
                epoch_act_dist[name] = pairwise_distances(act_np)[np.triu_indices(X_base.shape[0], k=1)]

            tracker["activations_clean"].append(acts)
            tracker["activation_distances_clean"].append(epoch_act_dist)

            if cfg["sd"] > 0:
                opt_probs = self.get_bayes_optimal_probabilities(X_noisy, cfg["sd"], cfg["features_types"],
                                                                 cfg["deciding_feature"])
                model_probs = torch.sigmoid(current_model(X_noisy))
                tracker["MAE_noisy"].append(torch.abs(model_probs - y).mean().item())
                tracker["accuracies_noisy"].append(((model_probs > 0.5) == y.bool()).float().mean().item())
                tracker["MAE_noisy_optimal"].append(torch.abs(opt_probs - y).mean().item())
                tracker["accuracies_noisy_optimal"].append(((opt_probs > 0.5) == y.bool()).float().mean().item())
                tracker["MAE_to_optimal"].append(torch.abs(model_probs - opt_probs).mean().item())
                tracker["accuracies_to_optimal"].append(
                    ((model_probs > 0.5) == (opt_probs > 0.5)).float().mean().item())

        return callback

    def get_bayes_optimal_probabilities(self, noisy_X, noise_sd, features_types, deciding_feature):
        """Calculates theoretical optimal probabilities using Bayesian inference."""
        start_idx = sum(features_types[:deciding_feature])
        n_types = features_types[deciding_feature]
        end_idx = start_idx + n_types
        noisy_deciding_feature = noisy_X[:, start_idx:end_idx]

        if noise_sd == 0.0:
            best_match = torch.argmax(noisy_deciding_feature, dim=1)
            threshold = n_types // 2
            return (best_match < threshold).float().unsqueeze(1)

        templates = torch.eye(n_types, device=noisy_X.device)
        dists = torch.cdist(noisy_deciding_feature, templates) ** 2
        likelihoods = torch.exp(-dists / (2 * (noise_sd ** 2)))

        threshold = n_types // 2
        L_class1 = torch.sum(likelihoods[:, :threshold], dim=1)
        L_class0 = torch.sum(likelihoods[:, threshold:], dim=1)
        optimal_probs = L_class1 / (L_class1 + L_class0)

        return optimal_probs.unsqueeze(1)

    def get_network_activations(self, model, x):
        """Manually passes data through the network layer by layer to collect activations."""
        activations = {}
        out = x
        for name, layer in model._layers.named_children():
            out = layer(out)
            activations[name] = out.detach().cpu().numpy()
        return activations
