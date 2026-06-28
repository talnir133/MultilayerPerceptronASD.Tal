import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
from core_ml import LogisticDecoder


# ==========================================
# Tracking & Metrics Collection
# ==========================================

def create_tracker(test_envs, X_global):
    tracker = {
        "weights": {}, "biases": {},
        "X_global": X_global.cpu().numpy(),
        "test_envs": {env: {"X": d["X"].cpu().numpy(), "y": d["y"].cpu().numpy()} for env, d in test_envs.items()},
        "activations": [], "activation_distances": [],
        "PR_weights": [], "bias_weight_correlation": []
    }

    env_specific_metrics = [
        "losses_clean", "MAE_clean", "accuracy_clean",
        "losses_noisy", "MAE_noisy", "accuracy_noisy",
        "losses_noisy_optimal", "MAE_noisy_optimal", "accuracy_noisy_optimal",
        "PR_activations", "decoder_evaluation"
    ]

    for metric in env_specific_metrics:
        tracker[metric] = {env: [] for env in test_envs.keys()}

    return tracker


def get_metric_callback(tracker, test_envs, X_global, decoder_cfg, features_types, target_layer="fc1"):
    epoch_counter = {"count": 0}

    def callback(current_model, loss_criterion):

        # ==========================================
        # A. GLOBAL METRICS
        # ==========================================

        # A1. Raw Parameters
        for name, param in current_model.named_parameters():
            if 'weight' in name:
                tracker["weights"].setdefault(name, []).append(param.detach().cpu().numpy().copy())
            elif 'bias' in name and param is not None:
                tracker["biases"].setdefault(name, []).append(param.detach().cpu().numpy().copy())

        # A2. Activations & Distances
        acts = get_network_activations(current_model, X_global)
        tracker["activations"].append(acts)

        distances = {name: pairwise_distances(act)[np.triu_indices(X_global.shape[0], k=1)]
                     for name, act in acts.items()}
        tracker["activation_distances"].append(distances)

        # A3. Layer Statistics
        layer = dict(current_model._layers.named_children()).get(target_layer)
        if layer:
            w_np = layer.weight.detach().cpu().numpy()
            b_np = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(w_np.shape[0])

            num_w, den_w = np.sum(w_np ** 2, axis=1) ** 2, np.sum(w_np ** 4, axis=1)
            pr_w = np.mean(np.divide(num_w, den_w, out=np.zeros_like(num_w), where=den_w != 0))
            tracker["PR_weights"].append(pr_w / w_np.shape[1])

            w_norm, abs_b = np.linalg.norm(w_np, axis=1), np.abs(b_np.flatten())
            corr = 0.0 if np.std(w_norm) == 0 or np.std(abs_b) == 0 else np.corrcoef(w_norm, abs_b)[0, 1]
            tracker["bias_weight_correlation"].append(corr)

        # ==========================================
        # B. ENVIRONMENT SPECIFIC METRICS
        # ==========================================

        for env_name, env_data in test_envs.items():
            X_env, y_env, env_sd = env_data["X"], env_data["y"], env_data["sd"]

            # B1. Activation PR
            env_acts = get_network_activations(current_model, X_env)[target_layer]
            num_a, den_a = np.sum(env_acts ** 2, axis=1) ** 2, np.sum(env_acts ** 4, axis=1)
            pr_a = np.mean(np.divide(num_a, den_a, out=np.zeros_like(num_a), where=den_a != 0))
            tracker["PR_activations"][env_name].append(pr_a / env_acts.shape[1])

            # B2. Clean Predictions
            clean_preds = current_model(X_env)[:, 0:1]
            clean_probs = torch.sigmoid(clean_preds)
            tracker["losses_clean"][env_name].append(loss_criterion(clean_preds, y_env).item())
            tracker["MAE_clean"][env_name].append(torch.abs(clean_probs - y_env).mean().item())
            tracker["accuracy_clean"][env_name].append(((clean_probs > 0.5) == y_env.bool()).float().mean().item())

            # B3. Noisy Predictions
            if env_sd > 0:
                X_noisy = X_env + (torch.randn_like(X_env) * env_sd * env_data["mask"])
                noisy_preds = current_model(X_noisy)[:, 0:1]
                model_probs = torch.sigmoid(noisy_preds)
                opt_probs = get_bayes_optimal_probabilities(X_noisy, env_sd, X_env, y_env)

                tracker["losses_noisy"][env_name].append(loss_criterion(noisy_preds, y_env).item())
                tracker["MAE_noisy"][env_name].append(torch.abs(model_probs - y_env).mean().item())
                tracker["accuracy_noisy"][env_name].append(((model_probs > 0.5) == y_env.bool()).float().mean().item())
                tracker["MAE_noisy_optimal"][env_name].append(torch.abs(opt_probs - y_env).mean().item())
                tracker["accuracy_noisy_optimal"][env_name].append(
                    ((opt_probs > 0.5) == y_env.bool()).float().mean().item())

            # B4. Logistic Regression Probing (Decoder Readout)
            if decoder_cfg:
                if epoch_counter["count"] % decoder_cfg["freq"] == 0:
                    decoder = LogisticDecoder(
                        input_size=env_acts.shape[1],
                        features_types=features_types,
                    ).to(X_env.device)

                    decoder.fit(
                        classification_model=current_model,
                        X_clean=X_env,
                        train_noise_sd=decoder_cfg["train_sd"],
                        noise_mask=env_data["mask"],
                        target_layer=target_layer,
                        epochs=decoder_cfg["epochs"],
                        lr=decoder_cfg["lr"]
                    )

                    eval_results = decoder.evaluate(
                        classification_model=current_model,
                        X_clean=X_env,
                        noise_mask=env_data["mask"],
                        target_layer=target_layer,
                        samples_per_point=decoder_cfg["samples_per_point"],
                        test_noise_sd=decoder_cfg["test_sd"]
                    )

                    tracker["decoder_evaluation"][env_name].append(eval_results)

            epoch_counter["count"] += 1

    return callback


# ==========================================
# Tracking Helper Functions
# ==========================================


def get_bayes_optimal_probabilities(noisy_X, noise_sd, clean_X, clean_y):
    if noise_sd <= 1e-7:
        return clean_y[torch.cdist(noisy_X, clean_X).argmin(dim=1)]

    sq_dists = torch.cdist(noisy_X, clean_X) ** 2
    likelihoods = torch.exp(-sq_dists / (2 * noise_sd ** 2))
    return (likelihoods @ clean_y) / likelihoods.sum(dim=1, keepdim=True)


def get_network_activations(model, x):
    activations = {}
    out = x
    for name, layer in model._layers.named_children():
        out = layer(out)
        activations[name] = out.detach().cpu().numpy()
    return activations


def get_noise_mask(block_config, X):
    noise_mask = torch.ones_like(X)
    start_idx = 0
    for i, dim in enumerate(block_config.get("features_types", [])):
        if i in block_config.get("zero_features", []):
            noise_mask[:, start_idx: start_idx + dim] = 0.0
        start_idx += dim
    return noise_mask