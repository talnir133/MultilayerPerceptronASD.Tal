from utils import *

# Parameters
PARAMS = {
    "num_epochs": 50,
    "features_types": [4, 4],
    "odd_dim": 8,
    "hidden_size": 30,
    "n_hidden": 1,
    "output_size": 1,
    "b_scale_low": 0,
    "b_scale_high": 0,
    "w_scale_low": 0.1,
    "w_scale_high": 50,
    "optimizer_type": "Adam",
    "activation_type": "Identity",
    "batch_size": 16,
}


def run_experiment(config, models=(0, 0), odd=False, deciding_feature=0, unique_points_only = False, seed=0, device=None):
    config = added_config(config, seed, device, odd, deciding_feature, unique_points_only)
    X, y, dataloader = create_dataset(**config)

    # Training
    model_low, data_low = train_mlp_model(models[0], X, y, dataloader, w_scale=config["w_scale_low"],
                                          b_scale=config["b_scale_low"],
                                          **config)
    model_high, data_high = train_mlp_model(models[1], X, y, dataloader, w_scale=config["w_scale_high"],
                                            b_scale=config["b_scale_high"],
                                            **config)

    return {"X": X, "y": y, "config": config, "model_low": model_low, "data_low": data_low, "model_high": model_high,
            "data_high": data_high}


if __name__ == '__main__':
    exp1 = run_experiment(PARAMS, odd=False, deciding_feature=0)
    exp1_trained_models = (exp1["model_low"], exp1["model_high"])
    exp2 = run_experiment(PARAMS, exp1_trained_models, odd=False, deciding_feature=1)
    exp3 = run_experiment(PARAMS, exp1_trained_models, odd=True, deciding_feature=0)

    # figures
    fig = Figures(exp1, exp2, exp3)
    fig.loss_graph(("initial", "odd"))
    fig.accuracy_graph(("initial", "flex"))
    fig.accuracy_graph(("initial", "odd"))
