import copy

from utils import *
import json

# Parameters
CONFIG1 = {
    "config_name": "config_1",
    "num_epochs": 1000,
    "features_types": [4, 4],
    "odd_dim": 8,
    "hidden_size": 30,
    "n_hidden": 1,
    "output_size": 1,
    "b_scale_low": 0.1,
    "b_scale_high": 2,
    "w_scale_low": 1,
    "w_scale_high": 1,
    "optimizer_type": "Adam",
    "activation_type": "Tanh",
    "batch_size": 128,
    "unique_points_only": False,
    "seed": 0,
    "exp_stages": ["initial", "flexibility"]
}

CONFIG2 = copy.deepcopy(CONFIG1)
CONFIG2["config_name"] = "config_2"
CONFIG2["exp_stages"] = ["initial", "generalization"]

FIGURES_CONFIG = ["accuracy_graph"]


def run_stage(stage, config, models=(0, 0)):
    config = copy.deepcopy(config)
    config = added_config(stage, config)
    print(config)

    X, y, dataloader = create_dataset(**config)
    print(X)
    print(y)

    # Training
    model_low, data_low = train_mlp_model(models[0], X, y, dataloader, w_scale=config["w_scale_low"],
                                          b_scale=config["b_scale_low"],
                                          **config)
    model_high, data_high = train_mlp_model(models[1], X, y, dataloader, w_scale=config["w_scale_high"],
                                            b_scale=config["b_scale_high"],
                                            **config)

    return {"X": X, "y": y, "config": config, "model_low": model_low, "data_low": data_low,
            "model_high": model_high,
            "data_high": data_high}


def run_experiment(config):
    exps_results = {"initial": None, "flexibility": None, "generalization": None}
    exp1 = run_stage("initial", config)
    exp1_trained_models = (exp1["model_low"], exp1["model_high"])
    exps_results["initial"] = exp1
    if len(config["exp_stages"]) > 1:
        second_stage = config["exp_stages"][1]
        exp2 = run_stage(second_stage, exp1["config"], models=exp1_trained_models)
        exps_results[second_stage] = exp2
    return exps_results


def create_figures(results, config, figures_config, save=True):
    os.makedirs("figures", exist_ok=True)
    if save:
        folder_name = config["config_name"]
        path = f"figures/{folder_name}"
        os.makedirs(path, exist_ok=True)
        with open(path + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    fig = Figures(results, save)
    for method_name in figures_config:
        plot_method = getattr(fig, method_name)
        plot_method()


if __name__ == '__main__':
    # RESULTS = run_experiment(CONFIG1)
    # create_figures(RESULTS, CONFIG1, FIGURES_CONFIG)

    RESULTS = run_experiment(CONFIG2)
    create_figures(RESULTS, CONFIG2, FIGURES_CONFIG)
