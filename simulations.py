from utils import *
import json
from gui_app import launch_gui

FIGURES_CONFIG = ["accuracy_graph"]


def run_stage(stage_config, config, models=(None, None)):
    print(f"--- Running Stage: {stage_config['stage_name']} ---")
    config = copy.deepcopy(config)
    config = merge_configs(stage_config, config)
    X, y, dataloader = create_dataset(**config)

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
    exps_results = []
    if len(config["exp_stages"]) == 0:
        raise ValueError("Experiment stages list is empty. Please provide at least one stage in the configuration.")

    models = (None,None)
    for stage_config in config["exp_stages"]:
        stage_results = run_stage(stage_config, config, models)
        models = (stage_results["model_low"], stage_results["model_high"])
        exps_results.append([stage_config["stage_name"],stage_results])
    return exps_results


def create_figures(exp_results, config, figures_config, save=True):
    os.makedirs("figures", exist_ok=True)
    if save:
        folder_name = config["exp_name"]
        path = f"figures/{folder_name}"
        os.makedirs(path, exist_ok=True)
        with open(path + "/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    fig = Figures(exp_results, config, save)
    for method_name in figures_config:
        plot_method = getattr(fig, method_name)
        plot_method()


def run_simulation_from_gui(figures_config):
    def simulation_callback(config):
        # Runs experiment and creates figures using the global FIGURES_CONFIG
        results = run_experiment(config)
        create_figures(results, config, figures_config)

    launch_gui(simulation_callback)

def run_simulation_from_config_file(json_name, figures_config):
    file_path = f"configs/{json_name}.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"--- Running Simulation from configuration: {json_name} ---")
    results = run_experiment(config)
    create_figures(results, config, figures_config)
    print(f"--- Finished! Results saved in figures/{config['exp_name']} ---")

CONFIG = {
    "exp_name": "config_1",
    "features_types": [4, 4],
    "odd_dim": 8,
    "hidden_size": 30,
    "n_hidden": 1,
    "output_size": 1,
    "b_scale_low": 0.1,
    "b_scale_high": 2.0,
    "w_scale_low": 1.0,
    "w_scale_high": 1.0,
    "optimizer_type": "Adam",
    "activation_type": "Tanh",
    "batch_size": 128,
    "unique_points_only": False,
    "seed": 0,
    "exp_stages": [{"stage_name": "M1", "deciding_feature": 0, "odd": False, "epoches": 100},
                   {"stage_name": "M1-Flex", "deciding_feature": 1, "odd": False, "epoches": 100}]
}

if __name__ == '__main__':
    # OPTION A: Launch the GUI
    # run_simulation_from_gui(FIGURES_CONFIG)

    # OPTION B: Bypass GUI and run from a specific JSON file
    # run_simulation_from_config_file("Data Test", FIGURES_CONFIG)

    results = run_experiment(CONFIG)
    create_figures(results, CONFIG, FIGURES_CONFIG)
