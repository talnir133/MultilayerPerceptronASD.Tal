from utils import *
import json
from gui_app import launch_gui

FIGURES_CONFIG = ["accuracy_graph"]


def run_experiment(config):
    exps_results = []
    if len(config["exp_stages"]) == 0:
        raise ValueError("Experiment stages list is empty. Please provide at least one stage in the configuration.")

    dataset = Dataset(config["features_types"], config["sd"], config["seed"])
    dataset.create_exp_data()

    models = (None, None)
    for stage_config in config["exp_stages"]:
        stage_results = run_stage(dataset, models, stage_config, config)
        models = (stage_results["model_low"], stage_results["model_high"])
        exps_results.append([stage_config["stage_name"], stage_results])
    return exps_results


def run_stage(dataset, models, stage_config, global_config):
    print(f"\n--- Running Stage: {stage_config['stage_name']} ---")
    current_cfg = copy.deepcopy(global_config)
    current_cfg = merge_configs(stage_config, current_cfg)
    X, y = dataset.get_stage_data_and_labels(**current_cfg)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    model_low, data_low = train_mlp_model(
        models[0], X, y,
        w_scale=current_cfg["w_scale_low"],
        b_scale=current_cfg["b_scale_low"],
        **current_cfg
    )

    model_high, data_high = train_mlp_model(
        models[1], X, y,
        w_scale=current_cfg["w_scale_high"],
        b_scale=current_cfg["b_scale_high"],
        **current_cfg
    )

    return {
        "X": X, "y": y, "config": current_cfg,
        "model_low": model_low, "data_low": data_low,
        "model_high": model_high, "data_high": data_high
    }


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
        results = run_experiment(config)
        create_figures(results, config, figures_config)

    launch_gui(simulation_callback)


def run_simulation_from_config_file(json_name, figures_config):
    file_path = f"configs/{json_name}.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"--- Running Simulation from configuration: {json_name} ---\n ")
    results = run_experiment(config)
    create_figures(results, config, figures_config)
    print(f"--- Finished! Results saved in figures/{config['exp_name']} ---")


def run_simulation_from_dictionary(config, figures_config):
    print(f"--- Running Simulation from provided configuration dictionary ---")
    results = run_experiment(config)
    create_figures(results, config, figures_config)
    print(f"--- Finished! Results saved in figures/{config['exp_name']} ---")


CONFIG = {
    "exp_name": "config_1",
    "features_types": [4, 4, 8],
    "hidden_size": 30,
    "n_hidden": 0,
    "output_size": 1,
    "b_scale_low": 0,
    "b_scale_high": 0,
    "w_scale_low": 0.1,
    "w_scale_high": 50,
    "optimizer_type": "Adam",
    "activation_type": "Identity",
    "batch_size": 1,
    "seed": 0,
    "sd": 0,
    "exp_stages": [{"stage_name": "M1", "deciding_feature": 0, "zero_features": (2,), "epoches": 25},
                   {"stage_name": "M1-Flex", "deciding_feature": 1, "zero_features": (2,), "epoches": 25}]
}

if __name__ == '__main__':
    # OPTION A: Launch the GUI
    run_simulation_from_gui(FIGURES_CONFIG)

    # OPTION B: Bypass GUI and run from a specific JSON file
    # run_simulation_from_config_file("test_config_gui", FIGURES_CONFIG)

    # OPTION C: Bypass GUI and run from a provided configuration dictionary
    # run_simulation_from_dictionary(CONFIG, FIGURES_CONFIG)
