from utils import *
import json
from gui_app import launch_gui


def run_experiment(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    exps_results = []
    if len(config["exp_blocks"]) == 0:
        raise ValueError("Experiment blocks list is empty. Please provide at least one block in the configuration.")

    dataset = Dataset(config["features_types"])
    dataset.create_exp_data()

    optimizers, models = (None, None), (None, None)
    for block_config in config["exp_blocks"]:
        block_results = run_block(dataset, models, optimizers, block_config, config)
        models = (block_results["model_low"], block_results["model_high"])
        optimizers = (block_results["optimizer_low"], block_results["optimizer_high"])
        exps_results.append([block_config["block_name"], block_results])
    return exps_results


def run_block(dataset, models, optimizers, block_config, global_config):

    print(f"\n--- Running Block: {block_config['block_name']} ---")
    current_cfg = copy.deepcopy(global_config)
    current_cfg = merge_configs(block_config, current_cfg)
    X, y = dataset.get_block_data_and_labels(**current_cfg)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    print(current_cfg)

    torch_rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()

    model_low, optimizer_low, data_low = train_mlp_model(
        models[0], optimizers[0], X, y,
        w_scale=current_cfg["w_scale_low"],
        b_scale=current_cfg["b_scale_low"],
        **current_cfg
    )

    torch.set_rng_state(torch_rng_state)
    np.random.set_state(np_rng_state)

    model_high, optimizer_high, data_high = train_mlp_model(
        models[1], optimizers[1], X, y,
        w_scale=current_cfg["w_scale_high"],
        b_scale=current_cfg["b_scale_high"],
        **current_cfg
    )

    return {
        "X": X, "y": y, "config": current_cfg,
        "model_low": model_low, "optimizer_low": optimizer_low, "data_low": data_low,
        "model_high": model_high, "optimizer_high": optimizer_high, "data_high": data_high
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
    "exp_name": "test",
    "features_types": [4, 4],
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
    "sd": 0.5,
    "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": (), "epochs": 25},
                   {"block_name": "M1", "deciding_feature": 0, "zero_features": (), "epochs": 25}]
}

CONFIG_2 = {
    "exp_name": "test",
    "features_types": [4, 4],
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
    "sd": 0.5,
    "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": (), "epochs": 50}]
}

FIGURES_CONFIG = ["accuracy_graph"]

if __name__ == '__main__':
    # OPTION A: Launch the GUI
    # run_simulation_from_gui(FIGURES_CONFIG)

    # OPTION B: Bypass GUI and run from a specific JSON file
    # run_simulation_from_config_file("test", FIGURES_CONFIG)

    # OPTION C: Bypass GUI and run from a provided configuration dictionary
    run_simulation_from_dictionary(CONFIG_2, FIGURES_CONFIG)
