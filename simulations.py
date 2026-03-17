from utils import *
import json
from gui_app import launch_gui

FIGURES_CONFIG = ["accuracy_graph", "loss_graph"]


def run_stage(stage, config, models=(0, 0)):
    config = copy.deepcopy(config)
    config = added_config(stage, config)
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
    fig = Figures(results, config, save)
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
    print(f"--- Finished! Results saved in figures/{config['config_name']} ---")


if __name__ == '__main__':
    # OPTION A: Launch the GUI
    run_simulation_from_gui(FIGURES_CONFIG)

    # OPTION B: Bypass GUI and run from a specific JSON file
    # run_from_config_file("Test - Initial (seed 2)", FIGURES_CONFIG)
