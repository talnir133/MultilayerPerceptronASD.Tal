import os
import json
from simulation import Simulation
from analysis import SimulationAnalyzer, IDR_check
from gui_app import launch_gui

CONFIG = {
    "exp_name": "test2",
    "features_types": [4,4],
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
    "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": (), "epochs": 100}]
}


def run_simulation(config_source):
    """
    Retrieves the configuration from the specified source,
    runs the simulation sequentially, and returns the SimulationAnalyzer object.
    """
    if config_source == "gui":
        print("Launching GUI to build configuration...")
        # The code halts here until the user closes the GUI
        config = launch_gui()
        if config is None:
            print("GUI was closed without running the simulation.")
            return None

    elif isinstance(config_source, dict):
        print("Using provided configuration dictionary...")
        config = config_source

    elif isinstance(config_source, str):
        print(f"Loading specific config from {config_source}...")
        with open(os.path.join("configs", config_source+".json"), "r", encoding="utf-8") as f:
            config = json.load(f)


    else:
        raise ValueError("Invalid source. Use 'gui', 'load', a 'filename.json', or a config dictionary.")

    # ==========================================
    # Run the Simulation & Create Analyzer
    # ==========================================
    print(f"\n--- Running Simulation: {config.get('exp_name', 'Unnamed')} ---")
    results = Simulation(config).run()

    analyzer = SimulationAnalyzer(results, config, save_figures=True)
    return analyzer


if __name__ == '__main__':
    # Simulation Running
    # s = run_simulation("gui")
    # s = run_simulation("test")
    # s = run_simulation(CONFIG)
    # s.plot_mae()

    DRs = IDR_check(sd=0,
                    activation_type="Identity",
                    w_scale_low=0.1,
                    w_scale_high=50,
                    b_scale_low=0,
                    b_scale_high=0,
                    epochs=25)
    # DRs.plot_sigmoids(seed=0)
    DRs.plot_histograms(10)
