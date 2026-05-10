import json
from simulation import Simulation, averaged_simulation
from analysis import SimulationAnalyzer
from gui_app import launch_gui
from dynamic_ranges import IDR_check

CONFIG = {
    "exp_name": "Summerfield_Replication",
    "features_types": [2, 2,2],
    "hidden_size": 30, "n_hidden": 1,
    "b_scale_low": 0.1, "b_scale_high": 1,
    "w_scale_low": 0.1, "w_scale_high": 50.0,
    "optimizer_type": "Adam", "activation_type": "Identity",
    "batch_size": 1, "seed": 0, "lr": 0.004,
    "exp_blocks": [
            {"block_name": "B1", "rule": "upper_half", "zero_features": (), "epochs": 3, "sd": 0.0, "alpha_class": 1.0, "alpha_rec": 0.0, "deciding_feature": 0}]
}

#

def run_simulation(config_source):
    match config_source:
        case "gui": config = launch_gui()
        case dict(): config = config_source
        case str():
            with open(f"configs/{config_source}.json", "r", encoding="utf-8") as f: config = json.load(f)
        case _: raise ValueError("Invalid source. Use 'gui', a dict, or a filename.")

    return SimulationAnalyzer(Simulation(config).run(), config) if config else None



if __name__ == '__main__':
    # Simulation Running
    s = run_simulation(CONFIG)
    # s = run_simulation("gui")
    # s = run_simulation("test")
    # s = averaged_simulation(CONFIG,5)
    s = SimulationAnalyzer(s, CONFIG)
    # s.plot_mae()
    # s.plot_mae(sub_type="noisy")
    # s.plot_mds((0,2,18), mode='animation')
    # s.plot_loss(sub_type="noisy")
    # s.plot_accuracy()
    # s.plot_parameter_distributions()
    # s.plot_mds(tuple(range(20)), mode='animation')
    #
    # drs = IDR_check(sd=0.5,
    #                 activation_type="Tanh",
    #                 w_scale_low=0.1,
    #                 w_scale_high=0.1,
    #                 b_scale_low=0.1,
    #                 b_scale_high=2,
    #                 epochs=50)

    # drs.plot_sigmoids(seed=0)
    # drs.plot_histograms(20)
    # drs.plot_sigmoid_SDs(samples_per_b_w = 50, b_range=(0.1, 2), w_range=(0.1, 20), dots_density = 10)