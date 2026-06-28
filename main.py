from simulation import Simulation, averaged_simulation
from analysis import SimulationAnalyzer
from dynamic_ranges import Simple_DR_Simulation

CONFIG = {
    "exp_name": "Simulation_1",
    "features_types": [2],
    "hidden_size": 2, "n_hidden": 1,
    "b_scale_low": 0.1, "b_scale_high": 2,
    "w_scale_low": 0.1, "w_scale_high": 0.1,
    "optimizer_type": "Adam", "activation_type": "Tanh",
    "batch_size": 1, "seed": 0, "lr": 0.04,
    "decoder": {"train_sd": 0.1, "test_sd": 0.3, "samples_per_point": 20, "freq": 2, "epochs": 200, "lr": 0.1},
    "exp_blocks": [
            {"block_name": "B1", "rule": "upper_half", "zero_features": (), "epochs": 10, "sd": 0, "alpha_class": 1.0, "alpha_rec": 0.0, "deciding_feature": 0}]
}



if __name__ == '__main__':
    # Simulation Running
    # s = Simulation(CONFIG).run()
    s = averaged_simulation(CONFIG,10)
    s = SimulationAnalyzer(s, CONFIG)
    s.plot_metric_over_epochs("MAE_noisy")
    s.plot_PCs((0,2,9))
    s.plot_parameter_distributions()
    s.plot_dr_tracker()