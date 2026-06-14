import json
from simulation import Simulation, averaged_simulation
from analysis import SimulationAnalyzer
from dynamic_ranges import IDR_check

CONFIG = {
    "exp_name": "Simulation_1",
    "features_types": [2,2],
    "hidden_size": 4, "n_hidden": 1,
    "b_scale_low": 0.1, "b_scale_high": 2,
    "w_scale_low": 1, "w_scale_high": 1,
    "optimizer_type": "Adam", "activation_type": "Tanh",
    "batch_size": 1, "seed": 0, "lr": 0.04,
    "decoder": {"sd": 0.5, "samples_per_point": 20, "freq": 5, "epochs": 100, "lr": 0.1},
    "exp_blocks": [
            {"block_name": "B1", "rule": "upper_half", "zero_features": (), "epochs": 100, "sd": 1, "alpha_class": 1.0, "alpha_rec": 0.0, "deciding_feature": 0}]
}



if __name__ == '__main__':
    # Simulation Running
    s = Simulation(CONFIG).run()
    # s = averaged_simulation(CONFIG,1)
    s = SimulationAnalyzer(s, CONFIG)
    s.plot_metric_over_epochs("MAE_noisy")
    s.plot_dr_tracker()
    # s.plot_mae(sub_type="noisy")
    # s.plot_mae(sub_type="clean")
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
    #                 epochs=50,
    #                 hidden_size=1000)
    #
    # # drs.plot_sigmoids(seed=0)
    # # drs.plot_histograms(20)
    # drs.plot_sigmoid_SDs(samples_per_b_w = 100, b_range=(0.1, 2), w_range=(0.1, 20), dots_density = 3, epochs_per_simulation=50)