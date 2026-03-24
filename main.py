from simulation import Simulation

CONFIG = {
    "exp_name": "test1",
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
    "sd": 0,
    "exp_blocks": [{"block_name": "M1", "deciding_feature": 0, "zero_features": (1,), "epochs": 100}]
}

results = Simulation(CONFIG)