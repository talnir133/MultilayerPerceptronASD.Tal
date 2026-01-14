from utils import *

# Parameters
PARAMS = {
    "num_epochs": 50,
    "features_types": [4, 4],
    "odd_dim": 0,
    "hidden_size": 30,
    "n_hidden": 1,
    "output_size": 1,
    "b_scale_low": 0,
    "b_scale_high": 0,
    "w_scale": [0.1, 50],
    "optim_type": "Adam",
    "activation_type": "Identity"
}


def run_experiment1(config, seed, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    input_size = sum(config["features_types"]) + config["odd_dim"]
    X, y, dataloader = create_dataset(config["features_types"], odd_dim=config["odd_dim"], seed=seed, device=device)
    opt = optim.Adam if config["optim_type"] == "Adam" else optim.SGD

    # Training
    (model_low_bias, data_low) = train_mlp_model(input_size, config["hidden_size"], config["n_hidden"],
                                                 config["output_size"], config["w_scale"][0], config["b_scale_low"],
                                                 X, y, dataloader,
                                                 optimizer_type=opt,
                                                 num_epochs=config["num_epochs"],
                                                 device=device,
                                                 activation_type=config["activation_type"])
    (model_high_bias, data_high) = train_mlp_model(input_size, config["hidden_size"], config["n_hidden"],
                                                   config["output_size"], config["w_scale"][1],
                                                   config["b_scale_high"],
                                                   X, y, dataloader,
                                                   optimizer_type=opt,
                                                   num_epochs=config["num_epochs"],
                                                   device=device,
                                                   activation_type=config["activation_type"])

    # Trained Model Data:
    model_low_bias.eval()
    model_high_bias.eval()

    activations = {}
    activations_wide = {}
    model_low_bias.set_activations_hook(activations)
    model_high_bias.set_activations_hook(activations_wide)

    # tests
    torch.set_printoptions(precision=2, sci_mode=False)
    # print(f"\n true y: {y}")
    # print("\n y_pred low: ", data_low["final predicted y"])
    # print("\n y_pred high: ", data_high["final predicted y"])

    # figures
    fig = Figures(config, data_low, data_high)
    fig.loss_graph()
    fig.accuracy_graph()


if __name__ == '__main__':
    run_experiment1(PARAMS, seed=0)
