from utils import *

# Parameters
NUM_EPOCHS = 10000
FEATURES_TYPES = [4, 4]
ODD_DIM = 0
HIDDEN_SIZE = 30
N_HIDDEN = 1
OUTPUT_SIZE = 1
B_SCALE = .1
B_SCALE_HIGH = 5.
W_SCALE = np.sqrt(5. * (2 / HIDDEN_SIZE))
OPTIM_TYPE = "SGD"
ACTIVATION_TYPE = 'Identity'


def run_experiment1(
        features_types, odd_dim, n_hidden, hidden_size, output_size,
        w_scale, b_scale, b_scale_high, num_epochs, opt_type, activation_type, seed, device=None,):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    input_size = sum(features_types)+odd_dim
    X, y, dataloader = create_dataset(features_types, odd_dim, seed=seed, device=device)
    opt = optim.Adam if opt_type == "Adam" else optim.SGD

    # Training
    (model_low_bias, analysis_data_low) = train_mlp_model(input_size, hidden_size, n_hidden,
                                                                                     output_size, w_scale, b_scale,
                                                                                     X, y, dataloader,
                                                                                     optimizer_type=opt,
                                                                                     num_epochs=num_epochs,
                                                                                     device=device,
                                                                                     activation_type=activation_type)
    (model_high_bias, analysis_data_high) = train_mlp_model(input_size, hidden_size, n_hidden,
                                                                                      output_size, w_scale,
                                                                                      b_scale_high,
                                                                                      X, y, dataloader,
                                                                                      optimizer_type=opt,
                                                                                      num_epochs=num_epochs,
                                                                                      device=device,
                                                                                      activation_type=activation_type)

    # Trained Model Data:
    model_low_bias.eval()
    model_high_bias.eval()

    activations = {}
    activations_wide = {}
    model_low_bias.set_activations_hook(activations)
    model_high_bias.set_activations_hook(activations_wide)

    # analysis:
    input_dist_low, layer_distances_low, loss_history_low = analysis_data_low
    input_dist_high, layer_distances_high, loss_history_high = analysis_data_high
    fig = Figures()
    fig.loss_graph(loss_history_low, loss_history_high)


if __name__ == '__main__':
    run_experiment1(FEATURES_TYPES, ODD_DIM,  N_HIDDEN, HIDDEN_SIZE, OUTPUT_SIZE,
    W_SCALE, B_SCALE, B_SCALE_HIGH, NUM_EPOCHS, OPTIM_TYPE, ACTIVATION_TYPE, seed=0)

