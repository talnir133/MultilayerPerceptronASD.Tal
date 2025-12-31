from utils import *

# Parameters
NUM_EPOCHS = 150
INPUT_SIZE = 2
HIDDEN_SIZE = 30
N_HIDDEN = 1
OUTPUT_SIZE = 1
NUM_SAMPLES = 500
B_SCALE = .1
B_SCALE_HIGH = 10.
W_SCALE = np.sqrt(5. * (2 / HIDDEN_SIZE))
SCALE = 1
LOC = 2
OPTIM_TYPE = "SGD"
ACTIVATION_TYPE = 'Sigmoid'


def run_experiment(
    input_size, num_samples, loc, scale, n_hidden, hidden_size, output_size,
    w_scale, b_scale, b_scale_high, num_epochs, opt_type, activation_type, seed, dataset_id, run_id, device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    os.makedirs("figures", exist_ok=True)
    X_train, X_test, y_train, y_test, dg, grid, dataloader = create_gaussian_dataset(
        input_size, num_samples, loc, scale, 2, seed=seed, device=device)
    opt = optim.Adam if opt_type == "Adam" else optim.SGD

    # Training
    (model_low_bias, resps_low_bias, inp_dist, pretrain_distances) = train_mlp_model(
        input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, X_train,
        y_train, X_test, y_test, dataloader, dg, grid, opt, num_epochs, device=device, activation_type=None)
    (model_high_bias, resps_high_bias, _, pretrain_distances_wide) = train_mlp_model(
        input_size, hidden_size, n_hidden, output_size, w_scale, b_scale_high,
        X_train, y_train, X_test, y_test, dataloader, dg, grid, opt, num_epochs, device=device, activation_type=None)

    # Trained Model Data:
    model_low_bias.eval()
    model_high_bias.eval()

    activations = {}
    activations_wide = {}
    model_low_bias.set_activations_hook(activations)
    model_high_bias.set_activations_hook(activations_wide)


if __name__ == '__main__':
    for dataset_id in range(1,2):
        print("Dataset", dataset_id)
        for run_id in range(1, 2):
            print("Run", run_id)
            run_experiment(
                INPUT_SIZE, NUM_SAMPLES, LOC, SCALE, N_HIDDEN, HIDDEN_SIZE, OUTPUT_SIZE,
                W_SCALE, B_SCALE, B_SCALE_HIGH, NUM_EPOCHS, OPTIM_TYPE, ACTIVATION_TYPE, seed=dataset_id*10+run_id,
                dataset_id=dataset_id, run_id=run_id)