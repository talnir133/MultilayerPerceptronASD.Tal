import torch
import torch.nn as nn


# Define the MLP model
class MLP(nn.Module):
    """
    A simple MLP model with ReLU activation and Sigmoid output, that can be reinitialized and set to specific bias scale
    """

    def __init__(self, input_size, hidden_size, n_hidden, output_size, w_scale, b_scale, activation_type=None):
        super(MLP, self).__init__()
        self._layers = nn.Sequential()

        def act_func():
            if activation_type == 'Identity':
                return nn.Identity()
            if activation_type == 'RelU':
                return nn.ReLU()
            if activation_type == 'Sigmoid':
                return nn.Sigmoid()
            return nn.Tanh()

        self._layers.add_module('fc1', nn.Linear(input_size, hidden_size, bias=bool(b_scale)))
        self._layers.add_module('activation_func1', act_func())
        self.w_scale = w_scale
        self.b_scale = b_scale
        for i in range(n_hidden):
            self._layers.add_module(f'fc{i + 2}', nn.Linear(hidden_size, hidden_size, bias=bool(b_scale)))
            self._layers.add_module(f'activation_func{i + 2}', act_func())
        self._layers.add_module('fc_last', nn.Linear(hidden_size, output_size, bias=bool(b_scale)))
        # self._layers.add_module('sigmoid', nn.Sigmoid())
        self._handles = []

    def set_activations_hook(self, activations):
        def hook_generator(name, activations):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()

            return hook

        self._handles = []
        for name, m in self._layers.named_modules():
            self._handles.append(m.register_forward_hook(hook_generator(name, activations)))

    def remove_activations_hook(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def get_out_activation(self):
        return self._layers[-1]

    def forward(self, x):
        return self._layers(x)

    def reinitialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if bool(self.b_scale):
                    nn.init.normal_(m.bias, 0, self.b_scale)
                if 'fc1' in name:
                    nn.init.xavier_normal_(m.weight, gain=self.w_scale)
                else:
                    nn.init.xavier_normal_(m.weight)
