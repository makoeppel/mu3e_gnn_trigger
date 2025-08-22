import torch
import torch.nn as nn

def get_mlp(input_dim, output_dim, num_layers=2, hidden_dim=None):
    if hidden_dim is None:
        layer_nodes = [input_dim] + [torch.floor((output_dim / input_dim)**((layer_id)/ num_layers) * input_dim) for layer_id in range(1, num_layers)] + [output_dim]
    else:
        layer_nodes = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

    layers = []
    for i in range(len(layer_nodes) - 1):
        layers.append(nn.Linear(int(layer_nodes[i]), int(layer_nodes[i+1])))
        if i < len(layer_nodes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
