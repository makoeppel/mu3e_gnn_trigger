import torch
import torch.nn as nn
import numpy as np

def get_mlp(input_dim, output_dim, num_layers=2, hidden_dim=None, dropout=0.0):
    if hidden_dim is None:
        layer_nodes = [input_dim] + [np.floor((output_dim / input_dim)**((layer_id)/ num_layers) * input_dim) for layer_id in range(1, num_layers)] + [output_dim]
    else:
        layer_nodes = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

    layers = []
    for i in range(len(layer_nodes) - 1):
        layers.append(nn.Linear(int(layer_nodes[i]), int(layer_nodes[i+1])))
        if i < len(layer_nodes) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, hidden_dim=None):
        super(Classifier, self).__init__()
        self.mlp = get_mlp(input_dim, num_classes, num_layers, hidden_dim)
        self.num_classes = num_classes

    def forward(self, x):
        feed_forward = self.mlp(x)
        softmaxed = torch.softmax(feed_forward, dim=-1)
        if self.num_classes == 2:
            return softmaxed[:, 1]
        return softmaxed