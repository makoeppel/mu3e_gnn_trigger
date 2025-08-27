import numpy as np
import matplotlib.pyplot as plt

def plot_graph(nodes, edges, ax, color = 'tab:blue', nodes_class = None):
    ax.scatter(nodes[:, 0], nodes[:, 1], c=color)
    for edge in edges.T:
        ax.plot(
            [nodes[edge[0], 0], nodes[edge[1], 0]],
            [nodes[edge[0], 1], nodes[edge[1], 1]],
            color=color,
            linewidth=0.5,
        )