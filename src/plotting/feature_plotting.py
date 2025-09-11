import numpy as np
import matplotlib.pyplot as plt


def plot_detector(ax):
    layer_1_circle = plt.Circle((0, 0), 23, color="red", fill=False, linestyle="dashed", alpha = 0.5)
    layer_2_circle = plt.Circle((0, 0), 29.5, color="red", fill=False, linestyle="dashed", alpha = 0.5)
    scifi_layer_circle = plt.Circle((0, 0), 61, color="blue", fill=False, linewidth=10, alpha = 0.5)
    layer_3_circle = plt.Circle((0, 0), 72, color="red", fill=False, linestyle="dashed", alpha = 0.5)
    layer_4_circle = plt.Circle((0, 0), 86, color="red", fill=False, linestyle="dashed", alpha = 0.5)


    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.add_patch(layer_1_circle)
    ax.add_patch(layer_2_circle)
    ax.add_patch(scifi_layer_circle)
    ax.add_patch(layer_3_circle)
    ax.add_patch(layer_4_circle)
    ax.scatter(0, 0, c="black", s=50, label="Target position")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)



def plot_layer_hetero_graph_to_axis(ax, graph, no_edges = False):
    mppc_nodes = graph["mppc"].x
    pixel_nodes = np.concatenate(
        [
            graph["layer_1"].x,
            graph["layer_2"].x,
            graph["layer_3"].x,
            graph["layer_4"].x,
        ],
        axis=0,
    )
    ax.scatter(mppc_nodes[:, 0], mppc_nodes[:, 1], c="blue", s=20, label="MPPC hits")
    ax.scatter(pixel_nodes[:, 0], pixel_nodes[:, 1], c="red", s=20, label="Pixel hits")
    if no_edges:
        return
    layer_1_to_layer_2_edges = graph["layer_1", "to", "layer_2"]
    for edge_idx in range(layer_1_to_layer_2_edges.edge_index.shape[1]):
        if layer_1_to_layer_2_edges.edge_labels[edge_idx] != 1:
            color = "orange"
        else:
            color = "green"

        edge = layer_1_to_layer_2_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["layer_1"].x[edge[0], 0],
                graph["layer_2"].x[edge[1], 0],
            ],
            [
                graph["layer_1"].x[edge[0], 1],
                graph["layer_2"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )
    layer_2_to_mppc_edges = graph["layer_2", "to", "mppc"]
    for edge_idx in range(layer_2_to_mppc_edges.edge_index.shape[1]):
        if layer_2_to_mppc_edges.edge_labels[edge_idx] != 1:
            color="orange"
        else:
            color="green"
        edge = layer_2_to_mppc_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["layer_2"].x[edge[0], 0],
                graph["mppc"].x[edge[1], 0],
            ],
            [
                graph["layer_2"].x[edge[0], 1],
                graph["mppc"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )
    mppc_to_layer_3_edges = graph["mppc", "to", "layer_3"]
    for edge_idx in range(mppc_to_layer_3_edges.edge_index.shape[1]):
        if mppc_to_layer_3_edges.edge_labels[edge_idx] != 1:
            color="orange"
        else:
            color="green"
        edge = mppc_to_layer_3_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["mppc"].x[edge[0], 0],
                graph["layer_3"].x[edge[1], 0],
            ],
            [
                graph["mppc"].x[edge[0], 1],
                graph["layer_3"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )
    layer_3_to_layer_4_edges = graph["layer_3", "to", "layer_4"]
    for edge_idx in range(layer_3_to_layer_4_edges.edge_index.shape[1]):
        if layer_3_to_layer_4_edges.edge_labels[edge_idx] != 1:
            color="orange"
        else:
            color="green"
        edge = layer_3_to_layer_4_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["layer_3"].x[edge[0], 0],
                graph["layer_4"].x[edge[1], 0],
            ],
            [
                graph["layer_3"].x[edge[0], 1],
                graph["layer_4"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )

def plot_hetero_graph_to_axis(ax, graph, no_edges = False):
    mppc_nodes = graph["mppc"].x
    pixel_nodes = graph["pixel"].x
    ax.scatter(mppc_nodes[:, 0], mppc_nodes[:, 1], c="blue", s=20, label="MPPC hits")
    ax.scatter(pixel_nodes[:, 0], pixel_nodes[:, 1], c="red", s=20, label="Pixel hits")
    if no_edges:
        return
    pixel_to_mppc_edges = graph["pixel", "to", "mppc"]
    for edge_idx in range(pixel_to_mppc_edges.edge_index.shape[1]):
        if pixel_to_mppc_edges.edge_labels[edge_idx] != 1:
            color="orange"
        else:
            color="green"
        edge = pixel_to_mppc_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["pixel"].x[edge[0], 0],
                graph["mppc"].x[edge[1], 0],
            ],
            [
                graph["pixel"].x[edge[0], 1],
                graph["mppc"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )
    pixel_to_pixel_edges = graph["pixel", "to", "pixel"]
    for edge_idx in range(pixel_to_pixel_edges.edge_index.shape[1]):
        if pixel_to_pixel_edges.edge_labels[edge_idx] != 1:
            color="orange"
        else:
            color="green"
        edge = pixel_to_pixel_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["pixel"].x[edge[0], 0],
                graph["pixel"].x[edge[1], 0],
            ],
            [
                graph["pixel"].x[edge[0], 1],
                graph["pixel"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )
    mppc_to_mppc_edges = graph["mppc", "to", "mppc"]
    for edge_idx in range(mppc_to_mppc_edges.edge_index.shape[1]):
        if mppc_to_mppc_edges.edge_labels[edge_idx] != 1:
            color="orange"
        else:
            color="green"
        edge = mppc_to_mppc_edges.edge_index[:, edge_idx]
        ax.plot(
            [
                graph["mppc"].x[edge[0], 0],
                graph["mppc"].x[edge[1], 0],
            ],
            [
                graph["mppc"].x[edge[0], 1],
                graph["mppc"].x[edge[1], 1],
            ],
            c=color,
            alpha=0.3,
        )
    ax.legend(loc="upper right", fontsize=8)