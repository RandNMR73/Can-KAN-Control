import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import efficient_kan.kan as kan
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_b_spline_basis_on_edge(grid, order, ax):
    """
    Plot all B-spline basis functions on a specific edge.

    Args:
        grid (torch.Tensor): Knot points (grid values).
        order (int): Spline order.
        ax (matplotlib.axes.Axes): Axes object to plot on.
    """
    grid_np = grid.cpu().numpy()

    # Validate grid size
    if len(grid_np) < order + 1:
        raise ValueError(
            f"Grid size ({len(grid_np)}) must be at least order + 1 ({order + 1}) for B-splines."
        )

    # Generate x-values for plotting
    x_vals = np.linspace(grid.min().item(), grid.max().item(), 500)

    # Compute B-spline basis functions
    bases = np.zeros((len(x_vals), len(grid_np) - order - 1))
    for i in range(len(grid_np) - order - 1):
        bases[:, i] = compute_single_basis(x_vals, grid_np, i, order)

    # Plot the basis functions
    for i in range(bases.shape[1]):
        ax.plot(x_vals, bases[:, i])
    ax.axis("off")  # Turn off axis for cleaner embedding


def compute_single_basis(x, grid, i, k):
    """
    Recursive computation of a single B-spline basis function.

    Args:
        x (numpy.ndarray): Points to evaluate the basis on.
        grid (numpy.ndarray): Knot points.
        i (int): Basis function index.
        k (int): Spline order.

    Returns:
        numpy.ndarray: Values of the basis function.
    """
    if k == 0:
        return np.where((x >= grid[i]) & (x < grid[i + 1]), 1.0, 0.0)
    else:
        denom1 = grid[i + k] - grid[i]
        denom2 = grid[i + k + 1] - grid[i + 1]

        term1 = ((x - grid[i]) / denom1 * compute_single_basis(x, grid, i, k - 1)) if denom1 > 0 else 0
        term2 = ((grid[i + k + 1] - x) / denom2 * compute_single_basis(x, grid, i + 1, k - 1)) if denom2 > 0 else 0

        return term1 + term2


def visualize_kan_with_basis(layers_hidden, kan_model, spline_order=3):
    """
    Visualize the KAN network with B-spline basis function plots embedded on edges.

    Args:
        layers_hidden (list): List defining the number of nodes in each layer.
        kan_model (KAN): The KAN model containing layers.
        spline_order (int): Order of the spline.
    """
    G = nx.DiGraph()
    pos = {}

    # Create nodes for the graph
    for layer_idx, layer_size in enumerate(layers_hidden):
        for node_idx in range(layer_size):
            node_name = f"L{layer_idx}_N{node_idx}"
            G.add_node(node_name)
            pos[node_name] = (layer_idx, -node_idx)

    # Create edges for the graph
    for layer_idx in range(len(layers_hidden) - 1):
        for in_idx in range(layers_hidden[layer_idx]):
            for out_idx in range(layers_hidden[layer_idx + 1]):
                G.add_edge(f"L{layer_idx}_N{in_idx}", f"L{layer_idx+1}_N{out_idx}")

    # Create the visualization
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray")

    # Embed basis function plots on edges
    ax = plt.gca()
    for edge in G.edges():
        node1, node2 = edge
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]

        # Create a new axis for the basis function plot
        inset_ax = ax.inset_axes([0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.1, 0.1])  # Adjust placement and size
        layer_idx = int(node1.split("_")[0][1:])  # Extract layer index from node name
        plot_b_spline_basis_on_edge(kan_model.layers[layer_idx].grid, spline_order, inset_ax)

    plt.title("KAN Network with B-spline Basis Functions")
    plt.show()


# Example Usage
if __name__ == "__main__":
    # Define the KAN model
    kan_model = kan.KAN(
        layers_hidden=[3, 4, 2],  # Example network structure
        grid_size=8,  # Increase grid size to at least 4 for spline_order=3
        spline_order=3
    )


    # Visualize the network with embedded B-spline plots
    visualize_kan_with_basis(
        layers_hidden=[3, 4, 2],
        kan_model=kan_model,
        spline_order=3
    )
