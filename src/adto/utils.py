"""Utilities for BC specification and plotting."""
import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# ============================================================================
# Node Selection Helper Functions
# ============================================================================


def select_edge_nodes(nodeNrs: np.ndarray, edge: str) -> np.ndarray:
    """
    Select nodes on a specified edge.

    Args:
        nodeNrs: Node numbering array (Ny+1 x Nx+1).
        edge: Which edge to select ('top', 'bottom', 'left', 'right').

    Returns:
        Array of selected node indices.
    """
    if edge == "top":
        return nodeNrs[0, :]
    elif edge == "bottom":
        return nodeNrs[-1, :]
    elif edge == "left":
        return nodeNrs[:, 0]
    elif edge == "right":
        return nodeNrs[:, -1]
    else:
        raise ValueError(
            f"Unknown edge: {edge}. Use 'top', 'bottom', 'left', or 'right'."
        )


def select_corner_nodes(nodeNrs: np.ndarray, corner: str) -> np.ndarray:
    """
    Select nodes at a specified corner.

    Args:
        nodeNrs: Node numbering array (Ny+1 x Nx+1).
        corner: Which corner ('top-left', 'top-right', 'bottom-left', 'bottom-right').

    Returns:
        Array with single selected node index.
    """
    if corner == "top-left":
        return np.array([nodeNrs[0, 0]])
    elif corner == "top-right":
        return np.array([nodeNrs[0, -1]])
    elif corner == "bottom-left":
        return np.array([nodeNrs[-1, 0]])
    elif corner == "bottom-right":
        return np.array([nodeNrs[-1, -1]])
    else:
        raise ValueError(f"Unknown corner: {corner}.")


def select_center_node(nodeNrs: np.ndarray) -> np.ndarray:
    """
    Select the center node of the mesh.

    Args:
        nodeNrs: Node numbering array (Ny+1 x Nx+1).

    Returns:
        Array with single selected node index.
    """
    Ny, Nx = nodeNrs.shape[0] - 1, nodeNrs.shape[1] - 1
    return np.array([nodeNrs[Ny // 2, Nx // 2]])


def select_region_nodes(
    nodeNrs: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float]
) -> np.ndarray:
    """
    Select nodes within a rectangular region.

    Args:
        nodeNrs: Node numbering array (Ny+1 x Nx+1).
        x_range: (x_min, x_max) as fraction of domain width (0 to 1).
        y_range: (y_min, y_max) as fraction of domain height (0 to 1).

    Returns:
        Array of node indices in the region.

    Example:
        # Select all nodes in left half of domain
        nodes = select_region_nodes(nodeNrs, x_range=(0, 0.5), y_range=(0, 1))
    """
    Ny, Nx = nodeNrs.shape[0] - 1, nodeNrs.shape[1] - 1

    x_min_idx = int(np.round(x_range[0] * Nx))
    x_max_idx = int(np.round(x_range[1] * Nx))
    y_min_idx = int(np.round(y_range[0] * Ny))
    y_max_idx = int(np.round(y_range[1] * Ny))

    return nodeNrs[y_min_idx : y_max_idx + 1, x_min_idx : x_max_idx + 1].ravel()


def nodes_to_dofs(nodes: np.ndarray, dof_type: str = "both") -> np.ndarray:
    """
    Convert node indices to DOF indices.

    Args:
        nodes: Array of node indices.
        dof_type: Which DOFs to extract ('x', 'y', or 'both').
                 'x' returns even indices (x-direction),
                 'y' returns odd indices (y-direction),
                 'both' returns all DOFs for the nodes.

    Returns:
        Array of DOF indices.
    """
    if dof_type == "x":
        return 2 * nodes
    elif dof_type == "y":
        return 2 * nodes + 1
    elif dof_type == "both":
        return np.concatenate([2 * nodes, 2 * nodes + 1])
    else:
        raise ValueError(f"Unknown dof_type: {dof_type}. Use 'x', 'y', or 'both'.")
    

def visualize_bc(
    problem_data: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    scale_arrow: float = 1.0,
) -> None:
    """
    Visualize boundary conditions including fixed DOFs and applied loads.

    Creates a plot showing:
    - Mesh nodes
    - Fixed DOFs with constraint symbols:
      - 'X' for nodes fixed in both x and y directions
      - '→' for nodes fixed only in x direction
      - '↑' for nodes fixed only in y direction
    - Load vectors as arrows with magnitude and direction

    Args:
        problem_data: Dictionary containing problem data from setup_fea_problem.
                      Must have keys: 'Nx', 'Ny', 'fixed', 'F'.
        figsize: Figure size as (width, height).
        scale_arrow: Scale factor for arrow lengths in visualization.

    Returns:
        None - displays the plot
    """
    Nx = problem_data["Nx"]
    Ny = problem_data["Ny"]
    fixed = problem_data["fixed"]
    F = problem_data["F"]

    # Node coordinates: element width/height = 1.0
    # Flip y-coordinates so that y=0 is at bottom and y=Ny is at top
    x_coords = np.tile(np.arange(Nx + 1), (Ny + 1, 1))
    y_coords = np.tile(np.arange(Ny, -1, -1)[:, np.newaxis], (1, Nx + 1))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Draw design domain as a lightly shaded rectangle
    design_domain = mpatches.Rectangle(
        (0, 0),
        Nx,
        Ny,
        linewidth=2,
        edgecolor="black",
        facecolor="lightblue",
        alpha=0.15,
        zorder=0,
    )
    ax.add_patch(design_domain)

    # Plot mesh nodes
    ax.scatter(
        x_coords.ravel(),
        y_coords.ravel(),
        s=30,
        c="lightgray",
        zorder=1,
        edgecolors="gray",
    )

    # Classify fixed DOFs
    fixed_x_only = []  # Fixed in x direction only
    fixed_y_only = []  # Fixed in y direction only
    fixed_both = []  # Fixed in both directions

    for dof in fixed:
        node_idx = dof // 2
        dof_dir = dof % 2  # 0 = x, 1 = y

        # Convert node index to 2D coordinates
        node_2d = np.unravel_index(node_idx, (Ny + 1, Nx + 1), order="F")
        node_coords = (x_coords[node_2d], y_coords[node_2d])

        # Check if this node has other constraints
        other_dof = 1 - dof_dir  # The other DOF direction
        other_dof_idx = node_idx * 2 + other_dof

        if other_dof_idx in fixed:
            if node_coords not in fixed_both:
                fixed_both.append(node_coords)
        else:
            if dof_dir == 0:  # x-direction
                if node_coords not in fixed_x_only:
                    fixed_x_only.append(node_coords)
            else:  # y-direction
                if node_coords not in fixed_y_only:
                    fixed_y_only.append(node_coords)

    # Plot fixed DOF markers
    if fixed_both:
        fx, fy = zip(*fixed_both)
        ax.scatter(
            fx,
            fy,
            s=200,
            marker="X",
            c="red",
            zorder=3,
            label="Fixed (x & y)",
            linewidths=2,
            edgecolors="darkred",
        )

    if fixed_x_only:
        fx, fy = zip(*fixed_x_only)
        ax.scatter(
            fx,
            fy,
            s=150,
            marker="|",
            c="blue",
            zorder=3,
            label="Fixed (x)",
            linewidths=2,
        )

    if fixed_y_only:
        fx, fy = zip(*fixed_y_only)
        ax.scatter(
            fx,
            fy,
            s=150,
            marker="_",
            c="green",
            zorder=3,
            label="Fixed (y)",
            linewidths=2,
        )

    # Plot load vectors
    load_indices = np.where(np.abs(F) > 1e-10)[0]

    for dof_idx in load_indices:
        node_idx = dof_idx // 2
        dof_dir = dof_idx % 2  # 0 = x, 1 = y

        # Convert node index to 2D coordinates
        node_2d = np.unravel_index(node_idx, (Ny + 1, Nx + 1), order="F")
        x = x_coords[node_2d]
        y = y_coords[node_2d]

        load_magnitude = F[dof_idx]

        # Determine arrow direction and color
        if dof_dir == 0:  # x-direction
            dx = (
                scale_arrow * load_magnitude / np.abs(load_magnitude)
                if load_magnitude != 0
                else 0
            )
            dy = 0
            color = "blue"
            label_text = f"Load X\n{load_magnitude:.2e}"
        else:  # y-direction
            dx = 0
            dy = (
                scale_arrow * load_magnitude / np.abs(load_magnitude)
                if load_magnitude != 0
                else 0
            )
            color = "green"
            label_text = f"Load Y\n{load_magnitude:.2e}"

        # Draw arrow
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.2,
            head_length=0.15,
            fc=color,
            ec=color,
            linewidth=2,
            zorder=2,
            alpha=0.7,
        )

        # Add load value text
        offset_x = dx * 0.3 if dx != 0 else 0.3
        offset_y = dy * 0.3 if dy != 0 else 0.3
    ax.text(
        x + offset_x,
        y + offset_y,
        label_text,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
    )

    # Set labels and formatting
    ax.set_aspect("equal")
    ax.axis("off")  # Hide axes and labels
    ax.grid(False)

    # Create legend
    handles = []
    if fixed_both:
        handles.append(
            mpatches.Patch(facecolor="red", edgecolor="darkred", label="Fixed (x & y)")
        )
    if fixed_x_only:
        handles.append(mpatches.Patch(facecolor="blue", label="Fixed (x)"))
    if fixed_y_only:
        handles.append(mpatches.Patch(facecolor="green", label="Fixed (y)"))
    if load_indices.size > 0:
        handles.append(
            mpatches.Patch(facecolor="yellow", alpha=0.5, label="Applied Load")
        )

    if handles:
        ax.legend(handles=handles, loc="best", fontsize=10)

    plt.tight_layout()
    plt.show()