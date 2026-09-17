"""Memory usage profiler for neural networks on different input grid sizes.

This script systematically measures the peak memory footprint of Equinox MLPs
via JAX's XLA compilation analysis. It varies two parameters:
  - NN depth: Controls model complexity (number of layers)
  - Grid size: Controls input size (number of evaluation points)

For each combination, the script:
  1. Creates a random MLP with specified depth and fixed width=256
  2. Constructs a 2D grid input of the specified size
  3. Compiles a jacobian-traced loss function with JAX JIT
  4. Extracts peak memory from XLA's analysis (temp + argument + output - alias)
  5. Records results in a CSV for analysis

Use this to understand how memory scales with network architecture and
input resolution, useful for planning GPU memory budgets in topology optimization.

Output:
  - Matplotlib plot showing memory vs. depth for each grid size
  - CSV file: memory_usage_vs_nn_depth.csv
"""
import gc

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from adto.nn_models import create_network_and_input

# with or without float 64 is irrelevant for memory analysis,
#  but we disable it to avoid unnecessary memory overhead
jax.config.update("jax_enable_x64", False)

NN_TYPE = "mlp"
WIDTH = 256


def get_memory_usage(nn, grid_size):
    """Measure peak memory footprint of a compiled neural network.

    Estimates the peak memory used by JAX's XLA compiler when evaluating
    a neural network with vmap on a 2D grid input. The memory includes:
      - Temporary buffers created during computation
      - Argument size (input, parameters)
      - Output buffers
    minus alias buffers (in-place operations).

    Args:
        nn (eqx.Module): Equinox neural network model
        grid_size (int): Side length of 2D grid; total inputs = grid_size^2

    Returns:
        float: Peak memory in MB, or None if memory analysis unavailable
    """
    # Create a 2D grid of points in [-1, 1]^2
    x = jnp.linspace(-1, 1, grid_size)
    y = jnp.linspace(-1, 1, grid_size)
    xx, yy = jnp.meshgrid(x, y)  # Broadcast to (grid_size, grid_size) meshes
    inputs = jnp.stack([xx.ravel(), yy.ravel()], axis=-
                       1)  # Shape: (grid_size^2, 2)
    trainable_vars = [v.value for v in nn.trainable_variables]
    non_trainable_vars = [v.value for v in nn.non_trainable_variables]

    def loss_fn(trainable_vars, non_trainable_vars, inputs):
        """Compute mean of NN predictions over all input points."""
        output, non_train_vars = nn.stateless_call(
            trainable_vars, non_trainable_vars, inputs
        )
        return jnp.mean(output)  # Aggregate to scalar

    # Jacobian of loss w.r.t. parameters (traces backward pass)
    grad_fn = jax.jacrev(loss_fn)
    f = jax.jit(grad_fn)  # Compile to XLA for memory analysis

    # Extract compiled XLA program and query its memory requirements
    compiled_step = f.lower(trainable_vars, non_trainable_vars, inputs).compile()
    compiled_stats = compiled_step.memory_analysis()

    if compiled_stats is not None:
        # NOTE: Peak memory in XLA includes:
        # temp_size: intermediate tensors during computation
        # argument_size: input and parameter buffers
        # output_size: result buffers
        # alias_size: overlapped/reused memory (subtract to avoid double-counting)
        total = (compiled_stats.temp_size_in_bytes + compiled_stats.argument_size_in_bytes
                 + compiled_stats.output_size_in_bytes - compiled_stats.alias_size_in_bytes)
        return total / (1024**2)  # Convert to MB

    return None


# Define sweep parameters
# Grid sizes: 100x100, 300x300, ..., 900x900 points
grid_sizes = jnp.arange(100, 1000, 200)
# NN depths from 2 to 9 layers, all with width=256
nn_depths = jnp.arange(2, 10, 1)

# Initialize DataFrame to store results (rows=depth, columns=grid sizes)
columns = ["x_depths"] + ["y_mem_grid" + str(size) for size in grid_sizes]
df = pd.DataFrame(columns=columns)
df["x_depths"] = nn_depths

# Nested loop: for each grid size, measure memory across all NN depths
for grid_size in grid_sizes:
    depth_mem_usages = []
    print(f"Grid Size: {grid_size}")
    for nn_depth in nn_depths:
        # Create random MLP: 2D input -> 1D output, depth layers, width 256
        nn, _ = create_network_and_input(
                                nn_type=NN_TYPE,
                                hyper_params={
                                            'num_hidden_layers': nn_depth,
                                            'hidden_units': WIDTH,
                                            })
        # Measure peak memory for this (grid_size, depth) combination
        mem_usage = get_memory_usage(nn, int(grid_size))
        depth_mem_usages.append(mem_usage)
        print(
            f"Grid Size: {grid_size}, NN Depth: {nn_depth}, Memory Usage: {mem_usage:.2f} MB")
        del nn  # Explicitly free model after measurement
        gc.collect()  # Force garbage collection
    # Plot memory vs. depth for this grid size (one line per grid size)
    plt.plot(nn_depths, depth_mem_usages,
             label=f"Grid Size {grid_size}", marker='o')
    # Store results in DataFrame
    df["y_mem_grid" + str(grid_size)] = depth_mem_usages

# Finalize and display results
plt.title("Memory Usage vs. NN depth for Various Grid Sizes")
plt.xlabel("NN depth (width=256)")
plt.ylabel("Memory Usage (MB)")
plt.legend()
plt.show()

# Export results to CSV for further analysis and documentation
df.to_csv("memory_usage_vs_nn_depth.csv", index=False)
