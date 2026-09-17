""" === Neural Topology Optimization Data Generation Script ===

Generates the data shown in the paper Appendix comparing
different neural network architectures and boundary conditions for topology 
optimization.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_topology_optimization_data(
    nn_type: str = "siren",
    Nx: int = 96,
    Ny: int = 64,
    bc_str: str = "tensile",
    volfrac: float = 0.5,
    max_iterations: int = 120,
    random_seed: int = 0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    nu: float = 0.3,
    rmin: float = 2.0,
    penal: float = 3.0,
    optimizer_str: str = "adam",
    optimizer_hyper_params: dict = None,
    ML_framework_to_use: str = "jax",
    nn_arch_details: dict = None,
):
    """
    Run neural topology optimization and optionally save results.

    Args:
        nn_type: Neural network type - "mlp", "siren", "cnn", or "simp"
        Nx: Number of elements along x-axis (CNN requires multiple of 8)
        Ny: Number of elements along y-axis (CNN requires multiple of 8)
        bc_str: Boundary condition - "mbb", "tensile", "bridge", or "michell"
        volfrac: Target volume fraction (0-1)
        max_iterations: Number of optimization steps
        random_seed: Seed for network initialization
        E0: Young's modulus of solid material
        Emin: Young's modulus of void
        nu: Poisson's ratio
        rmin: Radius for density filter
        penal: SIMP penalization factor
        optimizer_str: Optimizer choice - "adam", "sgd", "rmsprop", or "adagrad"
        optimizer_hyper_params: Dict with optimizer parameters (learning_rate, global_clipnorm, etc.)
        ML_framework_to_use: ML backend - "torch" or "jax"
        nn_arch_details: Dict with NN architecture settings (num_hidden_layers, hidden_units, etc.)

    Returns:
        losses: List of loss values per iteration
        designs: List of design arrays per iteration
        model_state: Tuple of (trainable_vars, non_trainable_vars)
    """

    # Set default hyperparameters if not provided
    if optimizer_hyper_params is None:
        optimizer_hyper_params = {
            "learning_rate": 1e-4 if nn_type.lower() == "siren" else 1e-2,
            "global_clipnorm": 1.0,
        }

    if nn_arch_details is None:
        nn_arch_details = {
            'num_hidden_layers': 3,
            'hidden_units': 256,
            'frequency_factor': 30.0,
            "latent_size": 128
        }

    # Sanity checks
    assert ML_framework_to_use in ["jax", "torch"], f"Backend must be 'jax' or 'torch', got {ML_framework_to_use}"
    assert isinstance(random_seed, int), "random_seed must be an integer"
    assert penal >= 1, "penal must be >= 1"
    assert rmin >= 1, "rmin must be >= 1"
    assert optimizer_str in ["adam", "sgd", "rmsprop", "adagrad"], f"Unknown optimizer: {optimizer_str}"
    assert bc_str in ["mbb", "tensile", "bridge", "michell"], f"Unknown boundary condition: {bc_str}"
    assert 0 < volfrac <= 1, "volfrac must be between 0 and 1"

    if nn_type.lower() == "cnn":
        assert Nx % 8 == 0 and Ny % 8 == 0, "CNN requires Nx and Ny to be multiples of 8"

    # Set backend before importing
    os.environ["ML_BACKEND"] = ML_framework_to_use

    # Import after setting backend
    from adto import (
        apply_density_filter,
        assemble_stiffness_matrix_parts,
        reduce_K,
        setup_fea_problem,
        solve,
        volume_enforcing_filter,
    )
    from adto.nn_models import create_network_and_input, get_optimizer
    from adto.non_ad_ops import bridge_bc, mbb_bc, michell_bc, tensile_bc

    if ML_framework_to_use == "jax":
        import jax
        import jax.numpy as jnp
    else:
        import torch

    def simp_and_reduced_solve(physical_densities, problem_data):
        """Apply SIMP and solve the linear system."""
        E = problem_data['E_min'] + physical_densities**penal * \
            (problem_data['E0'] - problem_data['E_min'])
        iK, jK, sK = assemble_stiffness_matrix_parts(E, problem_data)
        free_dofs = problem_data['free']
        f = problem_data['F']
        f_f = f[free_dofs]
        iK_f, jK_f, sK_f = reduce_K(iK, jK, sK, free_dofs, len(f))

        if ML_framework_to_use == "torch":
            u_f = solve(sK_f, iK_f, jK_f, torch.tensor(f_f, device=sK_f.device, dtype=sK_f.dtype))
        else:
            u_f = solve(sK_f, iK_f, jK_f, f_f)
        return u_f

    def run_with_jax_backend(problem_data, nn_model, nn_input, max_iterations=100,
                             trainable_vars=None, non_trainable_vars=None):
        """Training with JAX backend"""
        print(f"Training with JAX backend: {nn_type}, BC={bc_str}, Grid={Nx}x{Ny}, volfrac={volfrac}")

        def loss_fn(train_vars, non_train_vars):
            output, non_train_vars = nn_model.stateless_call(
                    train_vars, non_train_vars, nn_input)
            output = output.astype(jnp.float64)
            rho = volume_enforcing_filter(output, volfrac)
            rho = rho.ravel(order='F')
            physical_densities = apply_density_filter(rho, problem_data)
            physical_densities = physical_densities * problem_data['mask'].ravel(order='F')
            u_f = simp_and_reduced_solve(physical_densities, problem_data)
            f_f = problem_data['F'][problem_data['free']]
            compliance = u_f.T @ f_f
            return compliance, (non_train_vars, physical_densities)

        optimizer = get_optimizer(optimizer_str, **optimizer_hyper_params)

        trainable_vars = [v.value for v in nn_model.trainable_variables]
        non_trainable_vars = nn_model.non_trainable_variables
        optimizer.build(nn_model.trainable_variables)
        opt_vars = optimizer.variables

        losses = []
        designs = []
        for epoch in range(max_iterations):
            (loss, (non_trainable_vars, design)), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(trainable_vars, non_trainable_vars)
            trainable_vars, opt_vars = optimizer.stateless_apply(
                opt_vars, grads, trainable_vars)
            losses.append(loss)
            designs.append(design)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        return losses, designs, (trainable_vars, non_trainable_vars)

    def run_neural_optimization(problem_data, nn_model, nn_input, max_iterations=100,
                                trainable_vars=None, non_trainable_vars=None):
        """Train the neural network model."""
        if ML_framework_to_use == "jax":
            return run_with_jax_backend(problem_data, nn_model, nn_input, max_iterations=max_iterations,
                                        trainable_vars=trainable_vars, non_trainable_vars=non_trainable_vars)
        elif ML_framework_to_use == "torch":
            raise NotImplementedError("Torch backend not implemented yet.")
        else:
            raise ValueError(f"Unknown backend: {ML_framework_to_use}")

    # Setup boundary conditions
    bc_fn_dict = {
        "mbb": mbb_bc,
        "tensile": tensile_bc,
        "bridge": bridge_bc,
        "michell": michell_bc
    }
    bc_fn = bc_fn_dict[bc_str]

    if ML_framework_to_use == "jax":
        import jax.numpy as jnp
        mask = jnp.ones((Ny, Nx))
        if bc_str == "mbb":
            mask = jnp.ones((Ny, Nx)).at[0, :].set(1.0)
    else:
        import torch
        mask = torch.ones((Ny, Nx))
        if bc_str == "mbb":
            mask[0, :] = 1.0

    # Setup FEA problem
    problem_data = setup_fea_problem(Nx=Nx, Ny=Ny, rmin=rmin, E0=E0, Emin=Emin,
                                      penal=penal, nu=nu, bc_fn=bc_fn)
    problem_data['mask'] = mask

    # Create neural network and run optimization
    if nn_type.lower() == "simp":
        raise NotImplementedError("SIMP without neural network not yet supported in this function")
    else:
        nn, nn_input = create_network_and_input(nn_type=nn_type,
                                                hyper_params=nn_arch_details,
                                            random_seed=random_seed,
                                            grid_size=(Ny, Nx))

    losses, designs, (trainable_vars, non_trainable_vars) = run_neural_optimization(
        problem_data, nn, nn_input, max_iterations=max_iterations)

    return losses[-1], designs[-1]


# Example usage when run as script
if __name__ == "__main__":
    nns_to_test = ["mlp", "siren", "cnn"]
    bc_to_test = [ "michell", "bridge", "mbb", "tensile"]
    bc_settings = {
        "tensile": {"Nx": 96, "Ny": 64, "volfrac": 0.3},
        "mbb": {"Nx": 144, "Ny": 48, "volfrac": 0.2},
        "bridge": {"Nx": 96, "Ny": 96, "volfrac": 0.5},
        "michell": {"Nx": 128, "Ny": 64, "volfrac": 0.3}
    }
    # Run all of them, tabulate the losses and save the final design as png
    results = []
    for bc in bc_to_test:
        for nn in nns_to_test:
            print(f"Running {nn} with {bc} BC...")
            settings = bc_settings[bc]
            final_loss, final_design = generate_topology_optimization_data(
                nn_type=nn,
                Nx=settings["Nx"],
                Ny=settings["Ny"],
                bc_str=bc,
                volfrac=settings["volfrac"],
                max_iterations=200,
                random_seed=0,
                E0=1.0,
                Emin=1e-9,
                nu=0.3,
                rmin=2.0,
                penal=3.0,
                optimizer_str="adam",
                optimizer_hyper_params={"learning_rate": 1e-4 if nn == "siren"
                                        else 5e-3, "global_clipnorm": 1.0},
                nn_arch_details={'num_hidden_layers': 3, 'hidden_units': 256,
                                 'frequency_factor': 30.0, "latent_size": 128},
                save_results=True,
                output_prefix=f"{nn}_{bc}"
            )
            results.append((nn, bc, final_loss))
            plt.imshow(final_design.reshape(settings["Ny"], settings["Nx"],
                                             order="F"), cmap="Greys")
            # plt.title(f"{nn} - {bc} - Loss: {final_loss:.4f}")
            plt.axis('off')
            plt.savefig(f"{nn}_{bc}_final_design.png", dpi=600,
                        bbox_inches='tight')
            plt.close()

    # print results in tabular format
    df_results = pd.DataFrame(results,
                               columns=["NN_Type", "Boundary_Condition",
                                         "Final_Loss"])
    print(df_results)
    df_results.to_csv("results.csv", index=False)
