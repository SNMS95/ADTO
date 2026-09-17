""" === Topology Optimization Data Generation Script ===

Generates the data shown in the paper Appendix for standard TO
with different boundary conditions.
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

    Returns:
        losses: List of loss values per iteration
        designs: List of design arrays per iteration
        model_state: Tuple of (trainable_vars, non_trainable_vars)
    """
    ML_framework_to_use = "jax"  # or "torch"
    assert isinstance(random_seed, int), "random_seed must be an integer"
    assert penal >= 1, "penal must be >= 1"
    assert rmin >= 1, "rmin must be >= 1"
    assert bc_str in ["mbb", "tensile", "bridge", "michell"], f"Unknown boundary condition: {bc_str}"
    assert 0 < volfrac <= 1, "volfrac must be between 0 and 1"

    if nn_type.lower() == "cnn":
        assert Nx % 8 == 0 and Ny % 8 == 0, "CNN requires Nx and Ny to be multiples of 8"

    # Set backend before importing
    os.environ["ML_BACKEND"] = "jax"  # or "torch"

    # Import after setting backend
    from adto import (
        apply_density_filter,
        assemble_stiffness_matrix_parts,
        optimality_criteria,
        reduce_K,
        setup_fea_problem,
        solve,
    )
    from adto.non_ad_ops import bridge_bc, mbb_bc, michell_bc, tensile_bc

    if ML_framework_to_use == "jax":
        import jax
        import jax.numpy as jnp
    else:
        import torch

    def simp_and_reduced_solve(physical_densities, problem_data):
        # Compute compliance
        E = problem_data["E_min"] + physical_densities**penal * (
            problem_data["E0"] - problem_data["E_min"]
        )
        iK, jK, sK = assemble_stiffness_matrix_parts(E, problem_data)
        free_dofs = problem_data["free"]
        f = problem_data["F"]
        f_f = f[free_dofs]
        # reduce K to K_f
        iK_f, jK_f, sK_f = reduce_K(iK, jK, sK, free_dofs, len(f))
        # Solve system
        if ML_framework_to_use == "torch":
            # Torch needs tensor inputs to be passed since we provide gradients w.r.t the force vector as well in teh custom VJP rule
            u_f = solve(
                sK_f, iK_f, jK_f, torch.tensor(f_f, device=sK_f.device, dtype=sK_f.dtype)
            )
        else:
            u_f = solve(sK_f, iK_f, jK_f, f_f)
        return u_f


    def run_with_jax_backend(problem_data, rho_init=None, max_iterations=100):
        """Training example with JAX backend"""
        print("Training with JAX backend...")

        # Initialize design variables
        if rho_init is None:
            rho_init = (
                jnp.ones((problem_data["Ny"], problem_data["Nx"])) * volfrac
            )  # Initial guess
        else:
            assert rho_init.shape == (problem_data["Ny"], problem_data["Nx"])
        # Create loss function and optimizer

        def volume_constraint_fn(rho):
            """Needed only for OC update"""
            rho = rho.ravel(order="F")
            physical_densities = apply_density_filter(rho, problem_data)
            # apply mask
            physical_densities = physical_densities * problem_data["mask"].ravel(order="F")
            constraint = jnp.mean(physical_densities) - volfrac
            return constraint

        def obj_and_constraint_fn(rho):
            rho = rho.ravel(order="F")
            physical_densities = apply_density_filter(rho, problem_data)
            physical_densities = physical_densities * problem_data["mask"].ravel(order="F")
            # Compute compliance - SIMP, assemble K, Remove free DOFs, solve system
            u_f = simp_and_reduced_solve(physical_densities, problem_data)
            f_f = problem_data["F"][problem_data["free"]]
            compliance = u_f.T @ f_f
            constraint = jnp.mean(physical_densities) - volfrac
            aux_info = (compliance, constraint, physical_densities)
            return (compliance, constraint), aux_info

        # Training loop
        objs = []
        constraints = []
        designs = []

        rho = rho_init

        for itr in range(max_iterations):
            jacobian, aux = jax.jacrev(obj_and_constraint_fn, has_aux=True)(rho)
            obj_grads = jacobian[0]
            constraint_grads = jacobian[1]
            rho = optimality_criteria(
                rho.ravel(),
                obj_grads.ravel(),
                constraint_grads.ravel(),
                vol_constr_fn=volume_constraint_fn,
            )
            obj, constr, design = aux
            objs.append(obj)
            constraints.append(constr)
            designs.append(design)
            if itr % 1 == 0:
                print(f"JAX - Iteration {itr}, Obj: {obj:.6f}, constr: {constr:.6f}")

        return objs, constraints, designs

    def run_optimization(problem_data, rho_init=None, max_iterations=100):
        """Train the neural network model."""
        backend = ML_framework_to_use
        if backend == "jax":
            return run_with_jax_backend(
                problem_data, rho_init=rho_init, max_iterations=max_iterations
            )
        elif backend == "torch":
            raise NotImplementedError("PyTorch backend not yet implemented for this function")
        else:
            raise ValueError(f"Unknown backend: {backend}")

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

    objs, constraints, designs = run_optimization(
        problem_data, rho_init=None, max_iterations=max_iterations)

    return objs[-1], designs[-1]


# Example usage when run as script
if __name__ == "__main__":
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
            print(f"Running with {bc} BC...")
            settings = bc_settings[bc]
            final_loss, final_design = generate_topology_optimization_data(
                nn_type="simp",
                Nx=settings["Nx"],
                Ny=settings["Ny"],
                bc_str=bc,
                volfrac=settings["volfrac"],
                max_iterations=100,
                random_seed=0,
                E0=1.0,
                Emin=1e-9,
                nu=0.3,
                rmin=2.0,
                penal=3.0,
            )
            results.append((bc, final_loss))
            plt.imshow(final_design.reshape(settings["Ny"], settings["Nx"], order="F"), cmap="Greys")
            # plt.title(f"{nn} - {bc} - Loss: {final_loss:.4f}")
            plt.axis('off')
            plt.savefig(f"simp_{bc}_final_design.png", dpi=600, bbox_inches='tight')
            plt.close()

    # print results in tabular format
    df_results = pd.DataFrame(results, columns=["Boundary_Condition", "Final_Loss"])
    print(df_results)
    df_results.to_csv("results_simo.csv", index=False)
