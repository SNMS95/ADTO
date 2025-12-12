"""
Provides backend-specific, differentiable implementations for TopOpt.

This module offers implementations of core topology optimization operations
for both JAX and PyTorch. It leverages the custom automatic differentiation
(AD) capabilities of these frameworks to provide gradients for operations
that are not natively differentiable, such as the finite element analysis
(FEA) solver and bisection root-finding algorithm.

The active backend is determined by the `ML_BACKEND` environment variable.
"""
import os
from non_ad_ops import compute_compliance, bisection, external_linear_solver

# Detect active backend
BACKEND = os.environ.get("ML_BACKEND", "jax")
print(f"Using backend: {BACKEND}")

# ==============================================================================
# JAX BACKEND IMPLEMENTATION
# ==============================================================================
if BACKEND == "jax":
    import jax
    from jax.scipy.signal import convolve
    import jax.numpy as jnp

    # Enable float64 for better numerical stability (especially in FEA)
    jax.config.update("jax_enable_x64", True)

    # --------------------------------------------------------------------------
    # 1. Differentiable Compliance Calculation (Custom VJP)
    # --------------------------------------------------------------------------

    # Declare compliance function as a custom_vjp primitive
    compute_compliance_differentiable = jax.custom_vjp(compute_compliance)

    def compute_compliance_fwd(x, problem_data):
        """
        Forward pass for compliance calculation.

        This function executes the original `compute_compliance` function and
        saves necessary intermediate results (residuals) for the backward pass.

        Returns:
            A tuple containing the compliance and element-wise compliance, and
            a dictionary of residuals for the backward pass.
        """
        c, ce = compute_compliance(x, problem_data)
        residuals = {
            "elem_comp": ce,         # Element-wise compliance
            "xphy": x,               # Physical densities
            # SIMP parameters
            "penal": problem_data['penal'],
            "E0": problem_data["E0"],
            "Emin": problem_data["E_min"]
        }
        return (c, ce), residuals  # Outputs and residuals for backward

    # Backward pass (using analytical gradient)
    def compute_compliance_bwd(residuals, down_cotangents):
        """
        Backward pass (VJP) for compliance calculation.

        Computes the vector-Jacobian product using the analytical sensitivity
        of compliance with respect to physical densities.

        Returns:
            The gradient with respect to the physical densities `xphy`.
        """
        c_dot, ce_dot = down_cotangents
        del ce_dot  # ce has no downstream relevance

        # Unpack residuals
        ce = residuals["elem_comp"]
        xphy = residuals["xphy"]
        penal = residuals["penal"]
        E0 = residuals["E0"]
        Emin = residuals["Emin"]

        # dC/dx_phys
        dc_dxphy = -penal * xphy**(penal - 1) * (E0 - Emin) * ce
        vjp_xphys = dc_dxphy * c_dot  # Apply chain rule
        # Second arg is ∂L/∂problem_data = None
        return vjp_xphys.reshape(xphy.shape, order='F'), None

    # Register VJP rules
    compute_compliance_differentiable.defvjp(
        compute_compliance_fwd, compute_compliance_bwd)

    # --------------------------------------------------------------------------
    # 2. Differentiable Bisection Root-Finding (Custom VJP)
    # --------------------------------------------------------------------------

    # Mark bisection as custom_vjp with `root_fn` as
    #  nondiff_argnum (it's a Python function)
    bisection_differentiable = jax.custom_vjp(
        bisection, nondiff_argnums=(0,))

    def bisection_fwd(root_fn, x, lb, ub, max_iter, tol):
        """
        Forward pass for the bisection algorithm.

        Executes the standard bisection and saves the input `x` and the
        found root `eta_star` for the backward pass.

        Returns:
            The root `eta_star` and a tuple of residuals.
        """
        eta_star = bisection(root_fn, x, lb, ub, max_iter, tol)
        residuals = (x, eta_star)  # Needed for backward
        return eta_star, residuals

    def bisection_vjp(root_fn, residuals, down_cotangents):
        """
        Backward pass (VJP) for the bisection algorithm.

        Computes the gradient using the Implicit Function Theorem (IFT).
        The root `eta_star` is an implicit function of the input `x`, defined
        by `root_fn(eta_star, x) = 0`.

        The IFT states: ∂η*/∂x = - (∂F/∂x) / (∂F/∂η), where F is the root_fn.

        Returns:
            A tuple containing the gradient with respect to `x` and None for
            other non-differentiable arguments.
        """
        x, eta_star = residuals
        # Compute ∂F/∂x and ∂F/∂eta at solution
        #  (eta_star s.t. F(x, eta_star) = 0)
        df_deta, df_dx = jax.grad(root_fn, (0, 1))(eta_star, x)

        # IFT: ∂η*/∂x = - (∂F/∂x) / (∂F/∂η)
        lambda_val = down_cotangents / df_deta
        vjp_x = -lambda_val * df_dx
        # Only x is differentiable
        return (vjp_x.reshape(x.shape), None, None, None, None)

    # Register VJP
    bisection_differentiable.defvjp(bisection_fwd, bisection_vjp)

    # --------------------------------------------------------------------------
    # 3. Filters
    # --------------------------------------------------------------------------

    def apply_density_filter(x, problem_data):
        """
        Applies a density filter to the design variables using convolution.

        This is a JAX-compatible version that uses `jax.scipy.signal.convolve`.
        The filter helps prevent checkerboarding and ensures a min.
        length scale.
        """
        Ny, Nx = problem_data['Ny'], problem_data['Nx']
        h, Hs = problem_data['h'], problem_data['Hs']
        x_2d = x.reshape((Ny, Nx), order='F')
        x_filtered = convolve(x_2d, h, mode='same') / Hs
        return x_filtered.ravel(order='F')

    def volume_enforcing_filter(x, volfrac):
        """
        Applies a volume-preserving sigmoid projection filter.

        This function finds a scalar `eta` such that the average value of
        `sigmoid(x + eta)` equals the target volume fraction `volfrac`.
        The root-finding for `eta` is performed using the differentiable
        bisection method.
        """
        def root_fn(eta, x_inp):
            return jax.nn.sigmoid(eta + x_inp).mean() - volfrac

        eta_star = bisection_differentiable(root_fn, x)
        return jax.nn.sigmoid(eta_star + x)

    # ------------------------
    # 4. Differentiable FEA Solver via External Callback
    # ------------------------

    def solve_host_pure(A_data, i_inds, j_inds, b):
        """
        Wraps the external SciPy sparse solver for use within JAX.

        `jax.pure_callback` allows calling a non-JAX function (like SciPy's
        `spsolve`) from within a JAX-jitted function. We must provide the
        shape and dtype of the expected output.
        """
        return jax.pure_callback(
            # The function to call
            external_linear_solver,
            # The shape and dtype of the output
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            A_data, i_inds, j_inds, b
        )

    def assemble_stiffness_matrix_parts(E, problem_data):
        """Assemble global stiffness matrix"""
        KE, cMat = problem_data['KE'], problem_data['cMat']
        # Build sparse matrix
        iK = jnp.kron(cMat, jnp.ones((8, 1), dtype=int)).T.ravel(order='F')
        jK = jnp.kron(cMat, jnp.ones((1, 8), dtype=int)).ravel()
        sK = (KE.ravel(order='F')[jnp.newaxis, :] * E[:, jnp.newaxis]).ravel()
        return iK, jK, sK

    def reduce_K(iK, jK, sK, free_dofs, n_dofs):
        """
        Reduces the global stiffness matrix to only include free dofs.

        This is equivalent to `K[np.ix_(free, free)]` but implemented in a way
        that is traceable by JAX.
        """
        # We need a mapping old_index -> new_index
        # -1 for fixed dofs
        inv_map = -jnp.ones((n_dofs,), dtype=int)
        inv_map = inv_map.at[free_dofs].set(jnp.arange(len(free_dofs)))

        # Mask: keep entries where BOTH row and col are free dofs
        mask = (inv_map[iK] >= 0) & (inv_map[jK] >= 0)

        # Filter triplets
        iK_f = inv_map[iK[mask]]
        jK_f = inv_map[jK[mask]]
        sK_f = sK[mask]

        # build BCOO matrix
        K_f = jax.experimental.sparse.BCOO((sK_f, jnp.stack([iK_f, jK_f]).T),
                                           shape=(len(free_dofs),
                                                  len(free_dofs)))

        return K_f

    def F_opt(problem_params, u_f, problem_data):
        """
        Residual function for the FEA system, F(params, u) = K(params)u - f = 0.

        This function is used by the Implicit Function Theorem in the backward
        pass of the FEA solver. It must be a pure JAX function.
        """
        penal, rho = problem_params
        # Form K from rho
        K_f = get_reduced_K(penal, rho, problem_data)
        free_dofs = problem_data['free']
        f = problem_data['F']
        # reduce f to f_f
        f_f = f[free_dofs]
        residual = K_f @ u_f - f_f
        return residual

    def get_reduced_K(penal, rho, problem_data):
        """Get reduced stiffness matrix K_f."""
        E = problem_data['E_min'] + rho**penal * \
            (problem_data['E0'] - problem_data['E_min'])
        iK, jK, sK = assemble_stiffness_matrix_parts(E, problem_data)
        free_dofs = problem_data['free']
        f = problem_data['F']
        # reduce K to K_f
        K_f = reduce_K(iK, jK, sK, free_dofs, len(f))
        return K_f

    @jax.custom_vjp
    def solve_from_params(problem_params, problem_data):
        """
        Solves the FEA system for displacements `u_f` given problem parameters.

        This function is decorated with `custom_vjp` to define a custom
        gradient for the entire FEA solve process.
        """
        penal, rho = problem_params
        K_f = get_reduced_K(penal, rho, problem_data)
        # Reduce f to f_f
        free_dofs = problem_data['free']
        f = problem_data['F']
        f_f = f[free_dofs]
        # Solve K_f u_f = f_f
        # Extract sparse data for the external solver
        sK_f = K_f.data
        iK_f, jK_f = K_f.indices[:, 0], K_f.indices[:, 1]
        u_f = solve_host_pure(sK_f, iK_f, jK_f, f_f)
        return u_f

    def solve_from_params_fwd(problem_params, problem_data):
        """Forward pass for the FEA solver."""
        u_f = solve_from_params(problem_params, problem_data)
        return u_f, (u_f, problem_params, problem_data)

    def solve_from_params_bwd(res, g):
        """Backward pass (VJP) for the FEA solver using IFT."""
        u_f, problem_params, problem_data = res
        penal, rho = problem_params
        # We use IFT for the VJP
        # First compute dF/du_f = K_f
        K_f = get_reduced_K(penal, rho, problem_data)
        K_f = K_f.transpose()
        # from BCOO get data, i_inds, j_inds
        sK_f = K_f.data
        iK_f, jK_f = K_f.indices[:, 0], K_f.indices[:, 1]
        # Solve adjoint problem K_f^T * lambda = g
        lambda_f = solve_host_pure(sK_f, iK_f, jK_f, g)
        # Compute dF/dparams * lambda using jax.vjp

        def F_opt_params(params):
            return F_opt(params, u_f, problem_data)

        vjp_fun = jax.vjp(F_opt_params, problem_params)[1]
        dparams = vjp_fun(lambda_f)[0]
        # Multiply by -1 since we have du/dparams = - (dF/du)^-1 * dF/dparams
        dparams = jax.tree.map(lambda x: -x, dparams)
        return (dparams, None)

    solve_from_params.defvjp(solve_from_params_fwd, solve_from_params_bwd)

# ==============================================================================
# PYTORCH BACKEND IMPLEMENTATION
# ==============================================================================
elif BACKEND == "torch":
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)

    # --------------------------------------------------------------------------
    # 1. Differentiable Compliance Calculation (torch.autograd.Function)
    # --------------------------------------------------------------------------

    class ComplianceAD(torch.autograd.Function):
        """Custom PyTorch autograd Function for compliance computation."""

        @staticmethod
        def forward(ctx, x_tensor, problem_data):
            """Forward pass: compute compliance and save residuals"""
            x_np = x_tensor.detach().cpu().numpy()
            c, ce = compute_compliance(x_np, problem_data)

            # Save for backward
            ctx.save_for_backward(
                x_tensor, torch.tensor(ce, dtype=x_tensor.dtype))
            ctx.problem_data = problem_data

            c_tensor = torch.tensor(c, dtype=x_tensor.dtype,
                                    device=x_tensor.device)
            ce_tensor = torch.tensor(
                ce, dtype=x_tensor.dtype, device=x_tensor.device)
            return c_tensor, ce_tensor

        @staticmethod
        def backward(ctx, *grad_outputs):
            """
            Backward pass for compliance.

            Computes the gradient of the loss with respect to the input `x_tensor`
            using the chain rule and the analytical sensitivity of compliance.
            """
            x_tensor, ce_tensor = ctx.saved_tensors
            x = x_tensor.detach().cpu().numpy()
            ce = ce_tensor.detach().cpu().numpy()

            penal = ctx.problem_data['penal']
            E0 = ctx.problem_data['E0']
            Emin = ctx.problem_data['E_min']

            dc_dx = -penal * x**(penal - 1) * (E0 - Emin) * ce
            dc_dx_tensor = torch.tensor(
                dc_dx, dtype=x_tensor.dtype, device=x_tensor.device)

            # grad_output[0] for compliance scalar
            grad_x = grad_outputs[0] * dc_dx_tensor
            return grad_x, None  # Only gradients w.r.t x_tensor

    def compute_compliance_differentiable(x, problem_data):
        return ComplianceAD.apply(x, problem_data)

    class BisectionAD(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, root_fn, lb=-10.0, ub=10.0, max_iter=100,
                    tol=1e-10):
            # Call the external root solver (e.g., NumPy-based bisection)
            # Convert x to numpy
            x_np = x.detach().cpu().numpy()

            # Wrap the PyTorch root_fn to accept NumPy and return NumPy
            def root_fn_wrapped(eta_np, x_np_local=x_np):
                eta_tensor = torch.tensor(eta_np, dtype=x.dtype)
                x_tensor = torch.tensor(x_np_local, dtype=x.dtype)
                return root_fn(eta_tensor, x_tensor).item()

            eta_star = bisection(
                root_fn_wrapped, x_np, lb, ub, max_iter, tol)
            eta_star_tensor = torch.tensor(
                eta_star, dtype=x.dtype, device=x.device)
            ctx.save_for_backward(x, eta_star_tensor)
            ctx.root_fn = root_fn
            return eta_star_tensor

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass for bisection using the Implicit Function Theorem (IFT).

            The root `eta_star` is an implicit function of `x`. We use
            `torch.autograd.grad` to compute the partial derivatives required
            by the IFT: ∂η*/∂x = - (∂F/∂x) / (∂F/∂η), where F is the root_fn.
            """

            x, eta_star = ctx.saved_tensors
            root_fn = ctx.root_fn

            # Ensure eta_star is differentiable
            eta_star = eta_star.detach().requires_grad_()
            x = x.detach().requires_grad_()

            # Re-evaluate the root function at the solution to build the graph
            # for autograd.
            def func(eta):
                return root_fn(x, eta)

            # Compute partial derivatives using torch.autograd
            with torch.enable_grad():
                f = func(eta_star)
                df_deta, = torch.autograd.grad(
                    f, eta_star, retain_graph=True, create_graph=True)
                df_dx, = torch.autograd.grad(
                    f, x, retain_graph=True, create_graph=True)

            # Apply IFT: dη*/dx = - (∂F/∂x) / (∂F/∂η)
            lambda_val = grad_output / df_deta
            grad_x = -lambda_val * df_dx

            return grad_x, None, None, None, None, None

    def bisection_differentiable(x, root_fn, lb=-10.0, ub=10.0, max_iter=100,
                                 tol=1e-10):
        """
        Differentiable bisection root-finding function for PyTorch.

        Wraps the `BisectionAD` custom autograd function.
        """
        return BisectionAD.apply(x, root_fn, lb, ub, max_iter, tol)

    # --------------------------------------------------------------------------
    # 3. Filters
    # --------------------------------------------------------------------------

    def apply_density_filter(x, problem_data):
        """
        Applies a density filter using 2D convolution in PyTorch.
        """
        Ny, Nx = problem_data['Ny'], problem_data['Nx']
        # Convert numpy arrays to torch tensors
        h = torch.tensor(problem_data['h'], dtype=x.dtype, device=x.device)
        Hs = torch.tensor(problem_data['Hs'], dtype=x.dtype, device=x.device)
        # To enable fortran encoding in torch
        x_2d = x.reshape(Nx, Ny).t()
        kernel = h.unsqueeze(0).unsqueeze(0)
        x_input = x_2d.unsqueeze(0).unsqueeze(0)
        x_filtered = F.conv2d(
            x_input, kernel, padding='same') / Hs.unsqueeze(0).unsqueeze(0)
        return x_filtered.squeeze().T.ravel()

    def volume_enforcing_filter(x, volfrac):
        """
        Applies a volume-preserving sigmoid projection filter in PyTorch.
        """
        def root_fn(eta, x_inp):
            return torch.sigmoid(eta + x_inp).mean() - volfrac

        eta_star = BisectionAD.apply(x, root_fn)
        return torch.sigmoid(eta_star + x)

else:
    raise ValueError(f"Unsupported backend: {BACKEND}")

# Export the correct functions based on the selected backend
__all__ = ['apply_density_filter', 'volume_enforcing_filter',
           'compute_compliance_differentiable', 'bisection_differentiable']
