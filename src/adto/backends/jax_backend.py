try:
    import jax
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is not installed. Please install JAX to use the JAX backend."
    ) from e
from jax.scipy.signal import convolve
from adto.non_ad_ops import compute_compliance, bisection, external_linear_solver

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


@jax.custom_vjp
def solve(A_data, i_inds, j_inds, b):
    x = solve_host_pure(A_data, i_inds, j_inds, b)
    return x


def solve_host_fwd(A_data, i_inds, j_inds, b):
    x = solve(A_data, i_inds, j_inds, b)
    return x, (x, A_data, i_inds, j_inds)


def solve_host_bwd(res, g):
    x, A_data, i_inds, j_inds = res
    # Solve adjoint problem A^T * lambda = g
    # note the transpose by swapping i and j
    lambda_ = solve(A_data, j_inds, i_inds, g)
    # Compute gradient w.r.t. A_data
    dA_data = -lambda_[i_inds] * x[j_inds]
    db = lambda_
    return (dA_data, None, None, db)


solve.defvjp(solve_host_fwd, solve_host_bwd)


def assemble_stiffness_matrix_parts(E, problem_data):
    """Assemble global stiffness matrix"""
    KE, cMat = problem_data['KE'], problem_data['cMat']
    # Build sparse matrix
    iK = jnp.kron(cMat, jnp.ones((8, 1), dtype=int)).T.ravel(order='F')
    jK = jnp.kron(cMat, jnp.ones((1, 8), dtype=int)).ravel()
    sK = (KE.ravel(order='F')[jnp.newaxis, :] * E[:, jnp.newaxis]).ravel()
    return iK, jK, sK


def reduce_K(iK, jK, sK, free_dofs, n_dofs, return_sparse=False):
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
    if return_sparse:
        K_f = jsparse.BCOO((sK_f, jnp.stack([iK_f, jK_f]).T),
                           shape=(len(free_dofs),
                                  len(free_dofs)))
        return K_f
    else:
        return iK_f, jK_f, sK_f


def F_opt(problem_params, u_f, problem_data):
    """
    Residual function for the FEA system

    F(params, u) = K(params)u - f = 0.

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
