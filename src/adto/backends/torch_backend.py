try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(
        "PyTorch is not installed. Please install PyTorch to use the PyTorch backend."
    ) from e

from adto.non_ad_ops import compute_compliance, bisection, external_linear_solver

torch.set_default_dtype(torch.float64)


def solve_host_pure(A_data, i_inds, j_inds, b):
    # convert to numpy on host
    A_data = A_data.detach().cpu().numpy()
    i_inds = i_inds.detach().cpu().numpy()
    j_inds = j_inds.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return torch.from_numpy(external_linear_solver(A_data, i_inds, j_inds, b))


class SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_data, i_inds, j_inds, b):
        # Forward solve
        x = solve_host_pure(A_data, i_inds, j_inds, b)

        # Save tensors for backward
        ctx.save_for_backward(A_data, i_inds, j_inds, x)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        A_data, i_inds, j_inds, x = ctx.saved_tensors

        # Solve adjoint system: A^T lambda = grad_x
        lambda_ = solve_host_pure(
            A_data,
            j_inds,   # swapped
            i_inds,
            grad_x
        )

        # Gradient w.r.t. A_data
        # dA_ij = -lambda_i * x_j
        dA_data = -lambda_[i_inds] * x[j_inds]

        # Gradient w.r.t. b
        db = lambda_

        # None for integer indices
        return dA_data, None, None, db


def solve(A_data, i_inds, j_inds, b):
    return SparseSolve.apply(A_data, i_inds, j_inds, b)


def assemble_stiffness_matrix_parts(E, problem_data):
    """Assemble global stiffness matrix parts (PyTorch version)."""
    device, dtype = E.device, E.dtype
    # Convert NumPy → Torch
    KE = torch.as_tensor(
        problem_data["KE"],
        dtype=dtype,
        device=device
    )  # (8, 8), float

    cMat = torch.as_tensor(
        problem_data["cMat"],
        dtype=torch.long,
        device=device
    )  # (n_elem, 8), indices

    # iK = kron(cMat, ones((8, 1))).T.ravel(order='F')
    iK = torch.kron(cMat, torch.ones(
        (8, 1), device=device, dtype=cMat.dtype))
    iK = iK.reshape(-1)

    # jK = kron(cMat, ones((1, 8))).ravel()
    jK = torch.kron(cMat, torch.ones(
        (1, 8), device=device, dtype=cMat.dtype))
    jK = jK.reshape(-1)

    # sK = (KE.ravel(order='F')[None, :] * E[:, None]).ravel()
    KE_flat = KE.t().reshape(-1)  # column-major flatten
    sK = (KE_flat.unsqueeze(0) * E.unsqueeze(1)).reshape(-1)

    return iK, jK, sK


def reduce_K(iK, jK, sK, free_dofs, n_dofs):
    """
    Reduce the global stiffness matrix to free DOFs only.

    Equivalent to K[np.ix_(free, free)] but implemented via triplet filtering.
    """

    device = iK.device
    free_dofs = torch.as_tensor(
        free_dofs, dtype=torch.long, device=device
    )

    # Build inverse map: old_dof -> new_dof index, -1 for fixed DOFs
    inv_map = -torch.ones(
        n_dofs, dtype=torch.long, device=device
    )
    inv_map[free_dofs] = torch.arange(
        free_dofs.numel(), device=device
    )
    # Mask: keep entries where both row and column are free
    mask = (inv_map[iK] >= 0) & (inv_map[jK] >= 0)
    # Filter and remap triplets
    iK_f = inv_map[iK[mask]]
    jK_f = inv_map[jK[mask]]
    sK_f = sK[mask]

    return iK_f, jK_f, sK_f

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
