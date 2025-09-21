import os
from common_numpy import compute_compliance, bisection
# Detect active backend
BACKEND = os.environ.get("ML_BACKEND", "jax")
print(f"Using backend: {BACKEND}")

if BACKEND == "jax":
    import jax
    from jax.scipy.signal import convolve
    # Enable float64 for better numerical stability (especially in FEA)
    jax.config.update("jax_enable_x64", True)

    # ------------------------
    # 1. Compliance with custom VJP
    # ------------------------

    # Declare compliance function as a custom_vjp primitive
    compute_compliance_differentiable = jax.custom_vjp(compute_compliance)

    # Forward pass

    def compute_compliance_fwd(x, problem_data):
        c, ce = compute_compliance(x, problem_data)
        residuals = {
            "elem_comp": ce,         # Element-wise compliance
            "xphy": x,               # Physical densities
            "penal": problem_data['penal'],
            "E0": problem_data["E0"],
            "Emin": problem_data["E_min"]
        }
        return (c, ce), residuals  # Outputs and residuals for backward

    # Backward pass (using analytical gradient)

    def compute_compliance_bwd(residuals, down_cotangents):
        # VJP seed (∂L/∂c, ∂L/∂ce) from upstream
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

    # ------------------------
    # 2. Bisection with custom VJP
    # ------------------------
    # Mark bisection as custom_vjp with `root_fn` as
    #  nondiff_argnum (it's a Python function)
    bisection_differentiable = jax.custom_vjp(
        bisection, nondiff_argnums=(0,))

    # Forward pass

    def bisection_fwd(root_fn, x, lb, ub, max_iter, tol):
        eta_star = bisection(root_fn, x, lb, ub, max_iter, tol)
        residuals = (x, eta_star)  # Needed for backward
        return eta_star, residuals

    # Backward pass via Implicit Function Theorem

    def bisection_vjp(root_fn, residuals, down_cotangents):
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

    # ------------------------
    # 3. Filters
    # ------------------------

    def apply_density_filter(x, problem_data):
        """JAX version of density filter"""
        Ny, Nx = problem_data['Ny'], problem_data['Nx']
        h, Hs = problem_data['h'], problem_data['Hs']
        x_2d = x.reshape((Ny, Nx), order='F')
        x_filtered = convolve(x_2d, h, mode='same') / Hs
        return x_filtered.ravel(order='F')

    def volume_enforcing_filter(x, volfrac):
        """JAX version with differentiable sigmoid"""
        def root_fn(eta, x_inp):
            return jax.nn.sigmoid(eta + x_inp).mean() - volfrac

        eta_star = bisection_differentiable(root_fn, x)
        return jax.nn.sigmoid(eta_star + x)


elif BACKEND == "torch":
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)

    class ComplianceAD(torch.autograd.Function):
        """Custom PyTorch autograd function for compliance computation"""

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
            """Backward pass: compute dC/dx"""
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
            x, eta_star = ctx.saved_tensors
            root_fn = ctx.root_fn

            # Ensure eta_star is differentiable
            eta_star = eta_star.detach().requires_grad_()
            x = x.detach().requires_grad_()

            # Define f(x, eta*) ≈ 0
            def func(eta):
                return root_fn(x, eta)

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
        return BisectionAD.apply(x, root_fn, lb, ub, max_iter, tol)

    def apply_density_filter(x, problem_data):
        """PyTorch version of density filter"""
        Ny, Nx = problem_data['Ny'], problem_data['Nx']
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
        """PyTorch version with differentiable sigmoid"""
        def root_fn(eta, x_inp):
            return torch.sigmoid(eta + x_inp).mean() - volfrac

        eta_star = BisectionAD.apply(x, root_fn)
        return torch.sigmoid(eta_star + x)

else:
    raise ValueError(f"Unsupported backend: {BACKEND}")

# Export the right functions based on backend
__all__ = ['apply_density_filter', 'volume_enforcing_filter',
           'compute_compliance_differentiable', 'bisection_differentiable']
