import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve

# Set seed for numpy


def set_random_seed(random_seed):
    np.random.seed(random_seed)


def setup_fea_problem(Nx=64, Ny=32, rmin=2.0, E0=1.0, Emin=1e-6, penal=3.0,
                      nu=0.3, random_seed=0):
    """
    Precompute all problem-specific matrices and parameters.
    Returns a dictionary containing all precomputed data.
    """
    set_random_seed(random_seed)
    # Element stiffness matrix for 4-node quad element
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0],
                   [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6],
                   [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4],
                   [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2],
                   [4, 9, 2, 3], [-9, -2, 3, 2]])

    KE = E0/(1-nu**2)/24 * (np.block([[A11, A12], [A12.T, A11]]) +
                            nu * np.block([[B11, B12], [B12.T, B11]]))

    # DOF connectivity matrix
    nodeNrs = np.arange((1 + Nx) * (1 + Ny)).reshape(
        (1 + Ny), (1 + Nx), order='F')
    cVec = (nodeNrs[:-1, :-1] * 2 + 2).reshape(-1, 1, order='F').ravel()
    offsets = np.array([0, 1, 2*Ny + 2, 2*Ny + 3,
                       2*Ny, 2*Ny + 1, -2, -1])
    cMat = cVec[:, None] + offsets

    # Boundary conditions
    fixed1 = np.arange(0, 2 * (Ny + 1), 2)  # Fix left edge in x
    fixed2 = 2 * nodeNrs[-1, -1] + 1        # Fix bottom-right corner in y
    fixed = np.union1d(fixed1, fixed2)

    nDof = 2 * (Nx + 1) * (Ny + 1)
    free = np.setdiff1d(np.arange(nDof), fixed)

    # Load vector
    F = np.zeros(nDof)
    F[1] = -1.0  # Downward unit load

    # # Density filter
    range_val = np.arange(-np.ceil(rmin) + 1, np.ceil(rmin))
    dx, dy = np.meshgrid(range_val, range_val)
    h = np.maximum(0, rmin - np.sqrt(dx**2 + dy**2))
    Hs = convolve(np.ones((Ny, Nx)), h, mode='same')

    problem_data = {
        'Nx': Nx, 'Ny': Ny, 'KE': KE, 'cMat': cMat,
        'fixed': fixed, 'free': free, 'F': F,
        'E0': E0, 'E_min': Emin, 'penal': penal,
        'h': h, 'Hs': Hs
    }
    return problem_data


def assemble_stiffness_matrix(E, problem_data):
    """Assemble global stiffness matrix"""
    KE, cMat, F = problem_data['KE'], problem_data['cMat'], problem_data['F']
    nDof = len(F)

    # Build sparse matrix
    iK = np.kron(cMat, np.ones((8, 1), dtype=int)).T.ravel(order='F')
    jK = np.kron(cMat, np.ones((1, 8), dtype=int)).ravel()
    sK = (KE.ravel(order='F')[np.newaxis, :] * E[:, np.newaxis]).ravel()
    K = coo_matrix((sK, (iK, jK)), shape=(nDof, nDof)).tocsr()
    return K


def solve_displacement(K, problem_data):
    """Solve for displacements"""
    F, free = problem_data['F'], problem_data['free']
    u = np.zeros(len(F))
    u[free] = spsolve(K[np.ix_(free, free)], F[free])
    return u


def compute_compliance(xphy, problem_data):
    """
    Compute compliance and its gradient w.r.t. design variables.

    This is the main physics function that will be wrapped with custom AD.

    Args:
        xphy: Physical densities, shape (Nx*Ny,)
        problem_data: Dictionary with precomputed problem data

    Returns:
        compliance: Scalar compliance value
        ce: Element-wise compliance gradients, shape (Nx*Ny,)
    """
    # Apply density filter
    xphy = xphy.ravel(order='F')

    # SIMP material interpolation
    E0, E_min, penal = problem_data['E0'], problem_data['E_min'], problem_data['penal']
    E = E_min + xphy**penal * (E0 - E_min)

    # Assemble and solve
    K = assemble_stiffness_matrix(E, problem_data)
    u = solve_displacement(K, problem_data)

    # Compute element-wise compliance for sensitivity
    cMat, KE = problem_data['cMat'], problem_data['KE']
    u_elem = u[cMat]
    ce_unscaled = np.sum((u_elem @ KE) * u_elem, axis=1)
    ce_scaled = E * ce_unscaled
    return ce_scaled.sum(), ce_unscaled


def bisection(root_fn, x, lb=-10, ub=10, max_iter=100, tol=1e-10):
    """Standard bisection algorithm to find root of 
    root_fn(eta, fixed_inp) = 0. Assumes that function is monotonically increasing."""
    for _ in range(max_iter):
        mid = (lb + ub) / 2
        mid_val = root_fn(mid, x)
        if mid_val > 0:
            ub = mid
        else:
            lb = mid
        if np.abs(mid_val) < tol:
            break
    return mid


# def optimality_criteria(x, dc, dv, g, move=0.2, tol=1e-3):
#     """
#     Optimality Criteria (OC) update for topology optimization.

#     Parameters
#     ----------
#     nelx, nely : int
#         Number of elements in x and y directions.
#     x : ndarray
#         Current design variable array of size (nelx * nely,).
#     volfrac : float
#         Prescribed volume fraction.
#     dc : ndarray
#         Sensitivity of compliance w.r.t. x.
#     dv : ndarray
#         Sensitivity of volume w.r.t. x.
#     g : float
#         Current constraint violation.
#     move : float, optional
#         Maximum change in design variable per iteration (default 0.2).
#     tol : float, optional
#         Relative tolerance for the bisection loop (default 1e-3).

#     Returns
#     -------
#     xnew : ndarray
#         Updated design variable array.
#     gt : float
#         Updated constraint violation.
#     """

#     # Function defining the volume constraint balance
#     def constraint_balance(lmid, x):
#         """Returns volume constraint violation for a given Lagrange multiplier lmid."""
#         x_candidate = np.maximum(0.0, np.maximum(
#             x - move,
#             np.minimum(1.0, np.minimum(
#                 x + move, x * np.sqrt(-dc / (dv * lmid))))
#         ))
#         # Need to flip sign here since bisection is for monotonically increasing functions
#         return -1*(g + np.sum(dv * (x_candidate - x)))

#     # Solve for the Lagrange multiplier using bisection
#     lmid = bisection(constraint_balance, x=x, lb=1e-9,
#                      ub=1e9, tol=tol, max_iter=500)

#     # Final update with computed multiplier
#     xnew = np.maximum(0.0, np.maximum(
#         x - move,
#         np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / (dv * lmid))))
#     ))
#     gt = g + np.sum(dv * (xnew - x))

#     return xnew, gt


def optimality_criteria(rhoi, dc, dv, max_move=0.2,
                        vol_constr_fn=None):
    """Fully differentiable version of the optimality criteria."""

    def compute_xnew(lmid, rho):
        rho_candidate = np.maximum(0.0, np.maximum(
            rho - max_move,
            np.minimum(1.0, np.minimum(
                rho + max_move, rho * np.sqrt(-dc / (dv * lmid))))
        ))
        return rho_candidate

    def f(lambda_, rho):
        xnew = compute_xnew(lambda_, rho)
        # Multiply by -1 so that bisection works
        return -1*vol_constr_fn(xnew)

    lambda_final = bisection(
        f, x=rhoi, lb=1e-9, ub=1e9, tol=1e-3, max_iter=500)
    return compute_xnew(lambda_final, rhoi)
