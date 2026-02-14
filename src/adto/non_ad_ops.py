"""
Non-differentiable operations for Topology Optimization.

This module contains the core Finite Element Analysis (FEA)
routines implemented using NumPy and SciPy. These operations are NOT natively
compatible with automatic differentiation (AD) frameworks like JAX or PyTorch.

To use these in an AD pipeline, one must usually define custom rules.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve
from typing import Dict, Tuple, Callable, Any, Optional
from adto.utils import (
    select_edge_nodes,
    select_corner_nodes,
    select_center_node,
    select_region_nodes,
    nodes_to_dofs,
    visualize_bc,
)
# Set seed for numpy


def set_random_seed(random_seed: int) -> None:
    """Sets the random seed for NumPy to ensure reproducibility."""
    np.random.seed(random_seed)


def mbb_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cantilever beam boundary conditions: fixed left edge, load at bottom-right.

    Args:
        Nx: Number of elements in x-direction.
        Ny: Number of elements in y-direction.
        nodeNrs: Node numbering array (Ny+1 x Nx+1).
        nDof: Total number of degrees of freedom.

    Returns:
        fixed: Array of fixed DOF indices.
        free: Array of free DOF indices.
        F: Load vector with applied forces.
    """
    # Fix left edge in x direction
    left_nodes = select_edge_nodes(nodeNrs, "left")
    fixed_x = nodes_to_dofs(left_nodes, "x")

    # Fix bottom-right corner in y direction
    br_node = select_corner_nodes(nodeNrs, "bottom-right")
    fixed_y = nodes_to_dofs(br_node, "y")

    fixed = np.union1d(fixed_x, fixed_y)
    free = np.setdiff1d(np.arange(nDof), fixed)

    # Load vector: downward unit load at bottom-right
    F = np.zeros(nDof)
    F[1] = -1.0

    return fixed, free, F


def tensile_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tensile beam boundary conditions: fixed left edge, tensileload at mid-right.

    Args:
        Nx: Number of elements in x-direction.
        Ny: Number of elements in y-direction.
        nodeNrs: Node numbering array (Ny+1 x Nx+1).
        nDof: Total number of degrees of freedom.

    Returns:
        fixed: Array of fixed DOF indices.
        free: Array of free DOF indices.
        F: Load vector with applied forces.
    """
    # Fix left edge in x direction
    left_nodes = select_edge_nodes(nodeNrs, "left")
    fixed_x = nodes_to_dofs(left_nodes, "x")
    # fixed point at the middle of the left edge
    mid_left_node = left_nodes[Ny // 2]
    fixed_y = nodes_to_dofs(np.array([mid_left_node]), "y")

    fixed = np.union1d(fixed_x, fixed_y)
    free = np.setdiff1d(np.arange(nDof), fixed)

    # Load vector: right unit load at mid-right
    F = np.zeros(nDof)
    mid_right_node = select_edge_nodes(nodeNrs, "right")[Ny // 2]
    F[nodes_to_dofs(np.array([mid_right_node]), "x")[0]] = 1.0

    return fixed, free, F


def bridge_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bridge boundary conditions: fixed left edge, load at mid-right.

    Args:
        Nx: Number of elements in x-direction.
        Ny: Number of elements in y-direction.
        nodeNrs: Node numbering array (Ny+1 x Nx+1).
        nDof: Total number of degrees of freedom.

    Returns:
        fixed: Array of fixed DOF indices.
        free: Array of free DOF indices.
        F: Load vector with applied forces.
    """
    # Bottom left & rights corners fixed in both directions
    bl_node = select_corner_nodes(nodeNrs, "bottom-left")
    br_node = select_corner_nodes(nodeNrs, "bottom-right")
    fixed = nodes_to_dofs(np.union1d(bl_node, br_node), "both")
    free = np.setdiff1d(np.arange(nDof), fixed)
    # Uniform load on the top edge
    # Apply a downward load of 1.0 distributed across the top edge nodes
    F = np.zeros(nDof)
    top_nodes = select_edge_nodes(nodeNrs, "top")
    for node in top_nodes:
        F[nodes_to_dofs(np.array([node]), "y")[0]] = -1.0 / len(top_nodes)

    return fixed, free, F


def setup_fea_problem(
    Nx: int = 64,
    Ny: int = 32,
    rmin: float = 2.0,
    E0: float = 1.0,
    Emin: float = 1e-6,
    penal: float = 3.0,
    nu: float = 0.3,
    random_seed: int = 0,
    bc_fn: Optional[
        Callable[[int, int, np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = None,
) -> Dict[str, Any]:
    """
    Precompute all problem-specific matrices and parameters.

    Sets up a 2D FEA problem with flexible boundary conditions. By default,
    uses a Cantilever beam with a fixed left edge and a downward load on the
    bottom-right corner. Uses 4-node quadrilateral elements.

    Args:
        Nx: Number of elements in the x-direction.
        Ny: Number of elements in the y-direction.
        rmin: Filter radius for the density filter.
        E0: Young's modulus of the solid material.
        Emin: Young's modulus of the void material (to avoid singularity).
        penal: Penalization factor for SIMP.
        nu: Poisson's ratio.
        random_seed: Seed for random number generation.
        bc_fn: Boundary condition function. Takes (Nx, Ny, nodeNrs, nDof) and
               returns (fixed, free, F). If None, uses mbb_bc.

               Example BC functions available:
               - mbb_bc: Standard MBB beam.
               - tensile_bc: Fixed left edge, tensile load at mid-right.
               - bridge_bc: Fixed bottom corners, uniform load on top edge.

    Returns:
        A dictionary containing all precomputed data:
        - 'Nx', 'Ny': Grid dimensions.
        - 'KE': Element stiffness matrix (8x8).
        - 'cMat': Connectivity matrix mapping elements to DOFs (nDof_per_elem x nElem).
        - 'fixed': Indices of fixed degrees of freedom.
        - 'free': Indices of free degrees of freedom.
        - 'F': Global load vector.
        - 'E0', 'E_min', 'penal': Material properties.
        - 'h': Filter kernel for density filtering.
        - 'Hs': Normalization factor for the filter (sum of weights).

    Note:
    - We use Fortran ('F') order for reshaping to maintain consistency with
        element numbering as in 88 lines code.
    - Nodes (and thus DOFs) are numbered in a grid pattern, from top-left to bottom-right.
        (column-wise numbering).
        e.g., for Nx=2, Ny=1:
        Node numbers: [[0, 2, 4],
                       [1, 3, 5]]
    """
    set_random_seed(random_seed)

    # Element stiffness matrix for 4-node quad element (analytical integration)
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
    # Map node indices to a grid
    nodeNrs = np.arange((1 + Nx) * (1 + Ny)).reshape(
        (1 + Ny), (1 + Nx), order='F')
    # Calculate global DOF indices for the top-left node of each element
    cVec = (nodeNrs[:-1, :-1] * 2 + 2).reshape(-1, 1, order='F').ravel()
    # Offsets to get all 8 DOFs for a quad element relative to the top-left node
    offsets = np.array([0, 1, 2*Ny + 2, 2*Ny + 3,
                       2*Ny, 2*Ny + 1, -2, -1])
    cMat = cVec[:, None] + offsets

    # Apply boundary conditions
    nDof = 2 * (Nx + 1) * (Ny + 1)
    if bc_fn is None:
        # Default to cantilever boundary conditions
        bc_fn = mbb_bc

    fixed, free, F = bc_fn(Nx, Ny, nodeNrs, nDof)

    # Density filter setup
    range_val = np.arange(-np.ceil(rmin) + 1, np.ceil(rmin))
    dx, dy = np.meshgrid(range_val, range_val)
    h = np.maximum(0, rmin - np.sqrt(dx**2 + dy**2))
    # Hs is the sum of filter weights for each element (used for normalization)
    Hs = convolve(np.ones((Ny, Nx)), h, mode='same')

    problem_data = {
        'Nx': Nx, 'Ny': Ny, 'KE': KE, 'cMat': cMat,
        'fixed': fixed, 'free': free, 'F': F,
        'E0': E0, 'E_min': Emin, 'penal': penal,
        'h': h, 'Hs': Hs
    }
    return problem_data


def external_linear_solver(A_data: np.ndarray, i_inds: np.ndarray,
                           j_inds: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve a linear system Ax = b using SciPy's sparse solver.

    Args:
        A_data: Non-zero values of the sparse matrix A.
        i_inds: Row indices of the non-zero values.
        j_inds: Column indices of the non-zero values.
        b: Right-hand side vector.

    Returns:
        x: Solution vector.
    """
    A = csr_matrix((A_data, (i_inds, j_inds)),
                   shape=(b.shape[0], b.shape[0]))
    x = spsolve(A, b)
    return x


def assemble_stiffness_matrix(E: np.ndarray, problem_data: Dict[str, Any]
                              ) -> csr_matrix:
    """
    Assemble the global stiffness matrix K from element stiffness matrices.

    Args:
        E: Young's modulus for each element (Nx*Ny,).
        problem_data: Dictionary containing 'KE' and 'cMat'.

    Returns:
        K: Global stiffness matrix in CSR format.
    """
    KE, cMat, F = problem_data['KE'], problem_data['cMat'], problem_data['F']
    nDof = len(F)

    # Build sparse matrix
    # Repeat indices for COO format construction
    iK = np.kron(cMat, np.ones((8, 1), dtype=int)).T.ravel(order='F')
    jK = np.kron(cMat, np.ones((1, 8), dtype=int)).ravel()
    # Scale element stiffness matrix by material properties E
    sK = (KE.ravel(order='F')[np.newaxis, :] * E[:, np.newaxis]).ravel()

    K = coo_matrix((sK, (iK, jK)), shape=(nDof, nDof)).tocsr()
    return K


def solve_displacement(K: csr_matrix, problem_data: Dict[str, Any]
                       ) -> np.ndarray:
    """Solve Ku = F for displacements u, accounting for boundary conditions."""
    F, free = problem_data['F'], problem_data['free']
    u = np.zeros(len(F))
    # Solve only for free DOFs - Any solver can be used here
    u[free] = spsolve(K[np.ix_(free, free)], F[free])
    return u


def compute_compliance(xphy: np.ndarray, problem_data: Dict[str, Any]
                       ) -> Tuple[float, np.ndarray]:
    """
    Compute compliance and its gradient w.r.t. design variables.

    This is the main physics function that will be wrapped with custom AD.

    Args:
        xphy: Physical densities, shape (Nx*Ny,)
        problem_data: Dictionary with precomputed problem data

    Returns:
        compliance: Scalar compliance value
        ce_unscaled: Element-wise strain energy term (u_e^T k_0 u_e), shape (Nx*Ny,).
                     Used for sensitivity calculation: dc/dx = -p * x^(p-1) * (E0-Emin) * ce_unscaled.
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
    # ce_unscaled = u_e^T * KE * u_e for each element
    ce_unscaled = np.sum((u_elem @ KE) * u_elem, axis=1)
    ce_scaled = E * ce_unscaled
    return ce_scaled.sum(), ce_unscaled


def bisection(root_fn: Callable[[float, Any], float], x: Any,
              lb: float = -10.0, ub: float = 10.0,
              max_iter: int = 100, tol: float = 1e-10) -> float:
    """
    Standard bisection algorithm to find root of root_fn(eta, fixed_inp) = 0.

    Assumes that function is monotonically increasing.

    Args:
        root_fn: Function to find root of. Signature: f(variable, fixed_input) -> float.
        x: Fixed input passed to root_fn.
        lb: Lower bound for search.
        ub: Upper bound for search.
        max_iter: Maximum iterations.
        tol: Tolerance for convergence.
    """
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


def optimality_criteria(rhoi: np.ndarray, dc: np.ndarray, dv: np.ndarray,
                        max_move: float = 0.2,
                        vol_constr_fn: Optional[Callable[[
                            np.ndarray], float]] = None
                        ) -> np.ndarray:
    """
    Optimality Criteria (OC) update scheme for topology optimization.

    Updates the density distribution to satisfy the volume constraint while
    decreasing compliance, based on the KKT conditions.

    Args:
        rhoi: Current density distribution.
        dc: Sensitivity of objective (compliance) w.r.t density.
        dv: Sensitivity of volume w.r.t density (usually constant).
        max_move: Maximum change in density per iteration (damping).
        vol_constr_fn: Function that returns (current_vol - target_vol).
    """

    def compute_xnew(lmid, rho):
        # Heuristic update rule derived from KKT conditions
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
