# src/adto/__init__.pyi
"""Type stubs for adto package"""

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

# From non_ad_ops
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
) -> Dict[str, Any]: ...
def cantilever_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def mbb_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def tensile_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def bridge_bc(
    Nx: int, Ny: int, nodeNrs: np.ndarray, nDof: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

# From utils (node selection and visualization)
def select_edge_nodes(nodeNrs: np.ndarray, edge: str) -> np.ndarray: ...
def select_corner_nodes(nodeNrs: np.ndarray, corner: str) -> np.ndarray: ...
def select_center_node(nodeNrs: np.ndarray) -> np.ndarray: ...
def select_region_nodes(
    nodeNrs: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float]
) -> np.ndarray: ...

def nodes_to_dofs(nodes: np.ndarray, dof_type: str = "both") -> np.ndarray: ...
def visualize_bc(
    problem_data: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 8),
    scale_arrow: float = 1.0,
) -> None: ...
def optimality_criteria(
    rhoi: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    max_move: float = 0.2,
    vol_constr_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> np.ndarray: ...

# From backends
def solve(K: Any, F: np.ndarray) -> np.ndarray: ...
def assemble_stiffness_matrix_parts(*args: Any, **kwargs: Any) -> Any: ...
def compute_compliance_differentiable(*args: Any, **kwargs: Any) -> Any: ...
def bisection_differentiable(*args: Any, **kwargs: Any) -> Any: ...
def apply_density_filter(x: np.ndarray, radius: float) -> np.ndarray: ...
def volume_enforcing_filter(*args: Any, **kwargs: Any) -> Any: ...
def reduce_K(*args: Any, **kwargs: Any) -> Any: ...

# Module-level attributes
BACKEND: str
non_ad_ops: Any
nn_models: Any
