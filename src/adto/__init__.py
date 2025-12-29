from adto.non_ad_ops import (
    setup_fea_problem,
    optimality_criteria,
)
from adto.backends import (
    solve,
    assemble_stiffness_matrix_parts,
    compute_compliance_differentiable,
    bisection_differentiable,
    apply_density_filter,
    volume_enforcing_filter,
    reduce_K,
)

# Optionally export other modules
from adto import non_ad_ops
from adto import nn_models

from adto.backends.interface import get_backend
_backend = get_backend()
BACKEND = _backend.__name__.split(".")[-1].replace("_backend", "")

__all__ = [
    "setup_fea_problem",
    "optimality_criteria",
    "solve",
    "assemble_stiffness_matrix_parts",
    "compute_compliance_differentiable",
    "bisection_differentiable",
    "apply_density_filter",
    "volume_enforcing_filter",
    "reduce_K",
    "BACKEND",
    "non_ad_ops",
    "nn_models",
]
