from adto.backends.interface import get_backend

_backend = get_backend()

# Re-export backend functions
solve = _backend.solve
assemble_stiffness_matrix_parts = _backend.assemble_stiffness_matrix_parts
compute_compliance_differentiable = _backend.compute_compliance_differentiable
bisection_differentiable = _backend.bisection_differentiable
apply_density_filter = _backend.apply_density_filter
volume_enforcing_filter = _backend.volume_enforcing_filter
assemble_stiffness_matrix_parts = _backend.assemble_stiffness_matrix_parts
reduce_K = _backend.reduce_K

# Explicitly declare for type checkers
__all__ = [
    'solve',
    'assemble_stiffness_matrix_parts',
    'compute_compliance_differentiable',
    'bisection_differentiable',
    'apply_density_filter',
    'volume_enforcing_filter',
    'reduce_K',
]
