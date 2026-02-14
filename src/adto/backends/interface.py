"""Interface to select the appropriate backend based on environment variable.
"""
import os
import importlib
import warnings


def get_backend():
    backend = os.environ.get("ML_BACKEND", None)
    if backend is None:
        backend = "jax"  # Default to JAX if not specified
        # Set it to os so that it can be accessed by nn_models when it imports keras
        os.environ["ML_BACKEND"] = backend
        warnings.warn(
            "ML_BACKEND not set. Defaulting to 'jax'. Changing run-time needs kernel restart."
        )

    if backend == "jax":
        return importlib.import_module("adto.backends.jax_backend")
    elif backend == "torch":
        return importlib.import_module("adto.backends.torch_backend")
    else:
        raise ValueError(f"Unknown ML_BACKEND: {backend}")
