import os
import importlib


def get_backend():
    backend = os.environ.get("ML_BACKEND", None)
    if backend is None:
        raise RuntimeError(
            "ML_BACKEND must be set before importing adto"
        )

    if backend == "jax":
        return importlib.import_module("adto.backends.jax_backend")
    elif backend == "torch":
        return importlib.import_module("adto.backends.torch_backend")
    else:
        raise ValueError(f"Unknown ML_BACKEND: {backend}")
