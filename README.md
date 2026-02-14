# Automatic Differentiation in (Neural) Topology Optimization

Educational implementation of topology optimization using automatic differentiation (AD) in modern ML frameworks (JAX and PyTorch) for sensitivity analysis. Contains examples of writing custom rules for AD integration, for linear solvers and root finders (using the Implicit function Theorem).

## Repository Structure

```
ADTO/
├── src/adto/                 # Main package
│   ├── __init__.py
│   ├── nn_models.py         # Neural network architectures
│   ├── non_ad_ops.py        # Non-AD operations
│   ├── utils.py             # Utility functions
│   └── backends/
│       ├── interface.py      # Backend interface
│       ├── jax_backend.py    # JAX implementation
│       ├── torch_backend.py  # PyTorch implementation
│       └── ad_backend.py     # AD utilities
├── examples/                 # Jupyter notebooks
│   ├── TO.ipynb             # Standard TO (OC method)
│   └── neuralTO.ipynb       # Neural TO
└── pyproject.toml           # Package configuration
```

## Quick Start with Google Colab (Recommended)

The easiest way to get started is using Google Colab—no installation needed:

1. Open any notebook in Colab:
   - [TO.ipynb](https://colab.research.google.com/github/SNMS95/ADTO/blob/main/examples/TO.ipynb)
   - [neuralTO.ipynb](https://colab.research.google.com/github/SNMS95/ADTO/blob/main/examples/neuralTO.ipynb)

2. Select your backend (JAX or PyTorch) and run the cells.

## Installation

### Local Installation with Conda

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SNMS95/ADTO.git
   cd ADTO
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n adto_env python=3.12
   conda activate adto_env
   ```

3. **Install the package and dependencies:**
   ```bash
   pip install -e .
   ```

4. **Install a backend (choose one):**
   - **JAX:** https://docs.jax.dev/en/latest/installation.html
   - **PyTorch:** https://pytorch.org/get-started/locally/

5. **For notebook support:**
   ```bash
   conda install jupyter ipykernel
   ```

## Quick Start

Run any notebook in the `examples/` directory:
```bash
jupyter notebook examples/neuralTO.ipynb
```

Select your backend (JAX or PyTorch) within the notebook and execute cells sequentially.

## Citation

If you use this code, please cite the accompanying article in Structural and Multidisciplinary Optimization.
