# Automatic Differentiation in (Neural) Topology Optimization

This repository accompanies the educational article
“Leveraging Automatic Differentiation in Modern Machine Learning Frameworks for (Neural) Topology Optimization”
(submitted to Structural and Multidisciplinary Optimization).

---

## Overview

This repository provides minimal, educational implementations demonstrating how automatic differentiation (AD) can be applied to topology optimization (TO) using modern ML frameworks such as JAX and PyTorch.

No installation is required — simply open any of the notebooks (*.ipynb) in Google Colab by clicking on the “Open in Colab” badge.
You can choose the backend (either PyTorch or JAX), and the autodiff capabilities of the selected ML framework will be used for sensitivity analysis.

---

## Repository Contents
- common_numpy.py
Contains core topology optimization operations implemented in NumPy, which does not support automatic differentiation.
Includes:
	-   FEA preprocessing
	-   Compliance computation
	-	Optimality Criteria (OC) method
	-	Simple bisection algorithm for scalar root-finding
-	backend_utils.py
Provides JAX and PyTorch implementations of the same operations, including custom autodiff rules to make them differentiable and compatible with larger ML workflows.
-	nn_keras.py
Defines neural network architectures using the Keras API.
The interface is backend-agnostic and provides a simple way to parameterize the density field.
-	TO.ipynb
Demonstrates standard topology optimization using the Optimality Criteria method (without neural networks).
-	neuralTO.ipynb
Demonstrates Neural Topology Optimization (NTO), where a neural network parameterizes the density field, and sensitivities are obtained via automatic differentiation.

---

## How to Use

Ideally
	1.	Open any notebook (e.g., TO.ipynb or neuralTO.ipynb) using Google Colab.
	2.	Select the desired backend (JAX or PyTorch) within the notebook.
	3.	Run the cells sequentially to explore differentiable TO workflows.

For running locally,
....
---

## Citation

If you use this code or find it helpful, please cite the accompanying article once it is published in Structural and Multidisciplinary Optimization.