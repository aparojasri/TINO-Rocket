# TINO-Rocket: Turbulence-Informed Neural Operator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Accelerated Emulation of Regenerative-Cooling Solvers via Uncertainty-Aware Neural Operators.**

## 💡 Abstract
TINO (Turbulence-Informed Neural Operator) is a Bayesian deep learning framework designed to replace slow CFD solvers in flight-critical control loops. Unlike standard surrogates, TINO provides **Epistemic Uncertainty Quantification**, allowing the control system to know *when* the model is unsure.

This project demonstrates a **500x speedup** over traditional solvers while retaining the ability to predict thermal runaway events during rocket engine startup.

## 🧠 Architecture
* **Model:** Fourier Neural Operator (FNO) learning in the frequency domain.
* **Uncertainty:** Monte Carlo (MC) Dropout for probabilistic forecasting.
* **Input:** Time-varying boundary conditions (Mass Flow, Inlet Pressure).
* **Output:** Spatiotemporal wall temperature field $T_w(x, t)$ + Confidence Intervals $(2\sigma)$.

## 🛡️ The "Soft Sensor" Hypothesis
A key contribution of this work is the validation of the AI as a sensor health monitor.
* **Hypothesis:** Model uncertainty correlates with input data quality.
* **Result:** When synthetic noise (simulating sensor degradation) is injected, TINO's uncertainty bounds widen linearly, providing a robust "Safety Signal" to the flight computer.

## ⚡ Performance
| Metric | Traditional CFD | TINO (Ours) |
| :--- | :--- | :--- |
| **Inference Time** | ~600 ms | **1.24 ms** |
| **Resolution** | Grid Dependent | Mesh Invariant |
| **Uncertainty** | Deterministic | **Probabilistic** |

## 🚀 Usage
1.  **Train the Model:**
    ```bash
    python train_fno.py --epochs 500 --batch_size 32
    ```
2.  **Run Inference (with Uncertainty):**
    ```bash
    python evaluate.py --mode probabilistic --samples 100
    ```

## 📚 Tech Stack
* **Core:** PyTorch, Torch-FFT.
* **Data Handling:** NumPy, Pandas.
* **Visualization:** Matplotlib, Seaborn.
