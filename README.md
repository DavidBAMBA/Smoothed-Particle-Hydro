# Smoothed-Particle-Hydro
## SPH Simulation: Sock capturing tests

This repository contains the implementation and documentation for simulating the **Sod Shock Tube** and the **Sedov Blast Wave** problems using **Smoothed Particle Hydrodynamics (SPH)**. These benchmarks are fundamental in computational fluid dynamics and demonstrate the accuracy and robustness of SPH methods in shock-capturing and energy dissipation.

## Features

- **Density Computation**: Iterative calculation of density using adaptive smoothing lengths (`h`) with a Newton-Raphson scheme.
- **Momentum and Energy Conservation**: Includes dissipative terms for shock handling with velocity-dependent signal propagation.
- **Higher-Order Kernels**: Support for compact kernels like the M5 quartic spline to improve resolution and reduce bias.
- **Newton-Raphson Iteration**: For consistent coupling between smoothing length (`h`) and density (`ρ`).
- **Shock Capturing**: Dissipative terms in momentum and energy to handle shocks efficiently.

## Equations Implemented

1. **Density Computation**:
   \[
   \rho_a = \sum_b m_b W(\mathbf{r}_a - \mathbf{r}_b, h_a)
   \]

2. **Momentum Conservation**:
   \[
   \frac{d\mathbf{v}_a}{dt} = -\sum_b m_b \left[
   \frac{P_a}{\Omega_a \rho_a^2} \nabla_a W_{ab}(h_a) 
   + \frac{P_b}{\Omega_b \rho_b^2} \nabla_a W_{ab}(h_b)
   \right]
   \]

3. **Energy Conservation**:
   \[
   \frac{du_a}{dt} = \frac{P_a}{\Omega_a \rho_a^2} \sum_b m_b (\mathbf{v}_a - \mathbf{v}_b) \cdot \nabla_a W_{ab}(h_a)
   \]

4. **Dissipative Terms** (Shock Capturing):
   \[
   \Pi_\text{shock}^a = -\sum_b m_b \left[
   \frac{q_{ab}^a}{\rho_a^2 \Omega_a} \nabla_a W_{ab}(h_a) 
   + \frac{q_{ab}^b}{\rho_b^2 \Omega_b} \nabla_a W_{ab}(h_b)
   \right]
   \]

## Test Cases

1. **Sod Shock Tube**:
   - Solves a one-dimensional Riemann problem with discontinuities in density and pressure.
   - Validates the ability of the SPH method to capture shocks and contact discontinuities.

2. **Sedov Blast Wave**:
   - Models a spherical shock wave expanding from a central explosion.
   - Tests the robustness of SPH in extreme conditions and highly dynamic systems.

## Kernel Functions

The repository supports the following kernel formulations:
- **Cubic Spline Kernel**
- **M5 Quartic Kernel** for increased accuracy and reduced bias.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
