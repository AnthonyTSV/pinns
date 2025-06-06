# A Physics-Informed Machine Learning Approach for Heat Conduction Equation

Welcome to the **PINNs** project! This repository contains code and resources for solving the heat conduction equation in 2D and 3D using Physics-Informed Neural Networks (PINNs). Developed as part of a machine learning course FYS5429 at the university of Oslo.

---

## Project Overview

- **Goal:**  
  Leverage PINNs to approximate solutions to the heat conduction equation in both two and three dimensions.
- **Context:**  
  This project was completed as part of an advanced machine learning course FYS5429 at the university of Oslo, focusing on scientific machine learning and physics-based modeling.

---

## How It Works

Physics-Informed Neural Networks (PINNs) combine data-driven neural networks with the governing physical laws (expressed as PDEs). The loss function is augmented with the residuals of the heat equation, ensuring the model respects the underlying physics.

- **2D/3D Heat Equation:**  
  PINNs are trained to solve:
  $$-\nabla \cdot(\kappa\ \nabla T) = f$$
  where `T` is temperature, `kappa` is thermal conductivity.

- **Key Features:**
  - Solves both 2D and 3D variants
  - Customizable boundary conditions
  - Trains using NVIDIa PhysicsNeMo

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnthonyTSV/pinns.git
   cd pinns
   ```
2. **Install requirements for NVIDIA PhysicsNeMo:**
   ```bash
   docker pull nvcr.io/nvidia/physicsnemo/physicsnemo:25.03
   ```
