# LiteMish

LiteMish is a smooth, algebraic activation function designed as a lightweight alternative to Mish, especially for resource-constrained applications such as TinyML, embedded systems, and robot control. LiteMish achieves performance comparable to Mish while using only addition and multiplication, making it significantly faster and more energy-efficient on low-power hardware.

# Motivation
Modern smooth activation functions (e.g., Mish, Swish, Softplus) offer good learning performance but rely on expensive operations such as exponentials and logarithms. These operations are slow on microcontrollers and embedded CPUs.

LiteMish is designed to:

1. Preserve the smooth, non-monotonic behavior of Mish
1. Eliminate transcendental operations
1. Enable fast and energy-efficient inference on resource-constrained hardware

# Key Properties

LiteMish is:

1. Smooth (C¹ continuous)
1. Non-monotonic
1. Bounded below and unbounded above
1. Purely algebraic (addition and multiplication only)
1. Piecewise polynomial
1. Computationally efficient

# LiteMish Graph
<img width="554" height="432" alt="image" src="https://github.com/user-attachments/assets/bf3c24fc-7ef8-487c-9f73-6266067e23f4" />

