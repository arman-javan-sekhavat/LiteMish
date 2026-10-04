# LiteMish

<p align="center">
  <img src="https://img.shields.io/github/stars/arman-javan-sekhavat/LiteMish?style=for-the-badge" alt="GitHub Stars">
  <img src="https://img.shields.io/github/forks/arman-javan-sekhavat/LiteMish?style=for-the-badge" alt="GitHub Forks">
  <img src="https://img.shields.io/github/license/arman-javan-sekhavat/LiteMish?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/last-commit/arman-javan-sekhavat/LiteMish?style=for-the-badge" alt="Last Commit">
</p>

<h3 align="center">
A fast, smooth, algebraic alternative to Mish for TinyML and resource-constrained AI
</h3>

<p align="center">
  <a href="#why-litemish">Why LiteMish</a> •
  <a href="#benchmark-results">Results</a> •
  <a href="#mathematical-definition">Mathematical Definition</a> •
  <a href="#implementation">Implementation</a> •
  <a href="#reproducibility">Reproducibility</a> •
  <a href="#citation">Citation</a>
</p>

---

## What is LiteMish?

**LiteMish** is a smooth, piecewise-polynomial activation function designed as a lightweight alternative to **Mish**, particularly for **TinyML, embedded systems, edge AI, robot control, and other resource-constrained environments**.

Unlike Mish and several other smooth activation functions that rely on operations such as exponential, logarithm, hyperbolic tangent, division, or square root, LiteMish is implemented using only **addition and multiplication** in its core algebraic form.

The function was designed to retain several useful properties associated with Mish while substantially reducing computational complexity and making it more energy-efficient on low-power hardware.

### Highlights

* **~28.66× faster than Mish** in an ESP32 scalar benchmark
* Uses only **addition and multiplication**
* Designed with **TinyML and embedded inference** in mind
* **Continuously differentiable (C¹)**
* **Non-monotonic**
* **Piecewise polynomial**
* **Bounded below**
* **Unbounded above**

---

## Why LiteMish?

Modern activation functions have demonstrated excellent learning capability, but their computational cost can matter when neural networks run on microcontrollers or other hardware with limited compute and energy resources.

For example, **Mish** is defined using operations involving `softplus` and `tanh`. These operations can be significantly more expensive on CPUs than simple arithmetic.

LiteMish takes a different approach:

> **It approximates the useful shape and behavior of Mish with a smooth piecewise-polynomial function that requires only basic arithmetic operations.**

This makes LiteMish particularly interesting for applications where **computational simplicity and learning capability** are important at the same time.

---

## Benchmark Results

The following results were reported in the accompanying research study.

### ESP32 computation time

A scalar call to each activation function was measured on an **ESP32 development board** using single-precision floating-point arithmetic.

Each activation function was evaluated **1,000 times**, and the total elapsed time was divided by 1,000.

| Activation Function | Time per Scalar Call (μs) |
| :------------------ | ------------------------: |
| **LiteMish**        |                 **0.130** |
| Squareplus          |                     0.430 |
| ELU                 |                     0.669 |
| Swish               |                     1.422 |
| Softplus            |                     2.178 |
| Mish                |                     3.726 |

LiteMish required **0.130 μs** per scalar call compared with **3.726 μs** for Mish, corresponding to approximately **28.66× lower measured computation time** in this experiment.

> **Important:** These are results from the reported ESP32 experiment. Performance might vary with hardware, compiler, optimization settings, numerical precision, and implementation details.

---

## Learning Performance

LiteMish was also evaluated on a regression task and on the **MNIST** and **CIFAR-100** classification datasets.

### MNIST

| Activation Function | Maximum Accuracy (%) | Average Accuracy (%) |
| :------------------ | -------------------: | -------------------: |
| **LiteMish**        |            **99.17** |            **99.10** |
| Mish                |                99.16 |                99.05 |
| Swish               |                99.06 |                99.04 |
| ELU                 |                99.12 |                99.07 |
| Softplus            |                99.13 |                99.10 |
| Squareplus          |                99.11 |                99.08 |

In the reported experiments, LiteMish achieved the highest maximum test accuracy and matched the best reported average accuracy on MNIST.

### CIFAR-100

| Activation Function | Test Accuracy (%) |
| :------------------ | ----------------: |
| **LiteMish**        |         **64.59** |
| Mish                |             61.57 |
| Swish               |             60.72 |
| ELU                 |             64.69 |
| Softplus            |             0.99* |
| Squareplus          |             48.22 |

* The reported Softplus experiment failed and produced a test accuracy of 0.99%.

LiteMish achieved **64.59%** accuracy, close to ELU at **64.69%**, while outperforming Mish, Swish, Softplus, and Squareplus in the reported experiment.

---

## Mathematical Definition
<img width="615" height="170" alt="image" src="https://github.com/user-attachments/assets/f6d5abdd-fcab-401f-9c5f-bbf5dba2210f" />

## LiteMish Graph
<img width="554" height="432" alt="image" src="https://github.com/user-attachments/assets/bf3c24fc-7ef8-487c-9f73-6266067e23f4" />

---

## Implementation

### TensorFlow

The TensorFlow implementation of LiteMish is:

```python
import tensorflow as tf

x1 = -7.42218730
x2 = -0.73744017
a = -0.0074315585222961555
b = +0.7453866928666856000

def LiteMish(x):
return tf.where(x < x1, 0.0, tf.where(x < x2, a*tf.square(x - x1), tf.where(x < 0.0, b*x*x + x, x)))
```

### JAX

The [`LiteMish.ipynb`](./LiteMish.ipynb) notebook contains the JAX implementation and parameter optimization experiments.

The core JAX implementation is:

```python
import jax.numpy as jnp
from jax import jit

x1 = -7.42218730
x2 = -0.73744017
a = -0.0074315585222961555
b = +0.7453866928666856000

@jit
def LiteMish(x):
return jnp.where(x < x1, 0.0, jnp.where(x < x2, a*jnp.square(x - x1), jnp.where(x < 0.0, b*x*x + x, x)))
```

---

### C++

The [`LiteMish.cpp`](./LiteMish.cpp) file contains the C++ implementation used for the ESP32 timing experiment.

A minimal implementation is:

```cpp
#define x1 -7.422187f
#define x2 -0.737440f
#define a -0.0074315f
#define b +0.7453866f

static inline float LiteMish(float x) {
  if (x > 0.0f){
    return x;
  }else if (x > x2){
    return b*x*x + x;
  }else if (x > x1){
    return a*(x-x1)*(x-x1);
  }else{
    return 0.0f;
  }
}
```

This implementation contains no calls to `exp`, `log`, `tanh`, `sqrt`, or floating-point division.

## Quick Example

In a neural network, LiteMish can conceptually replace another activation function:

```python
activation = LiteMish
```

For JAX-based experiments:

```python
import jax.numpy as jnp

x = jnp.array([-10.0, -5.0, -1.0, -0.5, 0.0, 1.0])

y = LiteMish(x)

print(y)
```

For embedded systems, the C++ implementation can be integrated directly into firmware.

---

## Repository Structure

```text
LiteMish/
├── LiteMish.cpp      # C++ / ESP32 implementation and benchmark
├── LiteMish.ipynb    # JAX implementation and parameter optimization
├── LICENSE            # MIT License
└── README.md          # Project documentation
```

---

## Reproducibility

The reported experiments compare LiteMish with:

* Mish
* Swish
* ELU
* Softplus
* Squareplus

For the learning experiments, the activation function was changed while the remaining model/training hyperparameters were kept unchanged within each comparison.

The ESP32 timing experiment used:

* ESP32 development board
* Single-precision floating-point arithmetic
* 1,000 scalar calls per measurement
* `esp_timer_get_time()` for timing

See [`LiteMish.cpp`](./LiteMish.cpp) and [`LiteMish.ipynb`](./LiteMish.ipynb) for the available experimental code.

---

## Research Paper

**LiteMish: A Computationally Efficient and Smooth Algebraic Alternative to Mish**

**Author:** Arman Javan Sekhavat Pishkhani

**Paper / Preprint:**
[TechRxiv](https://doi.org/10.36227/techrxiv.176591866.68698045/v2)

The study investigates the mathematical properties, learning performance, and computational efficiency of LiteMish, including experiments on regression, MNIST, CIFAR-100, and an ESP32 development board.

---

## Citation

If LiteMish is useful in your research or project, please cite the accompanying paper.

```bibtex
@article{javansekhavatpishkhani2025litemish,
  title   = {LiteMish: A Computationally Efficient and Smooth Algebraic Alternative to Mish},
  author  = {Javan Sekhavat Pishkhani, Arman},
  doi     = {10.36227/techrxiv.176591866.68698045},
  year    = {2025}
}
```

---

## Applications

LiteMish is especially beneficial for applications where computational efficiency matters, including:

* TinyML
* Microcontrollers
* Edge AI
* Embedded robotics
* Resource-constrained inference
* Real-time neural control systems

---

## License

This project is released under the **MIT License**.

See [`LICENSE`](./LICENSE) for details.

---

## Acknowledgements

LiteMish was developed as an exploration of powerful yet smooth and computationally efficient activation functions for neural networks operating under resource constraints.

If you use LiteMish, benchmark it on your hardware, or build on the idea, feel free to open an issue or submit a pull request.

---

## Support the Project

If LiteMish is useful to you, consider giving the repository a **⭐ star** on GitHub.

Stars help the project become more visible to researchers and developers working on:

**Embedded Neural Control · TinyML · Edge AI · Embedded Deep Learning · Activation Functions · Efficient Neural Networks**
