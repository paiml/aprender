# Trueno: NumPy-like Compute Primitives Specification

**Version**: 2.0
**Date**: 2025-12-16
**Status**: Living Document

---

## Executive Summary

Trueno is a high-performance compute library providing **NumPy-like primitives** for Rust. It is NOT a machine learning framework and does NOT include autograd or training capabilities.

**Trueno's Role in the Ecosystem**:
- **Trueno** = NumPy equivalent (compute primitives: vectors, matrices, SIMD, GPU acceleration)
- **Aprender** = sklearn/PyTorch equivalent (ML algorithms, neural networks, autograd, training)

Trueno serves as the **backend compute engine** for higher-level ML libraries like aprender, similar to how NumPy serves as the backend for scikit-learn and PyTorch.

---

## 1. Ecosystem Positioning

### 1.1 What Trueno IS

Trueno is a **compute primitives library** providing:

- **Vector Operations**: Element-wise arithmetic, dot products, norms, reductions
- **Matrix Operations**: Matrix multiplication, transpose, eigendecomposition
- **Activation Functions**: ReLU, GELU, sigmoid, tanh, softmax (forward pass only)
- **SIMD Acceleration**: SSE2, AVX, AVX2, AVX-512, NEON, WASM SIMD128
- **GPU Acceleration**: wgpu/CUDA for large matrices (via trueno-gpu)

```rust
use trueno::{Vector, Matrix, SymmetricEigen};

// Vector operations (NumPy-like)
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let sum = a.add(&b).unwrap();           // [6.0, 8.0, 10.0, 12.0]
let dot = a.dot(&b).unwrap();           // 70.0

// Matrix operations
let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
let product = m.matmul(&m).unwrap();    // Matrix multiplication

// Eigendecomposition
let cov = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).unwrap();
let eigen = SymmetricEigen::new(&cov).unwrap();
```

### 1.2 What Trueno is NOT

Trueno does **NOT** include:

- ❌ **Autograd**: No automatic differentiation (use aprender)
- ❌ **Training**: No gradient descent, optimizers, or backpropagation
- ❌ **Neural Network Layers**: No nn::Linear, Conv2d, BatchNorm
- ❌ **Loss Functions**: No CrossEntropyLoss, MSELoss
- ❌ **Model Serialization**: No checkpoint saving/loading (use aprender's .apr format)

**These features belong in aprender, which uses trueno as its backend.**

### 1.3 Comparison Table

| Feature | NumPy | Trueno | PyTorch | Aprender |
|---------|-------|--------|---------|----------|
| Vector/Matrix ops | ✅ | ✅ | ✅ | ✅ (via trueno) |
| SIMD acceleration | ✅ | ✅ | ✅ | ✅ (via trueno) |
| GPU compute | ✅ (CuPy) | ✅ | ✅ | ✅ (via trueno) |
| Autograd | ❌ | ❌ | ✅ | ✅ |
| Neural networks | ❌ | ❌ | ✅ | ✅ |
| Training loops | ❌ | ❌ | ✅ | ✅ |
| Model format | ❌ | ❌ | .pth | .apr |
| ML algorithms | ❌ | ❌ | ❌ | ✅ |

---

## 2. Current Capabilities (v0.8.x)

### 2.1 Vector Operations

| Operation | Status | SIMD | GPU |
|-----------|--------|------|-----|
| add, sub, mul, div | ✅ | ✅ | ❌ |
| dot product | ✅ | ✅ | ❌ |
| sum, mean, variance | ✅ | ✅ | ❌ |
| min, max, argmin, argmax | ✅ | ✅ | ❌ |
| norm_l1, norm_l2, normalize | ✅ | ✅ | ❌ |

### 2.2 Matrix Operations

| Operation | Status | SIMD | GPU |
|-----------|--------|------|-----|
| matmul | ✅ | ✅ | ✅ |
| transpose | ✅ | ✅ | ❌ |
| matvec | ✅ | ✅ | ❌ |
| eigendecomposition | ✅ | ✅ | ❌ |
| convolve2d | ✅ | ✅ | ❌ |

### 2.3 Activation Functions (Forward Pass Only)

| Activation | Status | SIMD | GPU |
|------------|--------|------|-----|
| ReLU, Leaky ReLU, ELU | ✅ | ✅ | ❌ |
| Sigmoid, Tanh | ✅ | ✅ | ❌ |
| GELU, Swish | ✅ | ✅ | ❌ |
| Softmax, Log-Softmax | ✅ | ✅ | ❌ |

**Note**: These activations are inference-only (forward pass). For training with gradients, use aprender.

### 2.4 Statistics

| Operation | Status | SIMD |
|-----------|--------|------|
| mean, variance, stddev | ✅ | ✅ |
| covariance, correlation | ✅ | ✅ |
| zscore | ✅ | ✅ |

---

## 3. Architecture: Trueno + Aprender

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                         │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               │               ▼
┌─────────────────────┐       │       ┌─────────────────────┐
│      Aprender       │       │       │    trueno-db        │
│  (ML Framework)     │       │       │ (Analytics Database)│
│  - Neural Networks  │       │       │ - SQL queries       │
│  - Autograd         │       │       │ - Aggregations      │
│  - Training         │       │       │                     │
│  - .apr format      │       │       │                     │
└─────────────────────┘       │       └─────────────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Trueno (Compute)                        │
│  - Vector operations (add, dot, reduce)                     │
│  - Matrix operations (matmul, transpose, eigen)             │
│  - Activation functions (relu, sigmoid, softmax)            │
│  - SIMD backends (SSE2, AVX2, AVX-512, NEON)               │
│  - GPU backend (wgpu, trueno-gpu for CUDA)                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 How Aprender Uses Trueno

Aprender uses trueno as its SIMD-accelerated compute backend:

```rust
// aprender (ML framework) - has autograd
use aprender::{Tensor, nn, optim};

let model = nn::Sequential::new()
    .add(nn::Linear::new(784, 128))
    .add(nn::ReLU)
    .add(nn::Linear::new(128, 10));

let optimizer = optim::Adam::new(model.parameters(), 0.001);

// Training loop with autograd
for batch in dataloader {
    let output = model.forward(&batch.x);
    let loss = nn::cross_entropy(&output, &batch.y);
    loss.backward();  // Autograd computes gradients
    optimizer.step();
}

// Save model in .apr format
model.save("model.apr")?;
```

```rust
// trueno (compute primitives) - no autograd
use trueno::{Vector, Matrix};

// Just compute, no gradients
let hidden = input.matmul(&weights).unwrap();
let activated = hidden.relu().unwrap();
let output = activated.matmul(&weights2).unwrap();
// No backward(), no optimizer - that's aprender's job
```

---

## 4. Roadmap

### Phase 1: Complete (v0.1 - v0.8)
- ✅ Vector operations with SIMD
- ✅ Matrix operations
- ✅ Eigendecomposition
- ✅ GPU matrix multiply
- ✅ Activation functions (forward pass)
- ✅ Statistics operations

### Phase 2: Future Work
- [ ] f16/f64 data types
- [ ] Sparse matrix support
- [ ] Additional GPU operations
- [ ] WASM SIMD128 improvements

**Note**: Autograd, training, and neural network layers are OUT OF SCOPE for trueno. These belong in aprender.

---

## 5. Migration Guide

### From NumPy to Trueno

```python
# NumPy
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.dot(a, b)
```

```rust
// Trueno
use trueno::Vector;
let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let result = a.dot(&b).unwrap();
```

### From PyTorch to Aprender (NOT Trueno)

```python
# PyTorch - has autograd
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0, 6.0]
```

```rust
// Aprender - has autograd (NOT trueno)
use aprender::Tensor;
let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad(true);
let y = x.pow(2.0).sum();
y.backward();
println!("{:?}", x.grad());  // [2.0, 4.0, 6.0]
```

---

## 6. Summary

| Library | Role | Python Equivalent |
|---------|------|-------------------|
| **trueno** | Compute primitives | NumPy |
| **aprender** | ML framework | scikit-learn + PyTorch |
| **trueno-gpu** | GPU kernels | CuPy |
| **trueno-db** | Analytics database | DuckDB |
| **trueno-graph** | Graph algorithms | NetworkX |
| **trueno-rag** | RAG pipeline | LangChain |

Trueno is the **compute foundation** of the Pragmatic AI Labs ecosystem. For machine learning with autograd and training, use aprender which builds on trueno.
