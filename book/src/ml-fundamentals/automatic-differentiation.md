# Automatic Differentiation Theory

Automatic differentiation (autodiff) is the foundation of modern deep learning, enabling efficient computation of gradients for neural network training.

## The Differentiation Landscape

| Method | Accuracy | Speed | Scalability |
|--------|----------|-------|-------------|
| Manual | Exact | Fast | Poor (error-prone) |
| Symbolic | Exact | Slow | Poor (expression swell) |
| Numerical | Approximate | Slow | Moderate |
| **Automatic** | **Exact** | **Fast** | **Excellent** |

## Forward vs Reverse Mode

### Forward Mode (Tangent)

Computes derivatives alongside values:

```
For f: ℝⁿ → ℝᵐ
Forward mode computes one column of the Jacobian per pass.
Cost: O(n) passes for full Jacobian
Best when: n << m (few inputs, many outputs)
```

**Example:** Computing d/dx of f(x) = x² + 2x

```
Forward pass with tangent ẋ = 1:
  f = x²    → ḟ = 2x·ẋ = 2x
  g = 2x    → ġ = 2·ẋ = 2
  h = f + g → ḣ = ḟ + ġ = 2x + 2 ✓
```

### Reverse Mode (Adjoint / Backpropagation)

Computes gradients backwards from output:

```
For f: ℝⁿ → ℝᵐ
Reverse mode computes one row of the Jacobian per pass.
Cost: O(m) passes for full Jacobian
Best when: n >> m (many inputs, few outputs)
```

**Why reverse mode dominates deep learning:**
- Neural networks: millions of parameters (n), scalar loss (m=1)
- One backward pass computes all gradients!

## Computational Graph

Operations form a directed acyclic graph (DAG):

```
     x       w
     │       │
     ▼       ▼
   ┌───────────┐
   │  multiply │
   └─────┬─────┘
         │
         ▼ z = x·w
   ┌───────────┐
   │    sum    │
   └─────┬─────┘
         │
         ▼ L = Σz
```

### Forward Pass

Values flow forward through the graph, with operations recorded on a **tape**.

### Backward Pass

Gradients flow backward via the **chain rule**:

```
∂L/∂x = ∂L/∂z · ∂z/∂x
```

## Chain Rule Mechanics

For composed functions f(g(x)):

```
df/dx = df/dg · dg/dx
```

In neural networks with layers h₁, h₂, ..., hₙ:

```
∂L/∂W₁ = ∂L/∂hₙ · ∂hₙ/∂hₙ₋₁ · ... · ∂h₂/∂h₁ · ∂h₁/∂W₁
```

## Common Operation Gradients

### Element-wise Operations

| Operation | Forward | Backward (∂L/∂x) |
|-----------|---------|------------------|
| y = x + c | y = x + c | ∂L/∂y |
| y = x · c | y = x · c | c · ∂L/∂y |
| y = x² | y = x² | 2x · ∂L/∂y |
| y = eˣ | y = eˣ | eˣ · ∂L/∂y |
| y = log(x) | y = log(x) | (1/x) · ∂L/∂y |
| y = √x | y = √x | (1/2√x) · ∂L/∂y |

### Binary Operations

| Operation | ∂L/∂x | ∂L/∂y |
|-----------|-------|-------|
| z = x + y | ∂L/∂z | ∂L/∂z |
| z = x - y | ∂L/∂z | -∂L/∂z |
| z = x · y | y · ∂L/∂z | x · ∂L/∂z |
| z = x / y | (1/y) · ∂L/∂z | (-x/y²) · ∂L/∂z |

### Activation Functions

| Activation | Forward | Backward |
|------------|---------|----------|
| ReLU | max(0, x) | 1 if x > 0, else 0 |
| Sigmoid | σ(x) = 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) |
| Tanh | tanh(x) | 1 - tanh²(x) |
| GELU | x·Φ(x) | Φ(x) + x·φ(x) |
| Softmax | eˣⁱ/Σeˣʲ | softmax(x)·(δᵢⱼ - softmax(x)) |

### Reduction Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| sum(x) | Σxᵢ | ones_like(x) · ∂L/∂y |
| mean(x) | Σxᵢ/n | (1/n) · ones_like(x) · ∂L/∂y |
| max(x) | maxᵢ xᵢ | indicator(xᵢ = max) · ∂L/∂y |

### Matrix Operations

**Matrix multiply (C = A @ B):**
```
∂L/∂A = ∂L/∂C @ Bᵀ
∂L/∂B = Aᵀ @ ∂L/∂C
```

**Transpose (Bᵀ):**
```
∂L/∂B = (∂L/∂Bᵀ)ᵀ
```

## Tape-Based Implementation

### Define-by-Run (Dynamic Graph)

Operations recorded as they execute:

```rust
// Each operation adds to the tape
let z = x.mul(&w);  // Tape: [MulBackward]
let y = z.sum();    // Tape: [MulBackward, SumBackward]

// Backward traverses tape in reverse
y.backward();       // Process: SumBackward → MulBackward
```

**Advantages:**
- Debugging-friendly (can print tensors mid-forward)
- Supports control flow (if/loops) naturally
- Used by: PyTorch, Aprender

### Define-and-Run (Static Graph)

Graph defined before execution:

```python
# Define graph
x = placeholder()
y = x @ w + b

# Then run
session.run(y, feed_dict={x: data})
```

**Advantages:**
- Whole-graph optimization
- Better for deployment
- Used by: TensorFlow 1.x, JAX (XLA)

## Gradient Accumulation

When a tensor is used multiple times:

```
     x
    / \
   f   g
    \ /
     h
     |
     L
```

Gradients must be **summed**:

```
∂L/∂x = ∂L/∂f · ∂f/∂x + ∂L/∂g · ∂g/∂x
```

## No-Grad Context

Disable gradient tracking for inference:

```rust
let prediction = no_grad(|| {
    model.forward(&input)
});
// No tape recorded, no gradients computed
```

Benefits:
- Memory savings (no tape storage)
- Faster execution
- Required for validation/inference

## Numerical Stability

### Gradient Clipping

Prevent exploding gradients:

```
if ||∇L|| > threshold:
    ∇L = threshold · ∇L / ||∇L||
```

### Log-Sum-Exp Trick

For softmax with large values:

```
log(Σeˣⁱ) = max(x) + log(Σe^(xᵢ-max(x)))
```

Prevents overflow while maintaining gradients.

## Memory Optimization

### Checkpointing (Gradient Checkpointing)

Trade compute for memory:

1. Only save activations at checkpoints
2. Recompute intermediate values during backward
3. Reduces memory from O(n) to O(√n)

### In-Place Operations

Modify tensors directly (use with caution):

```rust
// Careful: invalidates any computation graph using x
x.add_(&y);  // x = x + y in-place
```

## References

- Baydin, A. G., et al. (2018). "Automatic differentiation in machine learning: a survey." JMLR.
- Rumelhart, D. E., et al. (1986). "Learning representations by back-propagating errors." Nature.
- Griewank, A., & Walther, A. (2008). "Evaluating derivatives." SIAM.
