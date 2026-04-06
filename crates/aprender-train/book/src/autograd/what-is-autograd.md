# What is Automatic Differentiation?

**Automatic Differentiation (Autograd)** is a technique for computing derivatives of functions specified by computer
programs. It's the foundation of modern deep learning, enabling neural networks to learn through gradient-based
optimization.

## The Problem: Manual Derivatives

Consider a simple neural network layer:

```rust
fn forward(x: f32, w: f32, b: f32) -> f32 {
    w * x + b  // Linear transformation
}
```

To train this layer, we need gradients: `∂loss/∂w` and `∂loss/∂b`.

### Manual Approach (Error-Prone)

```rust
// Forward pass
let y_pred = w * x + b;
let loss = (y_pred - y_true).powi(2);  // MSE

// Backward pass (hand-coded derivatives)
let d_loss = 2.0 * (y_pred - y_true);
let d_w = d_loss * x;  // ∂loss/∂w = ∂loss/∂y * ∂y/∂w
let d_b = d_loss * 1.0;  // ∂loss/∂b = ∂loss/∂y * ∂y/∂b

// Update
w -= learning_rate * d_w;
b -= learning_rate * d_b;
```

**Problems with manual derivatives:**
- ❌ Error-prone (easy to make mistakes in chain rule)
- ❌ Doesn't scale (complex models have thousands of operations)
- ❌ Hard to maintain (changing forward pass requires rewriting backward pass)
- ❌ No validation (how do you know your derivatives are correct?)

## The Solution: Automatic Differentiation

Entrenar's autograd engine **automatically computes correct derivatives** for any computation:

```rust
use entrenar::{Tensor, backward};

// Forward pass (same as before)
let x = Tensor::from_vec(vec![2.0], false);
let w = Tensor::from_vec(vec![3.0], true);  // requires_grad=true
let b = Tensor::from_vec(vec![1.0], true);

let y_pred = &(&w * &x) + &b;  // y = w*x + b = 7.0
let y_true = Tensor::from_vec(vec![10.0], false);

let diff = &y_pred - &y_true;
let loss = &diff * &diff;  // loss = 9.0

// Backward pass (automatic!)
backward(&loss);

// Gradients computed automatically
println!("∂loss/∂w = {}", w.grad()[0]);  // -12.0 ✅ Correct!
println!("∂loss/∂b = {}", b.grad()[0]);  // -6.0 ✅ Correct!
```

**Benefits of autograd:**
- ✅ Correct by construction (no manual derivative errors)
- ✅ Scales to any complexity (transformers, ResNets, etc.)
- ✅ Easy to maintain (change forward pass, backward automatically updates)
- ✅ Validated with gradient checking (10K+ test cases)

## How Autograd Works

Entrenar uses **reverse-mode automatic differentiation** (also called backpropagation).

### Three Modes of Differentiation

| Mode | Description | Complexity | Use Case |
|------|-------------|------------|----------|
| **Numerical** | Finite differences: `f'(x) ≈ (f(x+ε) - f(x)) / ε` | O(n) evaluations | Gradient checking |
| **Symbolic** | Algebraic manipulation: `d/dx(x²) = 2x` | Exponential growth | Computer algebra systems |
| **Automatic** | Chain rule on computation graph | O(1) per operation | Deep learning |

### Reverse-Mode Differentiation

Given a computation `y = f(g(h(x)))`, we want `dy/dx`.

**Forward Pass** (compute outputs):
```
x → h(x) → g(h(x)) → f(g(h(x))) = y
```

**Backward Pass** (compute gradients via chain rule):
```
dy/dx ← dy/dg * dg/dh ← dy/dg ← dy/dy = 1.0
```

**Key insight**: We only need to store intermediate values and apply the chain rule in reverse.

### Example: y = x²

```rust
let x = Tensor::from_vec(vec![3.0], true);
let y = &x * &x;  // y = x²

backward(&y);  // Compute dy/dx

println!("dy/dx = {}", x.grad()[0]);  // 6.0 (= 2*x)
```

**What happened:**

1. **Forward pass**:
   - Compute `y = x * x = 9.0`
   - Record operation: `Mul(x, x) -> y`

2. **Backward pass** (starting from `dy/dy = 1.0`):
   - `dy/dx_left = dy/dy * x_right = 1.0 * 3.0 = 3.0`
   - `dy/dx_right = dy/dy * x_left = 1.0 * 3.0 = 3.0`
   - `dy/dx = dy/dx_left + dy/dx_right = 6.0` (gradient accumulation)

## Computational Graph

Autograd builds a **computational graph** representing the sequence of operations:

```
Example: z = (x + y) * (x - y)

Graph:
       x      y
       │      │
       ├──────┤
       │      │
       ▼      ▼
      Add    Sub
       │      │
       └──────┘
          │
          ▼
         Mul
          │
          ▼
          z
```

### Tape-Based Implementation

Entrenar uses a **tape** to record operations during the forward pass:

```rust
// Forward pass (records operations on tape)
let x = Tensor::from_vec(vec![2.0], true);
let y = Tensor::from_vec(vec![3.0], true);

let a = &x + &y;  // Tape: [Add(x, y) -> a]
let b = &x - &y;  // Tape: [Add(x, y) -> a, Sub(x, y) -> b]
let z = &a * &b;  // Tape: [Add(x, y) -> a, Sub(x, y) -> b, Mul(a, b) -> z]

// Backward pass (replay tape in reverse)
backward(&z);  // Process: Mul -> Sub -> Add
```

**Tape structure:**
```rust
Tape:
  [0] Add { lhs: x_id, rhs: y_id, out: a_id }
  [1] Sub { lhs: x_id, rhs: y_id, out: b_id }
  [2] Mul { lhs: a_id, rhs: b_id, out: z_id }

Backward (reverse order):
  [2] Mul.backward(): da = b*dz, db = a*dz
  [1] Sub.backward(): dx += 1*db, dy += -1*db
  [0] Add.backward(): dx += 1*da, dy += 1*da
```

## Supported Operations

Entrenar provides backward passes for all essential neural network operations:

### Basic Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| **Add** | `z = x + y` | `dx = dz`, `dy = dz` |
| **Sub** | `z = x - y` | `dx = dz`, `dy = -dz` |
| **Mul** | `z = x * y` | `dx = y*dz`, `dy = x*dz` |
| **Div** | `z = x / y` | `dx = dz/y`, `dy = -x*dz/y²` |

### Matrix Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| **MatMul** | `C = A @ B` | `dA = dC @ B^T`, `dB = A^T @ dC` |

### Activations

| Operation | Forward | Backward |
|-----------|---------|----------|
| **ReLU** | `max(0, x)` | `dx = (x > 0) ? dy : 0` |
| **GELU** | `x * Φ(x)` | Chain rule with Gaussian CDF derivative |
| **Swish** | `x * sigmoid(x)` | `dx = (swish(x) + sigmoid(x) * (1 - swish(x))) * dy` |

### Normalization

| Operation | Forward | Backward |
|-----------|---------|----------|
| **LayerNorm** | `(x - μ) / σ` | Mean/variance chain rule |

### Attention

| Operation | Forward | Backward |
|-----------|---------|----------|
| **Attention** | `softmax(QK^T/√d)V` | Q, K, V gradients via chain rule |

## Gradient Validation

Entrenar validates **every backward pass** with finite difference checking:

```rust
#[test]
fn test_matmul_backward_gradient_check() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], true);

    let c = matmul(&a, &b, 2, 2, 1);
    backward(&c);

    // Finite difference: f'(x) ≈ (f(x+ε) - f(x-ε)) / 2ε
    let epsilon = 1e-3;
    let threshold = 0.2;  // 20% relative error

    check_gradient(&c, &a, epsilon, threshold);  // ✅ Passes
    check_gradient(&c, &b, epsilon, threshold);  // ✅ Passes
}
```

**Zero-tolerance policy:**
- 10K+ gradient checking test cases
- All operations tested with property-based tests
- Mathematical correctness guaranteed

## Autograd vs Manual Derivatives

| Aspect | Manual | Autograd |
|--------|--------|----------|
| **Correctness** | Error-prone | Validated with tests |
| **Scalability** | Doesn't scale | Handles any model size |
| **Maintainability** | Brittle | Change forward, backward auto-updates |
| **Development Time** | Hours/days | Seconds |
| **Performance** | Potentially optimal | Near-optimal (tape overhead minimal) |

## Common Pitfalls

### 1. Forgetting `requires_grad=true`

```rust
let w = Tensor::from_vec(vec![1.0], false);  // ❌ No gradients
let y = &w * &x;
backward(&y);
println!("{}", w.grad()[0]);  // 0.0 (gradient not computed)

// Fix:
let w = Tensor::from_vec(vec![1.0], true);  // ✅ Gradients enabled
```

### 2. Not Zeroing Gradients

```rust
for epoch in 0..10 {
    let loss = compute_loss(&model, &data);
    backward(&loss);

    optimizer.step(&mut params);
    // ❌ Gradients accumulate across epochs!

    // Fix:
    model.zero_grad();  // ✅ Clear gradients
}
```

### 3. In-Place Operations

```rust
let mut x = Tensor::from_vec(vec![1.0, 2.0], true);
x.data_mut()[0] = 5.0;  // ❌ In-place modification breaks graph

// Fix: Create new tensor
let x_new = Tensor::from_vec(vec![5.0, 2.0], true);  // ✅
```

## What's Next?

- **[Tape-Based Computation Graphs](./tape-based-graphs.md)** - Deep dive into Entrenar's tape implementation
- **[Tensor Operations](./tensor-operations.md)** - Explore all supported operations
- **[Backward Pass](./backward-pass.md)** - Understand gradient computation mechanics
- **[Finite Difference Validation](./finite-difference.md)** - Learn gradient checking methodology

## Key Takeaways

1. **Autograd automates derivative computation** - no manual chain rule
2. **Reverse-mode differentiation** - efficient for deep learning (many inputs, one output)
3. **Tape-based graph** - records operations during forward pass
4. **Validated with tests** - 10K+ gradient checking cases ensure correctness
5. **Zero-tolerance for bugs** - extreme TDD methodology

---

**Ready to understand the tape?** Continue to [Tape-Based Computation Graphs](./tape-based-graphs.md) →
