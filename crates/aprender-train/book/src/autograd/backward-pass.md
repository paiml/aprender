# Backward Pass

The **backward pass** computes gradients by traversing the computational graph in reverse order, applying the chain rule
at each operation. This chapter explains the mechanics of gradient propagation in Entrenar.

## The Chain Rule

The foundation of backpropagation is the **multivariate chain rule**:

```
Given: z = f(y) and y = g(x)
Then:  dz/dx = dz/dy * dy/dx
```

For neural networks with many layers:

```
Loss = f_n(f_{n-1}(...f_2(f_1(x))))

dLoss/dx = dLoss/df_n * df_n/df_{n-1} * ... * df_2/df_1 * df_1/dx
```

Entrenar automates this chain rule application.

## Backward Pass Algorithm

### High-Level Steps

1. **Seed the gradient**: Set output gradient to 1.0
2. **Traverse in reverse**: Process tape entries from end to start
3. **Apply local gradients**: Each operation computes input gradients from output gradient
4. **Accumulate gradients**: Sum contributions when tensors have multiple consumers

### Pseudocode

```python
def backward(output_tensor):
    # Step 1: Seed gradient
    output_tensor.grad = 1.0

    # Step 2: Get tape entries
    tape = get_global_tape()

    # Step 3: Reverse traversal
    for entry in reversed(tape):
        # Get output gradient (already computed)
        grad_output = entry.output.grad

        # Step 4: Compute input gradients (chain rule)
        grad_inputs = entry.operation.backward(grad_output)

        # Step 5: Accumulate into input tensors
        for (input_tensor, grad_input) in zip(entry.inputs, grad_inputs):
            input_tensor.grad += grad_input  # Accumulation!
```

## Operation-Specific Backward Rules

Each operation implements a `backward` method that computes input gradients from output gradients.

### Addition: z = x + y

**Forward**: `z_i = x_i + y_i`

**Backward**:
```
∂z/∂x = 1  (gradient passes through unchanged)
∂z/∂y = 1

Therefore:
∂Loss/∂x = ∂Loss/∂z * 1 = ∂Loss/∂z
∂Loss/∂y = ∂Loss/∂z * 1 = ∂Loss/∂z
```

**Implementation**:
```rust
fn add_backward(grad_output: &[f32], x: &Tensor, y: &Tensor) {
    // Gradient flows equally to both inputs
    x.accumulate_grad(grad_output);  // dx = dz
    y.accumulate_grad(grad_output);  // dy = dz
}
```

### Multiplication: z = x * y

**Forward**: `z_i = x_i * y_i`

**Backward**:
```
∂z/∂x = y  (gradient scaled by other input)
∂z/∂y = x

Therefore:
∂Loss/∂x = ∂Loss/∂z * y
∂Loss/∂y = ∂Loss/∂z * x
```

**Implementation**:
```rust
fn mul_backward(grad_output: &[f32], x: &Tensor, y: &Tensor) {
    // Gradient to x scaled by y's value
    let grad_x: Vec<f32> = grad_output.iter()
        .zip(y.data().iter())
        .map(|(g, y_val)| g * y_val)
        .collect();
    x.accumulate_grad(&grad_x);

    // Gradient to y scaled by x's value
    let grad_y: Vec<f32> = grad_output.iter()
        .zip(x.data().iter())
        .map(|(g, x_val)| g * x_val)
        .collect();
    y.accumulate_grad(&grad_y);
}
```

### Matrix Multiplication: C = A @ B

**Forward**: `C = A @ B` (dimensions: `C[m,n] = A[m,k] @ B[k,n]`)

**Backward**:
```
∂Loss/∂A = ∂Loss/∂C @ B^T
∂Loss/∂B = A^T @ ∂Loss/∂C
```

**Derivation** (element-wise):
```
C[i,j] = Σ_k A[i,k] * B[k,j]

∂C[i,j]/∂A[i,k] = B[k,j]  => ∂Loss/∂A[i,k] = Σ_j ∂Loss/∂C[i,j] * B[k,j]
                                             = (∂Loss/∂C @ B^T)[i,k]

∂C[i,j]/∂B[k,j] = A[i,k]  => ∂Loss/∂B[k,j] = Σ_i ∂Loss/∂C[i,j] * A[i,k]
                                             = (A^T @ ∂Loss/∂C)[k,j]
```

**Implementation**:
```rust
fn matmul_backward(
    grad_output: &Tensor,  // dC
    a: &Tensor,            // A
    b: &Tensor,            // B
    m: usize,              // rows of A
    k: usize,              // cols of A = rows of B
    n: usize,              // cols of B
) {
    // dA = dC @ B^T
    let b_transpose = transpose(b, k, n);
    let grad_a = matmul(grad_output, &b_transpose, m, n, k);
    a.accumulate_grad(grad_a.data());

    // dB = A^T @ dC
    let a_transpose = transpose(a, m, k);
    let grad_b = matmul(&a_transpose, grad_output, k, m, n);
    b.accumulate_grad(grad_b.data());
}
```

### ReLU: y = max(0, x)

**Forward**: `y_i = max(0, x_i)`

**Backward**:
```
∂y/∂x = {1 if x > 0, 0 otherwise}

Therefore:
∂Loss/∂x_i = ∂Loss/∂y_i * (x_i > 0 ? 1 : 0)
```

**Implementation**:
```rust
fn relu_backward(grad_output: &[f32], x: &Tensor) {
    let grad_x: Vec<f32> = grad_output.iter()
        .zip(x.data().iter())
        .map(|(g, &x_val)| {
            if x_val > 0.0 {
                *g  // Gradient passes through
            } else {
                0.0  // Gradient blocked
            }
        })
        .collect();

    x.accumulate_grad(&grad_x);
}
```

### GELU: y = x * Φ(x)

**Forward**: `y = x * Φ(x)` where `Φ` is the Gaussian CDF

**Backward** (using product rule):
```
∂y/∂x = Φ(x) + x * φ(x)

where φ(x) = (1/√(2π)) * exp(-x²/2) is the Gaussian PDF
```

**Implementation**:
```rust
fn gelu_backward(grad_output: &[f32], x: &Tensor) {
    const SQRT_2_PI: f32 = 2.5066282746;  // √(2π)

    let grad_x: Vec<f32> = grad_output.iter()
        .zip(x.data().iter())
        .map(|(g, &x_val)| {
            let phi = gaussian_cdf(x_val);          // Φ(x)
            let phi_prime = (-0.5 * x_val.powi(2)).exp() / SQRT_2_PI;  // φ(x)
            let local_grad = phi + x_val * phi_prime;

            g * local_grad
        })
        .collect();

    x.accumulate_grad(&grad_x);
}
```

### Layer Normalization

**Forward**:
```
y = (x - μ) / σ

where:
  μ = mean(x)
  σ = √(variance(x) + ε)
```

**Backward** (complex chain rule):
```
∂Loss/∂x_i = (1/σ) * [∂Loss/∂y_i - (1/n)Σ_j ∂Loss/∂y_j - (1/n)y_i Σ_j(∂Loss/∂y_j * y_j)]
```

**Implementation**:
```rust
fn layernorm_backward(
    grad_output: &[f32],
    x: &Tensor,
    normalized: &[f32],  // y values from forward pass
    mean: f32,
    variance: f32,
) {
    let n = grad_output.len() as f32;
    let std_inv = 1.0 / (variance + 1e-5).sqrt();

    // Compute sum terms
    let sum_grad: f32 = grad_output.iter().sum();
    let sum_grad_y: f32 = grad_output.iter()
        .zip(normalized.iter())
        .map(|(g, y)| g * y)
        .sum();

    // Compute gradient for each element
    let grad_x: Vec<f32> = grad_output.iter()
        .zip(normalized.iter())
        .map(|(g, y)| {
            std_inv * (g - sum_grad / n - y * sum_grad_y / n)
        })
        .collect();

    x.accumulate_grad(&grad_x);
}
```

## Gradient Accumulation

When a tensor is used multiple times, gradients accumulate:

### Example: z = x + x

```rust
let x = Tensor::from_vec(vec![2.0], true);
let z = &x + &x;  // z = 2x

backward(&z);

println!("dz/dx = {}", x.grad()[0]);  // 2.0 ✅
```

**Why 2.0?**

```
Graph:
    x ─┬─> Add -> z
       └─>

Backward:
  From first input:  dx = dz * 1 = 1.0
  From second input: dx = dz * 1 = 1.0
  Total:             dx = 1.0 + 1.0 = 2.0 ✅
```

**Implementation**:
```rust
// Always use += for gradient accumulation
x.grad_mut()[i] += gradient_contribution;
```

### Complex Example

```rust
let x = Tensor::from_vec(vec![3.0], true);
let y = Tensor::from_vec(vec![4.0], true);

let a = &x + &y;   // a = x + y = 7
let b = &x * &y;   // b = x * y = 12
let c = &a + &b;   // c = a + b = 19

backward(&c);
```

**Gradient computation:**

```
Tape (forward order):
  [0] Add(x, y) -> a
  [1] Mul(x, y) -> b
  [2] Add(a, b) -> c

Backward (reverse order):
  [2] Add: da = dc = 1.0, db = dc = 1.0
  [1] Mul: dx += db * y = 1.0 * 4 = 4.0
           dy += db * x = 1.0 * 3 = 3.0
  [0] Add: dx += da = 1.0
           dy += da = 1.0

Final gradients:
  dx = 4.0 + 1.0 = 5.0  ✅ (= y + 1)
  dy = 3.0 + 1.0 = 4.0  ✅ (= x + 1)
```

**Manual verification:**
```
c = (x + y) + (x * y) = x + y + xy
dc/dx = 1 + y = 1 + 4 = 5.0 ✅
dc/dy = 1 + x = 1 + 3 = 4.0 ✅
```

## Handling Non-Differentiable Points

Some operations have non-differentiable points where we use **subgradients**.

### ReLU at x=0

```
ReLU(x) = max(0, x)

Derivative:
  d/dx ReLU(x) = {1 if x > 0, 0 if x < 0, ??? if x = 0}
```

**Solution**: Use subgradient convention:
```rust
if x_val > 0.0 {
    1.0
} else {
    0.0  // Subgradient at x=0 (could also use 1.0 or 0.5)
}
```

**In practice**: Exact x=0 is rare with floating-point numbers, so the choice rarely matters.

## Detaching Gradients

Sometimes you want to **stop gradients** from flowing:

```rust
let x = Tensor::from_vec(vec![2.0], true);
let y = &x * &x;  // y = x²

// Detach: treat y as a constant for further operations
let y_detached = Tensor::from_vec(y.data().clone(), false);  // requires_grad=false

let z = &y_detached + &x;  // z = y_detached + x (y treated as constant)

backward(&z);

println!("dz/dx = {}", x.grad()[0]);  // 1.0 (only from addition, not from y)
```

**Use case**: Stopping gradient flow in certain model parts (e.g., frozen layers).

## In-Place Operations Warning

**In-place modifications break the computational graph:**

```rust
let mut x = Tensor::from_vec(vec![1.0, 2.0], true);
let y = &x * &x;

// ❌ BAD: Modify x in-place
x.data_mut()[0] = 5.0;

backward(&y);  // ⚠️ Undefined behavior! x changed after being used
```

**Solution**: Entrenar prevents in-place modifications for tensors with `requires_grad=true`:

```rust
// Entrenar's safeguard
if x.requires_grad() {
    panic!("Cannot modify tensor with requires_grad=true in-place");
}
```

## Computational Complexity

| Operation | Forward | Backward | Total |
|-----------|---------|----------|-------|
| **Add/Mul** | O(n) | O(n) | O(n) |
| **MatMul** | O(mnk) | O(mnk) | O(mnk) |
| **ReLU** | O(n) | O(n) | O(n) |
| **LayerNorm** | O(n) | O(n) | O(n) |
| **Attention** | O(n²d) | O(n²d) | O(n²d) |

**Key insight**: Backward pass has same asymptotic complexity as forward pass.

## Debugging Gradients

### Check if Gradients are Computed

```rust
let x = Tensor::from_vec(vec![2.0], true);
let y = &x * &x;

backward(&y);

if x.grad()[0] == 0.0 {
    eprintln!("Warning: Gradient is zero (might indicate issue)");
}
```

### Gradient Explosion/Vanishing

```rust
fn check_gradients(params: &[&Tensor]) {
    for param in params {
        let grad_norm = param.grad().iter().map(|g| g * g).sum::<f32>().sqrt();

        if grad_norm > 100.0 {
            eprintln!("Warning: Gradient explosion (norm={})", grad_norm);
        } else if grad_norm < 1e-7 {
            eprintln!("Warning: Gradient vanishing (norm={})", grad_norm);
        }
    }
}
```

### Gradient Checking

Always validate custom operations with finite differences:

```rust
#[test]
fn test_my_operation_backward() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let y = my_custom_operation(&x);

    backward(&y);

    // Compare with numerical gradient
    check_gradient(&y, &x, epsilon=1e-3, threshold=0.2);
}
```

## Key Takeaways

1. **Backward pass applies chain rule** in reverse topological order
2. **Each operation implements local gradient rule** (e.g., mul: dx = y*dz)
3. **Gradients accumulate** when tensors have multiple consumers
4. **Matrix operations** use transposition for gradient computation
5. **Nonlinear activations** use derivative of activation function
6. **Normalization** requires saved statistics from forward pass
7. **Complexity** of backward equals forward (asymptotically)

## What's Next?

- **[Gradient Computation](./gradient-computation.md)** - Mathematical derivations
- **[Finite Difference Validation](./finite-difference.md)** - Testing gradients
- **[Tensor Operations](./tensor-operations.md)** - All supported operations

---

**Ready to dive into the math?** Continue to [Gradient Computation](./gradient-computation.md) →
