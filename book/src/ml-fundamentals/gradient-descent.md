# Gradient Descent Theory

Gradient descent is the fundamental optimization algorithm used to train machine learning models. It iteratively adjusts model parameters to minimize a loss function by following the direction of steepest descent.

## Mathematical Foundation

### The Core Idea

Given a differentiable loss function **L(θ)** where **θ** represents model parameters, gradient descent finds parameters that minimize the loss:

```
θ* = argmin L(θ)
       θ
```

The algorithm works by repeatedly taking steps proportional to the **negative gradient** of the loss function:

```
θ(t+1) = θ(t) - η ∇L(θ(t))
```

Where:
- **θ(t)**: Parameters at iteration t
- **η**: Learning rate (step size)
- **∇L(θ(t))**: Gradient of loss with respect to parameters

### Why the Negative Gradient?

The gradient **∇L(θ)** points in the direction of **steepest ascent** (maximum increase in loss). By moving in the **negative gradient direction**, we move toward the steepest descent (minimum loss).

**Intuition**: Imagine standing on a mountain in thick fog. You can feel the slope beneath your feet but can't see the valley. Gradient descent is like repeatedly taking a step in the direction that slopes most steeply downward.

## Variants of Gradient Descent

### 1. Batch Gradient Descent (BGD)

Computes the gradient using the **entire training dataset**:

```
∇L(θ) = (1/N) Σ ∇L_i(θ)
              i=1..N
```

**Advantages**:
- Stable convergence (smooth trajectory)
- Guaranteed to converge to global minimum (convex functions)
- Theoretical guarantees

**Disadvantages**:
- Slow for large datasets (must process all samples)
- Memory intensive
- May converge to poor local minima (non-convex functions)

**When to use**: Small datasets (N < 10,000), convex optimization problems

### 2. Stochastic Gradient Descent (SGD)

Computes gradient using a **single random sample** at each iteration:

```
∇L(θ) ≈ ∇L_i(θ)    where i ~ Uniform(1..N)
```

**Advantages**:
- Fast updates (one sample per iteration)
- Can escape shallow local minima (noise helps exploration)
- Memory efficient
- Online learning capable

**Disadvantages**:
- Noisy convergence (zig-zagging trajectory)
- May not converge exactly to minimum
- Requires learning rate decay

**When to use**: Large datasets, online learning, non-convex optimization

**aprender implementation**:
```rust
use aprender::optim::SGD;

let mut optimizer = SGD::new(0.01) // learning rate = 0.01
    .with_momentum(0.9);           // momentum coefficient

// In training loop:
let gradients = compute_gradients(&params, &data);
optimizer.step(&mut params, &gradients);
```

### 3. Mini-Batch Gradient Descent

Computes gradient using a **small batch of samples** (typically 32-256):

```
∇L(θ) ≈ (1/B) Σ ∇L_i(θ)    where B << N
             i∈batch
```

**Advantages**:
- Balance between BGD stability and SGD speed
- Vectorized operations (GPU/SIMD acceleration)
- Reduced variance compared to SGD
- Memory efficient

**Disadvantages**:
- Batch size is a hyperparameter to tune
- Still has some noise

**When to use**: Default choice for most ML problems, deep learning

**Batch size guidelines**:
- Small batches (32-64): Better generalization, more noise
- Large batches (128-512): Faster convergence, more stable
- Powers of 2: Better hardware utilization

## The Learning Rate

The learning rate **η** is the **most critical hyperparameter** in gradient descent.

### Too Small Learning Rate

```
η = 0.001 (very small)

Loss over time:
1000 ┤
 900 ┤
 800 ┤
 700 ┤●
 600 ┤ ●
 500 ┤  ●
 400 ┤   ●●●●●●●●●●●●●●●  ← Slow convergence
     └─────────────────────→
        Iterations (10,000)
```

**Problem**: Training is very slow, may not converge within time budget.

### Too Large Learning Rate

```
η = 1.0 (very large)

Loss over time:
1000 ┤    ●
 900 ┤   ● ●
 800 ┤  ●   ●
 700 ┤ ●     ●  ← Oscillation
 600 ┤●       ●
     └──────────→
      Iterations
```

**Problem**: Loss oscillates or diverges, never converges to minimum.

### Optimal Learning Rate

```
η = 0.1 (just right)

Loss over time:
1000 ┤●
 800 ┤ ●●
 600 ┤    ●●●
 400 ┤       ●●●●  ← Smooth, fast convergence
 200 ┤           ●●●●
     └───────────────→
         Iterations
```

**Guidelines for choosing η**:
1. Start with **η = 0.1** and adjust by factors of 10
2. Use **learning rate schedules** (decay over time)
3. Monitor loss: if exploding → reduce η; if stagnating → increase η
4. Try **adaptive methods** (Adam, RMSprop) that auto-tune η

## Convergence Analysis

### Convex Functions

For **convex loss functions** (e.g., linear regression with MSE), gradient descent with fixed learning rate converges to the **global minimum**:

```
L(θ(t)) - L(θ*) ≤ C / t
```

Where **C** is a constant. The gap to the optimal loss decreases as **1/t**.

**Convergence rate**: O(1/t) for fixed learning rate

### Non-Convex Functions

For **non-convex functions** (e.g., neural networks), gradient descent may converge to:
- Local minimum
- Saddle point
- Plateau region

**No guarantees** of finding the global minimum, but SGD's noise helps escape poor local minima.

### Stopping Criteria

**When to stop iterating?**

1. **Gradient magnitude**: Stop when ||∇L(θ)|| < ε
   - ε = 1e-4 typical threshold

2. **Loss change**: Stop when |L(t) - L(t-1)| < ε
   - Measures improvement per iteration

3. **Maximum iterations**: Stop after T iterations
   - Prevents infinite loops

4. **Validation loss**: Stop when validation loss stops improving
   - Prevents overfitting

**aprender example**:
```rust
// SGD with convergence monitoring
let mut optimizer = SGD::new(0.01);
let mut prev_loss = f32::INFINITY;
let tolerance = 1e-4;

for epoch in 0..max_epochs {
    let loss = compute_loss(&model, &data);

    // Check convergence
    if (prev_loss - loss).abs() < tolerance {
        println!("Converged at epoch {}", epoch);
        break;
    }

    let gradients = compute_gradients(&model, &data);
    optimizer.step(&mut model.params, &gradients);
    prev_loss = loss;
}
```

## Common Pitfalls and Solutions

### 1. Exploding Gradients

**Problem**: Gradients become very large, causing parameters to explode.

**Symptoms**:
- Loss becomes NaN or infinity
- Parameters grow to extreme values
- Occurs in deep networks or RNNs

**Solutions**:
- Reduce learning rate
- Gradient clipping: `g = min(g, threshold)`
- Use batch normalization
- Better initialization (Xavier, He)

### 2. Vanishing Gradients

**Problem**: Gradients become very small, preventing parameter updates.

**Symptoms**:
- Loss stops decreasing but hasn't converged
- Parameters barely change
- Occurs in very deep networks

**Solutions**:
- Use ReLU activation (instead of sigmoid/tanh)
- Skip connections (ResNet architecture)
- Batch normalization
- Better initialization

### 3. Learning Rate Decay

**Strategy**: Start with large learning rate, gradually decrease it.

**Common schedules**:

```rust
// 1. Step decay: Reduce by factor every K epochs
η(t) = η₀ × 0.1^(floor(t / K))

// 2. Exponential decay: Smooth reduction
η(t) = η₀ × e^(-λt)

// 3. 1/t decay: Theoretical convergence guarantee
η(t) = η₀ / (1 + λt)

// 4. Cosine annealing: Cyclical with restarts
η(t) = η_min + 0.5(η_max - η_min)(1 + cos(πt/T))
```

**aprender pattern** (manual implementation):
```rust
let initial_lr = 0.1;
let decay_rate = 0.95;

for epoch in 0..num_epochs {
    let lr = initial_lr * decay_rate.powi(epoch as i32);
    let mut optimizer = SGD::new(lr);

    // Training step
    optimizer.step(&mut params, &gradients);
}
```

### 4. Saddle Points

**Problem**: Gradient is zero but point is not a minimum.

```
Surface shape at saddle point:
    ╱╲    (upward in one direction)
   ╱  ╲
  ╱    ╲
 ╱______╲  (downward in another)
```

**Solutions**:
- Add momentum (helps escape saddle points)
- Use SGD noise to explore
- Second-order methods (Newton, L-BFGS)

## Momentum Enhancement

Standard SGD can be slow in regions with:
- High curvature (steep in some directions, flat in others)
- Noisy gradients

**Momentum** accelerates convergence by accumulating past gradients:

```
v(t) = βv(t-1) + ∇L(θ(t))      // Velocity accumulation
θ(t+1) = θ(t) - η v(t)          // Update with velocity
```

Where **β ∈ [0, 1]** is the momentum coefficient (typically 0.9).

**Effect**:
- Smooths out noisy gradients
- Accelerates in consistent directions
- Dampens oscillations

**Analogy**: A ball rolling down a hill builds momentum, doesn't stop at small bumps.

**aprender implementation**:
```rust
let mut optimizer = SGD::new(0.01)
    .with_momentum(0.9);  // β = 0.9

// Momentum is applied automatically in step()
optimizer.step(&mut params, &gradients);
```

## Practical Guidelines

### Choosing Gradient Descent Variant

| Dataset Size | Recommendation | Reason |
|-------------|----------------|---------|
| N < 1,000 | Batch GD | Fast enough, stable convergence |
| N = 1K-100K | Mini-batch GD (32-128) | Good balance |
| N > 100K | Mini-batch GD (128-512) | Leverage vectorization |
| Streaming data | SGD | Online learning required |

### Hyperparameter Tuning Checklist

1. **Learning rate η**:
   - Start: 0.1
   - Grid search: [0.001, 0.01, 0.1, 1.0]
   - Use learning rate finder

2. **Momentum β**:
   - Default: 0.9
   - Range: [0.5, 0.9, 0.99]

3. **Batch size B**:
   - Default: 32 or 64
   - Range: [16, 32, 64, 128, 256]
   - Powers of 2 for hardware efficiency

4. **Learning rate schedule**:
   - Option 1: Fixed (simple baseline)
   - Option 2: Step decay every 10-30 epochs
   - Option 3: Cosine annealing (state-of-the-art)

### Debugging Convergence Issues

**Loss increasing**: Learning rate too large
→ Reduce η by 10x

**Loss stagnating**: Learning rate too small or stuck in local minimum
→ Increase η by 2x or add momentum

**Loss NaN**: Exploding gradients
→ Reduce η, clip gradients, check data preprocessing

**Slow convergence**: Poor learning rate or no momentum
→ Use adaptive optimizer (Adam), add momentum

## Connection to aprender

The `aprender::optim::SGD` implementation provides:

```rust
use aprender::optim::{SGD, Optimizer};

// Create SGD optimizer
let mut sgd = SGD::new(learning_rate)
    .with_momentum(momentum_coef);

// In training loop:
for epoch in 0..num_epochs {
    for batch in data_loader {
        // 1. Forward pass
        let predictions = model.predict(&batch.x);

        // 2. Compute loss and gradients
        let loss = loss_fn(predictions, batch.y);
        let grads = compute_gradients(&model, &batch);

        // 3. Update parameters using gradient descent
        sgd.step(&mut model.params, &grads);
    }
}
```

**Key methods**:
- `SGD::new(η)`: Create optimizer with learning rate
- `with_momentum(β)`: Add momentum coefficient
- `step(&mut params, &grads)`: Perform one gradient descent step
- `reset()`: Reset momentum buffers

## Further Reading

- **Theory**: Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
- **Momentum**: Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods"
- **Adaptive methods**: See [Advanced Optimizers Theory](./advanced-optimizers.md)

## Related Examples

- [Optimizer Demo](../examples/optimizer-demo.md) - Visualizing SGD with momentum
- [Logistic Regression](../examples/logistic-regression.md) - SGD for classification
- [Regularized Regression](../examples/regularized-regression.md) - Coordinate descent vs SGD

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Core algorithm** | θ(t+1) = θ(t) - η ∇L(θ(t)) |
| **Learning rate** | Most critical hyperparameter; start with 0.1 |
| **Variants** | Batch (stable), SGD (fast), Mini-batch (best of both) |
| **Momentum** | Accelerates convergence, smooths gradients |
| **Convergence** | Guaranteed for convex functions with proper η |
| **Debugging** | Loss ↑ → reduce η; Loss flat → increase η or add momentum |

Gradient descent is the workhorse of machine learning optimization. Understanding its variants, hyperparameters, and convergence properties is essential for training effective models.
