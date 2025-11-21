# Advanced Optimizers Theory

Modern optimizers go beyond vanilla gradient descent by adapting learning rates, incorporating momentum, and using gradient statistics to achieve faster and more stable convergence. This chapter covers state-of-the-art optimization algorithms used in deep learning and machine learning.

## Why Advanced Optimizers?

Standard SGD with momentum works well but has limitations:

1. **Fixed learning rate**: Same η for all parameters
   - Problem: Different parameters may need different learning rates
   - Example: Rare features need larger updates than frequent ones

2. **Manual tuning required**: Finding optimal η is time-consuming
   - Grid search expensive
   - Different datasets need different learning rates

3. **Slow convergence**: Without careful tuning, training can be slow
   - Especially in non-convex landscapes
   - High-dimensional parameter spaces

**Solution**: Adaptive optimizers that automatically adjust learning rates per parameter.

## Optimizer Comparison Table

| Optimizer | Key Feature | Best For | Pros | Cons |
|-----------|-------------|----------|------|------|
| **SGD + Momentum** | Velocity accumulation | General purpose | Simple, well-understood | Requires manual tuning |
| **AdaGrad** | Per-parameter lr | Sparse gradients | Adapts to data | lr decays too aggressively |
| **RMSprop** | Exponential moving average | RNNs, non-stationary | Fixes AdaGrad decay | No bias correction |
| **Adam** | Momentum + RMSprop | Deep learning (default) | Fast, robust | Can overfit on small data |
| **AdamW** | Adam + decoupled weight decay | Transformers | Better generalization | Slightly slower |
| **Nadam** | Adam + Nesterov momentum | Computer vision | Faster convergence | More complex |

## AdaGrad: Adaptive Gradient Algorithm

**Key idea**: Accumulate squared gradients and divide learning rate by their square root, giving smaller updates to frequently updated parameters.

### Algorithm

```
Initialize:
  θ₀ = initial parameters
  G₀ = 0  (accumulated squared gradients)
  η = learning rate (typically 0.01)
  ε = 1e-8 (numerical stability)

For t = 1, 2, ...
  g_t = ∇L(θ_{t-1})             // Compute gradient
  G_t = G_{t-1} + g_t ⊙ g_t      // Accumulate squared gradients
  θ_t = θ_{t-1} - η / √(G_t + ε) ⊙ g_t  // Adaptive update
```

Where **⊙** denotes element-wise multiplication.

### How It Works

**Per-parameter learning rate**:
```
η_i(t) = η / √(Σ(g_i^2) + ε)
                s=1..t
```

- Frequently updated parameters → large accumulated gradient → small effective η
- Infrequently updated parameters → small accumulated gradient → large effective η

### Example

Consider two parameters with gradients:

```
Parameter θ₁: Gradients = [10, 10, 10, 10]  (frequent updates)
Parameter θ₂: Gradients = [1, 0, 0, 1]      (sparse updates)

After 4 iterations (η = 0.1):

θ₁: G = 10² + 10² + 10² + 10² = 400
    Effective η₁ = 0.1 / √400 = 0.1 / 20 = 0.005  (small)

θ₂: G = 1² + 0² + 0² + 1² = 2
    Effective η₂ = 0.1 / √2 = 0.1 / 1.41 ≈ 0.071  (large)
```

**Result**: θ₂ gets ~14x larger updates despite having smaller gradients!

### Advantages

1. **Automatic learning rate adaptation**: No manual tuning per parameter
2. **Great for sparse data**: NLP, recommender systems
3. **Handles different scales**: Features with different ranges

### Disadvantages

1. **Learning rate decay**: Accumulation never decreases
   - Eventually η → 0, stopping learning
   - Problem for deep learning (many iterations)

2. **Requires careful initialization**: Poor initial η can hurt performance

### When to Use

- **Sparse gradients**: NLP (word embeddings), recommender systems
- **Convex optimization**: Guaranteed convergence for convex functions
- **Short training**: If iteration count is small

**Not recommended for**: Deep neural networks (use RMSprop or Adam instead)

## RMSprop: Root Mean Square Propagation

**Key idea**: Fix AdaGrad's aggressive learning rate decay by using exponential moving average instead of sum.

### Algorithm

```
Initialize:
  θ₀ = initial parameters
  v₀ = 0  (moving average of squared gradients)
  η = learning rate (typically 0.001)
  β = decay rate (typically 0.9)
  ε = 1e-8

For t = 1, 2, ...
  g_t = ∇L(θ_{t-1})                    // Compute gradient
  v_t = β·v_{t-1} + (1-β)·(g_t ⊙ g_t)  // Exponential moving average
  θ_t = θ_{t-1} - η / √(v_t + ε) ⊙ g_t  // Adaptive update
```

### Key Difference from AdaGrad

**AdaGrad**: `G_t = G_{t-1} + g_t²` (sum, always increasing)
**RMSprop**: `v_t = β·v_{t-1} + (1-β)·g_t²` (exponential moving average)

The **exponential moving average** forgets old gradients, allowing learning rate to increase again if recent gradients are small.

### Effect of Decay Rate β

```
β = 0.9 (typical):
  - Averages over ~10 iterations
  - Balance between stability and adaptivity

β = 0.99:
  - Averages over ~100 iterations
  - More stable, slower adaptation

β = 0.5:
  - Averages over ~2 iterations
  - Fast adaptation, more noise
```

### Advantages

1. **No learning rate decay problem**: Can train indefinitely
2. **Works well for RNNs**: Handles non-stationary problems
3. **Less sensitive to initialization**: Compared to AdaGrad

### Disadvantages

1. **No bias correction**: Early iterations biased toward 0
2. **Still requires tuning**: η and β hyperparameters

### When to Use

- **RNNs and LSTMs**: Originally designed for this
- **Non-stationary problems**: Changing data distributions
- **Deep learning**: Better than AdaGrad for many epochs

## Adam: Adaptive Moment Estimation

**The most popular optimizer** in modern deep learning. Combines the best ideas from momentum and RMSprop.

### Core Concept

Adam maintains two moving averages:
1. **First moment** (m): Exponential moving average of gradients (momentum)
2. **Second moment** (v): Exponential moving average of squared gradients (RMSprop)

### Algorithm

```
Initialize:
  θ₀ = initial parameters
  m₀ = 0  (first moment: mean gradient)
  v₀ = 0  (second moment: uncentered variance)
  η = 0.001  (learning rate)
  β₁ = 0.9   (exponential decay for first moment)
  β₂ = 0.999 (exponential decay for second moment)
  ε = 1e-8

For t = 1, 2, ...
  g_t = ∇L(θ_{t-1})                     // Gradient

  m_t = β₁·m_{t-1} + (1-β₁)·g_t         // Update first moment
  v_t = β₂·v_{t-1} + (1-β₂)·(g_t ⊙ g_t)  // Update second moment

  m̂_t = m_t / (1 - β₁^t)                // Bias correction for m
  v̂_t = v_t / (1 - β₂^t)                // Bias correction for v

  θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)  // Parameter update
```

### Why Bias Correction?

Initially, m and v are zero. Exponential moving averages are biased toward zero at the start.

**Example** (β₁ = 0.9, g₁ = 1.0):
```
Without bias correction:
  m₁ = 0.9 × 0 + 0.1 × 1.0 = 0.1  (underestimates true mean of 1.0)

With bias correction:
  m̂₁ = 0.1 / (1 - 0.9¹) = 0.1 / 0.1 = 1.0  (correct!)
```

The correction factor `1 - β^t` approaches 1 as t increases, so correction matters most early in training.

### Hyperparameters

**Default values** (from paper, work well in practice):
- **η = 0.001**: Learning rate (most important to tune)
- **β₁ = 0.9**: First moment decay (rarely changed)
- **β₂ = 0.999**: Second moment decay (rarely changed)
- **ε = 1e-8**: Numerical stability

**Tuning guidelines**:
1. Start with defaults
2. If unstable: reduce η to 0.0001
3. If slow: increase η to 0.01
4. Adjust β₁ for more/less momentum (rarely needed)

### aprender Implementation

```rust
use aprender::optim::{Adam, Optimizer};

// Create Adam optimizer with default hyperparameters
let mut adam = Adam::new(0.001)  // learning rate
    .with_beta1(0.9)             // optional: momentum coefficient
    .with_beta2(0.999)           // optional: RMSprop coefficient
    .with_epsilon(1e-8);         // optional: numerical stability

// Training loop
for epoch in 0..num_epochs {
    for batch in data_loader {
        // Forward pass
        let predictions = model.predict(&batch.x);
        let loss = loss_fn(predictions, batch.y);

        // Compute gradients
        let grads = compute_gradients(&model, &batch);

        // Adam step (handles momentum + adaptive lr internally)
        adam.step(&mut model.params, &grads);
    }
}
```

**Key methods**:
- `Adam::new(η)`: Create with learning rate
- `with_beta1(β₁)`, `with_beta2(β₂)`: Set moment decay rates
- `step(&mut params, &grads)`: Perform one update step
- `reset()`: Reset moment buffers (for multiple training runs)

### Advantages

1. **Robust**: Works well with default hyperparameters
2. **Fast convergence**: Combines momentum + adaptive lr
3. **Memory efficient**: Only 2x parameter memory (m and v)
4. **Well-studied**: Extensive empirical validation

### Disadvantages

1. **Can overfit**: On small datasets or with insufficient regularization
2. **Generalization**: Sometimes SGD with momentum generalizes better
3. **Memory overhead**: 2x parameter count

### When to Use

- **Default choice**: For most deep learning problems
- **Fast prototyping**: Converges quickly, minimal tuning
- **Large-scale training**: Handles high-dimensional problems well

**When to avoid**:
- Very small datasets (<1000 samples): Try SGD + momentum
- Need best generalization: Consider SGD with learning rate schedule

## AdamW: Adam with Decoupled Weight Decay

**Problem with Adam**: Weight decay (L2 regularization) interacts badly with adaptive learning rates.

**Solution**: Decouple weight decay from gradient-based optimization.

### Standard Adam with Weight Decay (Wrong)

```
g_t = ∇L(θ_{t-1}) + λ·θ_{t-1}  // Add L2 penalty to gradient
// ... normal Adam update with modified gradient
```

**Problem**: Weight decay gets adapted by second moment estimate, weakening regularization.

### AdamW (Correct)

```
// Normal Adam update (no λ in gradient)
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·(g_t ⊙ g_t)
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

// Separate weight decay step
θ_t = θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ·θ_{t-1})
```

Weight decay applied directly to parameters, not through adaptive learning rates.

### When to Use

- **Transformers**: Essential for BERT, GPT models
- **Large models**: Better generalization on big networks
- **Transfer learning**: Fine-tuning pre-trained models

**Hyperparameters**:
- Same as Adam, plus:
- **λ = 0.01**: Weight decay coefficient (typical)

## Optimizer Selection Guide

### Decision Tree

```
Start
  │
  ├─ Need fast prototyping?
  │    └─ YES → Adam (default: η=0.001)
  │
  ├─ Training RNN/LSTM?
  │    └─ YES → RMSprop (default: η=0.001, β=0.9)
  │
  ├─ Working with transformers?
  │    └─ YES → AdamW (η=0.001, λ=0.01)
  │
  ├─ Sparse gradients (NLP embeddings)?
  │    └─ YES → AdaGrad (η=0.01)
  │
  ├─ Need best generalization?
  │    └─ YES → SGD + momentum (η=0.1, β=0.9) + lr schedule
  │
  └─ Small dataset (<1000 samples)?
       └─ YES → SGD + momentum (less overfitting)
```

### Practical Recommendations

| Task | Recommended Optimizer | Learning Rate | Notes |
|------|----------------------|---------------|-------|
| Image classification (CNN) | Adam or SGD+momentum | 0.001 (Adam), 0.1 (SGD) | SGD often better final accuracy |
| NLP (word embeddings) | AdaGrad or Adam | 0.01 (AdaGrad), 0.001 (Adam) | AdaGrad for sparse features |
| RNN/LSTM | RMSprop or Adam | 0.001 | RMSprop traditional choice |
| Transformers | AdamW | 0.0001-0.001 | Essential for BERT, GPT |
| Small dataset | SGD + momentum | 0.01-0.1 | Less prone to overfitting |
| Reinforcement learning | Adam or RMSprop | 0.0001-0.001 | Non-stationary problem |

## Learning Rate Schedules

Even with adaptive optimizers, **learning rate schedules** improve performance.

### 1. Step Decay

Reduce η by factor every K epochs:

```rust
let initial_lr = 0.001;
let decay_factor = 0.1;
let decay_epochs = 30;

for epoch in 0..num_epochs {
    let lr = initial_lr * decay_factor.powi((epoch / decay_epochs) as i32);
    let mut adam = Adam::new(lr);
    // ... training
}
```

### 2. Exponential Decay

Smooth exponential reduction:

```rust
let initial_lr = 0.001;
let decay_rate = 0.96;

for epoch in 0..num_epochs {
    let lr = initial_lr * decay_rate.powi(epoch as i32);
    let mut adam = Adam::new(lr);
    // ... training
}
```

### 3. Cosine Annealing

Smooth reduction following cosine curve:

```rust
use std::f32::consts::PI;

let lr_max = 0.001;
let lr_min = 0.00001;
let T_max = 100; // periods

for epoch in 0..num_epochs {
    let lr = lr_min + 0.5 * (lr_max - lr_min) *
             (1.0 + f32::cos(PI * (epoch as f32) / (T_max as f32)));
    let mut adam = Adam::new(lr);
    // ... training
}
```

### 4. Warm-up + Decay

Start small, increase, then decay (used in transformers):

```rust
fn learning_rate_schedule(step: usize, d_model: usize, warmup_steps: usize) -> f32 {
    let d_model = d_model as f32;
    let step = step as f32;
    let warmup_steps = warmup_steps as f32;

    let arg1 = 1.0 / step.sqrt();
    let arg2 = step * warmup_steps.powf(-1.5);

    d_model.powf(-0.5) * arg1.min(arg2)
}
```

## Comparison: SGD vs Adam

### When SGD + Momentum is Better

**Advantages**:
- Better final generalization (lower test error)
- Flatter minima (more robust to perturbations)
- Less memory (no moment estimates)

**Requirements**:
- Careful learning rate tuning
- Learning rate schedule essential
- More training time may be needed

### When Adam is Better

**Advantages**:
- Faster initial convergence
- Minimal hyperparameter tuning
- Works across many problem types

**Trade-offs**:
- Can overfit more easily
- May find sharper minima
- Slightly worse generalization on some tasks

### Empirical Rule

**Adam for**:
- Fast prototyping and experimentation
- Baseline models
- Large-scale problems (many parameters)

**SGD + momentum for**:
- Final production models (after tuning)
- When computational budget allows careful tuning
- Small to medium datasets

## Debugging Optimizer Issues

### Loss Not Decreasing

**Possible causes**:
1. Learning rate too small
   - **Fix**: Increase η by 10x
2. Vanishing gradients
   - **Fix**: Check gradient norms, adjust architecture
3. Bug in gradient computation
   - **Fix**: Use gradient checking

### Loss Exploding (NaN)

**Possible causes**:
1. Learning rate too large
   - **Fix**: Reduce η by 10x
2. Gradient explosion
   - **Fix**: Gradient clipping, better initialization

### Slow Convergence

**Possible causes**:
1. Poor learning rate
   - **Fix**: Try different optimizer (Adam if using SGD)
2. No momentum
   - **Fix**: Add momentum (β=0.9)
3. Suboptimal batch size
   - **Fix**: Try 32, 64, 128

### Overfitting

**Possible causes**:
1. Optimizer too aggressive (Adam on small data)
   - **Fix**: Switch to SGD + momentum
2. No regularization
   - **Fix**: Add weight decay (AdamW)

## aprender Optimizer Example

```rust
use aprender::optim::{Adam, SGD, Optimizer};
use aprender::linear_model::LogisticRegression;
use aprender::prelude::*;

// Example: Comparing Adam vs SGD
fn compare_optimizers(x_train: &Matrix<f32>, y_train: &Vector<i32>) {
    // Optimizer 1: Adam (fast convergence)
    let mut model_adam = LogisticRegression::new();
    let mut adam = Adam::new(0.001);

    println!("Training with Adam...");
    for epoch in 0..50 {
        let loss = train_epoch(&mut model_adam, x_train, y_train, &mut adam);
        if epoch % 10 == 0 {
            println!("  Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }

    // Optimizer 2: SGD + momentum (better generalization)
    let mut model_sgd = LogisticRegression::new();
    let mut sgd = SGD::new(0.1).with_momentum(0.9);

    println!("\nTraining with SGD + Momentum...");
    for epoch in 0..50 {
        let loss = train_epoch(&mut model_sgd, x_train, y_train, &mut sgd);
        if epoch % 10 == 0 {
            println!("  Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }
}

fn train_epoch<O: Optimizer>(
    model: &mut LogisticRegression,
    x: &Matrix<f32>,
    y: &Vector<i32>,
    optimizer: &mut O,
) -> f32 {
    // Compute loss and gradients
    let predictions = model.predict_proba(x);
    let loss = compute_cross_entropy_loss(&predictions, y);
    let grads = compute_gradients(model, x, y);

    // Update parameters
    optimizer.step(&mut model.coefficients_mut(), &grads);

    loss
}
```

## Further Reading

**Seminal Papers**:
- **Adam**: Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"
- **AdamW**: Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
- **RMSprop**: Hinton (unpublished, Coursera lecture)
- **AdaGrad**: Duchi et al. (2011). "Adaptive Subgradient Methods"

**Practical Guides**:
- Ruder, S. (2016). "An overview of gradient descent optimization algorithms"
- CS231n Stanford: Optimization notes

## Related Chapters

- [Gradient Descent Theory](./gradient-descent.md) - Foundation for all optimizers
- [Optimizer Demo](../examples/optimizer-demo.md) - Visual comparison of SGD and Adam
- [Regularized Regression](../examples/regularized-regression.md) - Coordinate descent alternative

## Summary

| Optimizer | Core Innovation | When to Use | aprender Support |
|-----------|----------------|-------------|------------------|
| **AdaGrad** | Per-parameter learning rates | Sparse gradients, convex problems | Not yet (v0.5.0) |
| **RMSprop** | Exponential moving average of squared gradients | RNNs, non-stationary | Not yet (v0.5.0) |
| **Adam** | Momentum + RMSprop + bias correction | Default choice, deep learning | ✅ Implemented |
| **AdamW** | Adam + decoupled weight decay | Transformers, large models | Not yet (v0.5.0) |

**Key Takeaways**:
1. **Adam is the default** for most deep learning: fast, robust, minimal tuning
2. **SGD + momentum** often achieves better final accuracy with proper tuning
3. **Learning rate schedules** improve all optimizers
4. **AdamW essential** for training transformers
5. **Monitor convergence**: Loss curves reveal optimizer issues

Modern optimizers dramatically accelerate machine learning by adapting learning rates automatically. Understanding their trade-offs enables choosing the right tool for each problem.
