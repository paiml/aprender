# Bayesian Linear Regression

Bayesian Linear Regression extends ordinary least squares (OLS) regression by treating coefficients as random variables with a prior distribution, enabling uncertainty quantification and natural regularization.

## Theory

### Model

$$
y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

Where:
- $y \in \mathbb{R}^n$: target vector
- $X \in \mathbb{R}^{n \times p}$: feature matrix
- $\beta \in \mathbb{R}^p$: coefficient vector
- $\sigma^2$: noise variance

### Conjugate Prior (Normal-Inverse-Gamma)

$$
\begin{aligned}
\beta &\sim \mathcal{N}(\beta_0, \Sigma_0) \\
\sigma^2 &\sim \text{Inv-Gamma}(\alpha, \beta)
\end{aligned}
$$

### Analytical Posterior

With conjugate priors, the posterior has a closed form:

$$
\begin{aligned}
\beta | y, X &\sim \mathcal{N}(\beta_n, \Sigma_n) \\
\text{where:} \\
\Sigma_n &= (\Sigma_0^{-1} + \sigma^{-2} X^T X)^{-1} \\
\beta_n &= \Sigma_n (\Sigma_0^{-1} \beta_0 + \sigma^{-2} X^T y)
\end{aligned}
$$

### Key Properties

1. **Posterior mean**: $\beta_n$ balances prior belief ($\beta_0$) and data evidence ($X^T y$)
2. **Posterior covariance**: $\Sigma_n$ quantifies uncertainty
3. **Weak prior**: As $\Sigma_0 \to \infty$, $\beta_n \to (X^T X)^{-1} X^T y$ (OLS)
4. **Strong prior**: As $\Sigma_0 \to 0$, $\beta_n \to \beta_0$ (ignore data)

## Example: Univariate Regression with Weak Prior

```rust,ignore
use aprender::bayesian::BayesianLinearRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    // Training data: y ≈ 2x + noise
    let x = Matrix::from_vec(10, 1, vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
    ]).unwrap();
    let y = Vector::from_vec(vec![
        2.1, 3.9, 6.2, 8.1, 9.8, 12.3, 13.9, 16.1, 18.2, 20.0
    ]);

    // Create model with weak prior
    let mut model = BayesianLinearRegression::new(1);

    // Fit: compute analytical posterior
    model.fit(&x, &y).unwrap();

    // Posterior estimates
    let beta = model.posterior_mean().unwrap();
    let sigma2 = model.noise_variance().unwrap();

    println!("β (slope): {:.4}", beta[0]);          // ≈ 2.0094
    println!("σ² (noise): {:.4}", sigma2);           // ≈ 0.0251

    // Make predictions
    let x_test = Matrix::from_vec(3, 1, vec![11.0, 12.0, 13.0]).unwrap();
    let predictions = model.predict(&x_test).unwrap();

    println!("Prediction at x=11: {:.2}", predictions[0]);  // ≈ 22.10
    println!("Prediction at x=12: {:.2}", predictions[1]);  // ≈ 24.11
    println!("Prediction at x=13: {:.2}", predictions[2]);  // ≈ 26.12
}
```

**Output:**
```text
β (slope): 2.0094
σ² (noise): 0.0251
Prediction at x=11: 22.10
Prediction at x=12: 24.11
Prediction at x=13: 26.12
```

With a weak prior, the posterior mean is nearly identical to the OLS estimate.

## Example: Informative Prior (Ridge-like Regularization)

```rust,ignore
use aprender::bayesian::BayesianLinearRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    // Small dataset (prone to overfitting)
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let y = Vector::from_vec(vec![2.5, 4.1, 5.8, 8.2, 9.9]);

    // Weak prior model
    let mut weak_model = BayesianLinearRegression::new(1);
    weak_model.fit(&x, &y).unwrap();

    // Informative prior: β ~ N(1.5, 1.0)
    let mut strong_model = BayesianLinearRegression::with_prior(
        1,
        vec![1.5],  // Prior mean: expect slope around 1.5
        1.0,        // Prior precision (variance = 1.0)
        3.0,        // Noise shape
        2.0,        // Noise scale
    ).unwrap();
    strong_model.fit(&x, &y).unwrap();

    let beta_weak = weak_model.posterior_mean().unwrap();
    let beta_strong = strong_model.posterior_mean().unwrap();

    println!("Weak prior:       β = {:.4}", beta_weak[0]);
    println!("Informative prior: β = {:.4}", beta_strong[0]);
}
```

**Output:**
```text
Weak prior:       β = 2.0073
Informative prior: β = 2.0065
```

The informative prior shrinks the coefficient toward the prior mean (1.5), acting as **L2 regularization** (ridge regression).

## Example: Multivariate Regression

```rust,ignore
use aprender::bayesian::BayesianLinearRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    // Two features: y ≈ 2x₁ + 3x₂ + noise
    let x = Matrix::from_vec(8, 2, vec![
        1.0, 1.0,  // row 0
        2.0, 1.0,  // row 1
        3.0, 2.0,  // row 2
        4.0, 2.0,  // row 3
        5.0, 3.0,  // row 4
        6.0, 3.0,  // row 5
        7.0, 4.0,  // row 6
        8.0, 4.0,  // row 7
    ]).unwrap();

    let y = Vector::from_vec(vec![
        5.1, 7.2, 11.9, 14.1, 19.2, 21.0, 25.8, 27.9
    ]);

    // Fit multivariate model
    let mut model = BayesianLinearRegression::new(2);
    model.fit(&x, &y).unwrap();

    let beta = model.posterior_mean().unwrap();
    let sigma2 = model.noise_variance().unwrap();

    println!("β₁: {:.4}", beta[0]);    // ≈ 1.9785
    println!("β₂: {:.4}", beta[1]);    // ≈ 3.0343
    println!("σ²: {:.4}", sigma2);     // ≈ 0.0262

    // Predictions
    let x_test = Matrix::from_vec(3, 2, vec![
        9.0, 5.0,   // Expected: 2*9 + 3*5 = 33
        10.0, 5.0,  // Expected: 2*10 + 3*5 = 35
        10.0, 6.0,  // Expected: 2*10 + 3*6 = 38
    ]).unwrap();

    let predictions = model.predict(&x_test).unwrap();
    for i in 0..3 {
        println!("Prediction {}: {:.2}", i, predictions[i]);
    }
}
```

**Output:**
```text
β₁: 1.9785
β₂: 3.0343
σ²: 0.0262
Prediction 0: 32.98
Prediction 1: 34.96
Prediction 2: 37.99
```

## Comparison: Bayesian vs. OLS

| Aspect | Bayesian Linear Regression | OLS Regression |
|--------|----------------------------|----------------|
| **Output** | Posterior distribution over β | Point estimate β̂ |
| **Uncertainty** | Full posterior covariance Σₙ | Standard errors (requires additional computation) |
| **Regularization** | Natural via prior (e.g., ridge) | Requires explicit penalty term |
| **Interpretation** | Probability statements: P(β ∈ [a, b] \| data) | Frequentist confidence intervals |
| **Computation** | Analytical (conjugate case) | Analytical (normal equations) |
| **Small Data** | Regularizes via prior | May overfit |

## Implementation Details

### Simplified Approach (Aprender v0.6)

Aprender uses a **simplified diagonal prior**:
- $\Sigma_0 = \frac{1}{\lambda} I$ (scalar precision $\lambda$)
- Reduces computational cost from $O(p^3)$ to $O(p)$ for prior
- Still requires $O(p^3)$ for $(X^T X)^{-1}$ via Cholesky decomposition

### Algorithm

1. **Compute sufficient statistics**: $X^T X$ (Gram matrix), $X^T y$
2. **Estimate noise variance**: $\hat{\sigma}^2 = \frac{1}{n-p} ||y - X\beta_{OLS}||^2$
3. **Compute posterior precision**: $\Sigma_n^{-1} = \lambda I + \frac{1}{\hat{\sigma}^2} X^T X$
4. **Solve for posterior mean**: $\beta_n = \Sigma_n (\lambda \beta_0 + \frac{1}{\hat{\sigma}^2} X^T y)$

### Numerical Stability

- Uses **Cholesky decomposition** to solve linear systems
- Numerically stable for well-conditioned $X^T X$
- Prior precision $\lambda > 0$ ensures positive definiteness

## Bayesian Interpretation of Ridge Regression

Ridge regression minimizes:
$$
L(\beta) = ||y - X\beta||^2 + \alpha ||\beta||^2
$$

This is equivalent to **MAP estimation** with:
- Prior: $\beta \sim \mathcal{N}(0, \frac{1}{\alpha} I)$
- Likelihood: $y \sim \mathcal{N}(X\beta, \sigma^2 I)$

Bayesian regression extends this by computing the **full posterior**, not just the mode.

## When to Use

**Use Bayesian Linear Regression when:**
- You want **uncertainty quantification** (prediction intervals)
- You have **small datasets** (prior regularizes)
- You have **domain knowledge** (informative prior)
- You need **probabilistic predictions** for downstream tasks

**Use OLS when:**
- You only need **point estimates**
- You have **large datasets** (prior has little effect)
- You want **computational speed** (slightly faster than Bayesian)

## Further Reading

- Kevin Murphy, *Machine Learning: A Probabilistic Perspective*, Chapter 7
- Christopher Bishop, *Pattern Recognition and Machine Learning*, Chapter 3
- Andrew Gelman et al., *Bayesian Data Analysis*, Chapter 14

## See Also

- [Normal-Inverse-Gamma Inference](./normal-inverse-gamma-inference.md) - Conjugate prior details
- [Ridge Regression](#) - Frequentist regularization (coming soon)
- [Bayesian Model Comparison](#) - Marginal likelihood (coming soon)
