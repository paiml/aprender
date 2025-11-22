# Bayesian Logistic Regression

Bayesian Logistic Regression extends maximum likelihood logistic regression by treating coefficients as random variables with a prior distribution, enabling uncertainty quantification for classification tasks.

## Theory

### Model

$$
y \sim \text{Bernoulli}(\sigma(X\beta)), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- $y \in \{0, 1\}^n$: binary labels
- $X \in \mathbb{R}^{n \times p}$: feature matrix
- $\beta \in \mathbb{R}^p$: coefficient vector
- $\sigma$: sigmoid (logistic) function

### Prior (Gaussian)

$$
\beta \sim \mathcal{N}(0, \lambda^{-1} I)
$$

Where $\lambda$ is the precision (inverse variance). Higher $\lambda$ → stronger regularization.

### Posterior Approximation (Laplace)

The posterior $p(\beta | y, X)$ is **non-conjugate** and has no closed form. The **Laplace approximation** fits a Gaussian at the posterior mode (MAP):

$$
\beta | y, X \approx \mathcal{N}(\beta_{\text{MAP}}, H^{-1})
$$

Where:
- $\beta_{\text{MAP}}$: maximum a posteriori estimate
- $H$: Hessian of the negative log-posterior at $\beta_{\text{MAP}}$

### MAP Estimation

Find $\beta_{\text{MAP}}$ by maximizing the log-posterior:

$$
\begin{aligned}
\log p(\beta | y, X) &= \log p(y | X, \beta) + \log p(\beta) + \text{const} \\
&= \sum_{i=1}^n \left[ y_i \log \sigma(x_i^T \beta) + (1 - y_i) \log(1 - \sigma(x_i^T \beta)) \right] - \frac{\lambda}{2} ||\beta||^2
\end{aligned}
$$

Use **gradient ascent**:

$$
\nabla_\beta \log p(\beta | y, X) = X^T (y - p) - \lambda \beta
$$

where $p_i = \sigma(x_i^T \beta)$.

### Hessian (for Uncertainty)

The Hessian at $\beta_{\text{MAP}}$ is:

$$
H = X^T W X + \lambda I
$$

where $W = \text{diag}(p_i (1 - p_i))$ is the Fisher information matrix.

The posterior covariance is $\Sigma = H^{-1}$.

## Example: Binary Classification with Weak Prior

```rust,ignore
use aprender::bayesian::BayesianLogisticRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    // Training data: y = 1 if x > 0, else 0
    let x = Matrix::from_vec(8, 1, vec![
        -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0
    ]).unwrap();
    let y = Vector::from_vec(vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0
    ]);

    // Create model with weak prior (precision = 0.1)
    let mut model = BayesianLogisticRegression::new(0.1);

    // Fit: compute MAP estimate and Hessian
    model.fit(&x, &y).unwrap();

    // MAP estimate
    let beta = model.coefficients_map().unwrap();
    println!("β (coefficient): {:.4}", beta[0]);  // ≈ 1.4765

    // Make predictions
    let x_test = Matrix::from_vec(3, 1, vec![-1.0, 0.0, 1.0]).unwrap();
    let probas = model.predict_proba(&x_test).unwrap();

    println!("P(y=1 | x=-1.0): {:.4}", probas[0]);  // ≈ 0.1860
    println!("P(y=1 | x= 0.0): {:.4}", probas[1]);  // ≈ 0.5000
    println!("P(y=1 | x= 1.0): {:.4}", probas[2]);  // ≈ 0.8140
}
```

**Output:**
```text
β (coefficient): 1.4765
P(y=1 | x=-1.0): 0.1860
P(y=1 | x= 0.0): 0.5000
P(y=1 | x= 1.0): 0.8140
```

## Example: Uncertainty Quantification

The Laplace approximation provides **credible intervals** for predicted probabilities:

```rust,ignore
use aprender::bayesian::BayesianLogisticRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    // Small dataset (higher uncertainty)
    let x = Matrix::from_vec(6, 1, vec![
        -1.5, -1.0, -0.5, 0.5, 1.0, 1.5
    ]).unwrap();
    let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    let mut model = BayesianLogisticRegression::new(0.1);
    model.fit(&x, &y).unwrap();

    // Predict with 95% credible intervals
    let x_test = Matrix::from_vec(2, 1, vec![-2.0, 2.0]).unwrap();
    let probas = model.predict_proba(&x_test).unwrap();
    let (lower, upper) = model.predict_proba_interval(&x_test, 0.95).unwrap();

    for i in 0..2 {
        println!(
            "x={:.1}: P(y=1)={:.4}, 95% CI=[{:.4}, {:.4}]",
            x_test.get(i, 0), probas[i], lower[i], upper[i]
        );
    }
}
```

**Output:**
```text
x=-2.0: P(y=1)=0.0433, 95% CI=[0.0007, 0.7546]
x= 2.0: P(y=1)=0.9567, 95% CI=[0.2454, 0.9993]
```

The credible intervals are **wide** due to the small dataset, reflecting high posterior uncertainty.

## Example: Prior Regularization

The prior precision $\lambda$ acts as **L2 regularization** (ridge penalty):

```rust,ignore
use aprender::bayesian::BayesianLogisticRegression;
use aprender::primitives::{Matrix, Vector};

fn main() {
    // Tiny dataset (4 samples)
    let x = Matrix::from_vec(4, 1, vec![-1.0, -0.3, 0.3, 1.0]).unwrap();
    let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    // Weak prior (low regularization)
    let mut weak_model = BayesianLogisticRegression::new(0.1);
    weak_model.fit(&x, &y).unwrap();

    // Strong prior (high regularization)
    let mut strong_model = BayesianLogisticRegression::new(2.0);
    strong_model.fit(&x, &y).unwrap();

    let beta_weak = weak_model.coefficients_map().unwrap();
    let beta_strong = strong_model.coefficients_map().unwrap();

    println!("Weak prior (λ=0.1):   β = {:.4}", beta_weak[0]);
    println!("Strong prior (λ=2.0): β = {:.4}", beta_strong[0]);
}
```

**Output:**
```text
Weak prior (λ=0.1):   β = 1.4927
Strong prior (λ=2.0): β = 0.1519
```

The strong prior **shrinks** the coefficient toward zero, preventing overfitting on the tiny dataset.

## Comparison: Bayesian vs. MLE Logistic Regression

| Aspect | Bayesian (Laplace) | Maximum Likelihood |\n|--------|--------------------|--------------------|
| **Output** | Posterior distribution over β | Point estimate β̂ |
| **Uncertainty** | Credible intervals via $H^{-1}$ | Standard errors (asymptotic) |
| **Regularization** | Natural via prior (λ) | Requires explicit penalty |
| **Interpretation** | Posterior probability: $p(\beta \| \text{data})$ | Frequentist confidence intervals |
| **Computation** | Gradient ascent + Hessian | Gradient descent (IRLS) |
| **Small Data** | Regularizes via prior | May overfit |

## Implementation Details

### Laplace Approximation Algorithm

1. **Initialize**: $\beta \leftarrow 0$
2. **Gradient Ascent** (find MAP):
   - Repeat until convergence:
     - Compute predictions: $p_i = \sigma(x_i^T \beta)$
     - Compute gradient: $\nabla = X^T (y - p) - \lambda \beta$
     - Update: $\beta \leftarrow \beta + \eta \nabla$ (learning rate $\eta$)
3. **Compute Hessian**:
   - $W = \text{diag}(p_i (1 - p_i))$
   - $H = X^T W X + \lambda I$
4. **Store**: $\beta_{\text{MAP}}$ and $H$

### Credible Intervals for Predictions

For a test point $x_*$:

1. Compute linear predictor variance: $\text{Var}(x_*^T \beta) = x_*^T H^{-1} x_*$
2. Compute z-score for desired level (e.g., 1.96 for 95%)
3. Compute interval for $z_* = x_*^T \beta$:
   - $z_{\text{lower}} = z_* - 1.96 \sqrt{\text{Var}(z_*)}$
   - $z_{\text{upper}} = z_* + 1.96 \sqrt{\text{Var}(z_*)}$
4. Apply sigmoid to get probability bounds:
   - $p_{\text{lower}} = \sigma(z_{\text{lower}})$
   - $p_{\text{upper}} = \sigma(z_{\text{upper}})$

### Numerical Stability

- **Cholesky decomposition** to solve $H v = x_*$ (avoids explicit inversion)
- **Gradient averaging** by number of samples for stability
- **Convergence check** on parameter updates (tolerance $10^{-4}$)

## Bayesian Interpretation of Ridge Regularization

Logistic regression with L2 penalty minimizes:

$$
L(\beta) = -\sum_{i=1}^n \left[ y_i \log \sigma(x_i^T \beta) + (1 - y_i) \log(1 - \sigma(x_i^T \beta)) \right] + \frac{\lambda}{2} ||\beta||^2
$$

This is equivalent to **MAP estimation** with Gaussian prior $\beta \sim \mathcal{N}(0, \lambda^{-1} I)$.

Bayesian logistic regression extends this by computing the **full posterior**, not just the mode.

## When to Use

**Use Bayesian Logistic Regression when:**
- You want **uncertainty quantification** for predictions
- You have **small datasets** (prior regularizes)
- You need **probabilistic predictions** with confidence
- You want **interpretable regularization** via priors

**Use MLE Logistic Regression when:**
- You only need **point estimates** and class labels
- You have **large datasets** (prior has little effect)
- You want **computational speed** (no Hessian computation)

## Limitations

**Laplace Approximation:**
- Assumes posterior is **Gaussian** (may be poor for highly skewed posteriors)
- Only captures **first-order uncertainty** (ignores higher moments)
- Requires **MAP convergence** (may fail for ill-conditioned problems)

**For Better Posterior Estimates:**
- Use **MCMC** (Phase 2) for full posterior samples
- Use **Variational Inference** (Phase 2) for scalability
- Use **Expectation Propagation** for non-Gaussian posteriors

## Further Reading

- Kevin Murphy, *Machine Learning: A Probabilistic Perspective*, Chapter 8
- Christopher Bishop, *Pattern Recognition and Machine Learning*, Chapter 4
- Radford Neal, *Bayesian Learning for Neural Networks* (Laplace approximation)

## See Also

- [Bayesian Linear Regression](./bayesian-linear-regression.md) - Conjugate case with analytical posterior
- [Logistic Regression](#) - Maximum likelihood baseline (coming soon)
- [MCMC Methods](#) - Full posterior sampling (Phase 2)
