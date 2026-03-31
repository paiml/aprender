# Comprehensive Bayesian Probability Features Specification

**Version:** 1.0
**Date:** 2025-11-22
**Status:** Planning
**Target Release:** v0.7.0+

## Executive Summary

This specification defines a complete implementation of Bayesian probability theory and methods for the Aprender machine learning library, following E.T. Jaynes' foundational work *Probability Theory: The Logic of Science* [1] and modern computational advances. The implementation will cover classical inference, modern scalable algorithms, and production-grade Bayesian machine learning, all adhering to EXTREME TDD with zero tolerance for defects.

**Scope**: 45+ Bayesian methods across 8 major categories, from simple conjugate priors to scalable variational inference and Gaussian processes.

**Philosophy**: Bayesian probability as the principled extension of logic to uncertain propositions, implemented with mathematical rigor and computational efficiency.

---

## 1. Foundation: Bayesian Reasoning

### 1.1 Core Principles (Jaynes 2003)

**Bayes' Theorem** - The fundamental equation:

```text
P(θ|D) = P(D|θ) × P(θ) / P(D)

where:
  P(θ|D) = Posterior (belief after seeing data D)
  P(D|θ) = Likelihood (probability of data given parameters θ)
  P(θ)   = Prior (belief before seeing data)
  P(D)   = Evidence/Marginal likelihood (normalizing constant)
```

**Cox's Theorems** [1, §2.2]: Any system of plausible reasoning that satisfies basic consistency requirements is isomorphic to probability theory.

**Key Properties**:
- **Coherence**: Bayesian inference is internally consistent
- **Optimal**: Minimizes expected loss under proper scoring rules [2]
- **Sequential**: Posterior becomes next prior (Cromwell's rule)
- **Invariant**: Results independent of parameterization (with proper priors)

### 1.2 Computational Challenges

**Posterior Computation**:
```text
P(θ|D) = P(D|θ)P(θ) / ∫ P(D|θ')P(θ') dθ'
                        └─────────────────┘
                           Intractable integral
```

**Solution Categories**:
1. **Conjugate families**: Analytical posteriors (Gaussian, Beta-Binomial, etc.)
2. **Numerical integration**: Grid methods, quadrature (low dimensions only)
3. **MCMC**: Sample-based approximation (general but slow) [3]
4. **Variational inference**: Optimization-based approximation (fast, approximate) [4]
5. **Laplace approximation**: Gaussian approximation at posterior mode

---

## 2. Classical Bayesian Inference

### 2.1 Conjugate Prior Families

**Benefits**: Analytical posterior computation, interpretable hyperparameters, computational efficiency.

#### 2.1.1 Beta-Binomial (Binary Data)

**Use Case**: Coin flips, click-through rates, A/B testing

```rust
// Prior: Beta(α, β)
// Likelihood: Binomial(n, θ)
// Posterior: Beta(α + k, β + n - k)

pub struct BetaBinomial {
    alpha: f32,  // Prior successes
    beta: f32,   // Prior failures
}

impl BetaBinomial {
    pub fn new(alpha: f32, beta: f32) -> Self;
    pub fn update(&mut self, successes: usize, trials: usize);
    pub fn posterior_mean(&self) -> f32;
    pub fn posterior_variance(&self) -> f32;
    pub fn credible_interval(&self, level: f32) -> (f32, f32);
    pub fn sample(&self, rng: &mut impl Rng) -> f32;
}
```

**Reference**: Gelman et al. (2013), *Bayesian Data Analysis* [5]

#### 2.1.2 Gamma-Poisson (Count Data)

**Use Case**: Event rates, arrivals, defect counts

```rust
// Prior: Gamma(α, β)
// Likelihood: Poisson(λ)
// Posterior: Gamma(α + Σxᵢ, β + n)

pub struct GammaPoisson {
    alpha: f32,  // Shape parameter
    beta: f32,   // Rate parameter
}

impl GammaPoisson {
    pub fn new(alpha: f32, beta: f32) -> Self;
    pub fn update(&mut self, counts: &[usize]);
    pub fn posterior_mean(&self) -> f32;
    pub fn posterior_variance(&self) -> f32;
    pub fn predictive_probability(&self, k: usize) -> f32;  // Negative binomial
}
```

#### 2.1.3 Normal-InverseGamma (Gaussian Unknown Mean & Variance)

**Use Case**: Regression with unknown noise, measurement uncertainty

```rust
// Prior: NormalInverseGamma(μ₀, κ₀, α₀, β₀)
// Likelihood: N(μ, σ²)
// Posterior: NormalInverseGamma(μₙ, κₙ, αₙ, βₙ)

pub struct NormalInverseGamma {
    mu_0: f32,     // Prior mean
    kappa_0: f32,  // Prior precision (inverse variance scaling)
    alpha_0: f32,  // Inverse-gamma shape
    beta_0: f32,   // Inverse-gamma scale
}

impl NormalInverseGamma {
    pub fn new(mu_0: f32, kappa_0: f32, alpha_0: f32, beta_0: f32) -> Self;
    pub fn update(&mut self, data: &[f32]);
    pub fn posterior_mean(&self) -> (f32, f32);  // (μ, σ²)
    pub fn sample(&self, rng: &mut impl Rng) -> (f32, f32);
    pub fn predictive_distribution(&self) -> StudentT;  // Returns Student's t
}
```

#### 2.1.4 Dirichlet-Multinomial (Categorical Data)

**Use Case**: Text classification, topic modeling, market segmentation

```rust
// Prior: Dirichlet(α)
// Likelihood: Multinomial(n, θ)
// Posterior: Dirichlet(α + counts)

pub struct DirichletMultinomial {
    alpha: Vec<f32>,  // Concentration parameters
}

impl DirichletMultinomial {
    pub fn new(alpha: Vec<f32>) -> Self;
    pub fn symmetric(k: usize, alpha: f32) -> Self;  // All αᵢ = alpha
    pub fn update(&mut self, counts: &[usize]);
    pub fn posterior_mean(&self) -> Vec<f32>;
    pub fn entropy(&self) -> f32;
    pub fn sample(&self, rng: &mut impl Rng) -> Vec<f32>;
}
```

**Reference**: Minka (2000), "Estimating a Dirichlet distribution" [6]

### 2.2 Bayesian Point Estimation

#### 2.2.1 Maximum A Posteriori (MAP)

**Definition**: θ̂_MAP = argmax_θ P(θ|D) = argmax_θ [P(D|θ)P(θ)]

**Connection to Regularization**:
- L2 regularization ≡ Gaussian prior
- L1 regularization ≡ Laplace prior
- Elastic net ≡ Mixture of Gaussian + Laplace

```rust
pub trait BayesianEstimator {
    fn map_estimate(&self, data: &Matrix, targets: &Vector) -> Result<Vector, AprenderError>;
    fn posterior_mode(&self) -> Vector;
    fn log_posterior(&self, theta: &Vector) -> f32;
}
```

#### 2.2.2 Posterior Mean (Minimum MSE Estimator)

**Definition**: θ̂_mean = E[θ|D] = ∫ θ P(θ|D) dθ

**Properties**:
- Minimizes expected squared error
- Optimal under quadratic loss
- Requires full posterior (not just mode)

### 2.3 Bayesian Credible Intervals

**vs Frequentist Confidence Intervals**: Credible intervals have the interpretation "95% probability that θ ∈ [a,b]" (frequentist: "95% of intervals contain true θ").

```rust
pub trait CredibleInterval {
    /// Equal-tailed interval: 2.5% and 97.5% quantiles for 95% CI
    fn equal_tailed_interval(&self, level: f32) -> (f32, f32);

    /// Highest Posterior Density (HPD): Shortest interval containing level% probability
    fn hpd_interval(&self, level: f32) -> (f32, f32);
}
```

**Reference**: Kruschke (2014), *Doing Bayesian Data Analysis* [7]

---

## 3. Bayesian Regression & GLMs

### 3.1 Bayesian Linear Regression

**Model**:
```text
y = Xβ + ε,  ε ~ N(0, σ²I)
β ~ N(β₀, Σ₀)     # Prior on coefficients
σ² ~ InvGamma(α, β)  # Prior on noise variance
```

**Posterior** (conjugate):
```text
β|y ~ N(βₙ, Σₙ)
where:
  Σₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹
  βₙ = Σₙ(Σ₀⁻¹β₀ + σ⁻²Xᵀy)
```

**Implementation**:
```rust
pub struct BayesianLinearRegression {
    beta_prior_mean: Vector,
    beta_prior_cov: Matrix,
    noise_alpha: f32,  // InvGamma shape
    noise_beta: f32,   // InvGamma scale

    // Posterior (after fitting)
    posterior_mean: Option<Vector>,
    posterior_cov: Option<Matrix>,
}

impl BayesianLinearRegression {
    pub fn new() -> Self;
    pub fn with_prior(beta_mean: Vector, beta_cov: Matrix) -> Self;

    /// Analytical posterior update (conjugate case)
    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), AprenderError>;

    /// Posterior predictive distribution
    pub fn predict_distribution(&self, x_test: &Matrix) -> Result<Normal, AprenderError>;

    /// Point prediction (posterior mean)
    pub fn predict(&self, x_test: &Matrix) -> Result<Vector, AprenderError>;

    /// Prediction intervals (not confidence intervals!)
    pub fn predict_interval(&self, x_test: &Matrix, level: f32) -> Result<(Vector, Vector), AprenderError>;

    /// Marginal likelihood P(y|X) for model comparison
    pub fn log_marginal_likelihood(&self) -> f32;
}
```

**Benefits over OLS**:
- Uncertainty quantification (prediction intervals)
- Natural regularization via prior
- Model comparison via marginal likelihood
- Sequential updating (online learning)

### 3.2 Bayesian Logistic Regression

**Model**:
```text
y ~ Bernoulli(σ(Xβ))
β ~ N(0, λ⁻¹I)  # Gaussian prior (no analytical posterior)
```

**Inference Methods**:
1. **Laplace Approximation**: Gaussian approximation at MAP
2. **MCMC**: Full posterior sampling (gold standard)
3. **Variational Inference**: Fast approximation

```rust
pub struct BayesianLogisticRegression {
    prior_precision: f32,  // λ (inverse variance)
    inference_method: InferenceMethod,
}

pub enum InferenceMethod {
    Laplace,           // Fast, approximate
    MCMC(MCMCConfig),  // Slow, exact
    VI(VIConfig),      // Fast, approximate
}

impl BayesianLogisticRegression {
    pub fn new(inference_method: InferenceMethod) -> Self;

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), AprenderError>;

    /// Posterior predictive probability
    pub fn predict_proba(&self, x_test: &Matrix) -> Result<Vector, AprenderError>;

    /// Uncertainty in predictions
    pub fn predict_proba_interval(&self, x_test: &Matrix, level: f32) -> Result<(Vector, Vector), AprenderError>;

    /// Posterior samples (if using MCMC)
    pub fn posterior_samples(&self) -> Option<&Matrix>;
}
```

### 3.3 Bayesian Generalized Linear Models (GLMs)

**Family**: Extends to Poisson, Negative Binomial, Gamma, etc.

```rust
pub trait BayesianGLM {
    fn link_function(&self, eta: f32) -> f32;
    fn inverse_link(&self, mu: f32) -> f32;
    fn log_likelihood(&self, y: &Vector, mu: &Vector) -> f32;
}

pub struct BayesianPoissonRegression {
    // log(λ) = Xβ
}

pub struct BayesianGammaRegression {
    // E[y] = exp(Xβ), variance ∝ mean²
}
```

**Reference**: Gelman & Hill (2006), *Data Analysis Using Regression and Multilevel/Hierarchical Models* [8]

---

## 4. Hierarchical Bayesian Models

### 4.1 Multilevel Models

**Motivation**: Partial pooling between groups (neither complete pooling nor no pooling).

**Example**: Student test scores across schools
```text
yᵢⱼ ~ N(θⱼ, σ²)           # Student i in school j
θⱼ ~ N(μ, τ²)             # School effects
μ ~ N(μ₀, σ₀²)            # Population mean
τ² ~ InvGamma(α, β)       # Between-school variance
```

**Key Concept**: Shrinkage - schools with few students shrink toward population mean.

```rust
pub struct HierarchicalModel {
    group_means: Vec<f32>,         // θⱼ
    population_mean: f32,           // μ
    within_group_variance: f32,     // σ²
    between_group_variance: f32,    // τ²
}

impl HierarchicalModel {
    pub fn new() -> Self;
    pub fn fit(&mut self, data: &GroupedData) -> Result<(), AprenderError>;
    pub fn group_posterior(&self, group: usize) -> Normal;
    pub fn shrinkage_factor(&self, group: usize) -> f32;
}
```

### 4.2 Empirical Bayes

**Type II Maximum Likelihood**: Estimate hyperparameters from data, then use for inference.

```rust
pub trait EmpiricalBayes {
    /// Estimate hyperparameters from data
    fn estimate_hyperparameters(&mut self, data: &Matrix) -> Result<(), AprenderError>;

    /// Plug in estimated hyperparameters for inference
    fn fit_with_empirical_prior(&mut self, data: &Matrix) -> Result<(), AprenderError>;
}
```

**Trade-off**: Faster than full Bayes, but underestimates uncertainty.

---

## 5. Bayesian Model Selection & Comparison

### 5.1 Bayes Factors

**Definition**: Ratio of marginal likelihoods
```text
BF₁₂ = P(D|M₁) / P(D|M₂) = ∫ P(D|θ₁,M₁)P(θ₁|M₁) dθ₁ / ∫ P(D|θ₂,M₂)P(θ₂|M₂) dθ₂
```

**Interpretation** (Kass & Raftery, 1995):
- BF > 100: Decisive evidence for M₁
- BF > 10: Strong evidence
- BF > 3: Positive evidence
- BF ≈ 1: No preference

```rust
pub struct BayesFactor {
    log_marginal_likelihood_m1: f32,
    log_marginal_likelihood_m2: f32,
}

impl BayesFactor {
    pub fn compute(model1: &dyn BayesianModel, model2: &dyn BayesianModel, data: &Matrix) -> Self;
    pub fn log_bayes_factor(&self) -> f32;
    pub fn bayes_factor(&self) -> f32;
    pub fn interpretation(&self) -> &str;
}
```

### 5.2 Information Criteria

#### 5.2.1 Deviance Information Criterion (DIC)

**Formula**: DIC = D(θ̄) + 2pD, where pD = effective number of parameters

```rust
pub struct DIC {
    pub deviance_at_mean: f32,
    pub effective_parameters: f32,
    pub dic: f32,
}

impl DIC {
    pub fn compute(model: &dyn BayesianModel, posterior_samples: &Matrix) -> Self;
}
```

#### 5.2.2 Widely Applicable Information Criterion (WAIC)

**Formula**: WAIC = -2(lppd - pWAIC), where lppd = log pointwise predictive density

**Advantage**: Fully Bayesian (averages over posterior, not just point estimate).

```rust
pub struct WAIC {
    pub lppd: f32,           // Log pointwise predictive density
    pub p_waic: f32,         // Effective parameters
    pub waic: f32,
    pub standard_error: f32,
}

impl WAIC {
    pub fn compute(model: &dyn BayesianModel, data: &Matrix, posterior_samples: &Matrix) -> Self;
}
```

**Reference**: Vehtari et al. (2017), "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC" [9]

### 5.3 Bayesian Model Averaging

**Prediction**: Average over all models weighted by posterior probabilities
```text
P(y_new|D) = Σₖ P(y_new|Mₖ,D) P(Mₖ|D)
```

```rust
pub struct BayesianModelAveraging {
    models: Vec<Box<dyn BayesianModel>>,
    model_priors: Vec<f32>,
    model_posteriors: Option<Vec<f32>>,
}

impl BayesianModelAveraging {
    pub fn new(models: Vec<Box<dyn BayesianModel>>) -> Self;
    pub fn fit(&mut self, data: &Matrix) -> Result<(), AprenderError>;
    pub fn predict(&self, x_test: &Matrix) -> Result<Vector, AprenderError>;
}
```

---

## 6. Markov Chain Monte Carlo (MCMC)

### 6.1 Metropolis-Hastings Algorithm

**Core MCMC**: Sample from posterior via Markov chain with correct stationary distribution.

```rust
pub struct MetropolisHastings {
    proposal_std: f32,
    n_samples: usize,
    n_burn: usize,
    n_thin: usize,
}

impl MetropolisHastings {
    pub fn new(n_samples: usize) -> Self;

    pub fn sample<F>(&self, log_posterior: F, initial: Vector) -> MCMCResult
    where
        F: Fn(&Vector) -> f32;
}

pub struct MCMCResult {
    pub samples: Matrix,        // n_samples × n_params
    pub acceptance_rate: f32,
    pub effective_sample_size: Vec<f32>,  // ESS per parameter
    pub rhat: Vec<f32>,         // Gelman-Rubin statistic
}
```

### 6.2 Gibbs Sampling

**Conditional Sampling**: Sample each parameter given all others.

**Use Case**: Conjugate components, faster than MH when conditionals are available.

```rust
pub struct GibbsSampler {
    n_samples: usize,
    n_burn: usize,
}

impl GibbsSampler {
    pub fn sample(&self, model: &dyn GibbsModel, data: &Matrix) -> MCMCResult;
}

pub trait GibbsModel {
    fn sample_conditional(&self, param: usize, current_state: &Vector, data: &Matrix) -> f32;
}
```

### 6.3 Hamiltonian Monte Carlo (HMC)

**Gradient-Based**: Uses gradient of log-posterior for efficient exploration.

**Advantages**:
- Faster mixing than random walk
- Fewer correlated samples
- Gold standard for modern Bayesian inference (Stan, PyMC)

```rust
pub struct HamiltonianMC {
    n_samples: usize,
    step_size: f32,
    n_leapfrog_steps: usize,
}

impl HamiltonianMC {
    pub fn new(n_samples: usize) -> Self;
    pub fn with_step_size(mut self, step_size: f32) -> Self;

    pub fn sample<F, G>(&self, log_posterior: F, gradient: G, initial: Vector) -> MCMCResult
    where
        F: Fn(&Vector) -> f32,
        G: Fn(&Vector) -> Vector;
}
```

### 6.4 No-U-Turn Sampler (NUTS)

**Adaptive HMC**: Automatically tunes step size and number of leapfrog steps.

**Reference**: Hoffman & Gelman (2014), "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" [10]

```rust
pub struct NUTS {
    n_samples: usize,
    target_acceptance: f32,  // Default 0.8
    max_tree_depth: usize,   // Default 10
}

impl NUTS {
    pub fn new(n_samples: usize) -> Self;
    pub fn sample<F, G>(&self, log_posterior: F, gradient: G, initial: Vector) -> MCMCResult
    where
        F: Fn(&Vector) -> f32,
        G: Fn(&Vector) -> Vector;
}
```

### 6.5 MCMC Diagnostics

```rust
pub struct MCMCDiagnostics;

impl MCMCDiagnostics {
    /// Gelman-Rubin R̂ statistic (should be < 1.1)
    pub fn gelman_rubin(chains: &[Matrix]) -> Vec<f32>;

    /// Effective sample size
    pub fn effective_sample_size(samples: &Matrix) -> Vec<f32>;

    /// Autocorrelation function
    pub fn autocorrelation(samples: &Vector, max_lag: usize) -> Vec<f32>;

    /// Trace plots, density plots (return plot data)
    pub fn trace_plot_data(samples: &Matrix) -> Vec<Vec<(usize, f32)>>;
}
```

---

## 7. Variational Inference (VI)

### 7.1 Mean-Field Variational Bayes

**Idea**: Approximate intractable P(θ|D) with tractable Q(θ) by minimizing KL divergence.

**Optimization**:
```text
Q*(θ) = argmin_Q KL(Q||P)
      = argmax_Q E_Q[log P(D,θ)] - E_Q[log Q(θ)]  (ELBO)
```

```rust
pub struct MeanFieldVI {
    max_iter: usize,
    tolerance: f32,
    learning_rate: f32,
}

impl MeanFieldVI {
    pub fn new() -> Self;

    pub fn fit<M: BayesianModel>(&self, model: &M, data: &Matrix) -> VIResult;
}

pub struct VIResult {
    pub variational_params: Vec<f32>,  // Parameters of Q(θ)
    pub elbo_history: Vec<f32>,
    pub converged: bool,
}
```

**Reference**: Blei et al. (2017), "Variational Inference: A Review for Statisticians" [4]

### 7.2 Automatic Differentiation Variational Inference (ADVI)

**Automatic**: Works for any differentiable model (no manual derivation).

**Transformation**: Maps constrained parameters to unconstrained space.

```rust
pub struct ADVI {
    n_samples: usize,      // Monte Carlo samples for ELBO gradient
    max_iter: usize,
    learning_rate: f32,
}

impl ADVI {
    pub fn fit(&self, model: &dyn DifferentiableModel, data: &Matrix) -> VIResult;
}
```

### 7.3 Black Box Variational Inference (BBVI)

**Gradient Estimation**: Reparameterization trick or score function estimator.

```rust
pub struct BBVI {
    gradient_estimator: GradientEstimator,
}

pub enum GradientEstimator {
    Reparameterization,  // Low variance, requires reparameterizable distributions
    ScoreFunction,       // High variance, works for any distribution
    ReinforcementLearning,  // Policy gradient methods
}
```

---

## 8. Scalable Bayesian Methods

### 8.1 Stochastic Variational Inference (SVI)

**Minibatch Training**: Scale VI to large datasets via stochastic gradients.

```rust
pub struct StochasticVI {
    batch_size: usize,
    learning_rate_schedule: LearningRateSchedule,
    max_epochs: usize,
}

impl StochasticVI {
    pub fn fit(&self, model: &dyn BayesianModel, data: &Matrix) -> VIResult;
}
```

### 8.2 Streaming Variational Bayes

**Online Learning**: Update posterior sequentially as new data arrives.

```rust
pub trait StreamingBayesian {
    fn update_posterior(&mut self, new_data: &Matrix) -> Result<(), AprenderError>;
    fn predict_next(&self, x: &Vector) -> Result<Distribution, AprenderError>;
}
```

### 8.3 Distributed MCMC

**Parallel Chains**: Embarrassingly parallel (run independent chains, check convergence).

```rust
pub struct ParallelMCMC {
    n_chains: usize,
    sampler: Box<dyn MCMCSampler>,
}

impl ParallelMCMC {
    pub fn sample_parallel(&self, log_posterior: impl Fn(&Vector) -> f32 + Sync) -> Vec<MCMCResult>;
}
```

---

## 9. Gaussian Processes (GPs)

### 9.1 GP Regression

**Model**: f(x) ~ GP(m(x), k(x,x'))

**Prediction**:
```text
f*|X,y,X* ~ N(μ*, Σ*)
where:
  μ* = K*ᵀ(K + σ²I)⁻¹y
  Σ* = K** - K*ᵀ(K + σ²I)⁻¹K*
```

```rust
pub struct GaussianProcessRegressor {
    kernel: Box<dyn Kernel>,
    noise_variance: f32,

    // Training data
    x_train: Option<Matrix>,
    y_train: Option<Vector>,

    // Cached computations
    k_inv: Option<Matrix>,  // (K + σ²I)⁻¹
    alpha: Option<Vector>,  // (K + σ²I)⁻¹y
}

pub trait Kernel {
    fn compute(&self, x1: &Matrix, x2: &Matrix) -> Matrix;
    fn gradient(&self, x1: &Matrix, x2: &Matrix) -> Vec<Matrix>;
}

impl GaussianProcessRegressor {
    pub fn new(kernel: Box<dyn Kernel>) -> Self;

    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), AprenderError>;

    /// Posterior predictive mean
    pub fn predict(&self, x_test: &Matrix) -> Result<Vector, AprenderError>;

    /// Posterior predictive mean + variance
    pub fn predict_with_uncertainty(&self, x_test: &Matrix) -> Result<(Vector, Vector), AprenderError>;

    /// Sample from posterior predictive
    pub fn sample_posterior(&self, x_test: &Matrix, n_samples: usize) -> Result<Matrix, AprenderError>;

    /// Optimize kernel hyperparameters
    pub fn optimize_hyperparameters(&mut self) -> Result<(), AprenderError>;
}
```

### 9.2 GP Kernels

```rust
pub struct RBFKernel {
    length_scale: f32,
    variance: f32,
}

pub struct Matern52Kernel {
    length_scale: f32,
    variance: f32,
}

pub struct PeriodicKernel {
    period: f32,
    length_scale: f32,
}

pub struct CompositeKernel {
    kernels: Vec<Box<dyn Kernel>>,
    operation: KernelOperation,
}

pub enum KernelOperation {
    Sum,      // k₁ + k₂
    Product,  // k₁ × k₂
}
```

### 9.3 Sparse GP Approximations

**Inducing Points**: Reduce O(n³) to O(m²n) using m << n inducing points.

```rust
pub struct SparseGP {
    kernel: Box<dyn Kernel>,
    n_inducing: usize,
    inducing_points: Option<Matrix>,
}

impl SparseGP {
    pub fn fit(&mut self, x: &Matrix, y: &Vector) -> Result<(), AprenderError>;
}
```

**Reference**: Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning* [11]

---

## 10. Bayesian Neural Networks

### 10.1 Weight Uncertainty in Neural Networks

**Idea**: Place priors over network weights, propagate uncertainty.

```rust
pub struct BayesianNeuralNetwork {
    layers: Vec<BayesianLayer>,
    prior_variance: f32,
}

pub struct BayesianLayer {
    weight_mean: Matrix,
    weight_log_variance: Matrix,
    bias_mean: Vector,
    bias_log_variance: Vector,
}

impl BayesianNeuralNetwork {
    pub fn new(layer_sizes: &[usize]) -> Self;

    /// Variational inference training
    pub fn fit(&mut self, x: &Matrix, y: &Vector, n_epochs: usize) -> Result<(), AprenderError>;

    /// Monte Carlo prediction (sample multiple networks)
    pub fn predict_with_uncertainty(&self, x: &Matrix, n_samples: usize) -> Result<(Vector, Vector), AprenderError>;
}
```

### 10.2 Bayes by Backprop

**Training**: Minimize KL divergence using reparameterization trick.

```text
Loss = KL(Q(w)||P(w)) - E_Q[log P(D|w)]
```

### 10.3 MC Dropout as Bayesian Approximation

**Approximation**: Dropout at test time = approximate Bayesian inference.

```rust
pub struct MCDropout {
    network: NeuralNetwork,
    dropout_rate: f32,
    n_samples: usize,
}

impl MCDropout {
    pub fn predict_with_uncertainty(&self, x: &Matrix) -> (Vector, Vector);
}
```

---

## 11. Advanced Topics

### 11.1 Approximate Bayesian Computation (ABC)

**Simulation-Based**: When likelihood is intractable but can simulate from model.

```rust
pub struct ABC {
    simulator: Box<dyn Simulator>,
    distance_metric: Box<dyn DistanceMetric>,
    tolerance: f32,
}

pub trait Simulator {
    fn simulate(&self, params: &Vector) -> Matrix;
}

impl ABC {
    pub fn sample_posterior(&self, observed_data: &Matrix, n_samples: usize) -> Matrix;
}
```

### 11.2 Bayesian Optimization

**Black-Box Optimization**: Optimize expensive functions using GPs.

**Acquisition Functions**:
- Expected Improvement (EI)
- Probability of Improvement (PI)
- Upper Confidence Bound (UCB)

```rust
pub struct BayesianOptimizer {
    gp: GaussianProcessRegressor,
    acquisition: AcquisitionFunction,
}

pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement { threshold: f32 },
    UpperConfidenceBound { kappa: f32 },
}

impl BayesianOptimizer {
    pub fn optimize<F>(&mut self, objective: F, bounds: &[(f32, f32)], n_iterations: usize) -> Vector
    where
        F: Fn(&Vector) -> f32;
}
```

**Reference**: Shahriari et al. (2016), "Taking the Human Out of the Loop: A Review of Bayesian Optimization" [12]

### 11.3 Bayesian Non-Parametrics

**Infinite Models**: Number of parameters grows with data.

#### 11.3.1 Dirichlet Process Mixture Models

**Use Case**: Automatic determination of number of clusters.

```rust
pub struct DirichletProcessGMM {
    alpha: f32,  // Concentration parameter
    base_distribution: Normal,
}

impl DirichletProcessGMM {
    pub fn fit(&mut self, data: &Matrix) -> Result<(), AprenderError>;
    pub fn n_components(&self) -> usize;  // Inferred from data
}
```

#### 11.3.2 Gaussian Process Latent Variable Models

**Dimensionality Reduction**: Bayesian PCA with automatic relevance determination.

---

## 12. The Toyota Way as Bayesian Inference

### 12.1 The Convergence of Philosophy and Probability

The Toyota Production System (TPS) is frequently viewed as a cultural or management philosophy distinct from rigorous statistical methods. However, a deep structural analysis reveals that TPS is fundamentally a **physical manifestation of Bayesian inference**. Every principle of the Toyota Way, when stripped of cultural artifacts and examined through an epistemological lens, maps directly to operations in Bayesian probability theory.

**Core Thesis**: The factory floor is a Bayesian inference engine. Lean manufacturing is the practice of reducing entropy—the systematic tightening of probability distributions. Every act of Kaizen (continuous improvement) is a Bayesian update: movement from a prior belief about a process through the likelihood function of observed data to a posterior understanding closer to reality.

**Waste (Muda) as Uncertainty**: The seven types of waste—overproduction, waiting, transport, overprocessing, inventory, motion, defects—are symptoms of a single underlying cause: **uncertainty**. Inventory exists because posterior probability distributions regarding demand, machine uptime, and supplier delivery have high variance. Perfect information (Dirac delta functions) would eliminate the need for buffers. Therefore, **waste elimination = variance reduction = information maximization**.

**Reference**: Spear & Bowen (1999), "Decoding the DNA of the Toyota Production System" [20]

### 12.2 PDCA as Recursive Bayesian Updating

The Plan-Do-Check-Act (PDCA) cycle, perfected by Toyota, is a recursive Bayesian algorithm:

#### 12.2.1 Plan: Establishing the Prior P(H)

**Standardized Work** is the formalization of the Prior distribution. As Masaaki Imai stated: "Where there is no standard, there can be no kaizen." Why? Because without a defined Prior, any observation is equally likely, and evidence carries no information content.

```rust
pub struct StandardizedWork {
    /// Prior distribution over process outcomes
    prior_mean: f32,
    prior_variance: f32,

    /// Specification limits
    lower_spec_limit: f32,
    upper_spec_limit: f32,
}

impl StandardizedWork {
    /// Creates a strong informative prior from historical data
    pub fn from_historical_data(outcomes: &[f32]) -> Self;

    /// Checks if observed outcome is within expected distribution
    pub fn is_normal(&self, observed: f32) -> bool {
        let z_score = (observed - self.prior_mean) / self.prior_variance.sqrt();
        z_score.abs() < 3.0  // 3-sigma rule
    }
}
```

**Hansei (Reflection)**: Review of historical posteriors to inform new priors, ensuring plans are informed (not uniform) priors weighted by past experience.

#### 12.2.2 Do: The Experiment and Likelihood P(E|H)

The "Do" phase generates data under controlled conditions. Strict adherence to standards is required not to "enslave workers" but to **preserve data integrity** for valid Bayesian updating. Arbitrary deviations contaminate the likelihood function—we cannot distinguish whether results come from the process H or from deviations.

```rust
pub struct ProcessExecution {
    standard: StandardizedWork,
    observations: Vec<Observation>,
}

pub struct Observation {
    timestamp: DateTime,
    measurement: f32,
    operator: String,
    conditions: HashMap<String, f32>,
}

impl ProcessExecution {
    /// Records high-fidelity observation
    pub fn record(&mut self, obs: Observation);

    /// Compute likelihood of observations given standard
    pub fn log_likelihood(&self) -> f32;
}
```

#### 12.2.3 Check/Study: Computing Posterior P(H|E)

**Not mere inspection**, but analysis of the relationship between prediction and outcome. The Extended PDCA framework emphasizes process mining to rigorously compare "simulation model behavior" (Prior) with "observed log behavior" (Evidence)—functionally minimizing Kullback-Leibler divergence.

```rust
pub struct BayesianProcessControl {
    prior: Distribution,
    observations: Vec<f32>,
}

impl BayesianProcessControl {
    /// Compute posterior distribution
    pub fn update_posterior(&mut self) -> Distribution {
        // Bayes' theorem: P(θ|D) ∝ P(D|θ) × P(θ)
        let likelihood = self.compute_likelihood();
        let posterior = self.prior.update(likelihood, &self.observations);

        // Check if evidence is in low-probability region
        if self.is_anomaly(&posterior) {
            // Trigger problem-solving (Act phase)
            self.trigger_andon();
        }

        posterior
    }

    fn is_anomaly(&self, posterior: &Distribution) -> bool {
        // Low marginal likelihood P(E|H) indicates problem
        posterior.marginal_likelihood() < THRESHOLD
    }

    fn trigger_andon(&self) {
        // Stop the line - Bayesian filter prevents noise propagation
    }
}
```

**Reference**: Larrick & Feiler (2015), "Improving Bayesian Reasoning: Advice for Decision Makers" [21]

#### 12.2.4 Act: Bayesian Decision Theory

If posterior confirms prior (P(H|E) is high): Standardize further—reduce variance.
If posterior refutes prior (P(H|E) is low): Formulate new hypothesis—pivot.

This maps to **Bayesian Decision Theory**: choose action to minimize expected loss given posterior distribution.

### 12.3 The 14 Principles as Bayesian Priors

Jeffrey Liker's 14 Toyota Way principles transform from behavioral guidelines into structural requirements for a Bayesian inference engine:

#### Principle 1: Long-Term Philosophy

**Bayesian Interpretation**: Infinite time horizon for Bayesian Decision Theory. Shifts balance from exploitation (short-term greedy optimization) to exploration (learning that tightens posterior distributions for future iterations).

Research via Latent Semantic Analysis of Toyota's corporate memory reveals "Team Work" and "Respect for People" are emphasized over "Kaizen" because the long-term cultural prior provides the container for short-term Bayesian updates.

**Reference**: Hino (2006), "LSA analysis of Toyota corporate memory" [22]

#### Principle 2: Create Continuous Flow

**Flow as Signal Generation**: Batch-and-queue systems create temporal lag between events (H) and evidence (E), degrading the likelihood function P(E|H). One-piece flow creates immediate temporal linkage, maximizing signal-to-noise ratio for high-fidelity Bayesian updates.

#### Principle 6: Standardized Tasks

**Creating Strong Informative Priors**: Codifies 50+ years of experience into baseline distributions. Without standards (flat priors), massive data is required to converge on truth. Standards enable immediate anomaly detection.

#### Principle 7: Visual Control

**Probability Dashboard**: Kanban boards, Andon lights, shadow boards visualize system state and deviations from expected distributions in real-time, broadcasting probabilistic conflicts to the entire network.

```rust
pub struct VisualControl {
    expected_state: State,
    actual_state: State,
}

impl VisualControl {
    /// Binary probability map
    pub fn status(&self) -> ControlStatus {
        if self.expected_state == self.actual_state {
            ControlStatus::Normal(1.0)  // P = 1.0
        } else {
            ControlStatus::Abnormal(0.0)  // P = 0.0, flag low-likelihood event
        }
    }
}
```

### 12.4 Set-Based Design: Bayesian Filtering in Product Development

Traditional "Point-Based Design" selects one solution early (single hypothesis H₁) and iterates. Toyota's **Set-Based Concurrent Engineering** maintains broad solution sets and eliminates infeasible options through evidence.

**Mechanism as Bayesian Sequential Monte Carlo**:

1. **Initialization**: Define feasible region S₀ = {H₁, H₂, ..., Hₙ} (broad, flat prior)
2. **Evidence Collection**: Trade-off curves and constraint data arrive from manufacturing, suppliers, styling
3. **Intersection (Update)**: Eliminate weak solutions—intersect feasible sets from different departments
   ```
   S_new = S_manufacturing ∩ S_styling ∩ S_engineering
   ```
4. **Convergence**: Set narrows (posterior tightens) until final solution emerges by elimination of infeasible options

**Second Toyota Paradox** (Ward et al., 1995): "Delaying decisions, communicating ambiguously, and pursuing excessive prototypes" appears wasteful to frequentists but is **optimal Sequential Monte Carlo filtering** to Bayesians. By delaying decision collapse, Toyota maintains high-entropy prior until likelihood information is strong enough for certainty.

**Efficiency**: Verifying infeasibility is cheaper than verifying optimality. Progressive pruning avoids local optima that trap point-based designers.

**Reference**: Ward, Liker, Cristiano & Sobek (1995), "The Second Toyota Paradox" [23]

### 12.5 Genchi Genbutsu: High-Fidelity Sampling

**"Go and See for Yourself"** - Safeguard against biased priors and noisy likelihoods.

**Problem**: Managers using aggregated data construct office-based priors P(H_office) from lossy compression. Aggregation destroys variance information. Going to Gemba (the floor) collects raw, uncompressed evidence for accurate P(H_gemba).

**Modular Neural Networks**: Recent computational biology research shows Bayesian inference in biological brains requires modular networks with different timescales. TPS's decentralized teams and cell-based manufacturing reflect this—local Bayesian processors update local priors P(H_local) at fast timescales, while management updates global priors P(H_global) at slower timescales.

**Reference**: Ichikawa & Kaneko (2024), "Modular neural networks for Bayesian inference" [24]

```rust
pub trait BayesianSensor {
    /// Collect high-fidelity, uncompressed observations
    fn gemba_observation(&self) -> RawData;

    /// Reject aggregated/lossy data
    fn reject_office_prior(&self) -> bool;
}

pub struct WorkCell {
    local_prior: Distribution,
    update_rate: Duration,  // Fast timescale
}

pub struct ManagementLayer {
    global_prior: Distribution,
    update_rate: Duration,  // Slow timescale
}
```

### 12.6 Bayesian Statistical Process Control (SPC)

Traditional Shewhart charts assume fixed sample sizes, normality, and fixed parameters—failing in high-mix, low-volume Lean manufacturing.

**Bayesian Control Charts** treat process parameters (μ, σ²) as random variables with prior distributions. Even small sample sizes allow updates and shift detection.

#### 12.6.1 Bayesian EWMA Charts

```rust
pub struct BayesianEWMA {
    prior_mean: f32,
    prior_variance: f32,
    smoothing_factor: f32,
    measurement_error_model: Option<Distribution>,
}

impl BayesianEWMA {
    /// Update with new observation, accounting for measurement error
    pub fn update(&mut self, observation: f32) -> ControlResult {
        // Incorporate measurement error model into likelihood
        let posterior = self.bayesian_update(observation);

        // Distinguish process drift from sensor noise
        if self.is_process_drift(&posterior) {
            ControlResult::OutOfControl(posterior)
        } else if self.is_sensor_noise(&posterior) {
            ControlResult::SensorWarning(posterior)
        } else {
            ControlResult::InControl(posterior)
        }
    }

    /// Detect shifts with fewer samples than Shewhart
    pub fn detect_shift(&self, data: &[f32]) -> Option<usize> {
        // Bayesian charts achieve higher detection rate, fewer false alarms
    }
}
```

**Empirical Evidence**: Ali et al. (2025) compared Shewhart vs. Bayesian average charts in blood glucose monitoring (variable process similar to Lean manufacturing). **Result**: "Bayesian chart more accurate in detecting rapid changes, higher detection rate, fewer false alarms." [25]

**Measurement Error**: Bayesian VEWMA charts incorporate measurement error models, probabilistically "doubting" sensor readings that conflict strongly with prior process state, waiting for confirmation before stopping the line.

**Reference**: Ho & Quinino (2013), "Bayesian SPC with measurement error" [26]

#### 12.6.2 System Reliability as Coherent Structures

Toyota views the factory as a "coherent structure" of components (people, machines, suppliers). Bayesian methods synthesize component test data (likelihood) with engineering judgment (prior) to estimate system reliability.

```rust
pub struct CoherentSystem {
    components: Vec<Component>,
}

pub struct Component {
    reliability_prior: BetaDistribution,
    test_data: Vec<bool>,  // Success/failure
}

impl CoherentSystem {
    /// Estimate system reliability from component tests
    pub fn system_reliability(&self) -> f32 {
        self.components
            .iter()
            .map(|c| c.posterior_reliability())
            .product()  // Assuming series system
    }
}
```

**TPS Insight**: Obsession with supplier development mathematically increases reliability of entire coherent structure by tightening prior distributions of supplier quality.

**Reference**: Mastran & Singpurwalla (1978), "Bayesian estimation of coherent structure reliability" [27]

### 12.7 Lean Startup: Bayesian Innovation

Eric Ries adapted TPS principles for startups, making implicit Bayesian nature explicit.

#### 12.7.1 MVP as Information Probe

**Minimum Viable Product** = instrument to maximize Bayesian updating rate per dollar spent.

**Mathematical Model** (Yoo, Huang & Arifoğlu, 2024): "Build-test-learn" cycle as two-stage decision process with Bayesian belief updating. [28]

**Key Findings**:

1. **Optimal MVP Quality**: Intermediate level maximizes discriminative power
   - Too high: everyone buys (false positive)
   - Too low: no one buys (false negative)
   - Neither provides information about concept

2. **Failure as Information**: "Failure to sell can be equally informative as success" if experiment designed to be discriminative

```rust
pub struct MVP {
    quality_level: f32,  // Intermediate for max information
    design_location: DesignPoint,
}

impl MVP {
    /// Design experiment to maximize information gain
    pub fn optimal_design(candidate_designs: &[DesignPoint]) -> Self {
        // Select quality to maximize discriminative power
        let quality = Self::intermediate_quality();

        // Design to rule out less likely or confirm more likely
        let location = Self::select_test_location(candidate_designs);

        MVP { quality_level: quality, design_location: location }
    }

    /// Bayesian update from test results
    pub fn update_belief(&self, sales_data: &[bool]) -> Distribution {
        // Update posterior over business model space
    }
}
```

#### 12.7.2 Cognitive Bias Problem

Entrepreneurs with strong priors P(Idea is Great) ≈ 1 ignore negative evidence (confirmation bias). York & Danes (2017) recommend formal decision aids—stopping rules, pre-mortems—to force Bayesian updates despite ego resistance. [29]

#### 12.7.3 Advice Networks and Correlated Priors

Cohen & Koning (2020): If 10 mentors from same background (ex-Google engineers) all say "Yes," effective sample size ≈ 1 due to correlated priors. Sophisticated Bayesian entrepreneurs must:
- Account for correlation
- Seek diverse sources
- Actively challenge own priors
- Seek "negative advice" (fatal flaws) to maximize information gain

**Reference**: Cohen & Koning (2020), "The Bayesian Entrepreneur" [30]

### 12.8 Measuring Kaizen Readiness

**Bayesian Best-Worst Method (BBWM)** evaluates organizational readiness for Kaizen:

Key success factors with Bayesian weights:
- Training and Education: 0.119
- Employee Attitude: 0.112
- Senior Management Support: High significance

**Poetic Insight**: Using Bayesian methods to determine if organization is ready to use... Bayesian methods (Kaizen). **Finding**: Organizational prior (culture/attitude) is strongest predictor of success. If prior is resistant (dogmatic), no amount of evidence (Kaizen events) leads to successful updates.

**Toyota Maxim Proven**: "Build people, then build cars"—must condition the Bayesian Brain (workforce) before expecting effective information processing.

**Reference**: Mahdiraji et al. (2023), "Bayesian Kaizen readiness evaluation" [31]

### 12.9 Implementation in Aprender Development

**Aprender itself follows TPS/Bayesian principles**:

1. **Standardized Work** = Coding standards, clippy rules (Prior)
2. **PDCA Cycle** = Red-Green-Refactor (Bayesian update loop)
3. **Visual Control** = Test coverage dashboard, mutation scores
4. **Andon** = Clippy warnings, failing tests stop the build
5. **Genchi Genbutsu** = Read the code, run the benchmarks (not just read reports)
6. **Flow** = One feature at a time, immediate CI feedback
7. **Kaizen** = Continuous improvement via PMAT quality gates

```rust
// Example: Aprender development as Bayesian process
pub struct DevelopmentCycle {
    standard: CodingStandard,  // Prior
    implementation: Code,       // Experiment
    tests: TestResults,         // Evidence
    coverage: f32,             // Likelihood metric
}

impl DevelopmentCycle {
    /// PDCA = Red-Green-Refactor as Bayesian update
    pub fn pdca_cycle(&mut self) -> Result<(), AprenderError> {
        // RED: Write failing test (define expected distribution)
        let expected_behavior = self.write_test();

        // DO: Implement feature (generate evidence)
        let actual_behavior = self.implement();

        // CHECK: Compare expected vs actual (compute posterior)
        let posterior = self.bayesian_update(expected_behavior, actual_behavior);

        // ACT: If test fails (low P(E|H)), refine implementation
        if posterior.is_low_probability() {
            self.refactor()?;
        } else {
            self.standardize();  // Update standard (new prior)
        }

        Ok(())
    }
}
```

### 12.10 Summary: TPS-Bayesian Mapping

| TPS Concept | Bayesian Equivalent | Function |
|-------------|-------------------|----------|
| Standardized Work | Prior Distribution P(H) | Baseline prediction, anomaly detection |
| Production/Operation | Experiment | Generates data (Evidence E) |
| Check/Study | Posterior P(H\|E) | Updates belief in process stability |
| Problem (Deviation) | Low P(E\|H) | Signal that Prior doesn't match Reality |
| Set-Based Design | Sequential Monte Carlo | Multiple hypotheses until data eliminates infeasible |
| Root Cause Analysis | Causal Inference | Identify hidden variable explaining likelihood |
| Kanban | Demand Signal | Propagate information to update inventory requirements |
| Genchi Genbutsu | High-Fidelity Sampling | Replace biased priors with empirical priors |
| Andon (Stop Line) | Bayesian Filter | Stop error propagation downstream |
| Kaizen Readiness | Prior Belief State | Cultural condition for update mechanism |
| True North | Zero Entropy (Asymptote) | P(Defect)=0, requires infinite information |

**Conclusion**: The Toyota Production System is a rigorous, empirically-driven, probabilistic system for navigating uncertainty. To practice TPS without understanding probability is cargo cult. To the True Believer, the factory is a cathedral of data, and work is the holy act of aligning beliefs with reality—one Bayesian update at a time.

---

## 13. Implementation Roadmap

### Phase 1: Classical Bayesian Inference (v0.7.0, 4-6 weeks)

**Conjugate Families**:
- [ ] Beta-Binomial
- [ ] Gamma-Poisson
- [ ] Normal-InverseGamma
- [ ] Dirichlet-Multinomial

**Bayesian Regression**:
- [ ] Bayesian Linear Regression (analytical)
- [ ] Bayesian Ridge Regression
- [ ] Bayesian Logistic Regression (Laplace approximation)

**Model Selection**:
- [ ] Bayes Factors
- [ ] DIC
- [ ] WAIC

**Tests**: 150+ (conjugate priors, regression, model selection)
**Documentation**: 3 book chapters (classical inference, regression, model selection)
**Examples**: Beta-Binomial A/B testing, Bayesian linear regression with prediction intervals

### Phase 2: MCMC Methods (v0.8.0, 6-8 weeks)

**Samplers**:
- [ ] Metropolis-Hastings
- [ ] Gibbs Sampling
- [ ] Hamiltonian Monte Carlo
- [ ] NUTS (No-U-Turn Sampler)

**Diagnostics**:
- [ ] Gelman-Rubin R̂
- [ ] Effective Sample Size
- [ ] Autocorrelation
- [ ] Trace plots

**Applications**:
- [ ] Bayesian Logistic Regression (MCMC)
- [ ] Hierarchical Models
- [ ] Bayesian GLMs

**Tests**: 120+ (samplers, diagnostics, convergence)
**Documentation**: 2 book chapters (MCMC theory, hierarchical models)
**Examples**: Hierarchical model for grouped data, MCMC diagnostics

### Phase 3: Variational Inference (v0.9.0, 4-6 weeks)

**Algorithms**:
- [ ] Mean-Field VI
- [ ] ADVI (Automatic Differentiation VI)
- [ ] BBVI (Black Box VI)
- [ ] Stochastic VI (scalable)

**Applications**:
- [ ] Variational Bayesian Linear Regression
- [ ] Variational Bayesian GMM
- [ ] Bayesian Neural Networks (Bayes by Backprop)

**Tests**: 100+ (VI algorithms, convergence, ELBO)
**Documentation**: 1 book chapter (variational inference)
**Examples**: Scalable VI on large datasets

### Phase 4: Gaussian Processes (v1.0.0, 6-8 weeks)

**Core GP**:
- [ ] GP Regression
- [ ] GP Classification
- [ ] Sparse GP (inducing points)

**Kernels**:
- [ ] RBF (Squared Exponential)
- [ ] Matérn (ν=3/2, ν=5/2)
- [ ] Periodic
- [ ] Composite kernels (sum, product)

**Applications**:
- [ ] Bayesian Optimization
- [ ] Time series forecasting
- [ ] Active learning

**Tests**: 130+ (kernels, regression, optimization)
**Documentation**: 2 book chapters (GP theory, Bayesian optimization)
**Examples**: Hyperparameter tuning with Bayesian optimization

### Phase 5: Advanced Topics (v1.1.0+, 8-10 weeks)

**Bayesian Neural Networks**:
- [ ] Weight uncertainty propagation
- [ ] MC Dropout
- [ ] Ensembling

**Non-Parametrics**:
- [ ] Dirichlet Process Mixture Models
- [ ] GP Latent Variable Models
- [ ] Infinite Hidden Markov Models

**ABC & Simulation**:
- [ ] Approximate Bayesian Computation
- [ ] Sequential Monte Carlo (SMC)

**Tests**: 150+ (neural networks, non-parametrics, ABC)
**Documentation**: 3 book chapters
**Examples**: DP-GMM clustering, ABC for complex models

---

## 13. Quality Standards

### 13.1 EXTREME TDD Requirements

**All implementations must satisfy**:
- ✅ 95%+ test coverage
- ✅ Property-based tests (convergence, invariances)
- ✅ Mutation score ≥85%
- ✅ Zero clippy warnings
- ✅ Zero unwrap() calls (use expect() with descriptive messages)
- ✅ Comprehensive rustdoc with examples
- ✅ Book chapter for each major method
- ✅ Runnable example for each algorithm

### 13.2 Numerical Stability

**Critical for Bayesian methods**:
- Use log-probabilities to avoid underflow
- Cholesky decomposition for positive-definite matrices
- Numerically stable log-sum-exp
- Proper handling of zero/negative probabilities

```rust
/// Numerically stable log-sum-exp
pub fn log_sum_exp(log_probs: &[f32]) -> f32 {
    let max_log = log_probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    max_log + log_probs.iter().map(|&x| (x - max_log).exp()).sum::<f32>().ln()
}
```

### 13.3 Convergence Guarantees

**MCMC Diagnostics**:
- R̂ < 1.1 for all parameters (Gelman-Rubin)
- ESS > 100 per chain (effective sample size)
- Visual inspection of trace plots (automated in tests)

**VI Convergence**:
- ELBO increase plateau detection
- Gradient norm threshold
- Maximum iteration cutoff

### 13.4 Reproducibility

**Random Number Generation**:
- All stochastic methods accept seed parameter
- Deterministic behavior when seed is provided
- Document sources of randomness

```rust
pub trait Reproducible {
    fn set_seed(&mut self, seed: u64);
}
```

---

## 14. Performance Benchmarks

### 14.1 Conjugate Priors

**Target**: Sub-microsecond updates for simple conjugates

| Method | Update Time (n=1000) |
|--------|---------------------|
| Beta-Binomial | < 100ns |
| Gamma-Poisson | < 150ns |
| Normal-InverseGamma | < 500ns |
| Dirichlet-Multinomial | < 1µs |

### 14.2 Bayesian Regression

**Target**: Competitive with scikit-learn (within 2x)

| Method | Training (n=10k, p=10) | Prediction (n=1k) |
|--------|----------------------|------------------|
| Bayesian Linear (analytical) | < 100ms | < 1ms |
| Bayesian Logistic (Laplace) | < 500ms | < 5ms |
| Bayesian Logistic (MCMC) | < 30s | < 10ms |

### 14.3 MCMC Sampling

**Target**: 1000 samples/second for simple models (single chain)

| Sampler | Samples/sec (10 params) |
|---------|------------------------|
| Metropolis-Hastings | 5000+ |
| Gibbs | 3000+ |
| HMC | 500+ |
| NUTS | 200+ |

### 14.4 Gaussian Processes

**Target**: Handle n=10k training points in reasonable time

| Operation | Time (n=1000) | Time (n=10k, sparse) |
|-----------|--------------|-------------------|
| Fit (exact) | < 1s | N/A |
| Fit (sparse, m=100) | < 500ms | < 5s |
| Predict (k=100) | < 10ms | < 50ms |

---

## 15. Academic References

1. **Jaynes, E. T. (2003)**. *Probability Theory: The Logic of Science*. Cambridge University Press.
   - Foundation: Cox's theorems, principled Bayesian reasoning
   - Used for: Core philosophy, classical inference

2. **Gneiting, T., & Raftery, A. E. (2007)**. "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association*, 102(477), 359-378.
   - Bayesian inference is optimal under proper scoring rules
   - Used for: Justification of Bayesian approach

3. **Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013)**. *Bayesian Data Analysis* (3rd ed.). CRC Press.
   - Comprehensive reference for practical Bayesian methods
   - Used for: MCMC, hierarchical models, model checking

4. **Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017)**. "Variational inference: A review for statisticians." *Journal of the American Statistical Association*, 112(518), 859-877.
   - Modern VI techniques, ADVI, BBVI
   - Used for: Variational inference implementation

5. **Gelman, A., et al. (2013)**. *Bayesian Data Analysis*, Chapter 2.
   - Conjugate prior families, posterior computation
   - Used for: Beta-Binomial, Gamma-Poisson, etc.

6. **Minka, T. (2000)**. "Estimating a Dirichlet distribution." Technical report, MIT.
   - Parameter estimation for Dirichlet distributions
   - Used for: Dirichlet-Multinomial implementation

7. **Kruschke, J. K. (2014)**. *Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan* (2nd ed.). Academic Press.
   - Pedagogical approach to Bayesian methods
   - Used for: Documentation, examples, interpretation

8. **Gelman, A., & Hill, J. (2006)**. *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
   - Hierarchical models, GLMs
   - Used for: Multilevel models, Bayesian GLMs

9. **Vehtari, A., Gelman, A., & Gabry, J. (2017)**. "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing*, 27(5), 1413-1432.
   - Model comparison, WAIC computation
   - Used for: Model selection implementation

10. **Hoffman, M. D., & Gelman, A. (2014)**. "The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research*, 15(1), 1593-1623.
    - NUTS algorithm, automatic tuning
    - Used for: State-of-the-art MCMC implementation

11. **Rasmussen, C. E., & Williams, C. K. I. (2006)**. *Gaussian Processes for Machine Learning*. MIT Press.
    - Comprehensive GP theory and practice
    - Used for: GP implementation, kernels, sparse approximations

12. **Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016)**. "Taking the human out of the loop: A review of Bayesian optimization." *Proceedings of the IEEE*, 104(1), 148-175.
    - Bayesian optimization, acquisition functions
    - Used for: Hyperparameter tuning, black-box optimization

---

## 16. Integration with Aprender

### 16.1 API Consistency

All Bayesian methods follow the `Estimator` trait:

```rust
pub trait BayesianEstimator: Estimator {
    /// Posterior predictive distribution (not just point estimate)
    fn predict_distribution(&self, x: &Matrix) -> Result<Distribution, AprenderError>;

    /// Credible intervals for predictions
    fn predict_interval(&self, x: &Matrix, level: f32) -> Result<(Vector, Vector), AprenderError>;

    /// Log marginal likelihood for model comparison
    fn log_marginal_likelihood(&self) -> f32;

    /// Posterior samples (if using MCMC)
    fn posterior_samples(&self) -> Option<&Matrix>;
}
```

### 16.2 Serialization

All models support SafeTensors format:

```rust
impl BayesianLinearRegression {
    pub fn save_safetensors(&self, path: &str) -> Result<(), AprenderError>;
    pub fn load_safetensors(path: &str) -> Result<Self, AprenderError>;
}
```

### 16.3 Ruchy Integration

```python
# Ruchy transpiles to native Rust
from aprender.bayesian import BayesianLinearRegression

model = BayesianLinearRegression()
model.fit(X_train, y_train)

# Get prediction intervals (not available in scikit-learn!)
mean, lower, upper = model.predict_interval(X_test, level=0.95)
```

---

## 17. Success Criteria

### 17.1 Functional Requirements

- ✅ All 45+ methods implemented with comprehensive tests
- ✅ MCMC converges (R̂ < 1.1) on standard test problems
- ✅ VI achieves within 1% of MCMC posterior means
- ✅ GP predictions match reference implementations (GPy, scikit-learn)
- ✅ Numerical stability on extreme inputs (verified via property tests)

### 17.2 Documentation Requirements

- ✅ Book chapter for each major topic (10+ chapters)
- ✅ Runnable example for each algorithm (30+ examples)
- ✅ All public APIs have rustdoc with examples
- ✅ Academic references for all methods

### 17.3 Performance Requirements

- ✅ Conjugate priors: sub-microsecond updates
- ✅ MCMC: 1000+ samples/sec for simple models
- ✅ VI: converges in < 100 iterations for standard models
- ✅ GP: handles n=1000 exactly, n=10k with sparse approximations
- ✅ All operations scale linearly or better with data size

---

## 18. Conclusion

This specification defines a complete, production-grade implementation of Bayesian probability theory for the Aprender library. Following Jaynes' principled approach and modern computational advances, the implementation will provide:

1. **Classical inference**: Conjugate priors, Bayesian regression, model selection
2. **Modern algorithms**: MCMC (NUTS), variational inference (ADVI), Gaussian processes
3. **Scalability**: Stochastic VI, sparse GP, distributed MCMC
4. **Quality**: EXTREME TDD, 95%+ coverage, comprehensive documentation

**Total Scope**: 45+ methods, 500+ tests, 10+ book chapters, 30+ examples

**Timeline**: 5 releases over 6-9 months (v0.7.0 to v1.1.0)

**Outcome**: Best-in-class Bayesian ML library in pure Rust, matching PyMC/Stan capabilities with superior performance and safety guarantees.
