//! Bayesian inference and probability methods.
//!
//! This module implements classical and modern Bayesian methods including:
//! - Conjugate priors (Beta-Binomial, Gamma-Poisson, Normal-InverseGamma, Dirichlet-Multinomial)
//! - Bayesian regression and classification
//! - Hierarchical models
//! - MCMC methods (future)
//! - Variational inference (future)
//! - Gaussian processes (future)
//!
//! # Design Philosophy
//!
//! Following E.T. Jaynes' "Probability Theory: The Logic of Science", this module
//! treats probability as an extension of logic under uncertainty. All methods provide:
//! - Prior specification with sensible defaults
//! - Exact posterior computation where tractable
//! - Credible intervals (not confidence intervals)
//! - Predictive distributions
//!
//! # Example: Beta-Binomial (Binary Outcomes)
//!
//! ```
//! use aprender::bayesian::BetaBinomial;
//!
//! // Prior: Beta(1, 1) = Uniform(0, 1)
//! let mut model = BetaBinomial::uniform();
//!
//! // Observe 7 successes in 10 trials
//! model.update(7, 10);
//!
//! // Posterior mean (point estimate)
//! let mean = model.posterior_mean();
//! assert!((mean - 0.6667).abs() < 0.001);
//!
//! // 95% credible interval
//! let (lower, upper) = model.credible_interval(0.95).expect("valid confidence level");
//! assert!(lower < mean && mean < upper);
//!
//! // Predict next trial
//! let prob_success = model.posterior_predictive();
//! ```
//!
//! # Example: Gamma-Poisson (Count Data)
//!
//! ```
//! use aprender::bayesian::GammaPoisson;
//!
//! // Prior: Gamma(0.001, 0.001) = weakly informative
//! let mut model = GammaPoisson::noninformative();
//!
//! // Observe counts: [3, 5, 4, 6, 2] events per interval
//! model.update(&[3, 5, 4, 6, 2]);
//!
//! // Posterior mean event rate
//! let mean = model.posterior_mean();
//! assert!((mean - 4.0).abs() < 0.5);
//!
//! // 95% credible interval for rate
//! let (lower, upper) = model.credible_interval(0.95).expect("valid confidence level");
//! assert!(lower < mean && mean < upper);
//! ```
//!
//! # Example: Normal-InverseGamma (Continuous Data)
//!
//! ```
//! use aprender::bayesian::NormalInverseGamma;
//!
//! // Prior: weakly informative for both mean and variance
//! let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("valid prior parameters");
//!
//! // Observe continuous data
//! model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
//!
//! // Posterior mean of μ (location)
//! let mean_mu = model.posterior_mean_mu();
//! assert!((mean_mu - 4.3).abs() < 0.3);
//!
//! // Posterior mean of σ² (variance)
//! let mean_var = model.posterior_mean_variance().expect("mean variance exists when alpha > 1");
//! assert!(mean_var > 0.0);
//!
//! // 95% credible interval for μ
//! let (lower, upper) = model.credible_interval_mu(0.95).expect("valid confidence level with alpha > 1");
//! assert!(lower < mean_mu && mean_mu < upper);
//! ```
//!
//! # Example: Dirichlet-Multinomial (Categorical Data)
//!
//! ```
//! use aprender::bayesian::DirichletMultinomial;
//!
//! // Prior: uniform over 3 categories [A, B, C]
//! let mut model = DirichletMultinomial::uniform(3);
//!
//! // Observe categorical data: 10 A's, 5 B's, 3 C's
//! model.update(&[10, 5, 3]);
//!
//! // Posterior probabilities for each category
//! let probs = model.posterior_mean();
//! assert!((probs[0] - 11.0/21.0).abs() < 0.01); // P(A) ≈ 0.524
//! assert!((probs[1] - 6.0/21.0).abs() < 0.01);  // P(B) ≈ 0.286
//! assert!((probs[2] - 4.0/21.0).abs() < 0.01);  // P(C) ≈ 0.190
//!
//! // Probabilities sum to 1
//! assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
//!
//! // 95% credible intervals for each category
//! let intervals = model.credible_intervals(0.95).expect("valid confidence level");
//! for i in 0..3 {
//!     assert!(intervals[i].0 < probs[i] && probs[i] < intervals[i].1);
//! }
//! ```

mod conjugate;
mod logistic;
mod regression;

pub use conjugate::{BetaBinomial, DirichletMultinomial, GammaPoisson, NormalInverseGamma};
pub use logistic::BayesianLogisticRegression;
pub use regression::BayesianLinearRegression;
