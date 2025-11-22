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
//! let (lower, upper) = model.credible_interval(0.95);
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
//! let (lower, upper) = model.credible_interval(0.95);
//! assert!(lower < mean && mean < upper);
//! ```

mod conjugate;

pub use conjugate::{BetaBinomial, DirichletMultinomial, GammaPoisson, NormalInverseGamma};
