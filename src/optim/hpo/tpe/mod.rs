//! Tree-structured Parzen Estimator (TPE) optimizer
//!
//! Based on Bergstra et al. (2011) - Algorithms for Hyper-Parameter Optimization

mod optimizer;
mod sampling;
#[cfg(test)]
mod tests;

pub use optimizer::TPEOptimizer;
