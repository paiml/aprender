//! Property-based convergence tests for optimizers
//!
//! These tests validate optimizer correctness using:
//! - Quadratic convergence (convex, optimal solution at origin)
//! - Rosenbrock function (non-convex, tests valley navigation)
//! - Ill-conditioned problems (tests numerical stability)
//! - High-dimensional problems (tests scalability)
//! - Numerical edge cases (very small/large gradients)
//!
//! Tests are organized by optimizer type:
//! - `sgd_tests` - SGD optimizer tests
//! - `adam_tests` - Adam optimizer tests
//! - `adamw_tests` - AdamW optimizer tests
//! - `integration_tests` - Cross-optimizer comparisons

mod helpers;

mod adam_tests;
mod adamw_tests;
mod integration_tests;
mod sgd_tests;
