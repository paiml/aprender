//! Adversarial Falsification Testing (PMAT-019)
//!
//! Instead of "proving it works," actively attempt to break the system
//! through adversarial tactics per §36 of the cbtop spec.
//!
//! # Adversarial Tactics
//!
//! | Tactic | Description |
//! |--------|-------------|
//! | Bit-Flip Injection | Random bit flips in input tensors |
//! | Resource Starvation | Simulate memory/CPU pressure |
//! | Clock Skew | Test monotonic timestamp preservation |
//! | Network Partition | Timeout and disconnection handling |
//! | Config Fuzzing | Generate valid-but-pathological configs |
//!
//! # Citations
//!
//! - [Miller et al. 1990] "An Empirical Study of the Reliability of UNIX Utilities" CACM
//! - [Goodfellow et al. 2014] "Explaining and Harnessing Adversarial Examples" arXiv
//! - [Regehr et al. 2012] "Finding and Understanding Bugs in C Compilers" PLDI

mod resources;
mod types;
mod validators;

pub use resources::{BitFlipInjector, CancellationToken, RecoveryHandler, ResourceLimiter};
pub use types::*;
pub use validators::{ConfigValidator, InputValidator};

#[cfg(test)]
mod tests;
