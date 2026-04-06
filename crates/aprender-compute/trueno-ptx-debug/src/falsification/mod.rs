//! Popperian Falsification Framework
//!
//! Following Karl Popper's falsificationism: we cannot prove PTX correct,
//! but we can systematically attempt to falsify it. Each bug class represents
//! a falsifiable hypothesis.
//!
//! The framework implements a 100-point falsification matrix with tests
//! grouped into 10 categories.

mod framework;
mod tests;

pub use framework::{
    Category, FalsificationRegistry, FalsificationReport, FalsificationTest, TestResult,
};
