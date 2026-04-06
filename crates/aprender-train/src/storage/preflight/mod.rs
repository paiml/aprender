//! Pre-flight Validation System (Jidoka)
//!
//! Validates data integrity and environment before training starts.
//! Catches 30-50% of ML pipeline failures before training.
//!
//! # Toyota Way: 自働化 (Jidoka)
//!
//! Built-in quality through automatic defect detection at source.
//!
//! # Example
//!
//! ```
//! use entrenar::storage::preflight::{Preflight, PreflightCheck, CheckResult};
//!
//! let preflight = Preflight::new()
//!     .add_check(PreflightCheck::no_nan_values())
//!     .add_check(PreflightCheck::no_inf_values())
//!     .add_check(PreflightCheck::min_samples(2))
//!     .add_check(PreflightCheck::disk_space_mb(1));
//!
//! let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
//! let results = preflight.run(&data);
//! assert!(results.all_passed());
//! ```

mod check_result;
mod checks;
mod results;
mod types;
mod validator;

// Re-export all public types for API compatibility
pub use check_result::CheckResult;
pub use checks::PreflightCheck;
pub use results::PreflightResults;
pub use types::{CheckMetadata, CheckType, PreflightContext, PreflightError};
pub use validator::Preflight;
