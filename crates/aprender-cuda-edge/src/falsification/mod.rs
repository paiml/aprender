//! Falsification checklist and reporting.
//!
//! Tracks 50 falsification claims across all frameworks. Each claim
//! represents a property that tests should attempt to falsify.
//!
//! # Coverage Calculation
//!
//! Coverage = (verified + violated) / (total - skipped)
//!
//! - **Verified**: Property held under all tests
//! - **Violated**: Property was falsified (bug found!)
//! - **Skipped**: Test was skipped (e.g., no GPU available)
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::falsification::{
//!     FalsificationReport, all_claims, claims_for_framework, Framework
//! };
//!
//! // All 50 claims
//! assert_eq!(all_claims().len(), 50);
//!
//! // Filter by framework
//! let nf_claims = claims_for_framework(Framework::NullFuzzer);
//! assert_eq!(nf_claims.len(), 10);
//!
//! // Track progress
//! let mut report = FalsificationReport::new();
//! report.mark_verified("NF-001");
//! assert_eq!(report.coverage(), 1.0 / 50.0);
//! ```

pub mod checklist;
pub mod report;

pub use checklist::{all_claims, claims_for_framework, ClaimStatus, FalsificationClaim, Framework};
pub use report::FalsificationReport;
