//! APR QA Scenario Generator
//!
//! Property-based test scenario generation for model qualification.
//! Implements the Popperian falsification methodology from the APR Playbook Spec.
//!
//! # Design Philosophy
//!
//! > "The criterion of the scientific status of a theory is its falsifiability."
//! > — Karl Popper, *Conjectures and Refutations* (1963)
//!
//! Every generated scenario is a falsifiable hypothesis about model behavior.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Allow common patterns
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::needless_raw_string_hashes)]
// Allow common patterns in test code
#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::disallowed_methods))] // test assertions — unwrap acceptable
#![cfg_attr(test, allow(clippy::redundant_closure_for_method_calls))]
#![cfg_attr(test, allow(clippy::redundant_clone))]

pub mod bootstrapper;
pub mod error;
pub mod hf_parity;
pub mod kernel_class;
pub mod kernel_coverage;
pub mod kernel_profile;
pub mod models;
pub mod oracle;
pub mod proptest_impl;
pub mod scenario;

pub use bootstrapper::{bootstrap_playbook, to_yaml, BootstrapConfig, BootstrappedPlaybook};
pub use error::{Error, Result};
pub use hf_parity::{hash_prompt, GoldenOutput, HfParityOracle, TensorDiff, Tolerance};
pub use kernel_class::{models_in_class, KernelClass};
pub use kernel_coverage::{
    BindingVerification, BindingVerificationReport, ClassSummary, CoverageContext, CoverageReport,
    ImplementationStatus, KernelBinding, KernelGap, ModelCoverage, ModelCoverageSummary,
};
pub use kernel_profile::{
    profile_from_constraints, ArchConstraints, ArchSizeVariant, KernelOp, KernelProfile,
    PromptCategory,
};
pub use models::{ModelId, ModelRegistry, SizeCategory};
pub use oracle::{Oracle, OracleResult};
pub use scenario::{AprTool, Backend, Format, Modality, QaScenario, ScenarioGenerator, TraceLevel};
