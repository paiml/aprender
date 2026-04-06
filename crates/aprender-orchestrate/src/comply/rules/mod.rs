//! Stack Compliance Rules
//!
//! Individual rule implementations for checking cross-project consistency.

mod cargo_toml;
mod ci_workflow;
mod duplication;
mod makefile;

pub use cargo_toml::CargoTomlRule;
pub use ci_workflow::CiWorkflowRule;
pub use duplication::DuplicationRule;
pub use makefile::MakefileRule;
