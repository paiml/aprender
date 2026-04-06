//! Orchestration Recipes Module
//!
//! Provides high-level workflows for integrating experiment tracking,
//! cost-performance benchmarking, and sovereign deployment pipelines
//! with the Batuta orchestration framework.

mod benchmark;
mod cicd;
mod experiment_tracking;
mod research_artifact;
mod sovereign_deployment;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types to maintain the same public API
pub use benchmark::CostPerformanceBenchmarkRecipe;
pub use cicd::CiCdBenchmarkRecipe;
pub use experiment_tracking::ExperimentTrackingRecipe;
pub use research_artifact::ResearchArtifactRecipe;
pub use sovereign_deployment::{SovereignDeploymentConfig, SovereignDeploymentRecipe};
pub use types::{ExperimentTrackingConfig, RecipeResult};

// Re-export experiment types that are used in the public API
pub use crate::experiment::{
    ComputeDevice, CostPerformancePoint, CreditRole, ModelParadigm, PlatformEfficiency,
};
