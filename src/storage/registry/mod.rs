//! Model Registry with Staging Workflows (MLOPS-008)
//!
//! Kanban-style model lifecycle management.
//!
//! # Toyota Way: (Kanban)
//!
//! Visual workflow stages for model promotion with pull-based progression.
//! Models flow: None -> Development -> Staging -> Production -> Archived
//!
//! # Example
//!
//! ```ignore
//! use entrenar::storage::registry::{ModelRegistry, ModelStage, InMemoryRegistry};
//!
//! let mut registry = InMemoryRegistry::new();
//! registry.register_model("llama-7b-finetuned", "path/to/model.safetensors")?;
//! registry.transition_stage("llama-7b-finetuned", 1, ModelStage::Staging, Some("alice"))?;
//! ```

mod comparison;
mod error;
mod memory;
mod policy;
mod stage;
mod traits;
mod transition;
mod version;

// Re-export all public types for API compatibility
pub use comparison::{Comparison, MetricRequirement, VersionComparison};
pub use error::{RegistryError, Result};
pub use memory::InMemoryRegistry;
pub use policy::{PolicyCheckResult, PromotionPolicy};
pub use stage::ModelStage;
pub use traits::ModelRegistry;
pub use transition::StageTransition;
pub use version::ModelVersion;
