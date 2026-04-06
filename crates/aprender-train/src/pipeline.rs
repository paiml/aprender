//! Pipeline — standardized connectors for training pipelines
//!
//! Re-exports pipeline components from the training module to provide
//! a unified pipeline interface (MTD-05: Pipeline Glue Code Minimization).

pub use crate::train::{PipelineAction, PipelineActivationBuffer, PipelineStage};
