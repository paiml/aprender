//! 5-phase transpilation pipeline with Jidoka stop-on-error validation.
//!
//! The pipeline orchestrates the transformation of source code to Rust:
//! 1. Analysis - Detects languages and dependencies
//! 2. Transpilation - Converts source to Rust using external transpilers
//! 3. Optimization - Applies MoE-based backend selection for GPU/SIMD
//! 4. Validation - Verifies semantic equivalence using syscall tracing
//! 5. Build - Compiles the final binary

#![allow(dead_code, unused_imports)]

mod execution;
mod stages;
#[cfg(test)]
mod tests;
mod types;

// Re-export all public types from types module
pub use types::{
    PipelineContext, PipelineOutput, PipelineStage, ValidationResult, ValidationStrategy,
};

// Re-export execution pipeline
pub use execution::TranspilationPipeline;

// Re-export all stage implementations
pub use stages::{
    AnalysisStage, BuildStage, OptimizationStage, TranspilationStage, ValidationStage,
};
