//! Industry Baseline Validation (PMAT-016)
//!
//! Compare cbtop throughput against industry baselines (vLLM, TGI, Triton).
//! Per cbtop spec §21.7 and §21.8.
//!
//! # Design Principles
//!
//! - Use vLLM/llama.cpp as **reference**, not dependency
//! - Side-by-side validation without polluting Pure Rust codebase
//! - No foreign code in cbtop binary (F976)
//!
//! # Citations
//!
//! - [Satna 2026] "LLM Inference Benchmarking Framework" GitHub
//! - [vLLM 2023] "vLLM: Easy, Fast, Cheap LLM Serving with PagedAttention" UCB

mod comparison;
mod types;

pub use comparison::{BaselineComparison, BaselineValidator, ValidationSummary};
pub use types::{
    GpuClass, ServerBaseline, SingleComparison, SmHealth, ThroughputGrade, INDUSTRY_BASELINES,
    TGI_BASELINE, TRITON_BASELINE, VLLM_BASELINE,
};

#[cfg(test)]
mod tests;
