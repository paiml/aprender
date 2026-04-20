#![allow(dead_code, unused_imports)]
//! Experiment Tracking Integration Module
//!
//! Integrates with Entrenar Experiment Tracking Spec v1.8.0 for orchestrating
//! ML experiment workflows with full traceability, cost optimization, and
//! academic research support.
//!
//! # Entrenar CLI (v0.2.4)
//!
//! The entrenar crate provides a comprehensive CLI:
//!
//! ```bash
//! # Training (YAML Mode v1.0)
//! entrenar train config.yaml           # Train from declarative YAML
//! entrenar validate config.yaml        # Validate configuration
//! entrenar init --template lora        # Generate config template
//!
//! # Model Operations
//! entrenar quantize model.safetensors --bits 4
//! entrenar merge model1.st model2.st --method ties
//!
//! # Research Workflows
//! entrenar research init --id my-dataset
//! entrenar research cite artifact.yaml --format bibtex
//!
//! # Inspection & Auditing
//! entrenar inspect model.safetensors   # Model/data inspection
//! entrenar audit data.parquet --type bias
//! entrenar monitor data.parquet        # Drift detection
//!
//! # Benchmarking (entrenar-bench)
//! entrenar-bench temperature --start 1.0 --end 8.0
//! entrenar-bench cost-performance --gpu a100-80gb
//! ```
//!
//! # MCP Tooling (pmcp v2.3)
//!
//! The stack uses PAIML's pmcp SDK (github.com/paiml/rust-mcp-sdk) for both
//! Model Context Protocol client and server roles — full TypeScript SDK parity
//! covering JSON-RPC framing, stdio/SSE/WebSocket transports, tool registration,
//! and session lifecycle.
//!
//! # Features
//! - ComputeDevice abstraction (CPU/GPU/TPU/AppleSilicon)
//! - EnergyMetrics and CostMetrics for efficiency tracking
//! - ModelParadigm classification
//! - CostPerformanceBenchmark with Pareto frontier analysis
//! - SovereignDistribution for air-gapped deployments
//! - ResearchArtifact with ORCID/CRediT academic support
//! - CitationMetadata for BibTeX/CFF generation
//! - Experiment tree visualization for run comparison (MLflow replacement)
//! - YAML Mode Training v1.0 declarative configuration

// Submodules
pub mod benchmark;
pub mod metrics;
pub mod research;
pub mod run;
pub mod sovereign;
pub mod tree;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export all public types from submodules for convenient access

// Types module
pub use types::{
    AppleChip, ComputeDevice, ComputeIntensity, CpuArchitecture, ExperimentError, GpuVendor,
    ModelParadigm, PlatformEfficiency, TpuVersion,
};

// Metrics module
pub use metrics::{CostMetrics, EnergyMetrics};

// Benchmark module
pub use benchmark::{CostPerformanceBenchmark, CostPerformancePoint};

// Research module
pub use research::{
    CitationMetadata, CitationType, CreditRole, Orcid, PreRegistration, ResearchArtifact,
    ResearchContributor,
};

// Run module
pub use run::{ExperimentRun, ExperimentStorage, InMemoryExperimentStorage, RunStatus};

// Sovereign module
pub use sovereign::{
    ArtifactSignature, ArtifactType, OfflineRegistryConfig, SignatureAlgorithm, SovereignArtifact,
    SovereignDistribution,
};
