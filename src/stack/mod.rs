//! PAIML Stack Dependency Orchestration
//!
//! This module provides dependency management and coordinated release capabilities
//! for the PAIML (Pragmatic AI Labs) Rust ecosystem.
//!
//! ## Commands
//!
//! - `batuta stack check` - Dependency health analysis
//! - `batuta stack release` - Coordinated multi-crate release
//! - `batuta stack status` - Stack dashboard
//! - `batuta stack sync` - Dependency synchronization
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Pre-flight checks stop broken releases
//! - **Just-in-Time**: Pull-based release ordering
//! - **Heijunka**: Version alignment across stack
//! - **Genchi Genbutsu**: Real-time crates.io verification

#![allow(unused_imports)] // Public API re-exports

pub mod checker;
pub mod crates_io;
pub mod diagnostics;
pub mod diagnostics_ml;
pub mod drift;
pub mod graph;
pub mod hero_image;
pub mod publish_status;
pub mod quality;
pub mod quality_checker;
pub mod quality_format;
pub mod releaser;
pub mod releaser_preflight;
pub mod releaser_types;
pub mod tree;
pub mod tui;
pub mod types;

pub use checker::StackChecker;
pub use crates_io::CratesIoClient;
pub use diagnostics::{
    render_dashboard, AndonStatus, Anomaly, AnomalyCategory, ComponentMetrics, ComponentNode,
    GraphMetrics, HealthStatus, HealthSummary, StackDiagnostics,
};
pub use diagnostics_ml::{ErrorForecaster, ForecastMetrics, IsolationForest};
pub use drift::{format_drift_errors, format_drift_json, DriftChecker, DriftReport, DriftSeverity};
pub use graph::DependencyGraph;
pub use hero_image::{HeroImageResult, ImageFormat};
pub use publish_status::{
    format_report_json as format_publish_status_json,
    format_report_text as format_publish_status_text, CrateStatus, PublishAction,
    PublishStatusCache, PublishStatusReport, PublishStatusScanner,
};
pub use quality::{
    ComponentQuality, QualityGrade, QualityIssue, QualitySummary, Score, StackLayer,
    StackQualityReport,
};
pub use quality_checker::QualityChecker;
pub use quality_format::{
    format_report_json as format_quality_report_json,
    format_report_text as format_quality_report_text,
};
pub use releaser_types::{format_plan_text, BumpType, ReleaseConfig, ReleaseResult, ReleasedCrate};
pub use types::*;

/// PAIML stack crate names for identification
pub const PAIML_CRATES: &[&str] = &[
    // Core compute layer
    "trueno",
    "trueno-viz",
    "trueno-db",
    "trueno-graph",
    "trueno-rag",
    "trueno-rag-cli",
    "trueno-zram-core",
    "trueno-ublk",
    "trueno-cupti",
    "cbtop",
    // ML layer
    "aprender",
    "aprender-shell",
    "aprender-tsp",
    "realizar",
    "alimentar",
    "entrenar",
    // Infrastructure layer
    "renacer",
    "repartir",
    "pacha",
    "duende",
    "ttop",
    "pzsh",
    "forjar",
    // Simulation & games
    "simular",
    "jugar",
    "jugar-probar",
    // Hardware integration
    "manzana",
    "pepita",
    "wos",
    // Speech & inference
    "whisper-apr",
    // Quality/Tooling
    "certeza",
    "provable-contracts",
    "tiny-model-ground-truth",
    "batuta",
    "presentar",
    "presentar-cli",
    "presentar-core",
    "ruchy",
    "decy",
    "depyler",
    "pmat",
    // Transpilers
    "bashrs",
    "bashrs-runtime",
    "bashrs-oracle",
    // Intelligence
    "organizational-intelligence-plugin",
    // Documentation
    "sovereign-ai-stack-book",
    "apr-cookbook",
    "alm-cookbook",
    "pres-cookbook",
    "batuta-cookbook",
    // Canary benchmarks
    "qwen-train-canary",
    "qwen-coder-deploy",
];

/// Check if a crate name is part of the PAIML stack
pub fn is_paiml_crate(name: &str) -> bool {
    PAIML_CRATES.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paiml_crate_detection() {
        assert!(is_paiml_crate("trueno"));
        assert!(is_paiml_crate("aprender"));
        assert!(is_paiml_crate("batuta"));
        assert!(!is_paiml_crate("serde"));
        assert!(!is_paiml_crate("tokio"));
        assert!(!is_paiml_crate("arrow"));
    }

    #[test]
    fn test_cookbook_detection() {
        assert!(is_paiml_crate("apr-cookbook"));
        assert!(is_paiml_crate("alm-cookbook"));
        assert!(is_paiml_crate("pres-cookbook"));
        assert!(is_paiml_crate("pmat"));
    }

    #[test]
    fn test_new_stack_crates() {
        // Infrastructure
        assert!(is_paiml_crate("duende"));
        assert!(is_paiml_crate("ttop"));
        assert!(is_paiml_crate("forjar"));
        // Simulation & games
        assert!(is_paiml_crate("simular"));
        assert!(is_paiml_crate("jugar"));
        assert!(is_paiml_crate("jugar-probar"));
        // Hardware
        assert!(is_paiml_crate("manzana"));
        // Compression
        assert!(is_paiml_crate("trueno-zram-core"));
        assert!(is_paiml_crate("trueno-ublk"));
        // GPU monitoring
        assert!(is_paiml_crate("trueno-cupti"));
        assert!(is_paiml_crate("cbtop"));
        // Quality
        assert!(is_paiml_crate("provable-contracts"));
        assert!(is_paiml_crate("tiny-model-ground-truth"));
    }

    #[test]
    fn test_paiml_crates_count() {
        // Stack now has 47 crates (added qwen-train-canary, qwen-coder-deploy)
        assert!(PAIML_CRATES.len() >= 47);
    }
}
