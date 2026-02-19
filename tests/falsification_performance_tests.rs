#![allow(clippy::disallowed_methods)]
//! F081-F100: Performance Regression Falsification Tests
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md S9.4
//!
//! STATUS: IMPLEMENTED - Infrastructure verified, hardware tests gracefully skip
//!
//! These tests verify performance targets (2x llama.cpp) are met.
//! Tests requiring GPU hardware skip gracefully when unavailable.
//!
//! FALSIFICATION: If performance doesn't meet targets, optimization incomplete.

use std::time::Instant;

/// Check if CUDA is available
fn cuda_available() -> bool {
    std::path::Path::new("/proc/driver/nvidia/version").exists()
        || std::process::Command::new("nvidia-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
}

/// Calculate coefficient of variation
fn calculate_cv(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance: f64 =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    let std_dev = variance.sqrt();
    (std_dev / mean) * 100.0
}

include!("includes/falsification_performance_part_01.rs");
include!("includes/falsification_performance_part_02.rs");
