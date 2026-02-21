//! SQLite-style conversion test harness for SafeTensors <-> APR round-trips.
//!
//! Uses `TempDir` for RAII cleanup (no manual `fs::remove_file`), pygmy builders
//! for input data, and read-back verification with configurable tolerance.
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::format::test_factory::harness::ConversionTestHarness;
//! use crate::format::test_factory::PygmyConfig;
//!
//! ConversionTestHarness::assert_import_ok(PygmyConfig::llama_style());
//! ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::default());
//! ```

use super::{build_pygmy_apr_with_config, build_pygmy_safetensors_with_config, PygmyConfig};
use crate::format::converter::{
    apr_export, apr_import, ExportFormat, ExportOptions, ImportOptions,
};
use crate::format::v2::AprV2Reader;
use crate::serialization::safetensors::MappedSafeTensors;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Tolerance thresholds per dtype for tensor data comparison.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_field_names)] // Postfix naming is intentional for clarity
pub(crate) struct ToleranceConfig {
    pub(crate) f32_atol: f32,
    pub(crate) f16_atol: f32,
    pub(crate) q8_atol: f32,
    pub(crate) q4_atol: f32,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            f32_atol: 1e-6,
            f16_atol: 1e-3,
            q8_atol: 0.1,
            q4_atol: 0.5,
        }
    }
}

/// A single tensor mismatch found during verification.
#[derive(Debug)]
pub(crate) struct TensorMismatch {
    pub(crate) tensor_name: String,
    pub(crate) kind: MismatchKind,
}

/// What went wrong with a tensor comparison.
#[derive(Debug)]
pub(crate) enum MismatchKind {
    Missing,
    /// T-QKV-02: Output contains a tensor not present in source (possible fusion/split)
    Extra,
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    DataMismatch {
        index: usize,
        expected: f32,
        actual: f32,
        tolerance: f32,
    },
}

impl core::fmt::Display for TensorMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.kind {
            MismatchKind::Missing => {
                write!(f, "tensor '{}': missing in output", self.tensor_name)
            }
            MismatchKind::Extra => {
                write!(
                    f,
                    "tensor '{}': extra in output (not in source)",
                    self.tensor_name
                )
            }
            MismatchKind::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor '{}': shape mismatch expected={:?} actual={:?}",
                    self.tensor_name, expected, actual
                )
            }
            MismatchKind::DataMismatch {
                index,
                expected,
                actual,
                tolerance,
            } => {
                write!(
                    f,
                    "tensor '{}': data[{}] expected={} actual={} (tol={})",
                    self.tensor_name, index, expected, actual, tolerance
                )
            }
        }
    }
}

/// Result of a verification pass.
#[derive(Debug)]
pub(crate) struct VerificationResult {
    pub(crate) mismatches: Vec<TensorMismatch>,
}

impl VerificationResult {
    /// Panics with detailed info if any mismatches were found.
    pub(crate) fn assert_passed(&self) {
        if !self.mismatches.is_empty() {
            let msgs: Vec<String> = self.mismatches.iter().map(ToString::to_string).collect();
            panic!(
                "Verification failed with {} mismatch(es):\n  {}",
                self.mismatches.len(),
                msgs.join("\n  ")
            );
        }
    }

    #[must_use]
    pub(crate) fn passed(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// RAII conversion test harness. The `TempDir` is dropped (cleaned up)
/// when the harness goes out of scope.
pub(crate) struct ConversionTestHarness {
    dir: TempDir,
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    /// Original pygmy tensor data for verification (name -> (data, shape))
    source_tensors: Vec<(String, Vec<f32>, Vec<usize>)>,
    pub(crate) tolerance: ToleranceConfig,
}

include!("harness_impl.rs");
include!("collection.rs");
include!("harness_strict_tests.rs");
include!("harness_roundtrip_tests.rs");
