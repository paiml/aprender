//! APR Converter Coverage Tests - Type & Helper Coverage
//! Split from coverage.rs (PMAT-197) for file size reduction.
//!
//! Contains: TensorExpectation coverage, Architecture coverage,
//! Source type coverage, ValidationConfig, QuantizationType, Compression,
//! ImportOptions, ConvertOptions, TensorStats, internal helper function tests
//! (ROSETTA-ML-001), TensorAccumulator, quantization roundtrip, Pygmy-based tests,
//! ExportFormat/ExportOptions, MergeOptions/MergeReport.

#[allow(unused_imports)]
use super::super::*;

#[cfg(test)]
#[path = "coverage_types_tests.rs"]
mod tests;
