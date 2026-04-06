//! Popperian Falsification Tests for Format×Device Matrix
//!
//! Reference: docs/specifications/model-fixture-setup-teardown.md
//!
//! This module subjects ALL format×device combinations to rigorous testing.
//! Each test is a "risky prediction" - if it fails, the implementation is falsified.
//!
//! ## Synthetic Fixture Matrix (T001-T006)
//!
//! | Format      | CPU  | CUDA | Status |
//! |-------------|------|------|--------|
//! | GGUF        | T001 | T002 | FIXTURE BUG: Q4_K data size calculation |
//! | APR         | T003 | T004 | ✓ PMAT-111 FIXED: Tensor indexing now works |
//! | SafeTensors | T005 | T006 | ✓ CPU WORKS, CUDA NOT WIRED |
//!
//! ## Real Model Tests (T100+) - EMPIRICAL SCIENCE
//!
//! Uses `artifacts/models/` - run `./scripts/sync-models.sh` to populate.
//!
//! | Test | Format | Device | Model | Status |
//! |------|--------|--------|-------|--------|
//! | T100 | GGUF   | CPU    | qwen2-0.5b-q4_0.gguf | ✓ CORROBORATED (argmax=262) |
//! | T200 | SafeTensors | CPU | qwen2-0.5b.safetensors | ✓ CORROBORATED (argmax=262) |
//! | T201 | APR    | CPU    | synthetic fixture | ✓ EMPIRICAL (PMAT-111 FIXED) |
//!
//! ## Falsification Conditions
//!
//! - F012: Output contains NaN → FALSIFIED
//! - F013: Output contains Inf → FALSIFIED
//! - F015: Argmax(A) != Argmax(B) for same input → FALSIFIED
//! - F021: |Output(CPU) - Output(CUDA)| > ε → FALSIFIED
//!
//! ## Current Status (2026-01-26)
//!
//! **T100 (GGUF:CPU):** CORROBORATED - Real Qwen2-0.5B produces 151936 valid logits.
//! **T200 (APR:CPU):** TESTABLE - Real APR model from GGUF conversion.
//! **T005 (SafeTensors:CPU):** CORROBORATED - 100 valid logits.
//!
//! Synthetic fixtures (T001-T006) have known bugs; they test the fixture generator,
//! not the inference engine. Use T100+ tests for real falsification.

use super::{ModelConfig, ModelFixture, ModelFormat};

/// Device enum for test matrix
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU device (SIMD-accelerated)
    Cpu,
    /// CUDA GPU device
    Cuda,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            Device::Cuda => write!(f, "CUDA"),
        }
    }
}

/// Result of a forward pass
#[derive(Debug)]
pub struct ForwardResult {
    /// Output logits from model forward pass
    pub logits: Vec<f32>,
    /// Model format (GGUF, APR, SafeTensors)
    pub format: ModelFormat,
    /// Device used for inference
    pub device: Device,
    /// Number of tokens processed
    pub tokens_processed: usize,
}

impl ForwardResult {
    /// F012: Check for NaN
    pub fn has_nan(&self) -> bool {
        self.logits.iter().any(|x| x.is_nan())
    }

    /// F013: Check for Inf
    pub fn has_inf(&self) -> bool {
        self.logits.iter().any(|x| x.is_infinite())
    }

    /// F015: Get argmax for comparison
    pub fn argmax(&self) -> Option<usize> {
        self.logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// F016/F021: L2 distance to reference
    pub fn l2_distance(&self, other: &ForwardResult) -> f32 {
        if self.logits.len() != other.logits.len() {
            return f32::INFINITY;
        }
        self.logits
            .iter()
            .zip(other.logits.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Relative L2 as percentage
    pub fn relative_l2(&self, other: &ForwardResult) -> f32 {
        let norm: f32 = other.logits.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return f32::INFINITY;
        }
        self.l2_distance(other) / norm * 100.0
    }
}

/// Falsification result
#[derive(Debug)]
pub struct FalsificationResult {
    /// Falsification condition ID (e.g., "F012")
    pub id: &'static str,
    /// Human-readable description of the check
    pub description: &'static str,
    /// Whether the check passed (not falsified)
    pub passed: bool,
    /// Details about the result
    pub details: String,
}

impl FalsificationResult {
    /// Create a passing (corroborated) result
    pub fn pass(id: &'static str, desc: &'static str) -> Self {
        Self {
            id,
            description: desc,
            passed: true,
            details: "Corroborated".to_string(),
        }
    }

    /// Create a failing (falsified) result
    pub fn fail(id: &'static str, desc: &'static str, details: String) -> Self {
        Self {
            id,
            description: desc,
            passed: false,
            details,
        }
    }
}

/// Run falsification checks on a forward result
pub fn falsify(result: &ForwardResult) -> Vec<FalsificationResult> {
    let mut results = Vec::new();

    // F012: NaN check
    results.push(if result.has_nan() {
        FalsificationResult::fail(
            "F012",
            "Output contains NaN",
            format!(
                "FALSIFIED: {}:{} produced NaN logits",
                result.format, result.device
            ),
        )
    } else {
        FalsificationResult::pass("F012", "No NaN in output")
    });

    // F013: Inf check
    results.push(if result.has_inf() {
        FalsificationResult::fail(
            "F013",
            "Output contains Inf",
            format!(
                "FALSIFIED: {}:{} produced Inf logits",
                result.format, result.device
            ),
        )
    } else {
        FalsificationResult::pass("F013", "No Inf in output")
    });

    // Check logits are non-empty
    results.push(if result.logits.is_empty() {
        FalsificationResult::fail(
            "F099",
            "Output is empty",
            format!(
                "FALSIFIED: {}:{} produced empty logits",
                result.format, result.device
            ),
        )
    } else {
        FalsificationResult::pass("F099", "Output is non-empty")
    });

    results
}

/// Compare two results for parity
pub fn falsify_parity(a: &ForwardResult, b: &ForwardResult) -> Vec<FalsificationResult> {
    let mut results = Vec::new();

    // F015: Argmax parity
    let argmax_a = a.argmax();
    let argmax_b = b.argmax();
    results.push(if argmax_a != argmax_b {
        FalsificationResult::fail(
            "F015",
            "Argmax mismatch",
            format!(
                "FALSIFIED: {}:{} argmax={:?} vs {}:{} argmax={:?}",
                a.format, a.device, argmax_a, b.format, b.device, argmax_b
            ),
        )
    } else {
        FalsificationResult::pass("F015", "Argmax matches")
    });

    // F021: L2 distance
    let l2_pct = a.relative_l2(b);
    results.push(if l2_pct > 10.0 {
        FalsificationResult::fail(
            "F021",
            "L2 distance > 10%",
            format!(
                "FALSIFIED: {}:{} vs {}:{} L2={:.2}%",
                a.format, a.device, b.format, b.device, l2_pct
            ),
        )
    } else {
        FalsificationResult::pass("F021", "L2 within tolerance")
    });

    results
}

// =============================================================================
// Format×Device Forward Functions
// These use the public API - implementations will be filled in as API stabilizes
// =============================================================================

use crate::error::Result;

/// GGUF CPU forward - uses public gguf module
pub fn forward_gguf_cpu(fixture: &ModelFixture, tokens: &[u32]) -> Result<ForwardResult> {
    forward_gguf_cpu_path(fixture.path(), tokens)
}

/// GGUF CPU forward from path - for testing with real models
pub fn forward_gguf_cpu_path(path: &std::path::Path, tokens: &[u32]) -> Result<ForwardResult> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let logits = model.forward(tokens)?;

    Ok(ForwardResult {
        logits,
        format: ModelFormat::Gguf,
        device: Device::Cpu,
        tokens_processed: tokens.len(),
    })
}

/// APR CPU forward - uses public apr module
pub fn forward_apr_cpu(fixture: &ModelFixture, tokens: &[u32]) -> Result<ForwardResult> {
    forward_apr_cpu_path(fixture.path(), tokens)
}

/// APR CPU forward from path - for testing with real models
pub fn forward_apr_cpu_path(path: &std::path::Path, tokens: &[u32]) -> Result<ForwardResult> {
    use crate::apr::AprV2Model;

    let model = AprV2Model::load(path)?;
    let logits = model.forward(tokens)?;

    Ok(ForwardResult {
        logits,
        format: ModelFormat::Apr,
        device: Device::Cpu,
        tokens_processed: tokens.len(),
    })
}

/// SafeTensors CPU forward - converts to AprTransformer then runs forward
pub fn forward_safetensors_cpu(fixture: &ModelFixture, tokens: &[u32]) -> Result<ForwardResult> {
    use crate::safetensors_infer::SafetensorsToAprConverter;

    let transformer = SafetensorsToAprConverter::convert(fixture.path())?;
    let logits = transformer.forward(tokens)?;

    Ok(ForwardResult {
        logits,
        format: ModelFormat::SafeTensors,
        device: Device::Cpu,
        tokens_processed: tokens.len(),
    })
}

/// GGUF CUDA forward
#[cfg(feature = "cuda")]
pub fn forward_gguf_cuda(fixture: &ModelFixture, tokens: &[u32]) -> Result<ForwardResult> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let mapped = MappedGGUFModel::from_path(fixture.path())?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(cpu_model, 0)?;
    let logits = cuda_model.forward_cuda(tokens)?;

    Ok(ForwardResult {
        logits,
        format: ModelFormat::Gguf,
        device: Device::Cuda,
        tokens_processed: tokens.len(),
    })
}

/// APR CUDA forward
#[cfg(feature = "cuda")]
pub fn forward_apr_cuda(fixture: &ModelFixture, tokens: &[u32]) -> Result<ForwardResult> {
    use crate::apr::{AprV2Model, AprV2ModelCuda};

    let apr_model = AprV2Model::load(fixture.path())?;
    let mut cuda_model = AprV2ModelCuda::new(apr_model, 0)?;
    let logits = cuda_model.forward_cuda(tokens)?;

    Ok(ForwardResult {
        logits,
        format: ModelFormat::Apr,
        device: Device::Cuda,
        tokens_processed: tokens.len(),
    })
}

/// SafeTensors CUDA forward
#[cfg(feature = "cuda")]
pub fn forward_safetensors_cuda(_fixture: &ModelFixture, _tokens: &[u32]) -> Result<ForwardResult> {
    // SafeTensors CUDA path not yet wired up — returns UnsupportedOperation
    Err(crate::error::RealizarError::UnsupportedOperation {
        operation: "forward_safetensors_cuda".to_string(),
        reason: "SafeTensors CUDA path not yet wired to public API".to_string(),
    })
}

include!("falsification_tests_t005_safetensors.rs");
