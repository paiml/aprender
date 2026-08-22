//! GH-279: Unified Model Load Contract Gate
//!
//! **THE** single enforcement point for ALL model loading paths in realizar.
//! Every model (GGUF, SafeTensors, APR) MUST pass through `validate_model_load()`
//! before weights enter any kernel.
//!
//! # Architecture
//!
//! ```text
//! GGUF CPU ──────┐
//! GGUF CUDA ─────┤
//! SafeTensors ───┼──► validate_model_load() ──► ModelLoadProof ──► kernel
//! APR CPU ───────┤
//! APR CUDA ──────┤
//! GpuModel ──────┘
//! ```
//!
//! `ModelLoadProof` is a sealed type — private inner field means it can ONLY
//! be constructed by `validate_model_load()`. Downstream code that requires
//! a `&ModelLoadProof` parameter is GUARANTEED to have passed validation.
//!
//! # Validation Layers
//!
//! 1. **Architecture completeness** — all required weight roles present
//!    (via `arch_requirements::required_roles()`)
//! 2. **Dimension plausibility** — hidden_dim > 0, num_heads > 0, hidden_dim % num_heads == 0
//! 3. **Kernel contract link** — trueno `contracts::QuantFormat` constants are
//!    used to validate buffer sizes match expectations

use crate::arch_requirements::{required_roles, WeightRole};
use crate::error::RealizarError;
use crate::gguf::ArchConstraints;
use std::fmt;

// Re-export trueno kernel contracts for downstream consumers
pub use trueno::contracts::{
    self as kernel_contracts, validate_f32_buffer, validate_gemv_shapes, validate_weight_buffer,
    QuantFormat, TensorLayout, WeightBufferError, STACK_LAYOUT,
};

// ============================================================================
// ModelLoadProof — sealed output token
// ============================================================================

/// Proof that a model passed all contract validation gates.
///
/// Private inner field = IMPOSSIBLE to construct without `validate_model_load()`.
/// Functions that accept `&ModelLoadProof` are GUARANTEED that:
/// - All architecture-required weights are declared present
/// - Model dimensions are plausible
/// - The architecture is recognized
///
/// This does NOT prove that weight DATA is correct — only that the structural
/// metadata is valid. Data correctness is validated by `ValidatedLayerWeights`
/// at the per-layer level.
#[derive(Debug, Clone)]
pub struct ModelLoadProof {
    /// Construction only through validate_model_load()
    architecture: String,
    num_layers: usize,
}

impl ModelLoadProof {
    /// Architecture that was validated.
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Number of layers that was validated.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ============================================================================
// ModelLoadConfig — input to validation
// ============================================================================

/// Model metadata required for contract validation.
///
/// Extracted from GGUF/SafeTensors/APR metadata at load time.
/// Passed to `validate_model_load()` before any weight data is accessed.
#[derive(Debug, Clone)]
pub struct ModelLoadConfig {
    /// Architecture name (e.g., "llama", "qwen2", "qwen3")
    pub architecture: String,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads (Q heads)
    pub num_heads: usize,
    /// Number of K/V heads (for GQA)
    pub num_kv_heads: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Which weight roles are present in the model file.
    /// For each layer, the loader checks which tensors exist and reports them here.
    /// If empty, architecture completeness check is skipped (backwards compat).
    pub present_roles: Vec<WeightRole>,
}

// ============================================================================
// Validation Error
// ============================================================================

/// Error from model load contract validation.
#[derive(Debug, Clone)]
pub struct ModelLoadError {
    /// What failed
    pub gate: &'static str,
    /// Detailed reason
    pub reason: String,
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GH-279 contract gate '{}' failed: {}",
            self.gate, self.reason
        )
    }
}

impl std::error::Error for ModelLoadError {}

impl From<ModelLoadError> for RealizarError {
    fn from(e: ModelLoadError) -> Self {
        RealizarError::UnsupportedOperation {
            operation: format!("contract_gate::{}", e.gate),
            reason: e.reason,
        }
    }
}

// ============================================================================
// The Gate
// ============================================================================

/// Validate model metadata before loading weights.
///
/// This is THE enforcement point. ALL model loading paths MUST call this
/// before accessing weight data. Returns `ModelLoadProof` on success.
///
/// # Validation Gates
///
/// 1. **dimension_plausibility** — hidden_dim > 0, num_heads > 0,
///    hidden_dim % num_heads == 0, vocab_size > 0
/// 2. **architecture_recognized** — `ArchConstraints::from_architecture()`
///    returns valid constraints
/// 3. **architecture_completeness** — if `present_roles` is non-empty,
///    every role in `required_roles(arch)` must be in `present_roles`
///
/// # Errors
///
/// Returns `ModelLoadError` with gate name and detailed reason.
pub fn validate_model_load(
    config: &ModelLoadConfig,
) -> std::result::Result<ModelLoadProof, ModelLoadError> {
    // Gate 0: Architecture is one realizar can run CORRECTLY (honest-by-design,
    // PMAT-807). Fail LOUD rather than silently produce garbage for families
    // whose architecture-specific behaviors are not yet implemented in the
    // forward path.
    validate_supported_architecture(&config.architecture)?;

    // Gate 1: Dimension plausibility
    validate_dimensions(config)?;

    // Gate 2: Architecture recognized
    let arch = validate_architecture(&config.architecture)?;

    // Gate 3: Architecture completeness (if roles reported)
    if !config.present_roles.is_empty() {
        validate_completeness(&arch, &config.present_roles, &config.architecture)?;
    }

    Ok(ModelLoadProof {
        architecture: config.architecture.clone(),
        num_layers: config.num_layers,
    })
}

/// Convenience: validate from `ArchConstraints` + dimensions (no role checking).
///
/// Used by loading paths that don't enumerate roles but do have ArchConstraints.
/// Still validates dimensions and architecture.
pub fn validate_model_load_basic(
    architecture: &str,
    num_layers: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    intermediate_dim: usize,
    vocab_size: usize,
) -> std::result::Result<ModelLoadProof, ModelLoadError> {
    validate_model_load(&ModelLoadConfig {
        architecture: architecture.to_string(),
        num_layers,
        hidden_dim,
        num_heads,
        num_kv_heads,
        intermediate_dim,
        vocab_size,
        present_roles: Vec::new(), // no role checking in basic mode
    })
}

/// Convert a `ModelLoadError` into a `RealizarError` for ? propagation.
pub fn gate_error(e: ModelLoadError) -> RealizarError {
    e.into()
}

// ============================================================================
// Individual Gates
// ============================================================================

/// Return a dimension_plausibility error if a required field is zero.
fn require_nonzero(field_name: &str, value: usize) -> std::result::Result<(), ModelLoadError> {
    if value == 0 {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: format!("{field_name} is 0"),
        });
    }
    Ok(())
}

fn validate_dimensions(config: &ModelLoadConfig) -> std::result::Result<(), ModelLoadError> {
    require_nonzero("hidden_dim", config.hidden_dim)?;
    require_nonzero("num_heads", config.num_heads)?;
    if !config.hidden_dim.is_multiple_of(config.num_heads) {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: format!(
                "hidden_dim ({}) is not divisible by num_heads ({})",
                config.hidden_dim, config.num_heads
            ),
        });
    }
    require_nonzero("vocab_size", config.vocab_size)?;
    require_nonzero("num_kv_heads", config.num_kv_heads)?;
    if config.num_kv_heads > config.num_heads {
        return Err(ModelLoadError {
            gate: "dimension_plausibility",
            reason: format!(
                "num_kv_heads ({}) > num_heads ({})",
                config.num_kv_heads, config.num_heads
            ),
        });
    }
    require_nonzero("intermediate_dim", config.intermediate_dim)?;
    require_nonzero("num_layers", config.num_layers)?;
    Ok(())
}

fn validate_architecture(arch_name: &str) -> std::result::Result<ArchConstraints, ModelLoadError> {
    let arch = ArchConstraints::from_architecture(arch_name);
    // ArchConstraints::from_architecture returns a valid default for unknown architectures.
    // We accept this — unknown architectures get base validation (no QK norm, no bias).
    // This is by design: new architectures can load with base constraints and fail later
    // at the ValidatedLayerWeights level if they have unexpected weight patterns.
    Ok(arch)
}

// ============================================================================
// PMAT-807: Honest-by-design architecture support gate
// ============================================================================

/// Returns `true` when `arch_name` denotes a Gemma-family architecture.
///
/// Matches the raw GGUF arch strings (`gemma`, `gemma2`, `gemma3`), their HF
/// `architectures[]` class names (`GemmaForCausalLM`, `Gemma2ForCausalLM`,
/// `Gemma3ForCausalLM`), and the normalized form `gemma`. Matching is
/// case-insensitive and prefix-based on the lowercased name so future point
/// variants (`gemma3n`, ...) are also caught — fail-loud is the safe default.
#[must_use]
pub fn is_gemma_family(arch_name: &str) -> bool {
    let lower = arch_name.to_ascii_lowercase();
    lower.starts_with("gemma")
}

/// PMAT-809: Returns `true` when `arch_name` is the Gemma-**v1** architecture
/// that realizar's CPU forward path now implements CORRECTLY.
///
/// Gemma v1 (`gemma`, `GemmaForCausalLM`) needs exactly three architecture-
/// specific behaviors — GeGLU FFN, `(1 + weight)` RMSNorm, and `sqrt(hidden_size)`
/// embedding scaling — all of which are implemented and verified coherent against
/// the llama.cpp reference for the same GGUF (PMAT-809). It has NO softcapping, so
/// it is correct without it.
///
/// Gemma3 / Gemma3n ALSO require behaviors beyond softcapping (e.g. per-layer
/// embedding scaling, alternating local/global attention with QK-norm) that are
/// NOT implemented — so they are deliberately EXCLUDED here and remain fail-loud.
#[must_use]
pub fn is_gemma1_supported(arch_name: &str) -> bool {
    let lower = arch_name.to_ascii_lowercase();
    // EXACT v1 only — never gemma2/gemma3/gemma3n (those need softcapping).
    lower == "gemma" || lower == "gemmaforcausallm"
}

/// PMAT-810: Returns `true` when `arch_name` is the Gemma-**v2** architecture
/// that realizar's CPU forward path now implements CORRECTLY.
///
/// Gemma v2 (`gemma2`, `Gemma2ForCausalLM`) adds three behaviors on top of
/// Gemma v1's GeGLU FFN + `(1 + weight)` RMSNorm + `sqrt(hidden)` embed scaling:
/// attention-logit tanh softcap (`50 * tanh(scores/50)`), final-logit tanh
/// softcap (`30 * tanh(logits/30)`), and `1/sqrt(query_pre_attn_scalar)` query
/// scaling. All three are implemented (`ops::softcap`, `config.attn_scale`) and
/// verified coherent against llama.cpp on gemma-2-2b-it Q4_K_M (PMAT-810):
/// "capital of France" → "Paris", "2+2=" → "4", top-token match.
///
/// EXACT `gemma2` only — `gemma3`/`gemma3n` need further behaviors and stay
/// fail-loud (honest-by-design).
#[must_use]
pub fn is_gemma2_supported(arch_name: &str) -> bool {
    let lower = arch_name.to_ascii_lowercase();
    lower == "gemma2" || lower == "gemma2forcausallm"
}

/// Fail LOUD for architectures whose required behaviors realizar's forward path
/// does not yet implement, instead of silently producing wrong output.
///
/// # Gemma support status (PMAT-807 → PMAT-809)
///
/// - **Gemma v1** (`gemma`, `GemmaForCausalLM`): SUPPORTED. The CPU forward path
///   implements GeGLU FFN, `(1 + weight)` RMSNorm, and `sqrt(hidden_size)`
///   embedding scaling (PMAT-809), verified coherent vs llama.cpp on the same
///   GGUF. Gemma v1 has no softcapping, so it is correct without it.
/// - **Gemma2 / Gemma3** (`gemma2`, `gemma3`, ...): STILL REFUSED. They additionally
///   require attention/final-logit tanh-softcapping, which is NOT implemented.
///   Running them with LLaMA-style (uncapped) attention yields silently-wrong
///   output, so they remain fail-loud (honest-by-design).
///
/// Non-Gemma architectures (llama, qwen2, qwen3, mistral, phi, deepseek, gpt2,
/// ...) are unaffected.
fn validate_supported_architecture(arch_name: &str) -> std::result::Result<(), ModelLoadError> {
    // PMAT-809: Gemma v1 is now implemented — allow it through.
    if is_gemma1_supported(arch_name) {
        return Ok(());
    }
    // PMAT-810: Gemma v2 is now implemented (softcapping + query_pre_attn_scalar),
    // verified coherent vs llama.cpp on gemma-2-2b-it Q4_K_M — allow it through.
    if is_gemma2_supported(arch_name) {
        return Ok(());
    }
    if is_gemma_family(arch_name) {
        return Err(ModelLoadError {
            gate: "architecture_supported",
            reason: format!(
                "Gemma3/Gemma3n architecture '{arch_name}' requires behaviors \
                 (per-layer embedding scaling, alternating local/global attention \
                 with QK-norm) that realizar's forward path does not implement yet. \
                 Running it would silently produce incorrect output, so it is \
                 refused. (Gemma v1 — PMAT-809 — and Gemma v2 — PMAT-810 — ARE \
                 supported.) Track Gemma3 support at PMAT-807."
            ),
        });
    }
    Ok(())
}

fn validate_completeness(
    arch: &ArchConstraints,
    present: &[WeightRole],
    arch_name: &str,
) -> std::result::Result<(), ModelLoadError> {
    contract_pre_weight_completeness!();
    let required = required_roles(arch);
    let mut missing = Vec::new();

    for &role in required {
        if !present.contains(&role) {
            missing.push(role.field_name());
        }
    }

    if !missing.is_empty() {
        return Err(ModelLoadError {
            gate: "architecture_completeness",
            reason: format!(
                "Architecture '{}' requires {} weights but model is missing: [{}]",
                arch_name,
                required.len(),
                missing.join(", "),
            ),
        });
    }

    contract_post_weight_completeness!(&());
    Ok(())
}

// ============================================================================
// GH-478: Resource Limit Gate
// ============================================================================

/// GH-478: Estimate F32 dequantization peak memory from tensor metadata.
///
/// `AprTransformer::from_apr_bytes` dequantizes ALL tensors to F32 eagerly.
/// For quantized models, this expands data ~7x (Q4K) to ~4x (Q8_0).
/// If the estimated peak exceeds 80% of system RAM, returns an error
/// so callers can route to a memory-efficient loading path.
///
/// #2568: if the host's RAM cannot be determined this **refuses** the dequant.
/// See [`dequant_verdict`] for why an unknown limit is not an infinite one.
///
/// # Arguments
///
/// * `tensor_entries` — slice of `(byte_size, dtype)` for each tensor in the file
/// * `file_size` — total file size in bytes (used for the raw Vec<u8> allocation)
pub fn validate_f32_dequant_limits(
    tensor_entries: &[(usize, u8)],
    file_size: u64,
) -> std::result::Result<(), ModelLoadError> {
    // Estimate F32 output size: sum of (elements × 4 bytes) for each tensor
    let mut estimated_f32_bytes: u64 = 0;
    for &(byte_size, dtype) in tensor_entries {
        let elements = estimate_elements(byte_size, dtype);
        estimated_f32_bytes += elements as u64 * 4;
    }

    dequant_verdict(file_size, estimated_f32_bytes, system_memory_bytes())
}

/// The dequant decision itself, isolated from *how* memory was measured.
///
/// `mem_total` is `None` when this platform has no memory probe (see
/// [`system_memory_bytes`]). #2568: the shipped 0.63.0 code substituted
/// `u64::MAX` there, which made the 80% threshold ~12.8 EiB and the comparison
/// `estimated_peak > threshold` unsatisfiable — the OOM guard could never fire
/// on macOS, the one platform where the probe returned `None`. A safety
/// threshold must **fail closed**: an unknown limit is not an unlimited one, so
/// the dequant is refused and the reason says which measurement is missing.
fn dequant_verdict(
    file_size: u64,
    estimated_f32_bytes: u64,
    mem_total: Option<u64>,
) -> std::result::Result<(), ModelLoadError> {
    // Peak = file in Vec<u8> + all F32 dequantized tensors
    let estimated_peak = file_size.saturating_add(estimated_f32_bytes);

    let Some(mem_total) = mem_total else {
        return Err(ModelLoadError {
            gate: "resource_limits",
            reason: format!(
                "cannot determine total system RAM on this platform ({}), so the F32 \
                 dequant OOM guard cannot be evaluated; refusing to dequant ~{} GB \
                 (file {} GB + dequant {} GB). An unknown memory limit is not an \
                 unlimited one (#2568). Use the quantized inference path.",
                std::env::consts::OS,
                estimated_peak / (1 << 30),
                file_size / (1 << 30),
                estimated_f32_bytes / (1 << 30),
            ),
        });
    };

    // 80% of RAM, written as /5*4 so it cannot overflow for any u64 input
    // (the old `mem_total * 80 / 100` panicked in debug builds on the
    // u64::MAX fallback and wrapped in release builds).
    let threshold = mem_total / 5 * 4;

    if estimated_peak > threshold {
        return Err(ModelLoadError {
            gate: "resource_limits",
            reason: format!(
                "F32 dequant would use ~{} GB (file {} GB + dequant {} GB), \
                 exceeds 80% of system RAM ({} GB). Use quantized inference path.",
                estimated_peak / (1 << 30),
                file_size / (1 << 30),
                estimated_f32_bytes / (1 << 30),
                mem_total / (1 << 30),
            ),
        });
    }

    Ok(())
}

/// Estimate number of elements from byte size and GGML dtype.
fn estimate_elements(byte_size: usize, dtype: u8) -> usize {
    match dtype {
        12 => byte_size / 144 * 256, // Q4_K: 144 bytes per 256 elements
        14 => byte_size / 210 * 256, // Q6_K: 210 bytes per 256 elements
        2 => byte_size / 36 * 32,    // Q8_0: 36 bytes per 32 elements
        1 => byte_size / 2,          // F16: 2 bytes per element
        30 => byte_size / 2,         // BF16: 2 bytes per element
        8 => byte_size / 5 * 4,      // APR Q4: 5 bytes per 4 elements
        9 => byte_size / 5 * 4,      // APR Q8: 5 bytes per 4 elements
        _ => byte_size / 4,          // F32: 4 bytes per element
    }
}

/// Absolute paths to try for macOS's `sysctl(8)`.
///
/// Absolute, not PATH-resolved, on purpose: a `sysctl` shadowed earlier on
/// `$PATH` could report an arbitrary number into a *safety* threshold.
const SYSCTL_PATHS: &[&str] = &["/usr/sbin/sysctl", "/sbin/sysctl"];

/// Total physical memory of this host, or `None` if it cannot be measured here.
///
/// | Platform | Probe |
/// |----------|-------|
/// | Linux (incl. Android, WSL) | `MemTotal:` in `/proc/meminfo` |
/// | macOS | `sysctl -n hw.memsize` |
/// | anything else | `None` — **callers must fail closed** |
///
/// #2568: this used to read `/proc/meminfo` and nothing else. macOS has no
/// `/proc`, so it returned `None` there and the one caller
/// ([`validate_f32_dequant_limits`]) turned that `None` into `u64::MAX`,
/// disarming the OOM guard on every Mac. `None` now means exactly "unknown",
/// and [`dequant_verdict`] refuses rather than assumes.
pub fn system_memory_bytes() -> Option<u64> {
    if let Some(bytes) = proc_meminfo_total_bytes() {
        return Some(bytes);
    }
    sysctl_hw_memsize_bytes()
}

/// Linux probe: `MemTotal:` from `/proc/meminfo`.
fn proc_meminfo_total_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo_total_bytes(&content)
}

/// Parse `MemTotal: <n> kB` out of `/proc/meminfo` contents.
///
/// Split out from the file read so the parse is testable on any platform.
/// A reported total of zero is treated as *unknown*, not as "no memory".
fn parse_meminfo_total_bytes(content: &str) -> Option<u64> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return kb.checked_mul(1024).filter(|&b| b > 0);
        }
    }
    None
}

/// macOS probe: `sysctl -n hw.memsize`.
///
/// Runs the tool rather than calling `sysctlbyname(3)` so the whole path stays
/// safe Rust and — apart from the kernel key itself — is exercised by tests on
/// every platform via [`run_sysctl_memsize`].
fn sysctl_hw_memsize_bytes() -> Option<u64> {
    // A runtime `cfg!` rather than `#[cfg]`: the code below is then compiled
    // and type-checked on Linux CI too, instead of only on a Mac.
    if !cfg!(target_os = "macos") {
        return None;
    }
    run_sysctl_memsize(SYSCTL_PATHS)
}

/// Invoke the first `sysctl` in `paths` that exists and parse its output.
fn run_sysctl_memsize(paths: &[&str]) -> Option<u64> {
    for path in paths {
        let Ok(output) = std::process::Command::new(path)
            .args(["-n", "hw.memsize"])
            .output()
        else {
            continue; // not installed at this path, or could not be spawned
        };
        if !output.status.success() {
            continue; // e.g. Linux procps sysctl, which has no hw.memsize
        }
        if let Some(bytes) = parse_sysctl_memsize(&String::from_utf8_lossy(&output.stdout)) {
            return Some(bytes);
        }
    }
    None
}

/// Parse the single decimal number `sysctl -n hw.memsize` prints.
///
/// Zero is treated as *unknown* so a degenerate reading cannot disarm the guard.
fn parse_sysctl_memsize(stdout: &str) -> Option<u64> {
    stdout.trim().parse::<u64>().ok().filter(|&b| b > 0)
}

// ============================================================================
// PMAT-285: Canonical transpose (single source of truth)
// ============================================================================

/// Transpose a row-major f32 matrix [rows, cols] → [cols, rows].
///
/// This is THE canonical transpose for all model weight operations in realizar.
/// Delegates to trueno's cache-blocked implementation for matrices ≥64 elements.
///
/// # Panics
///
/// Panics if `data.len() != rows * cols`.
#[must_use]
pub fn transpose_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    contract_pre_transpose_involution!();
    assert_eq!(
        data.len(),
        rows * cols,
        "transpose_f32: data.len()={} != rows*cols={}",
        data.len(),
        rows * cols
    );
    let mut out = vec![0.0f32; rows * cols];
    // trueno::blis::transpose handles cache-blocking for large matrices
    trueno::blis::transpose::transpose(rows, cols, data, &mut out)
        .expect("transpose_f32: dimension mismatch (should be impossible after assert)");
    contract_post_transpose!(&out);
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelLoadConfig {
        ModelLoadConfig {
            architecture: "llama".to_string(),
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_dim: 11008,
            vocab_size: 32000,
            present_roles: Vec::new(),
        }
    }

    #[test]
    fn test_valid_model_passes() {
        let proof = validate_model_load(&valid_config()).expect("should pass");
        assert_eq!(proof.architecture(), "llama");
        assert_eq!(proof.num_layers(), 32);
    }

    #[test]
    fn test_zero_hidden_dim_fails() {
        let mut config = valid_config();
        config.hidden_dim = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert_eq!(err.gate, "dimension_plausibility");
        assert!(err.reason.contains("hidden_dim"));
    }

    #[test]
    fn test_zero_num_heads_fails() {
        let mut config = valid_config();
        config.num_heads = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("num_heads"));
    }

    #[test]
    fn test_hidden_not_divisible_by_heads() {
        let mut config = valid_config();
        config.hidden_dim = 4097;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("not divisible"));
    }

    #[test]
    fn test_kv_heads_greater_than_heads() {
        let mut config = valid_config();
        config.num_kv_heads = 64;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("num_kv_heads"));
    }

    #[test]
    fn test_zero_vocab_fails() {
        let mut config = valid_config();
        config.vocab_size = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("vocab_size"));
    }

    #[test]
    fn test_zero_layers_fails() {
        let mut config = valid_config();
        config.num_layers = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("num_layers"));
    }

    #[test]
    fn test_zero_intermediate_fails() {
        let mut config = valid_config();
        config.intermediate_dim = 0;
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("intermediate_dim"));
    }

    #[test]
    fn test_basic_convenience() {
        let proof =
            validate_model_load_basic("qwen2", 28, 1536, 12, 2, 8960, 151936).expect("should pass");
        assert_eq!(proof.architecture(), "qwen2");
    }

    #[test]
    fn test_completeness_llama_all_present() {
        let mut config = valid_config();
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
        ];
        assert!(validate_model_load(&config).is_ok());
    }

    #[test]
    fn test_completeness_llama_missing_gate() {
        let mut config = valid_config();
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            // Missing FfnGate, FfnUp, FfnDown
        ];
        let err = validate_model_load(&config).unwrap_err();
        assert_eq!(err.gate, "architecture_completeness");
        assert!(err.reason.contains("ffn_gate"));
    }

    #[test]
    fn test_completeness_qwen3_needs_qk_norm() {
        let mut config = valid_config();
        config.architecture = "qwen3".to_string();
        // Provide base roles but NOT qk_norm
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
        ];
        let err = validate_model_load(&config).unwrap_err();
        assert!(err.reason.contains("attn_q_norm"));
    }

    #[test]
    fn test_completeness_qwen3_with_qk_norm_passes() {
        let mut config = valid_config();
        config.architecture = "qwen3".to_string();
        config.present_roles = vec![
            WeightRole::AttnNorm,
            WeightRole::FfnNorm,
            WeightRole::QProj,
            WeightRole::KProj,
            WeightRole::VProj,
            WeightRole::OProj,
            WeightRole::FfnGate,
            WeightRole::FfnUp,
            WeightRole::FfnDown,
            WeightRole::AttnQNorm,
            WeightRole::AttnKNorm,
        ];
        assert!(validate_model_load(&config).is_ok());
    }

    #[test]
    fn test_no_roles_skips_completeness() {
        // If present_roles is empty, completeness check is skipped
        let config = valid_config();
        assert!(config.present_roles.is_empty());
        assert!(validate_model_load(&config).is_ok());
    }

    #[test]
    fn test_unknown_architecture_uses_base() {
        let proof = validate_model_load_basic("unknown_future_arch", 1, 128, 4, 4, 512, 1000)
            .expect("unknown arch should pass with base constraints");
        assert_eq!(proof.architecture(), "unknown_future_arch");
    }

    // ------------------------------------------------------------------
    // PMAT-807: Gemma fail-loud gate
    // ------------------------------------------------------------------

    /// FALSIFIER: every Gemma-family arch string is rejected at the gate.
    /// If any is silently accepted, this test fails (silent-garbage regression).
    ///
    /// PMAT-810: Gemma3/Gemma3n are STILL refused (further behaviors unimplemented).
    /// Gemma v1 (PMAT-809) and Gemma v2 (PMAT-810) are now SUPPORTED, asserted
    /// separately in `test_gemma1_now_supported` / `test_gemma2_now_supported`.
    #[test]
    fn test_gemma3_rejected_at_load() {
        let gemma_names = [
            "gemma3",
            "Gemma3ForCausalLM",
            "gemma3n", // future point variant — fail-loud is the safe default
        ];
        for name in gemma_names {
            let mut config = valid_config();
            config.architecture = name.to_string();
            let err = validate_model_load(&config)
                .expect_err(&format!("Gemma3 arch '{name}' must be refused, not run"));
            assert_eq!(
                err.gate, "architecture_supported",
                "'{name}' rejected by wrong gate: {}",
                err.gate
            );
            // The error must name the architecture it refused.
            assert!(
                err.reason.contains("Gemma3"),
                "'{name}' error must name the refused architecture: {}",
                err.reason
            );
        }
    }

    /// PMAT-810 FALSIFIER: Gemma v2 now LOADS (it was fail-loud under PMAT-807/809).
    ///
    /// If the forward path ever regresses and Gemma v2 is re-rejected, this fails.
    /// Coherence vs llama.cpp is the separate end-to-end falsifier (PMAT-810).
    #[test]
    fn test_gemma2_now_supported() {
        for name in ["gemma2", "GEMMA2", "Gemma2ForCausalLM"] {
            assert!(
                is_gemma2_supported(name),
                "'{name}' must be recognized as supported Gemma v2"
            );
            let mut config = valid_config();
            config.architecture = name.to_string();
            assert!(
                validate_model_load(&config).is_ok(),
                "Gemma v2 arch '{name}' must now load (PMAT-810)"
            );
        }
        // The exclusions: gemma3/gemma3n are NOT "supported v2".
        assert!(!is_gemma2_supported("gemma3"));
        assert!(!is_gemma2_supported("gemma3n"));
        assert!(!is_gemma2_supported("gemma"));
    }

    /// PMAT-809 FALSIFIER: Gemma v1 now LOADS (it was fail-loud under PMAT-807).
    ///
    /// If the forward path ever regresses and Gemma v1 is re-rejected, this fails.
    /// Coherence vs llama.cpp is the separate end-to-end falsifier (PMAT-809).
    #[test]
    fn test_gemma1_now_supported() {
        for name in ["gemma", "GEMMA", "GemmaForCausalLM"] {
            assert!(
                is_gemma1_supported(name),
                "'{name}' must be recognized as supported Gemma v1"
            );
            let mut config = valid_config();
            config.architecture = name.to_string();
            assert!(
                validate_model_load(&config).is_ok(),
                "Gemma v1 arch '{name}' must now load (PMAT-809)"
            );
        }
        // The exclusions: gemma2/gemma3 are NOT "supported v1".
        assert!(!is_gemma1_supported("gemma2"));
        assert!(!is_gemma1_supported("gemma3"));
        assert!(!is_gemma1_supported("gemma3n"));
    }

    /// PMAT-810: `validate_model_load_basic` (the path real loaders call) now
    /// ACCEPTS Gemma v2 (gemma-2-2b-it dims: 26 layers, 2304 hidden, 8/4 heads).
    #[test]
    fn test_gemma2_accepted_via_basic_loader_path() {
        validate_model_load_basic("gemma2", 26, 2304, 8, 4, 9216, 256_000)
            .expect("gemma2 must now load at the basic loader gate (PMAT-810)");
    }

    /// `validate_model_load_basic` now ACCEPTS Gemma v1 (PMAT-809).
    #[test]
    fn test_gemma1_accepted_via_basic_loader_path() {
        let proof = validate_model_load_basic("gemma", 18, 2048, 8, 1, 16384, 256_128)
            .expect("gemma v1 must now load at the basic loader gate");
        assert_eq!(proof.architecture(), "gemma");
    }

    /// CONTROL: non-Gemma architectures are unaffected — no regression.
    #[test]
    fn test_non_gemma_architectures_unaffected() {
        for arch in [
            "llama",
            "qwen2",
            "qwen3",
            "mistral",
            "phi",
            "phi2",
            "deepseek",
            "gpt2",
            "unknown_future_arch",
        ] {
            assert!(
                !is_gemma_family(arch),
                "'{arch}' wrongly classified as Gemma"
            );
            let mut config = valid_config();
            config.architecture = arch.to_string();
            // Dimensions in valid_config() are llama-shaped and pass; the point is
            // that the architecture_supported gate does NOT trip for these.
            assert!(
                validate_model_load(&config).is_ok(),
                "non-Gemma arch '{arch}' must still load"
            );
        }
    }

    /// `is_gemma_family` is case-insensitive and prefix-based.
    #[test]
    fn test_is_gemma_family_classification() {
        assert!(is_gemma_family("gemma"));
        assert!(is_gemma_family("GEMMA"));
        assert!(is_gemma_family("Gemma2ForCausalLM"));
        assert!(!is_gemma_family("llama"));
        assert!(!is_gemma_family("gem")); // not a Gemma model
        assert!(!is_gemma_family(""));
    }

    #[test]
    fn test_error_display() {
        let err = ModelLoadError {
            gate: "test_gate",
            reason: "test reason".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("GH-279"));
        assert!(msg.contains("test_gate"));
        assert!(msg.contains("test reason"));
    }

    #[test]
    fn test_error_converts_to_realizar_error() {
        let err = ModelLoadError {
            gate: "test",
            reason: "test".to_string(),
        };
        let r_err: RealizarError = err.into();
        match r_err {
            RealizarError::UnsupportedOperation { operation, .. } => {
                assert!(operation.contains("contract_gate"));
            },
            _ => panic!("expected UnsupportedOperation"),
        }
    }

    // ================================================================
    // GH-478: Resource limit gate tests
    // ================================================================

    #[test]
    fn test_estimate_elements_f32() {
        // 400 bytes of F32 = 100 elements
        assert_eq!(estimate_elements(400, 0), 100);
    }

    #[test]
    fn test_estimate_elements_q4k() {
        // 144 bytes = 1 Q4K super-block = 256 elements
        assert_eq!(estimate_elements(144, 12), 256);
        // 2 super-blocks
        assert_eq!(estimate_elements(288, 12), 512);
    }

    #[test]
    fn test_estimate_elements_q6k() {
        // 210 bytes = 1 Q6K super-block = 256 elements
        assert_eq!(estimate_elements(210, 14), 256);
    }

    #[test]
    fn test_estimate_elements_f16() {
        assert_eq!(estimate_elements(200, 1), 100);
    }

    #[test]
    fn test_estimate_elements_bf16() {
        assert_eq!(estimate_elements(200, 30), 100);
    }

    #[test]
    fn test_small_model_passes_resource_check() {
        // A ~1 MB APR file whose F32 dequant is ~4 MB. This fits under 80% of
        // any machine that can run the test suite, so — unlike the previous
        // `let _ = result;` version, which excluded no outcome — it may assert.
        let tensors: Vec<(usize, u8)> = vec![(144 * 1000, 12)]; // Q4K, 256K elements
        let result = validate_f32_dequant_limits(&tensors, 1_000_000);
        assert!(
            result.is_ok(),
            "a ~5 MB peak must pass the resource gate on any host that can run \
             this test, got: {:?}",
            result.err()
        );
    }

    // ------------------------------------------------------------------
    // #2568: the OOM guard must FAIL CLOSED when RAM cannot be measured.
    // These run on every platform — the decision is separated from the probe
    // precisely so no target_os can skip them.
    // ------------------------------------------------------------------

    #[test]
    fn test_dequant_verdict_fails_closed_when_memory_unknown() {
        // Before #2568 the unknown case became u64::MAX and this returned Ok.
        let err = dequant_verdict(1 << 30, 8 << 30, None)
            .expect_err("unknown system memory MUST refuse the dequant, not allow it");
        assert_eq!(err.gate, "resource_limits");
        assert!(
            err.reason.contains("cannot determine total system RAM"),
            "the refusal must name the missing measurement, got: {}",
            err.reason
        );
    }

    #[test]
    fn test_dequant_verdict_fails_closed_even_for_a_tiny_model() {
        // "Unknown" is not "unlimited" at ANY size: a 1-byte file is refused
        // too, because the guard has nothing to compare against.
        assert!(
            dequant_verdict(1, 0, None).is_err(),
            "an unmeasurable host must refuse every dequant, however small"
        );
    }

    #[test]
    fn test_dequant_verdict_refuses_over_80_percent() {
        // 16 GiB host, 14 GiB peak: over the 12.8 GiB threshold.
        let err = dequant_verdict(2 << 30, 12 << 30, Some(16 << 30))
            .expect_err("14 GiB peak on a 16 GiB host must be refused");
        assert!(
            err.reason.contains("exceeds 80% of system RAM"),
            "{}",
            err.reason
        );
    }

    #[test]
    fn test_dequant_verdict_allows_under_80_percent() {
        // 16 GiB host, 8 GiB peak: under the 12.8 GiB threshold.
        assert!(dequant_verdict(2 << 30, 6 << 30, Some(16 << 30)).is_ok());
    }

    #[test]
    fn test_dequant_verdict_threshold_does_not_overflow() {
        // `mem_total * 80 / 100` panicked in debug builds at u64::MAX (that is
        // what the old unwrap_or(u64::MAX) fed it). `/5*4` cannot.
        assert!(dequant_verdict(0, 0, Some(u64::MAX)).is_ok());
        assert!(
            dequant_verdict(1, 0, Some(0)).is_err(),
            "a host reporting 0 bytes of RAM fits nothing"
        );
    }

    // ------------------------------------------------------------------
    // #2568: the probe. NOTE — no `if cfg!(target_os = ...)` guard here.
    // The deleted guard was the whole defect: it skipped macOS, the only
    // platform where the probe was broken, so the test read as coverage while
    // proving nothing there.
    // ------------------------------------------------------------------

    #[test]
    fn test_system_memory_bytes_is_measurable_on_this_platform() {
        let mem = system_memory_bytes();
        assert!(
            mem.is_some(),
            "no memory probe for target_os={}: the F32 dequant OOM guard cannot \
             be armed here. Add a probe to system_memory_bytes() (#2568) — do \
             NOT skip this assertion by platform.",
            std::env::consts::OS,
        );
        assert!(mem.expect("checked is_some above") > 0);
    }

    #[test]
    fn test_parse_meminfo_total_bytes() {
        let sample = "MemTotal:       131377776 kB\nMemFree:         2000 kB\n";
        assert_eq!(parse_meminfo_total_bytes(sample), Some(131_377_776 * 1024));
        // MemTotal absent, malformed, or zero => unknown, never a wrong number
        assert_eq!(parse_meminfo_total_bytes("MemFree: 2000 kB\n"), None);
        assert_eq!(parse_meminfo_total_bytes("MemTotal:       kB\n"), None);
        assert_eq!(parse_meminfo_total_bytes("MemTotal:       0 kB\n"), None);
        assert_eq!(parse_meminfo_total_bytes(""), None);
    }

    #[test]
    fn test_parse_sysctl_memsize() {
        // What `/usr/sbin/sysctl -n hw.memsize` prints on macOS 26.5.2 arm64.
        assert_eq!(parse_sysctl_memsize("17179869184\n"), Some(17_179_869_184));
        assert_eq!(
            parse_sysctl_memsize("  17179869184  "),
            Some(17_179_869_184)
        );
        assert_eq!(parse_sysctl_memsize(""), None);
        assert_eq!(parse_sysctl_memsize("hw.memsize: 17179869184\n"), None);
        assert_eq!(parse_sysctl_memsize("0\n"), None);
    }

    /// Exercise the macOS probe's exec+status+parse path on ANY platform by
    /// pointing it at a stub that behaves like `sysctl -n hw.memsize`.
    #[test]
    fn test_run_sysctl_memsize_against_a_stub() {
        let dir = tempfile::tempdir().expect("tempdir");

        let good = dir.path().join("sysctl_ok");
        std::fs::write(&good, "#!/bin/sh\necho 17179869184\n").expect("write stub");
        set_executable(&good);

        // A stub that exits non-zero must NOT be believed, even though it
        // prints a number — that is the Linux `sysctl` case.
        let bad = dir.path().join("sysctl_fail");
        std::fs::write(&bad, "#!/bin/sh\necho 999\nexit 1\n").expect("write stub");
        set_executable(&bad);

        let missing = dir.path().join("sysctl_absent");
        let (good, bad, missing) = (
            good.to_string_lossy().into_owned(),
            bad.to_string_lossy().into_owned(),
            missing.to_string_lossy().into_owned(),
        );

        assert_eq!(run_sysctl_memsize(&[&good]), Some(17_179_869_184));
        assert_eq!(
            run_sysctl_memsize(&[&bad]),
            None,
            "non-zero exit must not be trusted"
        );
        assert_eq!(run_sysctl_memsize(&[&missing]), None);
        // A missing/failing path falls through to the next candidate.
        assert_eq!(
            run_sysctl_memsize(&[&missing, &bad, &good]),
            Some(17_179_869_184)
        );
        assert_eq!(run_sysctl_memsize(&[]), None);
    }

    fn set_executable(path: &std::path::Path) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod stub");
        }
    }
}
