//! VRAM-aware quantization auto-select classifier (CRUX-A-10).
//!
//! Contract: `contracts/crux-A-10-v1.yaml`.
//!
//! Pure classifier — implements the contract equations
//! `vram_footprint_model` and `auto_quant_selection` without touching
//! any GPU, network, or filesystem. Given a model's shape
//! (parameter count, layer/KV-head/head-dim triple), a list of
//! available quants, a detected free-VRAM byte count, a context
//! length, and a safety factor, returns the highest-quality quant
//! whose estimated footprint ≤ budget — or `None` (cpu_fallback) if
//! no quant fits.
//!
//! The integration-level claim
//!   * `apr pull --auto-quant --json` emits a result whose
//!     `estimated_footprint_bytes ≤ free_vram_bytes * safety_factor`,
//! is discharged by a separate CLI-wiring harness. This module proves
//! the algorithm-level precondition: the SELECTION function is
//! monotone, budget-respecting, and arg-max of quality_rank.

/// Canonical GGUF / llama.cpp quant tags ordered by quality_rank
/// (ascending — Q2_K worst, F16 best). Matches the codomain listed in
/// `auto_quant_selection` codomain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum QuantTag {
    Q2K = 1,
    Q3KS = 2,
    Q3KM = 3,
    Q4KS = 4,
    Q4KM = 5,
    Q5KS = 6,
    Q5KM = 7,
    Q6K = 8,
    Q8_0 = 9,
    F16 = 10,
}

impl QuantTag {
    /// Quality ordinal used by `auto_quant_selection` arg-max. Higher
    /// = better. Matches the explicit rank in the FALSIFY-003 golden.
    pub const fn quality_rank(self) -> u8 {
        self as u8
    }

    /// Bits per weight for the quant. Used by the footprint formula.
    /// Matches llama.cpp GGUF quant bit-per-weight reference table.
    pub const fn bits_per_weight(self) -> f64 {
        match self {
            // Block quants average BPW is taken from llama.cpp
            // `ggml-quants.c` comment blocks. Conservative lower
            // bounds that guarantee the classifier never under-
            // estimates footprint.
            QuantTag::Q2K => 2.625,
            QuantTag::Q3KS => 3.4375,
            QuantTag::Q3KM => 3.8125,
            QuantTag::Q4KS => 4.5,
            QuantTag::Q4KM => 4.85,
            QuantTag::Q5KS => 5.5,
            QuantTag::Q5KM => 5.7,
            QuantTag::Q6K => 6.5625,
            QuantTag::Q8_0 => 8.5,
            QuantTag::F16 => 16.0,
        }
    }

    /// Human-readable tag — round-trips through `from_str`.
    pub const fn as_str(self) -> &'static str {
        match self {
            QuantTag::Q2K => "Q2_K",
            QuantTag::Q3KS => "Q3_K_S",
            QuantTag::Q3KM => "Q3_K_M",
            QuantTag::Q4KS => "Q4_K_S",
            QuantTag::Q4KM => "Q4_K_M",
            QuantTag::Q5KS => "Q5_K_S",
            QuantTag::Q5KM => "Q5_K_M",
            QuantTag::Q6K => "Q6_K",
            QuantTag::Q8_0 => "Q8_0",
            QuantTag::F16 => "F16",
        }
    }
}

/// Shape of a transformer model needed to estimate VRAM footprint.
/// All fields are read from GGUF / APR tensor metadata — never
/// name-guessed (contract invariant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelShape {
    pub n_params: u64,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Overhead not accounted for by weights or KV cache — activations,
    /// workspace buffers, CUDA context. Conservative upper bound.
    pub overhead_bytes: u64,
}

/// Reason the auto-quant selector cannot pick a quant.
#[derive(Debug, Clone, PartialEq)]
pub enum AutoQuantError {
    /// No quants at all were offered — likely a repo metadata bug.
    EmptyQuantList,
    /// `safety_factor` was outside (0, 1].
    InvalidSafetyFactor(f64),
    /// `ctx_len` was 0 — nonsensical inference request.
    ZeroCtxLen,
}

impl std::fmt::Display for AutoQuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoQuantError::EmptyQuantList => {
                write!(f, "no available quants to choose from")
            }
            AutoQuantError::InvalidSafetyFactor(s) => {
                write!(f, "safety_factor must be in (0, 1], got {s}")
            }
            AutoQuantError::ZeroCtxLen => write!(f, "ctx_len must be > 0"),
        }
    }
}

impl std::error::Error for AutoQuantError {}

/// Default safety factor matches ollama's ≈ 10% headroom.
pub const DEFAULT_SAFETY_FACTOR: f64 = 0.90;

/// Dtype size in bytes for the KV cache. GGUF default is F16.
pub const KV_CACHE_DTYPE_BYTES: u64 = 2;

/// Estimated weight bytes for `(n_params, quant)`. Rounds UP so the
/// classifier never under-estimates.
pub fn weight_bytes(n_params: u64, quant: QuantTag) -> u64 {
    let bpw = quant.bits_per_weight();
    let bits = (n_params as f64) * bpw;
    (bits / 8.0).ceil() as u64
}

/// KV-cache bytes formula from the contract:
///   `2 * n_layers * n_kv_heads * head_dim * ctx_len * dtype_size`
pub fn kv_cache_bytes(shape: ModelShape, ctx_len: u32) -> u64 {
    2u64 * (shape.n_layers as u64)
        * (shape.n_kv_heads as u64)
        * (shape.head_dim as u64)
        * (ctx_len as u64)
        * KV_CACHE_DTYPE_BYTES
}

/// Total footprint per contract `vram_footprint_model`.
pub fn footprint_bytes(shape: ModelShape, quant: QuantTag, ctx_len: u32) -> u64 {
    weight_bytes(shape.n_params, quant)
        .saturating_add(kv_cache_bytes(shape, ctx_len))
        .saturating_add(shape.overhead_bytes)
}

/// One candidate evaluated by the selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub quant: QuantTag,
    pub footprint_bytes: u64,
    pub fits: bool,
}

/// Full selection decision. `selected` is `None` iff every candidate
/// overflowed budget (`cpu_fallback` in contract terms).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionDecision {
    pub budget_bytes: u64,
    pub candidates: Vec<Candidate>,
    pub selected: Option<QuantTag>,
}

/// Choose the highest-quality quant whose footprint ≤ budget.
///
/// Contract `auto_quant_selection` — exact implementation:
///   `budget = free_vram * safety_factor`
///   `fitting = { q | footprint(q) ≤ budget }`
///   `pick = argmax(quality_rank, fitting) if fitting else None`
///
/// Returns every candidate (for FALSIFY-002 "no strictly-better quant
/// would have fit" proofs), the applied budget, and the selected
/// quant (None = cpu_fallback).
pub fn select_auto_quant(
    shape: ModelShape,
    available: &[QuantTag],
    free_vram_bytes: u64,
    ctx_len: u32,
    safety_factor: f64,
) -> Result<SelectionDecision, AutoQuantError> {
    if available.is_empty() {
        return Err(AutoQuantError::EmptyQuantList);
    }
    if ctx_len == 0 {
        return Err(AutoQuantError::ZeroCtxLen);
    }
    if !(safety_factor > 0.0 && safety_factor <= 1.0) {
        return Err(AutoQuantError::InvalidSafetyFactor(safety_factor));
    }

    // Round DOWN on the budget so we never exceed.
    let budget_bytes = ((free_vram_bytes as f64) * safety_factor).floor() as u64;

    let mut candidates: Vec<Candidate> = available
        .iter()
        .copied()
        .map(|q| {
            let fp = footprint_bytes(shape, q, ctx_len);
            Candidate {
                quant: q,
                footprint_bytes: fp,
                fits: fp <= budget_bytes,
            }
        })
        .collect();

    // Sort by quality_rank ASC so iteration is stable; selection
    // still picks the MAX-rank that fits.
    candidates.sort_by_key(|c| c.quant.quality_rank());

    let selected = candidates
        .iter()
        .filter(|c| c.fits)
        .max_by_key(|c| c.quant.quality_rank())
        .map(|c| c.quant);

    Ok(SelectionDecision {
        budget_bytes,
        candidates,
        selected,
    })
}

/// FALSIFY-001 sub-claim predicate: the selected quant's footprint
/// never exceeds budget.
pub fn decision_respects_budget(d: &SelectionDecision) -> bool {
    match d.selected {
        None => true,
        Some(q) => d
            .candidates
            .iter()
            .find(|c| c.quant == q)
            .map(|c| c.footprint_bytes <= d.budget_bytes)
            .unwrap_or(false),
    }
}

/// FALSIFY-002 sub-claim predicate: no strictly-better quant fits
/// within budget. Equivalent to "selected is arg-max of fitting
/// candidates".
pub fn decision_is_argmax(d: &SelectionDecision) -> bool {
    let picked_rank = match d.selected {
        None => 0u8,
        Some(q) => q.quality_rank(),
    };
    // If there's no selection, the claim reduces to "no candidate fit
    // the budget" — verify that directly.
    if d.selected.is_none() {
        return d.candidates.iter().all(|c| !c.fits);
    }
    d.candidates
        .iter()
        .filter(|c| c.fits)
        .all(|c| c.quant.quality_rank() <= picked_rank)
}

#[cfg(test)]
#[path = "auto_quant_tests.rs"]
mod tests;
