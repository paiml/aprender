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
mod tests {
    use super::*;

    /// Approximate Qwen2.5-Coder-7B shape — 28 layers, GQA 4 KV heads,
    /// head_dim 128, 7.6B params. Used by FALSIFY-001/002/003 shell
    /// tests; the algorithm-level proof operates on the same shape.
    fn qwen25_coder_7b() -> ModelShape {
        ModelShape {
            n_params: 7_615_616_512,
            n_layers: 28,
            n_kv_heads: 4,
            head_dim: 128,
            overhead_bytes: 512 * 1024 * 1024, // 512 MiB CUDA + activations
        }
    }

    /// ~3B model — Qwen2.5-Coder-3B shape approximation.
    fn qwen25_coder_3b() -> ModelShape {
        ModelShape {
            n_params: 3_085_938_688,
            n_layers: 36,
            n_kv_heads: 2,
            head_dim: 128,
            overhead_bytes: 256 * 1024 * 1024,
        }
    }

    fn all_quants() -> Vec<QuantTag> {
        vec![
            QuantTag::Q2K,
            QuantTag::Q3KS,
            QuantTag::Q3KM,
            QuantTag::Q4KS,
            QuantTag::Q4KM,
            QuantTag::Q5KS,
            QuantTag::Q5KM,
            QuantTag::Q6K,
            QuantTag::Q8_0,
            QuantTag::F16,
        ]
    }

    #[test]
    fn quality_rank_is_monotone_across_enum() {
        // All known quants have strictly ascending quality_rank.
        let q = all_quants();
        for pair in q.windows(2) {
            assert!(pair[0].quality_rank() < pair[1].quality_rank());
        }
    }

    #[test]
    fn weight_bytes_matches_bpw_formula() {
        // 1B params at Q4_K_M: 1e9 * 4.85 / 8 ≈ 606 MB.
        let got = weight_bytes(1_000_000_000, QuantTag::Q4KM);
        let expected = ((1_000_000_000f64 * 4.85) / 8.0).ceil() as u64;
        assert_eq!(got, expected);
    }

    #[test]
    fn kv_cache_matches_contract_formula() {
        // Qwen2.5-7B @ ctx 2048:
        //   2 * 28 * 4 * 128 * 2048 * 2 bytes = 117,440,512 bytes
        let shape = qwen25_coder_7b();
        let got = kv_cache_bytes(shape, 2048);
        let expected = 2u64 * 28 * 4 * 128 * 2048 * 2;
        assert_eq!(got, expected);
    }

    #[test]
    fn falsify_001_sub_claim_selected_quant_under_budget() {
        // CRUX-A-10 ALGO-001 sub-claim of FALSIFY-001: selected quant's
        // footprint ≤ free_vram * safety_factor. Algorithm-level
        // analogue of the shell test's post-selection assertion.
        let shape = qwen25_coder_7b();
        // Typical RTX 4090: 24 GiB free.
        let free = 24u64 * 1024 * 1024 * 1024;
        let d = select_auto_quant(shape, &all_quants(), free, 2048, DEFAULT_SAFETY_FACTOR)
            .unwrap();
        assert!(decision_respects_budget(&d));
    }

    #[test]
    fn falsify_002_sub_claim_argmax_of_fitting() {
        // CRUX-A-10 ALGO-002 sub-claim of FALSIFY-002: no strictly-
        // higher-quality quant fits within budget.
        let shape = qwen25_coder_7b();
        let free = 16u64 * 1024 * 1024 * 1024; // 16 GiB — tighter
        let d = select_auto_quant(shape, &all_quants(), free, 8192, DEFAULT_SAFETY_FACTOR)
            .unwrap();
        assert!(decision_is_argmax(&d));
    }

    #[test]
    fn falsify_003_sub_claim_ctx_doubling_never_raises_quality() {
        // CRUX-A-10 ALGO-003 sub-claim of FALSIFY-003: doubling
        // ctx_len never raises the selected quant's quality_rank.
        let shape = qwen25_coder_7b();
        let free = 12u64 * 1024 * 1024 * 1024;
        let a = select_auto_quant(shape, &all_quants(), free, 2048, DEFAULT_SAFETY_FACTOR)
            .unwrap();
        let b = select_auto_quant(shape, &all_quants(), free, 4096, DEFAULT_SAFETY_FACTOR)
            .unwrap();
        let c = select_auto_quant(shape, &all_quants(), free, 32768, DEFAULT_SAFETY_FACTOR)
            .unwrap();
        let rank_a = a.selected.map(|q| q.quality_rank()).unwrap_or(0);
        let rank_b = b.selected.map(|q| q.quality_rank()).unwrap_or(0);
        let rank_c = c.selected.map(|q| q.quality_rank()).unwrap_or(0);
        assert!(rank_b <= rank_a, "2048→4096 raised rank {rank_a}→{rank_b}");
        assert!(rank_c <= rank_b, "4096→32768 raised rank {rank_b}→{rank_c}");
    }

    #[test]
    fn empty_quant_list_is_error() {
        let shape = qwen25_coder_7b();
        let err = select_auto_quant(shape, &[], 1 << 34, 2048, 0.9).unwrap_err();
        assert_eq!(err, AutoQuantError::EmptyQuantList);
    }

    #[test]
    fn zero_ctx_is_error() {
        let shape = qwen25_coder_7b();
        let err = select_auto_quant(shape, &all_quants(), 1 << 34, 0, 0.9).unwrap_err();
        assert_eq!(err, AutoQuantError::ZeroCtxLen);
    }

    #[test]
    fn safety_factor_out_of_range_is_error() {
        let shape = qwen25_coder_7b();
        let err = select_auto_quant(shape, &all_quants(), 1 << 34, 2048, 0.0).unwrap_err();
        assert!(matches!(err, AutoQuantError::InvalidSafetyFactor(_)));
        let err = select_auto_quant(shape, &all_quants(), 1 << 34, 2048, 1.5).unwrap_err();
        assert!(matches!(err, AutoQuantError::InvalidSafetyFactor(_)));
        let err = select_auto_quant(shape, &all_quants(), 1 << 34, 2048, -0.1).unwrap_err();
        assert!(matches!(err, AutoQuantError::InvalidSafetyFactor(_)));
    }

    #[test]
    fn safety_factor_one_still_valid() {
        // Boundary: safety_factor = 1.0 is valid (no headroom).
        let shape = qwen25_coder_7b();
        let d = select_auto_quant(shape, &all_quants(), 1u64 << 36, 2048, 1.0).unwrap();
        assert!(d.selected.is_some());
    }

    #[test]
    fn budget_overflow_returns_cpu_fallback() {
        // 7B F16 won't fit in 4 GiB.
        let shape = qwen25_coder_7b();
        let free = 4u64 * 1024 * 1024 * 1024;
        let d = select_auto_quant(shape, &[QuantTag::F16], free, 2048, 0.9)
            .unwrap();
        assert!(d.selected.is_none(), "expected cpu_fallback");
        assert!(d.candidates.iter().all(|c| !c.fits));
        assert!(decision_is_argmax(&d));
    }

    #[test]
    fn rtx_4090_24gib_7b_picks_q6k_or_better() {
        // Sanity: 7B on a 24 GiB card at ctx 2048 should comfortably
        // fit Q6_K or higher — the high-quality regime.
        let shape = qwen25_coder_7b();
        let free = 24u64 * 1024 * 1024 * 1024;
        let d = select_auto_quant(shape, &all_quants(), free, 2048, 0.9).unwrap();
        let q = d.selected.unwrap();
        assert!(
            q.quality_rank() >= QuantTag::Q6K.quality_rank(),
            "expected ≥ Q6_K on 24 GiB 7B @ 2048, got {:?}",
            q
        );
    }

    #[test]
    fn selection_is_deterministic() {
        let shape = qwen25_coder_7b();
        let a = select_auto_quant(shape, &all_quants(), 1 << 34, 2048, 0.9).unwrap();
        let b = select_auto_quant(shape, &all_quants(), 1 << 34, 2048, 0.9).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn footprint_monotone_in_ctx_len() {
        // Contract invariant: footprint non-decreasing in ctx_len.
        let shape = qwen25_coder_7b();
        let prev = [2048u32, 4096, 8192, 16_384, 32_768];
        let mut last = 0u64;
        for c in prev {
            let fp = footprint_bytes(shape, QuantTag::Q4KM, c);
            assert!(fp >= last, "non-monotone at ctx={c}: {fp} < {last}");
            last = fp;
        }
    }

    #[test]
    fn footprint_monotone_in_quality() {
        // Contract invariant: footprint non-decreasing in quality(quant).
        let shape = qwen25_coder_7b();
        let quants = all_quants();
        let mut last = 0u64;
        for q in quants {
            let fp = footprint_bytes(shape, q, 2048);
            assert!(fp >= last, "non-monotone at {:?}: {fp} < {last}", q);
            last = fp;
        }
    }

    #[test]
    fn small_model_selects_f16_with_headroom() {
        // Tiny 100M-param model on a 24 GiB card: F16 fits easily.
        let shape = ModelShape {
            n_params: 100_000_000,
            n_layers: 12,
            n_kv_heads: 12,
            head_dim: 64,
            overhead_bytes: 256 * 1024 * 1024,
        };
        let free = 24u64 * 1024 * 1024 * 1024;
        let d = select_auto_quant(shape, &all_quants(), free, 2048, 0.9).unwrap();
        assert_eq!(d.selected, Some(QuantTag::F16));
        assert!(decision_respects_budget(&d));
        assert!(decision_is_argmax(&d));
    }

    #[test]
    fn three_b_laptop_8gib_picks_mid_range_quant() {
        // Laptop GPU 8 GiB, 3B model @ 2048 ctx.
        let shape = qwen25_coder_3b();
        let free = 8u64 * 1024 * 1024 * 1024;
        let d = select_auto_quant(shape, &all_quants(), free, 2048, 0.9).unwrap();
        assert!(d.selected.is_some());
        assert!(decision_respects_budget(&d));
        assert!(decision_is_argmax(&d));
    }

    #[test]
    fn argmax_never_skips_a_fitting_candidate() {
        // Stress: across a range of VRAM budgets, decision_is_argmax
        // holds — every candidate that fits has quality_rank ≤ picked.
        let shape = qwen25_coder_7b();
        for gib in 4..=48u64 {
            let free = gib * 1024 * 1024 * 1024;
            let d = select_auto_quant(shape, &all_quants(), free, 4096, 0.9).unwrap();
            assert!(
                decision_is_argmax(&d),
                "argmax violated at {gib} GiB: picked {:?}, candidates {:?}",
                d.selected,
                d.candidates,
            );
        }
    }

    #[test]
    fn available_subset_restricts_selection() {
        // If F16 is not offered, selector cannot pick it even with
        // massive VRAM. Ensures we respect the repo's `available_quants`.
        let shape = qwen25_coder_7b();
        let free = 80u64 * 1024 * 1024 * 1024;
        let offered = vec![QuantTag::Q4KM, QuantTag::Q5KM];
        let d = select_auto_quant(shape, &offered, free, 2048, 0.9).unwrap();
        assert!(matches!(d.selected, Some(QuantTag::Q5KM)));
    }

    #[test]
    fn cpu_fallback_branch_respects_both_invariants() {
        // When no quant fits, both predicates still hold trivially.
        let shape = qwen25_coder_7b();
        let free = 1u64 * 1024 * 1024 * 1024; // 1 GiB — too small
        let d = select_auto_quant(shape, &all_quants(), free, 2048, 0.9).unwrap();
        assert!(d.selected.is_none());
        assert!(decision_respects_budget(&d));
        assert!(decision_is_argmax(&d));
    }
}
