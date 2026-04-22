//! Unit tests for `auto_quant` (extracted from `auto_quant.rs` to keep file-size invariant).
//!
//! Included via `#[cfg(test)] #[path = "auto_quant_tests.rs"] mod tests;` in the parent.

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
    let d = select_auto_quant(shape, &all_quants(), free, 2048, DEFAULT_SAFETY_FACTOR).unwrap();
    assert!(decision_respects_budget(&d));
}

#[test]
fn falsify_002_sub_claim_argmax_of_fitting() {
    // CRUX-A-10 ALGO-002 sub-claim of FALSIFY-002: no strictly-
    // higher-quality quant fits within budget.
    let shape = qwen25_coder_7b();
    let free = 16u64 * 1024 * 1024 * 1024; // 16 GiB — tighter
    let d = select_auto_quant(shape, &all_quants(), free, 8192, DEFAULT_SAFETY_FACTOR).unwrap();
    assert!(decision_is_argmax(&d));
}

#[test]
fn falsify_003_sub_claim_ctx_doubling_never_raises_quality() {
    // CRUX-A-10 ALGO-003 sub-claim of FALSIFY-003: doubling
    // ctx_len never raises the selected quant's quality_rank.
    let shape = qwen25_coder_7b();
    let free = 12u64 * 1024 * 1024 * 1024;
    let a = select_auto_quant(shape, &all_quants(), free, 2048, DEFAULT_SAFETY_FACTOR).unwrap();
    let b = select_auto_quant(shape, &all_quants(), free, 4096, DEFAULT_SAFETY_FACTOR).unwrap();
    let c = select_auto_quant(shape, &all_quants(), free, 32768, DEFAULT_SAFETY_FACTOR).unwrap();
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
    let d = select_auto_quant(shape, &[QuantTag::F16], free, 2048, 0.9).unwrap();
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
