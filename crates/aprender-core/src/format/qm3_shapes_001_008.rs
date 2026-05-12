// SHIP-TWO-001 — `qwen3moe-shapes-v1` algorithm-level PARTIAL discharge
// for FALSIFY-QM3-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/qwen3moe-shapes-v1.yaml`.
// Spec: M32 milestone (Qwen3-MoE forward parity).

// ===========================================================================
// Canonical Qwen3-MoE-235B-A22B shape constants
// ===========================================================================

pub const AC_QM3_HIDDEN_DIM: u64 = 4096;
pub const AC_QM3_NUM_HEADS: u64 = 64;
pub const AC_QM3_NUM_KV_HEADS: u64 = 4;
pub const AC_QM3_HEAD_DIM: u64 = 128;
pub const AC_QM3_Q_DIM: u64 = 8192;
pub const AC_QM3_KV_DIM: u64 = 512;
pub const AC_QM3_EXPERT_INTERMEDIATE: u64 = 1536;
/// Per-expert parameter count: 3 (gate, up, down) × hidden × intermediate.
pub const AC_QM3_PER_EXPERT_PARAMS: u64 = 3 * AC_QM3_HIDDEN_DIM * AC_QM3_EXPERT_INTERMEDIATE;
pub const AC_QM3_NUM_EXPERTS: u64 = 128;
pub const AC_QM3_TOP_K: u64 = 8;

// ===========================================================================
// QM3-001 — Q projection: n_h * d_k = 8192
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_q_projection(n_h: u64, d_k: u64) -> Qm3Shape001Verdict {
    if n_h == 0 || d_k == 0 { return Qm3Shape001Verdict::Fail; }
    if n_h * d_k == AC_QM3_Q_DIM { Qm3Shape001Verdict::Pass } else { Qm3Shape001Verdict::Fail }
}

// ===========================================================================
// QM3-002 — KV projection: n_kv * d_k = 512
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_kv_projection(n_kv: u64, d_k: u64) -> Qm3Shape002Verdict {
    if n_kv == 0 || d_k == 0 { return Qm3Shape002Verdict::Fail; }
    if n_kv * d_k == AC_QM3_KV_DIM { Qm3Shape002Verdict::Pass } else { Qm3Shape002Verdict::Fail }
}

// ===========================================================================
// QM3-003 — GQA divisibility: 64 % 4 == 0, ratio == 16
// ===========================================================================

pub const AC_QM3_GQA_RATIO: u64 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_gqa_divisibility(n_h: u64, n_kv: u64) -> Qm3Shape003Verdict {
    if n_h == 0 || n_kv == 0 { return Qm3Shape003Verdict::Fail; }
    if !n_h.is_multiple_of(n_kv) { return Qm3Shape003Verdict::Fail; }
    if n_h / n_kv == AC_QM3_GQA_RATIO { Qm3Shape003Verdict::Pass } else { Qm3Shape003Verdict::Fail }
}

// ===========================================================================
// QM3-004 — MoE expert shape: 3 * hidden * intermediate = 18,874,368
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_expert_params(hidden: u64, intermediate: u64) -> Qm3Shape004Verdict {
    if hidden == 0 || intermediate == 0 { return Qm3Shape004Verdict::Fail; }
    let computed = 3 * hidden * intermediate;
    if computed == AC_QM3_PER_EXPERT_PARAMS { Qm3Shape004Verdict::Pass } else { Qm3Shape004Verdict::Fail }
}

// ===========================================================================
// QM3-005 — MoE router: top_k=8 selects 8 of 128 experts
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape005Verdict { Pass, Fail }

/// Pass iff `top_k > 0 AND top_k < num_experts AND num_experts % top_k == 0
/// AND (top_k, num_experts) == (8, 128)` per spec.
#[must_use]
pub const fn verdict_from_router_top_k(top_k: u64, num_experts: u64) -> Qm3Shape005Verdict {
    if top_k == 0 || num_experts == 0 { return Qm3Shape005Verdict::Fail; }
    if top_k >= num_experts { return Qm3Shape005Verdict::Fail; }
    if !num_experts.is_multiple_of(top_k) { return Qm3Shape005Verdict::Fail; }
    if top_k == AC_QM3_TOP_K && num_experts == AC_QM3_NUM_EXPERTS {
        Qm3Shape005Verdict::Pass
    } else {
        Qm3Shape005Verdict::Fail
    }
}

// ===========================================================================
// QM3-006 — O projection is transpose of Q: O=[hidden, q_dim], Q=[q_dim, hidden]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_o_projection_transpose(q_shape: [u64; 2], o_shape: [u64; 2]) -> Qm3Shape006Verdict {
    if q_shape != [AC_QM3_Q_DIM, AC_QM3_HIDDEN_DIM] { return Qm3Shape006Verdict::Fail; }
    if o_shape != [AC_QM3_HIDDEN_DIM, AC_QM3_Q_DIM] { return Qm3Shape006Verdict::Fail; }
    if o_shape[0] != q_shape[1] || o_shape[1] != q_shape[0] { return Qm3Shape006Verdict::Fail; }
    Qm3Shape006Verdict::Pass
}

// ===========================================================================
// QM3-007 — RoPE freqs strictly decreasing
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape007Verdict { Pass, Fail }

#[must_use]
pub fn rope_freqs(d_k: u64, base: f32) -> Vec<f32> {
    if d_k == 0 || !d_k.is_multiple_of(2) || base <= 1.0 { return vec![]; }
    let half = d_k / 2;
    (0..half).map(|i| {
        let p = -2.0_f32 * (i as f32) / (d_k as f32);
        base.powf(p)
    }).collect()
}

#[must_use]
pub fn verdict_from_rope_decreasing(d_k: u64, base: f32) -> Qm3Shape007Verdict {
    let freqs = rope_freqs(d_k, base);
    if freqs.len() < 2 { return Qm3Shape007Verdict::Fail; }
    for w in freqs.windows(2) {
        if w[0] <= w[1] { return Qm3Shape007Verdict::Fail; }
    }
    Qm3Shape007Verdict::Pass
}

// ===========================================================================
// QM3-008 — SIMD vs scalar shape equivalence
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3Shape008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_simd_shape_match(scalar_shape: &[u64], simd_shape: &[u64]) -> Qm3Shape008Verdict {
    if scalar_shape.is_empty() || simd_shape.is_empty() { return Qm3Shape008Verdict::Fail; }
    if scalar_shape == simd_shape { Qm3Shape008Verdict::Pass } else { Qm3Shape008Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn qm3_001_pass() { assert_eq!(verdict_from_q_projection(64, 128), Qm3Shape001Verdict::Pass); }
    #[test] fn qm3_001_fail() { assert_eq!(verdict_from_q_projection(63, 128), Qm3Shape001Verdict::Fail); }

    #[test] fn qm3_002_pass() { assert_eq!(verdict_from_kv_projection(4, 128), Qm3Shape002Verdict::Pass); }
    #[test] fn qm3_002_fail() { assert_eq!(verdict_from_kv_projection(5, 128), Qm3Shape002Verdict::Fail); }

    #[test] fn qm3_003_pass() { assert_eq!(verdict_from_gqa_divisibility(64, 4), Qm3Shape003Verdict::Pass); }
    #[test] fn qm3_003_fail_indivisible() { assert_eq!(verdict_from_gqa_divisibility(64, 5), Qm3Shape003Verdict::Fail); }
    #[test] fn qm3_003_fail_wrong_ratio() { assert_eq!(verdict_from_gqa_divisibility(64, 8), Qm3Shape003Verdict::Fail); }

    #[test] fn qm3_004_pass() { assert_eq!(verdict_from_expert_params(4096, 1536), Qm3Shape004Verdict::Pass); }
    #[test] fn qm3_004_fail_wrong_intermediate() { assert_eq!(verdict_from_expert_params(4096, 1024), Qm3Shape004Verdict::Fail); }
    #[test] fn qm3_004_fail_wrong_hidden() { assert_eq!(verdict_from_expert_params(8192, 1536), Qm3Shape004Verdict::Fail); }

    #[test] fn qm3_005_pass_canonical() { assert_eq!(verdict_from_router_top_k(8, 128), Qm3Shape005Verdict::Pass); }
    #[test] fn qm3_005_fail_top_k_eq_experts() { assert_eq!(verdict_from_router_top_k(128, 128), Qm3Shape005Verdict::Fail); }
    #[test] fn qm3_005_fail_top_k_above_experts() { assert_eq!(verdict_from_router_top_k(129, 128), Qm3Shape005Verdict::Fail); }
    #[test] fn qm3_005_fail_indivisible() { assert_eq!(verdict_from_router_top_k(7, 128), Qm3Shape005Verdict::Fail); }
    #[test] fn qm3_005_fail_zero() { assert_eq!(verdict_from_router_top_k(0, 128), Qm3Shape005Verdict::Fail); }
    #[test] fn qm3_005_fail_off_top_k() {
        // Even though (4, 128) divides cleanly, contract pins top_k=8.
        assert_eq!(verdict_from_router_top_k(4, 128), Qm3Shape005Verdict::Fail);
    }

    #[test] fn qm3_006_pass() {
        assert_eq!(
            verdict_from_o_projection_transpose([8192, 4096], [4096, 8192]),
            Qm3Shape006Verdict::Pass
        );
    }
    #[test] fn qm3_006_fail_swapped() {
        assert_eq!(
            verdict_from_o_projection_transpose([4096, 8192], [8192, 4096]),
            Qm3Shape006Verdict::Fail
        );
    }

    #[test] fn qm3_007_pass() { assert_eq!(verdict_from_rope_decreasing(128, 1_000_000.0), Qm3Shape007Verdict::Pass); }
    #[test] fn qm3_007_fail_zero_d_k() { assert_eq!(verdict_from_rope_decreasing(0, 10000.0), Qm3Shape007Verdict::Fail); }

    #[test] fn qm3_008_pass() {
        assert_eq!(verdict_from_simd_shape_match(&[8192, 4096], &[8192, 4096]), Qm3Shape008Verdict::Pass);
    }
    #[test] fn qm3_008_fail() {
        assert_eq!(verdict_from_simd_shape_match(&[8192, 4096], &[4096, 8192]), Qm3Shape008Verdict::Fail);
    }

    #[test] fn provenance_constants() {
        assert_eq!(AC_QM3_HIDDEN_DIM, 4096);
        assert_eq!(AC_QM3_NUM_HEADS, 64);
        assert_eq!(AC_QM3_NUM_KV_HEADS, 4);
        assert_eq!(AC_QM3_HEAD_DIM, 128);
        assert_eq!(AC_QM3_Q_DIM, 8192);
        assert_eq!(AC_QM3_KV_DIM, 512);
        assert_eq!(AC_QM3_EXPERT_INTERMEDIATE, 1536);
        assert_eq!(AC_QM3_PER_EXPERT_PARAMS, 18_874_368);
        assert_eq!(AC_QM3_NUM_EXPERTS, 128);
        assert_eq!(AC_QM3_TOP_K, 8);
        assert_eq!(AC_QM3_GQA_RATIO, 16);
    }

    #[test] fn provenance_self_consistency() {
        assert_eq!(AC_QM3_NUM_HEADS * AC_QM3_HEAD_DIM, AC_QM3_Q_DIM);
        assert_eq!(AC_QM3_NUM_KV_HEADS * AC_QM3_HEAD_DIM, AC_QM3_KV_DIM);
        assert_eq!(AC_QM3_NUM_HEADS / AC_QM3_NUM_KV_HEADS, AC_QM3_GQA_RATIO);
        assert_eq!(AC_QM3_NUM_EXPERTS % AC_QM3_TOP_K, 0);
    }
}
