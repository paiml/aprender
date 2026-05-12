// SHIP-TWO-001 — `inference-pipeline-v1` algorithm-level PARTIAL
// discharge for FALSIFY-INF-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/inference-pipeline-v1.yaml`.

// ===========================================================================
// INF-001 — Prefill output shape == [seq_len, d_model]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_prefill_shape(observed: &[u64], seq_len: u64, d_model: u64) -> Inf001Verdict {
    if seq_len == 0 || d_model == 0 { return Inf001Verdict::Fail; }
    if observed.len() != 2 { return Inf001Verdict::Fail; }
    if observed[0] == seq_len && observed[1] == d_model { Inf001Verdict::Pass } else { Inf001Verdict::Fail }
}

// ===========================================================================
// INF-002 — Decode (single token) shape == [1, d_model]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_decode_shape(observed: &[u64], d_model: u64) -> Inf002Verdict {
    if d_model == 0 { return Inf002Verdict::Fail; }
    if observed.len() != 2 { return Inf002Verdict::Fail; }
    if observed[0] == 1 && observed[1] == d_model { Inf002Verdict::Pass } else { Inf002Verdict::Fail }
}

// ===========================================================================
// INF-003 — Residual stream: h_{l+1} == h_l + sublayer(norm(h_l))
// ===========================================================================

pub const AC_INF_003_TOLERANCE: f64 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf003Verdict { Pass, Fail }

/// Pass iff for every i: |observed_h_next[i] - (h_l[i] + sublayer_out[i])| < tol.
#[must_use]
pub fn verdict_from_residual_stream(
    h_l: &[f32],
    sublayer_out: &[f32],
    h_next: &[f32],
) -> Inf003Verdict {
    if h_l.is_empty() || h_l.len() != sublayer_out.len() || h_l.len() != h_next.len() {
        return Inf003Verdict::Fail;
    }
    if h_l.iter().chain(sublayer_out.iter()).chain(h_next.iter()).any(|v| !v.is_finite()) {
        return Inf003Verdict::Fail;
    }
    for i in 0..h_l.len() {
        let expected = (h_l[i] as f64) + (sublayer_out[i] as f64);
        let observed = h_next[i] as f64;
        if (observed - expected).abs() > AC_INF_003_TOLERANCE { return Inf003Verdict::Fail; }
    }
    Inf003Verdict::Pass
}

// ===========================================================================
// INF-004 — Layer schedule exhaustiveness: |attention| + |ffn| == num_layers
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_layer_schedule_exhaustive(
    num_attention: u64,
    num_layer_norm: u64,
    num_layers: u64,
) -> Inf004Verdict {
    if num_layers == 0 { return Inf004Verdict::Fail; }
    if num_attention + num_layer_norm == num_layers { Inf004Verdict::Pass } else { Inf004Verdict::Fail }
}

// ===========================================================================
// INF-005 — KV cache grows by fixed amount per token
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf005Verdict { Pass, Fail }

/// Pass iff every successive (cache[t+1] - cache[t]) is the same constant.
#[must_use]
pub fn verdict_from_kv_cache_growth(cache_sizes: &[u64]) -> Inf005Verdict {
    if cache_sizes.len() < 3 { return Inf005Verdict::Fail; }
    let first_delta = cache_sizes[1] as i64 - cache_sizes[0] as i64;
    if first_delta <= 0 { return Inf005Verdict::Fail; }
    for w in cache_sizes.windows(2) {
        let d = w[1] as i64 - w[0] as i64;
        if d != first_delta { return Inf005Verdict::Fail; }
    }
    Inf005Verdict::Pass
}

// ===========================================================================
// INF-006 — Residual dim preservation: shape(h_{l+1}) == shape(h_l)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_residual_dim_preserved(
    layer_shapes: &[Vec<u64>],
) -> Inf006Verdict {
    if layer_shapes.is_empty() { return Inf006Verdict::Fail; }
    let first = &layer_shapes[0];
    if first.is_empty() { return Inf006Verdict::Fail; }
    for shape in &layer_shapes[1..] {
        if shape != first { return Inf006Verdict::Fail; }
    }
    Inf006Verdict::Pass
}

// ===========================================================================
// INF-007 — Activation finiteness: no NaN/Inf in hidden states
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inf007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_activation_finite(hidden_states: &[f32]) -> Inf007Verdict {
    if hidden_states.is_empty() { return Inf007Verdict::Fail; }
    if hidden_states.iter().all(|v| v.is_finite()) { Inf007Verdict::Pass } else { Inf007Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // INF-001
    #[test] fn inf001_pass_canonical() {
        assert_eq!(verdict_from_prefill_shape(&[16, 3584], 16, 3584), Inf001Verdict::Pass);
    }
    #[test] fn inf001_fail_swapped() {
        assert_eq!(verdict_from_prefill_shape(&[3584, 16], 16, 3584), Inf001Verdict::Fail);
    }
    #[test] fn inf001_fail_wrong_rank() {
        assert_eq!(verdict_from_prefill_shape(&[1, 16, 3584], 16, 3584), Inf001Verdict::Fail);
    }
    #[test] fn inf001_fail_zero_d() {
        assert_eq!(verdict_from_prefill_shape(&[16, 0], 16, 0), Inf001Verdict::Fail);
    }

    // INF-002
    #[test] fn inf002_pass() {
        assert_eq!(verdict_from_decode_shape(&[1, 3584], 3584), Inf002Verdict::Pass);
    }
    #[test] fn inf002_fail_seq_not_one() {
        assert_eq!(verdict_from_decode_shape(&[2, 3584], 3584), Inf002Verdict::Fail);
    }
    #[test] fn inf002_fail_d_mismatch() {
        assert_eq!(verdict_from_decode_shape(&[1, 3584], 4096), Inf002Verdict::Fail);
    }

    // INF-003
    #[test] fn inf003_pass_canonical() {
        let h = vec![1.0_f32, 2.0, 3.0];
        let s = vec![0.1_f32, 0.2, 0.3];
        let n: Vec<f32> = h.iter().zip(&s).map(|(a, b)| a + b).collect();
        assert_eq!(verdict_from_residual_stream(&h, &s, &n), Inf003Verdict::Pass);
    }
    #[test] fn inf003_fail_missing_residual() {
        let h = vec![1.0_f32, 2.0];
        let s = vec![0.1_f32, 0.2];
        let n = vec![0.1_f32, 0.2]; // h dropped
        assert_eq!(verdict_from_residual_stream(&h, &s, &n), Inf003Verdict::Fail);
    }
    #[test] fn inf003_fail_dim_mismatch() {
        let h = vec![1.0_f32, 2.0];
        let s = vec![0.1_f32];
        let n = vec![1.1_f32, 2.2];
        assert_eq!(verdict_from_residual_stream(&h, &s, &n), Inf003Verdict::Fail);
    }

    // INF-004
    #[test] fn inf004_pass_28_layers() {
        // Qwen2.5-Coder-7B has 28 layers, each (attention + 2 norms typically).
        assert_eq!(verdict_from_layer_schedule_exhaustive(28, 0, 28), Inf004Verdict::Pass);
    }
    #[test] fn inf004_pass_split() {
        assert_eq!(verdict_from_layer_schedule_exhaustive(20, 8, 28), Inf004Verdict::Pass);
    }
    #[test] fn inf004_fail_short() {
        assert_eq!(verdict_from_layer_schedule_exhaustive(27, 0, 28), Inf004Verdict::Fail);
    }
    #[test] fn inf004_fail_zero() {
        assert_eq!(verdict_from_layer_schedule_exhaustive(0, 0, 0), Inf004Verdict::Fail);
    }

    // INF-005
    #[test] fn inf005_pass_constant_growth() {
        let sizes = vec![100, 110, 120, 130, 140];
        assert_eq!(verdict_from_kv_cache_growth(&sizes), Inf005Verdict::Pass);
    }
    #[test] fn inf005_fail_variable() {
        let sizes = vec![100, 110, 125, 130];
        assert_eq!(verdict_from_kv_cache_growth(&sizes), Inf005Verdict::Fail);
    }
    #[test] fn inf005_fail_no_growth() {
        let sizes = vec![100, 100, 100];
        assert_eq!(verdict_from_kv_cache_growth(&sizes), Inf005Verdict::Fail);
    }
    #[test] fn inf005_fail_too_few() {
        let sizes = vec![100, 110];
        assert_eq!(verdict_from_kv_cache_growth(&sizes), Inf005Verdict::Fail);
    }

    // INF-006
    #[test] fn inf006_pass_uniform() {
        let shapes = vec![vec![16, 3584]; 28];
        assert_eq!(verdict_from_residual_dim_preserved(&shapes), Inf006Verdict::Pass);
    }
    #[test] fn inf006_fail_dim_change() {
        let mut shapes = vec![vec![16_u64, 3584]; 28];
        shapes[5] = vec![16, 4096];
        assert_eq!(verdict_from_residual_dim_preserved(&shapes), Inf006Verdict::Fail);
    }
    #[test] fn inf006_fail_empty() {
        assert_eq!(verdict_from_residual_dim_preserved(&[]), Inf006Verdict::Fail);
    }

    // INF-007
    #[test] fn inf007_pass_finite() {
        let h = vec![1.0_f32, -2.0, 0.5, 3.14];
        assert_eq!(verdict_from_activation_finite(&h), Inf007Verdict::Pass);
    }
    #[test] fn inf007_fail_nan() {
        let h = vec![1.0_f32, f32::NAN];
        assert_eq!(verdict_from_activation_finite(&h), Inf007Verdict::Fail);
    }
    #[test] fn inf007_fail_inf() {
        let h = vec![1.0_f32, f32::INFINITY];
        assert_eq!(verdict_from_activation_finite(&h), Inf007Verdict::Fail);
    }
    #[test] fn inf007_fail_empty() {
        assert_eq!(verdict_from_activation_finite(&[]), Inf007Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_tolerance() {
        assert!((AC_INF_003_TOLERANCE - 1e-5).abs() < 1e-12);
    }
}
