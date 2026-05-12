// SHIP-TWO-001 — `kv-cache-sizing-v1` algorithm-level PARTIAL
// discharge for FALSIFY-KV-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/kv-cache-sizing-v1.yaml`.
// Spec: KV cache memory sizing and bias absence invariants
// (Qwen3 Performance Parity Spec, Qwen3.5 hybrid layer accounting).

// ===========================================================================
// KV-001 — Per-token KV bytes: 2 * n_kv * d_k * bytes_per_element
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv001Verdict { Pass, Fail }

#[must_use]
pub const fn kv_bytes_per_token_per_layer(n_kv: u64, d_k: u64, bytes_per_element: u64) -> u64 {
    2u64.saturating_mul(n_kv).saturating_mul(d_k).saturating_mul(bytes_per_element)
}

#[must_use]
pub const fn verdict_from_per_token_bytes(
    n_kv: u64,
    d_k: u64,
    bytes_per_element: u64,
    observed: u64,
) -> Kv001Verdict {
    if n_kv == 0 || d_k == 0 || bytes_per_element == 0 { return Kv001Verdict::Fail; }
    let expected = kv_bytes_per_token_per_layer(n_kv, d_k, bytes_per_element);
    if observed == expected { Kv001Verdict::Pass } else { Kv001Verdict::Fail }
}

// ===========================================================================
// KV-002 — KV total monotonic in sequence length: S1 < S2 ⇒ kv(S1) < kv(S2)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv002Verdict { Pass, Fail }

#[must_use]
pub const fn kv_total_bytes(
    layers: u64,
    seq_len: u64,
    n_kv: u64,
    d_k: u64,
    bytes_per_element: u64,
) -> u64 {
    layers
        .saturating_mul(seq_len)
        .saturating_mul(2)
        .saturating_mul(n_kv)
        .saturating_mul(d_k)
        .saturating_mul(bytes_per_element)
}

#[must_use]
pub const fn verdict_from_monotonic_seq(
    layers: u64,
    n_kv: u64,
    d_k: u64,
    bytes_per_element: u64,
    s1: u64,
    s2: u64,
) -> Kv002Verdict {
    if layers == 0 || n_kv == 0 || d_k == 0 || bytes_per_element == 0 {
        return Kv002Verdict::Fail;
    }
    if s1 == 0 || s2 == 0 { return Kv002Verdict::Fail; }
    let kv1 = kv_total_bytes(layers, s1, n_kv, d_k, bytes_per_element);
    let kv2 = kv_total_bytes(layers, s2, n_kv, d_k, bytes_per_element);
    // const fn can't use Ord::cmp; explicit branching is fine.
    let monotone = if s1 < s2 {
        kv1 < kv2
    } else if s1 > s2 {
        kv1 > kv2
    } else {
        kv1 == kv2
    };
    if monotone { Kv002Verdict::Pass } else { Kv002Verdict::Fail }
}

// ===========================================================================
// KV-003 — Hybrid accounting: kv_layers = count(attention) ≤ total_layers
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayerType { Attention, Linear }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv003Verdict { Pass, Fail }

#[must_use]
pub fn count_attention_layers(layer_types: &[KvLayerType]) -> u64 {
    layer_types.iter().filter(|t| **t == KvLayerType::Attention).count() as u64
}

/// Pass iff `observed == count(Attention)` AND `observed ≤ len(layer_types)`.
#[must_use]
pub fn verdict_from_hybrid_accounting(layer_types: &[KvLayerType], observed: u64) -> Kv003Verdict {
    if layer_types.is_empty() { return Kv003Verdict::Fail; }
    let total = layer_types.len() as u64;
    let attn = count_attention_layers(layer_types);
    if observed != attn { return Kv003Verdict::Fail; }
    if observed > total { return Kv003Verdict::Fail; }
    Kv003Verdict::Pass
}

// ===========================================================================
// KV-004 — Bias absence: has_bias=false ⇒ bias_tensor_count == 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_bias_absence(has_bias: bool, bias_tensor_count: u64) -> Kv004Verdict {
    if !has_bias && bias_tensor_count != 0 { return Kv004Verdict::Fail; }
    // When has_bias=true, any non-negative count is acceptable; only the
    // false→nonzero direction is the regression class this gate catches.
    Kv004Verdict::Pass
}

// ===========================================================================
// KV-005 — Zero input identity: bias-free W @ 0 = 0 (byte-exact)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv005Verdict { Pass, Fail }

/// Pure scalar matvec (row-major): output[i] = Σ_j W[i*n+j] * x[j]
#[must_use]
pub fn bias_free_matvec(w: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    if w.len() != m * n || x.len() != n { return vec![]; }
    let mut out = vec![0.0_f32; m];
    for i in 0..m {
        let mut acc = 0.0_f32;
        for j in 0..n {
            acc += w[i * n + j] * x[j];
        }
        out[i] = acc;
    }
    out
}

/// Pass iff bias-free matvec on the all-zero input produces the all-zero output.
#[must_use]
pub fn verdict_from_zero_input_identity(w: &[f32], m: usize, n: usize) -> Kv005Verdict {
    if w.is_empty() || w.len() != m * n || m == 0 || n == 0 {
        return Kv005Verdict::Fail;
    }
    if !w.iter().all(|v| v.is_finite()) { return Kv005Verdict::Fail; }
    let zero_input = vec![0.0_f32; n];
    let out = bias_free_matvec(w, &zero_input, m, n);
    if out.len() != m { return Kv005Verdict::Fail; }
    if out.iter().all(|&v| v == 0.0) { Kv005Verdict::Pass } else { Kv005Verdict::Fail }
}

// ===========================================================================
// KV-006 — SIMD KV equivalence (contract tolerance=0.0 → byte-exact)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kv006Verdict { Pass, Fail }

/// Pass iff scalar and SIMD byte-counting paths produce identical u64 totals.
#[must_use]
pub const fn verdict_from_simd_byte_parity(scalar_bytes: u64, simd_bytes: u64) -> Kv006Verdict {
    if scalar_bytes == 0 || simd_bytes == 0 { return Kv006Verdict::Fail; }
    if scalar_bytes == simd_bytes { Kv006Verdict::Pass } else { Kv006Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // KV-001 (per-token bytes)
    #[test] fn kv001_pass_canonical_qwen2_7b_f16() {
        // Qwen2-7B: n_kv=4 KV heads, d_k=128, f16=2 bytes.
        // Expected per-token-per-layer: 2 * 4 * 128 * 2 = 2048 bytes.
        let bytes = kv_bytes_per_token_per_layer(4, 128, 2);
        assert_eq!(bytes, 2048);
        assert_eq!(verdict_from_per_token_bytes(4, 128, 2, 2048), Kv001Verdict::Pass);
    }
    #[test] fn kv001_fail_missing_factor_of_2() {
        // Contract's stated falsifier: "Forget factor of 2 to halve KV cache".
        assert_eq!(verdict_from_per_token_bytes(4, 128, 2, 1024), Kv001Verdict::Fail);
    }
    #[test] fn kv001_fail_zero_n_kv() {
        assert_eq!(verdict_from_per_token_bytes(0, 128, 2, 0), Kv001Verdict::Fail);
    }
    #[test] fn kv001_fail_wrong_bpe() {
        // f32 should be 4 bytes; using 2 (treating as f16) drops half the size.
        assert_eq!(verdict_from_per_token_bytes(4, 128, 4, 2048), Kv001Verdict::Fail);
    }

    // KV-002 (monotonic in S)
    #[test] fn kv002_pass_strict() {
        assert_eq!(
            verdict_from_monotonic_seq(28, 4, 128, 2, 100, 200),
            Kv002Verdict::Pass
        );
    }
    #[test] fn kv002_pass_equal() {
        assert_eq!(
            verdict_from_monotonic_seq(28, 4, 128, 2, 200, 200),
            Kv002Verdict::Pass
        );
    }
    #[test] fn kv002_pass_decreasing_seq_decreasing_kv() {
        // Construct a hypothetical broken state by passing inverted observed
        // (algorithm-level we use the formula; if formula is wrong the answer
        // wouldn't satisfy the order). The verdict catches if the formula is
        // not S-linear: we can simulate by passing inputs that would inflate
        // S1 path. Here we just check the verdict logic is correct on
        // canonical inputs (which is what proptest would explore).
        assert_eq!(
            verdict_from_monotonic_seq(28, 4, 128, 2, 200, 100),
            Kv002Verdict::Pass
        );
    }
    #[test] fn kv002_fail_zero() {
        assert_eq!(
            verdict_from_monotonic_seq(0, 4, 128, 2, 100, 200),
            Kv002Verdict::Fail
        );
        assert_eq!(
            verdict_from_monotonic_seq(28, 4, 128, 2, 0, 200),
            Kv002Verdict::Fail
        );
    }

    // KV-003 (hybrid accounting)
    #[test] fn kv003_pass_pure_attention() {
        let lt = vec![KvLayerType::Attention; 28];
        assert_eq!(verdict_from_hybrid_accounting(&lt, 28), Kv003Verdict::Pass);
    }
    #[test] fn kv003_pass_hybrid() {
        // 8 attention + 24 linear = 32 total, kv_layers=8.
        let mut lt = vec![KvLayerType::Linear; 24];
        lt.extend(std::iter::repeat(KvLayerType::Attention).take(8));
        assert_eq!(verdict_from_hybrid_accounting(&lt, 8), Kv003Verdict::Pass);
    }
    #[test] fn kv003_fail_undercount() {
        // Linear layers were silently counted as KV-contributing.
        let mut lt = vec![KvLayerType::Linear; 24];
        lt.extend(std::iter::repeat(KvLayerType::Attention).take(8));
        assert_eq!(verdict_from_hybrid_accounting(&lt, 32), Kv003Verdict::Fail);
    }
    #[test] fn kv003_fail_zero_layers() {
        assert_eq!(verdict_from_hybrid_accounting(&[], 0), Kv003Verdict::Fail);
    }

    // KV-004 (bias absence)
    #[test] fn kv004_pass_no_bias_zero_count() {
        assert_eq!(verdict_from_bias_absence(false, 0), Kv004Verdict::Pass);
    }
    #[test] fn kv004_pass_with_bias_any_count() {
        assert_eq!(verdict_from_bias_absence(true, 0), Kv004Verdict::Pass);
        assert_eq!(verdict_from_bias_absence(true, 4), Kv004Verdict::Pass);
    }
    #[test] fn kv004_fail_no_bias_but_tensors_present() {
        // The exact regression class: config says no bias but a bias
        // tensor leaked in from import.
        assert_eq!(verdict_from_bias_absence(false, 1), Kv004Verdict::Fail);
        assert_eq!(verdict_from_bias_absence(false, 4), Kv004Verdict::Fail);
    }

    // KV-005 (zero-input identity)
    #[test] fn kv005_pass_identity_3x3() {
        let w = vec![
            1.0_f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        assert_eq!(verdict_from_zero_input_identity(&w, 3, 3), Kv005Verdict::Pass);
    }
    #[test] fn kv005_pass_random_nonzero_w() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
        assert_eq!(verdict_from_zero_input_identity(&w, 16, 16), Kv005Verdict::Pass);
    }
    #[test] fn kv005_fail_dim_mismatch() {
        let w = vec![1.0_f32; 8];
        // m * n = 9 but w.len() = 8.
        assert_eq!(verdict_from_zero_input_identity(&w, 3, 3), Kv005Verdict::Fail);
    }
    #[test] fn kv005_fail_zero_dim() {
        let w = vec![1.0_f32; 1];
        assert_eq!(verdict_from_zero_input_identity(&w, 0, 1), Kv005Verdict::Fail);
    }
    #[test] fn kv005_fail_nan_in_w() {
        let w = vec![1.0_f32, f32::NAN, 3.0, 4.0];
        assert_eq!(verdict_from_zero_input_identity(&w, 2, 2), Kv005Verdict::Fail);
    }

    // KV-006 (SIMD parity)
    #[test] fn kv006_pass_identical() {
        assert_eq!(verdict_from_simd_byte_parity(2048, 2048), Kv006Verdict::Pass);
    }
    #[test] fn kv006_fail_drift() {
        assert_eq!(verdict_from_simd_byte_parity(2048, 2049), Kv006Verdict::Fail);
    }
    #[test] fn kv006_fail_zero() {
        assert_eq!(verdict_from_simd_byte_parity(0, 2048), Kv006Verdict::Fail);
    }

    // bias_free_matvec sanity (helper)
    #[test] fn matvec_zero_input() {
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x = vec![0.0_f32, 0.0];
        let y = bias_free_matvec(&w, &x, 2, 2);
        assert_eq!(y, vec![0.0_f32, 0.0]);
    }
    #[test] fn matvec_canonical() {
        // [[1, 2], [3, 4]] @ [1, 1] = [3, 7]
        let w = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x = vec![1.0_f32, 1.0];
        let y = bias_free_matvec(&w, &x, 2, 2);
        assert_eq!(y, vec![3.0_f32, 7.0]);
    }
}
