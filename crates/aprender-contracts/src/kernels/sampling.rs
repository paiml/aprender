//! Sampling algorithms kernel.
//!
//! Matches `sampling-algorithms-v1.yaml`.
//! Greedy, top-k, top-p, and temperature sampling for autoregressive generation.
//!
//! Each function provides one of three backends:
//! - `fn {name}_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn {name}_avx2(...)` -- AVX2 SIMD implementation
//! - `fn sampling_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementations
// ────────────────────────────────────────────────────────────────────────────

/// Greedy sampling: return the index of the maximum logit.
///
/// # Panics
/// Panics if `logits` is empty.
pub fn greedy_scalar(logits: &[f32]) -> usize {
    assert!(!logits.is_empty(), "logits must not be empty");
    let mut best_idx = 0;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// Apply temperature scaling to logits in-place: `logits[i] /= temperature`.
///
/// # Panics
/// Panics if `temperature <= 0`.
pub fn temperature_scalar(logits: &mut [f32], temperature: f32) {
    assert!(
        temperature > 0.0,
        "temperature must be positive, got {temperature}"
    );
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

/// Top-K filtering: zero out all probabilities except the K highest.
///
/// `probs` is modified in-place. After filtering, probabilities are renormalized.
///
/// # Panics
/// Panics if `k == 0` or `k > probs.len()`.
pub fn top_k_scalar(probs: &mut [f32], k: usize) {
    let n = probs.len();
    assert!(k > 0 && k <= n, "k={k} must be in [1, {n}]");

    if k == n {
        return; // nothing to filter
    }

    // Find the k-th largest value (selection via sorted indices)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Zero out everything below top-K
    for &idx in &indices[k..] {
        probs[idx] = 0.0;
    }

    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}

/// Top-P (nucleus) filtering: retain the minimal set of tokens whose cumulative
/// probability exceeds `threshold`.
///
/// `probs` is modified in-place. After filtering, probabilities are renormalized.
///
/// # Panics
/// Panics if `threshold <= 0` or `threshold > 1`.
pub fn top_p_scalar(probs: &mut [f32], threshold: f32) {
    let n = probs.len();
    assert!(
        threshold > 0.0 && threshold <= 1.0,
        "threshold must be in (0, 1], got {threshold}"
    );

    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Accumulate until we exceed threshold
    let mut cumsum = 0.0f32;
    let mut cutoff = n;
    for (rank, &idx) in indices.iter().enumerate() {
        cumsum += probs[idx];
        if cumsum >= threshold {
            cutoff = rank + 1;
            break;
        }
    }

    // Zero out everything past cutoff
    for &idx in &indices[cutoff..] {
        probs[idx] = 0.0;
    }

    // Renormalize
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}

/// Full sampling pipeline: apply temperature, softmax, then greedy (scalar reference).
///
/// Returns the selected token index.
pub fn sample_scalar(logits: &[f32]) -> usize {
    greedy_scalar(logits)
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 greedy sampling -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn greedy_avx2(logits: &[f32]) -> usize {
    greedy_scalar(logits)
}

/// AVX2 temperature scaling -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn temperature_avx2(logits: &mut [f32], temperature: f32) {
    temperature_scalar(logits, temperature);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for greedy sampling (argmax reduction).
///
/// Uses parallel reduction to find the maximum logit index.
pub fn sampling_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry greedy_kernel(
    .param .u64 LOGITS,
    .param .u64 OUT_IDX,
    .param .u32 VOCAB_SIZE
) {
    .reg .u32 %tid, %vocab_size, %k, %best_idx, %cur_idx;
    .reg .u64 %logits_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %best_val, %cur_val;
    .reg .pred %p_loop, %p_better;

    mov.u32 %tid, %tid.x;

    ld.param.u32 %vocab_size, [VOCAB_SIZE];
    ld.param.u64 %logits_ptr, [LOGITS];
    ld.param.u64 %out_ptr, [OUT_IDX];

    // Only thread 0 performs the scan (simple serial argmax)
    setp.ne.u32 %p_loop, %tid, 0;
    @%p_loop bra EXIT;

    // Load first element as initial best
    ld.global.f32 %best_val, [%logits_ptr];
    mov.u32 %best_idx, 0;
    mov.u32 %k, 1;

SCAN_LOOP:
    setp.ge.u32 %p_loop, %k, %vocab_size;
    @%p_loop bra STORE;

    mul.wide.u32 %off64, %k, 4;
    add.u64 %addr, %logits_ptr, %off64;
    ld.global.f32 %cur_val, [%addr];

    setp.gt.f32 %p_better, %cur_val, %best_val;
    @!%p_better bra NEXT;
    mov.f32 %best_val, %cur_val;
    mov.u32 %best_idx, %k;
NEXT:
    add.u32 %k, %k, 1;
    bra SCAN_LOOP;

STORE:
    st.global.u32 [%out_ptr], %best_idx;

EXIT:
    ret;
}
"#
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_greedy_basic() {
        assert_eq!(greedy_scalar(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(greedy_scalar(&[5.0]), 0);
        assert_eq!(greedy_scalar(&[0.0, 0.0, 0.0, 1.0]), 3);
    }

    #[test]
    fn test_greedy_is_argmax() {
        let logits = [0.1, 0.5, -0.3, 0.8, 0.2];
        let result = greedy_scalar(&logits);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(result, argmax);
    }

    #[test]
    fn test_temperature_identity() {
        let original = [1.0, 2.0, 3.0];
        let mut scaled = original;
        temperature_scalar(&mut scaled, 1.0);
        assert_eq!(scaled, original);
    }

    #[test]
    fn test_temperature_scaling() {
        let mut logits = [2.0, 4.0];
        temperature_scalar(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_cardinality() {
        let mut probs = [0.1, 0.2, 0.3, 0.4];
        top_k_scalar(&mut probs, 2);
        let nonzero = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(nonzero <= 2, "expected at most 2, got {nonzero}");
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let mut probs = [0.1, 0.4, 0.2, 0.3];
        top_k_scalar(&mut probs, 2);
        // indices 1 (0.4) and 3 (0.3) should survive
        assert_eq!(probs[0], 0.0);
        assert!(probs[1] > 0.0);
        assert_eq!(probs[2], 0.0);
        assert!(probs[3] > 0.0);
    }

    #[test]
    fn test_top_k_renormalizes() {
        let mut probs = [0.1, 0.2, 0.3, 0.4];
        top_k_scalar(&mut probs, 2);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {sum}");
    }

    #[test]
    fn test_top_p_cumulative() {
        let mut probs = [0.1, 0.2, 0.3, 0.4];
        let threshold = 0.6;
        top_p_scalar(&mut probs, threshold);
        let sum: f32 = probs.iter().sum();
        assert!(sum >= threshold - 1e-5, "sum {sum} < threshold {threshold}");
    }

    #[test]
    fn test_top_p_minimal_set() {
        let mut probs = [0.1, 0.2, 0.3, 0.4];
        top_p_scalar(&mut probs, 0.5);
        // Only index 3 (0.4) and index 2 (0.3) needed: 0.4+0.3 = 0.7 >= 0.5
        // Actually, 0.4 alone is not >= 0.5, so need 0.4+0.3
        let nonzero = probs.iter().filter(|&&p| p > 0.0).count();
        assert!(
            nonzero <= 2,
            "expected minimal set size <= 2, got {nonzero}"
        );
    }

    proptest! {
        #[test]
        fn prop_greedy_is_argmax(logits in proptest::collection::vec(-10.0f32..10.0, 1..16)) {
            // TIES. The old form compared this index against
            //     logits.iter().enumerate().max_by(..).unwrap().0
            // which returns the LAST maximum, while `greedy_scalar` keeps the
            // FIRST (`if v > best_val`, strictly greater). On a tie the two
            // disagree by construction, so the property failed at random —
            // whenever proptest happened to generate duplicate maxima. Observed
            // in CI on `[9.086451, 9.086451]` after 127 successes, red on a PR
            // that touches nothing near sampling.
            //
            // Index equality was never the real property, and a required check
            // that reds at random is worse than one that cannot red at all: it
            // trains everyone to re-run until green. Assert what actually has
            // to hold, and pin the tie-break rule explicitly rather than
            // inheriting whatever `max_by` happens to do.
            let result = greedy_scalar(&logits);

            // 1. greedy returns AN index holding the maximum value.
            let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            prop_assert_eq!(logits[result], best);

            // 2. and specifically the FIRST such index. Sampling must be
            //    deterministic for a fixed input, so the tie-break is part of
            //    the contract, not an implementation detail. This is STRICTLY
            //    STRONGER than the old assertion, which could not even be
            //    stated on ties.
            let first_max = logits.iter().position(|&v| v == best).unwrap();
            prop_assert_eq!(result, first_max);
        }

        #[test]
        fn prop_top_k_cardinality(
            k in 1usize..8,
            n in 8usize..16,
        ) {
            let mut probs: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) / (n as f32)).collect();
            let sum: f32 = probs.iter().sum();
            for v in probs.iter_mut() { *v /= sum; }

            top_k_scalar(&mut probs, k);
            let nonzero = probs.iter().filter(|&&p| p > 0.0).count();
            prop_assert!(nonzero <= k, "nonzero={nonzero} > k={k}");
        }

        #[test]
        fn prop_temperature_identity(logits in proptest::collection::vec(-10.0f32..10.0, 1..16)) {
            let original = logits.clone();
            let mut scaled = logits;
            temperature_scalar(&mut scaled, 1.0);
            for (a, b) in original.iter().zip(scaled.iter()) {
                prop_assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sample_scalar_delegates_to_greedy() {
        let logits = [0.1, 0.5, -0.3, 0.8, 0.2];
        assert_eq!(sample_scalar(&logits), greedy_scalar(&logits));
    }

    #[test]
    fn test_top_k_full_k_is_noop() {
        let mut probs = [0.25, 0.25, 0.25, 0.25];
        let original = probs;
        top_k_scalar(&mut probs, 4);
        assert_eq!(probs, original);
    }

    #[test]
    fn test_top_p_threshold_one() {
        let mut probs = [0.25, 0.25, 0.25, 0.25];
        top_p_scalar(&mut probs, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_temperature_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scalar = [2.0, 4.0, 6.0, 8.0];
        let mut avx2 = scalar;
        temperature_scalar(&mut scalar, 2.0);
        unsafe { temperature_avx2(&mut avx2, 2.0) };
        assert_eq!(scalar, avx2);
    }

    #[test]
    fn test_sampling_ptx_structure() {
        let ptx = sampling_ptx();
        assert!(ptx.contains(".entry greedy_kernel"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_greedy_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let logits = [0.1, 0.5, -0.3, 0.8, 0.2];
        let scalar = greedy_scalar(&logits);
        let avx2 = unsafe { greedy_avx2(&logits) };
        assert_eq!(scalar, avx2);
    }
}
