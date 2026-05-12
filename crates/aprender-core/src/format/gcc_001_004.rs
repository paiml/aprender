// SHIP-TWO-001 — `gguf-cpu-cache-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CC-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/gguf-cpu-cache-v1.yaml`.
// Spec: GGUF CPU inference must use KV cache for O(n) generation
// (realizar#95: 11× speedup gap with vs without cache).

// ===========================================================================
// CC-001 — Output equivalence: cached and uncached produce identical tokens
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cc001Verdict { Pass, Fail }

/// Pass iff `cached_tokens == uncached_tokens` byte-exact (greedy decoding
/// is deterministic; any divergence indicates KV cache corruption).
#[must_use]
pub fn verdict_from_cached_uncached_parity(
    cached_tokens: &[u32],
    uncached_tokens: &[u32],
) -> Cc001Verdict {
    if cached_tokens.is_empty() || uncached_tokens.is_empty() { return Cc001Verdict::Fail; }
    if cached_tokens.len() != uncached_tokens.len() { return Cc001Verdict::Fail; }
    if cached_tokens == uncached_tokens { Cc001Verdict::Pass } else { Cc001Verdict::Fail }
}

// ===========================================================================
// CC-002 — Single-token forward: forward_single_with_cache processes 1 token
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cc002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_single_token_forward(tokens_processed_per_call: u64) -> Cc002Verdict {
    if tokens_processed_per_call == 1 { Cc002Verdict::Pass } else { Cc002Verdict::Fail }
}

// ===========================================================================
// CC-003 — Throughput bound: GGUF CPU tok/s ≥ 0.8 × APR CPU tok/s
// ===========================================================================

pub const AC_CC_003_MIN_RATIO: f32 = 0.8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cc003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_bound(gguf_cpu_tps: f32, apr_cpu_tps: f32) -> Cc003Verdict {
    if !gguf_cpu_tps.is_finite() || !apr_cpu_tps.is_finite() { return Cc003Verdict::Fail; }
    if gguf_cpu_tps <= 0.0 || apr_cpu_tps <= 0.0 { return Cc003Verdict::Fail; }
    let ratio = gguf_cpu_tps / apr_cpu_tps;
    if !ratio.is_finite() { return Cc003Verdict::Fail; }
    if ratio < AC_CC_003_MIN_RATIO { return Cc003Verdict::Fail; }
    Cc003Verdict::Pass
}

// ===========================================================================
// CC-004 — Greedy decoding parity: argmax(logits_cached) == argmax(logits_uncached)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cc004Verdict { Pass, Fail }

/// Pass iff argmax of cached logits matches argmax of uncached logits at
/// every step (per-step verdict over a generation trace).
#[must_use]
pub fn verdict_from_greedy_argmax_parity(
    cached_argmax_per_step: &[u32],
    uncached_argmax_per_step: &[u32],
) -> Cc004Verdict {
    if cached_argmax_per_step.is_empty() || uncached_argmax_per_step.is_empty() {
        return Cc004Verdict::Fail;
    }
    if cached_argmax_per_step.len() != uncached_argmax_per_step.len() {
        return Cc004Verdict::Fail;
    }
    if cached_argmax_per_step == uncached_argmax_per_step {
        Cc004Verdict::Pass
    } else {
        Cc004Verdict::Fail
    }
}

// ===========================================================================
// Helper: closed-form work ratio for the contract's complexity equation
// ===========================================================================

/// Without KV cache: O(n²) work = n*(n+1)/2 * L * M.
/// With KV cache: O(n) work = n * L * M.
/// Speedup = (n+1)/2.
#[must_use]
pub const fn no_cache_work(n: u64, layers: u64, matmul_cost: u64) -> u64 {
    n.saturating_mul(n.saturating_add(1))
        .saturating_div(2)
        .saturating_mul(layers)
        .saturating_mul(matmul_cost)
}

#[must_use]
pub const fn with_cache_work(n: u64, layers: u64, matmul_cost: u64) -> u64 {
    n.saturating_mul(layers).saturating_mul(matmul_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    // CC-001 (output equivalence)
    #[test] fn cc001_pass_identical() {
        let tokens = [42_u32, 100, 7, 99, 1];
        assert_eq!(verdict_from_cached_uncached_parity(&tokens, &tokens), Cc001Verdict::Pass);
    }
    #[test] fn cc001_fail_drift() {
        // The contract's stated falsifier: KV cache corrupts hidden state.
        let cached = [42_u32, 100, 7, 99, 1];
        let uncached = [42_u32, 100, 8, 99, 1]; // step 2 diverged
        assert_eq!(verdict_from_cached_uncached_parity(&cached, &uncached), Cc001Verdict::Fail);
    }
    #[test] fn cc001_fail_length() {
        let cached = [42_u32, 100];
        let uncached = [42_u32, 100, 7];
        assert_eq!(verdict_from_cached_uncached_parity(&cached, &uncached), Cc001Verdict::Fail);
    }
    #[test] fn cc001_fail_empty() {
        assert_eq!(verdict_from_cached_uncached_parity(&[], &[]), Cc001Verdict::Fail);
    }

    // CC-002 (single-token forward)
    #[test] fn cc002_pass_one_token() {
        assert_eq!(verdict_from_single_token_forward(1), Cc002Verdict::Pass);
    }
    #[test] fn cc002_fail_zero() {
        assert_eq!(verdict_from_single_token_forward(0), Cc002Verdict::Fail);
    }
    #[test] fn cc002_fail_above_one() {
        // Cache miss → full-sequence recomputation regression.
        assert_eq!(verdict_from_single_token_forward(20), Cc002Verdict::Fail);
    }

    // CC-003 (throughput bound)
    #[test] fn cc003_pass_above_threshold() {
        // GGUF 5.0 / APR 6.0 = 0.83 ≥ 0.8.
        assert_eq!(verdict_from_throughput_bound(5.0, 6.0), Cc003Verdict::Pass);
    }
    #[test] fn cc003_pass_at_threshold() {
        // 0.8 / 1.0 = 0.8.
        assert_eq!(verdict_from_throughput_bound(0.8, 1.0), Cc003Verdict::Pass);
    }
    #[test] fn cc003_fail_below_threshold() {
        // 4.0 / 6.0 = 0.67 < 0.8.
        assert_eq!(verdict_from_throughput_bound(4.0, 6.0), Cc003Verdict::Fail);
    }
    #[test] fn cc003_fail_pre_fix_baseline() {
        // The contract's baseline: GGUF 0.8 tok/s, APR 9.0 tok/s = 0.089 ratio.
        // Pre-fix state should fail this gate.
        assert_eq!(verdict_from_throughput_bound(0.8, 9.0), Cc003Verdict::Fail);
    }
    #[test] fn cc003_fail_zero_apr() {
        assert_eq!(verdict_from_throughput_bound(5.0, 0.0), Cc003Verdict::Fail);
    }
    #[test] fn cc003_fail_nan() {
        assert_eq!(verdict_from_throughput_bound(f32::NAN, 6.0), Cc003Verdict::Fail);
    }

    // CC-004 (greedy parity)
    #[test] fn cc004_pass_canonical() {
        let cached = [100_u32, 200, 300, 400];
        let uncached = [100_u32, 200, 300, 400];
        assert_eq!(verdict_from_greedy_argmax_parity(&cached, &uncached), Cc004Verdict::Pass);
    }
    #[test] fn cc004_fail_step_divergence() {
        let cached = [100_u32, 200, 300, 400];
        let uncached = [100_u32, 200, 999, 400]; // step 2 diverged
        assert_eq!(verdict_from_greedy_argmax_parity(&cached, &uncached), Cc004Verdict::Fail);
    }
    #[test] fn cc004_fail_length() {
        let cached = [100_u32];
        let uncached = [100_u32, 200];
        assert_eq!(verdict_from_greedy_argmax_parity(&cached, &uncached), Cc004Verdict::Fail);
    }

    // Work ratio helper
    #[test] fn work_speedup_at_n20() {
        // Per the contract: n=20 → speedup ≈ (20+1)/2 = 10.5x.
        let no_cache = no_cache_work(20, 28, 1_000_000);
        let with_cache = with_cache_work(20, 28, 1_000_000);
        let ratio = no_cache as f64 / with_cache as f64;
        assert!((ratio - 10.5).abs() < 0.001);
    }
    #[test] fn work_with_cache_linear() {
        // O(n): doubling n exactly doubles work.
        let w20 = with_cache_work(20, 28, 1000);
        let w40 = with_cache_work(40, 28, 1000);
        assert_eq!(w40, 2 * w20);
    }
    #[test] fn work_no_cache_quadratic() {
        // O(n²): doubling n quadruples work (approximately).
        let w20 = no_cache_work(20, 28, 1000);
        let w40 = no_cache_work(40, 28, 1000);
        // 40*41/2 / (20*21/2) = 820 / 210 = 3.905...
        assert!(w40 > 3 * w20);
        assert!(w40 < 4 * w20);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_CC_003_MIN_RATIO - 0.8).abs() < 1e-9);
    }
}
