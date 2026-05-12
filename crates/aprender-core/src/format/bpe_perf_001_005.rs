// SHIP-TWO-001 — `bpe-training-perf-v1` algorithm-level PARTIAL
// discharge for FALSIFY-BPE-TRAIN-PERF-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/bpe-training-perf-v1.yaml`.
// Spec: SHIP-TWO-001 §AC-SHIP2-002 (MODEL-2 BPE tokenizer training).

// ===========================================================================
// BPE-PERF-001 — Fast vs naive parity: same (vocab, merges)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpePerf001Verdict { Pass, Fail }

/// Pass iff vocab and merges JSON serializations are byte-identical.
#[must_use]
pub fn verdict_from_fast_vs_naive_parity(
    naive_vocab_json: &str,
    fast_vocab_json: &str,
    naive_merges_json: &str,
    fast_merges_json: &str,
) -> BpePerf001Verdict {
    if naive_vocab_json.is_empty() || fast_vocab_json.is_empty() { return BpePerf001Verdict::Fail; }
    if naive_vocab_json != fast_vocab_json { return BpePerf001Verdict::Fail; }
    if naive_merges_json != fast_merges_json { return BpePerf001Verdict::Fail; }
    BpePerf001Verdict::Pass
}

// ===========================================================================
// BPE-PERF-002 — Cross-run determinism: byte-identical (vocab, merges)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpePerf002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_cross_run_determinism(run1_json: &[u8], run2_json: &[u8]) -> BpePerf002Verdict {
    if run1_json.is_empty() || run2_json.is_empty() { return BpePerf002Verdict::Fail; }
    if run1_json == run2_json { BpePerf002Verdict::Pass } else { BpePerf002Verdict::Fail }
}

// ===========================================================================
// BPE-PERF-003 — Wall-clock <= 60 min on MODEL-2 workload (CPU-only)
// ===========================================================================

pub const AC_BPE_PERF_003_MAX_WALL_SECONDS: u64 = 3600;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpePerf003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_wall_clock_gate(wall_clock_seconds: u64) -> BpePerf003Verdict {
    if wall_clock_seconds == 0 { return BpePerf003Verdict::Fail; }
    if wall_clock_seconds <= AC_BPE_PERF_003_MAX_WALL_SECONDS {
        BpePerf003Verdict::Pass
    } else {
        BpePerf003Verdict::Fail
    }
}

// ===========================================================================
// BPE-PERF-004 — Progress observable: ≥ vocab_size/1000 [bpe] log lines
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpePerf004Verdict { Pass, Fail }

/// Pass iff `bpe_log_line_count >= vocab_size / 1000` AND vocab_size > 0.
#[must_use]
pub const fn verdict_from_progress_observable(
    bpe_log_line_count: u64,
    vocab_size: u64,
) -> BpePerf004Verdict {
    if vocab_size == 0 { return BpePerf004Verdict::Fail; }
    let required = vocab_size / 1000;
    if required == 0 {
        // Vocab too small to require >=1 progress line — pass vacuously.
        return BpePerf004Verdict::Pass;
    }
    if bpe_log_line_count >= required { BpePerf004Verdict::Pass } else { BpePerf004Verdict::Fail }
}

// ===========================================================================
// BPE-PERF-005 — 1.5× parity-replacement rule: naive_secs / fast_secs >= 1.5
// ===========================================================================

pub const AC_BPE_PERF_005_MIN_SPEEDUP: f64 = 1.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpePerf005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_speedup_ratio(naive_secs: f64, fast_secs: f64) -> BpePerf005Verdict {
    if !naive_secs.is_finite() || !fast_secs.is_finite() { return BpePerf005Verdict::Fail; }
    if naive_secs <= 0.0 || fast_secs <= 0.0 { return BpePerf005Verdict::Fail; }
    let ratio = naive_secs / fast_secs;
    if ratio >= AC_BPE_PERF_005_MIN_SPEEDUP { BpePerf005Verdict::Pass } else { BpePerf005Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // BPE-PERF-001
    #[test] fn bpe001_pass_match() {
        assert_eq!(
            verdict_from_fast_vs_naive_parity(
                "{\"a\": 0}", "{\"a\": 0}",
                "[\"a b\"]", "[\"a b\"]",
            ),
            BpePerf001Verdict::Pass
        );
    }

    #[test] fn bpe001_fail_vocab_drift() {
        assert_eq!(
            verdict_from_fast_vs_naive_parity(
                "{\"a\": 0}", "{\"a\": 1}",
                "[]", "[]",
            ),
            BpePerf001Verdict::Fail
        );
    }

    #[test] fn bpe001_fail_merges_drift() {
        assert_eq!(
            verdict_from_fast_vs_naive_parity(
                "{}", "{}",
                "[\"a b\"]", "[\"b a\"]",
            ),
            BpePerf001Verdict::Fail
        );
    }

    #[test] fn bpe001_fail_empty() {
        assert_eq!(
            verdict_from_fast_vs_naive_parity("", "{}", "[]", "[]"),
            BpePerf001Verdict::Fail
        );
    }

    // BPE-PERF-002
    #[test] fn bpe002_pass_byte_identical() {
        assert_eq!(verdict_from_cross_run_determinism(b"abc", b"abc"), BpePerf002Verdict::Pass);
    }

    #[test] fn bpe002_fail_drift() {
        assert_eq!(verdict_from_cross_run_determinism(b"abc", b"abd"), BpePerf002Verdict::Fail);
    }

    #[test] fn bpe002_fail_empty() {
        assert_eq!(verdict_from_cross_run_determinism(b"", b"abc"), BpePerf002Verdict::Fail);
    }

    // BPE-PERF-003
    #[test] fn bpe003_pass_under_limit() {
        assert_eq!(verdict_from_wall_clock_gate(1800), BpePerf003Verdict::Pass);
    }

    #[test] fn bpe003_pass_at_limit() {
        assert_eq!(verdict_from_wall_clock_gate(3600), BpePerf003Verdict::Pass);
    }

    #[test] fn bpe003_fail_above_limit() {
        assert_eq!(verdict_from_wall_clock_gate(3601), BpePerf003Verdict::Fail);
    }

    #[test] fn bpe003_fail_zero() {
        assert_eq!(verdict_from_wall_clock_gate(0), BpePerf003Verdict::Fail);
    }

    // BPE-PERF-004
    #[test] fn bpe004_pass_canonical() {
        // 50257 / 1000 = 50 progress lines required.
        assert_eq!(verdict_from_progress_observable(60, 50_257), BpePerf004Verdict::Pass);
    }

    #[test] fn bpe004_pass_at_minimum() {
        assert_eq!(verdict_from_progress_observable(50, 50_000), BpePerf004Verdict::Pass);
    }

    #[test] fn bpe004_fail_silent() {
        assert_eq!(verdict_from_progress_observable(0, 50_257), BpePerf004Verdict::Fail);
    }

    #[test] fn bpe004_fail_too_few() {
        assert_eq!(verdict_from_progress_observable(10, 50_257), BpePerf004Verdict::Fail);
    }

    #[test] fn bpe004_pass_small_vocab_vacuous() {
        // 500 / 1000 = 0 required, vacuous pass.
        assert_eq!(verdict_from_progress_observable(0, 500), BpePerf004Verdict::Pass);
    }

    #[test] fn bpe004_fail_zero_vocab() {
        assert_eq!(verdict_from_progress_observable(100, 0), BpePerf004Verdict::Fail);
    }

    // BPE-PERF-005
    #[test] fn bpe005_pass_2x() {
        // naive 60s, fast 30s → 2× speedup.
        assert_eq!(verdict_from_speedup_ratio(60.0, 30.0), BpePerf005Verdict::Pass);
    }

    #[test] fn bpe005_pass_at_1_5x() {
        assert_eq!(verdict_from_speedup_ratio(45.0, 30.0), BpePerf005Verdict::Pass);
    }

    #[test] fn bpe005_fail_1_4x() {
        assert_eq!(verdict_from_speedup_ratio(42.0, 30.0), BpePerf005Verdict::Fail);
    }

    #[test] fn bpe005_fail_slower() {
        assert_eq!(verdict_from_speedup_ratio(30.0, 60.0), BpePerf005Verdict::Fail);
    }

    #[test] fn bpe005_fail_zero_fast() {
        assert_eq!(verdict_from_speedup_ratio(60.0, 0.0), BpePerf005Verdict::Fail);
    }

    #[test] fn bpe005_fail_nan() {
        assert_eq!(verdict_from_speedup_ratio(f64::NAN, 30.0), BpePerf005Verdict::Fail);
    }

    // Provenance pins
    #[test] fn provenance_constants() {
        assert_eq!(AC_BPE_PERF_003_MAX_WALL_SECONDS, 3600);
        assert!((AC_BPE_PERF_005_MIN_SPEEDUP - 1.5).abs() < 1e-12);
    }
}
