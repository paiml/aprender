// SHIP-TWO-001 MODEL-2 — `apr-tokenize-parallel-bpe-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-TOK-PAR-002.
//
// Contract: `contracts/apr-tokenize-parallel-bpe-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2). First binding for the parallel-BPE
// contract surface.
//
// ## What FALSIFY-APR-TOK-PAR-002 says
//
//   rule: speedup ≥ 0.8N for 4-way
//   prediction: 4-way parallel on 4-core machine completes in
//               ≤ 1.25× single-threaded / 4 (i.e., parallel_seconds
//               ≤ serial_seconds / (0.8 * num_workers)).
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule:
//
//   speedup       = serial_seconds / parallel_seconds
//   efficiency    = speedup / num_workers
//   Pass iff efficiency ≥ 0.8
//
// Equivalently (avoiding floating-point division):
//
//   Pass iff serial_seconds * 100 ≥ parallel_seconds * num_workers * 80
//
// The 80% efficiency floor (0.8N speedup target) is bound as
// `AC_TOK_PAR_002_MIN_EFFICIENCY_PERCENT` so a future drift to 50% (would
// silently weaken the gate) or to 95% (would over-tighten and reject
// reasonable BPE parallelism) trips the provenance test.

/// Minimum parallel efficiency percent.
///
/// Per contract `FALSIFY-APR-TOK-PAR-002`: 80% means N-way parallelism
/// achieves at least 0.8N speedup over single-threaded baseline. BPE
/// encoding is CPU-bound with no shared mutable state across rows, so
/// 80% is achievable; deviation downward implies a synchronization or
/// thread-pool overhead bug that should be caught.
pub const AC_TOK_PAR_002_MIN_EFFICIENCY_PERCENT: u64 = 80;

/// Binary verdict for `FALSIFY-APR-TOK-PAR-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokPar002Verdict {
    /// Parallel efficiency ≥ 80% — `parallel_seconds * num_workers * 80
    /// ≤ serial_seconds * 100`.
    Pass,
    /// One or more of:
    /// - `serial_seconds == 0` (would imply zero work — caller error).
    /// - `parallel_seconds == 0` (caller error).
    /// - `num_workers == 0` (caller error).
    /// - `num_workers == 1` (cannot prove parallel speedup with no parallelism).
    /// - Multiplication overflow at the comparison step (absurd inputs).
    /// - Efficiency below 80%.
    Fail,
}

/// Pure verdict function for FALSIFY-APR-TOK-PAR-002.
///
/// Inputs:
/// - `serial_seconds`: wall-clock seconds for single-threaded encode.
/// - `parallel_seconds`: wall-clock seconds for parallel encode (with `num_workers`).
/// - `num_workers`: number of parallel workers (must be ≥ 2).
///
/// Pass iff:
/// 1. All three inputs are non-zero,
/// 2. `num_workers >= 2`,
/// 3. `serial_seconds * 100 >= parallel_seconds * num_workers * 80`
///    (computed via `checked_mul` to prevent overflow).
///
/// Otherwise `Fail`. Time inputs are integer microseconds (or any
/// consistent unit) — the verdict only cares about ratio.
///
/// # Examples
///
/// 4-way at 100% efficiency — `Pass`:
/// ```
/// use aprender::format::tok_par_002::{
///     verdict_from_speedup_observation, TokPar002Verdict,
/// };
/// // serial=4000ms, parallel=1000ms, 4 workers → speedup=4, efficiency=100%
/// let v = verdict_from_speedup_observation(4000, 1000, 4);
/// assert_eq!(v, TokPar002Verdict::Pass);
/// ```
///
/// 4-way at 50% efficiency (below 80% floor) — `Fail`:
/// ```
/// use aprender::format::tok_par_002::{
///     verdict_from_speedup_observation, TokPar002Verdict,
/// };
/// // serial=4000ms, parallel=2000ms, 4 workers → speedup=2, efficiency=50%
/// let v = verdict_from_speedup_observation(4000, 2000, 4);
/// assert_eq!(v, TokPar002Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_speedup_observation(
    serial_seconds: u64,
    parallel_seconds: u64,
    num_workers: u64,
) -> TokPar002Verdict {
    if serial_seconds == 0 || parallel_seconds == 0 || num_workers == 0 {
        return TokPar002Verdict::Fail;
    }
    if num_workers < 2 {
        return TokPar002Verdict::Fail;
    }
    // Pass iff serial * 100 >= parallel * num_workers * 80
    let lhs = match serial_seconds.checked_mul(100) {
        Some(v) => v,
        None => return TokPar002Verdict::Fail,
    };
    let rhs = match parallel_seconds.checked_mul(num_workers) {
        Some(v) => v,
        None => return TokPar002Verdict::Fail,
    };
    let rhs = match rhs.checked_mul(AC_TOK_PAR_002_MIN_EFFICIENCY_PERCENT) {
        Some(v) => v,
        None => return TokPar002Verdict::Fail,
    };
    if lhs >= rhs {
        TokPar002Verdict::Pass
    } else {
        TokPar002Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — efficiency floor matches contract.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_efficiency_is_eighty_percent() {
        assert_eq!(AC_TOK_PAR_002_MIN_EFFICIENCY_PERCENT, 80);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — efficiency ≥ 80%.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_perfect_4_way_speedup() {
        // serial=4000, parallel=1000, 4 workers → speedup=4, efficiency=100%.
        let v = verdict_from_speedup_observation(4000, 1000, 4);
        assert_eq!(v, TokPar002Verdict::Pass);
    }

    #[test]
    fn pass_exactly_at_80_percent_floor() {
        // serial=4000, parallel=1250, 4 workers → speedup=3.2, efficiency=80%.
        // 4000*100=400_000; 1250*4*80=400_000. lhs == rhs → Pass (inclusive).
        let v = verdict_from_speedup_observation(4000, 1250, 4);
        assert_eq!(v, TokPar002Verdict::Pass, "exact 80% must Pass (inclusive)");
    }

    #[test]
    fn pass_2_way_at_high_efficiency() {
        // serial=2000, parallel=1100, 2 workers → speedup=1.818, efficiency=91%.
        let v = verdict_from_speedup_observation(2000, 1100, 2);
        assert_eq!(v, TokPar002Verdict::Pass);
    }

    #[test]
    fn pass_8_way_at_perfect_efficiency() {
        // serial=8000, parallel=1000, 8 workers → speedup=8, efficiency=100%.
        let v = verdict_from_speedup_observation(8000, 1000, 8);
        assert_eq!(v, TokPar002Verdict::Pass);
    }

    #[test]
    fn pass_super_linear_speedup() {
        // serial=4000, parallel=900, 4 workers → speedup=4.44, efficiency=111%.
        // (Possible via cache effects on real hardware.)
        let v = verdict_from_speedup_observation(4000, 900, 4);
        assert_eq!(v, TokPar002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — efficiency < 80%.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_4_way_at_50_percent() {
        // serial=4000, parallel=2000, 4 workers → speedup=2, efficiency=50%.
        let v = verdict_from_speedup_observation(4000, 2000, 4);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    #[test]
    fn fail_4_way_at_75_percent_just_below_floor() {
        // 4000*100=400_000; parallel*4*80 must exceed 400_000 → parallel > 1250.
        // Use parallel=1251 → 1251*4*80 = 400_320 > 400_000 → Fail.
        let v = verdict_from_speedup_observation(4000, 1251, 4);
        assert_eq!(v, TokPar002Verdict::Fail, "75% efficiency must Fail");
    }

    #[test]
    fn fail_no_speedup_parallel_equals_serial() {
        // serial=parallel → speedup=1 → efficiency=1/N=25% for 4 workers.
        let v = verdict_from_speedup_observation(4000, 4000, 4);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    #[test]
    fn fail_parallel_slower_than_serial() {
        // Worst case: parallel is slower (sync overhead exceeds work).
        let v = verdict_from_speedup_observation(4000, 5000, 4);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — caller errors (zero inputs, num_workers < 2).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_serial() {
        let v = verdict_from_speedup_observation(0, 1000, 4);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    #[test]
    fn fail_zero_parallel() {
        let v = verdict_from_speedup_observation(4000, 0, 4);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    #[test]
    fn fail_zero_workers() {
        let v = verdict_from_speedup_observation(4000, 1000, 0);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    #[test]
    fn fail_one_worker_no_parallelism() {
        // num_workers = 1 cannot prove parallel speedup; conservative Fail.
        let v = verdict_from_speedup_observation(4000, 4000, 1);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Overflow protection — checked_mul on all 3 multiplications.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_serial_times_100_overflow() {
        // serial * 100 overflows u64 at serial > u64::MAX/100 ≈ 1.84e17.
        let huge = u64::MAX / 50; // > u64::MAX/100 → overflow
        let v = verdict_from_speedup_observation(huge, 1, 4);
        assert_eq!(
            v,
            TokPar002Verdict::Fail,
            "serial * 100 overflow must Fail (not silently wrap)"
        );
    }

    #[test]
    fn fail_parallel_times_workers_overflow() {
        // parallel * num_workers overflows.
        let near_max: u64 = u64::MAX / 2;
        let v = verdict_from_speedup_observation(1000, near_max, 4);
        assert_eq!(v, TokPar002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Efficiency sweep at fixed 4-way.
    // -------------------------------------------------------------------------
    #[test]
    fn efficiency_sweep_at_4_way() {
        let serial = 4000_u64;
        let workers = 4_u64;
        // Efficiency E means parallel = serial / (E * workers / 100).
        // E=100% → parallel=1000; E=90% → parallel=1111; E=80% → parallel=1250;
        // E=70% → parallel=1428; E=50% → parallel=2000.
        let probes: Vec<(u64, TokPar002Verdict)> = vec![
            (1000, TokPar002Verdict::Pass), // 100%
            (1100, TokPar002Verdict::Pass), // ~91%
            (1200, TokPar002Verdict::Pass), // ~83%
            (1250, TokPar002Verdict::Pass), // 80% (boundary, inclusive)
            (1251, TokPar002Verdict::Fail), // ~80% just below
            (1500, TokPar002Verdict::Fail), // ~67%
            (2000, TokPar002Verdict::Fail), // 50%
            (4000, TokPar002Verdict::Fail), // 25% (no speedup)
        ];
        for (parallel, expected) in probes {
            let v = verdict_from_speedup_observation(serial, parallel, workers);
            assert_eq!(
                v, expected,
                "serial={serial} parallel={parallel} workers={workers} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Worker-count sweep at fixed efficiency profile.
    // -------------------------------------------------------------------------
    #[test]
    fn worker_count_sweep_at_perfect_efficiency() {
        // For each N workers, perfect efficiency means parallel = serial / N.
        // serial=10_000, worker={2,4,8,16,32}.
        for n in [2_u64, 4, 8, 16, 32] {
            let parallel = 10_000 / n;
            let v = verdict_from_speedup_observation(10_000, parallel, n);
            assert_eq!(
                v,
                TokPar002Verdict::Pass,
                "perfect efficiency at {n} workers must Pass"
            );
        }
    }

    #[test]
    fn worker_count_sweep_at_50_percent_efficiency_all_fail() {
        // For each N workers, 50% efficiency means parallel = serial / (0.5 * N) = 2*serial/N.
        for n in [2_u64, 4, 8, 16] {
            let parallel = 2 * 10_000 / n;
            let v = verdict_from_speedup_observation(10_000, parallel, n);
            assert_eq!(
                v,
                TokPar002Verdict::Fail,
                "50% efficiency at {n} workers must Fail"
            );
        }
    }
}
