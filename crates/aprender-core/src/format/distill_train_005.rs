// SHIP-TWO-001 §35 — `apr-cli-distill-train-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-DISTILL-TRAIN-005.
//
// Contract: `contracts/apr-cli-distill-train-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §35.
//
// ## What FALSIFY-APR-DISTILL-TRAIN-005 says
//
//   rule: precompute is byte-deterministic
//   prediction: Two runs of `apr distill --stage precompute` with same
//               inputs produce byte-identical `teacher_logits/` output.
//   test:       diff -r run1/teacher_logits run2/teacher_logits → exit 0
//   if_fails:   non-determinism in teacher forward pass — likely a
//               kernel-launch ordering or atomic-add issue.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// The decision rule — "every per-file SHA-256 in run A matches the
// corresponding file's SHA-256 in run B, both directories have the same
// non-empty file set, and every hash is canonical lowercase 64-char hex"
// — is pinned. Composes with [`super::ship_010::verdict_from_sha256_match`]
// (already-merged SHA-256 byte-identity primitive) so the format
// validation rules (length, lowercase, hex-only) are the SAME ones
// enforced at MODEL-1 publish time.
//
// Future implementations cannot silently weaken determinism by:
// - dropping a file from one of the two runs (length mismatch → Fail)
// - emitting different hashes for the same logical file (Fail)
// - emitting non-canonical hex (uppercase, hyphens) (delegated to ship_010 → Fail)

use super::ship_010::{verdict_from_sha256_match, Ship010Verdict};

/// Binary verdict for `FALSIFY-APR-DISTILL-TRAIN-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillTrain005Verdict {
    /// Both runs produced the same set of files in the same order, all
    /// non-empty, and every per-file SHA-256 matches byte-identically.
    /// Precompute is deterministic.
    Pass,
    /// One or more of:
    /// - Either run is empty (no files emitted — caller error).
    /// - Different number of files between run A and run B.
    /// - At least one per-file SHA-256 mismatch.
    /// - At least one hash is malformed (length ≠ 64, uppercase, non-hex).
    Fail,
}

/// Pure verdict function for FALSIFY-APR-DISTILL-TRAIN-005.
///
/// Inputs: per-file SHA-256 hex strings from each run, in the same
/// canonical order (e.g. sorted by relative path under
/// `<output_dir>/teacher_logits/`).
///
/// Composes with [`super::ship_010::verdict_from_sha256_match`] so the
/// SHA-256 format validation rules (64 hex chars, lowercase) are the same
/// ones enforced at MODEL-1 publish time.
///
/// # Examples
///
/// Identical runs — `Pass`:
/// ```
/// use aprender::format::distill_train_005::{
///     verdict_from_paired_per_file_hashes, DistillTrain005Verdict,
/// };
/// let run_a = vec![
///     "0".repeat(64),
///     "a".repeat(64),
///     "b".repeat(64),
/// ];
/// let run_b = run_a.clone();
/// assert_eq!(
///     verdict_from_paired_per_file_hashes(&run_a, &run_b),
///     DistillTrain005Verdict::Pass,
/// );
/// ```
///
/// Mismatched hash — `Fail`:
/// ```
/// use aprender::format::distill_train_005::{
///     verdict_from_paired_per_file_hashes, DistillTrain005Verdict,
/// };
/// let run_a = vec!["0".repeat(64)];
/// let run_b = vec!["1".repeat(64)];
/// assert_eq!(
///     verdict_from_paired_per_file_hashes(&run_a, &run_b),
///     DistillTrain005Verdict::Fail,
/// );
/// ```
#[must_use]
pub fn verdict_from_paired_per_file_hashes(
    run_a: &[String],
    run_b: &[String],
) -> DistillTrain005Verdict {
    if run_a.is_empty() || run_b.is_empty() {
        return DistillTrain005Verdict::Fail;
    }
    if run_a.len() != run_b.len() {
        return DistillTrain005Verdict::Fail;
    }
    for (a, b) in run_a.iter().zip(run_b.iter()) {
        match verdict_from_sha256_match(a, b) {
            Ship010Verdict::Pass => continue,
            Ship010Verdict::Fail => return DistillTrain005Verdict::Fail,
        }
    }
    DistillTrain005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(c: char) -> String {
        std::iter::repeat(c).take(64).collect()
    }

    fn sample_run() -> Vec<String> {
        vec![h('0'), h('a'), h('b'), h('c'), h('d')]
    }

    // -------------------------------------------------------------------------
    // Section 1: Pass band — identical runs.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_identical_runs() {
        let run = sample_run();
        assert_eq!(
            verdict_from_paired_per_file_hashes(&run, &run),
            DistillTrain005Verdict::Pass
        );
    }

    #[test]
    fn pass_single_file_match() {
        let run = vec![h('a')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&run, &run),
            DistillTrain005Verdict::Pass
        );
    }

    #[test]
    fn pass_typical_size_339_files() {
        // Mirrors a realistic per-shard cache: 339 entries (one per
        // Qwen2.5-Coder-7B tensor's logits row, illustratively).
        let run: Vec<String> = (0..339)
            .map(|i| {
                let base = format!("{:064x}", i);
                base
            })
            .collect();
        assert_eq!(
            verdict_from_paired_per_file_hashes(&run, &run),
            DistillTrain005Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — single-file hash mismatch.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_single_file_mismatch() {
        let a = vec![h('a')];
        let b = vec![h('b')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    #[test]
    fn fail_mismatch_at_each_position() {
        for bad_idx in [0_usize, 2, 4] {
            let mut a = sample_run();
            let b = sample_run();
            a[bad_idx] = h('f');
            assert_eq!(
                verdict_from_paired_per_file_hashes(&a, &b),
                DistillTrain005Verdict::Fail,
                "mismatch at index {bad_idx} must Fail"
            );
        }
    }

    #[test]
    fn fail_one_byte_difference() {
        // sha256sum changing one bit anywhere in the 256-bit output
        // changes at least one nybble — pinned via single-char swap here.
        let a = vec!["0".repeat(63) + "0"];
        let b = vec!["0".repeat(63) + "1"];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Length drift — both runs must have the same file count.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_run_a_longer() {
        let a = vec![h('a'), h('b'), h('c')];
        let b = vec![h('a'), h('b')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    #[test]
    fn fail_run_b_longer() {
        let a = vec![h('a'), h('b')];
        let b = vec![h('a'), h('b'), h('c')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    #[test]
    fn fail_off_by_one_at_typical_size() {
        let a: Vec<String> = (0..339).map(|i| format!("{:064x}", i)).collect();
        let b: Vec<String> = (0..338).map(|i| format!("{:064x}", i)).collect();
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Empty input — caller error → conservative Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_both_empty() {
        let empty: Vec<String> = vec![];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&empty, &empty),
            DistillTrain005Verdict::Fail,
            "empty cache implies precompute did nothing — caller error"
        );
    }

    #[test]
    fn fail_run_a_empty_but_b_not() {
        let a: Vec<String> = vec![];
        let b = vec![h('a')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    #[test]
    fn fail_run_b_empty_but_a_not() {
        let a = vec![h('a')];
        let b: Vec<String> = vec![];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Format violation — delegated to ship_010 (composition test).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_uppercase_hex_via_ship_010() {
        // ship_010 enforces canonical lowercase only; uppercase Fails
        // even when the strings are byte-equal — pinned via composition.
        let a = vec!["A".repeat(64)];
        let b = vec!["A".repeat(64)];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail,
            "uppercase hex Fails even when byte-equal (ship_010 rule)"
        );
    }

    #[test]
    fn fail_too_short_hash() {
        let a = vec!["0".repeat(63)];
        let b = vec!["0".repeat(63)];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    #[test]
    fn fail_too_long_hash() {
        let a = vec!["0".repeat(65)];
        let b = vec!["0".repeat(65)];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    #[test]
    fn fail_non_hex_character() {
        let mut bad = "0".repeat(63);
        bad.push('z');
        let a = vec![bad.clone()];
        let b = vec![bad];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail,
            "non-hex character Fails (ship_010 rule)"
        );
    }

    #[test]
    fn fail_empty_string_in_list() {
        let a = vec![String::new()];
        let b = vec![String::new()];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Asymmetry probe — order matters.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_same_set_different_order() {
        // Per contract, both lists are expected in the same canonical
        // order (e.g., sorted relative paths). A reordered second run
        // FAILS — the future CLI is responsible for sorting before
        // comparison; this gate doesn't paper over that bug.
        let a = vec![h('a'), h('b'), h('c')];
        let b = vec![h('c'), h('b'), h('a')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail,
            "same hash set in different order must Fail (caller sorts)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Composition with ship_010 — sanity check that we delegate.
    // -------------------------------------------------------------------------
    #[test]
    fn ship_010_passes_imply_pass() {
        // Pure composition test: if ship_010 says Pass for every pair,
        // and the lengths match, this verdict says Pass.
        let a = vec![h('0'), h('1'), h('2')];
        let b = a.clone();
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Pass
        );
        // And confirm the underlying ship_010 verdict matches.
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(verdict_from_sha256_match(x, y), Ship010Verdict::Pass);
        }
    }

    #[test]
    fn ship_010_fails_imply_fail() {
        let a = vec![h('0')];
        let b = vec![h('1')];
        assert_eq!(
            verdict_from_paired_per_file_hashes(&a, &b),
            DistillTrain005Verdict::Fail
        );
        assert_eq!(
            verdict_from_sha256_match(&a[0], &b[0]),
            Ship010Verdict::Fail
        );
    }
}
