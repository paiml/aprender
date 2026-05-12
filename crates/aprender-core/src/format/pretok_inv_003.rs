// SHIP-TWO-001 MODEL-2 — `pretokenize-bin-v1` (C-PRETOK-BIN) algorithm-level
// PARTIAL discharge for INV-PRETOK-003.
//
// Contract: `contracts/pretokenize-bin-v1.yaml` v1.0.0 PROPOSED.
//
// ## What INV-PRETOK-003 says
//
//   description: total_tokens declared in manifest equals the sum over
//                shards of (file_size_bytes / 4). No drift between
//                declared and actual.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a declared `total_tokens` count and a list of
// per-shard byte sizes, Pass iff:
//
//   declared_total_tokens == sum(per_shard_bytes / 4)
//
// AND every per-shard size is u32-aligned (composes with INV-PRETOK-002,
// pretok_inv_002.rs in this same module). Sum-overflow protection via
// `checked_add` — a corrupted shard with size near u64::MAX cannot crash
// the verdict via wraparound.
//
// Future implementations cannot:
// - Drift the manifest's declared total by ±1 token (caught by exact ==).
// - Skip emitting one shard's bytes from the sum (caught by exact ==).
// - Allow a u64 wrap-around on the sum (caught by checked_add → Fail).

use super::pretok_inv_002::AC_PRETOK_INV_002_U32_BYTE_SIZE;

/// Binary verdict for `INV-PRETOK-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokInv003Verdict {
    /// Declared `total_tokens` exactly matches the sum of
    /// `(per_shard_byte_sizes[i] / 4)` across all shards. The shard list
    /// is non-empty and every size is u32-aligned.
    Pass,
    /// One or more of:
    /// - Empty shard list (caller error).
    /// - Some shard size is not u32-aligned (composes with INV-PRETOK-002).
    /// - Sum overflow (would imply absurd corpus size — conservative `Fail`).
    /// - Declared `total_tokens` ≠ summed-actual.
    Fail,
}

/// Pure verdict function for `INV-PRETOK-003`.
///
/// Inputs:
/// - `declared_total_tokens`: the manifest's claimed total token count.
/// - `per_shard_byte_sizes`: byte size of each emitted shard file.
///
/// Pass iff:
/// 1. The slice is non-empty.
/// 2. Every entry is `% 4 == 0` (u32-aligned, mirrors INV-PRETOK-002).
/// 3. Every entry is `>= 4` (at least one token).
/// 4. `sum(size / 4)` does not overflow u64.
/// 5. `sum(size / 4) == declared_total_tokens`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Manifest matches actual — `Pass`:
/// ```
/// use aprender::format::pretok_inv_003::{
///     verdict_from_manifest_vs_shards, PretokInv003Verdict,
/// };
/// let sizes = vec![1024_u64, 2048, 4096]; // 256 + 512 + 1024 = 1792 tokens
/// assert_eq!(
///     verdict_from_manifest_vs_shards(1792, &sizes),
///     PretokInv003Verdict::Pass,
/// );
/// ```
///
/// Manifest off by one — `Fail`:
/// ```
/// use aprender::format::pretok_inv_003::{
///     verdict_from_manifest_vs_shards, PretokInv003Verdict,
/// };
/// let sizes = vec![1024_u64, 2048, 4096]; // 1792 tokens, declared 1793
/// assert_eq!(
///     verdict_from_manifest_vs_shards(1793, &sizes),
///     PretokInv003Verdict::Fail,
/// );
/// ```
#[must_use]
pub fn verdict_from_manifest_vs_shards(
    declared_total_tokens: u64,
    per_shard_byte_sizes: &[u64],
) -> PretokInv003Verdict {
    if per_shard_byte_sizes.is_empty() {
        return PretokInv003Verdict::Fail;
    }
    let mut summed: u64 = 0;
    for &size in per_shard_byte_sizes {
        if size < AC_PRETOK_INV_002_U32_BYTE_SIZE {
            return PretokInv003Verdict::Fail;
        }
        if size % AC_PRETOK_INV_002_U32_BYTE_SIZE != 0 {
            return PretokInv003Verdict::Fail;
        }
        let tokens_in_shard = size / AC_PRETOK_INV_002_U32_BYTE_SIZE;
        match summed.checked_add(tokens_in_shard) {
            Some(new_sum) => summed = new_sum,
            None => return PretokInv003Verdict::Fail,
        }
    }
    if summed == declared_total_tokens {
        PretokInv003Verdict::Pass
    } else {
        PretokInv003Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — declared matches actual.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_simple_three_shards() {
        let sizes = vec![1024_u64, 2048, 4096];
        // 1024/4 + 2048/4 + 4096/4 = 256 + 512 + 1024 = 1792
        assert_eq!(
            verdict_from_manifest_vs_shards(1792, &sizes),
            PretokInv003Verdict::Pass
        );
    }

    #[test]
    fn pass_single_shard_one_token() {
        let sizes = vec![4_u64];
        assert_eq!(
            verdict_from_manifest_vs_shards(1, &sizes),
            PretokInv003Verdict::Pass
        );
    }

    #[test]
    fn pass_realistic_565m_token_corpus() {
        // Per spec: MODEL-2 trained on 565.6M token Python+permissive corpus.
        // Approximate by: 100 shards × 22_624_000 bytes (5.656M tokens each).
        let sizes = vec![22_624_000_u64; 100];
        let total = 100_u64 * (22_624_000 / 4);
        assert_eq!(
            verdict_from_manifest_vs_shards(total, &sizes),
            PretokInv003Verdict::Pass
        );
        assert_eq!(total, 565_600_000);
    }

    #[test]
    fn pass_uniform_shards() {
        let sizes = vec![400_u64; 50]; // 100 tokens × 50 shards = 5000
        assert_eq!(
            verdict_from_manifest_vs_shards(5000, &sizes),
            PretokInv003Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — declared ≠ actual (drift).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_off_by_one_high() {
        let sizes = vec![1024_u64, 2048, 4096];
        assert_eq!(
            verdict_from_manifest_vs_shards(1793, &sizes), // actual = 1792
            PretokInv003Verdict::Fail
        );
    }

    #[test]
    fn fail_off_by_one_low() {
        let sizes = vec![1024_u64, 2048, 4096];
        assert_eq!(
            verdict_from_manifest_vs_shards(1791, &sizes), // actual = 1792
            PretokInv003Verdict::Fail
        );
    }

    #[test]
    fn fail_declared_zero_with_real_data() {
        let sizes = vec![1024_u64];
        assert_eq!(
            verdict_from_manifest_vs_shards(0, &sizes),
            PretokInv003Verdict::Fail
        );
    }

    #[test]
    fn fail_declared_double_actual() {
        let sizes = vec![1024_u64];
        assert_eq!(
            verdict_from_manifest_vs_shards(512, &sizes), // actual = 256
            PretokInv003Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — alignment (composes with INV-PRETOK-002).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_misaligned_shard_short_circuit() {
        // 1025 % 4 != 0 → Fail before manifest comparison.
        let sizes = vec![1025_u64];
        assert_eq!(
            verdict_from_manifest_vs_shards(256, &sizes),
            PretokInv003Verdict::Fail,
            "alignment failure short-circuits manifest check"
        );
    }

    #[test]
    fn fail_zero_byte_shard() {
        let sizes = vec![0_u64];
        assert_eq!(
            verdict_from_manifest_vs_shards(0, &sizes),
            PretokInv003Verdict::Fail
        );
    }

    #[test]
    fn fail_size_below_one_token() {
        for size in [1_u64, 2, 3] {
            let sizes = vec![size];
            assert_eq!(
                verdict_from_manifest_vs_shards(0, &sizes),
                PretokInv003Verdict::Fail
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 4: Empty input — caller error → Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_shard_list_with_zero_declared() {
        let sizes: Vec<u64> = vec![];
        assert_eq!(
            verdict_from_manifest_vs_shards(0, &sizes),
            PretokInv003Verdict::Fail,
            "even matching zeros must Fail on empty list"
        );
    }

    #[test]
    fn fail_empty_shard_list_with_nonzero_declared() {
        let sizes: Vec<u64> = vec![];
        assert_eq!(
            verdict_from_manifest_vs_shards(100, &sizes),
            PretokInv003Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Overflow protection — checked_add cannot wrap silently.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_overflow_via_two_max_shards() {
        // u64::MAX is odd → first must catch via alignment.
        // To force a sum overflow, use the largest u32-aligned u64.
        let max_aligned: u64 = u64::MAX & !0b11; // = u64::MAX - 3
        let sizes = vec![max_aligned, max_aligned];
        // sum(max_aligned/4 + max_aligned/4) overflows u64.
        // Expect Fail (checked_add returns None).
        assert_eq!(
            verdict_from_manifest_vs_shards(0, &sizes),
            PretokInv003Verdict::Fail
        );
    }

    #[test]
    fn fail_one_max_aligned_within_range() {
        // A single max_aligned shard does NOT overflow on its own
        // (max_aligned/4 fits in u64). The verdict's manifest check
        // will Fail unless declared exactly equals that max-aligned/4.
        let max_aligned: u64 = u64::MAX & !0b11;
        let sizes = vec![max_aligned];
        let expected_tokens = max_aligned / 4;
        assert_eq!(
            verdict_from_manifest_vs_shards(expected_tokens, &sizes),
            PretokInv003Verdict::Pass,
            "single max-aligned shard with correct manifest must Pass"
        );
        assert_eq!(
            verdict_from_manifest_vs_shards(expected_tokens - 1, &sizes),
            PretokInv003Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Single-shard mutation — drift in any one shard fails.
    // -------------------------------------------------------------------------
    #[test]
    fn single_shard_mutation_at_each_index_fails() {
        // Baseline: 5 shards × 4 bytes = 5 tokens declared.
        for bad_idx in [0_usize, 2, 4] {
            let mut sizes = vec![4_u64; 5];
            sizes[bad_idx] = 8; // adds one token to the actual sum
            assert_eq!(
                verdict_from_manifest_vs_shards(5, &sizes), // declared still 5
                PretokInv003Verdict::Fail,
                "mutation at index {bad_idx} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Sweep — declared probe around actual.
    // -------------------------------------------------------------------------
    #[test]
    fn declared_sweep_around_actual() {
        let sizes = vec![400_u64; 10]; // 100 tokens × 10 = 1000 actual
        let probes: Vec<(u64, PretokInv003Verdict)> = vec![
            (0, PretokInv003Verdict::Fail),
            (999, PretokInv003Verdict::Fail),
            (1000, PretokInv003Verdict::Pass), // exactly matches
            (1001, PretokInv003Verdict::Fail),
            (10_000, PretokInv003Verdict::Fail),
            (u64::MAX, PretokInv003Verdict::Fail),
        ];
        for (declared, expected) in probes {
            assert_eq!(
                verdict_from_manifest_vs_shards(declared, &sizes),
                expected,
                "declared={declared} (actual=1000) expected {expected:?}"
            );
        }
    }
}
