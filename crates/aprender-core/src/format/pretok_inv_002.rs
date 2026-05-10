// SHIP-TWO-001 MODEL-2 — `pretokenize-bin-v1` (C-PRETOK-BIN) algorithm-level
// PARTIAL discharge for INV-PRETOK-002.
//
// Contract: `contracts/pretokenize-bin-v1.yaml` v1.0.0 PROPOSED.
//
// ## What INV-PRETOK-002 says
//
//   description: Every shard file size is a multiple of 4 bytes
//                (u32-aligned). A shard with a trailing 1/2/3-byte
//                fragment would cause off-by-one read errors in
//                `ShardBatchIter`.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a list of per-shard file sizes in bytes, Pass iff
// every size is divisible by 4 AND the list is non-empty AND every size
// is at least one u32 (4 bytes).
//
// Future implementations cannot:
// - Emit a shard with a 1/2/3-byte trailing fragment (would crash
//   ShardBatchIter::next at the partial u32 read).
// - Emit a 0-byte shard (no tokens).
// - Emit zero shards (caller bug).
//
// Sibling to [`super::pretok_inv_001`] (#1143, vocab bound) — both must
// Pass for the pretokenize step to be considered well-formed.

/// Number of bytes per u32 token id in the binary shard format.
///
/// Bound here because every shard file is a flat little-endian stream of
/// u32 token ids per `ShardBatchIter::new` (crates/aprender-train/src/
/// train/shard_reader.rs). Drift to e.g. u16 (2 bytes) or u64 (8 bytes)
/// would silently corrupt the shard format.
pub const AC_PRETOK_INV_002_U32_BYTE_SIZE: u64 = 4;

/// Binary verdict for `INV-PRETOK-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PretokInv002Verdict {
    /// Every shard is u32-aligned (size divisible by 4) AND non-zero
    /// (contains at least one token) AND the list is non-empty.
    /// `ShardBatchIter::next()` will not encounter a partial u32.
    Pass,
    /// One or more of:
    /// - Empty shard list (caller error — no work done).
    /// - Some shard size is 0 (no tokens).
    /// - Some shard size has a 1/2/3-byte trailing fragment (size % 4 != 0).
    Fail,
}

/// Pure verdict function for `INV-PRETOK-002`.
///
/// Inputs:
/// - `per_shard_byte_sizes`: byte size of each emitted shard file (e.g.,
///   from `std::fs::metadata(shard_path).len()`).
///
/// Pass iff:
/// 1. The slice is non-empty.
/// 2. Every entry is `>= AC_PRETOK_INV_002_U32_BYTE_SIZE` (4 bytes).
/// 3. Every entry is `% 4 == 0`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Three well-aligned shards — `Pass`:
/// ```
/// use aprender::format::pretok_inv_002::{
///     verdict_from_per_shard_byte_sizes, PretokInv002Verdict,
/// };
/// let sizes = vec![1024_u64, 2048, 4096];
/// assert_eq!(
///     verdict_from_per_shard_byte_sizes(&sizes),
///     PretokInv002Verdict::Pass,
/// );
/// ```
///
/// One trailing 3-byte fragment — `Fail`:
/// ```
/// use aprender::format::pretok_inv_002::{
///     verdict_from_per_shard_byte_sizes, PretokInv002Verdict,
/// };
/// let sizes = vec![1024_u64, 2051]; // 2051 % 4 == 3
/// assert_eq!(
///     verdict_from_per_shard_byte_sizes(&sizes),
///     PretokInv002Verdict::Fail,
/// );
/// ```
#[must_use]
pub fn verdict_from_per_shard_byte_sizes(per_shard_byte_sizes: &[u64]) -> PretokInv002Verdict {
    if per_shard_byte_sizes.is_empty() {
        return PretokInv002Verdict::Fail;
    }
    for &size in per_shard_byte_sizes {
        if size < AC_PRETOK_INV_002_U32_BYTE_SIZE {
            return PretokInv002Verdict::Fail;
        }
        if size % AC_PRETOK_INV_002_U32_BYTE_SIZE != 0 {
            return PretokInv002Verdict::Fail;
        }
    }
    PretokInv002Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — u32 byte size matches contract.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_u32_byte_size_is_four() {
        assert_eq!(AC_PRETOK_INV_002_U32_BYTE_SIZE, 4);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — well-aligned non-empty shards.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_typical_aligned_shards() {
        let sizes = vec![1024_u64, 2048, 4096, 8192];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Pass
        );
    }

    #[test]
    fn pass_minimum_size_4_bytes() {
        // Exactly one u32 (one token) per shard is valid.
        let sizes = vec![4_u64];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Pass
        );
    }

    #[test]
    fn pass_realistic_seq_length_2049_x_4_bytes() {
        // Per ShardBatchIter::new: (seq_length+1) u32s per sequence.
        // For seq_length=2048: each sequence is 2049 u32s = 8196 bytes.
        // batch_size=8 means 8 sequences = 65_568 bytes per shard.
        let sizes = vec![65_568_u64; 100];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Pass
        );
    }

    #[test]
    fn pass_single_small_shard() {
        let sizes = vec![64_u64];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — trailing 1/2/3-byte fragment.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_trailing_one_byte() {
        let sizes = vec![1025_u64]; // 1025 % 4 == 1
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail
        );
    }

    #[test]
    fn fail_trailing_two_bytes() {
        let sizes = vec![1026_u64]; // 1026 % 4 == 2
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail
        );
    }

    #[test]
    fn fail_trailing_three_bytes() {
        let sizes = vec![1027_u64]; // 1027 % 4 == 3
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail
        );
    }

    #[test]
    fn fail_mod4_remainder_sweep() {
        // Exhaustive: every non-zero remainder mod 4 must Fail.
        for remainder in [1_u64, 2, 3] {
            let sizes = vec![1024 + remainder];
            assert_eq!(
                verdict_from_per_shard_byte_sizes(&sizes),
                PretokInv002Verdict::Fail,
                "size with mod4 remainder {remainder} must Fail"
            );
        }
    }

    #[test]
    fn fail_one_misaligned_among_many_aligned() {
        let mut sizes = vec![4_u64; 100];
        sizes[42] = 5; // single shard violates alignment
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — sub-token sizes (< 4 bytes).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_byte_shard() {
        // Empty file is technically % 4 == 0 BUT has no tokens — caller
        // error. Conservative Fail per the `>= 4 byte` invariant.
        let sizes = vec![0_u64];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail,
            "0-byte shard contains no tokens; must Fail"
        );
    }

    #[test]
    fn fail_size_below_one_token() {
        // 1, 2, 3 bytes are simultaneously below the 4-byte minimum AND
        // not %4-aligned. Both rules catch it; either suffices to Fail.
        for size in [1_u64, 2, 3] {
            let sizes = vec![size];
            assert_eq!(
                verdict_from_per_shard_byte_sizes(&sizes),
                PretokInv002Verdict::Fail,
                "size {size} (< 4 bytes) must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — empty input list.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_shard_list() {
        let sizes: Vec<u64> = vec![];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail,
            "empty shard list implies pretokenize did nothing"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — sizes around the 4-byte minimum.
    // -------------------------------------------------------------------------
    #[test]
    fn boundary_sweep_at_minimum_size() {
        let probes: Vec<(u64, PretokInv002Verdict)> = vec![
            (0, PretokInv002Verdict::Fail),
            (1, PretokInv002Verdict::Fail),
            (2, PretokInv002Verdict::Fail),
            (3, PretokInv002Verdict::Fail),
            (4, PretokInv002Verdict::Pass), // exactly 1 token
            (5, PretokInv002Verdict::Fail), // 1.25 tokens — bad
            (6, PretokInv002Verdict::Fail),
            (7, PretokInv002Verdict::Fail),
            (8, PretokInv002Verdict::Pass), // exactly 2 tokens
            (9, PretokInv002Verdict::Fail),
            (10, PretokInv002Verdict::Fail),
            (11, PretokInv002Verdict::Fail),
            (12, PretokInv002Verdict::Pass), // exactly 3 tokens
        ];
        for (size, expected) in probes {
            let sizes = vec![size];
            assert_eq!(
                verdict_from_per_shard_byte_sizes(&sizes),
                expected,
                "size={size} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic-scale stress — 10K aligned shards.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_scale_10k_aligned_shards_pass() {
        // 10K shards each 65_568 bytes (typical 8 × 2049 × 4).
        let sizes = vec![65_568_u64; 10_000];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Pass
        );
    }

    #[test]
    fn realistic_scale_10k_one_misaligned_fails() {
        let mut sizes = vec![65_568_u64; 10_000];
        sizes[5_000] = 65_569; // one trailing byte violates
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail
        );
    }

    #[test]
    fn realistic_scale_u64_max_alignment() {
        // u64::MAX is odd (..._FFFF), not %4 aligned. Even though it's
        // an absurd size, the gate must catch it.
        let sizes = vec![u64::MAX];
        assert_eq!(
            verdict_from_per_shard_byte_sizes(&sizes),
            PretokInv002Verdict::Fail
        );
    }
}
