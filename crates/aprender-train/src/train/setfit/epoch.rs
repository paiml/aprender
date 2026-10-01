//! The epoch-shuffle domain — the trainer's own frozen Philox tag (D-14 hand-off).
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06). Requirement: TRN-06.
//!
//! Phase 2 fixed the per-epoch pair COUNT and explicitly left cross-epoch reshuffling to
//! this phase. This module is that hand-off, and it derives its keys under a tag of its
//! own: `apr-setfit-train-v1\0`, never Phase 2's `apr-contrastive-v1\0`. Sharing the tag
//! would correlate the training permutation with the pair-sampling stream — they share a
//! root seed by design — and nothing would fail while it happened.
//!
//! # The frozen byte encoding
//!
//! Identical in SHAPE to `aprender-contrastive-data::rng`, with one substitution: the tag.
//! Everything else is deliberately the same so a reader who has learned one has learned
//! both.
//!
//! | Decision | Value |
//! |---|---|
//! | domain tag | `b"apr-setfit-train-v1\0"` — 19 ASCII bytes plus one NUL terminator |
//! | root seed | `u64::to_le_bytes`, exactly 8 bytes |
//! | domain string | `"epoch-shuffle"` |
//! | key truncation | digest bytes `0..8` as two LITTLE-ENDIAN `u32` lanes; `8..32` discarded |
//! | counter | `[ordinal as u32, (ordinal >> 32) as u32, epoch, 0]` |
//! | 64-bit assembly | `((lanes[1] as u64) << 32) \| (lanes[0] as u64)` — lane 0 is the LOW half |
//! | bounded draw | `((x as u128 * n as u128) >> 64) as u64` — multiply-shift, never modulo |
//!
//! The trailing NUL on the tag is load-bearing: without a terminator the tag and the seed
//! bytes are ambiguous under concatenation, so a different tag with a different seed could
//! derive the same key.
//!
//! # Philox is a STATISTICAL generator, never a CSPRNG
//!
//! The key is derived from a caller-visible seed and the stream is seekable by
//! construction. Never reuse anything here for tokens, nonces, salts or key material.

use core::num::NonZeroU64;

use sha2::{Digest, Sha256};
use trueno_rand::Philox4x32;

/// The trainer's frozen domain-separation tag. NOT Phase 2's.
const DOMAIN_TAG: &[u8] = b"apr-setfit-train-v1\0";

/// The epoch-shuffle domain string.
pub const DOMAIN_EPOCH_SHUFFLE: &str = "epoch-shuffle";

/// A Philox key derived from a `(root_seed, domain)` pair under the trainer's tag.
///
/// Opaque on purpose: the only way to obtain one is [`derive_epoch_key`], so no call site
/// can invent a key that skips domain separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochKey([u32; 2]);

impl EpochKey {
    /// The two Philox key lanes, in derivation order.
    ///
    /// Exposed so a golden test can pin the byte encoding BY VALUE rather than by
    /// behaviour: an endianness or truncation change must show up as a number in a diff,
    /// not only as a training order that quietly moved.
    #[must_use]
    pub fn lanes(self) -> [u32; 2] {
        self.0
    }
}

/// Derive the trainer's domain-separated Philox key.
///
/// `key = trunc64_le(SHA-256(DOMAIN_TAG || root_seed.to_le_bytes() || domain.as_bytes()))`.
#[must_use]
pub fn derive_epoch_key(root_seed: u64, domain: &str) -> EpochKey {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(root_seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let lane0 = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let lane1 = u32::from_le_bytes([digest[4], digest[5], digest[6], digest[7]]);
    EpochKey([lane0, lane1])
}

/// A uniform-ish draw in `[0, n)` by 64-bit multiply-shift.
///
/// Modulo is FORBIDDEN — its bias at the top of the range is real and, worse, unauditable,
/// so two implementations can both look correct and disagree. Float scaling is FORBIDDEN —
/// a 23-bit mantissa cannot address a bucket beyond 2^24 without collisions. `n` is a
/// [`NonZeroU64`] so a zero bound is a type error rather than a silent constant.
#[must_use]
pub fn bounded_draw(key: EpochKey, epoch: u32, ordinal: u64, n: NonZeroU64) -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    let counter = [ordinal as u32, (ordinal >> 32) as u32, epoch, 0];
    let lanes = Philox4x32::generate_at(key.0, counter);
    let x = (u64::from(lanes[1]) << 32) | u64::from(lanes[0]);
    #[allow(clippy::cast_possible_truncation)]
    let scaled = ((u128::from(x) * u128::from(n.get())) >> 64) as u64;
    scaled
}

/// Epoch `epoch`'s pair order: a permutation of `0..n_pairs`.
///
/// # It is a pure function of `(root_seed, epoch, n_pairs)`
///
/// No ambient RNG state, no thread-count sensitivity, no dependence on how many epochs ran
/// before. Replaying epoch 3 of a run without replaying epochs 0-2 gives the same order,
/// which is what makes a partial re-verification meaningful.
///
/// # How the training loop consumes this
///
/// Batches are CONSECUTIVE FIXED-SIZE WINDOWS of this order — `[0..B)`, `[B..2B)`, and so
/// on. The final batch is SHORT when `n_pairs` is not a multiple of the batch size; it is
/// never padded and never resampled to fill. (RESEARCH Open Question 3, answered here so
/// 03-05 does not have to re-decide it.)
///
/// # The permutation costs O(B), not O(B^2)
///
/// The permuted stream is consumed through `PairSampler::pair_at(ordinal)`, which is O(1)
/// random access — so permuting the replay stream is a shuffle of indices, not a re-walk of
/// the sampler for each position. Recorded because the phase review raised the complexity
/// question; the existing Phase 2 API is the answer.
///
/// # Algorithm
///
/// Fisher-Yates descending: for `i` from `n_pairs - 1` down to `1`, draw
/// `j = bounded_draw(key, epoch, ordinal, i + 1)` and swap positions `i` and `j`, with
/// `ordinal` incrementing once per swap. Descending rather than ascending because that is
/// the direction the unbiased formulation is stated in; ascending with the same bounds is a
/// DIFFERENT (and biased) shuffle, which is precisely the kind of plausible-looking variant
/// a golden test exists to pin.
#[must_use]
pub fn epoch_pair_order(root_seed: u64, epoch: u32, n_pairs: u64) -> Vec<u64> {
    let mut order: Vec<u64> = (0..n_pairs).collect();
    if n_pairs < 2 {
        return order;
    }
    let key = derive_epoch_key(root_seed, DOMAIN_EPOCH_SHUFFLE);
    for (i, ordinal) in (1..n_pairs).rev().zip(0_u64..) {
        // `i + 1` is in `2..=n_pairs`, so it is never zero.
        let bound = NonZeroU64::new(i + 1).unwrap_or(NonZeroU64::MIN);
        let j = bounded_draw(key, epoch, ordinal, bound);
        // Both indices are `< n_pairs == order.len()`, so the swap is in bounds; `swap`
        // is used rather than indexing so the bound is checked once by the slice itself.
        let (Ok(i_index), Ok(j_index)) = (usize::try_from(i), usize::try_from(j)) else {
            // Unreachable on any 64-bit target for a budget bounded by DEFAULT_HARD_CAP.
            // Typed as a break rather than a panic so an exotic target degrades to a
            // shorter shuffle instead of aborting a training run.
            break;
        };
        order.swap(i_index, j_index);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Golden constants for the trainer's frozen byte encoding.
    ///
    /// **Derivation, so a future re-baseline is reviewable rather than invisible.** These
    /// were produced by an INDEPENDENT Python implementation of Philox 4x32-10, written
    /// from Salmon et al. (2011) and the byte-encoding table in this module's doc — not by
    /// running this code and blessing its output. (PROJECT.md permits Python solely as a
    /// numerical reference during verification; nothing Python entered the repository.)
    ///
    /// **The derivation carries its own control.** Before deriving anything under the new
    /// tag, that Python reproduced EVERY constant Phase 2 froze in
    /// `aprender-contrastive-data::rng::rng_tests` under the OLD tag — the three key-lane
    /// pairs, the block at ordinal 7, the 64-bit assembly, and all three bounded draws
    /// including the high-ordinal / non-zero-stream case. A reference implementation that
    /// reproduces eight independently frozen constants is not guessing at the ninth.
    ///
    /// The construction, restated so the derivation is reproducible from this comment
    /// alone:
    ///
    /// ```text
    /// key    = trunc64_le(SHA256(b"apr-setfit-train-v1\0" ‖ seed.to_le_bytes() ‖ "epoch-shuffle"))
    /// block  = Philox4x32-10(key, [ordinal_lo, ordinal_hi, epoch, 0])
    /// x      = (block[1] << 32) | block[0]
    /// draw   = (x * n) >> 64
    /// order  = Fisher-Yates descending over 0..n, one draw per swap, ordinal from 0
    /// ```
    ///
    /// If any of these go red, an endianness, truncation, counter-layout, lane-assembly or
    /// shuffle-direction decision changed. That is a versioned contract change, never a
    /// re-blessing.
    const KEY_42_EPOCH_SHUFFLE: [u32; 2] = [0x0127_0e6e, 0x0c64_8cc6];
    const GOLDEN_SEED42_EPOCH0_N8: [u64; 8] = [2, 3, 0, 7, 1, 5, 6, 4];
    const GOLDEN_SEED42_EPOCH1_N8: [u64; 8] = [7, 2, 5, 1, 3, 4, 0, 6];
    const GOLDEN_SEED43_EPOCH0_N8: [u64; 8] = [7, 2, 1, 4, 3, 0, 6, 5];

    #[test]
    fn epoch_byte_encoding_golden_is_frozen() {
        assert_eq!(derive_epoch_key(42, DOMAIN_EPOCH_SHUFFLE).lanes(), KEY_42_EPOCH_SHUFFLE);
        assert_eq!(epoch_pair_order(42, 0, 8), GOLDEN_SEED42_EPOCH0_N8);
        assert_eq!(epoch_pair_order(42, 1, 8), GOLDEN_SEED42_EPOCH1_N8);
        assert_eq!(epoch_pair_order(43, 0, 8), GOLDEN_SEED43_EPOCH0_N8);
    }

    /// The tag really is the trainer's own, asserted by VALUE.
    ///
    /// A key derived under Phase 2's tag for the same `(seed, domain)` must differ. If this
    /// module ever picked up `apr-contrastive-v1\0` the training permutation would silently
    /// correlate with the pair stream and every other test here would still pass.
    #[test]
    fn epoch_tag_is_the_trainers_own_not_phase_twos() {
        assert_eq!(DOMAIN_TAG, b"apr-setfit-train-v1\0");
        assert_eq!(DOMAIN_TAG.len(), 20, "19 ASCII bytes plus the NUL terminator");
        assert_eq!(
            *DOMAIN_TAG.last().expect("the tag is not empty"),
            0,
            "the NUL terminator is load-bearing",
        );

        // Phase 2's own frozen key for (13, "select/0") is [0x022810b9, 0x71ce22dc]. Under
        // THIS module's tag the same inputs must produce something else.
        let under_our_tag = derive_epoch_key(13, "select/0").lanes();
        assert_ne!(
            under_our_tag,
            [0x0228_10b9, 0x71ce_22dc],
            "this module must not be deriving under apr-contrastive-v1",
        );
    }

    #[test]
    fn epoch_order_is_a_permutation_for_every_size() {
        for n in [0_u64, 1, 2, 3, 7, 8, 16, 17, 64, 257] {
            let order = epoch_pair_order(13, 0, n);
            assert_eq!(order.len() as u64, n, "n={n}");
            let unique: BTreeSet<u64> = order.iter().copied().collect();
            assert_eq!(unique.len() as u64, n, "n={n}: every index must appear exactly once");
            assert!(order.iter().all(|&i| i < n), "n={n}: every index must be in range");
        }
    }

    #[test]
    fn epoch_order_replays_exactly() {
        for epoch in 0..4_u32 {
            let first = epoch_pair_order(29, epoch, 64);
            let second = epoch_pair_order(29, epoch, 64);
            assert_eq!(first, second, "epoch {epoch} must replay identically");
        }
    }

    #[test]
    fn epoch_order_differs_across_epochs_and_across_seeds() {
        let e0 = epoch_pair_order(29, 0, 64);
        let e1 = epoch_pair_order(29, 1, 64);
        assert_ne!(e0, e1, "consecutive epochs must reshuffle");

        let other_seed = epoch_pair_order(31, 0, 64);
        assert_ne!(e0, other_seed, "a different root seed must give a different order");
    }

    /// Purity: the order for epoch 3 does not depend on epochs 0-2 having been computed.
    #[test]
    fn epoch_order_is_a_pure_function_of_its_arguments() {
        let direct = epoch_pair_order(47, 3, 32);
        for epoch in 0..3_u32 {
            let _ = epoch_pair_order(47, epoch, 32);
        }
        assert_eq!(direct, epoch_pair_order(47, 3, 32));
    }

    #[test]
    fn epoch_degenerate_sizes_are_identity() {
        assert!(epoch_pair_order(42, 0, 0).is_empty());
        assert_eq!(epoch_pair_order(42, 0, 1), vec![0]);
    }

    #[test]
    fn epoch_bounded_draw_is_always_below_its_bound() {
        let key = derive_epoch_key(53, DOMAIN_EPOCH_SHUFFLE);
        for ordinal in 0..2_000_u64 {
            for n in [1_u64, 2, 3, 16, 587, 24_576] {
                let bound = NonZeroU64::new(n).expect("literal bounds are non-zero");
                assert!(bounded_draw(key, 0, ordinal, bound) < n, "n={n} ordinal={ordinal}");
            }
        }
    }

    #[test]
    fn epoch_bounded_draw_with_bound_one_is_always_zero() {
        let key = derive_epoch_key(53, DOMAIN_EPOCH_SHUFFLE);
        let one = NonZeroU64::new(1).expect("1 is not zero");
        for ordinal in 0..256_u64 {
            assert_eq!(bounded_draw(key, 0, ordinal, one), 0);
        }
    }

    /// Streams separate by epoch at the same ordinal — the property that makes the epoch a
    /// real coordinate rather than decoration.
    #[test]
    fn epoch_streams_separate_at_the_same_ordinal() {
        let key = derive_epoch_key(53, DOMAIN_EPOCH_SHUFFLE);
        let bound = NonZeroU64::new(u64::from(u32::MAX)).expect("non-zero");
        let mut seen = BTreeSet::new();
        for epoch in 0..8_u32 {
            seen.insert(bounded_draw(key, epoch, 0, bound));
        }
        assert_eq!(seen.len(), 8, "eight epochs must give eight distinct draws at ordinal 0");
    }
}
