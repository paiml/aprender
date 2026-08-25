//! Domain-separated Philox key derivation and the bounded-draw primitive.
//!
//! The byte encoding is frozen by the contract: the key is a little-endian 64-bit
//! truncation of a SHA-256 over a domain tag, the root seed, and a domain string drawn
//! from a closed table; the counter carries the draw ordinal. Bounded draws use 64-bit
//! multiply-shift. Modulo draws and `next_f32` are forbidden — the first is biased and
//! unauditable at the edges, the second has only 23 mantissa bits.
//!
//! # Every function here is STATELESS
//!
//! No function in this module takes `&mut self`, and no mutable generator state crosses
//! any boundary. Draw *i* is a pure function of `(key, stream_id, ordinal)`, so worker
//! count, thread scheduling and iteration order cannot change it (D-20). That is the
//! whole reason this crate uses a counter-based generator instead of the `rand_chacha`
//! stream `.planning/research/STACK.md` recommends: with a stateful stream, draw *i*
//! depends on every draw before it, and worker-count independence becomes a discipline
//! the implementation must maintain rather than a fact about its type.
//!
//! # Philox is a STATISTICAL generator, never a CSPRNG
//!
//! Philox 4x32-10 is used here solely for sampling determinism. It is **not**
//! cryptographic randomness: the key is derived from a caller-visible seed, the stream is
//! seekable by construction, and nothing about it resists an adversary who knows the
//! seed. Never reuse anything in this module for tokens, nonces, salts or key material.
//!
//! # The frozen byte encoding
//!
//! Every byte-level decision below is contracted (`rng_key_derivation`, `bounded_draw`)
//! and pinned by [`rng_tests::rng_byte_encoding_golden_is_frozen`], whose constants were
//! derived from the contract text by an independent implementation rather than captured
//! from a first run of this code:
//!
//! | Decision | Value |
//! |---|---|
//! | domain tag | `b"apr-contrastive-v1\0"` — 18 ASCII bytes plus one NUL terminator |
//! | root seed | `u64::to_le_bytes`, exactly 8 bytes |
//! | key truncation | digest bytes `0..8` as two LITTLE-ENDIAN `u32` lanes; `8..32` discarded |
//! | counter | `[ordinal as u32, (ordinal >> 32) as u32, stream_id, 0]` |
//! | 64-bit assembly | `((lanes[1] as u64) << 32) \| (lanes[0] as u64)` — lane 0 is the LOW half |
//! | bounded draw | `((x as u128 * n as u128) >> 64) as u64` — multiply-shift, never modulo |

use core::num::NonZeroU64;

use sha2::{Digest, Sha256};
use trueno_rand::Philox4x32;

/// The frozen domain-separation tag.
///
/// The trailing NUL is load-bearing: without a terminator, the tag and the seed bytes are
/// ambiguous under concatenation, so a different tag with a different seed could derive
/// the same key.
const DOMAIN_TAG: &[u8] = b"apr-contrastive-v1\0";

/// A Philox key derived from a `(root_seed, domain)` pair.
///
/// Opaque on purpose: the only way to obtain one is [`derive_key`], so no call site can
/// invent a key that skips domain separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainKey([u32; 2]);

impl DomainKey {
    /// The two Philox key lanes, in derivation order.
    ///
    /// Exposed so a golden test can pin the byte encoding by value rather than by
    /// behaviour: an endianness or truncation change must be visible as a number in a
    /// diff, not only as a downstream selection that quietly moved.
    pub fn lanes(self) -> [u32; 2] {
        self.0
    }
}

/// Derive a domain-separated Philox key.
///
/// `key = trunc64_le(SHA-256(DOMAIN_TAG ‖ root_seed.to_le_bytes() ‖ domain.as_bytes()))`,
/// where `trunc64_le` reads digest bytes `0..8` as
/// `[u32::from_le_bytes(d[0..4]), u32::from_le_bytes(d[4..8])]`. Digest bytes `8..32` are
/// discarded.
///
/// The little-endian choices are stated rather than inherited from the host: a big-endian
/// machine that read these bytes natively would derive a different key for the same seed,
/// and the divergence would surface only as a different set of selected examples.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "rng_key_derivation"
)]
pub fn derive_key(root_seed: u64, domain: &str) -> DomainKey {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(root_seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();

    // Indexing is safe by construction: a SHA-256 digest is exactly 32 bytes, so both
    // 4-byte windows exist. `try_into` on a fixed-size slice cannot fail here, and the
    // fallback keeps the function total without an `unwrap`.
    let lane0 = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let lane1 = u32::from_le_bytes([digest[4], digest[5], digest[6], digest[7]]);
    DomainKey([lane0, lane1])
}

/// One Philox 4x32-10 output block at `(key, stream_id, ordinal)`.
///
/// `counter = [ordinal as u32, (ordinal >> 32) as u32, stream_id, 0]`. The result depends
/// on nothing else — not on how many draws preceded it, not on which thread asks, not on
/// the order the ordinals are requested in.
pub fn draw(key: &DomainKey, stream_id: u32, ordinal: u64) -> [u32; 4] {
    let counter = [ordinal as u32, (ordinal >> 32) as u32, stream_id, 0];
    Philox4x32::generate_at(key.0, counter)
}

/// Assemble a 64-bit value from an output block: lane 0 is the LOW half.
///
/// Frozen for the same reason the seed encoding is — the opposite convention is equally
/// natural and would silently produce a different, equally plausible-looking stream.
fn assemble64(lanes: [u32; 4]) -> u64 {
    (u64::from(lanes[1]) << 32) | u64::from(lanes[0])
}

/// A uniform-ish draw in `[0, n)` by 64-bit multiply-shift.
///
/// `bounded(..) = ((x as u128 * n.get() as u128) >> 64) as u64`, where `x` is
/// [`assemble64`] of the block at `(key, stream_id, ordinal)`.
///
/// # This IS the contracted derivation
///
/// Its non-uniformity is below 2⁻⁴⁴ for every range this protocol reaches (class buckets
/// ≤ 587 rows, pair spaces ≤ ~10⁵), and it is branch-free and index-pure. Contract
/// assumption A3 records that as the derivation itself, not as an approximation of some
/// exact-uniform alternative — so a future reader does not "correct" it into rejection
/// sampling and change every sampled identity in the process.
///
/// Modulo is FORBIDDEN: its bias at the top of the range is real and, worse, unauditable —
/// two implementations can both look correct and disagree. Float scaling is FORBIDDEN: a
/// 23-bit mantissa cannot address a bucket beyond 2²⁴ without collisions.
///
/// # A zero bound is unrepresentable
///
/// `n` is a [`NonZeroU64`], so `bounded(.., 0)` is a type error rather than a silent
/// constant or a fault:
///
/// ```compile_fail
/// use aprender_contrastive_data::rng::{bounded, derive_key};
///
/// let key = derive_key(13, "select/0");
/// let _ = bounded(&key, 0, 0, 0u64);
/// ```
///
/// The same call with a real bound compiles, which is what stops the block above from
/// being green for an unrelated reason:
///
/// ```
/// use core::num::NonZeroU64;
/// use aprender_contrastive_data::rng::{bounded, derive_key};
///
/// let key = derive_key(13, "select/0");
/// let one = NonZeroU64::new(1).expect("1 is not zero");
/// assert_eq!(bounded(&key, 0, 0, one), 0);
/// ```
#[provable_contracts_macros::contract("contrastive-pair-protocol-v1", equation = "bounded_draw")]
pub fn bounded(key: &DomainKey, stream_id: u32, ordinal: u64, n: NonZeroU64) -> u64 {
    let x = assemble64(draw(key, stream_id, ordinal));
    ((u128::from(x) * u128::from(n.get())) >> 64) as u64
}

/// The six frozen domain strings of protocol v1.
///
/// Domain separation is the ONLY mechanism keeping the selection stream and the two pair
/// streams independent — they share a root seed by design — so a collision between two of
/// these strings would correlate the streams without any test failing. The table is
/// therefore closed: adding a seventh string, or changing how an existing one renders, is
/// a versioned contract change, because either alters every sampled identity.
///
/// The `#[contract]` annotation on [`select`] covers this whole table. It sits there
/// rather than on the five constants because an attribute macro cannot annotate a `const`,
/// and because `select` is the only entry that FORMATS — hence the only one that could
/// ever drift across platforms.
pub mod domains {
    /// The selection domain for one class label: `"select/{label}"`.
    ///
    /// `label` renders as a base-10 `usize` with NO zero padding, NO thousands separator
    /// and NO locale awareness: class 7 is `"select/7"`, never `"select/07"`.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "rng_domain_strings"
    )]
    pub fn select(label: usize) -> String {
        format!("select/{label}")
    }

    /// Positive-pair class choice.
    pub const PAIRS_POS_CLASS: &str = "pairs/pos/class";
    /// Positive-pair member unranking.
    pub const PAIRS_POS_RANK: &str = "pairs/pos/rank";
    /// Negative-pair class-pair choice.
    pub const PAIRS_NEG_CLASS: &str = "pairs/neg/class";
    /// Negative-pair first member.
    pub const PAIRS_NEG_FIRST: &str = "pairs/neg/first";
    /// Negative-pair second member.
    pub const PAIRS_NEG_SECOND: &str = "pairs/neg/second";
}

#[cfg(test)]
mod rng_tests {
    use super::{assemble64, bounded, derive_key, domains, draw, DomainKey};
    use core::num::NonZeroU64;
    use proptest::prelude::{prop_assert, proptest};

    /// Golden constants for the frozen byte encoding.
    ///
    /// **Derivation, so a future re-baseline is reviewable rather than invisible.** These
    /// were produced by an independent Python implementation written from the contract
    /// text (`rng_key_derivation`, `bounded_draw`) and the Philox 4x32-10 definition in
    /// Salmon et al. (2011) — not by running this module and blessing its output. The key
    /// lanes were additionally cross-checked against `shasum -a 256` over the literal
    /// byte string `apr-contrastive-v1\0` ‖ `0d 00 00 00 00 00 00 00` ‖ `select/0`, which
    /// yields digest `b9102802 dc22ce71 …`; reading its first two 4-byte windows
    /// little-endian gives exactly `KEY_13_SELECT_0`.
    ///
    /// If this test goes red, an endianness, truncation, counter-layout or lane-assembly
    /// decision changed. That is a versioned contract change, never a re-blessing.
    const KEY_13_SELECT_0: [u32; 2] = [0x0228_10b9, 0x71ce_22dc];
    const KEY_13_SELECT_1: [u32; 2] = [465_649_502, 1_683_967_742];
    const KEY_14_SELECT_0: [u32; 2] = [1_239_703_332, 3_359_937_302];
    const BLOCK_AT_ORDINAL_7: [u32; 4] =
        [1_281_016_082, 3_815_106_876, 1_099_144_567, 2_908_329_261];
    const ASSEMBLED_AT_ORDINAL_7: u64 = 16_385_759_264_445_743_378;
    const BOUNDED_7_587: u64 = 521;
    const BOUNDED_7_24576: u64 = 21_830;
    const BOUNDED_HIGH_ORDINAL_587: u64 = 419;

    fn nz(n: u64) -> NonZeroU64 {
        NonZeroU64::new(n).expect("test bounds are non-zero by construction")
    }

    #[test]
    fn rng_byte_encoding_golden_is_frozen() {
        let key = derive_key(13, "select/0");
        assert_eq!(key.lanes(), KEY_13_SELECT_0);
        assert_eq!(derive_key(13, "select/1").lanes(), KEY_13_SELECT_1);
        assert_eq!(derive_key(14, "select/0").lanes(), KEY_14_SELECT_0);

        // Counter layout and Philox invocation.
        assert_eq!(draw(&key, 0, 7), BLOCK_AT_ORDINAL_7);

        // Lane assembly: lane 0 is the LOW half. Pinned as a VALUE rather than restated
        // as `(lanes[1] << 32) | lanes[0]`, which would only re-derive the implementation
        // and would stay green if both sides were swapped together.
        assert_eq!(assemble64(BLOCK_AT_ORDINAL_7), ASSEMBLED_AT_ORDINAL_7);

        // Multiply-shift, including one case that exercises the HIGH ordinal word and a
        // non-zero stream id — the two counter lanes a naive implementation drops.
        assert_eq!(bounded(&key, 0, 7, nz(587)), BOUNDED_7_587);
        assert_eq!(bounded(&key, 0, 7, nz(24_576)), BOUNDED_7_24576);
        assert_eq!(
            bounded(&key, 3, 12_345_678_901, nz(587)),
            BOUNDED_HIGH_ORDINAL_587
        );
    }

    #[test]
    fn rng_derive_key_separates_domains_and_seeds() {
        let base = derive_key(42, "select/0");
        assert_ne!(base, derive_key(42, "select/1"), "domain separation");
        assert_ne!(base, derive_key(43, "select/0"), "seed separation");
        assert_ne!(
            derive_key(42, domains::PAIRS_POS_CLASS),
            derive_key(42, domains::PAIRS_NEG_CLASS),
            "the pair domains must not collide"
        );
        assert_eq!(base, derive_key(42, "select/0"), "and it is reproducible");
    }

    /// Purity: draw *i* is a function of its index, so requesting ordinals out of order
    /// yields the same values as requesting them in order.
    ///
    /// This is the property that makes worker-count independence structural. A stateful
    /// stream would fail it, and no amount of "we always draw in order" discipline would
    /// make it true.
    #[test]
    fn rng_draw_is_pure_and_order_independent() {
        let key = derive_key(13, domains::PAIRS_POS_RANK);

        let in_order: Vec<[u32; 4]> = [1_u64, 3, 5].iter().map(|i| draw(&key, 0, *i)).collect();
        let shuffled: Vec<[u32; 4]> = [5_u64, 1, 3].iter().map(|i| draw(&key, 0, *i)).collect();
        assert_eq!(in_order, vec![shuffled[1], shuffled[2], shuffled[0]]);

        // Twice is twice, and different ordinals are different draws.
        assert_eq!(draw(&key, 0, 5), draw(&key, 0, 5));
        assert_ne!(draw(&key, 0, 5), draw(&key, 0, 6));
        // Streams separate too, at the same ordinal.
        assert_ne!(draw(&key, 0, 5), draw(&key, 1, 5));
    }

    #[test]
    fn rng_bounded_with_bound_one_is_always_zero() {
        let key = derive_key(29, "select/2");
        for ordinal in 0..256 {
            assert_eq!(bounded(&key, 0, ordinal, nz(1)), 0);
        }
    }

    #[test]
    fn rng_domains_select_renders_base_ten_unpadded() {
        assert_eq!(domains::select(0), "select/0");
        assert_eq!(domains::select(7), "select/7");
        assert_eq!(domains::select(12), "select/12");
        assert_eq!(domains::select(1_024), "select/1024");
        // The other five are literals with no interpolation to drift.
        assert_eq!(domains::PAIRS_POS_CLASS, "pairs/pos/class");
        assert_eq!(domains::PAIRS_POS_RANK, "pairs/pos/rank");
        assert_eq!(domains::PAIRS_NEG_CLASS, "pairs/neg/class");
        assert_eq!(domains::PAIRS_NEG_FIRST, "pairs/neg/first");
        assert_eq!(domains::PAIRS_NEG_SECOND, "pairs/neg/second");
    }

    /// The key is opaque, so the only way to get one is through domain separation.
    #[test]
    fn rng_domain_key_is_copy_and_comparable() {
        let key: DomainKey = derive_key(17, "select/0");
        let copied = key;
        assert_eq!(key, copied);
    }

    proptest! {
        /// `bounded(..) < n` for every bound this protocol reaches, over 10_000 ordinals.
        #[test]
        fn rng_bounded_is_always_below_its_bound(ordinal in 0_u64..10_000) {
            let key = derive_key(31, "select/1");
            for n in [1_u64, 2, 3, 587, 24_576] {
                let bound = NonZeroU64::new(n).expect("literal bounds are non-zero");
                prop_assert!(bounded(&key, 0, ordinal, bound) < n);
            }
        }
    }
}
