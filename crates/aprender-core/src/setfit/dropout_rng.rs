//! Counter-based dropout masks for the SetFit encoder (TRN-06, D-15).
//!
//! The mask element at `(root_seed, site, forward_ordinal, i)` is a PURE FUNCTION
//! OF ITS INDEX: it is computed directly, without drawing elements `0..i`, and it
//! is bitwise identical on replay. That is what makes TRN-06's "two clean runs
//! agree bitwise" a fact about the type rather than a discipline the training
//! loop has to maintain.
//!
//! # What this displaces, and why
//!
//! [`crate::nn::Dropout`] holds a `Mutex<StdRng>` seeded through `seed_from_u64`.
//! Two independent defects follow from that on a reproducibility path:
//!
//! 1. **Draw `i` depends on every draw before it.** Worker count, evaluation
//!    passes, or an extra forward anywhere upstream shift the whole stream.
//! 2. **`StdRng` is explicitly not stable across `rand` versions.** A dependency
//!    bump silently moves every mask, and therefore every loss value, with no
//!    test failing and no diff to review.
//!
//! `nn::Dropout` is untouched — other consumers keep it. Only the SetFit
//! encoder's four dotted sites are rerouted here.
//!
//! # Every function here is STATELESS
//!
//! No mask function takes `&mut self`, and no generator state crosses any
//! boundary. [`SiteDropout`] does carry two atomics, but they are *coordinates*
//! (the mode flag and the current forward ordinal), never accumulated RNG state:
//! setting them to the same values always reproduces the same masks.
//!
//! # Philox is a STATISTICAL generator, never a CSPRNG
//!
//! Philox 4x32-10 is used here solely for training determinism. It is **not**
//! cryptographic randomness: the key is derived from a caller-visible seed, the
//! stream is seekable by construction, and nothing about it resists an adversary
//! who knows the seed. Never reuse anything in this module for tokens, nonces,
//! salts or key material.
//!
//! # The frozen byte encoding
//!
//! Pinned by [`dropout_rng_tests::dropout_rng_byte_encoding_golden_is_frozen`],
//! whose constants were derived from this table by an independent Python
//! implementation rather than captured from a first run of this code:
//!
//! | Decision | Value |
//! |---|---|
//! | domain tag | `b"apr-setfit-dropout-v1\0"` — 21 ASCII bytes plus one NUL terminator |
//! | root seed | `u64::to_le_bytes`, exactly 8 bytes |
//! | site | the DOTTED HF NAME, UTF-8, no terminator (it is last) |
//! | key truncation | digest bytes `0..8` as two LITTLE-ENDIAN `u32` lanes; `8..32` discarded |
//! | counter | `[element as u32, (element >> 32) as u32, forward_ordinal, 0]` |
//! | 64-bit assembly | `((lanes[1] as u64) << 32) \| (lanes[0] as u64)` — lane 0 is the LOW half |
//! | keep rule | `keep(i) iff assemble64(draw(i)) >= threshold` |
//! | threshold | `floor(p * 2^64)` as `u128`, clamped to `0 ..= 2^64` |
//!
//! ## The tag is this module's OWN
//!
//! `b"apr-setfit-dropout-v1\0"`, deliberately NOT Phase 2's
//! `b"apr-contrastive-v1\0"`. Reusing that tag would give the pair sampler and the
//! dropout sites the same key whenever they share a root seed and a domain string
//! — a silent seed-reuse bug that correlates two streams the design assumes are
//! independent, with nothing failing to announce it.
//!
//! ## Modulo and float scaling are FORBIDDEN
//!
//! The keep decision is a comparison against a `u128` threshold, never
//! `x % n < ...` and never `(x as f64 / 2^64) < p`. Modulo's bias at the top of
//! the range is real and, worse, unauditable — two implementations can both look
//! correct and disagree. Float scaling loses low bits and leaves the rounding
//! mode unstated, so the same `p` can produce different masks on different
//! targets. Both rules are asserted by a source grep in the plan's acceptance
//! criteria, because a comment alone does not survive an edit.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use sha2::{Digest, Sha256};
use trueno_rand::Philox4x32;

use crate::autograd::Tensor;
use crate::nn::transformer::AttentionDropoutMasks;

/// The frozen domain-separation tag.
///
/// The trailing NUL is load-bearing: without a terminator the tag and the seed
/// bytes are ambiguous under concatenation, so a different tag with a different
/// seed could derive the same key.
const DOMAIN_TAG: &[u8] = b"apr-setfit-dropout-v1\0";

/// `2^64`, as the exact `f64` it is (every power of two below 2^1024 is exact).
///
/// Spelled as a literal rather than `2f64.powi(64)` so the constant a reader
/// checks against the contract text is the one the code multiplies by.
const TWO_POW_64_F64: f64 = 18_446_744_073_709_551_616.0;

/// `2^64` as a `u128` — the saturating top of the threshold range.
const TWO_POW_64_U128: u128 = 1_u128 << 64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A dropout-mask derivation was asked for something it cannot represent.
///
/// Every variant names the OFFENDING VALUE. A rate rejection that says only
/// "invalid probability" cannot be told apart from a units mistake, and the
/// near-one case below is precisely the one a reader would otherwise assume was
/// fine.
///
/// `PartialEq` but not `Eq`: the payloads are floats.
#[derive(Debug, Clone, PartialEq)]
pub enum DropoutRngError {
    /// `p` was `NaN` or `±Inf`.
    RateNotFinite {
        /// The offending rate.
        observed: f32,
    },

    /// `p` was negative.
    RateNegative {
        /// The offending rate.
        observed: f32,
    },

    /// `p` was at or above 1.0: every element would be dropped.
    RateAtOrAboveOne {
        /// The offending rate.
        observed: f32,
    },

    /// `p` was below 1.0 and still produced a non-finite inverted-dropout scale.
    ///
    /// # This guard is UNREACHABLE for `f32`, and that is a measurement
    ///
    /// The plan this module implements asked for a rate check that rejects on the
    /// computed SCALE rather than only on `p >= 1.0`, naming `p = 1.0 - 1e-40` as
    /// the value a bare rate check would miss. **The named mechanism does not
    /// hold, and it was measured rather than argued** (CLAUDE.md verification
    /// discipline 2 and 6):
    ///
    /// * `1e-40` is subnormal in `binary32` (`9.9999461e-41`), and
    ///   `1.0 - 1e-40` evaluates to **exactly `1.0`** — in `f64` as well as `f32`.
    ///   So that value never reaches this clause; `p >= 1.0` catches it.
    /// * The largest `f32` strictly below 1.0 is `0x3F7F_FFFF` = `0.99999994`.
    ///   For it, `1.0 - p` is exactly `2^-24` and the scale is exactly `2^24`
    ///   (`16777216`) — finite. Since `1.0 - p` is monotone in `p`, that is the
    ///   worst case, so **no finite `f32` in `[0, 1)` produces a non-finite
    ///   scale**. Pinned by `dropout_rng_rate_scale_guard_is_unreachable_for_f32`.
    ///
    /// It is retained anyway, deliberately: it costs one comparison, it is the
    /// literal statement of the T-3-07 mitigation, and it stops being vacuous the
    /// moment the rate's type widens or the scale's formula changes. What it must
    /// NOT do is imply that a `p < 1.0` case exists which it catches — a guard
    /// believed to cover a case it cannot reach is worse than no guard, because it
    /// buys confidence nothing paid for.
    RateScaleNotFinite {
        /// The offending rate.
        observed: f32,
        /// The non-finite `1/(1-p)` it produced.
        scale: f32,
    },

    /// A forward-call ordinal did not fit the `u32` counter lane.
    ///
    /// `u32::MAX` itself is rejected, not only values beyond it: the boundary is
    /// exclusive so the representable range and the accepted range are the same
    /// set, and no caller has to reason about an off-by-one at the wrap point.
    /// An accepted wrap would silently REUSE an earlier step's masks, which is
    /// the one failure of this scheme that looks perfectly reproducible.
    ForwardOrdinalOverflow {
        /// The ordinal that does not fit.
        observed: u64,
    },
}

impl std::fmt::Display for DropoutRngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RateNotFinite { observed } => write!(
                f,
                "DropoutRngError::RateNotFinite(dropout rate {observed} is not finite)"
            ),
            Self::RateNegative { observed } => write!(
                f,
                "DropoutRngError::RateNegative(dropout rate {observed} is negative)"
            ),
            Self::RateAtOrAboveOne { observed } => write!(
                f,
                "DropoutRngError::RateAtOrAboveOne(dropout rate {observed} would drop every element)"
            ),
            Self::RateScaleNotFinite { observed, scale } => write!(
                f,
                "DropoutRngError::RateScaleNotFinite(dropout rate {observed} is below 1.0 but its \
                 inverted-dropout scale 1/(1-p) is {scale})"
            ),
            Self::ForwardOrdinalOverflow { observed } => write!(
                f,
                "DropoutRngError::ForwardOrdinalOverflow(forward ordinal {observed} does not fit \
                 the u32 counter lane; the limit is {} exclusive)",
                u32::MAX
            ),
        }
    }
}

impl std::error::Error for DropoutRngError {}

// ---------------------------------------------------------------------------
// Key derivation and draws
// ---------------------------------------------------------------------------

/// A Philox key derived from a `(root_seed, site)` pair.
///
/// Opaque on purpose: the only way to obtain one is [`derive_key`], so no call
/// site can invent a key that skips domain separation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainKey([u32; 2]);

impl DomainKey {
    /// The two Philox key lanes, in derivation order.
    ///
    /// Exposed so a golden test can pin the byte encoding BY VALUE rather than
    /// by behaviour: an endianness or truncation change must be visible as a
    /// number in a diff, not only as a training run that quietly moved.
    #[must_use]
    pub fn lanes(self) -> [u32; 2] {
        self.0
    }
}

/// Derive a domain-separated Philox key for one dotted dropout site.
///
/// `key = trunc64_le(SHA-256(DOMAIN_TAG ‖ root_seed.to_le_bytes() ‖ site.as_bytes()))`,
/// where `trunc64_le` reads digest bytes `0..8` as
/// `[u32::from_le_bytes(d[0..4]), u32::from_le_bytes(d[4..8])]`. Digest bytes
/// `8..32` are discarded.
///
/// Keying on the DOTTED NAME rather than a position is inherited from Phase 1's
/// `site_seed` and is the half of that design worth keeping: inserting a layer
/// must not renumber the streams of the layers after it. The trade is that
/// RENAMING a site changes its stream, which is acceptable — the names are HF's
/// own and are pinned by the parameter-order gate — whereas positional drift is
/// not, because it silently re-addresses every site downstream of an edit.
///
/// The little-endian choices are stated rather than inherited from the host: a
/// big-endian machine reading these bytes natively would derive a different key
/// for the same seed, and the divergence would surface only as a different
/// training trajectory.
#[must_use]
pub fn derive_key(root_seed: u64, site: &str) -> DomainKey {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN_TAG);
    hasher.update(root_seed.to_le_bytes());
    hasher.update(site.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();

    // A SHA-256 digest is exactly 32 bytes, so both 4-byte windows exist by
    // construction; the explicit element form keeps this total with no `unwrap`.
    let lane0 = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
    let lane1 = u32::from_le_bytes([digest[4], digest[5], digest[6], digest[7]]);
    DomainKey([lane0, lane1])
}

/// One Philox 4x32-10 output block at `(key, forward_ordinal, element)`.
///
/// `counter = [element as u32, (element >> 32) as u32, forward_ordinal, 0]`.
///
/// The `forward_ordinal` lane is D-15's `block` coordinate and it is the reason
/// this module exists in its current shape. `pair_cosine_mse(za, zb, labels)`
/// takes TWO `[B,H]` embedding matrices, so a training step performs TWO
/// SEPARATE encoder forwards — one per siamese branch. Keying only on the step
/// would hand branch A and branch B the identical mask at every corresponding
/// element: an artificial correlation between the two halves of the pair
/// objective, a silent divergence from the reference recipe, and perfectly
/// deterministic-looking. See [`forward_ordinal`] for the `2*step + branch`
/// mapping.
///
/// The result depends on nothing else — not on how many draws preceded it, not
/// on which thread asks, not on the order the elements are requested in.
#[must_use]
pub fn draw(key: &DomainKey, forward_ordinal: u32, element: u64) -> [u32; 4] {
    let counter = [element as u32, (element >> 32) as u32, forward_ordinal, 0];
    Philox4x32::generate_at(key.0, counter)
}

/// Assemble a 64-bit value from an output block: lane 0 is the LOW half.
///
/// Frozen for the same reason the seed encoding is — the opposite convention is
/// equally natural and would silently produce a different, equally
/// plausible-looking stream.
#[must_use]
fn assemble64(lanes: [u32; 4]) -> u64 {
    (u64::from(lanes[1]) << 32) | u64::from(lanes[0])
}

/// The keep threshold for rate `p`, exactly.
///
/// `keep(i)` iff `assemble64(draw(i)) >= keep_threshold(p)`, so the drop
/// probability is `threshold / 2^64`.
///
/// # The rule is spelled out because the obvious phrasing was ambiguous
///
/// The first draft said `round(p * 2^64)`, which does not state a rounding mode.
/// The rule here is instead:
///
/// ```text
/// threshold = clamp(floor(p * 18446744073709551616.0), 0, 2^64)   // as u128
/// ```
///
/// with `p` widened to `f64` first. Rust's `f64 -> u128` `as` cast is DEFINED
/// (round toward zero, saturating at both ends), so this is bit-reproducible on
/// every target — no `unsafe`, no UB, no platform-dependent intrinsic.
///
/// ## Where the rounding mode is actually observable (measured)
///
/// `p * 2^64` is an EXACT `f64` operation — multiplying by a power of two only
/// shifts the exponent, so no bits are lost and the product is representable.
/// The product is additionally an INTEGER whenever its magnitude is at least
/// `2^52`, i.e. whenever `p >= 2^-12` (`0.000244140625`). Every rate this
/// encoder uses is far above that, so at `p = 0.1` and `p = 0.5` the `.floor()`
/// **cannot be observed at all**: `floor`, `ceil` and `round` return the same
/// number.
///
/// That was not obvious and it was not assumed — replacing `.floor()` with
/// `.ceil()` left the entire suite GREEN, and only then was the exactness
/// argument worked out and a rate below `2^-12` added to the goldens
/// (`p = 1e-5`, where the product is `184467440737095.53` and the two modes
/// genuinely differ). Stating the mode therefore remains necessary — small rates
/// are reachable, e.g. in an ablation — but the interesting fact is that for the
/// production rates the threshold is EXACT, with no rounding decision at all.
///
/// # Why `u128` and not `u64`
///
/// `p == 1.0` must map to `2^64` — "drop everything", since no 64-bit draw is
/// ever `>= 2^64`. In a `u64` that value wraps to `0`, which means "drop
/// NOTHING": the exact inversion of the intent, produced silently. The threshold
/// is therefore a `u128` and the comparison widens the draw. (Rate validation
/// rejects `p == 1.0` before any [`SiteDropout`] is built; the rule is stated
/// totally anyway so the function is correct on its own terms and testable at
/// the boundary.)
#[must_use]
pub fn keep_threshold(p: f64) -> u128 {
    if !p.is_finite() || p <= 0.0 {
        // Also catches NaN, whose `as` cast is defined to be 0 but reads as an
        // accident at the call site.
        return 0;
    }
    let scaled = (p * TWO_POW_64_F64).floor();
    let raw = scaled as u128;
    raw.min(TWO_POW_64_U128)
}

/// Validate a dropout rate and return its inverted-dropout scale `1/(1-p)`.
///
/// Rejects, with the offending value named, when any of these holds:
/// `!p.is_finite()`, `p < 0.0`, `p >= 1.0`, or the computed `f32` scale is not
/// finite. The fourth clause is a deliberately-retained belt to the third's
/// braces and, for `f32`, is provably unreachable — see
/// [`DropoutRngError::RateScaleNotFinite`], which records the measurement instead
/// of the plausible-but-false story it replaced.
///
/// Kept elements are multiplied by the returned scale, matching
/// [`crate::nn::Dropout`]'s inverted-dropout semantics exactly, so switching a
/// site to this module changes WHICH elements are dropped and nothing else about
/// the arithmetic.
///
/// # Errors
///
/// [`DropoutRngError::RateNotFinite`], [`DropoutRngError::RateNegative`],
/// [`DropoutRngError::RateAtOrAboveOne`] or
/// [`DropoutRngError::RateScaleNotFinite`], each naming `p`.
pub fn validate_rate(p: f32) -> Result<f32, DropoutRngError> {
    if !p.is_finite() {
        return Err(DropoutRngError::RateNotFinite { observed: p });
    }
    if p < 0.0 {
        return Err(DropoutRngError::RateNegative { observed: p });
    }
    if p >= 1.0 {
        return Err(DropoutRngError::RateAtOrAboveOne { observed: p });
    }
    let scale = 1.0 / (1.0 - p);
    if !scale.is_finite() {
        return Err(DropoutRngError::RateScaleNotFinite { observed: p, scale });
    }
    Ok(scale)
}

/// D-15's `block`: the forward-call ordinal `2 * step + branch`.
///
/// `branch` is 0 for the pair's A sentence and 1 for its B sentence. The mapping
/// is strictly monotone and a pure function of `(step, branch)`, so it is
/// replay-exact and the two branches of one step necessarily draw from different
/// Philox streams.
///
/// # Errors
///
/// [`DropoutRngError::ForwardOrdinalOverflow`] naming the computed ordinal when
/// it does not fit a `u32` counter lane. Checked arithmetic throughout: a `u64`
/// overflow here would wrap to a SMALL ordinal and reuse an early step's masks.
pub fn forward_ordinal(step: u64, branch: u32) -> Result<u32, DropoutRngError> {
    let doubled = step
        .checked_mul(2)
        .and_then(|s| s.checked_add(u64::from(branch)));
    match doubled {
        Some(ordinal) => checked_forward_ordinal(ordinal),
        // `2*step + branch` overflowed u64 itself. There is no honest `observed`
        // to report other than the step that produced it, so report the step
        // doubled in the widest type available.
        None => Err(DropoutRngError::ForwardOrdinalOverflow { observed: u64::MAX }),
    }
}

/// Narrow a forward ordinal to the `u32` counter lane, or reject it.
///
/// # Errors
///
/// [`DropoutRngError::ForwardOrdinalOverflow`] naming `ordinal` when it is at or
/// above `u32::MAX`.
pub fn checked_forward_ordinal(ordinal: u64) -> Result<u32, DropoutRngError> {
    match u32::try_from(ordinal) {
        Ok(narrowed) if narrowed < u32::MAX => Ok(narrowed),
        _ => Err(DropoutRngError::ForwardOrdinalOverflow { observed: ordinal }),
    }
}

// ---------------------------------------------------------------------------
// The site
// ---------------------------------------------------------------------------

/// One dotted dropout site's mask source.
///
/// Replaces [`crate::nn::Dropout`] on the SetFit encoder route only. The mode
/// flag and the forward ordinal are interior-mutable because a forward pass runs
/// through `&self` all the way down and the ordinal has to reach four sites per
/// layer; they are COORDINATES, not accumulated state, so nothing about
/// reproducibility depends on how many times they were read.
#[derive(Debug)]
pub struct SiteDropout {
    /// The dotted HF name this site's stream is keyed on.
    site: String,
    key: DomainKey,
    p: f32,
    /// `1/(1-p)`, validated finite at construction.
    scale: f32,
    /// `floor(p * 2^64)`, clamped — see [`keep_threshold`].
    threshold: u128,
    training: AtomicBool,
    forward_ordinal: AtomicU32,
}

impl SiteDropout {
    /// Build a site keyed on `(root_seed, site)` at rate `p`.
    ///
    /// Starts in TRAINING mode at forward ordinal 0, matching
    /// [`crate::nn::Dropout::with_seed`], so the encoder's existing
    /// `set_training(false)` at the end of construction still lands the model in
    /// eval mode exactly as it did before.
    ///
    /// # Errors
    ///
    /// Whatever [`validate_rate`] rejects, naming `p`.
    pub fn new(root_seed: u64, site: &str, p: f32) -> Result<Self, DropoutRngError> {
        let scale = validate_rate(p)?;
        Ok(Self {
            site: site.to_string(),
            key: derive_key(root_seed, site),
            p,
            scale,
            threshold: keep_threshold(f64::from(p)),
            training: AtomicBool::new(true),
            forward_ordinal: AtomicU32::new(0),
        })
    }

    /// The dotted HF name this site is keyed on.
    #[must_use]
    pub fn site(&self) -> &str {
        &self.site
    }

    /// The derived Philox key.
    ///
    /// A READ accessor, so "every site has its own stream" can be asserted on the
    /// derivation itself rather than inferred from outputs that differ.
    #[must_use]
    pub fn key(&self) -> DomainKey {
        self.key
    }

    /// The dropout probability.
    #[must_use]
    pub fn probability(&self) -> f32 {
        self.p
    }

    /// The inverted-dropout scale `1/(1-p)` applied to kept elements.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The keep threshold, `floor(p * 2^64)`.
    #[must_use]
    pub fn threshold(&self) -> u128 {
        self.threshold
    }

    /// Whether this site is in training mode.
    #[must_use]
    pub fn training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }

    /// Flip the mode. Takes `&self`: see the type docs.
    pub fn set_training(&self, training: bool) {
        self.training.store(training, Ordering::Relaxed);
    }

    /// The forward-call ordinal this site currently draws at.
    #[must_use]
    pub fn current_forward_ordinal(&self) -> u32 {
        self.forward_ordinal.load(Ordering::Relaxed)
    }

    /// Point this site at forward ordinal `ordinal`.
    ///
    /// # Errors
    ///
    /// [`DropoutRngError::ForwardOrdinalOverflow`] naming `ordinal`; the site is
    /// left at its previous ordinal in that case.
    pub fn set_forward_ordinal(&self, ordinal: u64) -> Result<(), DropoutRngError> {
        let narrowed = checked_forward_ordinal(ordinal)?;
        self.forward_ordinal.store(narrowed, Ordering::Relaxed);
        Ok(())
    }

    /// The inverted-dropout multiplier for element `i` at `forward_ordinal`.
    ///
    /// `0.0` when dropped, [`Self::scale`] when kept. THE definition — every
    /// other mask function in this module is `(0..len).map(mask_element)` — so
    /// "pure function of the index" is structural rather than a property some
    /// vectorized fast path might not share.
    #[must_use]
    pub fn mask_element(&self, forward_ordinal: u32, i: u64) -> f32 {
        let x = assemble64(draw(&self.key, forward_ordinal, i));
        if u128::from(x) >= self.threshold {
            self.scale
        } else {
            0.0
        }
    }

    /// `len` inverted-dropout multipliers at an EXPLICIT forward ordinal.
    #[must_use]
    pub fn mask_at(&self, forward_ordinal: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| self.mask_element(forward_ordinal, i as u64))
            .collect()
    }

    /// `len` inverted-dropout multipliers at the CURRENT forward ordinal.
    #[must_use]
    pub fn mask(&self, len: usize) -> Vec<f32> {
        self.mask_at(self.current_forward_ordinal(), len)
    }

    /// Apply this site: identity in eval mode or at `p == 0`, masked otherwise.
    ///
    /// The mask is a non-grad CONSTANT tensor applied with the autograd-aware
    /// [`Tensor::mul`], exactly as `nn::Dropout` does post-PMAT-922. Building the
    /// scaled values into a fresh `Tensor::new` leaf instead would SEVER the
    /// graph and freeze every parameter upstream of the site — the whole reason
    /// this shape is copied rather than reinvented.
    #[must_use]
    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training() || self.p == 0.0 {
            return input.clone();
        }
        let mask_data = self.mask(input.data().len());
        let mask = Tensor::from_vec(mask_data, input.shape());
        input.mul(&mask)
    }
}

/// The attention-probs site (site 2) reaches this same implementation.
///
/// One mask implementation for all four dotted sites: the site that lives INSIDE
/// `MultiHeadAttention` cannot be a second hand-rolled derivation, or D-15's
/// branch independence would hold at three sites and silently not at the fourth.
impl AttentionDropoutMasks for SiteDropout {
    fn attention_dropout_mask(&self, len: usize) -> Vec<f32> {
        self.mask(len)
    }
}

#[cfg(test)]
mod dropout_rng_tests {
    use super::{
        assemble64, checked_forward_ordinal, derive_key, draw, forward_ordinal, keep_threshold,
        validate_rate, DropoutRngError, SiteDropout,
    };
    use crate::autograd::Tensor;

    /// The site the goldens below are pinned on.
    const GOLDEN_SITE: &str = "embeddings.dropout";
    const GOLDEN_SEED: u64 = 13;

    // -----------------------------------------------------------------------
    // Goldens
    //
    // DERIVATION (recorded so a future re-baseline is reviewable rather than
    // invisible). Every constant in this block was produced by an INDEPENDENT
    // Python implementation written from this module's frozen-encoding table and
    // the Philox 4x32-10 definition in Salmon et al. (2011), NOT by running the
    // Rust code under test and blessing its output:
    //
    // ```python
    // import hashlib, math, struct
    // TAG = b"apr-setfit-dropout-v1\x00"
    // M0, M1, W0, W1, MASK32 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85, 0xFFFFFFFF
    //
    // def derive_key(seed, site):
    //     d = hashlib.sha256(TAG + struct.pack('<Q', seed) + site.encode()).digest()
    //     return [struct.unpack('<I', d[0:4])[0], struct.unpack('<I', d[4:8])[0]]
    //
    // def rnd(c, k):
    //     p0, p1 = M0 * c[0], M1 * c[2]
    //     return [((p1 >> 32) & MASK32) ^ c[1] ^ k[0], p1 & MASK32,
    //             ((p0 >> 32) & MASK32) ^ c[3] ^ k[1], p0 & MASK32]
    //
    // def philox(c, k):
    //     c, k = list(c), list(k)
    //     for r in range(10):
    //         c = rnd(c, k)
    //         if r < 9:
    //             k = [(k[0] + W0) & MASK32, (k[1] + W1) & MASK32]
    //     return c
    //
    // def draw(k, ordinal, element):
    //     return philox([element & MASK32, (element >> 32) & MASK32, ordinal, 0], k)
    //
    // assemble64 = lambda l: ((l[1] << 32) | l[0]) & 0xFFFFFFFFFFFFFFFF
    // keep_threshold = lambda p: min(max(math.floor(p * 18446744073709551616.0), 0), 1 << 64)
    // ```
    //
    // CROSS-CHECK, independent of that script: `shasum -a 256` over the literal
    // bytes `apr-setfit-dropout-v1\0` ‖ `0d 00 00 00 00 00 00 00` ‖
    // `embeddings.dropout` yields
    // `32f3baa5 f961657e af38b482 d6c503aa 1df69d97 83cca03e 544bc1cd a5c2ee59`.
    // Reading its first two 4-byte windows LITTLE-ENDIAN gives exactly
    // `KEY_13_EMBEDDINGS` below — so the key encoding is pinned by two
    // derivations that share no code.
    //
    // If any of this goes red, an endianness, truncation, counter-layout,
    // lane-assembly or threshold decision changed. That is a versioned contract
    // change, never a re-blessing.
    // -----------------------------------------------------------------------

    const KEY_13_EMBEDDINGS: [u32; 2] = [0xa5ba_f332, 0x7e65_61f9];
    const KEY_13_LAYER0_ATTN: [u32; 2] = [0xd0cf_7225, 0x1ade_02c9];
    const KEY_14_EMBEDDINGS: [u32; 2] = [0x3a40_a784, 0x1322_f43f];

    const BLOCK_ORD0_ELEM0: [u32; 4] = [3_836_206_948, 4_227_518_855, 1_470_901_809, 3_372_378_841];
    const ASSEMBLED_ORD0_ELEM0: u64 = 18_157_055_229_284_573_028;
    const BLOCK_ORD7_ELEM3: [u32; 4] = [2_242_403_448, 2_398_921_103, 4_169_238_919, 3_718_170_778];
    /// Exercises the HIGH element word AND a non-zero ordinal — the two counter
    /// lanes a naive implementation drops.
    const BLOCK_ORD1_HIGH_ELEM: [u32; 4] =
        [1_286_292_542, 2_418_118_001, 4_129_126_676, 3_075_353_215];

    // `math.floor(p * 2**64)` for each rate. Note that 0.1 as an `f64` LITERAL and
    // `f64::from(0.1_f32)` are DIFFERENT numbers and therefore different
    // thresholds; both are pinned, because the encoder's `DROPOUT_P` is an `f32`
    // and the widening is part of the derivation rather than an accident.
    const THRESHOLD_P0: u128 = 0;
    const THRESHOLD_P0_1_F64: u128 = 1_844_674_407_370_955_264;
    const THRESHOLD_P0_5: u128 = 9_223_372_036_854_775_808;
    const THRESHOLD_DROPOUT_P: u128 = 1_844_674_434_858_745_856;
    const THRESHOLD_P1: u128 = 1_u128 << 64;
    /// `math.floor(1e-5 * 2**64)` — the ONLY golden here that can see the
    /// rounding mode. `1e-5 * 2^64 = 184467440737095.53`, so `floor` and `ceil`
    /// differ by one; at every production rate the product is an exact integer
    /// and the mode is unobservable. Added after a `.floor()` -> `.ceil()`
    /// mutation left the whole suite green.
    const THRESHOLD_P1E_5: u128 = 184_467_440_737_095;

    fn site(p: f32) -> SiteDropout {
        SiteDropout::new(GOLDEN_SEED, GOLDEN_SITE, p).expect("test rates are valid")
    }

    // -----------------------------------------------------------------------
    // Accessors and the trait forwarder
    //
    // Every assertion here exists because a specific mutant SURVIVED the
    // complete 03-10 mutation run of this file (79 mutants, run twice under
    // different test configs with byte-identical survivor sets): `site ->
    // ""/"xyzzy"`, `probability -> -1.0/0.0/1.0`, `scale -> 1.0`, `threshold
    // -> 0`, `current_forward_ordinal -> 0`, and all four
    // `attention_dropout_mask -> vec![...]` variants. The accessors were read
    // by ZERO tests, and the forwarder was never called THROUGH the trait by
    // any lib test. A behavioral test cannot kill an accessor mutant — the
    // mutation changes the accessor, not the field — so these read each
    // accessor against an INDEPENDENTLY derived expectation.
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_accessors_report_the_constructed_values() {
        let s = site(0.1);

        // Round-trip of the identity fields (kills `site -> "" / "xyzzy"`,
        // `probability -> -1.0 / 0.0 / 1.0`).
        assert_eq!(s.site(), GOLDEN_SITE);
        assert_eq!(s.probability(), 0.1);

        // The derived fields compare against derivations, not against a second
        // read of the same struct: scale is validate_rate's own output for this
        // p (kills `scale -> 1.0`; 1/(1-0.1) != 1.0), and threshold is the
        // FROZEN golden produced by the independent Python in this module's
        // header (kills `threshold -> 0`).
        assert_eq!(s.scale(), validate_rate(0.1).expect("0.1 is a valid rate"));
        assert_eq!(s.threshold(), THRESHOLD_DROPOUT_P);
        assert_eq!(s.threshold(), keep_threshold(f64::from(0.1_f32)));

        // The ordinal is asserted at a NON-default value: at construction it is
        // genuinely 0 — the mutant's constant — so an assertion there would
        // pass with or without the mutation and prove nothing.
        s.set_forward_ordinal(7).expect("7 fits u32");
        assert_eq!(s.current_forward_ordinal(), 7);
    }

    #[test]
    fn dropout_rng_attention_mask_trait_forwarder_reaches_the_real_mask() {
        use crate::nn::transformer::AttentionDropoutMasks;

        // p = 0.5 so a correct mask carries BOTH outcomes and the kept value is
        // exactly 2.0 — no constant vec can collide with it.
        let s = site(0.5);
        s.set_forward_ordinal(3).expect("3 fits u32");

        // THROUGH the trait object, the way the in-attention site is reached in
        // production. The direct-method tests in this module cannot see a
        // mutation of the forwarder (kills all four `attention_dropout_mask ->
        // vec![...]` variants).
        let via_trait: &dyn AttentionDropoutMasks = &s;
        let len = 64_usize;
        let mask = via_trait.attention_dropout_mask(len);

        assert_eq!(mask.len(), len, "a fixed 0- or 1-element vec is not a mask");
        assert_eq!(
            mask,
            s.mask(len),
            "the forwarder must return THE mask, not a lookalike"
        );
        // Non-vacuity for the equality above: the mask must have structure no
        // constant vec has — both outcomes present at p = 0.5. Deterministic:
        // fixed key and ordinal, so this either always holds or never does.
        assert!(
            mask.contains(&0.0),
            "no element dropped at p=0.5 across 64 draws"
        );
        assert!(mask.contains(&2.0), "no element kept-and-scaled at p=0.5");
    }

    // -----------------------------------------------------------------------
    // Byte encoding
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_byte_encoding_golden_is_frozen() {
        assert_eq!(
            derive_key(GOLDEN_SEED, GOLDEN_SITE).lanes(),
            KEY_13_EMBEDDINGS
        );
        assert_eq!(
            derive_key(GOLDEN_SEED, "encoder.layer.0.attention.self.dropout").lanes(),
            KEY_13_LAYER0_ATTN,
            "site separation"
        );
        assert_eq!(
            derive_key(14, GOLDEN_SITE).lanes(),
            KEY_14_EMBEDDINGS,
            "seed separation"
        );

        let key = derive_key(GOLDEN_SEED, GOLDEN_SITE);
        assert_eq!(draw(&key, 0, 0), BLOCK_ORD0_ELEM0, "counter layout");
        assert_eq!(
            draw(&key, 7, 3),
            BLOCK_ORD7_ELEM3,
            "ordinal + element lanes"
        );
        assert_eq!(
            draw(&key, 1, 12_345_678_901),
            BLOCK_ORD1_HIGH_ELEM,
            "the HIGH element word must reach counter lane 1"
        );

        // Lane assembly pinned as a VALUE, not restated as
        // `(lanes[1] << 32) | lanes[0]` — that would only re-derive the
        // implementation and would stay green if both sides were swapped.
        assert_eq!(assemble64(BLOCK_ORD0_ELEM0), ASSEMBLED_ORD0_ELEM0);
    }

    #[test]
    fn dropout_rng_tag_is_this_modules_own_and_not_phase_twos() {
        // A cross-phase domain collision is a silent seed-reuse bug: the pair
        // sampler and a dropout site would share a key wherever they share a root
        // seed and a domain string, correlating two streams the design assumes are
        // independent. Asserted by VALUE against the digest the contrastive tag
        // would have produced for the same (seed, name).
        let ours = derive_key(GOLDEN_SEED, GOLDEN_SITE).lanes();
        let phase_two_tag_would_give = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"apr-contrastive-v1\0");
            h.update(GOLDEN_SEED.to_le_bytes());
            h.update(GOLDEN_SITE.as_bytes());
            let d: [u8; 32] = h.finalize().into();
            [
                u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
                u32::from_le_bytes([d[4], d[5], d[6], d[7]]),
            ]
        };
        assert_ne!(
            ours, phase_two_tag_would_give,
            "this module derived the SAME key Phase 2's tag would — the domain tag \
             is not separating the phases"
        );
    }

    // -----------------------------------------------------------------------
    // Threshold
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_threshold_goldens_are_frozen() {
        assert_eq!(keep_threshold(0.0), THRESHOLD_P0);
        assert_eq!(keep_threshold(0.1), THRESHOLD_P0_1_F64);
        assert_eq!(keep_threshold(0.5), THRESHOLD_P0_5);
        assert_eq!(keep_threshold(f64::from(0.1_f32)), THRESHOLD_DROPOUT_P);
        assert_ne!(
            THRESHOLD_P0_1_F64, THRESHOLD_DROPOUT_P,
            "the f32 and f64 spellings of 0.1 must NOT collapse to one threshold — \
             if they do, the widening step was silently dropped"
        );

        // The rounding MODE, which none of the rates above can see: `p * 2^64` is
        // an exact exponent shift and is integral for every `p >= 2^-12`. This is
        // the one golden that distinguishes floor from ceil, and it exists
        // because a `.floor()` -> `.ceil()` mutation was measured to leave the
        // rest of this suite green.
        assert_eq!(keep_threshold(1e-5), THRESHOLD_P1E_5);
        assert_ne!(
            keep_threshold(1e-5),
            THRESHOLD_P1E_5 + 1,
            "the threshold is rounding AWAY from zero"
        );

        // `p == 1.0` maps to exactly 2^64 (drop everything). In a u64 this value
        // wraps to 0, which means drop NOTHING: the exact inversion of the intent,
        // produced silently. That is why the threshold is a u128.
        assert_eq!(keep_threshold(1.0), THRESHOLD_P1);
        assert!(THRESHOLD_P1 > u128::from(u64::MAX));

        // Total on garbage, without an `as` cast surprise.
        assert_eq!(keep_threshold(f64::NAN), 0);
        assert_eq!(keep_threshold(-0.5), 0);
        assert_eq!(keep_threshold(f64::INFINITY), 0);
    }

    #[test]
    fn dropout_rng_threshold_is_the_drop_boundary_not_the_keep_boundary() {
        // The band that would catch an inverted comparison. p = 0.9 must drop
        // ~90 % of a large mask; a flipped `>=` keeps ~90 % and every other test
        // here still passes.
        let s = SiteDropout::new(GOLDEN_SEED, GOLDEN_SITE, 0.9).expect("0.9 is valid");
        let mask = s.mask_at(0, 20_000);
        let dropped = mask.iter().filter(|v| **v == 0.0).count();
        assert!(
            (17_400..=18_600).contains(&dropped),
            "dropped {dropped} of 20000 at p = 0.9; expected ~18000, so the keep \
             comparison is inverted or the threshold is wrong"
        );
    }

    // -----------------------------------------------------------------------
    // Purity, replay, separation
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_mask_element_is_a_pure_function_of_its_index() {
        // THE property that makes worker-count independence structural. A stateful
        // stream fails it, and no amount of "we always draw in order" discipline
        // would make it true.
        let s = site(0.3);
        for (ordinal, len) in [(0_u32, 64_usize), (1, 64), (7, 256), (4_242, 33)] {
            let sequential = s.mask_at(ordinal, len);
            for i in [0_usize, 1, 5, len / 2, len - 1] {
                assert_eq!(
                    s.mask_element(ordinal, i as u64),
                    sequential[i],
                    "element {i} at ordinal {ordinal} differs when computed directly"
                );
            }
            // Out-of-order requests give the in-order answers.
            let shuffled: Vec<f32> = [len - 1, 0, len / 2]
                .iter()
                .map(|i| s.mask_element(ordinal, *i as u64))
                .collect();
            assert_eq!(
                shuffled,
                vec![sequential[len - 1], sequential[0], sequential[len / 2]]
            );
        }
    }

    #[test]
    fn dropout_rng_replay_is_bitwise_and_every_coordinate_separates() {
        let base = SiteDropout::new(7, "embeddings.dropout", 0.25).expect("valid");
        let a = base.mask_at(4, 512);
        let b = base.mask_at(4, 512);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "element {i}: replay is not exact");
        }
        // A SECOND, independently constructed site replays it too — otherwise
        // "replay" would only mean "this object is idempotent".
        let twin = SiteDropout::new(7, "embeddings.dropout", 0.25).expect("valid");
        assert_eq!(a, twin.mask_at(4, 512));

        // Changing any ONE coordinate changes the mask.
        let other_seed = SiteDropout::new(8, "embeddings.dropout", 0.25).expect("valid");
        let other_site =
            SiteDropout::new(7, "encoder.layer.0.output.dropout", 0.25).expect("valid");
        assert_ne!(a, other_seed.mask_at(4, 512), "root seed does not separate");
        assert_ne!(a, other_site.mask_at(4, 512), "site does not separate");
        assert_ne!(a, base.mask_at(5, 512), "forward ordinal does not separate");
    }

    // -----------------------------------------------------------------------
    // Branch independence — the load-bearing D-15 gate
    // -----------------------------------------------------------------------

    /// Masks at `2*step` and `2*step + 1` must differ like INDEPENDENT draws.
    ///
    /// Not "not equal": two streams that collapsed to one would give distance 0,
    /// but so would a hundred subtler defects give a distance that is merely
    /// small. Two independent inverted-dropout masks of length `n` at rate `p`
    /// disagree at each position with probability `2p(1-p)` (one kept, one
    /// dropped, either way round), so the distance is `Binomial(n, 2p(1-p))`. The
    /// band below is `±4 standard deviations` around that mean, computed FROM
    /// `(n, p)` rather than from the observed value — a band read off a first run
    /// would pass by construction.
    ///
    /// Both rates are exercised, because one failing (or passing) input is an
    /// anecdote.
    #[test]
    fn dropout_rng_branches_of_one_step_draw_independent_masks() {
        const LEN: usize = 512;
        for (p, step) in [(0.1_f32, 3_u64), (0.5, 3), (0.1, 0), (0.5, 17)] {
            let s = SiteDropout::new(GOLDEN_SEED, GOLDEN_SITE, p).expect("valid rate");
            let branch_a = forward_ordinal(step, 0).expect("2*step fits");
            let branch_b = forward_ordinal(step, 1).expect("2*step+1 fits");
            assert_eq!(u64::from(branch_a), 2 * step, "branch A is 2*step");
            assert_eq!(u64::from(branch_b), 2 * step + 1, "branch B is 2*step + 1");

            let mask_a = s.mask_at(branch_a, LEN);
            let mask_b = s.mask_at(branch_b, LEN);
            let hamming = mask_a
                .iter()
                .zip(mask_b.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();

            let q = 2.0 * f64::from(p) * (1.0 - f64::from(p));
            let n = LEN as f64;
            let mean = n * q;
            let sd = (n * q * (1.0 - q)).sqrt();
            let lo = (mean - 4.0 * sd).max(1.0);
            let hi = mean + 4.0 * sd;
            assert!(
                (lo..=hi).contains(&(hamming as f64)),
                "p = {p}, step = {step}: Hamming distance {hamming} is outside \
                 [{lo:.1}, {hi:.1}] (mean {mean:.1}, sd {sd:.2}). A distance of 0 \
                 means the two siamese branches collapsed onto ONE stream and are \
                 sharing a mask — D-15's whole point"
            );
        }
    }

    #[test]
    fn dropout_rng_forward_ordinal_is_monotone_in_step_and_branch() {
        let mut previous = None;
        for step in 0_u64..8 {
            for branch in 0_u32..2 {
                let ordinal = forward_ordinal(step, branch).expect("small ordinals fit");
                if let Some(prev) = previous {
                    assert!(ordinal > prev, "2*{step}+{branch} is not monotone");
                }
                previous = Some(ordinal);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Mode
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_eval_mode_is_the_identity_and_consumes_nothing() {
        let s = site(0.5);
        let x = Tensor::new(&[0.25_f32, -1.5, 3.0, 7.75], &[4]);
        s.set_training(false);
        let y = s.forward(&x);
        for (i, (a, b)) in x.data().iter().zip(y.data().iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: eval is not identity"
            );
        }
        // "Consumes nothing" is structural here: the ordinal is a coordinate, so
        // an eval pass cannot move it. Asserted anyway, because that IS the
        // property `nn::Dropout`'s Mutex<StdRng> could not offer.
        assert_eq!(s.current_forward_ordinal(), 0);

        // And training mode is not the identity, or the assertion above is
        // satisfied by a site that never drops anything.
        s.set_training(true);
        let z = s.forward(&x);
        assert!(
            x.data()
                .iter()
                .zip(z.data().iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "train mode was also the identity — no element was dropped or scaled"
        );
    }

    #[test]
    fn dropout_rng_rate_zero_keeps_everything_unscaled() {
        let s = site(0.0);
        assert_eq!(s.scale(), 1.0);
        assert_eq!(s.threshold(), 0);
        assert!(
            s.mask_at(0, 1_000).iter().all(|v| *v == 1.0),
            "p = 0 dropped or rescaled an element"
        );
        // The forward is the identity too — at p == 0 by the early return, and it
        // would be identical elementwise anyway.
        let x = Tensor::new(&[1.0_f32, 2.0, 3.0], &[3]);
        let y = s.forward(&x);
        assert_eq!(x.data(), y.data());
    }

    // -----------------------------------------------------------------------
    // Rate edges
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_rate_one_is_a_typed_error_naming_the_value() {
        let err = validate_rate(1.0).expect_err("p = 1.0 must be rejected");
        assert_eq!(err, DropoutRngError::RateAtOrAboveOne { observed: 1.0 });
        assert!(err.to_string().contains('1'), "got {err}");
        assert!(SiteDropout::new(1, "s", 1.0).is_err());
    }

    /// `p = 1.0 - 1e-40` — the value the plan named — IS rejected, and the
    /// message names it. The MECHANISM is not the one the plan predicted, and
    /// this test says which one it is.
    #[test]
    fn dropout_rng_rate_just_below_one_is_a_typed_error_naming_the_value() {
        let p: f32 = 1.0 - 1e-40;
        // Measured, not assumed: `1e-40` is subnormal in binary32 and the
        // subtraction rounds straight back to 1.0 — in f64 too. So this value
        // never reaches the scale clause; the rate clause catches it.
        assert_eq!(p.to_bits(), 1.0_f32.to_bits(), "1.0 - 1e-40 is exactly 1.0");
        let err = validate_rate(p).expect_err("must be rejected");
        assert_eq!(err, DropoutRngError::RateAtOrAboveOne { observed: p });
        let text = err.to_string();
        assert!(
            text.contains('1'),
            "the message must name the value: {text}"
        );
    }

    /// The scale guard cannot fire for any finite `f32` rate below 1.0.
    ///
    /// Pins the measurement that corrected the plan's stated mechanism, so nobody
    /// later "fixes" the near-one test by asserting a variant that is unreachable.
    #[test]
    fn dropout_rng_rate_scale_guard_is_unreachable_for_f32() {
        // The largest f32 strictly below 1.0.
        let worst = f32::from_bits(0x3F7F_FFFF);
        assert!(worst < 1.0);
        assert_eq!(1.0 - worst, 2.0_f32.powi(-24));
        let scale = validate_rate(worst).expect("the worst case is still valid");
        assert_eq!(
            scale, 16_777_216.0,
            "1/(1-p) at the boundary is exactly 2^24"
        );
        assert!(scale.is_finite());

        // Monotonicity does the rest: `1.0 - p` shrinks as `p` grows, so the
        // boundary above is the largest scale any accepted rate can produce.
        for bits in [0x3F7F_FFFEu32, 0x3F7F_FFF0, 0x3F7F_FF00, 0x3F00_0000] {
            let p = f32::from_bits(bits);
            let s = validate_rate(p).expect("still below one");
            assert!(s.is_finite() && s <= 16_777_216.0, "p = {p} gave scale {s}");
        }
    }

    #[test]
    fn dropout_rng_rate_rejects_nan_infinity_and_negatives_by_name() {
        // NaN is compared by PREDICATE, not by `==`: `NaN != NaN`, so an
        // `assert_eq!` on the whole variant fails even when the code is right —
        // and would have been "fixed" by weakening the assertion to `is_err()`,
        // which no longer checks WHICH variant fired.
        match validate_rate(f32::NAN).expect_err("NaN") {
            DropoutRngError::RateNotFinite { observed } => assert!(observed.is_nan()),
            other => panic!("NaN must be RateNotFinite, got {other}"),
        }
        assert_eq!(
            validate_rate(f32::INFINITY).expect_err("inf"),
            DropoutRngError::RateNotFinite {
                observed: f32::INFINITY
            }
        );
        let err = validate_rate(-0.25).expect_err("negative");
        assert_eq!(err, DropoutRngError::RateNegative { observed: -0.25 });
        assert!(err.to_string().contains("-0.25"), "got {err}");
    }

    // -----------------------------------------------------------------------
    // Forward-ordinal overflow
    // -----------------------------------------------------------------------

    #[test]
    fn dropout_rng_forward_ordinal_overflow_is_a_typed_error_naming_the_value() {
        // The boundary is EXCLUSIVE at u32::MAX, so the accepted set and the
        // representable set are the same and nobody has to reason about the wrap
        // point. An accepted wrap would silently reuse an early step's masks —
        // the one failure of this scheme that looks perfectly reproducible.
        let limit = u64::from(u32::MAX);
        assert!(checked_forward_ordinal(limit - 1).is_ok());

        for observed in [limit, limit + 1, u64::MAX] {
            let err = checked_forward_ordinal(observed).expect_err("must be rejected");
            assert_eq!(err, DropoutRngError::ForwardOrdinalOverflow { observed });
            assert!(
                err.to_string().contains(&observed.to_string()),
                "the message must name the observed ordinal: {err}"
            );
        }

        // And through the two callers that a training loop actually uses.
        let s = site(0.1);
        assert!(s.set_forward_ordinal(limit).is_err());
        assert_eq!(
            s.current_forward_ordinal(),
            0,
            "a rejected ordinal must leave the site where it was"
        );
        assert!(
            forward_ordinal(u64::MAX / 2, 1).is_err(),
            "2*step must be checked before it can wrap into a SMALL ordinal"
        );
    }
}
