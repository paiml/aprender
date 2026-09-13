//! Bounded pair sampling: canonical pairs, capacity math, budget resolution, and the
//! singleton and degenerate-layout policies.
//!
//! # Contract: contrastive-pair-protocol-v1.yaml
//!
//! Equations `canonical_pair`, `positive_capacity`, `negative_capacity`,
//! `default_epoch_budget`, `budget_resolution`, `pair_stream_degenerate_policy`,
//! `pair_stream`, `singleton_policy`, `untrusted_pair_ingest`, `split_span_fail_closed`.
//!
//! # Counting is cheap; enumerating is not
//!
//! SetFit's oversampling *count* is a closed form over the per-class sizes — `O(K)` — while
//! only ENUMERATING the pairs is quadratic. That asymmetry is the whole reason fidelity and
//! boundedness are compatible here (D-14): Aprender reproduces the reference's per-epoch
//! COUNT exactly while never materializing the pair set the reference's own
//! `np.triu_indices(n)` builds.
//!
//! # Everything fallible is typed
//!
//! Every capacity function returns `Result<u64, ContrastiveDataError>` and uses checked
//! arithmetic at every step. A wrapped capacity is not an obviously wrong huge number — it
//! is a small, plausible-looking one, and it would silently under-sample forever while
//! every balance and membership test stayed green.

use core::cmp::Ordering;
use core::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::error::ContrastiveDataError;
use crate::rng::{bounded, derive_key, domains, DomainKey};
use crate::select::{SelectedId, Selection};

/// The default pair hard cap: 2²⁰.
///
/// This value is contract-resident (`default_epoch_budget`) and must byte-match the
/// contract. "Configurable hard cap" is locked D-14 text; the *value* is a discretion
/// choice, sized so the clamp NEVER engages for a contracted layout — the largest
/// contracted closed form is 24,576, forty-two times below this cap. The cap is a
/// denial-of-service ceiling for adversarial inputs, not a tuning knob that quietly
/// reshapes normal runs.
pub const DEFAULT_HARD_CAP: u64 = 1_048_576;

/// The version tag of the degenerate-layout policy recorded in every pair replay record.
///
/// Version-tagged because an undefined degenerate case is not an edge case; it is a place
/// where two implementations silently disagree.
pub const DEGENERATE_POLICY_VERSION: u32 = 1;

// ===========================================================================================
// 1. Pair identity (D-10 / D-12)
// ===========================================================================================

/// An unordered, self-pair-free pair of [`SelectedId`] ordinals.
///
/// The fields are PRIVATE and [`CanonicalPair::new`] is the SOLE constructor, so both
/// orientations of one unordered pair are the same value and `(x, x)` is unrepresentable
/// (D-12). No configuration value can resurrect a self-pair, and a conflicting label for
/// the same unordered pair is structurally impossible rather than merely tested against.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CanonicalPair {
    lo: SelectedId,
    hi: SelectedId,
}

impl CanonicalPair {
    /// The only way to build a pair.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::SelfPair`] when both endpoints are the same ordinal.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "canonical_pair"
    )]
    pub fn new(a: SelectedId, b: SelectedId) -> Result<Self, ContrastiveDataError> {
        match a.cmp(&b) {
            Ordering::Less => Ok(Self { lo: a, hi: b }),
            Ordering::Greater => Ok(Self { lo: b, hi: a }),
            Ordering::Equal => Err(ContrastiveDataError::SelfPair {
                id: u64::from(a.ordinal()),
            }),
        }
    }

    /// The lower endpoint.
    pub fn lo(&self) -> SelectedId {
        self.lo
    }

    /// The upper endpoint.
    pub fn hi(&self) -> SelectedId {
        self.hi
    }
}

/// A canonical pair plus the target DERIVED from its endpoints' classes.
///
/// `1.0` when both endpoints share a class, `0.0` otherwise. The target is never accepted
/// from caller input, because a caller-supplied target is precisely how a poisoned pair
/// claims to be a positive.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LabeledPair {
    /// The unordered endpoint pair.
    pub pair: CanonicalPair,
    /// `1.0` same class, `0.0` different class. Derived, never supplied.
    pub target: f32,
}

// ===========================================================================================
// 2. Closed-form capacity (D-14) — all fallible
// ===========================================================================================

fn overflow(operation: &str) -> ContrastiveDataError {
    ContrastiveDataError::ArithmeticOverflow {
        operation: operation.to_string(),
    }
}

/// `Σ_k C(n_k, 2)` — the number of distinct same-class unordered pairs.
///
/// `O(K)` in the number of classes. Self-pairs are EXCLUDED, hence `n(n−1)/2` rather than
/// `n(n+1)/2`: that exclusion is an Aprender policy (deviation clause 3), NOT SetFit's
/// behaviour — the pinned `setfit==1.1.3` enumerates `np.triu_indices(n, 0)` and therefore
/// includes the diagonal, contradicting its own published documentation.
///
/// A singleton class contributes exactly 0, which is what makes
/// [`SingletonPolicy::NegativesOnly`] fall out of the arithmetic instead of needing a
/// special case.
///
/// # Errors
///
/// [`ContrastiveDataError::ArithmeticOverflow`] naming the operation that overflowed.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "positive_capacity"
)]
pub fn positive_capacity(class_sizes: &[u64]) -> Result<u64, ContrastiveDataError> {
    let mut total: u64 = 0;
    for &n in class_sizes {
        // `n * (n - 1)` is always even, so the halving is exact and loses nothing.
        let product = n
            .checked_mul(n.saturating_sub(1))
            .ok_or_else(|| overflow("positive_capacity/class_product"))?;
        total = total
            .checked_add(product / 2)
            .ok_or_else(|| overflow("positive_capacity/total"))?;
    }
    Ok(total)
}

/// `Σ_{j<k} n_j · n_k` — the number of distinct cross-class unordered pairs.
///
/// # Why the running-prefix evaluation order
///
/// The contract's formula line gives the algebraically equivalent `(S² − Σ n_k²) / 2`.
/// Both are `O(K)` and neither enumerates class PAIRS — the property the contract's
/// invariant actually protects — but `S²` overflows `u64` long before the true capacity
/// does, so it would report `ArithmeticOverflow` for layouts whose answer fits perfectly
/// well. This function therefore accumulates `Σ_k n_k · (Σ_{j<k} n_j)`, which overflows
/// exactly when the RESULT does. `negative_capacity_agrees_with_the_sum_of_squares_derivation`
/// pins the two against each other wherever the second is computable at all, so the
/// evaluation order cannot drift into a different quantity.
///
/// # Errors
///
/// [`ContrastiveDataError::ArithmeticOverflow`] naming the operation that overflowed.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "negative_capacity"
)]
pub fn negative_capacity(class_sizes: &[u64]) -> Result<u64, ContrastiveDataError> {
    let mut seen: u64 = 0;
    let mut total: u64 = 0;
    for &n in class_sizes {
        let cross = n
            .checked_mul(seen)
            .ok_or_else(|| overflow("negative_capacity/cross_product"))?;
        total = total
            .checked_add(cross)
            .ok_or_else(|| overflow("negative_capacity/total"))?;
        seen = seen
            .checked_add(n)
            .ok_or_else(|| overflow("negative_capacity/running_total"))?;
    }
    Ok(total)
}

/// The RAW closed form `2 · max(positive_capacity, negative_capacity)`, before any clamp.
///
/// This is D-14's oversampling count. Note it is NOT the contracted default budget —
/// [`effective_default_budget`] is, because the contracted equation includes the cap.
///
/// # Errors
///
/// [`ContrastiveDataError::ArithmeticOverflow`] naming the operation that overflowed.
pub fn default_epoch_budget(class_sizes: &[u64]) -> Result<u64, ContrastiveDataError> {
    let pos = positive_capacity(class_sizes)?;
    let neg = negative_capacity(class_sizes)?;
    pos.max(neg)
        .checked_mul(2)
        .ok_or_else(|| overflow("default_epoch_budget/balanced_total"))
}

/// The CONTRACTED default: `min(closed_form, hard_cap)`.
///
/// # Errors
///
/// [`ContrastiveDataError::ZeroHardCap`] for a zero cap;
/// [`ContrastiveDataError::ArithmeticOverflow`] from the closed form.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "default_epoch_budget"
)]
pub fn effective_default_budget(
    class_sizes: &[u64],
    hard_cap: u64,
) -> Result<u64, ContrastiveDataError> {
    if hard_cap == 0 {
        return Err(ContrastiveDataError::ZeroHardCap);
    }
    Ok(default_epoch_budget(class_sizes)?.min(hard_cap))
}

/// Pre-provisioned capacity check for the `unique` strategy, which does NOT ship in v1.
///
/// # Scope note, so this is not mistaken for dead API
///
/// D-11 locks the `unique` strategy's semantics *if it is present*, and the research
/// recommendation adopted by this plan ships `oversampling` only. The capacity check ships
/// anyway because D-11's hard part — a closed-form capacity that fails closed instead of
/// rejection-looping — is needed for the cap regardless, and because a typed error variant
/// no code path can raise is a claim nothing checks. It is public, documented and tested.
///
/// # Errors
///
/// [`ContrastiveDataError::BudgetExceedsCapacity`] when the budget exceeds `pos + neg`.
pub fn unique_capacity_check(class_sizes: &[u64], budget: u64) -> Result<(), ContrastiveDataError> {
    let pos = positive_capacity(class_sizes)?;
    let neg = negative_capacity(class_sizes)?;
    let capacity = pos
        .checked_add(neg)
        .ok_or_else(|| overflow("unique_capacity_check/total"))?;
    if budget > capacity {
        return Err(ContrastiveDataError::BudgetExceedsCapacity { budget, capacity });
    }
    Ok(())
}

// ===========================================================================================
// 3. Versioned policy enums and the pair configuration
// ===========================================================================================

/// The sampling strategy. v1 ships exactly one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub enum PairStrategy {
    /// Draw with replacement to the resolved budget (D-14).
    #[default]
    Oversampling,
}

impl PairStrategy {
    /// The wire name.
    ///
    /// Matched on `self` rather than returned as a constant: both of these values are
    /// written verbatim into every [`PairReplayRecord`](crate::manifest::PairReplayRecord),
    /// so a second variant added to this `#[non_exhaustive]` enum must not be able to
    /// inherit `oversampling`'s name and version silently. The `match` makes that a
    /// compile error instead.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oversampling => "oversampling",
        }
    }

    /// The version tag recorded beside the name.
    pub fn strategy_version(self) -> u32 {
        match self {
            Self::Oversampling => 1,
        }
    }
}

/// What a class with exactly one selected example does. v1 ships exactly one variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub enum SingletonPolicy {
    /// No positives (`C(1,2) = 0` falls out of the arithmetic), still present in negatives.
    #[default]
    NegativesOnly,
}

impl SingletonPolicy {
    /// The wire name.
    ///
    /// Matched on `self` for the same reason [`PairStrategy::as_str`] is: a second variant
    /// must not be able to inherit this one's wire name and version tag by default.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NegativesOnly => "negatives_only",
        }
    }

    /// The version tag recorded beside the name.
    pub fn policy_version(self) -> u32 {
        match self {
            Self::NegativesOnly => 1,
        }
    }
}

/// What the stream actually emitted, recorded in the replay record.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum EmittedKinds {
    /// Both kinds, alternating.
    Both,
    /// Positives only — a single class with two or more members.
    PositivesOnly,
    /// Negatives only — every class a singleton.
    NegativesOnly,
}

impl EmittedKinds {
    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Both => "both",
            Self::PositivesOnly => "positives_only",
            Self::NegativesOnly => "negatives_only",
        }
    }
}

/// The caller-supplied pair-stream configuration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PairConfig {
    /// The root seed every draw key is derived from.
    pub root_seed: u64,
    /// The strategy. v1: `Oversampling`.
    pub strategy: PairStrategy,
    /// The singleton policy. v1: `NegativesOnly`.
    pub singleton_policy: SingletonPolicy,
    /// `None` resolves to `min(closed_form, hard_cap)`.
    pub budget: Option<u64>,
    /// `None` resolves to [`DEFAULT_HARD_CAP`].
    pub hard_cap: Option<u64>,
}

impl PairConfig {
    /// A configuration with default policy versions and a default budget and cap.
    pub fn new(root_seed: u64) -> Self {
        Self {
            root_seed,
            strategy: PairStrategy::Oversampling,
            singleton_policy: SingletonPolicy::NegativesOnly,
            budget: None,
            hard_cap: None,
        }
    }

    /// The resolved hard cap.
    pub fn resolved_hard_cap(&self) -> u64 {
        self.hard_cap.unwrap_or(DEFAULT_HARD_CAP)
    }
}

/// Resolve the effective budget, and report whether the DEFAULT clamp engaged.
///
/// # Errors
///
/// [`ContrastiveDataError::ZeroHardCap`], [`ContrastiveDataError::ZeroBudget`],
/// [`ContrastiveDataError::BudgetExceedsHardCap`], [`ContrastiveDataError::NoPairCapacity`],
/// [`ContrastiveDataError::ArithmeticOverflow`].
/// # The cap BINDS an explicit budget — it does not clamp it
///
/// Silent clamping would keep the cap's denial-of-service role while discarding a number
/// the user typed: the run would succeed and produce a DIFFERENT dataset than the one
/// requested, with nothing red and a manifest that looks fine. That is the worst
/// reproducibility outcome available. Dropping the cap for explicit budgets would remove
/// its DoS role entirely. Failing loudly and letting the user raise `--hard-cap` keeps both
/// properties, and makes the decision visible in the command that was run.
///
/// # Ordering, and one place the contract's formula and its invariants disagree
///
/// The formula line orders `effective budget == 0 -> ZeroBudget` before
/// `pos == 0 and neg == 0 -> NoPairCapacity`, but the same equation's invariant prose says
/// "pos == 0 AND neg == 0 ... is `NoPairCapacity{...}`. There is nothing to emit." Taken
/// literally, the formula's order would report `ZeroBudget` for `[1]` under a DEFAULT
/// budget — because the closed form is then 0 — which names the request when the layout is
/// what is wrong. This function follows the invariant: a zero cap (configuration defect)
/// first, then an explicit zero or over-cap budget (request defects, wrong whatever the
/// layout is), then absent capacity, then resolution. Each rung names the thing the reader
/// has to change.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "budget_resolution"
)]
pub fn resolve_budget(
    cfg: &PairConfig,
    class_sizes: &[u64],
) -> Result<(u64, bool), ContrastiveDataError> {
    let hard_cap = cfg.resolved_hard_cap();
    if hard_cap == 0 {
        return Err(ContrastiveDataError::ZeroHardCap);
    }
    if let Some(budget) = cfg.budget {
        if budget == 0 {
            return Err(ContrastiveDataError::ZeroBudget);
        }
        if budget > hard_cap {
            return Err(ContrastiveDataError::BudgetExceedsHardCap { budget, hard_cap });
        }
    }

    let pos = positive_capacity(class_sizes)?;
    let neg = negative_capacity(class_sizes)?;
    classify_degenerate(pos, neg)?;

    match cfg.budget {
        Some(budget) => Ok((budget, false)),
        None => {
            let closed_form = default_epoch_budget(class_sizes)?;
            let resolved = closed_form.min(hard_cap);
            if resolved == 0 {
                // Unreachable while `classify_degenerate` above accepts only layouts with
                // capacity, but typed rather than asserted so a future edit to that rung
                // cannot silently produce an empty stream.
                return Err(ContrastiveDataError::ZeroBudget);
            }
            Ok((resolved, closed_form > hard_cap))
        }
    }
}

/// Which kinds a layout can emit — total over every degenerate case.
///
/// * `pos == 0 && neg == 0` — a single class of size ≤ 1, or no examples at all — is
///   [`ContrastiveDataError::NoPairCapacity`]. There is nothing to emit, and reporting a
///   budget problem here would point at the wrong file.
/// * `pos == 0 && neg > 0` — every class a singleton, the K ≈ N adversarial layout — emits
///   NEGATIVES ONLY. It is not an error: the layout is legal and its pair space is
///   non-empty. This is [`SingletonPolicy::NegativesOnly`] arriving as arithmetic.
/// * `neg == 0 && pos > 0` — one class with two or more members — emits POSITIVES ONLY.
/// * otherwise both kinds alternate.
///
/// `emitted_kinds` is recorded even in the ordinary both-kinds case, so its absence cannot
/// be confused with the ordinary case.
///
/// # Errors
///
/// [`ContrastiveDataError::NoPairCapacity`] when neither kind has any capacity.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "pair_stream_degenerate_policy"
)]
pub fn classify_degenerate(pos: u64, neg: u64) -> Result<EmittedKinds, ContrastiveDataError> {
    match (pos, neg) {
        (0, 0) => Err(ContrastiveDataError::NoPairCapacity {
            positive_capacity: 0,
            negative_capacity: 0,
        }),
        (0, _) => Ok(EmittedKinds::NegativesOnly),
        (_, 0) => Ok(EmittedKinds::PositivesOnly),
        _ => Ok(EmittedKinds::Both),
    }
}

// ===========================================================================================
// 4. Structural diagnostics (D-25, review finding F14)
// ===========================================================================================

/// STRUCTURAL counts of what a sampler RETAINS. Nothing here is self-reported "memory used".
///
/// # Why this is unconditionally public
///
/// Plan 02-08's capacity gate is an INTEGRATION test, and every file under `tests/` is its
/// own crate. A `#[cfg(test)]` introspection method on the library would therefore be
/// invisible to it, and the phase's headline boundedness gate could not be written at all.
/// The three sanctioned options were a `test-support` feature plus a self dev-dependency
/// (a known duplicate-crate hazard under feature unification), a `#[cfg(test)]` method
/// (unusable, as above), and this: a small, stable, public diagnostics object. A gate whose
/// evidence a cargo trick could silently unhook is not a gate.
///
/// # What each field counts
///
/// * `bucket_entries` — per-EXAMPLE entries the sampler retains a handle to. [`PairSampler`]
///   holds one borrowed slice per class covering every selected row, so it reports the
///   total example count. A bare [`PairLayout`] addresses an abstract index space and
///   retains none, so it reports 0. A sampler that COPIES rows reports its copy.
/// * `positive_weight_entries`, `negative_weight_entries`, `class_offset_entries` — the
///   three `O(K)` arrays. A class-PAIR design would report `~K²/2` in the negative slot,
///   which is exactly how the rejected design is detected at K ≈ N.
/// * `materialized_pairs` — pairs held in memory. An honest streaming sampler reports 0
///   forever, whatever the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerStateReport {
    /// Per-example bucket entries retained (see the type docs).
    pub bucket_entries: usize,
    /// Length of the positive class-weight array.
    pub positive_weight_entries: usize,
    /// Length of the negative class-weight array.
    pub negative_weight_entries: usize,
    /// Length of the class-offset array.
    pub class_offset_entries: usize,
    /// Pairs held in memory. Honest samplers report 0.
    pub materialized_pairs: usize,
}

impl SamplerStateReport {
    /// The single number the capacity invariant is stated over.
    pub fn total_retained_entries(&self) -> usize {
        self.bucket_entries
            .saturating_add(self.positive_weight_entries)
            .saturating_add(self.negative_weight_entries)
            .saturating_add(self.class_offset_entries)
            .saturating_add(self.materialized_pairs)
    }
}

/// A sampler that can be asked what it retains.
///
/// Implemented by [`PairLayout`] and [`PairSampler`] here, and by plan 02-08's in-band
/// `MaterializingSampler`, so the capacity gate is literally the SAME call for the honest
/// and the deliberately-wrong implementation. That is the whole point: a gate that reads a
/// different accessor for each subject is comparing two claims, not one property.
pub trait RetainedState {
    /// The structural report.
    fn state_report(&self) -> SamplerStateReport;
}

// ===========================================================================================
// 5. The O(K) layout — the sampler's entire retained state
// ===========================================================================================

/// Which half of the interleave a drawn pair came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairKind {
    /// Both endpoints in one class.
    Positive,
    /// Endpoints in two different classes.
    Negative,
}

/// One endpoint in LAYOUT space: a class index plus a member index inside that class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawEndpoint {
    /// Position of the class in ascending label order.
    pub class_index: usize,
    /// Position of the member inside that class's sorted bucket.
    pub member_index: u64,
}

/// A drawn pair in LAYOUT space, before any [`Selection`] is consulted.
///
/// Public because it is what makes the sampler's adversarial layouts reachable from OUTSIDE
/// the crate: a `Selection` always has `shots_per_class ∈ {8, 16, 32, 64}` members in every
/// class, so the K = N all-singleton layout DATA-05 must survive cannot be expressed as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawPair {
    /// Positive or negative.
    pub kind: PairKind,
    /// The first endpoint drawn.
    pub first: RawEndpoint,
    /// The second endpoint drawn.
    pub second: RawEndpoint,
}

/// The `O(K)` structural core of the pair sampler over a class-size layout.
///
/// # Why this is a separate public type
///
/// `PairSampler` borrows a [`Selection`], and a `Selection` always carries
/// `shots_per_class ∈ {8, 16, 32, 64}` rows in EVERY class. The adversarial layout DATA-05
/// has to survive — K = N, every class a singleton, under a fixed small budget — is
/// therefore not expressible as a `Selection` at all. Splitting the retained state out into
/// a layout that can be built from bare class sizes is what keeps that case reachable from
/// outside the crate, which is a requirement of plan 02-08's capacity gate rather than a
/// convenience. It is also the honest factoring: this struct IS the sampler's state, and
/// `PairSampler` is a borrowed identity mapping on top of it.
#[derive(Debug, Clone)]
pub struct PairLayout {
    offsets: Vec<u64>,
    pos_prefix: Vec<u64>,
    neg_prefix: Vec<u64>,
    total_examples: u64,
    positive_capacity: u64,
    negative_capacity: u64,
    budget: u64,
    default_was_clamped: bool,
    emitted_kinds: EmittedKinds,
    affected_singleton_classes: u64,
    strategy: PairStrategy,
    singleton_policy: SingletonPolicy,
    root_seed: u64,
    /// The five domain keys, derived once at construction.
    ///
    /// Each is a pure function of `root_seed` and a `&'static str` domain constant, so it is
    /// invariant for the life of the layout. Deriving them per draw cost a full SHA-256 each
    /// — two per positive draw and three per negative — which dominated the Philox block they
    /// feed. `select.rs` already hoists its key out of the Fisher-Yates loop; this is the same
    /// move in the one place it was missed.
    ///
    /// This is O(1) state — five 8-byte keys, independent of K and of the budget — so the
    /// D-14 capacity invariant and `state_report()` (which enumerates only structural counts)
    /// are unaffected.
    keys: PairKeys,
}

/// The five domain-separated Philox keys a [`PairLayout`] draws with.
///
/// Held as one struct rather than five fields so the derivation order stays adjacent to the
/// domain constants it mirrors, and so adding a sixth domain is one edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PairKeys {
    pos_class: DomainKey,
    pos_rank: DomainKey,
    neg_class: DomainKey,
    neg_first: DomainKey,
    neg_second: DomainKey,
}

impl PairKeys {
    /// Derive all five from the root seed. Order matches [`domains`].
    fn derive(root_seed: u64) -> Self {
        Self {
            pos_class: derive_key(root_seed, domains::PAIRS_POS_CLASS),
            pos_rank: derive_key(root_seed, domains::PAIRS_POS_RANK),
            neg_class: derive_key(root_seed, domains::PAIRS_NEG_CLASS),
            neg_first: derive_key(root_seed, domains::PAIRS_NEG_FIRST),
            neg_second: derive_key(root_seed, domains::PAIRS_NEG_SECOND),
        }
    }
}

impl PairLayout {
    /// Build the layout from bare per-class sizes, resolving the budget once.
    ///
    /// # Errors
    ///
    /// Everything [`resolve_budget`] and the capacity functions can raise.
    pub fn from_class_sizes(
        class_sizes: &[u64],
        cfg: &PairConfig,
    ) -> Result<Self, ContrastiveDataError> {
        let positive_capacity = positive_capacity(class_sizes)?;
        let negative_capacity = negative_capacity(class_sizes)?;
        let emitted_kinds = classify_degenerate(positive_capacity, negative_capacity)?;
        let (budget, default_was_clamped) = resolve_budget(cfg, class_sizes)?;

        let mut total_examples: u64 = 0;
        for &n in class_sizes {
            total_examples = total_examples
                .checked_add(n)
                .ok_or_else(|| overflow("pair_layout/total_examples"))?;
        }

        // Three O(K) arrays and nothing else. There is deliberately NO array indexed by
        // class PAIRS: that is the rejected O(K²) design, which is fine at K = 3 and fatal
        // at K ≈ N, the all-singleton layout.
        let mut offsets = Vec::with_capacity(class_sizes.len());
        let mut pos_prefix = Vec::with_capacity(class_sizes.len());
        let mut neg_prefix = Vec::with_capacity(class_sizes.len());
        let (mut running, mut pos_running, mut neg_running) = (0_u64, 0_u64, 0_u64);
        for &n in class_sizes {
            offsets.push(running);
            pos_running += n * n.saturating_sub(1) / 2;
            pos_prefix.push(pos_running);
            // w_c = n_c · (S − n_c). Σ_c w_c == 2 · negative_capacity, which is why the
            // running total is checked: the capacity itself can fit while its double does
            // not.
            let weight = n
                .checked_mul(total_examples - n)
                .ok_or_else(|| overflow("pair_layout/negative_class_weight"))?;
            neg_running = neg_running
                .checked_add(weight)
                .ok_or_else(|| overflow("pair_layout/negative_weight_total"))?;
            neg_prefix.push(neg_running);
            running += n;
        }

        Ok(Self {
            offsets,
            pos_prefix,
            neg_prefix,
            total_examples,
            positive_capacity,
            negative_capacity,
            budget,
            default_was_clamped,
            emitted_kinds,
            affected_singleton_classes: class_sizes.iter().filter(|n| **n == 1).count() as u64,
            strategy: cfg.strategy,
            singleton_policy: cfg.singleton_policy,
            root_seed: cfg.root_seed,
            keys: PairKeys::derive(cfg.root_seed),
        })
    }

    /// The resolved effective budget.
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// Whether the DEFAULT budget was clamped by the hard cap.
    pub fn default_was_clamped(&self) -> bool {
        self.default_was_clamped
    }

    /// Which kinds this layout emits.
    pub fn emitted_kinds(&self) -> EmittedKinds {
        self.emitted_kinds
    }

    /// How many classes hold exactly one member.
    pub fn affected_singleton_classes(&self) -> u64 {
        self.affected_singleton_classes
    }

    /// Number of classes, `K`.
    pub fn class_count(&self) -> usize {
        self.offsets.len()
    }

    /// Total examples across all classes, `S`.
    pub fn total_examples(&self) -> u64 {
        self.total_examples
    }

    /// `Σ_k C(n_k, 2)`.
    pub fn positive_capacity(&self) -> u64 {
        self.positive_capacity
    }

    /// `Σ_{j<k} n_j · n_k`.
    pub fn negative_capacity(&self) -> u64 {
        self.negative_capacity
    }

    /// The configured strategy.
    pub fn strategy(&self) -> PairStrategy {
        self.strategy
    }

    /// The configured singleton policy.
    pub fn singleton_policy(&self) -> SingletonPolicy {
        self.singleton_policy
    }

    /// The root seed every draw key derives from.
    pub fn root_seed(&self) -> u64 {
        self.root_seed
    }

    /// Members in class `class_index`, or 0 for an out-of-range index.
    pub fn class_size(&self, class_index: usize) -> u64 {
        let Some(start) = self.offsets.get(class_index) else {
            return 0;
        };
        let end = self
            .offsets
            .get(class_index + 1)
            .copied()
            .unwrap_or(self.total_examples);
        end - start
    }

    /// The pair at `ordinal`, in layout space.
    ///
    /// # The interleave
    ///
    /// When both kinds are available, ordinal `2t` is positive draw `t` and `2t + 1` is
    /// negative draw `t` — matching the pinned reference's `zip_longest` and giving strict
    /// balance at every even prefix. When only one kind is available, ordinal `m` is draw
    /// `m` of that kind.
    ///
    /// # One stream per `(selection, seed, policy, budget)`
    ///
    /// The stream is EPOCH-INDEPENDENT: consumers advance by offset rather than by epoch,
    /// which is also what makes it shardable and resumable without replaying the prefix. A
    /// consumer that wants a different order across epochs adds `epoch` to the domain
    /// string — a versioned, non-breaking policy extension (Open Question 3 / Assumption
    /// A4), documented here rather than left to be discovered.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::OrdinalOutOfRange`] at or beyond the resolved budget.
    #[provable_contracts_macros::contract("contrastive-pair-protocol-v1", equation = "pair_stream")]
    pub fn raw_pair_at(&self, ordinal: u64) -> Result<RawPair, ContrastiveDataError> {
        if ordinal >= self.budget {
            return Err(ContrastiveDataError::OrdinalOutOfRange {
                ordinal,
                budget: self.budget,
            });
        }
        // PRECONDITIONS OF THE INNER PATH, stated because they are what makes it total:
        // `ordinal < budget` (checked immediately above), and every capacity, weight and
        // offset was computed with checked arithmetic in `from_class_sizes`. `iter_from`
        // relies on exactly this to yield infallibly.
        match self.emitted_kinds {
            EmittedKinds::Both => {
                if ordinal % 2 == 0 {
                    self.positive_draw(ordinal / 2)
                } else {
                    self.negative_draw(ordinal / 2)
                }
            }
            EmittedKinds::PositivesOnly => self.positive_draw(ordinal),
            EmittedKinds::NegativesOnly => self.negative_draw(ordinal),
        }
    }

    /// Positive draw `t`: class by `C(n_k, 2)` weight, then triangular unranking.
    fn positive_draw(&self, t: u64) -> Result<RawPair, ContrastiveDataError> {
        let total = nonzero(self.positive_capacity, "positive_draw/total_weight")?;
        let target = bounded(&self.keys.pos_class, 0, t, total);
        let class_index = self.pos_prefix.partition_point(|prefix| *prefix <= target);

        let n = self.class_size(class_index);
        let capacity = nonzero(n * n.saturating_sub(1) / 2, "positive_draw/class_capacity")?;
        let rank = bounded(&self.keys.pos_rank, 0, t, capacity);
        let (first, second) = triangular_unrank(n, rank);

        Ok(RawPair {
            kind: PairKind::Positive,
            first: RawEndpoint {
                class_index,
                member_index: first,
            },
            second: RawEndpoint {
                class_index,
                member_index: second,
            },
        })
    }

    /// Negative draw `t` — `O(K)`, NOT `O(K²)`.
    ///
    /// Draw the first class `j` against the per-class weights `w_c = n_c · (S − n_c)`, draw
    /// the first endpoint uniformly inside `j`, then draw `u` uniformly in `[0, S − n_j)`
    /// and map it to a global bucket index that SKIPS class `j`'s contiguous block
    /// (`u < offset_j ? u : u + n_j`), resolving the second class by binary search over the
    /// `O(K)` offset array.
    ///
    /// # Why this is equivalent to D-14's `n_j · n_k` class-pair weights
    ///
    /// `P(class j) = w_j / Σ_c w_c`, and given `j` the first endpoint is uniform over its
    /// `n_j` members while the second is uniform over the `S − n_j` members outside it. So
    /// every ORDERED cross-class endpoint pair has probability
    /// `[n_j(S − n_j) / Σw] · (1/n_j) · (1/(S − n_j)) = 1 / Σw` — uniform. There are
    /// `Σ_c n_c(S − n_c) = 2 · Σ_{j<k} n_j n_k` such ordered pairs, so each UNORDERED class
    /// pair `{j, k}` receives weight proportional to `2 · n_j · n_k`, i.e. exactly the
    /// `n_j · n_k` weight D-14 specifies. The rewrite is a representation change, not a
    /// semantics change; `negative_draw_marginals_match_the_n_j_times_n_k_class_pair_weights`
    /// measures it rather than restating this algebra.
    ///
    /// THE REJECTED ALTERNATIVE, named so it is not reinvented: prefix sums over enumerated
    /// unordered class pairs. That is `O(K²)` retained state — fine at K = 3, fatal at
    /// K ≈ N. It is forbidden here.
    fn negative_draw(&self, t: u64) -> Result<RawPair, ContrastiveDataError> {
        let total_weight = self.neg_prefix.last().copied().unwrap_or_default();
        let total = nonzero(total_weight, "negative_draw/total_weight")?;
        let target = bounded(&self.keys.neg_class, 0, t, total);
        let first_class = self.neg_prefix.partition_point(|prefix| *prefix <= target);

        let n_j = nonzero(
            self.class_size(first_class),
            "negative_draw/first_class_size",
        )?;
        let member_index = bounded(&self.keys.neg_first, 0, t, n_j);

        let outside = nonzero(
            self.total_examples - n_j.get(),
            "negative_draw/outside_first_class",
        )?;
        let drawn = bounded(&self.keys.neg_second, 0, t, outside);
        let offset_j = self.offsets.get(first_class).copied().unwrap_or_default();
        let global = if drawn < offset_j {
            drawn
        } else {
            drawn + n_j.get()
        };
        let second_class = self.class_of_global(global);

        Ok(RawPair {
            kind: PairKind::Negative,
            first: RawEndpoint {
                class_index: first_class,
                member_index,
            },
            second: RawEndpoint {
                class_index: second_class,
                member_index: global - self.offsets.get(second_class).copied().unwrap_or_default(),
            },
        })
    }

    /// The LAST class whose offset is at or below `global` — which is what skips empty
    /// classes, since they share the offset of the class after them.
    fn class_of_global(&self, global: u64) -> usize {
        self.offsets
            .partition_point(|offset| *offset <= global)
            .saturating_sub(1)
    }
}

/// A non-zero bound, or a typed error naming which draw could not be made.
fn nonzero(value: u64, operation: &str) -> Result<NonZeroU64, ContrastiveDataError> {
    NonZeroU64::new(value).ok_or_else(|| overflow(operation))
}

/// Pairs of a class of `n` whose first member is strictly below `i`: `i·(2n−1−i)/2`.
///
/// Computed in `u128` because the intermediate `i·(2n−1−i)` reaches `2·C(n,2)`, which can
/// exceed `u64` while the result cannot.
fn triangular_prefix(n: u64, i: u64) -> u64 {
    let doubled = u128::from(i) * (2 * u128::from(n) - 1 - u128::from(i));
    (doubled / 2) as u64
}

/// Map `rank ∈ [0, C(n,2))` to the ordered member pair `(i, j)` with `i < j < n`.
///
/// Integer binary search, never `sqrt`: a floating-point closed form would make the pair
/// identities host-fragile for exactly the reason `next_f32` draws are forbidden.
/// Precondition: `n >= 2` and `rank < C(n, 2)`, both guaranteed by the weighted class draw.
fn triangular_unrank(n: u64, rank: u64) -> (u64, u64) {
    let (mut lo, mut hi) = (0_u64, n.saturating_sub(2));
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if triangular_prefix(n, mid) <= rank {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    (lo, lo + 1 + (rank - triangular_prefix(n, lo)))
}

impl RetainedState for PairLayout {
    fn state_report(&self) -> SamplerStateReport {
        SamplerStateReport {
            bucket_entries: 0,
            positive_weight_entries: self.pos_prefix.len(),
            negative_weight_entries: self.neg_prefix.len(),
            class_offset_entries: self.offsets.len(),
            materialized_pairs: 0,
        }
    }
}

// ===========================================================================================
// 6. The borrowed sampler
// ===========================================================================================

/// The bounded, deterministic pair stream over ONE [`Selection`].
///
/// Borrowing the selection is the mechanism, not a detail: a [`SelectedId`] can only be
/// obtained from the selection that produced it, so an endpoint naming a row that was never
/// selected is not rejected — it is inexpressible (`split_span_fail_closed`, structural
/// half). The typed half, for untrusted replayed bytes, is [`validate_pair_records`].
#[derive(Debug)]
pub struct PairSampler<'a> {
    selection: &'a Selection,
    layout: PairLayout,
    class_ids: Vec<&'a [SelectedId]>,
}

impl<'a> PairSampler<'a> {
    /// Build a sampler over a completed selection.
    ///
    /// # Errors
    ///
    /// Everything [`PairLayout::from_class_sizes`] can raise.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "singleton_policy"
    )]
    pub fn new(sel: &'a Selection, cfg: &PairConfig) -> Result<Self, ContrastiveDataError> {
        let sizes: Vec<u64> = sel.class_sizes().iter().map(|(_, n)| *n).collect();
        let layout = PairLayout::from_class_sizes(&sizes, cfg)?;
        let class_ids: Vec<&'a [SelectedId]> = sel
            .class_sizes()
            .iter()
            .map(|(label, _)| sel.ids_in_class(*label))
            .collect();
        Ok(Self {
            selection: sel,
            layout,
            class_ids,
        })
    }

    /// The structural layout underneath.
    pub fn layout(&self) -> &PairLayout {
        &self.layout
    }

    /// The selection this sampler is bound to.
    pub fn selection(&self) -> &'a Selection {
        self.selection
    }

    /// The resolved effective budget.
    pub fn budget(&self) -> u64 {
        self.layout.budget()
    }

    // `emitted_kinds`, `affected_singleton_classes` and `default_was_clamped` are NOT
    // forwarded here. They are layout properties and production already reads them through
    // `sampler.layout()` (see `PairReplayRecord::from_sampler`); forwarding copies existed
    // with test-only callers, which is one public surface per property too many. `budget()`
    // below stays because the sampler genuinely resolves it.

    /// The pair at `ordinal`. Pure: the same ordinal always yields the same pair.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::OrdinalOutOfRange`] at or beyond the resolved budget.
    pub fn pair_at(&self, ordinal: u64) -> Result<LabeledPair, ContrastiveDataError> {
        let raw = self.layout.raw_pair_at(ordinal)?;
        let a = self.endpoint(raw.first)?;
        let b = self.endpoint(raw.second)?;
        let pair = CanonicalPair::new(a, b)?;
        Ok(LabeledPair {
            target: derive_target(self.selection, &pair),
            pair,
        })
    }

    /// A cursor from `offset` to the budget.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::OrdinalOutOfRange`] when `offset > budget`.
    pub fn iter_from(&self, offset: u64) -> Result<PairIter<'_, 'a>, ContrastiveDataError> {
        let budget = self.layout.budget();
        if offset > budget {
            return Err(ContrastiveDataError::OrdinalOutOfRange {
                ordinal: offset,
                budget,
            });
        }
        Ok(PairIter {
            sampler: self,
            cursor: offset,
            budget,
        })
    }

    /// Resolve a layout-space endpoint to a selected ordinal.
    fn endpoint(&self, raw: RawEndpoint) -> Result<SelectedId, ContrastiveDataError> {
        let bucket =
            self.class_ids
                .get(raw.class_index)
                .ok_or(ContrastiveDataError::OrdinalOutOfRange {
                    ordinal: raw.class_index as u64,
                    budget: self.class_ids.len() as u64,
                })?;
        bucket.get(raw.member_index as usize).copied().ok_or(
            ContrastiveDataError::OrdinalOutOfRange {
                ordinal: raw.member_index,
                budget: bucket.len() as u64,
            },
        )
    }
}

impl RetainedState for PairSampler<'_> {
    fn state_report(&self) -> SamplerStateReport {
        SamplerStateReport {
            bucket_entries: self.class_ids.iter().map(|ids| ids.len()).sum(),
            ..self.layout.state_report()
        }
    }
}

/// A borrowed cursor over one sampler.
#[derive(Debug)]
pub struct PairIter<'s, 'a> {
    sampler: &'s PairSampler<'a>,
    cursor: u64,
    budget: u64,
}

impl Iterator for PairIter<'_, '_> {
    type Item = LabeledPair;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.budget {
            return None;
        }
        let item = self.sampler.pair_at(self.cursor).expect(
            "iter_from validated offset <= budget and PairSampler::new validated every \
             capacity and bucket size, so pair_at below the budget is total",
        );
        self.cursor += 1;
        Some(item)
    }
}

/// The 1.0 / 0.0 target, DERIVED from the endpoints' classes.
fn derive_target(sel: &Selection, pair: &CanonicalPair) -> f32 {
    if sel.label_of(pair.lo()) == sel.label_of(pair.hi()) {
        1.0
    } else {
        0.0
    }
}

// ===========================================================================================
// 7. The untrusted-input boundary (review finding F13)
// ===========================================================================================

/// A pair as it arrives from replayed or dumped bytes — NOT a [`LabeledPair`].
///
/// A trusted pair type with a private constructor CANNOT REPRESENT poisoned input, so
/// "validating at the boundary" is impossible without an untrusted representation to
/// validate FROM. That is why this DTO exists rather than deserializing straight into
/// [`CanonicalPair`], and it is what makes plan 02-08's endpoint-poisoning test
/// constructible at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UntrustedPairRecord {
    /// The lower endpoint's row identifier, as written.
    pub lo: String,
    /// The upper endpoint's row identifier, as written.
    pub hi: String,
    /// The target the bytes CLAIM. Checked against the derived one, never trusted.
    pub target: f32,
}

/// Parse a JSONL pair dump into untrusted records.
///
/// # Errors
///
/// [`ContrastiveDataError::MalformedRow`] naming the line index and the parser message.
pub fn parse_pair_dump(bytes: &[u8]) -> Result<Vec<UntrustedPairRecord>, ContrastiveDataError> {
    let text = core::str::from_utf8(bytes).map_err(|error| ContrastiveDataError::MalformedRow {
        split: PAIR_DUMP_SPLIT.to_string(),
        index: 0,
        reason: error.to_string(),
    })?;
    let mut records = Vec::new();
    for (index, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let record: UntrustedPairRecord =
            serde_json::from_str(line).map_err(|error| ContrastiveDataError::MalformedRow {
                split: PAIR_DUMP_SPLIT.to_string(),
                index,
                reason: error.to_string(),
            })?;
        records.push(record);
    }
    Ok(records)
}

/// The `split` name every pair-dump parse error carries.
pub(crate) const PAIR_DUMP_SPLIT: &str = "pair_dump";

/// Both endpoints of an untrusted record, resolved against the selection.
///
/// This is D-27's typed arm: for in-process construction the span guarantee is STRUCTURAL
/// (a [`SelectedId`] cannot name an unselected row), but a deserialized record can name
/// anything at all, so membership becomes a typed error naming the offending identifier.
/// "Leakage detected" is not diagnosable; `pair endpoint "validation:3" is not in the
/// selection` is.
///
/// `found_in` is deliberately `"unknown"`: this validator's universe is the Selection, so
/// it can prove ABSENCE but cannot locate the identifier. Claiming to know where a rejected
/// id came from would be a nicer message than the evidence supports.
///
/// The `split_span_fail_closed` `#[contract]` annotation sits on the PUBLIC
/// [`validate_pair_records`] that calls this, not here — see that function's docs for why.
fn assert_endpoints_in_selection(
    rec: &UntrustedPairRecord,
    sel: &Selection,
) -> Result<(SelectedId, SelectedId), ContrastiveDataError> {
    let resolve = |id: &str| {
        sel.selected_id(id)
            .ok_or_else(|| ContrastiveDataError::EndpointNotInSelection {
                id: id.to_string(),
                found_in: "unknown".to_string(),
            })
    };
    Ok((resolve(&rec.lo)?, resolve(&rec.hi)?))
}

/// Validate untrusted pair records against the selection they claim to belong to.
///
/// Validation order: membership of BOTH endpoints
/// ([`assert_endpoints_in_selection`], `split_span_fail_closed`), then distinctness and
/// canonical ordering through [`CanonicalPair::new`], then agreement of the declared target
/// with the one DERIVED from the endpoints' classes.
///
/// # The target check is not redundant with membership
///
/// A record naming two legitimate same-class endpoints while declaring target `0.0` is a
/// semantically poisoned pair made entirely of valid parts. A gate that only checked
/// membership would pass it.
///
/// # A swapped record is normalized, not refused
///
/// Orientation carries no information — that is the entire point of canonicalization
/// (D-12) — so `(b, a)` validates to the same [`LabeledPair`] as `(a, b)`. Refusing it
/// would be rejecting a spelling, not a threat.
///
/// # Two contract equations, stacked on one function
///
/// `untrusted_pair_ingest` is the whole ladder; `split_span_fail_closed` is its membership
/// arm, implemented by [`assert_endpoints_in_selection`] just above. The macro DOES accept
/// stacked attributes (verified by compiling both forms), so both annotations sit here — on
/// the PUBLIC entry point — rather than one of them on a private helper: plan 02-08's
/// binding registry and its `validate_pair_records` key-link both need an auditable public
/// path, and a binding that names a private function is harder to check from outside.
/// `split_span_fail_closed`'s other, STRUCTURAL arm needs no annotation at all, because it
/// is a type ([`SelectedId`]'s private constructor) rather than a function.
///
/// # Errors
///
/// [`ContrastiveDataError::EndpointNotInSelection`], [`ContrastiveDataError::SelfPair`],
/// [`ContrastiveDataError::PairTargetMismatch`].
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "untrusted_pair_ingest"
)]
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "split_span_fail_closed"
)]
pub fn validate_pair_records(
    recs: &[UntrustedPairRecord],
    sel: &Selection,
) -> Result<Vec<LabeledPair>, ContrastiveDataError> {
    let mut validated = Vec::with_capacity(recs.len());
    for rec in recs {
        let (lo, hi) = assert_endpoints_in_selection(rec, sel)?;
        let pair = CanonicalPair::new(lo, hi)?;
        let derived_target = derive_target(sel, &pair);
        if rec.target != derived_target {
            return Err(ContrastiveDataError::PairTargetMismatch {
                lo: rec.lo.clone(),
                hi: rec.hi.clone(),
                declared_target: rec.target,
                derived_target,
            });
        }
        validated.push(LabeledPair {
            pair,
            target: derived_target,
        });
    }
    Ok(validated)
}

#[cfg(test)]
mod pair_tests {
    use super::{
        classify_degenerate, default_epoch_budget, effective_default_budget, negative_capacity,
        positive_capacity, resolve_budget, unique_capacity_check, CanonicalPair, EmittedKinds,
        PairConfig, PairStrategy, SingletonPolicy, DEFAULT_HARD_CAP, DEGENERATE_POLICY_VERSION,
    };
    use crate::error::ContrastiveDataError;
    use crate::select::{test_corpus, SelectedId};

    /// Twenty-four selected ordinals (8 shots × 3 classes) in selection order.
    fn ordinals() -> Vec<SelectedId> {
        let (selection, _) = test_corpus::fresh_selection(12, 13, 8);
        selection
            .examples()
            .iter()
            .map(|row| {
                selection
                    .selected_id(&row.id)
                    .expect("every selected example resolves to its own ordinal")
            })
            .collect()
    }

    #[test]
    fn canonical_pair_is_orientation_free_and_rejects_self_pairs() {
        let ids = ordinals();
        let (a, b) = (ids[2], ids[9]);

        let forward = CanonicalPair::new(a, b).expect("distinct endpoints pair");
        let backward = CanonicalPair::new(b, a).expect("distinct endpoints pair");
        assert_eq!(forward, backward, "orientation carries no information");
        assert!(forward.lo() < forward.hi(), "lo < hi always");
        assert_eq!(forward.lo(), a);
        assert_eq!(forward.hi(), b);

        match CanonicalPair::new(a, a).expect_err("a self-pair must be refused") {
            ContrastiveDataError::SelfPair { id } => assert_eq!(id, u64::from(a.ordinal())),
            other => panic!("expected SelfPair, got {other:?}"),
        }
    }

    #[test]
    fn positive_capacity_matches_the_contracted_closed_form() {
        assert_eq!(positive_capacity(&[8, 4, 8]).expect("no overflow"), 62);
        assert_eq!(positive_capacity(&[8, 8, 8]).expect("no overflow"), 84);
        assert_eq!(positive_capacity(&[64, 64, 64]).expect("no overflow"), 6048);
        assert_eq!(positive_capacity(&[6]).expect("no overflow"), 15);
        assert_eq!(positive_capacity(&[]).expect("no overflow"), 0);
    }

    /// A singleton class contributes ZERO positive capacity; its four-member neighbour
    /// still yields six. That is what makes `NegativesOnly` arithmetic rather than a
    /// special case.
    #[test]
    fn positive_capacity_of_a_singleton_class_is_zero() {
        assert_eq!(positive_capacity(&[4, 1]).expect("no overflow"), 6);
        assert_eq!(positive_capacity(&[1]).expect("no overflow"), 0);
        assert_eq!(positive_capacity(&[1; 32]).expect("no overflow"), 0);
    }

    #[test]
    fn negative_capacity_matches_the_contracted_closed_form() {
        assert_eq!(negative_capacity(&[8, 4, 8]).expect("no overflow"), 128);
        assert_eq!(negative_capacity(&[8, 8, 8]).expect("no overflow"), 192);
        assert_eq!(
            negative_capacity(&[64, 64, 64]).expect("no overflow"),
            12288
        );
        assert_eq!(negative_capacity(&[4, 1]).expect("no overflow"), 4);
        assert_eq!(negative_capacity(&[6]).expect("no overflow"), 0);
        assert_eq!(negative_capacity(&[1; 32]).expect("no overflow"), 496);
    }

    /// Two derivations of the same quantity must agree. The shipped evaluation order is
    /// the running-prefix form, which overflows strictly later than `(S² − Σn²)/2`; this
    /// pins them together wherever the second one is computable at all.
    #[test]
    fn negative_capacity_agrees_with_the_sum_of_squares_derivation() {
        for layout in [
            vec![8_u64, 4, 8],
            vec![8, 8, 8],
            vec![64, 64, 64],
            vec![4, 1],
            vec![1; 32],
            vec![3, 5, 7],
            vec![0, 9, 0, 2],
        ] {
            let s: u64 = layout.iter().sum();
            let sum_squares: u64 = layout.iter().map(|n| n * n).sum();
            let via_squares = (s * s - sum_squares) / 2;
            assert_eq!(
                negative_capacity(&layout).expect("no overflow"),
                via_squares,
                "the two derivations disagree on {layout:?}"
            );
        }
    }

    #[test]
    fn default_epoch_budget_matches_the_worked_values() {
        assert_eq!(default_epoch_budget(&[8, 4, 8]).expect("no overflow"), 256);
        assert_eq!(default_epoch_budget(&[8, 8, 8]).expect("no overflow"), 384);
        assert_eq!(
            default_epoch_budget(&[64, 64, 64]).expect("no overflow"),
            24_576
        );
        assert_eq!(default_epoch_budget(&[4, 1]).expect("no overflow"), 12);
        assert_eq!(default_epoch_budget(&[1; 32]).expect("no overflow"), 992);
        assert_eq!(default_epoch_budget(&[6]).expect("no overflow"), 30);
    }

    #[test]
    fn effective_default_budget_leaves_contracted_layouts_unclamped() {
        assert_eq!(
            effective_default_budget(&[64, 64, 64], DEFAULT_HARD_CAP).expect("no overflow"),
            24_576
        );
        assert_eq!(
            effective_default_budget(&[8, 8, 8], DEFAULT_HARD_CAP).expect("no overflow"),
            384
        );
        assert_eq!(DEFAULT_HARD_CAP, 1_048_576);
    }

    #[test]
    fn effective_default_budget_clamp_engages_under_a_small_cap() {
        assert_eq!(
            effective_default_budget(&[64, 64, 64], 10_000).expect("no overflow"),
            10_000
        );
        assert!(matches!(
            effective_default_budget(&[8, 8, 8], 0),
            Err(ContrastiveDataError::ZeroHardCap)
        ));
    }

    #[test]
    fn capacity_functions_return_arithmetic_overflow_rather_than_wrapping() {
        let named = |result: Result<u64, ContrastiveDataError>, needle: &str| match result {
            Err(ContrastiveDataError::ArithmeticOverflow { operation }) => {
                assert!(
                    operation.contains(needle),
                    "operation {operation:?} does not name {needle:?}"
                );
            }
            other => panic!("expected ArithmeticOverflow naming {needle}, got {other:?}"),
        };

        named(positive_capacity(&[u64::MAX]), "positive_capacity");
        named(
            positive_capacity(&[u64::MAX - 1, u64::MAX - 1]),
            "positive_capacity",
        );
        named(negative_capacity(&[u64::MAX, 2]), "negative_capacity");
        named(
            negative_capacity(&[u64::MAX / 2, u64::MAX / 2]),
            "negative_capacity",
        );
        // pos and neg BOTH fit; only the doubling overflows, so the error must name the
        // balancing step rather than one of the capacities.
        named(
            default_epoch_budget(&[4_294_967_296, 65_537]),
            "default_epoch_budget",
        );
        named(
            effective_default_budget(&[4_294_967_296, 65_537], DEFAULT_HARD_CAP),
            "default_epoch_budget",
        );
    }

    // -- budget resolution: one named test per branch (review finding F3) ------------------

    fn cfg_with(budget: Option<u64>, hard_cap: Option<u64>) -> PairConfig {
        PairConfig {
            budget,
            hard_cap,
            ..PairConfig::new(13)
        }
    }

    #[test]
    fn budget_resolution_rejects_a_zero_hard_cap() {
        assert!(matches!(
            resolve_budget(&cfg_with(None, Some(0)), &[8, 8, 8]),
            Err(ContrastiveDataError::ZeroHardCap)
        ));
        // A configuration defect outranks a request defect: it is fixed in a different
        // place, so naming the cap first is what points at the file to edit.
        assert!(matches!(
            resolve_budget(&cfg_with(Some(0), Some(0)), &[8, 8, 8]),
            Err(ContrastiveDataError::ZeroHardCap)
        ));
    }

    #[test]
    fn budget_resolution_rejects_a_zero_explicit_budget() {
        assert!(matches!(
            resolve_budget(&cfg_with(Some(0), None), &[8, 8, 8]),
            Err(ContrastiveDataError::ZeroBudget)
        ));
    }

    #[test]
    fn budget_resolution_rejects_an_explicit_budget_above_the_hard_cap() {
        match resolve_budget(&cfg_with(Some(20_000), Some(10_000)), &[64, 64, 64]) {
            Err(ContrastiveDataError::BudgetExceedsHardCap { budget, hard_cap }) => {
                assert_eq!(budget, 20_000);
                assert_eq!(hard_cap, 10_000);
            }
            other => panic!("expected BudgetExceedsHardCap naming both numbers, got {other:?}"),
        }
    }

    #[test]
    fn budget_resolution_accepts_an_explicit_budget_at_or_below_the_hard_cap() {
        assert_eq!(
            resolve_budget(&cfg_with(Some(10_000), Some(10_000)), &[64, 64, 64])
                .expect("at the cap is accepted"),
            (10_000, false)
        );
        assert_eq!(
            resolve_budget(&cfg_with(Some(7), None), &[8, 8, 8]).expect("below the cap"),
            (7, false)
        );
    }

    #[test]
    fn budget_resolution_default_clamps_and_reports_whether_it_engaged() {
        assert_eq!(
            resolve_budget(&cfg_with(None, None), &[64, 64, 64]).expect("closed form"),
            (24_576, false)
        );
        assert_eq!(
            resolve_budget(&cfg_with(None, Some(10_000)), &[64, 64, 64]).expect("clamped"),
            (10_000, true)
        );
    }

    #[test]
    fn budget_resolution_refuses_a_layout_with_no_pair_capacity() {
        match resolve_budget(&cfg_with(None, None), &[1]) {
            Err(ContrastiveDataError::NoPairCapacity {
                positive_capacity: pos,
                negative_capacity: neg,
            }) => {
                assert_eq!((pos, neg), (0, 0));
            }
            other => panic!("expected NoPairCapacity, got {other:?}"),
        }
        assert!(matches!(
            resolve_budget(&cfg_with(Some(5), None), &[]),
            Err(ContrastiveDataError::NoPairCapacity { .. })
        ));
    }

    // -- degenerate policy: one named test per branch (review finding F2) ------------------

    #[test]
    fn degenerate_policy_no_capacity_of_either_kind_is_a_typed_error() {
        match classify_degenerate(0, 0) {
            Err(ContrastiveDataError::NoPairCapacity {
                positive_capacity: pos,
                negative_capacity: neg,
            }) => assert_eq!((pos, neg), (0, 0)),
            other => panic!("expected NoPairCapacity, got {other:?}"),
        }
    }

    #[test]
    fn degenerate_policy_all_singletons_emits_negatives_only() {
        let pos = positive_capacity(&[1; 32]).expect("no overflow");
        let neg = negative_capacity(&[1; 32]).expect("no overflow");
        assert_eq!((pos, neg), (0, 496));
        assert_eq!(
            classify_degenerate(pos, neg).expect("a legal layout"),
            EmittedKinds::NegativesOnly
        );
    }

    #[test]
    fn degenerate_policy_one_class_emits_positives_only() {
        let pos = positive_capacity(&[6]).expect("no overflow");
        let neg = negative_capacity(&[6]).expect("no overflow");
        assert_eq!((pos, neg), (15, 0));
        assert_eq!(
            classify_degenerate(pos, neg).expect("a legal layout"),
            EmittedKinds::PositivesOnly
        );
    }

    #[test]
    fn degenerate_policy_both_kinds_alternate() {
        assert_eq!(
            classify_degenerate(62, 128).expect("a legal layout"),
            EmittedKinds::Both
        );
        assert_eq!(EmittedKinds::Both.as_str(), "both");
        assert_eq!(EmittedKinds::PositivesOnly.as_str(), "positives_only");
        assert_eq!(EmittedKinds::NegativesOnly.as_str(), "negatives_only");
    }

    #[test]
    fn strategy_and_policy_version_tags_are_one_and_render_snake_case() {
        assert_eq!(PairStrategy::Oversampling.as_str(), "oversampling");
        assert_eq!(PairStrategy::Oversampling.strategy_version(), 1);
        assert_eq!(SingletonPolicy::NegativesOnly.as_str(), "negatives_only");
        assert_eq!(SingletonPolicy::NegativesOnly.policy_version(), 1);
        assert_eq!(DEGENERATE_POLICY_VERSION, 1);
    }

    /// D-11's capacity check ships even though the `unique` strategy does not, so the
    /// error variant is reachable and tested rather than dead API.
    #[test]
    fn unique_capacity_check_is_reachable_and_names_both_numbers() {
        unique_capacity_check(&[8, 4, 8], 190).expect("190 <= 62 + 128");
        match unique_capacity_check(&[8, 4, 8], 191) {
            Err(ContrastiveDataError::BudgetExceedsCapacity { budget, capacity }) => {
                assert_eq!(budget, 191);
                assert_eq!(capacity, 190);
            }
            other => panic!("expected BudgetExceedsCapacity, got {other:?}"),
        }
    }

    /// The clamp flag at the exact boundary (plan 02-08 mutation triage).
    ///
    /// `cargo mutants` found `closed_form > hard_cap` -> `>=` surviving: at
    /// `closed_form == hard_cap` the `min` picks the same number either way, so nothing
    /// downstream changes EXCEPT the reported flag. That flag is copied verbatim into the
    /// replay record, and a record claiming `default_was_clamped: true` for a run that was
    /// not clamped is a manifest asserting the stream differs from the request — the same
    /// class of quiet untruth the binding hard cap exists to prevent.
    #[test]
    fn a_default_budget_exactly_equal_to_the_hard_cap_is_not_reported_as_clamped() {
        let sizes = [8_u64, 8, 8];
        let closed = default_epoch_budget(&sizes).expect("[8,8,8] never overflows");
        assert_eq!(closed, 384, "2 * max(84, 192)");

        let at_the_boundary = resolve_budget(
            &PairConfig {
                hard_cap: Some(closed),
                ..PairConfig::new(13)
            },
            &sizes,
        )
        .expect("a cap equal to the closed form resolves");
        assert_eq!(at_the_boundary, (closed, false), "== is NOT a clamp");

        // One below, and it genuinely is — without this the assertion above would also
        // hold against a resolver that never reports a clamp at all.
        let one_below = resolve_budget(
            &PairConfig {
                hard_cap: Some(closed - 1),
                ..PairConfig::new(13)
            },
            &sizes,
        )
        .expect("a cap one below the closed form resolves");
        assert_eq!(
            one_below,
            (closed - 1, true),
            "one below, the clamp engages"
        );
    }
}

#[cfg(test)]
mod pair_stream_tests {
    //! The streaming sampler: interleave, purity, structure, and the `O(K)` evidence.
    //!
    //! Layouts other than `K × shots` are exercised through [`PairLayout`] rather than
    //! through a `Selection`, because a `Selection` always carries the same
    //! `shots_per_class ∈ {8, 16, 32, 64}` in every class — so `[4, 1]`, `[6]` and the
    //! K = N `[1; 32]` layout the DATA-05 bound must survive cannot be built as one.

    use super::{
        classify_degenerate, negative_capacity, parse_pair_dump, positive_capacity,
        validate_pair_records, EmittedKinds, LabeledPair, PairConfig, PairKind, PairLayout,
        PairSampler, RawPair, RetainedState, UntrustedPairRecord, DEFAULT_HARD_CAP,
    };
    use crate::error::ContrastiveDataError;
    use crate::select::{test_corpus, Selection};
    use std::collections::BTreeMap;

    fn layout(sizes: &[u64], cfg: &PairConfig) -> PairLayout {
        PairLayout::from_class_sizes(sizes, cfg).expect("a legal layout builds")
    }

    fn default_layout(sizes: &[u64]) -> PairLayout {
        layout(sizes, &PairConfig::new(13))
    }

    fn capped_layout(sizes: &[u64], budget: u64) -> PairLayout {
        layout(
            sizes,
            &PairConfig {
                budget: Some(budget),
                ..PairConfig::new(13)
            },
        )
    }

    fn all_raw(layout: &PairLayout) -> Vec<RawPair> {
        (0..layout.budget())
            .map(|ordinal| {
                layout
                    .raw_pair_at(ordinal)
                    .expect("every ordinal below the budget resolves")
            })
            .collect()
    }

    fn selection_8_shots() -> Selection {
        test_corpus::fresh_selection(12, 13, 8).0
    }

    // -- budget wiring ---------------------------------------------------------------------

    #[test]
    fn pair_sampler_default_budget_is_the_effective_default_for_its_selection() {
        let selection = selection_8_shots();
        let sampler =
            PairSampler::new(&selection, &PairConfig::new(29)).expect("8 shots x 3 classes");
        assert_eq!(selection.class_sizes(), [(0, 8), (1, 8), (2, 8)]);
        assert_eq!(sampler.budget(), 384);
        assert!(!sampler.layout().default_was_clamped());
        assert_eq!(sampler.layout().emitted_kinds(), EmittedKinds::Both);
        assert_eq!(sampler.layout().affected_singleton_classes(), 0);
    }

    #[test]
    fn pair_sampler_uses_the_clamped_budget_not_the_closed_form() {
        let selection = selection_8_shots();
        let cfg = PairConfig {
            hard_cap: Some(100),
            ..PairConfig::new(29)
        };
        let sampler = PairSampler::new(&selection, &cfg).expect("a clamped default is legal");
        assert_eq!(sampler.budget(), 100, "the CLAMPED value is what is used");
        assert!(sampler.layout().default_was_clamped());
        assert_eq!(sampler.iter_from(0).expect("from zero").count(), 100);
    }

    #[test]
    fn pair_sampler_refuses_an_explicit_budget_above_the_hard_cap() {
        let selection = selection_8_shots();
        let cfg = PairConfig {
            budget: Some(20_000),
            hard_cap: Some(10_000),
            ..PairConfig::new(29)
        };
        match PairSampler::new(&selection, &cfg) {
            Err(ContrastiveDataError::BudgetExceedsHardCap { budget, hard_cap }) => {
                assert_eq!((budget, hard_cap), (20_000, 10_000));
            }
            other => panic!("expected BudgetExceedsHardCap, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn pair_layout_counts_affected_singleton_classes_even_when_zero() {
        assert_eq!(default_layout(&[8, 4, 8]).affected_singleton_classes(), 0);
        assert_eq!(default_layout(&[4, 1]).affected_singleton_classes(), 1);
        assert_eq!(default_layout(&[1; 32]).affected_singleton_classes(), 32);
        assert_eq!(
            default_layout(&[5, 1, 1, 4]).affected_singleton_classes(),
            2
        );
    }

    // -- interleave and balance ------------------------------------------------------------

    #[test]
    fn pair_stream_alternates_positive_at_even_ordinals_and_negative_at_odd() {
        let layout = default_layout(&[8, 4, 8]);
        assert_eq!(layout.budget(), 256);
        for (ordinal, raw) in all_raw(&layout).into_iter().enumerate() {
            let want = if ordinal % 2 == 0 {
                PairKind::Positive
            } else {
                PairKind::Negative
            };
            assert_eq!(raw.kind, want, "ordinal {ordinal}");
        }
    }

    /// `|#pos − #neg| ≤ 1` at EVERY prefix and `== 0` at every EVEN prefix. Stated this way
    /// rather than as "strict 1:1" because that would be false for an odd budget and the
    /// test would then have to be quietly relaxed.
    #[test]
    fn pair_stream_balance_holds_at_every_prefix_including_an_odd_budget() {
        for budget in [255_u64, 256] {
            let layout = capped_layout(&[8, 4, 8], budget);
            let (mut pos, mut neg) = (0_i64, 0_i64);
            for (ordinal, raw) in all_raw(&layout).into_iter().enumerate() {
                match raw.kind {
                    PairKind::Positive => pos += 1,
                    PairKind::Negative => neg += 1,
                }
                assert!((pos - neg).abs() <= 1, "budget {budget} prefix {ordinal}");
                if ordinal % 2 == 1 {
                    assert_eq!(pos, neg, "even prefix of budget {budget}");
                }
            }
            let expected_pos = i64::try_from(budget.div_ceil(2)).expect("small");
            assert_eq!(pos, expected_pos, "ceil(B/2) positives at budget {budget}");
            assert_eq!(
                neg,
                i64::try_from(budget / 2).expect("small"),
                "floor(B/2) negatives at budget {budget}"
            );
        }
    }

    #[test]
    fn positives_only_and_negatives_only_streams_emit_exactly_one_kind() {
        let positives = capped_layout(&[6], 25);
        assert_eq!(positives.emitted_kinds(), EmittedKinds::PositivesOnly);
        assert!(all_raw(&positives)
            .iter()
            .all(|raw| raw.kind == PairKind::Positive));
        assert_eq!(all_raw(&positives).len(), 25);

        let negatives = capped_layout(&[1; 32], 17);
        assert_eq!(negatives.emitted_kinds(), EmittedKinds::NegativesOnly);
        assert!(all_raw(&negatives)
            .iter()
            .all(|raw| raw.kind == PairKind::Negative));
        assert_eq!(all_raw(&negatives).len(), 17);
    }

    // -- structure -------------------------------------------------------------------------

    #[test]
    fn positives_share_a_class_and_negatives_do_not() {
        for sizes in [vec![8_u64, 4, 8], vec![4, 1], vec![2, 2], vec![3, 5, 7]] {
            let layout = capped_layout(&sizes, 400);
            let raws = all_raw(&layout);
            // Vacuity guard: pin the population before asserting a relation over it. An
            // empty stream satisfies every clause below (02-04's lesson).
            assert_eq!(raws.len(), 400, "{sizes:?} must actually emit 400 pairs");
            assert!(raws.iter().any(|raw| raw.kind == PairKind::Positive));
            assert!(raws.iter().any(|raw| raw.kind == PairKind::Negative));
            for raw in raws {
                match raw.kind {
                    PairKind::Positive => {
                        assert_eq!(raw.first.class_index, raw.second.class_index, "{sizes:?}");
                        assert_ne!(
                            raw.first.member_index, raw.second.member_index,
                            "a positive pair is never a self-pair ({sizes:?})"
                        );
                    }
                    PairKind::Negative => {
                        assert_ne!(raw.first.class_index, raw.second.class_index, "{sizes:?}");
                    }
                }
                assert!(raw.first.member_index < layout.class_size(raw.first.class_index));
                assert!(raw.second.member_index < layout.class_size(raw.second.class_index));
            }
        }
    }

    /// The singleton class contributes no positives (its `C(1,2) = 0` weight is zero) but
    /// does appear in negatives — `NegativesOnly` observed structurally, not asserted.
    #[test]
    fn a_singleton_class_never_appears_in_a_positive_pair_but_does_in_negatives() {
        let layout = capped_layout(&[4, 1], 200);
        let singleton = 1_usize;
        let mut negatives_touching_singleton = 0;
        for raw in all_raw(&layout) {
            match raw.kind {
                PairKind::Positive => {
                    assert_ne!(raw.first.class_index, singleton);
                    assert_ne!(raw.second.class_index, singleton);
                }
                PairKind::Negative => {
                    if raw.first.class_index == singleton || raw.second.class_index == singleton {
                        negatives_touching_singleton += 1;
                    }
                }
            }
        }
        assert_eq!(
            negatives_touching_singleton, 100,
            "every negative in a two-class layout must touch the singleton"
        );
    }

    #[test]
    fn targets_are_derived_from_selection_label_of_for_every_emitted_pair() {
        let selection = selection_8_shots();
        let cfg = PairConfig {
            budget: Some(200),
            ..PairConfig::new(31)
        };
        let sampler = PairSampler::new(&selection, &cfg).expect("a legal budget");
        let mut same = 0;
        for labeled in sampler.iter_from(0).expect("from zero") {
            let lo = selection.label_of(labeled.pair.lo());
            let hi = selection.label_of(labeled.pair.hi());
            let want = if lo == hi { 1.0 } else { 0.0 };
            assert_eq!(labeled.target, want);
            assert!(labeled.pair.lo() < labeled.pair.hi());
            same += usize::from(lo == hi);
        }
        assert_eq!(
            same, 100,
            "both kinds must actually occur or this proves nothing"
        );
    }

    // -- purity and range --------------------------------------------------------------------

    #[test]
    fn pair_at_is_pure_and_order_independent() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(37)).expect("legal");
        let pick = |ordinal: u64| sampler.pair_at(ordinal).expect("below the budget");

        let forward: Vec<LabeledPair> = (0..24).map(pick).collect();
        let shuffled: Vec<LabeledPair> = [17_u64, 3, 22, 0, 11].iter().map(|i| pick(*i)).collect();
        for (slot, ordinal) in [17_usize, 3, 22, 0, 11].into_iter().enumerate() {
            assert_eq!(shuffled[slot], forward[ordinal], "ordinal {ordinal}");
        }

        let resumed: Vec<LabeledPair> = sampler
            .iter_from(10)
            .expect("mid-stream")
            .take(14)
            .collect();
        assert_eq!(resumed, forward[10..24].to_vec());
    }

    #[test]
    fn pair_at_and_iter_from_beyond_the_budget_are_typed_errors() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(41)).expect("legal");
        let budget = sampler.budget();
        // Vacuity guard: with a zero budget EVERY ordinal is out of range, so the clauses
        // below would hold against a sampler that emits nothing at all.
        assert_eq!(budget, 384);

        for ordinal in [budget, budget + 1] {
            match sampler.pair_at(ordinal) {
                Err(ContrastiveDataError::OrdinalOutOfRange {
                    ordinal: got,
                    budget: cap,
                }) => {
                    assert_eq!((got, cap), (ordinal, budget));
                }
                other => panic!("expected OrdinalOutOfRange, got {:?}", other.map(|_| ())),
            }
        }
        // `offset == budget` is a legal EMPTY cursor; only beyond it is an error.
        assert_eq!(sampler.iter_from(budget).expect("empty cursor").count(), 0);
        assert!(matches!(
            sampler.iter_from(budget + 1),
            Err(ContrastiveDataError::OrdinalOutOfRange { .. })
        ));
    }

    // -- O(K) evidence: the DATA-05 blocker (review finding F1) -------------------------------

    /// A class-PAIR design would report `C(32, 2) = 496` here. This is the layout that
    /// separates `O(K)` from `O(K²)`; at K = 3 the two are indistinguishable.
    #[test]
    fn state_report_weight_arrays_are_k_long_at_k_equals_n() {
        let layout = capped_layout(&[1; 32], 16);
        let report = layout.state_report();
        assert_eq!(report.negative_weight_entries, 32);
        assert_eq!(report.class_offset_entries, 32);
        assert_eq!(report.positive_weight_entries, 32);
        assert_eq!(report.materialized_pairs, 0);
        assert_eq!(
            negative_capacity(&[1; 32]).expect("no overflow"),
            496,
            "the rejected design's array length, named so the contrast is explicit"
        );
        assert!(report.total_retained_entries() < 496);
    }

    /// Non-quadratic growth proven by SCALING, not by inspection: at K = 8, 32 and 128
    /// all-singleton layouts under the SAME fixed budget, retained state grows LINEARLY.
    ///
    /// The honest sampler retains exactly three K-long arrays and no per-example entries,
    /// so `total_retained_entries() == 3K` and the ratio between successive K values must
    /// equal the ratio of the K values themselves. A class-pair design would retain
    /// `~K²/2` and the 4× step from 8 to 32 would show as ~16×.
    #[test]
    fn state_report_grows_linearly_in_k_under_a_fixed_budget() {
        let budget = 16_u64;
        let mut totals = Vec::new();
        for k in [8_usize, 32, 128] {
            let sizes = vec![1_u64; k];
            let layout = capped_layout(&sizes, budget);
            assert_eq!(layout.budget(), budget, "the budget is held FIXED across K");
            let total = layout.state_report().total_retained_entries();
            assert_eq!(total, 3 * k, "three O(K) arrays and nothing else");
            totals.push(total);
        }
        assert_eq!(totals, vec![24, 96, 384]);
        // 4x in K must be 4x in state. Quadratic growth would be 16x.
        assert_eq!(
            totals[1] * 8 / 32,
            totals[0],
            "8 -> 32 is linear, not quadratic"
        );
        assert_eq!(totals[2] * 32 / 128, totals[1], "32 -> 128 is linear");
    }

    #[test]
    fn state_report_does_not_grow_with_the_budget_consumed() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(43)).expect("legal");
        let before = sampler.state_report();
        let drained = sampler.iter_from(0).expect("from zero").count();
        let after = sampler.state_report();

        assert_eq!(drained, 384);
        assert_eq!(
            before, after,
            "retained state is independent of pairs emitted"
        );
        assert_eq!(
            after.materialized_pairs, 0,
            "the honest sampler stores no pairs"
        );
        assert_eq!(after.bucket_entries, 24, "one handle per selected example");
        assert_eq!(after.total_retained_entries(), 24 + 3 * 3);
    }

    /// The same report, at a layout whose pair space dwarfs its examples: 192 examples,
    /// 24,576 pairs. A materializing sampler is quadratic HERE; the honest one is not.
    #[test]
    fn state_report_is_flat_where_the_pair_space_dwarfs_the_examples() {
        let (selection, _) = test_corpus::fresh_selection(70, 47, 64);
        let sampler = PairSampler::new(&selection, &PairConfig::new(47)).expect("64 shots");
        assert_eq!(sampler.budget(), 24_576);
        let report = sampler.state_report();
        assert_eq!(report.total_retained_entries(), 192 + 3 * 3);
        assert_eq!(report.materialized_pairs, 0);
    }

    // -- marginal equivalence of the O(K) negative scheme -------------------------------------

    /// The O(K) rewrite is a REPRESENTATION change, not a semantics change. Sampling ordered
    /// cross-class endpoint pairs uniformly induces weight `n_j · n_k` on each unordered
    /// class pair, which is exactly what D-14 specifies. This test measures that empirically
    /// rather than restating the algebra.
    #[test]
    fn negative_draw_marginals_match_the_n_j_times_n_k_class_pair_weights() {
        let sizes = [3_u64, 5, 7];
        let draws = 40_000_u64;
        let layout = capped_layout(&sizes, draws);
        assert_eq!(layout.emitted_kinds(), EmittedKinds::Both);

        let mut counts: BTreeMap<(usize, usize), u64> = BTreeMap::new();
        let mut negatives = 0_u64;
        for raw in all_raw(&layout) {
            if raw.kind != PairKind::Negative {
                continue;
            }
            negatives += 1;
            let (a, b) = (raw.first.class_index, raw.second.class_index);
            *counts.entry((a.min(b), a.max(b))).or_default() += 1;
        }

        let total_weight = negative_capacity(&sizes).expect("no overflow") as f64;
        assert_eq!(counts.len(), 3, "all three class pairs must be reached");
        for ((j, k), hits) in &counts {
            let expected = (sizes[*j] * sizes[*k]) as f64 / total_weight;
            let observed = *hits as f64 / negatives as f64;
            assert!(
                (observed - expected).abs() < 0.02,
                "class pair ({j},{k}): expected {expected:.4}, observed {observed:.4}"
            );
        }
        // A uniform-over-class-pairs bug would give 1/3 each; the ordering below rules it out.
        assert!(counts[&(1, 2)] > counts[&(0, 2)]);
        assert!(counts[&(0, 2)] > counts[&(0, 1)]);
    }

    #[test]
    fn negative_draw_reaches_every_cross_class_endpoint_pair() {
        let layout = capped_layout(&[2, 2], 400);
        let mut seen: BTreeMap<(u64, u64), u64> = BTreeMap::new();
        for raw in all_raw(&layout) {
            if raw.kind != PairKind::Negative {
                continue;
            }
            let (mut a, mut b) = (
                layout_global(&layout, raw.first.class_index, raw.first.member_index),
                layout_global(&layout, raw.second.class_index, raw.second.member_index),
            );
            if a > b {
                core::mem::swap(&mut a, &mut b);
            }
            *seen.entry((a, b)).or_default() += 1;
        }
        assert_eq!(
            seen.len(),
            4,
            "all four cross-class endpoint pairs of [2,2] must be reachable, saw {seen:?}"
        );
    }

    fn layout_global(layout: &PairLayout, class_index: usize, member_index: u64) -> u64 {
        (0..class_index).map(|c| layout.class_size(c)).sum::<u64>() + member_index
    }

    // -- the untrusted boundary (review finding F13) -------------------------------------------

    fn honest_records(sampler: &PairSampler<'_>, count: u64) -> Vec<UntrustedPairRecord> {
        let selection = sampler.selection();
        sampler
            .iter_from(0)
            .expect("from zero")
            .take(count as usize)
            .map(|labeled| UntrustedPairRecord {
                lo: selection.id_of(labeled.pair.lo()).to_string(),
                hi: selection.id_of(labeled.pair.hi()).to_string(),
                target: labeled.target,
            })
            .collect()
    }

    #[test]
    fn validate_pair_records_accepts_the_samplers_own_pairs() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(53)).expect("legal");
        let records = honest_records(&sampler, 40);
        // Vacuity guard: two empty vectors are equal, which is exactly how this assertion
        // would pass against a validator that returns nothing (02-04's lesson).
        assert_eq!(records.len(), 40);
        let validated = validate_pair_records(&records, &selection).expect("honest records pass");
        assert_eq!(validated.len(), 40);

        let expected: Vec<LabeledPair> =
            sampler.iter_from(0).expect("from zero").take(40).collect();
        assert_eq!(validated, expected);
    }

    #[test]
    fn validate_pair_records_rejects_an_endpoint_outside_the_selection() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(53)).expect("legal");
        let mut records = honest_records(&sampler, 10);
        records[4].lo = "validation:1".to_string();

        match validate_pair_records(&records, &selection) {
            Err(ContrastiveDataError::EndpointNotInSelection { id, found_in }) => {
                assert_eq!(id, "validation:1");
                assert!(!found_in.is_empty());
            }
            other => panic!(
                "expected EndpointNotInSelection, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    #[test]
    fn validate_pair_records_rejects_a_self_pair() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(53)).expect("legal");
        let mut records = honest_records(&sampler, 10);
        records[2].hi = records[2].lo.clone();

        assert!(matches!(
            validate_pair_records(&records, &selection),
            Err(ContrastiveDataError::SelfPair { .. })
        ));
    }

    #[test]
    fn validate_pair_records_rejects_a_target_that_disagrees_with_the_endpoints() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(53)).expect("legal");
        let mut records = honest_records(&sampler, 10);
        let flipped = 1.0 - records[0].target;
        records[0].target = flipped;

        match validate_pair_records(&records, &selection) {
            Err(ContrastiveDataError::PairTargetMismatch {
                lo,
                hi,
                declared_target,
                derived_target,
            }) => {
                assert_eq!(lo, records[0].lo);
                assert_eq!(hi, records[0].hi);
                assert_eq!(declared_target, flipped);
                assert_eq!(derived_target, 1.0 - flipped);
            }
            other => panic!("expected PairTargetMismatch, got {:?}", other.map(|_| ())),
        }
    }

    /// Orientation carries no information (D-12), so a swapped record is NORMALIZED rather
    /// than refused — and must produce the identical `LabeledPair`.
    #[test]
    fn validate_pair_records_canonicalizes_a_swapped_record() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(53)).expect("legal");
        let records = honest_records(&sampler, 4);
        let swapped: Vec<UntrustedPairRecord> = records
            .iter()
            .map(|rec| UntrustedPairRecord {
                lo: rec.hi.clone(),
                hi: rec.lo.clone(),
                target: rec.target,
            })
            .collect();
        assert_ne!(swapped, records);
        assert_eq!(
            validate_pair_records(&swapped, &selection).expect("swapped records normalize"),
            validate_pair_records(&records, &selection).expect("honest records pass")
        );
    }

    #[test]
    fn parse_pair_dump_round_trips_and_rejects_an_unknown_field() {
        let selection = selection_8_shots();
        let sampler = PairSampler::new(&selection, &PairConfig::new(53)).expect("legal");
        let records = honest_records(&sampler, 3);

        let mut bytes = Vec::new();
        for rec in &records {
            bytes.extend_from_slice(
                serde_json::to_string(rec)
                    .expect("a record serializes")
                    .as_bytes(),
            );
            bytes.push(b'\n');
        }
        assert_eq!(parse_pair_dump(&bytes).expect("well-formed"), records);

        let poisoned = br#"{"lo":"a","hi":"b","target":1.0,"rogue":7}"#;
        match parse_pair_dump(poisoned) {
            Err(ContrastiveDataError::MalformedRow { index, reason, .. }) => {
                assert_eq!(index, 0);
                assert!(
                    reason.contains("rogue"),
                    "reason {reason:?} must name the field"
                );
            }
            other => panic!(
                "deny_unknown_fields must reject, got {:?}",
                other.map(|_| ())
            ),
        }
    }

    #[test]
    fn degenerate_layouts_are_refused_or_classified_at_construction() {
        for sizes in [vec![1_u64], vec![]] {
            assert!(matches!(
                PairLayout::from_class_sizes(&sizes, &PairConfig::new(13)),
                Err(ContrastiveDataError::NoPairCapacity { .. })
            ));
        }
        let singletons = default_layout(&[1; 32]);
        assert_eq!(singletons.emitted_kinds(), EmittedKinds::NegativesOnly);
        assert_eq!(singletons.budget(), 992);
        assert_eq!(
            classify_degenerate(
                positive_capacity(&[1; 32]).expect("no overflow"),
                negative_capacity(&[1; 32]).expect("no overflow")
            )
            .expect("legal"),
            singletons.emitted_kinds()
        );

        let one_class = default_layout(&[6]);
        assert_eq!(one_class.emitted_kinds(), EmittedKinds::PositivesOnly);
        assert_eq!(one_class.budget(), 30);
        assert_eq!(one_class.total_examples(), 6);
        assert_eq!(one_class.class_count(), 1);
        assert_eq!(DEFAULT_HARD_CAP, 1_048_576);
    }

    /// Every capacity accessor pinned on TWO layouts (plan 02-08 mutation triage).
    ///
    /// `cargo mutants` found `PairLayout::class_count -> 1`,
    /// `PairLayout::positive_capacity -> 0 | 1` and `PairLayout::negative_capacity -> 0 | 1`
    /// all surviving: the closed-form FUNCTIONS were pinned everywhere, but the accessors
    /// that hand their results to `PairReplayRecord::from_sampler` were not, so a manifest
    /// could have recorded a capacity of 0 for a layout with 192 negative pairs and nothing
    /// would have been red.
    ///
    /// Two layouts rather than one, deliberately: a single layout is satisfied by an
    /// accessor returning the right constant, and returning a constant is exactly the
    /// mutation being killed.
    #[test]
    fn layout_capacity_accessors_agree_with_the_closed_forms_on_two_layouts() {
        for sizes in [vec![8_u64, 8, 8], vec![8_u64, 4, 8]] {
            let subject = default_layout(&sizes);
            assert_eq!(
                subject.class_count(),
                sizes.len(),
                "class_count for {sizes:?}"
            );
            assert_eq!(
                subject.positive_capacity(),
                positive_capacity(&sizes).expect("no overflow"),
                "positive_capacity for {sizes:?}"
            );
            assert_eq!(
                subject.negative_capacity(),
                negative_capacity(&sizes).expect("no overflow"),
                "negative_capacity for {sizes:?}"
            );
        }

        // The two layouts must actually DISAGREE, or "agrees on two layouts" is one claim
        // wearing two hats.
        let even = default_layout(&[8, 8, 8]);
        let uneven = default_layout(&[8, 4, 8]);
        assert_eq!(
            (even.positive_capacity(), even.negative_capacity()),
            (84, 192)
        );
        assert_eq!(
            (uneven.positive_capacity(), uneven.negative_capacity()),
            (62, 128)
        );
    }
}

#[cfg(test)]
mod pair_proptests {
    //! The RUNNABLE evidence behind the two DECLARED-not-executed Kani harnesses
    //! (`KANI-CPP-001`, `KANI-CPP-002`). `cargo-kani` is not installed in this repository
    //! and there is no `#[kani::proof]` harness anywhere under `crates/`, so these
    //! proptests — bounded identically at 4 — are the evidence, not a placeholder for it.

    use super::{
        negative_capacity, positive_capacity, CanonicalPair, PairConfig, PairKind, PairLayout,
        RetainedState,
    };
    use crate::error::ContrastiveDataError;
    use crate::select::{test_corpus, SelectedId};
    use proptest::collection::vec as prop_vec;
    use proptest::prelude::{prop_assert, prop_assert_eq, proptest};

    fn ordinals() -> Vec<SelectedId> {
        let (selection, _) = test_corpus::fresh_selection(12, 17, 8);
        selection
            .examples()
            .iter()
            .map(|row| {
                selection
                    .selected_id(&row.id)
                    .expect("every selected example resolves to its own ordinal")
            })
            .collect()
    }

    /// Naive reference enumeration — deliberately quadratic, deliberately only used inside
    /// the bound-4 proptest, and therefore the independent second derivation the closed
    /// forms are checked against.
    fn naive_capacities(sizes: &[u64]) -> (u64, u64) {
        let mut pos = 0;
        let mut neg = 0;
        for (j, &nj) in sizes.iter().enumerate() {
            pos += nj * nj.saturating_sub(1) / 2;
            for &nk in &sizes[j + 1..] {
                neg += nj * nk;
            }
        }
        (pos, neg)
    }

    proptest! {
        /// KANI-CPP-001's backing proptest, bound 4.
        #[test]
        fn canonical_pair_ordering(a in 0_usize..4, b in 0_usize..4) {
            let ids = ordinals();
            let (x, y) = (ids[a], ids[b]);
            match CanonicalPair::new(x, y) {
                Ok(pair) => {
                    prop_assert!(a != b);
                    prop_assert!(pair.lo() < pair.hi());
                    prop_assert_eq!(pair, CanonicalPair::new(y, x).expect("the mirror pairs"));
                }
                Err(ContrastiveDataError::SelfPair { id }) => {
                    prop_assert_eq!(a, b);
                    prop_assert_eq!(id, u64::from(x.ordinal()));
                }
                Err(other) => prop_assert!(false, "unexpected error {:?}", other),
            }
        }

        /// KANI-CPP-002's backing proptest, bound 4.
        #[test]
        fn capacity_no_overflow(sizes in prop_vec(0_u64..4, 0..=4_usize)) {
            let (want_pos, want_neg) = naive_capacities(&sizes);
            prop_assert_eq!(positive_capacity(&sizes).expect("bounded sizes never overflow"), want_pos);
            prop_assert_eq!(negative_capacity(&sizes).expect("bounded sizes never overflow"), want_neg);
        }

        /// Every structural pair invariant, swept over the five contracted layout shapes
        /// and a random ordinal: the balanced case, the singleton case, the single-class
        /// case, the K = N case, and a two-class case.
        #[test]
        fn pair_stream_invariants_hold_across_layouts(
            which in 0_usize..5,
            ordinal in 0_u64..200,
        ) {
            let sizes: Vec<u64> = match which {
                0 => vec![8, 4, 8],
                1 => vec![4, 1],
                2 => vec![6],
                3 => vec![1; 32],
                _ => vec![2, 2],
            };
            let cfg = PairConfig { budget: Some(200), ..PairConfig::new(59) };
            let layout = PairLayout::from_class_sizes(&sizes, &cfg)
                .expect("every one of these layouts has capacity");
            let raw = layout.raw_pair_at(ordinal).expect("ordinal < 200 == budget");

            let (a, b) = (raw.first, raw.second);
            prop_assert!(a.member_index < layout.class_size(a.class_index));
            prop_assert!(b.member_index < layout.class_size(b.class_index));
            match raw.kind {
                PairKind::Positive => {
                    prop_assert_eq!(a.class_index, b.class_index);
                    prop_assert!(a.member_index < b.member_index);
                    prop_assert!(layout.class_size(a.class_index) >= 2);
                }
                PairKind::Negative => prop_assert!(a.class_index != b.class_index),
            }
            // Purity: the same ordinal, asked again, is the same pair.
            prop_assert_eq!(raw, layout.raw_pair_at(ordinal).expect("pure"));
            // Retained state is three K-long arrays whatever the ordinal.
            let report = layout.state_report();
            prop_assert_eq!(report.negative_weight_entries, sizes.len());
            prop_assert_eq!(report.materialized_pairs, 0);
        }
    }

    /// The adversarial half of KANI-CPP-002: near-`u64::MAX` vectors are a typed error,
    /// never a wrap. Outside `proptest!` because the inputs are enumerated, not sampled.
    #[test]
    fn capacity_no_overflow_adversarial_vectors_are_typed_errors() {
        for sizes in [
            vec![u64::MAX],
            vec![u64::MAX, u64::MAX],
            vec![u64::MAX - 1, 3],
            vec![u64::MAX / 2, u64::MAX / 2],
        ] {
            let pos = positive_capacity(&sizes);
            let neg = negative_capacity(&sizes);
            assert!(
                matches!(pos, Err(ContrastiveDataError::ArithmeticOverflow { .. }))
                    || matches!(neg, Err(ContrastiveDataError::ArithmeticOverflow { .. })),
                "{sizes:?} wrapped instead of erroring: pos={pos:?} neg={neg:?}"
            );
        }
    }
}
