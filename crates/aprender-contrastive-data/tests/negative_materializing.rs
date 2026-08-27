//! D-25 — the boundedness gate is not theater.
//!
//! Obligations: `OBLIG-CPP-CAPACITY-INVARIANT` and `OBLIG-CPP-KN-ADVERSARIAL`.
//! Requirement: DATA-05. Same three-element discipline as
//! `crates/aprender-core/tests/setfit_conformance/detach_negative.rs` (Phase 1, D-24) and
//! as this phase's `negative_leaky.rs`.
//!
//! "The sampler is O(K) and streams" is the phase's second fakeable claim. Nothing in a
//! green suite distinguishes a streaming sampler from one that quietly enumerates the whole
//! pair space and hands out slices of it: both produce the same pairs. This file builds the
//! materializing implementation on purpose — the trap the SetFit reference itself falls into
//! (RESEARCH Finding F2) — and requires the SAME public gate that passes the honest sampler
//! to FAIL it.
//!
//! # Measurement is STRUCTURAL, through the public `state_report()`
//!
//! Both subjects implement the crate's public
//! [`RetainedState`](aprender_contrastive_data::pairs::RetainedState) trait, so
//! [`check_capacity_invariant`] is literally ONE call for the honest and the deliberately
//! wrong implementation. A gate that read a different accessor per subject would be
//! comparing two claims rather than measuring one property, and a sampler-reported
//! "bytes used" number is exactly as trustworthy as a self-reported decreasing loss.
//!
//! # Why K = N is a required case and three classes can never be one
//!
//! At K = 3 the rejected O(K²) class-pair design retains 3 entries and the shipped O(K)
//! design retains 3 entries: `K²/2` and `K` are indistinguishable, so no amount of
//! three-class testing can tell them apart. At K = N = 32 all-singleton classes the two
//! designs retain 496 and 32. That gap is the evidence, and it is the layout plan 02-07's
//! class-pair mutation was measured against (`left: 496, right: 32`, while 46 of 50 tests
//! stayed green).
//!
//! Run with `cargo test -p aprender-contrastive-data --test negative_materializing`.

// `tests/common/mod.rs` is compiled into EVERY integration-test crate that names it, so
// the loaders this file does not call are dead code HERE while being the whole point of
// `reference_fixtures.rs`.
#[allow(dead_code)]
mod common;

use aprender_contrastive_data::pairs::{
    EmittedKinds, PairConfig, PairKind, PairLayout, PairSampler, RetainedState, SamplerStateReport,
};

// ===============================================================================
// THE BOUND. Defined once, from the honest sampler's STRUCTURE, and applied to
// every subject through one function.
// ===============================================================================

/// Per-example coefficient. The honest [`PairSampler`] retains exactly one borrowed slice
/// handle per class covering every selected row, so its `bucket_entries` is the example
/// count `S` — coefficient 1, no copy. A [`PairLayout`] retains none and reports 0.
const C_EXAMPLES: usize = 1;

/// Per-class coefficient: THREE O(K) arrays and no fourth — the positive weight prefix, the
/// negative weight prefix and the class-offset array. The rejected design's fourth array is
/// indexed by class PAIRS, which is `K(K-1)/2` entries and blows straight through this.
const C_CLASSES: usize = 3;

/// Slack for fixed-size scalar state that `state_report()` does not enumerate at all — the
/// iterator's `cursor` and `budget`, the layout's ten scalar parameters, and the five
/// precomputed domain keys (8 bytes each, derived once at construction rather than per draw).
/// Named rather than rounded so it is a stated allowance and not a fudge factor tuned until
/// the test passed.
///
/// This allowance does not track the field count and must not be grown to match it: the
/// bound applies to the STRUCTURAL counts `state_report()` returns, and no O(1) field
/// appears in any of them. What matters is that this state is fixed-size — a field added
/// here is free, a field indexed by class or by pair is not and would show up in the
/// per-class or per-example terms instead.
const C_CONST: usize = 8;

/// `C_EXAMPLES * examples + C_CLASSES * classes + C_CONST`.
///
/// This is TIGHTER than `OBLIG-CPP-CAPACITY-INVARIANT`'s `c * (examples + classes)` and
/// implies it at `c = 3`, so satisfying this satisfies the contract.
fn capacity_bound(examples: u64, classes: usize) -> usize {
    let examples = usize::try_from(examples).expect("test layouts are far below usize::MAX");
    C_EXAMPLES * examples + C_CLASSES * classes + C_CONST
}

/// THE GATE. One call, both implementations, structural counts only.
///
/// Returns the report on success so a caller can make further assertions about the same
/// measurement it just bounded, rather than calling `state_report()` a second time.
///
/// # Errors
///
/// A message naming every component count, because a bound that fails without saying which
/// term blew it is not diagnosable.
fn check_capacity_invariant(
    subject: &dyn RetainedState,
    examples: u64,
    classes: usize,
) -> Result<SamplerStateReport, String> {
    let report = subject.state_report();
    let total = report.total_retained_entries();
    let bound = capacity_bound(examples, classes);
    if total <= bound {
        return Ok(report);
    }
    Err(format!(
        "retained state {total} exceeds {bound} = {C_EXAMPLES}*{examples} + \
         {C_CLASSES}*{classes} + {C_CONST}; buckets={}, pos_weights={}, neg_weights={}, \
         offsets={}, materialized_pairs={}",
        report.bucket_entries,
        report.positive_weight_entries,
        report.negative_weight_entries,
        report.class_offset_entries,
        report.materialized_pairs,
    ))
}

// ===============================================================================
// THE IN-BAND WRONG IMPLEMENTATION.
// ===============================================================================

/// A sampler that enumerates the whole pair space up front and samples from the list.
///
/// This is not a straw man: it is what `setfit`'s own `shuffle_combinations` does
/// (`np.triu_indices` over the full example set, materialized, then shuffled), and it is
/// the design the contract's `pair_stream` names as REJECTED. It keeps the same three O(K)
/// arrays as the honest layout — so the difference the gate sees is the materialization
/// alone, not a second unrelated defect — and it OWNS its example ordinals rather than
/// borrowing them, which is the other thing a materializing implementation does.
struct MaterializingSampler {
    class_offsets: Vec<u64>,
    positive_weights: Vec<u64>,
    negative_weights: Vec<u64>,
    bucket: Vec<u32>,
    /// THE DEFECT: every unordered pair of distinct examples, held in memory.
    pairs: Vec<(u32, u32)>,
}

impl MaterializingSampler {
    fn from_class_sizes(class_sizes: &[u64]) -> Self {
        let total: u64 = class_sizes.iter().sum();
        let examples = u32::try_from(total).expect("test layouts stay small");

        let mut class_offsets = Vec::with_capacity(class_sizes.len());
        let mut positive_weights = Vec::with_capacity(class_sizes.len());
        let mut negative_weights = Vec::with_capacity(class_sizes.len());
        let mut running = 0_u64;
        for &n in class_sizes {
            class_offsets.push(running);
            positive_weights.push(n * n.saturating_sub(1) / 2);
            negative_weights.push(n * (total - n));
            running += n;
        }

        // The Cartesian enumeration. O(S²) entries, built before a single pair is emitted.
        let mut pairs = Vec::new();
        for first in 0..examples {
            for second in (first + 1)..examples {
                pairs.push((first, second));
            }
        }

        Self {
            class_offsets,
            positive_weights,
            negative_weights,
            bucket: (0..examples).collect(),
            pairs,
        }
    }
}

impl RetainedState for MaterializingSampler {
    fn state_report(&self) -> SamplerStateReport {
        SamplerStateReport {
            bucket_entries: self.bucket.len(),
            positive_weight_entries: self.positive_weights.len(),
            negative_weight_entries: self.negative_weights.len(),
            class_offset_entries: self.class_offsets.len(),
            materialized_pairs: self.pairs.len(),
        }
    }
}

// ===============================================================================
// Fixtures.
// ===============================================================================

/// The layout where the pair space dwarfs the examples: 3 classes x 64 shots = 192
/// examples, a 24,576-pair budget space and an 18,336-pair Cartesian set.
const WIDE_CLASSES: usize = 3;
const WIDE_SHOTS: u32 = 64;
/// The same three classes at the smallest legal shot count — the small end of the scaling
/// measurement. `FewShotSelector::select` admits only `{8, 16, 32, 64}`, so 8 -> 64 is the
/// widest span a real `Selection` can express and the factor below is 8, not 10.
const NARROW_SHOTS: u32 = 8;
const EXAMPLE_GROWTH_FACTOR: usize = (WIDE_SHOTS / NARROW_SHOTS) as usize;

const SEED: u64 = 31;

fn wide_class_sizes() -> Vec<u64> {
    vec![u64::from(WIDE_SHOTS); WIDE_CLASSES]
}

fn narrow_class_sizes() -> Vec<u64> {
    vec![u64::from(NARROW_SHOTS); WIDE_CLASSES]
}

fn layout(class_sizes: &[u64], budget: u64) -> PairLayout {
    PairLayout::from_class_sizes(
        class_sizes,
        &PairConfig {
            budget: Some(budget),
            ..PairConfig::new(SEED)
        },
    )
    .expect("every layout used here has pair capacity")
}

// ===============================================================================
// (a) The materializer is RED under the gate.
// ===============================================================================

#[test]
fn the_materializing_sampler_violates_the_capacity_invariant() {
    let sizes = wide_class_sizes();
    let examples: u64 = sizes.iter().sum();
    let materializing = MaterializingSampler::from_class_sizes(&sizes);

    // Pin the population before bounding it. An empty pair list would satisfy the bound
    // trivially and this negative would be green for the worst possible reason.
    let report = materializing.state_report();
    assert_eq!(
        report.materialized_pairs,
        (192 * 191) / 2,
        "the materializer must actually hold the whole Cartesian set"
    );

    let failure = check_capacity_invariant(&materializing, examples, WIDE_CLASSES).expect_err(
        "the capacity gate ACCEPTED a sampler holding the entire Cartesian pair set. \
         Every boundedness result in this crate is worthless.",
    );

    // The message must name what blew the bound, or a real failure is not diagnosable.
    assert!(
        failure.contains("materialized_pairs=18336"),
        "the failure does not report the materialized pair count: {failure}"
    );
    assert!(
        failure.contains("exceeds 209"),
        "the failure does not report the bound it broke: {failure}"
    );
}

// ===============================================================================
// (b) THE MIRROR — the honest sampler passes the identical call.
// ===============================================================================

#[test]
fn mirror_the_honest_sampler_satisfies_the_same_capacity_call_at_the_same_layout() {
    let selection =
        common::synthetic_selection(WIDE_CLASSES, WIDE_SHOTS as usize, SEED, WIDE_SHOTS);
    let sampler = PairSampler::new(&selection, &PairConfig::new(SEED))
        .expect("192 examples in 3 classes have pair capacity");

    let examples: u64 = selection.len() as u64;
    assert_eq!(
        examples, 192,
        "the mirror must be the SAME layout as the negative"
    );

    let report = check_capacity_invariant(&sampler, examples, WIDE_CLASSES)
        .expect("the identical call must ACCEPT the honest streaming sampler");

    assert_eq!(
        report.materialized_pairs, 0,
        "an honest streaming sampler holds no pairs at all"
    );
    // The budget here is the full closed-form default — 24,576 pairs of space — and none of
    // it is retained. Without this the mirror would be compatible with a sampler that is
    // bounded only because it was asked for very little.
    assert!(
        sampler.budget() >= 24_576,
        "the mirror must run at a budget large enough for materialization to be tempting; \
         got {}",
        sampler.budget()
    );
}

// ===============================================================================
// (c) Scaling. Non-quadratic behaviour is proven by GROWTH, never by one size.
// ===============================================================================

#[test]
fn honest_state_grows_no_faster_than_the_examples_while_the_materializer_grows_quadratically() {
    let small = common::synthetic_selection(WIDE_CLASSES, WIDE_SHOTS as usize, SEED, NARROW_SHOTS);
    let large = common::synthetic_selection(WIDE_CLASSES, WIDE_SHOTS as usize, SEED, WIDE_SHOTS);
    let budget = 128;
    let cfg = PairConfig {
        budget: Some(budget),
        ..PairConfig::new(SEED)
    };

    // The SAME budget at both sizes, so the measurement is about the layout rather than
    // about how much work was requested.
    let small_sampler = PairSampler::new(&small, &cfg).expect("24 examples support 128 pairs");
    let large_sampler = PairSampler::new(&large, &cfg).expect("192 examples support 128 pairs");
    assert_eq!(small_sampler.budget(), large_sampler.budget());
    assert_eq!(large.len(), small.len() * EXAMPLE_GROWTH_FACTOR);

    let honest_small = small_sampler.state_report().total_retained_entries();
    let honest_large = large_sampler.state_report().total_retained_entries();
    assert!(
        honest_large <= honest_small * (EXAMPLE_GROWTH_FACTOR + 1),
        "honest state grew from {honest_small} to {honest_large} over an \
         {EXAMPLE_GROWTH_FACTOR}x growth in examples — that is faster than linear"
    );

    let mat_small = MaterializingSampler::from_class_sizes(&narrow_class_sizes())
        .state_report()
        .total_retained_entries();
    let mat_large = MaterializingSampler::from_class_sizes(&wide_class_sizes())
        .state_report()
        .total_retained_entries();

    // The control that makes the assertion above mean something: the SAME span applied to a
    // quadratic implementation blows through the same linear allowance by a wide margin. If
    // this ever stopped holding, the growth assertion above would be untestable rather than
    // satisfied.
    assert!(
        mat_large > mat_small * EXAMPLE_GROWTH_FACTOR * 4,
        "the materializer grew only {mat_small} -> {mat_large}; it is not behaving \
         quadratically and cannot serve as the contrast"
    );
}

#[test]
fn the_honest_layout_arrays_are_unchanged_at_ten_times_the_example_count() {
    // Exactly 10x, which no `Selection` can express (shots are drawn from {8,16,32,64}) but
    // a bare `PairLayout` can. The class arrays are the part that could plausibly grow with
    // N, and they do not move at all: the growth in `PairSampler` is entirely the borrowed
    // per-example handles, counted separately as `bucket_entries`.
    let base = vec![8_u64; WIDE_CLASSES];
    let ten_x = vec![80_u64; WIDE_CLASSES];
    assert_eq!(
        ten_x.iter().sum::<u64>(),
        base.iter().sum::<u64>() * 10,
        "the two layouts must really differ by 10x or this proves nothing"
    );

    let small = layout(&base, 64);
    let large = layout(&ten_x, 64);
    assert_eq!(
        small.state_report(),
        large.state_report(),
        "ten times the examples must not change a single retained entry"
    );

    check_capacity_invariant(&small, base.iter().sum(), base.len())
        .expect("the small layout is bounded");
    check_capacity_invariant(&large, ten_x.iter().sum(), ten_x.len())
        .expect("the ten-times layout is bounded");
}

// ===============================================================================
// (d) THE ADVERSARIAL K = N CASE. The whole reason this file exists.
// ===============================================================================

#[test]
fn the_adversarial_k_equals_n_layout_retains_o_k_state_not_o_k_squared() {
    let sizes = common::all_singleton_layout(32);

    // The builder and plan 02-04's committed fixture must describe the SAME layout, so the
    // two artifacts cross-check rather than drift.
    assert_eq!(
        sizes,
        common::contracted_layout("singletons_32"),
        "the adversarial layout must be the one the contracted fixture records"
    );

    // 496 is READ, not typed: it is the contracted `negative_capacity` at this layout, which
    // is also C(32, 2) — the exact length of the array the REJECTED class-pair design would
    // allocate here. Typing it would prove only that someone typed it twice.
    let rejected_design_entries =
        usize::try_from(common::contracted_negative_capacity("singletons_32"))
            .expect("496 fits in a usize");
    assert_eq!(rejected_design_entries, (32 * 31) / 2);

    let adversarial = layout(&sizes, common::ADVERSARIAL_BUDGET);
    let report = check_capacity_invariant(&adversarial, 32, sizes.len())
        .expect("the shipped O(K) layout is bounded at K = N");

    assert_eq!(
        report.negative_weight_entries, 32,
        "the negative weight array must be K long, not K(K-1)/2 = {rejected_design_entries}"
    );
    assert_eq!(report.class_offset_entries, 32);
    assert_eq!(report.positive_weight_entries, 32);
    assert_eq!(
        report.materialized_pairs, 0,
        "nothing is materialized at any layout"
    );
    assert!(
        report.negative_weight_entries < rejected_design_entries,
        "at K = N the shipped design must retain strictly fewer entries than the rejected \
         one, or this case discriminates nothing"
    );

    // The budget is fixed and small, and the layout still reports the same arrays: retained
    // state is a function of the layout, never of how many pairs were asked for.
    assert_eq!(adversarial.budget(), common::ADVERSARIAL_BUDGET);
    assert_eq!(
        layout(&sizes, 992).state_report(),
        report,
        "sixty-two times the budget must not change one retained entry"
    );
}

#[test]
fn the_materializing_sampler_is_red_at_the_adversarial_layout_too() {
    // The K = N case must catch the trap as well, not merely admit the honest design. A
    // gate that only fires at 192 examples would say nothing about the layout the phase
    // singled out.
    let sizes = common::all_singleton_layout(32);
    let materializing = MaterializingSampler::from_class_sizes(&sizes);
    let failure = check_capacity_invariant(&materializing, 32, sizes.len())
        .expect_err("a materializing sampler must be RED at K = N as well");
    assert!(
        failure.contains("materialized_pairs=496"),
        "the failure does not report the 496 materialized pairs: {failure}"
    );
}

#[test]
fn honest_state_is_linear_in_the_class_count_across_three_all_singleton_layouts() {
    // A single size cannot distinguish O(K) from O(K²); a 4x step in K that produces a 4x
    // step in state can. Under a FIXED budget throughout.
    let mut totals = Vec::new();
    for k in [8_usize, 32, 128] {
        let sizes = common::all_singleton_layout(k);
        let subject = layout(&sizes, common::ADVERSARIAL_BUDGET);
        let report = check_capacity_invariant(&subject, k as u64, k)
            .unwrap_or_else(|e| panic!("K = {k} must be bounded: {e}"));
        assert_eq!(subject.budget(), common::ADVERSARIAL_BUDGET);
        totals.push(report.total_retained_entries());
    }
    assert_eq!(totals, vec![24, 96, 384]);
    assert_eq!(totals[1], totals[0] * 4, "4x in K must be 4x in state");
    assert_eq!(totals[2], totals[1] * 4, "and again");
}

/// FALSIFY-CPP-007, at the layout the contract actually names.
///
/// The obligation was discharged in substance at K = 32 and K = 128, but the recorded
/// PREDICTION is about N = 512 under a fixed budget of 64, and its headline number —
/// `negative_capacity == C(512, 2) == 130816` — appeared nowhere in the tree. A prediction
/// no test produces cannot fail for the reason it claims, which is the defect class this
/// phase kept finding elsewhere in the repo; leaving it in our own contract would be the
/// same theater.
///
/// Every literal here is read from the contract, not from a first run: `positive_capacity`
/// 0 (no class has two members), `negative_capacity` 130816, negatives-only emission, and
/// retained state inside `c * (N + K)`. The rejected O(K^2) design would need one entry per
/// unordered class pair — 130,816 of them — so the same gate call that passes here is what
/// it fails.
#[test]
fn falsify_cpp_007_pairs_at_n_512_singletons_stays_bounded() {
    const N: usize = 512;
    const BUDGET: u64 = 64;
    // C(512, 2) = 512 * 511 / 2. Written as the product so a reader can check it, and
    // asserted against the contract's literal so neither can drift alone.
    const EXPECTED_NEGATIVE_CAPACITY: u64 = (N as u64) * (N as u64 - 1) / 2;
    assert_eq!(
        EXPECTED_NEGATIVE_CAPACITY, 130_816,
        "the contract's C(512,2) literal"
    );

    let sizes = common::all_singleton_layout(N);
    let subject = layout(&sizes, BUDGET);

    // The gate — the SAME call both implementations go through, never self-reported.
    let report = check_capacity_invariant(&subject, N as u64, N)
        .unwrap_or_else(|e| panic!("K = N = {N} must be bounded: {e}"));

    assert_eq!(subject.budget(), BUDGET, "the budget is FIXED, not derived");
    assert_eq!(
        subject.positive_capacity(),
        0,
        "no singleton class can furnish a positive pair"
    );
    assert_eq!(
        subject.negative_capacity(),
        EXPECTED_NEGATIVE_CAPACITY,
        "negative_capacity must be C(512,2)"
    );
    assert_eq!(
        subject.emitted_kinds(),
        EmittedKinds::NegativesOnly,
        "positives are impossible at this layout"
    );

    // Linear, not quadratic: the K = 128 case above retains 384 entries, so K = 512 must
    // retain 4x that and not 4^2x. Anchored to the measured smaller case rather than to a
    // blessed constant, so the two move together or the test says so.
    let retained = report.total_retained_entries();
    assert_eq!(
        retained, 1536,
        "K = 512 must retain 4x the K = 128 total (384), not 16x"
    );
    assert!(
        (retained as u64) < EXPECTED_NEGATIVE_CAPACITY / 10,
        "retained {retained} is not comfortably below the {EXPECTED_NEGATIVE_CAPACITY} \
         entries the rejected class-pair design would need"
    );

    // "and the run does not error" — the contract says the stream works, not merely that
    // the arrays are small. Drain the whole fixed budget.
    let drawn: Vec<_> = (0..BUDGET)
        .map(|ordinal| {
            subject
                .raw_pair_at(ordinal)
                .unwrap_or_else(|e| panic!("ordinal {ordinal} must draw at K = N: {e}"))
        })
        .collect();
    assert_eq!(drawn.len(), BUDGET as usize);
    assert!(
        drawn.iter().all(|p| p.kind == PairKind::Negative),
        "every emitted pair at an all-singleton layout must be a negative"
    );
    // Endpoints must land in DIFFERENT classes -- at K = N that is also the leakage check,
    // since same-class would mean a singleton paired with itself.
    assert!(
        drawn
            .iter()
            .all(|p| p.first.class_index != p.second.class_index),
        "a negative pair must span two classes"
    );
}

// ===============================================================================
// (e) Budget independence, and the gate's own identity.
// ===============================================================================

#[test]
fn draining_the_whole_stream_changes_no_retained_entry() {
    let selection = common::synthetic_selection(WIDE_CLASSES, 20, SEED, NARROW_SHOTS);
    let cfg = PairConfig {
        budget: Some(256),
        ..PairConfig::new(SEED)
    };
    let sampler = PairSampler::new(&selection, &cfg).expect("24 examples support 256 pairs");

    let before = sampler.state_report();
    let drained = sampler
        .iter_from(0)
        .expect("offset 0 is within the budget")
        .count();
    assert_eq!(drained, 256, "the stream must actually be drained");
    assert_eq!(
        sampler.state_report(),
        before,
        "emitting 256 pairs must not change one retained entry"
    );
}

#[test]
fn the_capacity_gate_is_one_call_for_both_implementations() {
    // The identity is the whole point of D-25: two gates could be wrong in exactly the way
    // that lets both subjects pass. The needle is assembled at runtime so this scan cannot
    // trip on itself.
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/negative_materializing.rs");
    let text = std::fs::read_to_string(&path).expect("this file is readable");
    let call = format!("{}{}", "check_capacity_", "invariant(&");
    let hits = text
        .lines()
        .filter(|line| line.split("//").next().unwrap_or("").contains(&call))
        .count();
    assert!(
        hits >= 6,
        "expected every capacity assertion — honest and materializing alike — to go \
         through the SAME helper; found {hits} call sites"
    );

    // And there must be no second measurement channel: nothing here may assert against a
    // sampler-reported size string rather than the structural report. These needles are
    // assembled at runtime for the same reason as the one above — a literal would make the
    // scan find itself and the check would be permanently red.
    for needle in [
        format!("{}{}", "memory_", "used"),
        format!("{}{}", "retained_", "bytes"),
        format!("{}{}", "bytes_", "allocated"),
    ] {
        assert!(
            !text.contains(&needle),
            "`{needle}` — a self-reported size — has appeared in this file; the gate must \
             stay structural"
        );
    }
}
