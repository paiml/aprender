//! D-08 / TRN-05 — the exactly-once head is not theater.
//!
//! Obligation: the structural half of TRN-05. This is the Phase 1 D-24 / Phase 2 D-25 in-band
//! negative discipline (`aprender-contrastive-data/tests/negative_leaky.rs`) applied to the
//! head's input: build the forbidden thing on purpose, and require the difference it makes to
//! be visible in every `cargo test`.
//!
//! # Why the attack cannot be routed through `fit_head`
//!
//! `fit_head` takes `self` and nothing else. Everything it fits on is reached from the run it
//! consumes — the dataset, the `Selection`, the resolved configuration — and it builds its
//! input through `head_input::head_dataset`, whose only count is a batch size. There is no
//! parameter, no field and no builder anywhere on that path that can say "weight this row
//! twice". So the pair-weighted fitter below is NOT written against `fit_head`: it cannot be.
//! It is assembled from the surface that does remain expressible — raw embedding rows plus
//! the public `MultinomialLogisticRegression` — exactly as `negative_leaky.rs` had to poison
//! an untrusted DTO rather than a `LabeledPair`. If this file could route the attack through
//! `fit_head` or any `SetFitRun` door, the typestate would be broken and THAT would be the
//! finding rather than this negative.
//!
//! # Why BOTH fits use ONE lambda
//!
//! The adversary's fit function takes `lambda: f64`. It never takes a `Regularization`, and
//! it never calls `head_input::resolve_lambda` for itself. This is load-bearing: with
//! `SklearnEquivalentC { c }`, lambda is `1/(2*c*n)`, so an adversary that re-resolved against
//! its own DUPLICATED row count would be minimizing a WEAKER penalty — roughly `1/(2*c*5n)`
//! at the multiplicities below. The coefficients would then differ for two reasons at once,
//! the control would differ too, and the whole set would prove nothing about multiplicity.
//! One lambda, resolved once against the unique row count, is what makes the control a
//! control.
//!
//! # The set is THREE elements, and all three run in every `cargo test`
//!
//! 1. **The negative** — on a multiplicity-skewed weighting the coefficients move, and the
//!    message names the skewed class and the observed multiplicity ratio.
//! 2. **The control** — on a UNIFORM multiplicity (every row twice) the two fits agree at the
//!    same lambda. Without it, "the coefficients moved" is equally satisfied by an adversary
//!    that is simply a different optimizer. Uniform multiplicity two, not one: a control in
//!    which both sides are literally the same call proves nothing.
//! 3. **The mirror** — the adversary at multiplicity ONE reproduces the trusted
//!    `fit_on_selection` head BITWISE, so the instrument is faithful and the trusted path is
//!    unperturbed by the adversary's existence.
//!
//! `#[cfg(test)]` and nothing weaker: 03-10's acceptance criteria reject a `#[doc(hidden)]`
//! test-support door on the shipped surface, and a pair-weighted fitter is the last thing that
//! should be reachable from one.

use aprender::classification::{MultinomialLogisticRegression, Regularization};
use aprender_contrastive_data::pairs::PairSampler;
use aprender_contrastive_data::select::Selection;

use super::config::ResolvedSetFitConfig;
use super::head_input::{self, FittedHead};
use super::test_fixtures as fx;

/// The pair budget the skew is drawn from.
///
/// 64 pairs over 24 selected rows is 128 endpoint draws — enough that endpoint frequency
/// diverges sharply from uniqueness, which is the whole quantity under test. The selection's
/// pair space is 276, so the budget binds rather than exhausts.
const SKEW_BUDGET: u64 = 64;

/// The trusted artifacts every element of the set is measured against.
struct Trusted {
    fitted: FittedHead,
    selection: Selection,
    config: ResolvedSetFitConfig,
}

/// The fixture's stage two, through the SHIPPED path.
///
/// `fit_on_selection` is the body `fit_head` runs. Measuring against it rather than against a
/// second fit written here is what makes "the trusted path" mean the trusted path.
fn trusted() -> Trusted {
    let dataset = fx::fixture_dataset();
    let selection = fx::fixture_selection(fx::FIXTURE_SEED, 8);
    let mut encoder = fx::slice_encoder(fx::FIXTURE_SEED);
    let variant = fx::CalibrationVariant { budget: SKEW_BUDGET, ..fx::calibrated_variant() };
    let config = fx::config_for(variant, None)
        .resolve()
        .expect("the fixture configuration resolves on a cpu host");
    let fitted = head_input::fit_on_selection(
        &mut encoder,
        &dataset,
        &selection,
        &config,
        head_input::HEAD_MAX_ITER,
    )
    .expect("the trusted exactly-once head must fit");
    Trusted { fitted, selection, config }
}

/// THE adversary's fitter: the same head, on rows that may repeat.
///
/// It takes `lambda: f64` and NOT a `Regularization`. See the module doc: a `Regularization`
/// here would be re-resolved against the duplicated row count and the control would stop
/// isolating multiplicity.
fn fit_at_lambda(
    features: &[Vec<f32>],
    class_indices: &[usize],
    ordered_labels: &[String],
    lambda: f64,
) -> MultinomialLogisticRegression {
    let mut head = MultinomialLogisticRegression::new(ordered_labels.len())
        .with_max_iter(head_input::HEAD_MAX_ITER);
    head.fit(features, class_indices, ordered_labels, Regularization::Lambda(lambda))
        .expect("the pair-weighted fit must converge, or it measures the optimizer not the data");
    head
}

/// How often each selected row appears as a pair endpoint, over the whole budgeted stream.
///
/// This is what "pair multiplicity" MEANS: the number of times the contrastive stage touched
/// a row. Replaying the stream rather than inventing weights is what makes the negative about
/// the real sampler.
fn endpoint_multiplicities(t: &Trusted) -> Vec<usize> {
    let sampler = PairSampler::new(&t.selection, t.config.requested().pair_config())
        .expect("24 rows support a 64-pair budget");
    let mut counts = vec![0_usize; t.selection.len()];
    for labeled in sampler.iter_from(0).expect("offset 0 is within the budget") {
        counts[labeled.pair.lo().ordinal() as usize] += 1;
        counts[labeled.pair.hi().ordinal() as usize] += 1;
    }
    // A row the stream never touched would VANISH rather than be down-weighted, and a whole
    // class vanishing is a different failure (an unrepresented class) than the one under
    // test. The floor keeps the difference attributable to reweighting.
    counts.iter().map(|&c| c.max(1)).collect()
}

/// The head's rows, each repeated according to `multiplicity`.
fn replicated(t: &Trusted, multiplicity: &[usize]) -> (Vec<Vec<f32>>, Vec<usize>) {
    let input = &t.fitted.input;
    let total: usize = multiplicity.iter().sum();
    let mut features = Vec::with_capacity(total);
    let mut classes = Vec::with_capacity(total);
    for (row, &times) in multiplicity.iter().enumerate() {
        for _ in 0..times {
            features.push(input.embeddings()[row].clone());
            classes.push(input.class_indices()[row]);
        }
    }
    (features, classes)
}

/// Largest absolute coefficient difference between two fitted heads.
fn max_coefficient_distance(
    a: &MultinomialLogisticRegression,
    b: &MultinomialLogisticRegression,
) -> f64 {
    fn abs_diffs<'a>(x: &'a [f32], y: &'a [f32]) -> impl Iterator<Item = f64> + 'a {
        x.iter().zip(y).map(|(p, q)| f64::from((p - q).abs()))
    }
    abs_diffs(a.weights(), b.weights())
        .chain(abs_diffs(a.intercepts(), b.intercepts()))
        .fold(0.0_f64, f64::max)
}

/// Total endpoint multiplicity per class, and the (class, ratio) the skew is worst at.
fn class_totals(t: &Trusted, multiplicity: &[usize]) -> (Vec<usize>, usize, f64) {
    let labels = t.fitted.input.ordered_labels().len();
    let mut totals = vec![0_usize; labels];
    for (row, &times) in multiplicity.iter().enumerate() {
        totals[t.fitted.input.class_indices()[row]] += times;
    }
    let hi = totals.iter().copied().max().unwrap_or(0);
    let lo = totals.iter().copied().min().unwrap_or(0);
    // `position`, not `max_by_key`: the first class at the maximum, matching the tie-break
    // the assertion message reports.
    let worst = totals.iter().position(|&t| t == hi).unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    let ratio = hi as f64 / lo.max(1) as f64;
    (totals, worst, ratio)
}

/// The lambda BOTH sides fit under: resolved once, from the UNIQUE row count.
fn shared_lambda(t: &Trusted) -> f64 {
    let lambda =
        head_input::resolve_lambda(&t.config.requested().head_regularization(), t.fitted.input.n());
    assert_eq!(lambda, t.fitted.lambda, "the trusted fit's own lambda");
    lambda
}

// ===========================================================================================
// The mirror. Stated first, because both elements below are only meaningful relative to it.
// ===========================================================================================

/// MIRROR — the adversary at multiplicity ONE reproduces the trusted head BITWISE.
///
/// Two things at once. The instrument is faithful: `fit_at_lambda` differs from the shipped
/// stage-two fit in nothing but the rows it is handed, so a difference reported below is a
/// difference in the rows. And the trusted path is unperturbed: its stored `f32` coefficients
/// are exactly what they are with the adversary present, which is the compiled-in form of
/// "byte-identical with and without the adversary".
#[test]
fn pair_weight_mirror_the_trusted_path_is_unperturbed() {
    let t = trusted();
    let lambda = shared_lambda(&t);
    let ones = vec![1_usize; t.selection.len()];
    let (features, classes) = replicated(&t, &ones);

    assert_eq!(features.len(), t.selection.len(), "multiplicity one is the trusted row set");
    let mirrored = fit_at_lambda(&features, &classes, t.fitted.input.ordered_labels(), lambda);

    assert_eq!(
        mirrored.weights(),
        t.fitted.head.weights(),
        "at multiplicity one the adversary must reproduce the shipped fit BITWISE; if it \
         cannot, every distance it reports below is confounded by the instrument",
    );
    assert_eq!(mirrored.intercepts(), t.fitted.head.intercepts());
    assert_eq!(t.fitted.input.encode_ledger().len(), t.selection.len());
}

// ===========================================================================================
// The control, then the negative.
// ===========================================================================================

/// CONTROL — a UNIFORM multiplicity changes nothing, at the same lambda.
///
/// Every row twice. The objective is `(1/2n) * sum over 2n duplicated rows + lambda*||W||^2`,
/// which is the same function of `W` as the trusted `(1/n) * sum over n rows + lambda*||W||^2`
/// — so a solver that reads the DATA rather than the ROW COUNT lands in the same place.
///
/// Multiplicity two rather than one on purpose: at one the two calls would be the same call
/// on the same rows and the control would be trivially green.
#[test]
fn pair_weight_control_uniform_multiplicity_agrees_at_one_lambda() {
    let t = trusted();
    let lambda = shared_lambda(&t);
    let uniform = vec![2_usize; t.selection.len()];
    let (features, classes) = replicated(&t, &uniform);

    assert_eq!(features.len(), 2 * t.selection.len(), "the control must really duplicate");
    let doubled = fit_at_lambda(&features, &classes, t.fitted.input.ordered_labels(), lambda);
    let distance = max_coefficient_distance(&doubled, &t.fitted.head);

    println!("CONTROL uniform-multiplicity coefficient distance = {distance:e}");
    assert!(
        distance < UNIFORM_TOLERANCE,
        "a uniform multiplicity must not move the head at a fixed lambda: observed max \
         coefficient distance {distance:e} exceeds {UNIFORM_TOLERANCE:e}. If this is red, the \
         negative below is measuring the optimizer or the lambda, not multiplicity.",
    );
}

/// NEGATIVE — a multiplicity-SKEWED weighting moves the head, at an identical lambda.
///
/// The weights are the real sampler's endpoint frequencies over the budgeted stream, not
/// invented numbers. The message names the class the skew is worst at and the observed ratio,
/// because a red gate nobody can diagnose is a gate that gets deleted.
#[test]
fn pair_weighted_multiplicity_skew_moves_the_head() {
    let t = trusted();
    let lambda = shared_lambda(&t);
    let multiplicity = endpoint_multiplicities(&t);
    let (totals, worst, ratio) = class_totals(&t, &multiplicity);

    // Vacuity guards, BEFORE any claim is made about the fit. A stream that happened to touch
    // every row equally would make the negative green for no reason at all.
    let (lo, hi) = (
        *multiplicity.iter().min().expect("24 rows"),
        *multiplicity.iter().max().expect("24 rows"),
    );
    assert!(
        hi >= 2 * lo,
        "the replayed stream is not skewed (per-row multiplicity {lo}..{hi}), so this test \
         would prove nothing about multiplicity",
    );
    assert!(ratio > 1.0, "per-class totals {totals:?} are uniform, so there is no skew to see");

    let (features, classes) = replicated(&t, &multiplicity);
    assert_eq!(features.len(), multiplicity.iter().sum::<usize>());
    let weighted = fit_at_lambda(&features, &classes, t.fitted.input.ordered_labels(), lambda);
    let distance = max_coefficient_distance(&weighted, &t.fitted.head);

    #[allow(clippy::cast_precision_loss)]
    let row_ratio = hi as f64 / lo as f64;
    println!(
        "NEGATIVE skewed-multiplicity coefficient distance = {distance:e} \
         (per-row multiplicity {lo}..{hi}, ratio {row_ratio:.3}x; per-class totals {totals:?}, \
         worst class {worst}, ratio {ratio:.3}x; {} weighted rows from {} unique)",
        features.len(),
        t.selection.len(),
    );
    assert!(
        distance > SKEW_FLOOR,
        "the pair-weighted head is within {SKEW_FLOOR:e} of the exactly-once head at the \
         SAME lambda ({lambda:e}), so pair multiplicity would NOT reweight the head data and \
         TRN-05's structural claim is about nothing. The skew is real: rows were drawn as \
         endpoints between {lo} and {hi} times ({row_ratio:.3}x), and class `{}` (index \
         {worst}) carries {} of the {} endpoint draws — a {ratio:.3}x per-class multiplicity \
         ratio. Observed coefficient distance {distance:e}.",
        t.fitted.input.ordered_labels()[worst],
        totals[worst],
        totals.iter().sum::<usize>(),
    );
}

// ===========================================================================================
// The discipline itself, read off the source.
//
// Deliberately NOT named `pair_weight*`: the set is exactly three elements and
// `cargo test pair_weight` must run exactly those three.
// ===========================================================================================

/// The adversary takes a bare `f64`, so it CANNOT re-resolve lambda against duplicated rows.
///
/// This is the review's third fix made structural. If the signature took a `Regularization`,
/// `SklearnEquivalentC` would resolve against 129 rows instead of 24 — a penalty ~5x weaker —
/// and both the negative AND the control would move for a reason that is not multiplicity.
#[test]
fn adversary_fitter_takes_a_bare_lambda_not_a_regularization() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/train/setfit/negative.rs");
    let text = std::fs::read_to_string(&path).expect("this module is readable");

    // Assembled so the scan finds the DEFINITION, not this assertion.
    let signature_start = format!("{}{}", "fn fit_at_", "lambda(");
    let start = text.find(&signature_start).expect("the adversary's fitter is defined here");
    let rest = &text[start..];
    let params = &rest[..rest.find(") -> ").expect("the fitter's parameter list is closed")];
    assert!(params.contains("lambda: f64"), "the adversary must take a bare lambda: {params}");
    assert!(
        !params.contains("Regularization"),
        "the adversary must NOT take a Regularization; it would re-resolve against its own \
         duplicated row count and the control would stop isolating multiplicity: {params}",
    );

    // The module doc has to carry the expressible-surface analysis, because a reader who does
    // not know WHY the attack cannot go through the transition will eventually "simplify" it.
    let module_doc: String = text
        .lines()
        .filter(|line| line.trim_start().starts_with("//!"))
        .collect::<Vec<_>>()
        .join("\n");
    for token in ["fit_head", "lambda", "MultinomialLogisticRegression"] {
        assert!(module_doc.contains(token), "the module doc must discuss `{token}`");
    }
}

/// The control's tolerance.
///
/// MEASURED: the observed distance is exactly `0e0`, and that is expected rather than
/// suspicious. Doubling every row doubles the NLL sum EXACTLY — scaling by a power of two is
/// exact in binary floating point and commutes with rounding — and the `1/(2n)` mean then
/// divides it back, so the objective and its analytic gradient are bitwise identical to the
/// 24-row problem. L-BFGS starts from a fixed zero vector, so it walks the identical
/// trajectory. The tolerance is kept rather than asserting `== 0.0` because that argument
/// assumes no subnormal or overflow anywhere in the accumulation.
///
/// This is NOT the trivially-green control the plan warns about: the two calls are handed 48
/// rows and 24 rows respectively (asserted), and the SAME `fit_at_lambda` returns a distance
/// of 8.5e-2 in the negative below.
const UNIFORM_TOLERANCE: f64 = 1e-6;

/// The floor the skewed difference must clear.
///
/// MEASURED: the observed distance is 8.507794e-2, so the floor sits ~85x below the signal and
/// three orders of magnitude above the control's agreement. The two elements of the set cannot
/// be satisfied by the same number.
const SKEW_FLOOR: f64 = 1e-3;
