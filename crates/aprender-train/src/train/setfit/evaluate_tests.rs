//! Canonical-validation evaluation tests (plan 03-09, TRN-07).
//!
//! Every test name starts `evaluate_`, which is the filter this plan's Task 1 verification
//! runs.
//!
//! # What the source assertions are for
//!
//! Three of the claims this module makes are about what does NOT exist: no public constructor
//! on the evaluation, no API taking a metric value, no `String` variant on the metric kind. A
//! property of that shape cannot be witnessed by calling something, so it is asserted against
//! the module's own source text. That is weaker than a type error and stronger than a comment,
//! and it runs in every `cargo test` rather than living in a review checklist.

use super::super::test_fixtures as fx;
use super::*;

// The two source scanners the non-existence assertions read with. Shared with `lock_tests.rs`
// through the fixture module rather than restated here — see `fx::source_block_after` for why
// one copy is the point.
use fx::{source_block_after as block_after, source_signature_after as signature_after};

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::prepared::CanonicalDeclarations;
use aprender_contrastive_data::split::SplitDeclaration;

/// This module's own source, for the three non-existence assertions.
const EVALUATE_SOURCE: &str = include_str!("evaluate.rs");

// ===========================================================================================
// Fixtures
// ===========================================================================================

/// A complete calibrated pipeline through the shipped codec.
fn verified_run() -> SetFitRun<ArtifactReloadedAndVerified> {
    fx::verified_run(fx::calibrated_variant())
}

/// The fixture dataset, rebuilt INDEPENDENTLY of the run's copy.
///
/// Deliberately not `run.dataset()`: passing the run's own dataset back in would make the
/// evaluator's agreement check pass by identity, and the property being relied on is that it
/// passes by FINGERPRINT.
fn fixture_dataset() -> PreparedDataset<Canonical> {
    fx::fixture_dataset()
}

/// The same corpus with ONE validation row's bytes changed.
///
/// Same ids, same label map, same train and test splits — so the only thing that can make the
/// evaluator refuse it is the fingerprint pair, which is the thing under test.
fn dataset_with_altered_validation() -> PreparedDataset<Canonical> {
    let base = fixture_dataset();
    let label_names = base.label_names().to_vec();
    let classes = label_names.len();
    let train = base.train().rows().to_vec();
    let mut validation = base.validation().rows().to_vec();
    let test = base.test().rows().to_vec();
    validation[0].input = format!("{} and again .", validation[0].input);

    let train_per_class = train.len() / classes;
    let decl = |per_class: usize| SplitDeclaration {
        expected_class_counts: vec![per_class; classes],
        label_names: label_names.clone(),
    };
    let mut ledger = AccessLedger::new();
    PreparedDataset::<Canonical>::from_labeled_rows(
        train,
        validation,
        test,
        &CanonicalDeclarations {
            train: decl(train_per_class),
            validation: decl(1),
            test: decl(1),
            label_names: label_names.clone(),
        },
        &mut ledger,
    )
    .expect("altering one validation row must still yield a valid canonical dataset")
}

// ===========================================================================================
// The computed metric
// ===========================================================================================

/// Accuracy is computed over every validation row, and the evaluation commits its provenance.
#[test]
fn evaluate_accuracy_is_computed_over_every_validation_row() {
    let run = verified_run();
    let dataset = fixture_dataset();
    let evaluation = evaluate_validation(&run, &dataset, ValidationMetricKind::Accuracy)
        .expect("the fixture dataset is the one the run was prepared from");

    assert_eq!(evaluation.metric_kind(), ValidationMetricKind::Accuracy);
    assert_eq!(
        evaluation.n_rows(),
        dataset.validation().rows().len(),
        "every validation row must be predicted",
    );
    assert!(
        (0.0..=1.0).contains(&evaluation.value()),
        "accuracy is a fraction, got {}",
        evaluation.value(),
    );

    let witness = dataset.validation_witness();
    assert_eq!(
        evaluation.artifact_hash(),
        run.artifact_hash(),
        "the artifact hash must be READ OFF the run, never supplied",
    );
    assert_eq!(evaluation.validation_split_fingerprint(), witness.fingerprint_hex());
    assert_eq!(evaluation.dataset_fingerprint(), witness.dataset_fingerprint_hex());
    assert_ne!(
        evaluation.validation_split_fingerprint(),
        evaluation.dataset_fingerprint(),
        "Phase 2 made these two deliberately different values; committing one twice would \
         collapse the lock's two consistency checks into one",
    );
}

/// Macro-F1 is computed over the same rows and is a finite fraction.
#[test]
fn evaluate_macro_f1_is_computed_over_every_validation_row() {
    let run = verified_run();
    let dataset = fixture_dataset();
    let evaluation = evaluate_validation(&run, &dataset, ValidationMetricKind::MacroF1)
        .expect("the fixture dataset is the one the run was prepared from");

    assert_eq!(evaluation.metric_kind(), ValidationMetricKind::MacroF1);
    assert_eq!(evaluation.n_rows(), dataset.validation().rows().len());
    assert!(evaluation.value().is_finite(), "a macro-F1 must never be NaN");
    assert!(
        (0.0..=1.0).contains(&evaluation.value()),
        "macro-F1 is a fraction, got {}",
        evaluation.value(),
    );
}

/// Two evaluations of the same run agree down to the value's bit pattern.
#[test]
fn evaluate_is_deterministic_including_the_value_bit_pattern() {
    let run = verified_run();
    let dataset = fixture_dataset();
    let first = evaluate_validation(&run, &dataset, ValidationMetricKind::MacroF1)
        .expect("the first evaluation must succeed");
    let second = evaluate_validation(&run, &dataset, ValidationMetricKind::MacroF1)
        .expect("the second evaluation must succeed");

    assert_eq!(first, second, "the evaluation must be a pure function of its inputs");
    assert_eq!(
        first.value_bits(),
        second.value_bits(),
        "an assertion on the decimal form would accept two values one ULP apart",
    );
}

/// A dataset the run was not prepared from is refused, with both fingerprint pairs named.
#[test]
fn evaluate_refuses_a_dataset_the_run_was_not_prepared_from() {
    let run = verified_run();
    let other = dataset_with_altered_validation();

    let expected = run.dataset().validation_witness();
    let observed = other.validation_witness();
    assert_ne!(
        expected.fingerprint_hex(),
        observed.fingerprint_hex(),
        "the fixture must actually differ, or this test would pass vacuously",
    );

    let error = evaluate_validation(&run, &other, ValidationMetricKind::Accuracy)
        .expect_err("an evaluation against a different dataset is not evidence about this run");
    assert!(
        matches!(error, SetFitTrainError::ValidationDatasetMismatch { .. }),
        "expected a typed mismatch, got {error:?}",
    );

    let rendered = error.to_string();
    for fingerprint in [
        expected.fingerprint_hex(),
        observed.fingerprint_hex(),
        expected.dataset_fingerprint_hex(),
        observed.dataset_fingerprint_hex(),
    ] {
        assert!(
            rendered.contains(&fingerprint),
            "the rejection must name `{fingerprint}`; got `{rendered}`",
        );
    }
}

// ===========================================================================================
// The metric definitions
// ===========================================================================================

/// Accuracy against hand-computed values.
#[test]
fn evaluate_accuracy_matches_hand_computed_values() {
    assert_eq!(accuracy(&[0, 1, 2, 0], &[0, 1, 1, 2]), 0.5);
    assert_eq!(accuracy(&[0], &[0]), 1.0);
    assert_eq!(accuracy(&[0], &[1]), 0.0);
    assert_eq!(accuracy(&[2, 2, 2, 2], &[2, 2, 2, 2]), 1.0);
}

/// Macro-F1 against a hand-computed value, with the arithmetic written out.
///
/// truth `[0,0,1,1]`, predicted `[0,1,1,1]` over two classes.
/// Class 0: tp 1, fp 0, fn 1 -> 2*1 / (2*1 + 0 + 1) = 2/3.
/// Class 1: tp 2, fp 1, fn 0 -> 2*2 / (2*2 + 1 + 0) = 4/5.
/// Macro    = (2/3 + 4/5) / 2.
#[test]
fn evaluate_macro_f1_matches_a_hand_computed_value() {
    let observed = macro_f1(&[0, 0, 1, 1], &[0, 1, 1, 1], 2);
    let expected = (2.0_f64 / 3.0 + 4.0 / 5.0) / 2.0;
    assert_eq!(observed.to_bits(), expected.to_bits(), "expected {expected}, got {observed}",);
}

/// A declared class with NO actual and NO predicted positives scores 0.0, not NaN.
///
/// Both precision and recall are 0/0 for such a class. The convention here is 0.0 — the same
/// one `sklearn`'s `zero_division` default applies — because it keeps macro-F1 a TOTAL function
/// whose values are comparable across candidates. A NaN would propagate through the selection
/// rule's comparison and make every ordering meaningless; excluding the class instead would let
/// a candidate improve its score by declaring a class it never predicts.
#[test]
fn evaluate_macro_f1_defines_a_class_with_no_positives_at_all_as_zero() {
    let observed = macro_f1(&[0, 1], &[0, 1], 3);
    assert!(observed.is_finite(), "the empty-class convention must not produce NaN");
    assert_eq!(
        observed.to_bits(),
        (2.0_f64 / 3.0).to_bits(),
        "two perfect classes and one absent class average to 2/3, got {observed}",
    );
}

/// The empty-class convention is a CHOICE, and the alternative is visibly different.
#[test]
fn evaluate_macro_f1_absent_class_convention_is_not_vacuous() {
    let with_absent_class = macro_f1(&[0, 1], &[0, 1], 3);
    let without_absent_class = macro_f1(&[0, 1], &[0, 1], 2);
    assert_eq!(without_absent_class, 1.0);
    assert!(
        with_absent_class < without_absent_class,
        "declaring a class nobody predicts must cost macro-F1, or the convention is inert",
    );
}

// ===========================================================================================
// The non-existence assertions
// ===========================================================================================

/// No public API in this module takes a float PARAMETER at all.
///
/// The scan reads each `pub fn`'s PARAMETER LIST rather than the whole file, because the
/// evaluation's own private `value: f64` FIELD is the number this plan exists to protect and a
/// naive substring search reports the field as the violation. Reading the parameter list is
/// also what makes the assertion survive a multi-line signature, where the offending parameter
/// would not share a line with `pub fn` at all.
#[test]
fn evaluate_source_exposes_no_public_api_taking_a_float_parameter() {
    let mut checked = 0_usize;
    // BOTH spellings. `pub const fn` does not contain the substring `pub fn`, so a scan for the
    // latter alone silently skips every `const` accessor — measured: it saw 5 of the 9.
    for pattern in ["pub fn ", "pub const fn "] {
        for (offset, _) in EVALUATE_SOURCE.match_indices(pattern) {
            let rest = &EVALUATE_SOURCE[offset..];
            let open = rest.find('(').expect("a function signature has a parameter list");
            let close = rest[open..].find(')').expect("a parameter list closes") + open;
            let parameters = &rest[open..=close];
            let name = rest[..open].trim_end();
            assert!(
                !parameters.contains("f64") && !parameters.contains("f32"),
                "`{name}` takes a float parameter: `{parameters}`. A metric a caller can hand \
                 over is exactly the asserted number this plan removed.",
            );
            checked += 1;
        }
    }
    assert!(checked >= 9, "the scan must have found the public surface, saw {checked}");
    assert!(!EVALUATE_SOURCE.contains("metric_value:"));

    // The ONE float-taking door in this module is the lock tests' fixture constructor. The guard
    // NAMES it and asserts it is `#[cfg(test)]`-gated and not `pub`, rather than passing because
    // `pub(super) fn` happens not to match the patterns above — an exception a guard does not
    // mention is an exception nobody re-checks.
    assert_eq!(
        EVALUATE_SOURCE.matches("fn evaluation_for_tests(").count(),
        1,
        "exactly one test-only evaluation constructor",
    );
    assert!(
        EVALUATE_SOURCE.contains("#[cfg(test)]\npub(super) fn evaluation_for_tests("),
        "the test-only constructor must carry `#[cfg(test)]` immediately above it, so a shipped \
         build cannot contain a door that mints an evaluation from a caller's number",
    );

    // The SECOND `pub(super)` door, added by 04-07 so a fresh process can build a candidate.
    // It is named here rather than left to escape the `pub fn` / `pub const fn` patterns
    // above, on this guard's own stated principle: an exception a guard does not mention is an
    // exception nobody re-checks. `pub(super)` matches neither pattern, so a float-taking
    // sibling door would have slipped past in silence.
    let shared_tail = EVALUATE_SOURCE
        .find("pub(super) fn evaluation_from_predictions(")
        .expect("the shared prediction tail must exist; if it moved, fix this scan");
    let rest = &EVALUATE_SOURCE[shared_tail..];
    let open = rest.find('(').expect("a function signature has a parameter list");
    let close = rest[open..].find(')').expect("a parameter list closes") + open;
    let parameters = &rest[open..=close];
    assert!(
        !parameters.contains("f64") && !parameters.contains("f32"),
        "`evaluation_from_predictions` must take PREDICTIONS, never a value: `{parameters}`. \
         A metric a sibling module can hand over is the asserted number this module removed, \
         one call frame further out.",
    );
    // The complete `pub(super)` surface, enumerated: `from_wire` (the lock's reconstruction
    // path, which takes the wire struct and therefore BITS), `evaluation_for_tests` (gated),
    // and `evaluation_from_predictions` (the shared tail). A fourth must be examined against
    // the same float rule rather than added silently.
    for door in [
        "pub(super) fn from_wire(",
        "pub(super) fn evaluation_for_tests(",
        "pub(super) fn evaluation_from_predictions(",
    ] {
        assert_eq!(EVALUATE_SOURCE.matches(door).count(), 1, "`{door}` must exist exactly once",);
    }
    assert_eq!(
        EVALUATE_SOURCE.matches("pub(super) fn ").count(),
        3,
        "exactly three `pub(super)` doors in this module, all enumerated above",
    );
}

/// `ValidationEvaluation` has no public constructor.
#[test]
fn evaluate_source_has_no_public_constructor_on_the_evaluation() {
    let block = block_after(EVALUATE_SOURCE, "impl ValidationEvaluation {");
    assert!(
        !block.contains("pub fn new"),
        "the evaluator is the only production path to a ValidationEvaluation",
    );
    assert!(!block.contains("pub const fn new"));
    let declaration = block_after(EVALUATE_SOURCE, "pub struct ValidationEvaluation {");
    for field in declaration.lines().skip(1) {
        assert!(
            !field.trim_start().starts_with("pub "),
            "every field must be private; found `{field}`",
        );
    }
}

/// The evaluator's signature takes BOTH the verified run and the canonical dataset.
#[test]
fn evaluate_signature_takes_both_the_verified_run_and_the_dataset() {
    let signature = signature_after(EVALUATE_SOURCE, "pub fn evaluate_validation(");
    assert!(signature.contains("&SetFitRun<ArtifactReloadedAndVerified>"), "got `{signature}`",);
    assert!(signature.contains("&PreparedDataset<Canonical>"), "got `{signature}`");
    assert!(signature.contains("ValidationMetricKind"), "got `{signature}`");
}

/// `ValidationMetricKind` is closed, and cannot carry a free-form name.
#[test]
fn evaluate_metric_kind_is_a_closed_enum_with_no_string_variant() {
    let block = block_after(EVALUATE_SOURCE, "pub enum ValidationMetricKind {");
    assert!(
        !block.contains("String"),
        "a String-carrying variant would let two different quantities share a label",
    );
    assert!(!block.contains("#[non_exhaustive]"));

    // An exhaustive match with no wildcard: a new variant makes this fail to COMPILE.
    for kind in [ValidationMetricKind::Accuracy, ValidationMetricKind::MacroF1] {
        let tag = match kind {
            ValidationMetricKind::Accuracy => "accuracy",
            ValidationMetricKind::MacroF1 => "macro_f1",
        };
        assert_eq!(kind.tag(), tag);
    }
}

/// No wall-clock type appears in this module, so its output cannot drift with the clock.
#[test]
fn evaluate_source_carries_no_wall_clock_type() {
    for forbidden in ["Dur\u{61}tion", "Inst\u{61}nt", "SystemTim\u{65}"] {
        assert!(
            !EVALUATE_SOURCE.contains(forbidden),
            "a wall-clock type in the evaluation would make its canonical bytes irreproducible",
        );
    }
}

// ===========================================================================================
// The canonical wire form
// ===========================================================================================

/// The serialized evaluation carries the value's BITS, the schema version and the metric tag.
#[test]
fn evaluate_wire_form_carries_the_value_bit_pattern() {
    let run = verified_run();
    let dataset = fixture_dataset();
    let evaluation = evaluate_validation(&run, &dataset, ValidationMetricKind::Accuracy)
        .expect("the fixture dataset is the one the run was prepared from");

    let json = serde_json::to_string(&evaluation).expect("the evaluation serializes");
    assert!(json.contains(&format!("\"value_bits\":{}", evaluation.value_bits())), "{json}");
    assert!(json.contains("\"schema_version\":1"), "{json}");
    assert!(json.contains("\"metric_kind\":\"accuracy\""), "{json}");
    assert!(
        !json.contains("\"value\":"),
        "a decimal rendering in the canonical form would bind the digest to a float formatter",
    );
    assert!(json.contains(evaluation.artifact_hash()), "{json}");
    assert!(json.contains(evaluation.validation_split_fingerprint()), "{json}");
}
