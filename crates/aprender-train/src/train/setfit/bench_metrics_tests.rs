//! `QualityBlock` assembly tests (plan 05-08, EVAL-01).
//!
//! Every test name starts `bench_metrics_`.
//!
//! # The toy cases are HAND-computed, and the derivation is in the test
//!
//! A metric test whose expected value was produced by the code under test proves only that the
//! code is deterministic. Every number asserted below is derived in a comment from the counts,
//! so a reader can check it without running anything — and so a future refactor that changes a
//! convention has to argue with arithmetic rather than re-record a baseline.

use super::super::apr_evaluate::row_predictions_for_tests;
use super::*;

/// The contract's declared label order, which the evidence test below pins.
fn labels() -> Vec<String> {
    ["none", "against", "favor"].iter().map(|s| (*s).to_string()).collect()
}

/// The six-row TEST case whose every metric is hand-derived in the comments below.
///
/// ```text
/// row | truth | pred
///   0 |   0   |  0
///   1 |   0   |  1
///   2 |   1   |  1
///   3 |   1   |  1
///   4 |   2   |  0
///   5 |   2   |  2
///
/// confusion (true x pred):   [[1, 1, 0],
///                             [0, 2, 0],
///                             [1, 0, 1]]
/// ```
fn toy_test_rows() -> RowPredictions {
    let truth = vec![0, 0, 1, 1, 2, 2];
    let predicted = vec![0, 1, 1, 1, 0, 2];
    // The test split's probabilities play no part in any metric below (calibration is
    // validation-only, D-07), so they are the one-hot form of the predictions — the shape a
    // reader can check against `predicted` at a glance.
    let probabilities: Vec<Vec<f64>> = predicted
        .iter()
        .map(|&k| {
            let mut row = vec![0.0; 3];
            row[k] = 1.0;
            row
        })
        .collect();
    row_predictions_for_tests(predicted, probabilities, truth, labels(), "artifact-sha", "test")
}

/// The four-row VALIDATION case, whose ECE and Brier are hand-derived below.
///
/// ```text
/// row | probabilities        | truth | conf | pred | correct
///   0 | [0.75, 0.15, 0.10]   |   0   | 0.75 |  0   | yes
///   1 | [0.10, 0.75, 0.15]   |   1   | 0.75 |  1   | yes
///   2 | [0.05, 0.10, 0.85]   |   0   | 0.85 |  2   | NO
///   3 | [0.30, 0.45, 0.25]   |   1   | 0.45 |  1   | yes
/// ```
///
/// Every top confidence sits at a bin CENTRE (0.75 -> bin 7, 0.85 -> bin 8, 0.45 -> bin 4 at
/// `n_bins = 10`), 0.05 away from the nearest edge. That is deliberate: a confidence on a bin
/// boundary makes the assertion an f32-rounding coin flip rather than a metric check.
fn toy_validation_rows() -> RowPredictions {
    let probabilities = vec![
        vec![0.75, 0.15, 0.10],
        vec![0.10, 0.75, 0.15],
        vec![0.05, 0.10, 0.85],
        vec![0.30, 0.45, 0.25],
    ];
    let predicted = vec![0, 1, 2, 1];
    let truth = vec![0, 1, 0, 1];
    row_predictions_for_tests(
        predicted,
        probabilities,
        truth,
        labels(),
        "artifact-sha",
        "validation",
    )
}

/// Assemble the toy pair, or panic with the refusal.
fn toy_block() -> QualityBlock {
    assemble_quality_block(&toy_test_rows(), &toy_validation_rows(), &labels())
        .expect("the toy pair is well-formed")
}

// ===========================================================================================
// The hand-computed case
// ===========================================================================================

/// Per-class precision, recall and F1 are the counts, worked out by hand.
///
/// ```text
/// class 0 (none):    TP=1  FP=1  FN=1  ->  P=1/2      R=1/2      F1=0.5
/// class 1 (against): TP=2  FP=1  FN=0  ->  P=2/3      R=1        F1=2*(2/3)/(5/3) = 0.8
/// class 2 (favor):   TP=1  FP=0  FN=1  ->  P=1        R=1/2      F1=2*(1/2)/(3/2) = 2/3
/// ```
#[test]
fn bench_metrics_per_class_values_are_the_hand_computed_counts() {
    let block = toy_block();

    let expected_precision = [0.5, 2.0 / 3.0, 1.0];
    let expected_recall = [0.5, 1.0, 0.5];
    let expected_f1 = [0.5, 0.8, 2.0 / 3.0];

    assert_eq!(block.per_class_precision.len(), 3, "one entry per ordered label");
    for (i, expected) in expected_precision.iter().enumerate() {
        assert!(
            (block.per_class_precision[i] - expected).abs() < 1e-12,
            "precision[{i}] = {} but the counts give {expected}",
            block.per_class_precision[i],
        );
    }
    for (i, expected) in expected_recall.iter().enumerate() {
        assert!(
            (block.per_class_recall[i] - expected).abs() < 1e-12,
            "recall[{i}] = {} but the counts give {expected}",
            block.per_class_recall[i],
        );
    }
    for (i, expected) in expected_f1.iter().enumerate() {
        assert!(
            (block.per_class_f1[i] - expected).abs() < 1e-12,
            "f1[{i}] = {} but the counts give {expected}",
            block.per_class_f1[i],
        );
    }
}

/// `F_avg` is `(F1_against + F1_favor) / 2` — classes 1 and 2, NOT the three-class mean.
///
/// ```text
/// F_avg    = (0.8 + 2/3) / 2         = 0.7333333333333334
/// macro_F1 = (0.5 + 0.8 + 2/3) / 3   = 0.6555555555555556
/// ```
#[test]
fn bench_metrics_f_avg_is_the_official_two_class_average_and_not_the_macro() {
    let block = toy_block();

    let expected_f_avg = (0.8 + 2.0 / 3.0) / 2.0;
    let expected_macro = (0.5 + 0.8 + 2.0 / 3.0) / 3.0;

    assert!(
        (block.f_avg - expected_f_avg).abs() < 1e-12,
        "f_avg = {} but the official (F1_against + F1_favor)/2 is {expected_f_avg}",
        block.f_avg,
    );
    assert!(
        (block.macro_f1 - expected_macro).abs() < 1e-12,
        "macro_f1 = {} but the three-class mean is {expected_macro}",
        block.macro_f1,
    );
    assert!(
        (block.f_avg - block.macro_f1).abs() > 1e-9,
        "non-vacuity: on this case the two MUST differ, or a macro-for-F_avg substitution would \
         pass every assertion above",
    );
    assert_eq!(block.f_avg_bits, block.f_avg.to_bits(), "the bits siblings must be the bits");
    assert_eq!(block.macro_f1_bits, block.macro_f1.to_bits());
}

/// The confusion matrix is the counts, row-major true-by-predicted.
#[test]
fn bench_metrics_confusion_matrix_is_true_by_predicted_row_major() {
    let block = toy_block();
    assert_eq!(
        block.confusion_matrix,
        vec![vec![1_u64, 1, 0], vec![0, 2, 0], vec![1, 0, 1]],
        "element [i][j] is the count of TRUE i predicted as j; a transposed matrix would swap \
         every precision with its recall downstream",
    );
    assert_eq!(block.n_test_rows, 6);
    assert_eq!(block.ordered_labels, labels());
}

/// MCC over the three classes, from the same counts.
///
/// ```text
/// c = 1 + 2 + 1 = 4          (correct)
/// s = 6                      (total)
/// predicted totals p = [2, 3, 1]; true totals t = [2, 2, 2]
///
/// MCC = (c*s - sum_k p_k t_k) / sqrt((s^2 - sum p_k^2)(s^2 - sum t_k^2))
///     = (24 - 12) / sqrt((36 - 14)(36 - 12))
///     = 12 / sqrt(528)
///     = 0.5222329678670935
/// ```
#[test]
fn bench_metrics_mcc_is_the_hand_computed_coefficient() {
    let block = toy_block();
    let expected = 12.0 / 528.0_f64.sqrt();
    assert!(
        (block.mcc - expected).abs() < 1e-6,
        "mcc = {} but the counts give {expected}. (Tolerance is 1e-6 because the shipped \
         `matthews_corrcoef` accumulates in f32.)",
        block.mcc,
    );
    assert_eq!(block.mcc_bits, block.mcc.to_bits());
}

/// ECE and Brier come from the VALIDATION rows, hand-derived from the four-row case.
///
/// ```text
/// n_bins = 10, equal width. bin(conf) = min(floor(conf * 10), 9).
///
/// bin 7: rows 0 and 1, conf 0.75, both correct -> |1.00 - 0.75| = 0.25, weight 2/4
/// bin 8: row 2,        conf 0.85, wrong        -> |0.00 - 0.85| = 0.85, weight 1/4
/// bin 4: row 3,        conf 0.45, correct      -> |1.00 - 0.45| = 0.55, weight 1/4
/// ECE  = 0.5*0.25 + 0.25*0.85 + 0.25*0.55 = 0.125 + 0.2125 + 0.1375 = 0.475
///
/// Brier (UNNORMALISED, codomain [0, 2]):
///   row 0 (truth 0): (0.75-1)^2 + 0.15^2 + 0.10^2 = 0.0625 + 0.0225 + 0.0100 = 0.0950
///   row 1 (truth 1): 0.10^2 + (0.75-1)^2 + 0.15^2 = 0.0100 + 0.0625 + 0.0225 = 0.0950
///   row 2 (truth 0): (0.05-1)^2 + 0.10^2 + 0.85^2 = 0.9025 + 0.0100 + 0.7225 = 1.6350
///   row 3 (truth 1): 0.30^2 + (0.45-1)^2 + 0.25^2 = 0.0900 + 0.3025 + 0.0625 = 0.4550
///   BS = 2.28 / 4 = 0.57
/// ```
#[test]
fn bench_metrics_calibration_is_the_hand_computed_validation_diagnostics() {
    let block = toy_block();
    assert!(
        (block.ece_top_label_validation - 0.475).abs() < 1e-5,
        "ECE = {} but the bin table gives 0.475",
        block.ece_top_label_validation,
    );
    assert!(
        (block.brier_multiclass_validation - 0.57).abs() < 1e-5,
        "Brier = {} but the row sums give 0.57",
        block.brier_multiclass_validation,
    );
    assert_eq!(block.ece_top_label_validation_bits, block.ece_top_label_validation.to_bits());
    assert_eq!(block.brier_multiclass_validation_bits, block.brier_multiclass_validation.to_bits());
}

/// Calibration is structurally validation-only (D-07).
#[test]
fn bench_metrics_calibration_split_is_validation_and_test_probabilities_cannot_reach_it() {
    let block = toy_block();
    assert_eq!(
        block.calibration_split, CALIBRATION_SPLIT,
        "the row must record the split its calibration was measured on, and there is only one \
         legal value",
    );

    // The two splits are SEPARATE PARAMETERS, so the only way to calibrate on test data is to
    // pass test rows in the validation position — which is refused by the split tag.
    let error = assemble_quality_block(&toy_test_rows(), &toy_test_rows(), &labels()).expect_err(
        "test probabilities in the validation position must be refused, not calibrated",
    );
    assert!(
        matches!(error, BenchMetricsError::WrongSplit { parameter: "validation_rows", .. }),
        "the refusal must name the parameter; got: {error}",
    );

    // And symmetrically: validation rows may not stand in for the test measurement.
    let swapped = assemble_quality_block(&toy_validation_rows(), &toy_validation_rows(), &labels())
        .expect_err("validation rows in the test position must be refused");
    assert!(
        matches!(swapped, BenchMetricsError::WrongSplit { parameter: "test_rows", .. }),
        "got: {swapped}",
    );
}

// ===========================================================================================
// Label ordering: two-sided
// ===========================================================================================

/// Ordering is EXPLICIT: the same predictions under a different label vector attribute the same
/// numbers to different classes, and the assembly refuses to guess which one the caller meant.
#[test]
fn bench_metrics_label_ordering_changes_attribution_in_both_directions() {
    let block = toy_block();

    // Side one: the declared vector must MATCH the evidence's own. A mismatch is a refusal
    // rather than a silent re-attribution — the two vectors disagreeing is exactly the state in
    // which every per-class number means something other than what it is labelled.
    let permuted: Vec<String> =
        ["none", "favor", "against"].iter().map(|s| (*s).to_string()).collect();
    assert_ne!(permuted, labels(), "non-vacuity: the probe order must actually differ");
    let error = assemble_quality_block(&toy_test_rows(), &toy_validation_rows(), &permuted)
        .expect_err("a declared order that is not the artifact's own must be refused");
    assert!(matches!(error, BenchMetricsError::LabelOrderMismatch { .. }), "got: {error}",);

    // Side two: when BOTH sides carry the permuted order, the assembly accepts it and the
    // numbers stay attached to their INDICES — so the IDENTICAL `f_avg` now averages `favor`
    // and `against` in the other order, over a map in which index 1 is `favor`. The number does
    // not move; its MEANING does. That is the whole reason the label vector has to be evidence
    // (the test below) rather than a caption a reader supplies.
    let permuted_test = row_predictions_for_tests(
        toy_test_rows().predicted().to_vec(),
        toy_test_rows().probabilities().to_vec(),
        toy_test_rows().truth().to_vec(),
        permuted.clone(),
        "artifact-sha",
        "test",
    );
    let permuted_validation = row_predictions_for_tests(
        toy_validation_rows().predicted().to_vec(),
        toy_validation_rows().probabilities().to_vec(),
        toy_validation_rows().truth().to_vec(),
        permuted.clone(),
        "artifact-sha",
        "validation",
    );
    let relabelled = assemble_quality_block(&permuted_test, &permuted_validation, &permuted)
        .expect("a consistently permuted map is well-formed, and MEANS something different");
    assert_eq!(
        relabelled.per_class_f1, block.per_class_f1,
        "the numbers are attached to INDICES; only their labels moved",
    );
    assert_eq!(
        relabelled.ordered_labels[1], "favor",
        "and under this map index 1 is `favor`, so `f_avg` is no longer the official score — \
         which is what the evidence test below exists to prevent",
    );
}

/// Label identity is EXACT byte equality: no case folding, no trimming, no normalization.
#[test]
fn bench_metrics_label_identity_is_exact_byte_equality() {
    let folded: Vec<String> =
        ["none", "Against", "favor"].iter().map(|s| (*s).to_string()).collect();
    let padded: Vec<String> =
        ["none", "against ", "favor"].iter().map(|s| (*s).to_string()).collect();
    for probe in [folded, padded] {
        assert_ne!(probe, labels(), "non-vacuity: the probe must differ from the declared map");
        let error = assemble_quality_block(&toy_test_rows(), &toy_validation_rows(), &probe)
            .expect_err(
                "a label differing only in case or whitespace is a DIFFERENT label; folding or \
                 trimming here would silently accept a relabelled head",
            );
        assert!(matches!(error, BenchMetricsError::LabelOrderMismatch { .. }), "got: {error}");
    }
}

// ===========================================================================================
// LABEL-ORDER EVIDENCE (review residual): the assumption behind `[1, 2]` becomes a fact
// ===========================================================================================

/// The ordered labels are read from the PINNED dataset revision's own declaration, not memory.
///
/// `tweet-eval-stance-benchmark-v1.yaml` carries both the class-label map and the
/// `canonical_revision` it describes. This test resolves the map from those bytes and asserts it
/// is exactly `["none", "against", "favor"]`, which is what makes
/// `f1_average_for_classes(&f1, &[1, 2])` provably `(F1_against + F1_favor) / 2` — the official
/// TweetEval F_avg — rather than an assumption.
///
/// It pins the revision too. If the pinned revision ever moves, this test goes red BEFORE any
/// headline number is computed, and whoever moves it has to re-confirm the `ClassLabel` order at
/// the new revision instead of inheriting this one.
#[test]
fn bench_metrics_label_order_is_evidence_from_the_pinned_dataset_revision() {
    let contract: serde_yaml::Value = serde_yaml::from_str(TWEET_EVAL_CONTRACT_YAML)
        .expect("the tweet-eval contract must parse as YAML");
    let dataset = contract.get("dataset").expect("the contract declares a `dataset` block");

    let revision = dataset
        .get("canonical_revision")
        .and_then(serde_yaml::Value::as_str)
        .expect("the dataset block declares `canonical_revision`");
    assert_eq!(
        revision, PINNED_TWEET_EVAL_REVISION,
        "the pinned TweetEval revision moved. The label ORDER at the new revision must be \
         re-confirmed against its own ClassLabel declaration before this constant is updated — \
         `f1_average_for_classes(&f1, &[1, 2])` is only the official F_avg while index 1 is \
         `against` and index 2 is `favor`",
    );

    let declared = dataset
        .get("labels")
        .and_then(serde_yaml::Value::as_mapping)
        .expect("the dataset block declares `labels` as an index -> name mapping");
    let mut pairs: Vec<(u64, String)> = declared
        .iter()
        .map(|(key, value)| {
            let index = key.as_u64().expect("every label key is an integer class index");
            let name = value.as_str().expect("every label name is a string").to_string();
            (index, name)
        })
        .collect();
    pairs.sort_by_key(|(index, _)| *index);

    assert_eq!(
        pairs.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        vec![0_u64, 1, 2],
        "the declared class indices must be exactly 0, 1, 2 — a gap or a 1-based map would make \
         the ordered vector below a different thing entirely",
    );
    let resolved: Vec<String> = pairs.into_iter().map(|(_, name)| name).collect();

    assert_eq!(
        resolved,
        vec!["none", "against", "favor"],
        "the pinned revision's own class-label order",
    );
    assert_eq!(resolved[1], "against", "index 1 — the first class `f_avg` averages");
    assert_eq!(resolved[2], "favor", "index 2 — the second class `f_avg` averages");
    assert_eq!(
        OFFICIAL_F_AVG_CLASSES.to_vec(),
        vec![1_usize, 2],
        "and those are the indices the assembly selects",
    );

    // The contract states the same thing in prose; asserting the FORMULA too means a future
    // edit that changed the prose without the map, or the map without the prose, is red.
    let formula = contract
        .get("equations")
        .and_then(|equations| equations.get("official_f_avg"))
        .and_then(|equation| equation.get("formula"))
        .and_then(serde_yaml::Value::as_str)
        .expect("the contract declares the official_f_avg formula");
    assert_eq!(formula, "F_avg = (F1_against + F1_favor) / 2");
}

// ===========================================================================================
// Degenerate splits
// ===========================================================================================

/// A class with ZERO support gets a DEFINED F1 — the shipped zero-division literal, not a NaN.
///
/// `MultiClassMetrics::from_confusion_matrix` computes `P = TP/(TP+FP)` guarded by
/// `TP+FP > 0`, `R = TP/(TP+FN)` guarded by `TP+FN > 0`, and `F1 = 2PR/(P+R)` guarded by
/// `P+R > 0`, each falling back to **0.0**. So an absent class contributes a literal
/// `0.0` — the same convention `sklearn`'s `zero_division=0` default applies — and NOT a NaN
/// that would poison `f_avg`, `macro_f1` and every ordering downstream.
#[test]
fn bench_metrics_a_zero_support_class_yields_the_shipped_zero_division_literal() {
    // Class 2 (`favor`) appears in neither the truth nor the predictions.
    let truth = vec![0, 0, 1, 1];
    let predicted = vec![0, 1, 1, 1];
    let probabilities: Vec<Vec<f64>> = predicted
        .iter()
        .map(|&k| {
            let mut row = vec![0.0; 3];
            row[k] = 1.0;
            row
        })
        .collect();
    let rows = row_predictions_for_tests(
        predicted,
        probabilities,
        truth,
        labels(),
        "artifact-sha",
        "test",
    );

    let block = assemble_quality_block(&rows, &toy_validation_rows(), &labels())
        .expect("an absent class is not a malformed split");

    assert_eq!(
        block.per_class_f1.len(),
        3,
        "the class must still be REPRESENTED. A metric vector shortened to the observed classes \
         would make `f1[2]` mean `favor` on one row and nothing on the next",
    );
    // The literal, recorded rather than described:
    assert_eq!(block.per_class_precision[2], 0.0);
    assert_eq!(block.per_class_recall[2], 0.0);
    assert_eq!(block.per_class_f1[2], 0.0);
    assert!(
        block.per_class_f1[2].is_finite() && block.f_avg.is_finite() && block.macro_f1.is_finite(),
        "no NaN may reach a published number; serde_json renders one as `null`, which is a \
         MISSING cell rather than a visible failure",
    );
    assert_eq!(
        block.confusion_matrix.len(),
        3,
        "and the confusion matrix keeps its third row and column",
    );
}

/// A split with zero rows is a TYPED refusal, from either position.
#[test]
fn bench_metrics_a_zero_row_split_is_a_typed_refusal() {
    let empty_test =
        row_predictions_for_tests(vec![], vec![], vec![], labels(), "artifact-sha", "test");
    let empty_validation =
        row_predictions_for_tests(vec![], vec![], vec![], labels(), "artifact-sha", "validation");

    let from_test = assemble_quality_block(&empty_test, &toy_validation_rows(), &labels())
        .expect_err("a zero-row test split has no metrics to report");
    assert!(
        matches!(from_test, BenchMetricsError::EmptySplit { parameter: "test_rows" }),
        "got: {from_test}",
    );

    let from_validation = assemble_quality_block(&toy_test_rows(), &empty_validation, &labels())
        .expect_err("a zero-row validation split has no calibration to report");
    assert!(
        matches!(from_validation, BenchMetricsError::EmptySplit { parameter: "validation_rows" }),
        "got: {from_validation}",
    );
}

/// A class index outside the label map is a typed error, never a silent 0.
#[test]
fn bench_metrics_a_class_index_outside_the_label_map_is_a_typed_error() {
    // A two-label map cannot contain index 2, so the official F_avg is not computable over it —
    // which `f1_average_for_classes` reports as `None` and this function must NOT read as zero.
    let two: Vec<String> = ["none", "against"].iter().map(|s| (*s).to_string()).collect();
    let truth = vec![0, 1, 0, 1];
    let predicted = vec![0, 1, 1, 1];
    let one_hot = |k: usize| {
        let mut row = vec![0.0; 2];
        row[k] = 1.0;
        row
    };
    let probabilities: Vec<Vec<f64>> = predicted.iter().map(|&k| one_hot(k)).collect();
    let test_rows = row_predictions_for_tests(
        predicted.clone(),
        probabilities.clone(),
        truth.clone(),
        two.clone(),
        "artifact-sha",
        "test",
    );
    let validation_rows = row_predictions_for_tests(
        predicted,
        probabilities,
        truth,
        two.clone(),
        "artifact-sha",
        "validation",
    );

    let error = assemble_quality_block(&test_rows, &validation_rows, &two)
        .expect_err("F_avg over classes [1, 2] is not computable on a two-label map");
    assert!(
        matches!(error, BenchMetricsError::ClassIndexOutsideLabelMap { n_classes: 2, .. }),
        "the refusal must name the map's size; got: {error}",
    );
}

/// The refusals are distinct renderings, so a reader can tell them apart.
#[test]
fn bench_metrics_refusals_are_distinct() {
    let rendered: Vec<String> = vec![
        BenchMetricsError::EmptySplit { parameter: "test_rows" }.to_string(),
        BenchMetricsError::WrongSplit {
            parameter: "validation_rows",
            expected: "validation",
            observed: "test".to_string(),
        }
        .to_string(),
        BenchMetricsError::LabelOrderMismatch {
            parameter: "test_rows",
            declared: vec!["a".to_string()],
            observed: vec!["b".to_string()],
        }
        .to_string(),
        BenchMetricsError::TooFewLabels { declared: 1 }.to_string(),
        BenchMetricsError::ClassIndexOutsideLabelMap { classes: vec![1, 2], n_classes: 2 }
            .to_string(),
    ];
    for (i, left) in rendered.iter().enumerate() {
        for (j, right) in rendered.iter().enumerate() {
            if i != j {
                assert_ne!(left, right, "two refusals must not read the same");
            }
        }
    }
}

// ===========================================================================================
// The import audit (D-06 / T-05-08-03): the RNG/bootstrap evaluator may not enter
// ===========================================================================================

/// `ClassifyEvalReport` — the unfixtured, resampling-capable report — is not reachable here.
///
/// Comment-filtered, so the prose above (and this plan's own wording) cannot satisfy or trip it.
#[test]
fn bench_metrics_does_not_import_the_resampling_evaluator() {
    let code: String = BENCH_METRICS_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        code.contains("fn assemble_quality_block"),
        "non-vacuity: the comment filter must not have eaten the module",
    );
    for banned in ["ClassifyEvalReport", "bootstrap", "resample", "rand::", "thread_rng"] {
        assert!(
            !code.contains(banned),
            "`{banned}` here would put an RNG — or an unfixtured metric — into the claims path; \
             the calibration numbers must come from 05-04's contract-bound functions alone",
        );
    }
    // And the five shipped surfaces ARE the ones used.
    for required in [
        "f1_average_for_classes",
        "MultiClassMetrics::from_predictions",
        "matthews_corrcoef",
        "expected_calibration_error_top_label",
        "brier_score_multiclass",
    ] {
        assert!(code.contains(required), "the assembly must call `{required}`");
    }
}

// ===========================================================================================
// The closed-form recomputation (05-17, verifier advisory 2)
// ===========================================================================================

/// The committed benchmark rows, at a path derived from this crate's own manifest directory.
///
/// A runtime read rather than `include_str!` because there are forty of them — but the test
/// below FAILS if the directory is absent or holds a different count, on `bench_row`'s own
/// precedent: an evidence test that silently skips when its evidence is missing proves nothing.
const COMMITTED_ROWS_DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../benchmarks/tweeteval-stance/rows");

/// How many cells the ACTIVE scope declares, and therefore how many rows must be measured.
const COMMITTED_ROW_COUNT: usize = 40;

/// Read and parse every committed row's PAYLOAD, in a deterministic (sorted) order.
///
/// # Why this does NOT go through `BenchRow::from_bytes`, stated rather than left as a gap
///
/// It cannot, in this build. `BenchRow::from_bytes` recomputes the envelope digest through
/// `to_canonical_bytes`, which serializes via `serde_json::Value` — whose `Map` is an
/// `IndexMap` when the `serde_json/preserve_order` feature is on anywhere in the binary's
/// dependency graph, and a `BTreeMap` when it is not. **MEASURED, on the committed row
/// `setfit-s16-seed13.json`:** its envelope claims
/// `1c54f3b4e38540040a2a2224a424969bae473e7b28da6b33ee2fb4be9f1eae69`, which is the
/// DECLARATION-ORDER digest; under `cargo test -p aprender-train --lib` the same payload
/// hashes to `fafda6485f47531aaf277a095ea782cd86ba148e694956bb902e3e0d9c5ad04c`, which is the
/// KEY-SORTED digest. `pmcp v2.19.3` turns the feature on in `apr-cli`'s graph and is absent
/// from `aprender-train`'s, so the shipped `apr` binary accepts these rows and this test
/// binary would refuse all forty as `row_digest_mismatch`.
///
/// That is a FINDING about the row seal, reported in 05-17's SUMMARY and in
/// `deferred-items.md`; it is NOT this plan's to fix (`bench_row.rs` is outside its
/// `files_modified`, and the fix re-seals forty rows and their manifests). It is deliberately
/// routed around here rather than worked around silently: the subject of THIS measurement is
/// whether a published metric follows from the confusion matrix beside it, which the envelope
/// digest has nothing to do with. Deserializing the payload still exercises the schema,
/// including `deny_unknown_fields`.
fn committed_rows() -> Vec<(String, crate::train::setfit::bench_row::BenchRowPayload)> {
    use crate::train::setfit::bench_row::BenchRowPayload;
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(COMMITTED_ROWS_DIR)
        .unwrap_or_else(|error| {
            panic!("the committed rows must be readable at {COMMITTED_ROWS_DIR}: {error}")
        })
        .map(|entry| entry.expect("a readable directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let name = path.file_name().expect("a file name").to_string_lossy().to_string();
            let bytes = std::fs::read(&path).expect("the committed row reads");
            let envelope: serde_json::Value = serde_json::from_slice(&bytes)
                .unwrap_or_else(|error| panic!("committed row {name} is not JSON: {error}"));
            let payload = envelope
                .get("payload")
                .unwrap_or_else(|| panic!("committed row {name} carries no payload"))
                .clone();
            let payload: BenchRowPayload =
                serde_json::from_value(payload).unwrap_or_else(|error| {
                    panic!("committed row {name}'s payload does not parse: {error}")
                });
            (name, payload)
        })
        .collect()
}

/// One measured field: how many cells agreed BIT-IDENTICALLY, and the largest deviation seen.
struct FieldAgreement {
    field: &'static str,
    bit_identical: usize,
    max_abs_deviation: f64,
}

#[test]
fn bench_metrics_the_forty_committed_rows_agree_with_their_own_confusion_matrices() {
    // THE MEASUREMENT THE ACCEPTANCE BAND WAS CHOSEN FROM, and it was taken BEFORE the band
    // was set (05-17's own prohibition: "a tolerance must not be chosen because it makes the
    // committed evidence pass"). It reports, per field, the number of cells agreeing bit for
    // bit and the maximum absolute deviation — so the band is derived from evidence rather than
    // fitted to it.
    //
    // If any cell disagreed, the answer is NOT a widened band. A metric that disagrees with its
    // own confusion matrix on committed evidence is a finding about the evidence, to be
    // reported; the row is not edited to make the check pass.
    let rows = committed_rows();
    assert_eq!(
        rows.len(),
        COMMITTED_ROW_COUNT,
        "NON-VACUITY: the measurement must cover the whole ACTIVE expectation set; a glob that \
         matched nothing would report 0/0 agreement and pass"
    );

    let mut scalar = vec![
        FieldAgreement { field: "f_avg", bit_identical: 0, max_abs_deviation: 0.0 },
        FieldAgreement { field: "macro_f1", bit_identical: 0, max_abs_deviation: 0.0 },
        FieldAgreement { field: "mcc", bit_identical: 0, max_abs_deviation: 0.0 },
        FieldAgreement { field: "per_class_precision", bit_identical: 0, max_abs_deviation: 0.0 },
        FieldAgreement { field: "per_class_recall", bit_identical: 0, max_abs_deviation: 0.0 },
        FieldAgreement { field: "per_class_f1", bit_identical: 0, max_abs_deviation: 0.0 },
    ];
    let mut n_rows_agreed = 0_usize;
    let mut bits_agreed = 0_usize;

    for (name, payload) in &rows {
        let quality = &payload.quality;
        let recomputed =
            quality_from_confusion_matrix(&quality.confusion_matrix, &quality.ordered_labels)
                .unwrap_or_else(|error| panic!("committed row {name}: {error}"));

        if quality.n_test_rows == recomputed.n_rows {
            n_rows_agreed += 1;
        }
        let pairs: [(usize, f64, f64); 3] = [
            (0, quality.f_avg, recomputed.f_avg),
            (1, quality.macro_f1, recomputed.macro_f1),
            (2, quality.mcc, recomputed.mcc),
        ];
        for (index, claimed, expected) in pairs {
            if claimed.to_bits() == expected.to_bits() {
                scalar[index].bit_identical += 1;
            }
            scalar[index].max_abs_deviation =
                scalar[index].max_abs_deviation.max((claimed - expected).abs());
        }
        let vectors: [(usize, &Vec<f64>, &Vec<f64>); 3] = [
            (3, &quality.per_class_precision, &recomputed.per_class_precision),
            (4, &quality.per_class_recall, &recomputed.per_class_recall),
            (5, &quality.per_class_f1, &recomputed.per_class_f1),
        ];
        for (index, claimed, expected) in vectors {
            if claimed.len() == expected.len()
                && claimed.iter().zip(expected.iter()).all(|(l, r)| l.to_bits() == r.to_bits())
            {
                scalar[index].bit_identical += 1;
            }
            for (l, r) in claimed.iter().zip(expected.iter()) {
                scalar[index].max_abs_deviation =
                    scalar[index].max_abs_deviation.max((l - r).abs());
            }
        }
        // The five bits siblings, counted per ROW: all five must hold for the row to count.
        let bits_ok = quality.f_avg_bits == quality.f_avg.to_bits()
            && quality.macro_f1_bits == quality.macro_f1.to_bits()
            && quality.mcc_bits == quality.mcc.to_bits()
            && quality.ece_top_label_validation_bits == quality.ece_top_label_validation.to_bits()
            && quality.brier_multiclass_validation_bits
                == quality.brier_multiclass_validation.to_bits();
        if bits_ok {
            bits_agreed += 1;
        }
    }

    // THE MEASUREMENT, printed so it can be read off a run rather than taken on trust.
    println!(
        "[bench_metrics] COMMITTED_ROW_AGREEMENT rows={} field=n_test_rows bit_identical={}/{} \
         max_abs_dev=0 (integer)",
        rows.len(),
        n_rows_agreed,
        rows.len()
    );
    for measured in &scalar {
        println!(
            "[bench_metrics] COMMITTED_ROW_AGREEMENT rows={} field={} bit_identical={}/{} \
             max_abs_dev={:e}",
            rows.len(),
            measured.field,
            measured.bit_identical,
            rows.len(),
            measured.max_abs_deviation
        );
    }
    println!(
        "[bench_metrics] COMMITTED_ROW_AGREEMENT rows={} field=all_five_bits_siblings \
         bit_identical={}/{} max_abs_dev=0 (integer)",
        rows.len(),
        bits_agreed,
        rows.len()
    );

    assert_eq!(n_rows_agreed, rows.len(), "`n_test_rows` must equal its matrix's own total");
    assert_eq!(
        bits_agreed,
        rows.len(),
        "every `_bits` field must equal `to_bits()` of the f64 beside it"
    );
    for measured in &scalar {
        assert_eq!(
            measured.bit_identical,
            rows.len(),
            "`{}` agreed on {}/{} committed rows with a maximum deviation of {:e}. THE BAND IS \
             EXACT IEEE-754 BIT EQUALITY and it is not to be widened: a published metric that \
             disagrees with its own confusion matrix is a finding about the evidence, not a \
             tolerance to fit",
            measured.field,
            measured.bit_identical,
            rows.len(),
            measured.max_abs_deviation
        );
    }
}

#[test]
fn bench_metrics_the_recomputation_is_order_independent() {
    // THE ONE PROPERTY THE COUNT EXPANSION RESTS ON, asserted rather than argued. Every metric
    // involved accumulates from integers, so the order the reconstructed pairs are emitted in
    // cannot change any result — but "cannot" is cheaper to test than to reason about.
    let labels = labels();
    let matrix = vec![vec![31, 9, 5], vec![74, 66, 49], vec![6, 12, 28]];
    let forward = quality_from_confusion_matrix(&matrix, &labels).expect("the matrix is valid");

    // The SAME multiset of pairs, reached by reversing both the row and the column walk. A
    // transposed matrix would be a different multiset and is deliberately not what this does.
    let mut reversed: Vec<Vec<u64>> = matrix.clone();
    reversed.reverse();
    for row in &mut reversed {
        row.reverse();
    }
    let mut relabelled = vec![vec![0_u64; 3]; 3];
    for (true_index, row) in reversed.iter().enumerate() {
        for (pred_index, &count) in row.iter().enumerate() {
            relabelled[2 - true_index][2 - pred_index] = count;
        }
    }
    assert_eq!(relabelled, matrix, "the walk-reversal must reconstruct the same tally");

    let backward =
        quality_from_confusion_matrix(&relabelled, &labels).expect("the matrix is valid");
    assert_eq!(forward.f_avg.to_bits(), backward.f_avg.to_bits());
    assert_eq!(forward.macro_f1.to_bits(), backward.macro_f1.to_bits());
    assert_eq!(forward.mcc.to_bits(), backward.mcc.to_bits());
    assert_eq!(forward.n_rows, backward.n_rows);
    assert_eq!(forward, backward, "every field, not only the three headlines");
}

#[test]
fn bench_metrics_a_degenerate_confusion_matrix_is_refused_rather_than_producing_nans() {
    // serde_json renders a NaN as `null`, which a reader takes for a MISSING cell rather than a
    // visible failure. Each shape below is a state in which SOME number could still be
    // produced, and every one of those numbers would be a NaN or a wrong attribution.
    let labels = labels();
    // THE VARIANT IS ASSERTED, not a substring of the prose. A message scan for "NaN"/"null"
    // would be the wrong instrument twice over: `EmptySplit`'s own explanation deliberately
    // contains the word `null` (it is explaining why a NaN is worse than a refusal), so the
    // scan collides with the very sentence that documents the property — the same collision
    // 05-16 recorded when it stopped grepping serialized output for "NaN".
    let cases: [(&str, Vec<Vec<u64>>, &str); 4] = [
        ("ragged", vec![vec![1, 2, 3], vec![4, 5, 6]], "ConfusionMatrixShape"),
        (
            "four_by_four_against_three_labels",
            vec![vec![1; 4], vec![1; 4], vec![1; 4], vec![1; 4]],
            "ConfusionMatrixShape",
        ),
        ("all_zero", vec![vec![0; 3]; 3], "EmptySplit"),
        (
            "above_the_expansion_cap",
            vec![vec![u64::MAX, 0, 0], vec![0, 0, 0], vec![0, 0, 0]],
            "ConfusionMatrixTooLarge",
        ),
    ];
    for (label, matrix, expected_variant) in cases {
        let error = quality_from_confusion_matrix(&matrix, &labels)
            .expect_err("a degenerate matrix must be refused");
        let observed = match &error {
            BenchMetricsError::ConfusionMatrixShape { .. } => "ConfusionMatrixShape",
            BenchMetricsError::ConfusionMatrixTooLarge { .. } => "ConfusionMatrixTooLarge",
            BenchMetricsError::EmptySplit { .. } => "EmptySplit",
            other => panic!("[{label}] unexpected variant: {other:?}"),
        };
        assert_eq!(observed, expected_variant, "[{label}] wrong refusal: {error}");
        assert!(
            error.to_string().contains("confusion_matrix"),
            "[{label}] the refusal must name the field an operator has to look at: {error}"
        );
    }

    // AND FEWER THAN TWO LABELS, which is a statement about the map rather than the matrix.
    let one_label = vec!["none".to_string()];
    assert!(quality_from_confusion_matrix(&[vec![1]], &one_label).is_err());

    // NON-VACUITY: a well-formed matrix over the same label map is ACCEPTED, so the assertions
    // above are not holding for a function that refuses everything.
    quality_from_confusion_matrix(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]], &labels)
        .expect("a well-formed matrix is accepted");
}

#[test]
fn bench_metrics_a_zero_support_class_still_occupies_its_index() {
    // EVAL-01's empty edge. A class with zero support AND zero predictions must still be
    // present in every per-class vector at its own index and score the shipped zero-division
    // value — otherwise `per_class_f1[2]` would mean `favor` on one row of a results table and
    // be out of range on the next, and the official F_avg would report "not computable" for a
    // class that merely did not occur.
    let labels = labels();
    let matrix = vec![vec![4, 1, 0], vec![1, 5, 0], vec![0, 0, 0]];
    let recomputed = quality_from_confusion_matrix(&matrix, &labels).expect("valid");
    assert_eq!(recomputed.per_class_f1.len(), 3);
    assert_eq!(recomputed.per_class_precision.len(), 3);
    assert_eq!(recomputed.per_class_recall.len(), 3);
    assert_eq!(recomputed.per_class_f1[2].to_bits(), 0.0_f64.to_bits());
    assert!(recomputed.f_avg.is_finite(), "the headline must be a number, never a NaN");
    assert_eq!(recomputed.n_rows, 11);
}

#[test]
fn bench_metrics_the_recomputation_authors_no_metric_arithmetic() {
    // OPS-03, held structurally. The recomputation must ROUTE to the same surfaces
    // `assemble_quality_block` routes to; a second definition of `F_avg` living here would
    // disagree with the first eventually and invisibly. The whole function is scanned rather
    // than only its signature, because a local re-derivation is exactly what would not show up
    // in a signature.
    let source = BENCH_METRICS_SOURCE;
    let start = source
        .find("pub fn quality_from_confusion_matrix")
        .expect("the recomputation is in this file");
    let end = source[start..]
        .find("\n/// Expand `[true][predicted]` counts")
        .map(|offset| start + offset)
        .expect("the function is followed by the expansion helper");
    let body = &source[start..end];

    for required in [
        "MultiClassMetrics::from_predictions_with_min_classes",
        "f1_average_for_classes",
        "matthews_corrcoef",
        "f1_avg(Average::Macro)",
    ] {
        assert!(body.contains(required), "the recomputation must route to `{required}`");
    }
    for banned in ["2.0 *", "/ 2.0", ".sqrt()", "as f64 /", "powi"] {
        assert!(
            !body.contains(banned),
            "`{banned}` in the recomputation would be a SECOND definition of a metric this \
             crate already ships; route to the surface instead"
        );
    }
}
