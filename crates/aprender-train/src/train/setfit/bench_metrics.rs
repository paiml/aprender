//! `QualityBlock` assembly: per-row predictions -> the EVAL-01 metric set (05-08).
//!
//! Contract: `setfit-benchmark-claims-v1`, equation `bench_row_schema`; and
//! `tweet-eval-stance-benchmark-v1`, equation `official_f_avg`.
//! Requirement: EVAL-01.
//!
//! # This module computes NOTHING. It routes.
//!
//! Every number a benchmark row publishes comes from a surface that already exists and is
//! already falsified against a reference:
//!
//! | number | surface | its evidence |
//! |---|---|---|
//! | per-class P / R / F1, confusion matrix | [`MultiClassMetrics`] | this crate's own metric suite |
//! | `F_avg` | [`f1_average_for_classes`] | contract-bound `official_f_avg` |
//! | MCC | [`matthews_corrcoef`] | `aprender-core`'s agreement metrics |
//! | top-label ECE | [`expected_calibration_error_top_label`] | 05-04's pinned-env fixtures |
//! | multiclass Brier | [`brier_score_multiclass`] | 05-04's pinned-env fixtures |
//! | THE CLOSED-FORM RECOMPUTATION of every row above | [`quality_from_confusion_matrix`] | the SAME three surfaces, over the counts the row itself recorded (05-17) |
//!
//! The one thing this module DOES decide is which vectors reach which surface, and that is the
//! decision the whole plan is about — so it is made structurally rather than by convention.
//!
//! # The recomputation is HERE, and not in the gate (05-17)
//!
//! [`quality_from_confusion_matrix`] reads a published row's own `confusion_matrix` and
//! `ordered_labels` and returns what the accuracy family must be. It lives beside
//! [`assemble_quality_block`] because it is the same routing decision read backwards, and
//! putting a second metric computation in `bench_gate` would give the phase two definitions of
//! `F_avg` — which is exactly the failure mode that module's own header names. The gate calls
//! this function and compares; it computes nothing.
//!
//! It is NOT a reimplementation and the reason is structural rather than a tolerance: every
//! metric involved is a function of the counts alone, all of them accumulate from integers, and
//! the expansion feeds the SAME entry points [`assemble_quality_block`] fed. Agreement is
//! therefore BIT-IDENTICAL by construction — measured 40/40 exact on every field over the
//! committed `benchmarks/tweeteval-stance/rows/` before any acceptance band was chosen, which
//! is why the band is exact IEEE-754 bit equality and there is no epsilon to justify.
//!
//! # Calibration is validation-only, and the signature is what enforces it (D-07)
//!
//! The two splits are SEPARATE PARAMETERS, and each is checked against the split tag its
//! [`RowPredictions`] carries. There is no argument order that feeds test probabilities to the
//! calibration functions, and no boolean a caller can get wrong. `calibration_split` is then a
//! recorded fact rather than a promise.
//!
//! # The resampling evaluator is not reachable from here
//!
//! `ClassifyEvalReport` carries its own, unfixtured `ece`/`brier` and a bootstrap path. 05-04
//! built the contract-bound replacements precisely so those cannot drift into a published claim,
//! and `bench_metrics_does_not_import_the_resampling_evaluator` scans this file's non-comment
//! source for the whole vocabulary.
//!
//! # Why `F_avg` selects indices `[1, 2]`, stated once
//!
//! TweetEval stance is `ClassLabel(names=['none', 'against', 'favor'])`, so class 1 is `against`
//! and class 2 is `favor` and `(F1[1] + F1[2]) / 2` IS the official
//! `F_avg = (F1_against + F1_favor) / 2`. That is not an assumption here: it is asserted against
//! `tweet-eval-stance-benchmark-v1.yaml`'s own class-label map at its pinned
//! `canonical_revision` by `bench_metrics_label_order_is_evidence_from_the_pinned_dataset_revision`,
//! which also pins the revision — so a revision bump goes red before any headline number is
//! computed. **The indices are correct and must not be changed.**

use core::fmt;

use aprender::calibration::{brier_score_multiclass, expected_calibration_error_top_label};
use aprender::metrics::matthews_corrcoef;

use crate::eval::classification::{
    f1_average_for_classes, Average, ConfusionMatrix, MultiClassMetrics,
};

use super::apr_evaluate::RowPredictions;
use super::bench_row::{QualityBlock, CALIBRATION_SPLIT};

/// The class indices the official TweetEval `F_avg` averages: `against` and `favor`.
///
/// See the module header. Changing these changes which metric the phase publishes.
pub const OFFICIAL_F_AVG_CLASSES: [usize; 2] = [1, 2];

/// Equal-width bins for the top-label ECE, matching `calibration-v1`'s declared default.
pub const ECE_N_BINS: usize = 10;

/// The split tag a `test_rows` argument must carry.
pub const TEST_SPLIT: &str = "test";

/// The pinned upstream TweetEval revision the label order is evidence FROM.
///
/// Byte-equal to `dataset.canonical_revision` in `tweet-eval-stance-benchmark-v1.yaml`.
pub const PINNED_TWEET_EVAL_REVISION: &str = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66";

/// The dataset contract, embedded at compile time.
///
/// `include_str!` rather than a runtime read, on `bench_row`'s precedent: an evidence test that
/// silently skips when the contract is absent proves nothing.
#[cfg(test)]
pub(crate) const TWEET_EVAL_CONTRACT_YAML: &str =
    include_str!("../../../../../contracts/tweet-eval-stance-benchmark-v1.yaml");

/// This module's own source, for the import audit.
#[cfg(test)]
const BENCH_METRICS_SOURCE: &str = include_str!("bench_metrics.rs");

// ===========================================================================================
// The refusals
// ===========================================================================================

/// An assembly that cannot produce an honest `QualityBlock`.
///
/// Every variant is a state in which SOME number could still be computed — which is exactly why
/// each is an error. A published row with a plausible number derived from the wrong split, the
/// wrong label attribution or an empty denominator is worse than a missing row, because nothing
/// downstream can tell the two apart.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BenchMetricsError {
    /// One of the two splits has no rows.
    EmptySplit {
        /// Which parameter: `test_rows` or `validation_rows`.
        parameter: &'static str,
    },
    /// A split was supplied in the position of the other one.
    WrongSplit {
        /// Which parameter.
        parameter: &'static str,
        /// The tag that parameter requires.
        expected: &'static str,
        /// The tag the evidence carries.
        observed: String,
    },
    /// The declared label vector is not the one the evidence was produced under.
    LabelOrderMismatch {
        /// Which parameter disagreed.
        parameter: &'static str,
        /// The vector the caller declared.
        declared: Vec<String>,
        /// The vector the evidence carries.
        observed: Vec<String>,
    },
    /// Fewer than two labels: neither calibration metric is defined.
    TooFewLabels {
        /// How many were declared.
        declared: usize,
    },
    /// The official `F_avg` selects a class the label map does not contain.
    ClassIndexOutsideLabelMap {
        /// The selected indices.
        classes: Vec<usize>,
        /// The label map's size.
        n_classes: usize,
    },
    /// A recorded `confusion_matrix` that is not a square tally over the declared label map.
    ///
    /// `Vec<Vec<u64>>` is the wire type, so serde accepts a ragged or mis-dimensioned matrix
    /// and the shape is not a property the parse can hold.
    ConfusionMatrixShape {
        /// How many rows the matrix has.
        rows: usize,
        /// Each row's width, in row order.
        row_widths: Vec<usize>,
        /// How many labels were declared beside it.
        declared_labels: usize,
    },
    /// A recorded `confusion_matrix` whose counts total more than [`MAX_CROSS_CHECK_ROWS`].
    ///
    /// The recomputation expands the counts back into index vectors, so an unbounded total is a
    /// denial of service through a shipped door rather than a wrong number: a few hundred bytes
    /// of JSON can declare `u64::MAX` in one cell.
    ConfusionMatrixTooLarge {
        /// The running total when the cap was crossed — a lower bound, since the sum stops
        /// there rather than continuing (and rather than overflowing).
        at_least: u64,
        /// The cap.
        cap: u64,
    },
}

impl fmt::Display for BenchMetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySplit { parameter } => write!(
                f,
                "`{parameter}` measured zero rows. Every metric below divides by that count, so \
                 the alternative to this refusal is a row of NaNs — and serde_json renders a NaN \
                 as null, which reads as a MISSING cell rather than a visible failure",
            ),
            Self::WrongSplit { parameter, expected, observed } => write!(
                f,
                "`{parameter}` requires evidence from the `{expected}` split and was handed \
                 `{observed}`. Calibration measured on the test split is a diagnostic about data \
                 the model was never allowed to be tuned against (D-07), and no field in the row \
                 would show that it had happened",
            ),
            Self::LabelOrderMismatch { parameter, declared, observed } => write!(
                f,
                "the declared label order {declared:?} is not `{parameter}`'s own {observed:?}. \
                 Index i of one is not index i of the other, so every per-class number would be \
                 published under another class's name — comparison is EXACT byte equality, so a \
                 difference in case, whitespace or Unicode form is a different label",
            ),
            Self::TooFewLabels { declared } => write!(
                f,
                "{declared} label(s) declared; the multiclass calibration metrics are defined \
                 for two or more classes",
            ),
            Self::ClassIndexOutsideLabelMap { classes, n_classes } => write!(
                f,
                "the official F_avg averages classes {classes:?}, which is not inside a \
                 {n_classes}-label map. Reporting 0 here would publish a headline score for a \
                 class that does not exist",
            ),
            Self::ConfusionMatrixShape { rows, row_widths, declared_labels } => write!(
                f,
                "the recorded `confusion_matrix` has {rows} row(s) of widths {row_widths:?} \
                 against {declared_labels} `ordered_labels`. Every per-class number indexes \
                 this matrix by a DECLARED class, so a ragged or mis-dimensioned tally would \
                 publish class i's counts under class j's name — or index out of the matrix \
                 entirely. `Vec<Vec<u64>>` is the wire type, so serde cannot hold this and the \
                 recomputation has to",
            ),
            Self::ConfusionMatrixTooLarge { at_least, cap } => write!(
                f,
                "the recorded `confusion_matrix` totals at least {at_least} observations, above \
                 the cap of {cap}. The recomputation expands the counts back into index \
                 vectors, so an unbounded total is a denial of service through a shipped door \
                 rather than a wrong number: a few hundred bytes of JSON can declare u64::MAX \
                 in one cell",
            ),
        }
    }
}

impl std::error::Error for BenchMetricsError {}

// ===========================================================================================
// The assembly
// ===========================================================================================

/// Assemble one benchmark row's quality half from per-row predictions.
///
/// `test_rows` supplies every accuracy-family number; `validation_rows` supplies the calibration
/// diagnostics and nothing else. `ordered_labels` is the caller's DECLARATION of the label map,
/// checked against both evidence vectors by exact byte equality — it is not how the function
/// finds the labels, it is how a disagreement becomes visible.
///
/// # Errors
///
/// [`BenchMetricsError`]: an empty split, a split in the wrong position, a declared label order
/// that is not the evidence's own, fewer than two labels, or an `F_avg` class index outside the
/// map.
pub fn assemble_quality_block(
    test_rows: &RowPredictions,
    validation_rows: &RowPredictions,
    ordered_labels: &[String],
) -> Result<QualityBlock, BenchMetricsError> {
    let n_classes = ordered_labels.len();
    if n_classes < 2 {
        return Err(BenchMetricsError::TooFewLabels { declared: n_classes });
    }
    check_evidence(test_rows, "test_rows", TEST_SPLIT, ordered_labels)?;
    check_evidence(validation_rows, "validation_rows", CALIBRATION_SPLIT, ordered_labels)?;

    // ---- The accuracy family, over the TEST split ------------------------------------------
    //
    // `from_predictions_with_min_classes` rather than `from_predictions`: the latter infers the
    // class count from the observed indices, so a class with zero support AND zero predictions
    // would vanish from every per-class vector and `F_avg` would report "class 2 not present"
    // for a class that merely did not occur. The DECLARED map is the authority here.
    let metrics = MultiClassMetrics::from_predictions_with_min_classes(
        test_rows.predicted(),
        test_rows.truth(),
        n_classes,
    );
    let f_avg = f1_average_for_classes(&metrics.f1, &OFFICIAL_F_AVG_CLASSES).ok_or_else(|| {
        BenchMetricsError::ClassIndexOutsideLabelMap {
            classes: OFFICIAL_F_AVG_CLASSES.to_vec(),
            n_classes,
        }
    })?;
    let macro_f1 = metrics.f1_avg(Average::Macro);

    // MCC is `aprender-core`'s, over the same two index vectors. It accumulates in f32; the row
    // records the widened value and its bits, so what is hashed is what was computed.
    let mcc = f64::from(matthews_corrcoef(test_rows.predicted(), test_rows.truth()));

    let confusion_matrix = confusion_counts(test_rows.predicted(), test_rows.truth(), n_classes);

    // ---- The calibration diagnostics, over the VALIDATION split ONLY (D-07) -----------------
    //
    // The core functions take f32 row-major probabilities; the classify path reports f64. The
    // narrowing is recorded rather than hidden: these two numbers are diagnostics published to
    // ~4 significant figures, and 05-04's fixtures assert the f32 implementations at 1e-6, so
    // the conversion is inside the band the metric is trusted to.
    let flat: Vec<f32> = validation_rows
        .probabilities()
        .iter()
        .flat_map(|row| row.iter().map(|&p| p as f32))
        .collect();
    let truth = validation_rows.truth();
    let ece = f64::from(expected_calibration_error_top_label(&flat, n_classes, truth, ECE_N_BINS));
    let brier = f64::from(brier_score_multiclass(&flat, n_classes, truth));

    Ok(QualityBlock {
        f_avg,
        f_avg_bits: f_avg.to_bits(),
        macro_f1,
        macro_f1_bits: macro_f1.to_bits(),
        per_class_precision: metrics.precision.clone(),
        per_class_recall: metrics.recall.clone(),
        per_class_f1: metrics.f1.clone(),
        mcc,
        mcc_bits: mcc.to_bits(),
        confusion_matrix,
        n_test_rows: test_rows.n_rows() as u64,
        ordered_labels: ordered_labels.to_vec(),
        ece_top_label_validation: ece,
        ece_top_label_validation_bits: ece.to_bits(),
        brier_multiclass_validation: brier,
        brier_multiclass_validation_bits: brier.to_bits(),
        calibration_split: CALIBRATION_SPLIT.to_string(),
    })
}

/// One split's evidence must be non-empty, from the right split, and under the declared map.
fn check_evidence(
    rows: &RowPredictions,
    parameter: &'static str,
    expected_split: &'static str,
    ordered_labels: &[String],
) -> Result<(), BenchMetricsError> {
    if rows.split_tag() != expected_split {
        return Err(BenchMetricsError::WrongSplit {
            parameter,
            expected: expected_split,
            observed: rows.split_tag().to_string(),
        });
    }
    // EXACT byte equality of the UTF-8 strings — Rust's `==` on `str`. No case folding, no
    // Unicode normalization, no trimming: a head whose labels differ from the declared map in
    // any of those ways is a DIFFERENT head, and quietly accepting it would publish its numbers
    // under this map's names.
    if rows.ordered_labels() != ordered_labels {
        return Err(BenchMetricsError::LabelOrderMismatch {
            parameter,
            declared: ordered_labels.to_vec(),
            observed: rows.ordered_labels().to_vec(),
        });
    }
    // AFTER the two structural checks, so an empty split whose tag is also wrong reports the
    // mistake a caller can act on rather than the one that happens to be checked first.
    if rows.n_rows() == 0 {
        return Err(BenchMetricsError::EmptySplit { parameter });
    }
    Ok(())
}

/// Row-major `[K][K]` counts, `[true][predicted]`, widened for the row's wire form.
///
/// The counting is the SHIPPED [`ConfusionMatrix`] — the same constructor
/// [`MultiClassMetrics::from_predictions_with_min_classes`] reduces, over the same declared map
/// size — so the matrix a reader inspects and the per-class numbers beside it cannot come from
/// two different tallies. This function only widens `usize` to the `u64` the row schema
/// declares.
fn confusion_counts(predicted: &[usize], truth: &[usize], n_classes: usize) -> Vec<Vec<u64>> {
    ConfusionMatrix::from_predictions_with_min_classes(predicted, truth, n_classes)
        .matrix()
        .iter()
        .map(|row| row.iter().map(|&count| count as u64).collect())
        .collect()
}

// ===========================================================================================
// The closed-form recomputation (05-17, verifier advisory 2)
// ===========================================================================================

/// The most observations the recomputation will expand back into index vectors.
///
/// DERIVED FROM AN EXISTING DECLARED BOUND, not chosen: the expansion allocates two
/// `Vec<usize>` of `total` elements, so at 10^6 it allocates 16 MB — the same order as
/// `MAX_EVIDENCE_FILE_BYTES`, the cap the gate already applies to the bytes this matrix arrives
/// in. It is not a band fitted to the committed evidence: the forty committed rows total **280**
/// observations each, some three and a half thousand times below this cap.
pub const MAX_CROSS_CHECK_ROWS: u64 = 1_000_000;

/// The accuracy family a row's own `confusion_matrix` and `ordered_labels` determine.
///
/// Every field is what [`assemble_quality_block`] would have published for the same counts.
/// The two calibration diagnostics are deliberately ABSENT: they need per-row probability
/// vectors that no committed file carries, so they are not recomputable and this struct does
/// not pretend otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct RecomputedQuality {
    /// Official `F_avg = (F1_against + F1_favor) / 2`.
    pub f_avg: f64,
    /// Three-class macro F1.
    pub macro_f1: f64,
    /// Matthews correlation coefficient, widened from the `f32` the surface returns.
    pub mcc: f64,
    /// Per-class precision, in `ordered_labels` order.
    pub per_class_precision: Vec<f64>,
    /// Per-class recall, in `ordered_labels` order.
    pub per_class_recall: Vec<f64>,
    /// Per-class F1, in `ordered_labels` order.
    pub per_class_f1: Vec<f64>,
    /// The matrix's own total — what `n_test_rows` must equal.
    pub n_rows: u64,
}

/// Recompute a published row's accuracy family from the counts the row itself recorded.
///
/// # Why expanding the counts is sound, and is not a second implementation
///
/// The matrix is `[true][predicted]` counts. Pushing `count` copies of `(predicted, true)` for
/// each cell reconstructs the exact MULTISET of prediction pairs that produced it. Every metric
/// below is a function of the counts alone — [`MultiClassMetrics::from_confusion_matrix`]
/// reduces a tally, and [`matthews_corrcoef`] rebuilds its own tally from the pairs and
/// accumulates in `i64` — so the ORDER the pairs are emitted in cannot change any result, and
/// `bench_metrics_the_recomputation_is_order_independent` asserts that rather than arguing it.
///
/// The expansion then feeds the SAME entry points [`assemble_quality_block`] fed, with the same
/// declared class count and the same `f64::from` widening of the same `f32` MCC return. So the
/// agreement is bit-identical BY CONSTRUCTION rather than within a tolerance, which is why the
/// gate compares `to_bits()` and no epsilon appears anywhere in this file.
///
/// A class with zero support and zero predictions still occupies its index, because
/// `from_predictions_with_min_classes` is given `ordered_labels.len()` — so `per_class_f1[2]`
/// means `favor` on every row rather than being out of range on some.
///
/// # Errors
///
/// [`BenchMetricsError::TooFewLabels`] for fewer than two declared labels;
/// [`BenchMetricsError::ConfusionMatrixShape`] for a ragged matrix or one whose dimension
/// disagrees with the label map; [`BenchMetricsError::ConfusionMatrixTooLarge`] above
/// [`MAX_CROSS_CHECK_ROWS`]; [`BenchMetricsError::EmptySplit`] for counts totalling zero.
///
/// Each of those is a state in which SOME number could still be produced, and every one of
/// those numbers would be a NaN — which `serde_json` renders as `null`, and which a reader takes
/// for a MISSING cell rather than a visible failure.
pub fn quality_from_confusion_matrix(
    confusion_matrix: &[Vec<u64>],
    ordered_labels: &[String],
) -> Result<RecomputedQuality, BenchMetricsError> {
    let n_classes = ordered_labels.len();
    if n_classes < 2 {
        return Err(BenchMetricsError::TooFewLabels { declared: n_classes });
    }

    // ---- SHAPE FIRST, before any arithmetic and before any allocation ----------------------
    let row_widths: Vec<usize> = confusion_matrix.iter().map(Vec::len).collect();
    if confusion_matrix.len() != n_classes || row_widths.iter().any(|width| *width != n_classes) {
        return Err(BenchMetricsError::ConfusionMatrixShape {
            rows: confusion_matrix.len(),
            row_widths,
            declared_labels: n_classes,
        });
    }

    // ---- THEN THE TOTAL, saturating and short-circuited ------------------------------------
    // `saturating_add` cannot wrap and cannot panic, and the loop stops at the cap — so a
    // doctored `u64::MAX` is refused in a few comparisons rather than attempting an allocation
    // no machine can serve.
    let mut n_rows: u64 = 0;
    for &count in confusion_matrix.iter().flatten() {
        n_rows = n_rows.saturating_add(count);
        if n_rows > MAX_CROSS_CHECK_ROWS {
            return Err(BenchMetricsError::ConfusionMatrixTooLarge {
                at_least: n_rows,
                cap: MAX_CROSS_CHECK_ROWS,
            });
        }
    }
    if n_rows == 0 {
        return Err(BenchMetricsError::EmptySplit { parameter: "confusion_matrix" });
    }

    let (y_pred, y_true) = expand_confusion_counts(confusion_matrix, n_rows);

    // ---- ROUTE. No arithmetic is authored below this line ----------------------------------
    let metrics = MultiClassMetrics::from_predictions_with_min_classes(&y_pred, &y_true, n_classes);
    let f_avg = f1_average_for_classes(&metrics.f1, &OFFICIAL_F_AVG_CLASSES).ok_or_else(|| {
        BenchMetricsError::ClassIndexOutsideLabelMap {
            classes: OFFICIAL_F_AVG_CLASSES.to_vec(),
            n_classes,
        }
    })?;
    Ok(RecomputedQuality {
        f_avg,
        macro_f1: metrics.f1_avg(Average::Macro),
        mcc: f64::from(matthews_corrcoef(&y_pred, &y_true)),
        per_class_precision: metrics.precision,
        per_class_recall: metrics.recall,
        per_class_f1: metrics.f1,
        n_rows,
    })
}

/// Expand `[true][predicted]` counts back into the `(y_pred, y_true)` index vectors.
///
/// `n_rows` is the already-validated total, used only to size the allocation once.
fn expand_confusion_counts(confusion_matrix: &[Vec<u64>], n_rows: u64) -> (Vec<usize>, Vec<usize>) {
    let capacity = usize::try_from(n_rows).unwrap_or(usize::MAX);
    let mut y_pred = Vec::with_capacity(capacity);
    let mut y_true = Vec::with_capacity(capacity);
    for (true_index, row) in confusion_matrix.iter().enumerate() {
        for (pred_index, &count) in row.iter().enumerate() {
            for _ in 0..count {
                y_pred.push(pred_index);
                y_true.push(true_index);
            }
        }
    }
    (y_pred, y_true)
}

#[cfg(test)]
#[path = "bench_metrics_tests.rs"]
mod bench_metrics_tests;
