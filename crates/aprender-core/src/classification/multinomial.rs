//! Multinomial (softmax) logistic regression head for any ordered `K >= 2` label set.
//!
//! This is a **general, dataset-agnostic** classification capability. SetFit is its
//! first consumer, not its owner (phase-3 decision D-01). The binary
//! [`LogisticRegression`](super::LogisticRegression) that lives alongside it is
//! deliberately left untouched: it is binary-only, plain gradient descent, and reports
//! failure as an untyped string where this head is required to fail with a *typed*
//! error that a caller can branch on.
//!
//! # Objective
//!
//! The native objective is the **mean** negative log-likelihood plus an L2 penalty
//! on the weight matrix only:
//!
//! ```text
//! J(W, b) = (1/n) * sum_i [ -log p_{i, y_i} ]  +  lambda * ||W||_F^2
//! ```
//!
//! where `p_{i,k} = softmax(z_i)_k` and `z_{i,k} = <W_k, x_i> + b_k`. The intercept
//! vector `b` is **not** penalized.
//!
//! # Relation to scikit-learn's `C`
//!
//! scikit-learn's `LogisticRegression` minimizes
//!
//! ```text
//! (1/S) * sum_i NLL_i  +  (1/(S*C)) * (1/2) * ||W||_F^2      (S = n, unweighted)
//! ```
//!
//! The **half lives inside sklearn's `r(W) = (1/2)||W||_F^2`**, so matching this
//! head's `lambda * ||W||_F^2` form gives
//!
//! ```text
//! lambda = 1 / (2 * C * n)
//! ```
//!
//! `n` here is the number of **rows passed to `fit`** — never a pair count, never a
//! post-deduplication count, never a batch size. [`Regularization::SklearnEquivalentC`]
//! resolves the relation at fit time from exactly that row count.
//!
//! Contract: `contracts/multinomial-head-v1.yaml`.
//!
//! # Deliberate omissions
//!
//! * **No `class_weight` knob.** The phase-3 reference head is
//!   `sklearn.linear_model.LogisticRegression()` with defaults, i.e. uniform class
//!   weights (RESEARCH Open Question 8). Adding weighting here would be an undeclared
//!   deviation from the reference the falsification gate compares against.
//! * **No [`Estimator`](crate::traits::Estimator) impl.** The typed API is primary
//!   (RESEARCH Open Question 6): `Estimator` round-trips labels through `f32` and
//!   returns `crate::error::Result`, both of which erase the typed failure modes
//!   TRN-04 requires. An `Estimator` bridge can be added later without breaking this
//!   surface.
//!
//! # Precision
//!
//! The fit is carried out in `f64` on [`LbfgsF64`]; the fitted coefficients are stored
//! as `f32` (the APR artifact width). Prediction reads the `f32` store but accumulates
//! logits in `f64`, so a finite `f32` feature row cannot produce a spuriously infinite
//! logit.

use crate::optim::{ConvergenceStatus, LbfgsF64};
use crate::primitives::Vector;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt;

/// Default maximum L-BFGS iterations.
///
/// Provenance: `sklearn.linear_model.LogisticRegression().get_params()["max_iter"] == 100`.
pub const DEFAULT_MAX_ITER: usize = 100;

/// Default gradient-norm convergence tolerance.
///
/// Provenance: `sklearn.linear_model.LogisticRegression().get_params()["tol"] == 1e-4`.
pub const DEFAULT_TOL: f64 = 1e-4;

/// Default L-BFGS correction-pair history size.
pub const DEFAULT_HISTORY_SIZE: usize = 10;

// =========================================================================
// Regularization
// =========================================================================

/// How the L2 penalty is specified.
///
/// See the module documentation for the two fully expanded objective conventions
/// and the `lambda = 1/(2*C*n)` relation between them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Regularization {
    /// Native form: the penalty term is `lambda * ||W||_F^2` with an unpenalized
    /// intercept. `lambda` must be finite and non-negative.
    Lambda(f64),
    /// scikit-learn's inverse regularization strength `C`.
    ///
    /// Resolved at fit time to `lambda = 1 / (2 * c * n)` where `n` is the number of
    /// **rows passed to `fit`** (a row count, never a pair count). `c` must be finite
    /// and strictly positive.
    SklearnEquivalentC {
        /// scikit-learn's `C` (inverse regularization strength).
        c: f64,
    },
}

impl Regularization {
    /// Resolves this specification to the native `lambda` for a fit over `n_rows` rows.
    ///
    /// Callers must validate the specification first (see [`HeadInputError`]); this
    /// function performs the arithmetic only.
    #[must_use]
    pub fn resolve_lambda(self, n_rows: usize) -> f64 {
        match self {
            Self::Lambda(lambda) => lambda,
            // The half lives inside sklearn's r(W) = (1/2)||W||^2_F.
            Self::SklearnEquivalentC { c } => 1.0 / (2.0 * c * n_rows as f64),
        }
    }
}

// =========================================================================
// Errors
// =========================================================================

/// A rejected `fit` or `predict` input, naming the offending element.
///
/// Every variant is produced by exactly one enumerated failure mode, so a caller can
/// branch on the cause rather than parsing a message.
#[derive(Debug, Clone, PartialEq)]
pub enum HeadInputError {
    /// `n_classes < 2`. A multinomial head needs at least a binary label set.
    TooFewClasses {
        /// The rejected class count.
        k: usize,
    },
    /// `ordered_labels.len()` disagrees with `n_classes`.
    LabelCountMismatch {
        /// Number of labels supplied.
        labels: usize,
        /// Configured class count.
        k: usize,
    },
    /// An ordered label is the empty string.
    EmptyLabel {
        /// Index of the empty label.
        index: usize,
    },
    /// Two ordered labels are equal, so the index -> label map is not injective.
    DuplicateLabel {
        /// First index carrying the label.
        first: usize,
        /// Second index carrying the same label.
        second: usize,
        /// The duplicated label.
        label: String,
    },
    /// No rows were supplied.
    EmptyDataset,
    /// The feature row count and the class-index count disagree.
    RowCountMismatch {
        /// Number of feature rows.
        rows: usize,
        /// Number of class indices.
        class_indices: usize,
    },
    /// The feature dimension is zero.
    ZeroFeatureDimension,
    /// A feature row has a different length than the first row.
    RaggedRow {
        /// Index of the offending row.
        row: usize,
        /// Dimension established by row 0.
        expected: usize,
        /// Dimension found on this row.
        found: usize,
    },
    /// A feature value is NaN.
    NanFeature {
        /// Row index of the offending value.
        row: usize,
        /// Column index of the offending value.
        col: usize,
    },
    /// A feature value is an infinity.
    InfiniteFeature {
        /// Row index of the offending value.
        row: usize,
        /// Column index of the offending value.
        col: usize,
        /// The offending value.
        value: f32,
    },
    /// A class index is not in `0..n_classes`.
    LabelIndexOutOfRange {
        /// Row carrying the out-of-range index.
        row: usize,
        /// The offending index.
        index: usize,
        /// Configured class count.
        k: usize,
    },
    /// A class in `0..n_classes` has no rows assigned to it.
    UnrepresentedClass {
        /// The unrepresented class index.
        class: usize,
    },
    /// [`Regularization::Lambda`] was negative.
    NegativeLambda {
        /// The rejected value.
        lambda: f64,
    },
    /// [`Regularization::Lambda`] was NaN or an infinity.
    NonFiniteLambda {
        /// The rejected value.
        lambda: f64,
    },
    /// [`Regularization::SklearnEquivalentC`] was zero or negative.
    NonPositiveC {
        /// The rejected value.
        c: f64,
    },
    /// [`Regularization::SklearnEquivalentC`] was NaN or an infinity.
    NonFiniteC {
        /// The rejected value.
        c: f64,
    },
    /// A prediction-time row has a different dimension than the fitted one.
    FeatureDimMismatch {
        /// Row index of the offending row.
        row: usize,
        /// The fitted feature dimension.
        expected: usize,
        /// The dimension supplied.
        found: usize,
    },
    /// Stored coefficients do not describe a `K x d` weight matrix plus `K` intercepts.
    ///
    /// Reached only from [`MultinomialLogisticRegression::from_stored_coefficients`],
    /// the reload door: a fit produces its own coefficients and cannot get this wrong.
    CoefficientCountMismatch {
        /// Which array was the wrong size (`"weights"` or `"intercepts"`).
        array: &'static str,
        /// The count implied by the label map and the feature dimension.
        expected: usize,
        /// The count supplied.
        found: usize,
    },
    /// A stored coefficient is NaN or an infinity.
    ///
    /// Reached only from the reload door. A non-finite coefficient makes every
    /// logit non-finite, so it is refused at the boundary rather than surfacing
    /// later as a per-row prediction failure.
    NonFiniteCoefficient {
        /// Which array carried it (`"weights"` or `"intercepts"`).
        array: &'static str,
        /// Flat index of the offending value.
        index: usize,
        /// The offending value.
        value: f32,
    },
}

impl fmt::Display for HeadInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewClasses { k } => {
                write!(f, "n_classes = {k}, but a multinomial head requires K >= 2")
            }
            Self::LabelCountMismatch { labels, k } => {
                write!(f, "ordered_labels has {labels} entries but n_classes = {k}")
            }
            Self::EmptyLabel { index } => {
                write!(f, "ordered_labels[{index}] is the empty string")
            }
            Self::DuplicateLabel {
                first,
                second,
                label,
            } => write!(
                f,
                "ordered_labels[{first}] and ordered_labels[{second}] are both {label:?}"
            ),
            Self::EmptyDataset => write!(f, "no feature rows were supplied"),
            Self::RowCountMismatch {
                rows,
                class_indices,
            } => write!(f, "{rows} feature rows but {class_indices} class indices"),
            Self::ZeroFeatureDimension => write!(f, "the feature dimension is 0"),
            Self::RaggedRow {
                row,
                expected,
                found,
            } => write!(
                f,
                "features[{row}] has dimension {found}, expected {expected} (ragged input)"
            ),
            Self::NanFeature { row, col } => {
                write!(f, "features[{row}][{col}] is NaN")
            }
            Self::InfiniteFeature { row, col, value } => {
                write!(f, "features[{row}][{col}] is {value} (not finite)")
            }
            Self::LabelIndexOutOfRange { row, index, k } => {
                write!(f, "class_indices[{row}] = {index}, not in 0..{k}")
            }
            Self::UnrepresentedClass { class } => write!(
                f,
                "class {class} has no rows; every class in 0..K must be represented"
            ),
            Self::NegativeLambda { lambda } => {
                write!(f, "lambda = {lambda} is negative")
            }
            Self::NonFiniteLambda { lambda } => {
                write!(f, "lambda = {lambda} is not finite")
            }
            Self::NonPositiveC { c } => write!(f, "C = {c} must be strictly positive"),
            Self::NonFiniteC { c } => write!(f, "C = {c} is not finite"),
            Self::FeatureDimMismatch {
                row,
                expected,
                found,
            } => write!(
                f,
                "prediction row {row} has dimension {found}, but the head was fitted on {expected}"
            ),
            Self::CoefficientCountMismatch {
                array,
                expected,
                found,
            } => write!(
                f,
                "stored {array} has {found} values but the label map and feature dimension \
                 imply {expected}"
            ),
            Self::NonFiniteCoefficient {
                array,
                index,
                value,
            } => write!(f, "stored {array}[{index}] is {value} (not finite)"),
        }
    }
}

impl std::error::Error for HeadInputError {}

/// A typed failure from [`MultinomialLogisticRegression::fit`] or its predictors.
///
/// There is deliberately no `String` payload anywhere: TRN-04 requires that a caller
/// can distinguish "your data was bad" from "the optimizer ran out of budget" from
/// "the arithmetic went non-finite" without string matching.
#[derive(Debug, Clone, PartialEq)]
pub enum HeadFitError {
    /// Input validation rejected the call before any solve was attempted.
    InvalidInput(HeadInputError),
    /// The optimizer exhausted `max_iter` without reaching `tol`.
    ///
    /// **Declared deviation from scikit-learn:** sklearn emits a
    /// `ConvergenceWarning` and returns the non-converged coefficients anyway. This
    /// head fails instead, because TRN-04 requires explicit convergence *or* a typed
    /// failure and a warning is neither.
    NotConverged {
        /// Iterations performed.
        iterations: usize,
        /// Gradient norm reached.
        gradient_norm: f64,
        /// Tolerance that was not reached.
        tol: f64,
    },
    /// The line search could make no further progress.
    Stalled {
        /// Iterations performed.
        iterations: usize,
        /// Gradient norm reached.
        gradient_norm: f64,
    },
    /// A non-finite value reached the optimizer from some channel.
    NumericalError {
        /// Iterations performed.
        iterations: usize,
    },
    /// The optimizer returned a status unreachable from this call pattern.
    ///
    /// `Running` and `UserTerminated` cannot be produced by a completed
    /// [`LbfgsF64::minimize`]. If one ever arrives it is an internal invariant
    /// violation, and it is surfaced rather than silently treated as success.
    Internal {
        /// The unexpected status.
        status: ConvergenceStatus,
    },
    /// A logit was not finite. Logits accumulate in `f64`; this is returned instead of
    /// letting an infinity propagate into the softmax.
    NonFiniteLogit {
        /// Row index whose logit went non-finite.
        row: usize,
        /// Class index whose logit went non-finite.
        class: usize,
    },
    /// A predictor was called before `fit` succeeded.
    NotFitted,
}

impl fmt::Display for HeadFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(e) => write!(f, "invalid input: {e}"),
            Self::NotConverged {
                iterations,
                gradient_norm,
                tol,
            } => write!(
                f,
                "L-BFGS did not converge: {iterations} iterations, gradient norm {gradient_norm:e} > tol {tol:e}"
            ),
            Self::Stalled {
                iterations,
                gradient_norm,
            } => write!(
                f,
                "L-BFGS stalled after {iterations} iterations at gradient norm {gradient_norm:e}"
            ),
            Self::NumericalError { iterations } => write!(
                f,
                "L-BFGS hit a non-finite value after {iterations} iterations"
            ),
            Self::Internal { status } => write!(
                f,
                "L-BFGS returned {status:?}, which is unreachable from a completed minimize()"
            ),
            Self::NonFiniteLogit { row, class } => write!(
                f,
                "logit for row {row}, class {class} is not finite"
            ),
            Self::NotFitted => write!(f, "the head has not been fitted"),
        }
    }
}

impl std::error::Error for HeadFitError {}

impl From<HeadInputError> for HeadFitError {
    fn from(e: HeadInputError) -> Self {
        Self::InvalidInput(e)
    }
}

// =========================================================================
// Fit report
// =========================================================================

/// Deterministic record of one fit.
///
/// Every field is a function of the inputs alone. There is **no** wall-clock field of
/// any kind: this report is designed to be hashed into a reproducibility record, and a
/// timing field makes any such record irreproducible. Time a fit at the call site if
/// you need it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadFitReport {
    /// Terminal optimizer status (always `Converged` when `fit` returned `Ok`).
    pub status: ConvergenceStatus,
    /// Iterations performed.
    pub iterations: usize,
    /// Final gradient norm.
    pub final_grad_norm: f64,
    /// Final objective value (mean NLL + penalty).
    pub objective: f64,
}

// =========================================================================
// Objective and analytic gradient
// =========================================================================

/// Numerically safe `log(sum_k exp(z_k))` using the max-subtraction shift.
///
/// The shift is what makes this usable: `exp` overflows `f64` above an argument of
/// about 709, so a naive `sum exp(z)` is `inf` for a logit of 1000 while the shifted
/// form is exact to rounding.
pub(crate) fn log_sum_exp(logits: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &z in logits {
        if z > max {
            max = z;
        }
    }
    let mut sum = 0.0;
    for &z in logits {
        sum += (z - max).exp();
    }
    max + sum.ln()
}

/// Writes `softmax(logits)` into `out` using the same max-subtraction shift.
pub(crate) fn softmax_into(logits: &[f64], out: &mut [f64]) {
    let mut max = f64::NEG_INFINITY;
    for &z in logits {
        if z > max {
            max = z;
        }
    }
    let mut sum = 0.0;
    for (o, &z) in out.iter_mut().zip(logits.iter()) {
        let e = (z - max).exp();
        *o = e;
        sum += e;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// Index of the largest value, breaking exact ties to the **lowest** index.
///
/// The strict `>` comparison is what implements the tie-break: a later element that
/// merely equals the running maximum never displaces it.
pub(crate) fn argmax_lowest_index(values: &[f64]) -> usize {
    let mut best = 0;
    let mut best_value = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_value {
            best_value = v;
            best = i;
        }
    }
    best
}

/// The softmax-NLL + L2 problem handed to L-BFGS.
///
/// Parameter layout, flattened into a single `Vector<f64>` of length `K*d + K`:
///
/// ```text
/// x[k*d + j] = W[k][j]     for k in 0..K, j in 0..d   (row-major, K rows of d)
/// x[K*d + k] = b[k]        for k in 0..K              (the intercept block)
/// ```
pub(crate) struct SoftmaxNllProblem<'a> {
    pub(crate) features: &'a [Vec<f32>],
    pub(crate) class_indices: &'a [usize],
    pub(crate) n_classes: usize,
    pub(crate) n_features: usize,
    pub(crate) lambda: f64,
}

impl SoftmaxNllProblem<'_> {
    /// Offset of the intercept block within the flattened parameter vector.
    pub(crate) fn intercept_offset(&self) -> usize {
        self.n_classes * self.n_features
    }

    /// Length of the flattened parameter vector.
    pub(crate) fn n_params(&self) -> usize {
        self.n_classes * self.n_features + self.n_classes
    }

    /// Fills `logits` with `z_i = W x_i + b` for row `i`, accumulating in `f64`.
    /// `row_f64` is a caller-owned scratch of length `n_features`: widening the row once
    /// per row rather than once per class keeps the inner loop uniformly `f64`. The
    /// widening is exact, so the accumulated result is bit-identical either way.
    fn logits_for_row(&self, x: &Vector<f64>, row: usize, row_f64: &mut [f64], logits: &mut [f64]) {
        let d = self.n_features;
        let off = self.intercept_offset();
        // Indexed rather than zipped. A `zip` stops at the SHORTER side, so a feature row
        // narrower than `n_features` would silently leave the previous row's values in the
        // reused scratch and compute a logit from another example's features. Validation
        // rejects ragged input, but a scratch buffer that reads correct while carrying stale
        // data is the wrong shape for a "single gate" invariant to rest on: indexing panics
        // loudly instead.
        let features = &self.features[row];
        for j in 0..d {
            row_f64[j] = f64::from(features[j]);
        }
        for c in 0..self.n_classes {
            let mut z = x[off + c];
            for j in 0..d {
                z += x[c * d + j] * row_f64[j];
            }
            logits[c] = z;
        }
    }

    /// `J(W, b) = (1/n) * sum_i NLL_i + lambda * ||W||_F^2` (intercept unpenalized).
    ///
    /// Contract: `contracts/multinomial-head-v1.yaml`, equation `softmax_nll_objective`.
    #[provable_contracts_macros::contract(
        "multinomial-head-v1",
        equation = "softmax_nll_objective"
    )]
    pub(crate) fn objective(&self, x: &Vector<f64>) -> f64 {
        let k = self.n_classes;
        let n = self.features.len();
        let mut logits = vec![0.0_f64; k];
        let mut row_f64 = vec![0.0_f64; self.n_features];
        let mut nll = 0.0_f64;
        for i in 0..n {
            self.logits_for_row(x, i, &mut row_f64, &mut logits);
            // -log p_{i, y_i} = logsumexp(z_i) - z_{i, y_i}
            nll += log_sum_exp(&logits) - logits[self.class_indices[i]];
        }
        // The penalty covers the W block ONLY: the intercept block starts at
        // `intercept_offset()` and is deliberately excluded.
        let mut penalty = 0.0_f64;
        for p in 0..self.intercept_offset() {
            penalty += x[p] * x[p];
        }
        nll / n as f64 + self.lambda * penalty
    }

    /// Analytic gradient of [`Self::objective`].
    ///
    /// ```text
    /// dJ/dW[k][j] = (1/n) * sum_i (p_{i,k} - [k == y_i]) * x_i[j]  +  2*lambda*W[k][j]
    /// dJ/db[k]    = (1/n) * sum_i (p_{i,k} - [k == y_i])
    /// ```
    ///
    /// The penalty contributes `2*lambda*W` to the weight block and **exactly 0** to
    /// the intercept block. Validated against central differences in
    /// `tests_multinomial_contract.rs`, not merely against a converged reference
    /// optimum — a subtly wrong gradient can still reach the right optimum on some
    /// configurations.
    ///
    /// Contract: `contracts/multinomial-head-v1.yaml`, equation `analytic_gradient`.
    #[provable_contracts_macros::contract("multinomial-head-v1", equation = "analytic_gradient")]
    pub(crate) fn gradient(&self, x: &Vector<f64>) -> Vector<f64> {
        let k = self.n_classes;
        let d = self.n_features;
        let n = self.features.len();
        let off = self.intercept_offset();
        let inv_n = 1.0 / n as f64;

        let mut g = vec![0.0_f64; self.n_params()];
        let mut logits = vec![0.0_f64; k];
        let mut probs = vec![0.0_f64; k];
        let mut row_f64 = vec![0.0_f64; self.n_features];

        for i in 0..n {
            self.logits_for_row(x, i, &mut row_f64, &mut logits);
            softmax_into(&logits, &mut probs);
            let y_i = self.class_indices[i];
            let row = &self.features[i];
            for c in 0..k {
                let indicator = if c == y_i { 1.0 } else { 0.0 };
                let residual = (probs[c] - indicator) * inv_n;
                for j in 0..d {
                    g[c * d + j] += residual * f64::from(row[j]);
                }
                g[off + c] += residual;
            }
        }

        // Penalty: 2*lambda*W on the weight block, nothing on the intercept block.
        for p in 0..off {
            g[p] += 2.0 * self.lambda * x[p];
        }

        Vector::from_vec(g)
    }
}

// =========================================================================
// The head
// =========================================================================

/// Validated fit inputs: everything the solver needs that validation had to compute.
struct ValidatedFit {
    n_features: usize,
    lambda: f64,
}

/// THE label-set rule: at least two classes, arity agreement, no empty label, no
/// duplicate.
///
/// Extracted so the fit path and the reload path
/// ([`MultinomialLogisticRegression::from_stored_coefficients`]) apply the SAME
/// rule. A second copy is how a head reloaded from an artifact ends up accepting
/// a label map the fit would have refused, which would make the index -> label
/// map non-injective for exactly the models nobody re-fits.
fn validate_label_set(n_classes: usize, ordered_labels: &[String]) -> Result<(), HeadInputError> {
    if n_classes < 2 {
        return Err(HeadInputError::TooFewClasses { k: n_classes });
    }
    if ordered_labels.len() != n_classes {
        return Err(HeadInputError::LabelCountMismatch {
            labels: ordered_labels.len(),
            k: n_classes,
        });
    }
    for (index, label) in ordered_labels.iter().enumerate() {
        if label.is_empty() {
            return Err(HeadInputError::EmptyLabel { index });
        }
    }
    // ONE pass, not the `K^2` pairwise scan this used to be. The rule is applied to
    // `ordered_labels` taken straight off a parsed artifact
    // ([`MultinomialLogisticRegression::from_stored_coefficients`]), and it runs BEFORE
    // the coefficient-arity check that would otherwise bound `K` — so a payload
    // declaring millions of one-character labels turned the reload door into a hang
    // that the bundle's four size bounds cannot see, because none of them bounds the
    // label map. A `K`-time scan removes the amplification instead of bounding it.
    //
    // It reports the SAME pair the pairwise scan did. `first_seen` records each label's
    // first index; a repeat at `index` yields the candidate `(first_seen, index)`, and
    // the smallest `first` wins — which is the pair the outer-then-inner loop found,
    // because a strict `<` keeps the earliest repeat of the winning label.
    let mut first_seen: HashMap<&str, usize> = HashMap::with_capacity(ordered_labels.len());
    let mut duplicate: Option<(usize, usize)> = None;
    for (index, label) in ordered_labels.iter().enumerate() {
        match first_seen.entry(label.as_str()) {
            Entry::Vacant(slot) => {
                slot.insert(index);
            }
            Entry::Occupied(slot) => {
                let first = *slot.get();
                if duplicate.is_none_or(|(best, _)| first < best) {
                    duplicate = Some((first, index));
                }
            }
        }
    }
    if let Some((first, second)) = duplicate {
        return Err(HeadInputError::DuplicateLabel {
            first,
            second,
            label: ordered_labels[first].clone(),
        });
    }
    Ok(())
}

/// The SHAPE rung: non-empty, row counts agreeing, non-zero and uniform width.
///
/// Returns the feature dimension every row was required to have, so the caller cannot
/// re-derive it from `features[0]` and disagree with what was checked.
fn validate_fit_shape(
    features: &[Vec<f32>],
    class_indices: &[usize],
) -> Result<usize, HeadInputError> {
    if features.is_empty() {
        return Err(HeadInputError::EmptyDataset);
    }
    if features.len() != class_indices.len() {
        return Err(HeadInputError::RowCountMismatch {
            rows: features.len(),
            class_indices: class_indices.len(),
        });
    }
    let n_features = features[0].len();
    if n_features == 0 {
        return Err(HeadInputError::ZeroFeatureDimension);
    }
    for (row, values) in features.iter().enumerate() {
        if values.len() != n_features {
            return Err(HeadInputError::RaggedRow {
                row,
                expected: n_features,
                found: values.len(),
            });
        }
    }
    Ok(n_features)
}

/// The VALUE rung: every feature finite, NaN reported as NaN rather than as an infinity.
fn validate_fit_feature_values(features: &[Vec<f32>]) -> Result<(), HeadInputError> {
    for (row, values) in features.iter().enumerate() {
        for (col, &v) in values.iter().enumerate() {
            if v.is_nan() {
                return Err(HeadInputError::NanFeature { row, col });
            }
            if !v.is_finite() {
                return Err(HeadInputError::InfiniteFeature { row, col, value: v });
            }
        }
    }
    Ok(())
}

/// The LABEL-INDEX rung: in range, and every declared class actually represented.
fn validate_fit_class_indices(
    n_classes: usize,
    class_indices: &[usize],
) -> Result<(), HeadInputError> {
    let mut represented = vec![false; n_classes];
    for (row, &index) in class_indices.iter().enumerate() {
        if index >= n_classes {
            return Err(HeadInputError::LabelIndexOutOfRange {
                row,
                index,
                k: n_classes,
            });
        }
        represented[index] = true;
    }
    for (class, seen) in represented.iter().enumerate() {
        if !seen {
            return Err(HeadInputError::UnrepresentedClass { class });
        }
    }
    Ok(())
}

/// The REGULARIZATION rung.
///
/// Finiteness is checked FIRST in both arms: `NaN < 0.0` and `NaN <= 0.0` are both false,
/// so a NaN would otherwise slip past the sign checks and be reported as a valid penalty.
fn validate_fit_regularization(regularization: Regularization) -> Result<(), HeadInputError> {
    match regularization {
        Regularization::Lambda(lambda) => {
            if !lambda.is_finite() {
                return Err(HeadInputError::NonFiniteLambda { lambda });
            }
            if lambda < 0.0 {
                return Err(HeadInputError::NegativeLambda { lambda });
            }
        }
        Regularization::SklearnEquivalentC { c } => {
            if !c.is_finite() {
                return Err(HeadInputError::NonFiniteC { c });
            }
            if c <= 0.0 {
                return Err(HeadInputError::NonPositiveC { c });
            }
        }
    }
    Ok(())
}

/// Validates every `fit` input **before** any solve is attempted.
///
/// This is the single gate: no partially validated state can reach the optimizer,
/// because the optimizer is not called until this returns `Ok`.
///
/// # The rungs are separate functions, and the ORDER here is the contract
///
/// Split in plan 03-10 T3 to clear the project's cyclomatic ceiling of 10 (measured 20
/// before, by `pmat analyze complexity`). The extraction is deliberately order-preserving
/// and nothing else: the FALSIFY case table pins WHICH error each of the sixteen invalid
/// inputs produces, and several inputs are invalid on more than one rung — a ragged row of
/// NaNs is both `RaggedRow` and `NanFeature` — so reordering these calls would change the
/// reported error for an input that is still, correctly, rejected. That is the failure a
/// refactor of a validation ladder makes, so the sequence is stated once, here.
fn validate_fit_inputs(
    n_classes: usize,
    features: &[Vec<f32>],
    class_indices: &[usize],
    ordered_labels: &[String],
    regularization: Regularization,
) -> Result<ValidatedFit, HeadInputError> {
    validate_label_set(n_classes, ordered_labels)?;
    let n_features = validate_fit_shape(features, class_indices)?;
    validate_fit_feature_values(features)?;
    validate_fit_class_indices(n_classes, class_indices)?;
    validate_fit_regularization(regularization)?;

    Ok(ValidatedFit {
        n_features,
        lambda: regularization.resolve_lambda(features.len()),
    })
}

/// Multinomial (softmax) logistic regression over an ordered `K >= 2` label set.
///
/// See the module documentation for the objective, the sklearn `C` relation, and the
/// deliberate omissions.
#[derive(Debug, Clone)]
pub struct MultinomialLogisticRegression {
    n_classes: usize,
    max_iter: usize,
    tol: f64,
    history_size: usize,

    /// Fitted feature dimension (`None` until a successful fit).
    n_features: Option<usize>,
    /// `K * d` weights, row-major, stored at the APR artifact width.
    weights: Vec<f32>,
    /// `K` intercepts, stored at the APR artifact width.
    intercepts: Vec<f32>,
    /// The `f64` intercepts, kept out of the public surface: the gauge and gradient
    /// assertions must not be confounded by the `f32` downcast.
    intercepts_f64: Vec<f64>,
    /// Ordered labels; index == weight-matrix row.
    labels: Vec<String>,
    /// Report from the last successful fit.
    report: Option<HeadFitReport>,
}

impl MultinomialLogisticRegression {
    /// Creates an unfitted head for `n_classes` classes.
    ///
    /// `n_classes` is validated at fit time, not here, so that every rejection travels
    /// through the same typed channel.
    #[must_use]
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            max_iter: DEFAULT_MAX_ITER,
            tol: DEFAULT_TOL,
            history_size: DEFAULT_HISTORY_SIZE,
            n_features: None,
            weights: Vec::new(),
            intercepts: Vec::new(),
            intercepts_f64: Vec::new(),
            labels: Vec::new(),
            report: None,
        }
    }

    /// Rebuild a head from stored coefficients — the RELOAD door (plan 03-08).
    ///
    /// # This is not a fit, and it does not claim to be one
    ///
    /// [`Self::report`] stays `None`: no optimizer ran in this process, and a
    /// synthesized report would assert a convergence status nobody observed. The
    /// provenance of the coefficients belongs to whatever artifact carried them,
    /// and it is that artifact's job to record the fit that produced them.
    ///
    /// Every input is validated before the head exists, through the SAME
    /// label-set rule the fit path uses ([`validate_label_set`]) plus the
    /// coefficient-arity and finiteness checks a fit cannot get wrong. A head
    /// returned from here therefore satisfies `predict_proba`'s preconditions by
    /// construction, exactly as a fitted one does.
    ///
    /// # Errors
    ///
    /// [`HeadFitError::InvalidInput`] carrying the offending
    /// [`HeadInputError`]: too few classes, a mis-sized label map, an empty or
    /// duplicated label, a zero feature dimension, a weight or intercept array of
    /// the wrong length, or a non-finite coefficient.
    pub fn from_stored_coefficients(
        ordered_labels: Vec<String>,
        n_features: usize,
        weights: Vec<f32>,
        intercepts: Vec<f32>,
    ) -> Result<Self, HeadFitError> {
        let n_classes = ordered_labels.len();
        validate_label_set(n_classes, &ordered_labels)?;
        if n_features == 0 {
            return Err(HeadInputError::ZeroFeatureDimension.into());
        }
        // SATURATING, not checked-with-a-second-error-arm. The overflow arm had to invent
        // an `expected` it could not compute and reported `usize::MAX`, which rendered as a
        // claim that the label map implies 18446744073709551615 weights. Saturating makes
        // that value TRUE at the bound instead of a placeholder, and collapses two arms
        // constructing the same error into one: a product that saturates is necessarily
        // larger than any `Vec` length, so the mismatch below always fires.
        let expected_weights = n_classes.saturating_mul(n_features);
        if weights.len() != expected_weights {
            return Err(HeadInputError::CoefficientCountMismatch {
                array: "weights",
                expected: expected_weights,
                found: weights.len(),
            }
            .into());
        }
        if intercepts.len() != n_classes {
            return Err(HeadInputError::CoefficientCountMismatch {
                array: "intercepts",
                expected: n_classes,
                found: intercepts.len(),
            }
            .into());
        }
        for (array, values) in [("weights", &weights), ("intercepts", &intercepts)] {
            for (index, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(HeadInputError::NonFiniteCoefficient {
                        array,
                        index,
                        value: *value,
                    }
                    .into());
                }
            }
        }

        Ok(Self {
            n_classes,
            max_iter: DEFAULT_MAX_ITER,
            tol: DEFAULT_TOL,
            history_size: DEFAULT_HISTORY_SIZE,
            n_features: Some(n_features),
            // `intercepts_f64` is the gauge/gradient path's copy and exists to keep
            // the f32 downcast out of the optimizer's arithmetic. Widening the
            // stored f32 is exact, and it is the only value available here — the
            // f64 the fit held did not survive the artifact.
            intercepts_f64: intercepts.iter().map(|&b| f64::from(b)).collect(),
            weights,
            intercepts,
            labels: ordered_labels,
            report: None,
        })
    }

    /// Sets the maximum L-BFGS iteration budget (default [`DEFAULT_MAX_ITER`]).
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the gradient-norm convergence tolerance (default [`DEFAULT_TOL`]).
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the L-BFGS correction-pair history size (default [`DEFAULT_HISTORY_SIZE`]).
    #[must_use]
    pub fn with_history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Number of classes this head is configured for.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Fitted feature dimension, or `None` before a successful fit.
    #[must_use]
    pub fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Fitted weights as `K * d` values in row-major order, or an empty slice.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Fitted intercepts (`K` values), or an empty slice.
    #[must_use]
    pub fn intercepts(&self) -> &[f32] {
        &self.intercepts
    }

    /// Ordered labels; index == weight-matrix row.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Report from the last successful fit.
    #[must_use]
    pub fn report(&self) -> Option<&HeadFitReport> {
        self.report.as_ref()
    }

    /// The `f64` intercepts from the solve, deliberately kept off the public surface.
    ///
    /// `#[cfg(test)]` because the only reason to reach past the stored `f32` artifact
    /// is to assert a property the `f32` downcast would mask: the intercept gauge's
    /// 1e-9 band is tighter than `f32` resolution at O(1) magnitudes, so asserting it
    /// on the stored `f32` values would be asserting the rounding, not the gauge.
    /// Widen the gate if a later plan needs this outside tests.
    #[cfg(test)]
    pub(crate) fn intercepts_f64(&self) -> &[f64] {
        &self.intercepts_f64
    }

    /// Fits the head.
    ///
    /// Returns the [`HeadFitReport`] on convergence, or a typed [`HeadFitError`]
    /// otherwise. Non-convergence is an **error**, not a warning.
    ///
    /// # A failed `fit` leaves the head UNFITTED
    ///
    /// The fitted state is discarded on entry, before anything can fail. Otherwise a head
    /// that fitted once and was then re-fitted with data the gate rejects would keep the
    /// PREVIOUS fit's weights, `n_features`, labels and report — so `predict` would answer
    /// with a model the caller believes it failed to build, and `report()` would describe a
    /// converged run that the last call did not perform. `NotFitted` is the honest answer
    /// after a rejected fit, and it is only reachable if the state is cleared here.
    pub fn fit(
        &mut self,
        features: &[Vec<f32>],
        class_indices: &[usize],
        ordered_labels: &[String],
        regularization: Regularization,
    ) -> Result<HeadFitReport, HeadFitError> {
        self.n_features = None;
        self.weights.clear();
        self.intercepts.clear();
        self.intercepts_f64.clear();
        self.labels.clear();
        self.report = None;

        let validated = validate_fit_inputs(
            self.n_classes,
            features,
            class_indices,
            ordered_labels,
            regularization,
        )?;
        let d = validated.n_features;
        let problem = SoftmaxNllProblem {
            features,
            class_indices,
            n_classes: self.n_classes,
            n_features: d,
            lambda: validated.lambda,
        };

        // x0 = zeros. Fixed, not sampled: the fit path contains no randomness of any
        // kind, which is what makes two identical fits bitwise identical.
        let x0 = Vector::from_vec(vec![0.0_f64; problem.n_params()]);
        let mut solver = LbfgsF64::new(self.max_iter, self.tol, self.history_size);
        let result = solver.minimize(
            |x: &Vector<f64>| problem.objective(x),
            |x: &Vector<f64>| problem.gradient(x),
            &x0,
        );

        match result.status {
            ConvergenceStatus::Converged => {}
            ConvergenceStatus::MaxIterations => {
                return Err(HeadFitError::NotConverged {
                    iterations: result.iterations,
                    gradient_norm: result.gradient_norm,
                    tol: self.tol,
                })
            }
            ConvergenceStatus::Stalled => {
                return Err(HeadFitError::Stalled {
                    iterations: result.iterations,
                    gradient_norm: result.gradient_norm,
                })
            }
            ConvergenceStatus::NumericalError => {
                return Err(HeadFitError::NumericalError {
                    iterations: result.iterations,
                })
            }
            status @ (ConvergenceStatus::Running | ConvergenceStatus::UserTerminated) => {
                return Err(HeadFitError::Internal { status })
            }
        }

        let off = problem.intercept_offset();
        let solution = result.solution.as_slice();
        self.intercepts_f64 = solution[off..].to_vec();
        self.weights = solution[..off].iter().map(|&w| w as f32).collect();
        self.intercepts = self.intercepts_f64.iter().map(|&b| b as f32).collect();
        self.labels = ordered_labels.to_vec();
        self.n_features = Some(d);

        let report = HeadFitReport {
            status: result.status,
            iterations: result.iterations,
            final_grad_norm: result.gradient_norm,
            objective: result.objective_value,
        };
        self.report = Some(report.clone());
        Ok(report)
    }

    /// Per-row class logits, accumulated in `f64` from the `f32` store.
    ///
    /// THE single logit implementation: [`Self::predict_proba`] is this plus a
    /// softmax, so a caller that needs both — `VerifiedSetFitModel::classify`,
    /// which reports `logits` alongside `probabilities` — cannot obtain a pair
    /// that disagrees with itself. Extracting it also removed the standing
    /// temptation to write a second copy of the accumulation loop next to the
    /// caller that wanted logits.
    ///
    /// The accumulation ORDER is unchanged from the original `predict_proba`:
    /// intercept first, then `j` ascending. That order is what the artifact
    /// writer recorded its probe logits in, so it is load-bearing rather than
    /// incidental.
    pub fn predict_logits(&self, features: &[Vec<f32>]) -> Result<Vec<Vec<f64>>, HeadFitError> {
        let d = self.n_features.ok_or(HeadFitError::NotFitted)?;
        let k = self.n_classes;
        let mut out = Vec::with_capacity(features.len());
        for (row, values) in features.iter().enumerate() {
            if values.len() != d {
                return Err(HeadFitError::InvalidInput(
                    HeadInputError::FeatureDimMismatch {
                        row,
                        expected: d,
                        found: values.len(),
                    },
                ));
            }
            let mut logits = vec![0.0_f64; k];
            for c in 0..k {
                // Accumulate in f64 from the f32 store: a finite f32 row whose f32
                // dot product would overflow still yields a finite logit here.
                let mut z = f64::from(self.intercepts[c]);
                for j in 0..d {
                    z += f64::from(self.weights[c * d + j]) * f64::from(values[j]);
                }
                if !z.is_finite() {
                    return Err(HeadFitError::NonFiniteLogit { row, class: c });
                }
                logits[c] = z;
            }
            out.push(logits);
        }
        Ok(out)
    }

    /// Per-row class probabilities, computed with `f64` logit accumulation.
    ///
    /// Each returned row is finite and sums to 1 within `1e-6`.
    pub fn predict_proba(&self, features: &[Vec<f32>]) -> Result<Vec<Vec<f64>>, HeadFitError> {
        let rows = self.predict_logits(features)?;
        let mut out = Vec::with_capacity(rows.len());
        for logits in &rows {
            let mut probs = vec![0.0_f64; logits.len()];
            softmax_into(logits, &mut probs);
            out.push(probs);
        }
        Ok(out)
    }

    /// Per-row predicted class indices, breaking exact ties to the lowest index.
    pub fn predict_indices(&self, features: &[Vec<f32>]) -> Result<Vec<usize>, HeadFitError> {
        let probs = self.predict_proba(features)?;
        Ok(probs.iter().map(|p| argmax_lowest_index(p)).collect())
    }

    /// Per-row predicted labels, breaking exact ties to the lowest label index.
    pub fn predict(&self, features: &[Vec<f32>]) -> Result<Vec<String>, HeadFitError> {
        let indices = self.predict_indices(features)?;
        Ok(indices
            .into_iter()
            .map(|i| self.labels[i].clone())
            .collect())
    }
}

// =========================================================================
// Tests
//
// The tests live inline (rather than in a sibling file) because plan 03-04
// declares exactly one implementation file for this head, and a test module
// registered from elsewhere would be a second one. The contract-level
// falsification suite is separate: `tests_multinomial_contract.rs`.
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_string()).collect()
    }

    /// A separable K=3 toy set: three well-spaced clusters in 2-D, 3 rows each.
    fn k3_separable() -> (Vec<Vec<f32>>, Vec<usize>, Vec<String>) {
        let features = vec![
            vec![-2.0, 0.0],
            vec![-2.2, 0.3],
            vec![-1.8, -0.3],
            vec![0.0, 2.0],
            vec![0.3, 2.2],
            vec![-0.3, 1.8],
            vec![2.0, 0.0],
            vec![2.2, -0.3],
            vec![1.8, 0.3],
        ];
        let class_indices = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        (
            features,
            class_indices,
            labels(&["against", "favor", "none"]),
        )
    }

    /// A separable K=2 toy set — TRN-04's K >= 2 lower bound as a working fit path.
    fn k2_separable() -> (Vec<Vec<f32>>, Vec<usize>, Vec<String>) {
        let features = vec![
            vec![-1.5, -1.0],
            vec![-1.2, -1.4],
            vec![-1.8, -0.7],
            vec![1.5, 1.0],
            vec![1.2, 1.4],
            vec![1.8, 0.7],
        ];
        let class_indices = vec![0, 0, 0, 1, 1, 1];
        (features, class_indices, labels(&["no", "yes"]))
    }

    /// A dataset in which every feature row appears once under EVERY class.
    ///
    /// At the zero initialization this makes the analytic gradient exactly zero for
    /// K=2 (and ~1e-17 for K=3, far below any usable tolerance), so L-BFGS converges
    /// at iteration 0 with the solution still exactly zeros. Every logit is then
    /// exactly 0.0 and every class probability is exactly 1/K — an EXACT tie, not a
    /// near-tie, which is what the lowest-index tie-break has to be tested against.
    fn duplicate_row_tie(k: usize) -> (Vec<Vec<f32>>, Vec<usize>, Vec<String>) {
        let base = [vec![0.7_f32, -1.3], vec![-0.4, 0.9]];
        let mut features = Vec::new();
        let mut class_indices = Vec::new();
        for row in &base {
            for c in 0..k {
                features.push(row.clone());
                class_indices.push(c);
            }
        }
        let names: Vec<String> = (0..k).map(|c| format!("class{c}")).collect();
        (features, class_indices, names)
    }

    fn fit_k3() -> MultinomialLogisticRegression {
        let (x, y, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        head.fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect("k3 separable fit converges");
        head
    }

    // ---------------------------------------------------------------------
    // Numeric helpers (log-sum-exp shift, tie-break)
    // ---------------------------------------------------------------------

    /// The shift is what makes large logits usable at all: `exp(1000)` is `inf` in
    /// f64 (the overflow threshold is ~709), so an unshifted softmax would produce
    /// `inf / inf = NaN` here.
    #[test]
    fn log_sum_exp_is_finite_for_logits_near_1000() {
        assert!(
            1000.0_f64.exp().is_infinite(),
            "precondition: naive exp(1000) must overflow f64, else this test proves nothing"
        );
        let logits = [1000.0, 1001.0, 999.0];
        let lse = log_sum_exp(&logits);
        assert!(lse.is_finite(), "log_sum_exp({logits:?}) = {lse}");
        // log(e^1000 + e^1001 + e^999) = 1001 + log(e^-1 + 1 + e^-2)
        let expected = 1001.0 + ((-1.0_f64).exp() + 1.0 + (-2.0_f64).exp()).ln();
        assert!((lse - expected).abs() < 1e-9, "lse {lse} != {expected}");
    }

    #[test]
    fn softmax_into_is_finite_and_sums_to_one_for_logits_near_1000() {
        let logits = [1000.0, 1001.0, 999.0];
        let mut out = [0.0; 3];
        softmax_into(&logits, &mut out);
        let sum: f64 = out.iter().sum();
        for (c, p) in out.iter().enumerate() {
            assert!(p.is_finite(), "probability[{c}] = {p} is not finite");
            assert!(
                (0.0..=1.0).contains(p),
                "probability[{c}] = {p} out of [0,1]"
            );
        }
        assert!((sum - 1.0).abs() < 1e-12, "probabilities sum to {sum}");
        assert_eq!(argmax_lowest_index(&out), 1, "largest logit is index 1");
    }

    #[test]
    fn argmax_lowest_index_breaks_exact_ties_to_lowest_index() {
        assert_eq!(argmax_lowest_index(&[0.5, 0.5, 0.5]), 0);
        assert_eq!(argmax_lowest_index(&[0.1, 0.6, 0.6]), 1);
        assert_eq!(argmax_lowest_index(&[0.6, 0.1, 0.6]), 0);
        assert_eq!(argmax_lowest_index(&[0.1, 0.2, 0.7]), 2);
    }

    // ---------------------------------------------------------------------
    // Regularization relation
    // ---------------------------------------------------------------------

    /// D-04 as amended: `lambda = 1/(2*C*n)`, where `n` is the ROW count.
    #[test]
    fn sklearn_equivalent_c_resolves_to_one_over_two_c_n_rows() {
        let r = Regularization::SklearnEquivalentC { c: 1.0 };
        // 8-shot x K=3 => 24 rows. The correct value is 1/48, NOT 1/24: sklearn's
        // r(W) = (1/2)||W||^2_F carries the half that this head's lambda*||W||^2_F
        // does not.
        assert!((r.resolve_lambda(24) - 1.0 / 48.0).abs() < 1e-15);
        assert!(
            (r.resolve_lambda(24) - 1.0 / 24.0).abs() > 1e-3,
            "the factor-2 error must be separable at n=24"
        );
        assert!((Regularization::Lambda(0.07).resolve_lambda(24) - 0.07).abs() < 1e-15);
    }

    // ---------------------------------------------------------------------
    // Fit / predict behavior
    // ---------------------------------------------------------------------

    #[test]
    fn k3_separable_fit_returns_ok_with_finite_probabilities_summing_to_one() {
        let head = fit_k3();
        let report = head.report().expect("report recorded");
        assert_eq!(report.status, ConvergenceStatus::Converged);
        assert_eq!(head.n_features(), Some(2));
        assert_eq!(head.weights().len(), 3 * 2);
        assert_eq!(head.intercepts().len(), 3);

        let (x, y, l) = k3_separable();
        let probs = head.predict_proba(&x).expect("predict_proba");
        assert_eq!(probs.len(), x.len());
        for (i, row) in probs.iter().enumerate() {
            assert_eq!(row.len(), 3);
            let mut sum = 0.0;
            for (c, p) in row.iter().enumerate() {
                assert!(p.is_finite(), "p[{i}][{c}] = {p} is not finite");
                sum += p;
            }
            assert!((sum - 1.0).abs() < 1e-6, "row {i} sums to {sum}");
        }
        let predicted = head.predict(&x).expect("predict");
        for (i, label) in predicted.iter().enumerate() {
            assert_eq!(label, &l[y[i]], "row {i} misclassified on separable data");
        }
    }

    /// TRN-04's `K >= 2` lower bound, exercised as a working fit/predict path rather
    /// than merely as a `K < 2` rejection.
    #[test]
    fn k2_binary_boundary_fit_predict_proba_and_labels() {
        let (x, y, l) = k2_separable();
        let mut head = MultinomialLogisticRegression::new(2);
        let report = head
            .fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect("k2 separable fit converges");
        assert_eq!(report.status, ConvergenceStatus::Converged);
        assert_eq!(head.weights().len(), 2 * 2);
        assert_eq!(head.intercepts().len(), 2);

        let probs = head.predict_proba(&x).expect("predict_proba");
        for (i, row) in probs.iter().enumerate() {
            assert_eq!(row.len(), 2);
            let sum: f64 = row.iter().sum();
            assert!(row.iter().all(|p| p.is_finite()), "row {i} not finite");
            assert!((sum - 1.0).abs() < 1e-6, "row {i} sums to {sum}");
        }
        let predicted = head.predict(&x).expect("predict");
        for (i, label) in predicted.iter().enumerate() {
            assert!(
                l.contains(label),
                "row {i} predicted {label:?}, not in {l:?}"
            );
            assert_eq!(label, &l[y[i]], "row {i} misclassified on separable data");
        }
    }

    #[test]
    fn predict_breaks_exact_probability_tie_to_lowest_label_index() {
        for k in [2_usize, 3] {
            let (x, y, l) = duplicate_row_tie(k);
            let mut head = MultinomialLogisticRegression::new(k);
            head.fit(&x, &y, &l, Regularization::Lambda(0.0))
                .expect("degenerate duplicate-row fit converges at iteration 0");

            // The construction is only a valid tie test if the fit really did land on
            // exactly zero parameters.
            assert!(
                head.weights().iter().all(|w| *w == 0.0),
                "K={k}: weights are not exactly zero: {:?}",
                head.weights()
            );
            assert!(
                head.intercepts().iter().all(|b| *b == 0.0),
                "K={k}: intercepts are not exactly zero: {:?}",
                head.intercepts()
            );

            let probe = vec![vec![0.7_f32, -1.3]];
            let probs = head.predict_proba(&probe).expect("predict_proba");
            for c in 1..k {
                assert_eq!(
                    probs[0][c], probs[0][0],
                    "K={k}: probabilities must be EXACTLY tied, got {:?}",
                    probs[0]
                );
            }
            let predicted = head.predict(&probe).expect("predict");
            assert_eq!(
                predicted[0], l[0],
                "K={k}: an exact tie must resolve to the lowest label index"
            );
        }
    }

    /// Review fix 3: finite `f32` inputs can still overflow an `f32` accumulator. The
    /// head accumulates in `f64`, so this row yields finite probabilities where an
    /// `f32` dot product would have produced an infinity.
    #[test]
    fn predict_near_f32_max_row_accumulates_in_f64_where_f32_would_overflow() {
        let head = fit_k3();
        let d = head.n_features().expect("fitted");
        let w = head.weights();

        // Pick the class with the largest L1 weight and align the row's signs with it
        // so every product accumulates in the same direction.
        let mut best_class = 0;
        let mut best_l1 = 0.0_f32;
        for c in 0..head.n_classes() {
            let l1: f32 = (0..d).map(|j| w[c * d + j].abs()).sum();
            if l1 > best_l1 {
                best_l1 = l1;
                best_class = c;
            }
        }
        assert!(
            best_l1 > 1.0,
            "precondition: sum |w| = {best_l1} must exceed 1 for the f32 accumulation \
             to overflow at f32::MAX inputs"
        );
        let row: Vec<f32> = (0..d)
            .map(|j| {
                if w[best_class * d + j] >= 0.0 {
                    f32::MAX
                } else {
                    -f32::MAX
                }
            })
            .collect();
        assert!(
            row.iter().all(|v| v.is_finite()),
            "every individual feature value must be finite"
        );

        // Witness: the SAME accumulation carried out in f32 overflows.
        let mut z32 = head.intercepts()[best_class];
        for j in 0..d {
            z32 += w[best_class * d + j] * row[j];
        }
        assert!(
            !z32.is_finite(),
            "witness failed: the f32 accumulation produced {z32}, so this row does not \
             exercise the overflow this test exists for"
        );

        let probs = head
            .predict_proba(&[row])
            .expect("f64 accumulation keeps the logits finite");
        let sum: f64 = probs[0].iter().sum();
        assert!(
            probs[0].iter().all(|p| p.is_finite()),
            "probabilities must be finite, got {:?}",
            probs[0]
        );
        assert!((sum - 1.0).abs() < 1e-6, "probabilities sum to {sum}");
        assert_eq!(
            argmax_lowest_index(&probs[0]),
            best_class,
            "the aligned class must dominate"
        );
    }

    /// The typed error exists and is reachable: a non-finite feature at prediction
    /// time produces a non-finite logit, and that is reported as
    /// `NonFiniteLogit { row, class }` rather than allowed into the softmax.
    #[test]
    fn predict_row_with_infinite_feature_yields_nonfinite_logit_error() {
        let head = fit_k3();
        let err = head
            .predict_proba(&[vec![1.0, f32::INFINITY]])
            .expect_err("an infinite feature must not produce a probability");
        match err {
            HeadFitError::NonFiniteLogit { row, class } => {
                assert_eq!(row, 0);
                assert!(class < 3, "class {class} out of range");
            }
            other => panic!("expected NonFiniteLogit, got {other:?}"),
        }
    }

    #[test]
    fn max_iter_one_on_nontrivial_data_returns_not_converged() {
        let (x, y, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3).with_max_iter(1);
        let err = head
            .fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect_err("one iteration cannot converge on this data");
        match err {
            HeadFitError::NotConverged {
                iterations,
                gradient_norm,
                tol,
            } => {
                assert_eq!(iterations, 1, "the budget was 1 iteration");
                assert!(gradient_norm > tol, "grad {gradient_norm} vs tol {tol}");
                assert!((tol - DEFAULT_TOL).abs() < 1e-18);
            }
            other => panic!("expected NotConverged, got {other:?}"),
        }
        assert!(head.report().is_none(), "a failed fit records no report");
    }

    /// Review fix 4 — the intercept gauge is a PROPERTY, not an optimizer accident.
    ///
    /// The NLL gradient with respect to the intercept block is
    /// `g_b[k] = (1/n) sum_i (p_ik - [k == y_i])`, whose sum over `k` is
    /// `(1/n) sum_i (1 - 1) = 0` at EVERY point, and the penalty contributes nothing
    /// there. L-BFGS only ever moves along linear combinations of gradients and stored
    /// differences, all of which have a zero-sum intercept block, so a zero-initialized
    /// intercept never leaves the centered gauge. If this assertion ever fails, the
    /// analytic gradient is wrong.
    #[test]
    fn fitted_intercept_mean_is_zero_within_1e_9() {
        for (x, y, l, k) in [
            {
                let (x, y, l) = k3_separable();
                (x, y, l, 3)
            },
            {
                let (x, y, l) = k2_separable();
                (x, y, l, 2)
            },
        ] {
            let mut head = MultinomialLogisticRegression::new(k);
            head.fit(&x, &y, &l, Regularization::Lambda(0.01))
                .expect("fit converges");
            let b = head.intercepts_f64();
            assert_eq!(b.len(), k);
            let mean: f64 = b.iter().sum::<f64>() / k as f64;
            assert!(
                mean.abs() < 1e-9,
                "K={k}: intercept mean {mean:e} left the centered gauge (b = {b:?})"
            );
        }
    }

    #[test]
    fn two_identical_fits_produce_bitwise_identical_f32_weights() {
        let (x, y, l) = k3_separable();
        let mut a = MultinomialLogisticRegression::new(3);
        let mut b = MultinomialLogisticRegression::new(3);
        let ra = a
            .fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect("fit a");
        let rb = b
            .fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect("fit b");
        assert_eq!(ra, rb, "identical fits must produce identical reports");

        let bits_a: Vec<u32> = a.weights().iter().map(|w| w.to_bits()).collect();
        let bits_b: Vec<u32> = b.weights().iter().map(|w| w.to_bits()).collect();
        assert_eq!(
            bits_a, bits_b,
            "stored f32 weights are not bitwise identical"
        );

        let ib_a: Vec<u32> = a.intercepts().iter().map(|v| v.to_bits()).collect();
        let ib_b: Vec<u32> = b.intercepts().iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            ib_a, ib_b,
            "stored f32 intercepts are not bitwise identical"
        );
    }

    /// The report is hashable-by-construction: no wall-clock field, and its canonical
    /// JSON is byte-identical across two independent runs of the same fit.
    #[test]
    fn head_fit_report_serializes_stably_across_two_runs() {
        let (x, y, l) = k3_separable();
        let mut a = MultinomialLogisticRegression::new(3);
        let mut b = MultinomialLogisticRegression::new(3);
        let ra = a
            .fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect("fit a");
        let rb = b
            .fit(&x, &y, &l, Regularization::Lambda(0.01))
            .expect("fit b");
        let ja = serde_json::to_string(&ra).expect("serialize a");
        let jb = serde_json::to_string(&rb).expect("serialize b");
        assert_eq!(ja, jb, "report JSON is not stable across runs");
        assert!(
            ja.contains("final_grad_norm") && ja.contains("objective"),
            "report JSON missing expected fields: {ja}"
        );
        // EVERY field survives the round trip exactly, f64 fields included. This used to
        // be a one-ULP tolerance; `float_roundtrip` (see
        // `json_roundtrip_of_an_f64_is_bit_exact`) removed the reason for it, and the
        // tolerance is removed with it — a tolerance kept after its cause is gone is a
        // hole that admits a real drift later.
        let round: HeadFitReport = serde_json::from_str(&ja).expect("deserialize");
        assert_eq!(round.status, ra.status);
        assert_eq!(round.iterations, ra.iterations);
        assert_eq!(
            round.objective.to_bits(),
            ra.objective.to_bits(),
            "objective must survive the round trip bit for bit"
        );
        assert_eq!(
            round.final_grad_norm.to_bits(),
            ra.final_grad_norm.to_bits(),
            "final_grad_norm must survive the round trip bit for bit"
        );
        assert_eq!(
            serde_json::to_string(&round).expect("re-serialize"),
            ja,
            "re-serializing a parsed report must reproduce its own bytes"
        );
    }

    /// `from_str(to_string(x))` returns `x` bit for bit — the `float_roundtrip` proof.
    ///
    /// # This test used to assert the opposite, and the difference is a Cargo feature
    ///
    /// `serde_json`'s DEFAULT float parser is fast and not correctly rounded, so it can
    /// land one ULP from the value ryu wrote. This head's own converged gradient norm was
    /// such a value, measured on serde_json 1.0 without the feature:
    ///
    /// ```text
    /// v            = 2.1531120041346774e-5   bits 0x3ef693b74d831429
    /// to_string(v) = "0.000021531120041346774"      (ryu, positional, exact)
    /// from_str(..) = 2.1531120041346778e-5   bits 0x3ef693b74d83142a   (+1 ULP)
    /// ```
    ///
    /// This crate now declares `serde_json`'s `float_roundtrip` feature, which selects the
    /// correctly-rounding parse path, and the drift is gone. Plan 03-08 is what forced the
    /// question: its persistence boundary hashes a serialized bundle and then re-serializes
    /// the reloaded one and compares the bytes, and a one-ULP parse made every honest codec
    /// fail that check — a spurious mismatch reported as a tampered artifact.
    ///
    /// The test asserts the BEHAVIOUR, not the presence of the flag. A feature can be
    /// declared and inert — unification, a vendored copy, a future default change — and
    /// what the reload boundary depends on is the behaviour.
    #[test]
    fn json_roundtrip_of_an_f64_is_bit_exact() {
        // A value that round-tripped cleanly even before the feature, so this test is not
        // merely asserting that everything works.
        let clean = 0.11346603265462092_f64;
        let s_clean = serde_json::to_string(&clean).expect("serialize");
        let back_clean: f64 = serde_json::from_str(&s_clean).expect("deserialize");
        assert_eq!(
            back_clean.to_bits(),
            clean.to_bits(),
            "{s_clean} should round-trip"
        );

        // The value that did NOT, before `float_roundtrip`.
        let measured = 2.1531120041346774e-5_f64;
        assert_eq!(measured.to_bits(), 0x3ef6_93b7_4d83_1429);
        let serialized = serde_json::to_string(&measured).expect("serialize");
        assert_eq!(
            serialized, "0.000021531120041346774",
            "the SERIALIZED form is the stable, exact one; if this changed, re-measure the \
             whole observation below rather than patching the constant"
        );
        let back: f64 = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(
            back.to_bits(),
            measured.to_bits(),
            "`{serialized}` parsed back to bits {:#018x} instead of {:#018x}. serde_json's \
             `float_roundtrip` feature is not engaged for this build, and every artifact \
             whose reload is verified by comparing re-serialized bytes will report a \
             spurious mismatch.",
            back.to_bits(),
            measured.to_bits(),
        );
    }

    #[test]
    fn predict_before_fit_returns_not_fitted() {
        let head = MultinomialLogisticRegression::new(3);
        assert_eq!(
            head.predict_proba(&[vec![0.0, 0.0]]),
            Err(HeadFitError::NotFitted)
        );
        assert_eq!(
            head.predict_indices(&[vec![0.0, 0.0]]),
            Err(HeadFitError::NotFitted)
        );
    }

    // ---------------------------------------------------------------------
    // Input validation — one test per enumerated case, one distinct variant each
    // ---------------------------------------------------------------------

    /// Fits `head` on otherwise-valid K=3 data and returns the input error.
    fn expect_input_error(
        head: &mut MultinomialLogisticRegression,
        x: &[Vec<f32>],
        y: &[usize],
        l: &[String],
        r: Regularization,
    ) -> HeadInputError {
        match head.fit(x, y, l, r) {
            Err(HeadFitError::InvalidInput(e)) => e,
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn invalid_k_less_than_two() {
        let (x, y, _) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(1);
        let e = expect_input_error(
            &mut head,
            &x,
            &y,
            &labels(&["only"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(e, HeadInputError::TooFewClasses { k: 1 });
    }

    #[test]
    fn invalid_ordered_label_count_mismatch() {
        let (x, y, _) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(
            &mut head,
            &x,
            &y,
            &labels(&["a", "b"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(e, HeadInputError::LabelCountMismatch { labels: 2, k: 3 });
    }

    #[test]
    fn invalid_empty_ordered_label() {
        let (x, y, _) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(
            &mut head,
            &x,
            &y,
            &labels(&["a", "", "c"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(e, HeadInputError::EmptyLabel { index: 1 });
    }

    #[test]
    fn invalid_duplicate_ordered_labels() {
        let (x, y, _) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(
            &mut head,
            &x,
            &y,
            &labels(&["a", "b", "a"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(
            e,
            HeadInputError::DuplicateLabel {
                first: 0,
                second: 2,
                label: "a".to_string()
            }
        );
    }

    #[test]
    fn invalid_empty_dataset() {
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(
            &mut head,
            &[],
            &[],
            &labels(&["a", "b", "c"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(e, HeadInputError::EmptyDataset);
    }

    #[test]
    fn invalid_row_count_mismatch() {
        let (x, _, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &[0, 1, 2], &l, Regularization::Lambda(0.0));
        assert_eq!(
            e,
            HeadInputError::RowCountMismatch {
                rows: 9,
                class_indices: 3
            }
        );
    }

    #[test]
    fn invalid_zero_feature_dimension() {
        let mut head = MultinomialLogisticRegression::new(3);
        let x = vec![Vec::<f32>::new(), Vec::new(), Vec::new()];
        let e = expect_input_error(
            &mut head,
            &x,
            &[0, 1, 2],
            &labels(&["a", "b", "c"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(e, HeadInputError::ZeroFeatureDimension);
    }

    #[test]
    fn invalid_ragged_rows() {
        let mut head = MultinomialLogisticRegression::new(3);
        let x = vec![vec![0.0, 1.0], vec![1.0, 0.0, 2.0], vec![2.0, 2.0]];
        let e = expect_input_error(
            &mut head,
            &x,
            &[0, 1, 2],
            &labels(&["a", "b", "c"]),
            Regularization::Lambda(0.0),
        );
        assert_eq!(
            e,
            HeadInputError::RaggedRow {
                row: 1,
                expected: 2,
                found: 3
            }
        );
    }

    #[test]
    fn invalid_nan_feature() {
        let (mut x, y, l) = k3_separable();
        x[4][1] = f32::NAN;
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &y, &l, Regularization::Lambda(0.0));
        assert_eq!(e, HeadInputError::NanFeature { row: 4, col: 1 });
    }

    #[test]
    fn invalid_infinite_feature() {
        let (mut x, y, l) = k3_separable();
        x[7][0] = f32::INFINITY;
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &y, &l, Regularization::Lambda(0.0));
        assert_eq!(
            e,
            HeadInputError::InfiniteFeature {
                row: 7,
                col: 0,
                value: f32::INFINITY
            }
        );
    }

    #[test]
    fn invalid_label_index_out_of_range() {
        let (x, mut y, l) = k3_separable();
        y[5] = 3;
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &y, &l, Regularization::Lambda(0.0));
        assert_eq!(
            e,
            HeadInputError::LabelIndexOutOfRange {
                row: 5,
                index: 3,
                k: 3
            }
        );
    }

    #[test]
    fn invalid_unrepresented_class() {
        let (x, mut y, l) = k3_separable();
        for v in &mut y {
            if *v == 1 {
                *v = 0;
            }
        }
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &y, &l, Regularization::Lambda(0.0));
        assert_eq!(e, HeadInputError::UnrepresentedClass { class: 1 });
    }

    #[test]
    fn invalid_negative_lambda() {
        let (x, y, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &y, &l, Regularization::Lambda(-1e-6));
        assert_eq!(e, HeadInputError::NegativeLambda { lambda: -1e-6 });
    }

    #[test]
    fn invalid_nan_lambda() {
        let (x, y, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(&mut head, &x, &y, &l, Regularization::Lambda(f64::NAN));
        match e {
            HeadInputError::NonFiniteLambda { lambda } => assert!(lambda.is_nan()),
            other => panic!("expected NonFiniteLambda, got {other:?}"),
        }
    }

    #[test]
    fn invalid_sklearn_c_non_positive() {
        let (x, y, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(
            &mut head,
            &x,
            &y,
            &l,
            Regularization::SklearnEquivalentC { c: 0.0 },
        );
        assert_eq!(e, HeadInputError::NonPositiveC { c: 0.0 });
    }

    #[test]
    fn invalid_sklearn_c_non_finite() {
        let (x, y, l) = k3_separable();
        let mut head = MultinomialLogisticRegression::new(3);
        let e = expect_input_error(
            &mut head,
            &x,
            &y,
            &l,
            Regularization::SklearnEquivalentC { c: f64::INFINITY },
        );
        assert_eq!(e, HeadInputError::NonFiniteC { c: f64::INFINITY });
    }

    #[test]
    fn invalid_predict_feature_dim_mismatch() {
        let head = fit_k3();
        for probe in [vec![vec![1.0_f32]], vec![vec![1.0_f32, 2.0, 3.0]]] {
            let found = probe[0].len();
            let err = head
                .predict_proba(&probe)
                .expect_err("dimension mismatch must be rejected");
            assert_eq!(
                err,
                HeadFitError::InvalidInput(HeadInputError::FeatureDimMismatch {
                    row: 0,
                    expected: 2,
                    found
                })
            );
            let err = head
                .predict(&probe)
                .expect_err("dimension mismatch must be rejected by predict too");
            assert_eq!(
                err,
                HeadFitError::InvalidInput(HeadInputError::FeatureDimMismatch {
                    row: 0,
                    expected: 2,
                    found
                })
            );
        }
    }

    // ---------------------------------------------------------------------
    // Error surface
    // ---------------------------------------------------------------------

    #[test]
    fn typed_errors_render_informative_messages() {
        let cases: Vec<(HeadInputError, &str)> = vec![
            (HeadInputError::TooFewClasses { k: 1 }, "K >= 2"),
            (HeadInputError::EmptyDataset, "no feature rows"),
            (HeadInputError::ZeroFeatureDimension, "dimension is 0"),
            (HeadInputError::NonPositiveC { c: 0.0 }, "strictly positive"),
        ];
        for (e, needle) in cases {
            let rendered = e.to_string();
            assert!(
                rendered.contains(needle),
                "{rendered:?} does not mention {needle:?}"
            );
        }
        let wrapped = HeadFitError::from(HeadInputError::EmptyDataset);
        assert!(wrapped.to_string().starts_with("invalid input:"));
        assert!(HeadFitError::NotFitted
            .to_string()
            .contains("not been fitted"));
        assert!(HeadFitError::NonFiniteLogit { row: 2, class: 1 }
            .to_string()
            .contains("row 2, class 1"));
        assert!(HeadFitError::Internal {
            status: ConvergenceStatus::Running
        }
        .to_string()
        .contains("Running"));
    }

    // ---------------------------------------------------------------------
    // Problem layout
    // ---------------------------------------------------------------------

    #[test]
    fn problem_layout_places_intercepts_after_the_weight_block() {
        let (x, y, _) = k3_separable();
        let problem = SoftmaxNllProblem {
            features: &x,
            class_indices: &y,
            n_classes: 3,
            n_features: 2,
            lambda: 0.0,
        };
        assert_eq!(problem.intercept_offset(), 6);
        assert_eq!(problem.n_params(), 9);
    }

    /// At the zero initialization every class is equiprobable, so the objective is
    /// exactly `log K` and the penalty contributes nothing.
    #[test]
    fn objective_at_zero_is_log_k() {
        let (x, y, _) = k3_separable();
        let problem = SoftmaxNllProblem {
            features: &x,
            class_indices: &y,
            n_classes: 3,
            n_features: 2,
            lambda: 0.07,
        };
        let x0 = Vector::from_vec(vec![0.0; problem.n_params()]);
        let j = problem.objective(&x0);
        assert!(
            (j - 3.0_f64.ln()).abs() < 1e-12,
            "objective at zero = {j}, expected ln(3) = {}",
            3.0_f64.ln()
        );
    }

    /// The intercept block of the gradient sums to zero at every point — the identity
    /// the gauge property rests on.
    #[test]
    fn gradient_intercept_block_sums_to_zero_at_every_point() {
        let (x, y, _) = k3_separable();
        let problem = SoftmaxNllProblem {
            features: &x,
            class_indices: &y,
            n_classes: 3,
            n_features: 2,
            lambda: 0.07,
        };
        for scale in [0.0_f64, 0.3, -1.7] {
            let params: Vec<f64> = (0..problem.n_params())
                .map(|i| scale * ((i % 5) as f64 - 2.0))
                .collect();
            let g = problem.gradient(&Vector::from_vec(params));
            let off = problem.intercept_offset();
            let sum: f64 = (0..3).map(|c| g[off + c]).sum();
            assert!(
                sum.abs() < 1e-12,
                "scale {scale}: intercept gradient block sums to {sum:e}, not 0"
            );
        }
    }
}
