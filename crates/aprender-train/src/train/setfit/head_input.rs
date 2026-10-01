//! Stage two's INPUT: every unique selected row encoded exactly once (D-08, TRN-05).
//!
//! Contract: `setfit-train-lifecycle-v1`. Requirement: TRN-05.
//!
//! # The exactly-once claim is MEASURED, not inferred from a row count
//!
//! A row count cannot see the defect it is supposed to exclude. An implementation that
//! encodes row 3 twice and skips row 5 produces the same number of rows, in the same order,
//! deterministically — every count-based and every re-encode-determinism assertion stays
//! green. What distinguishes the two is the MULTISET of identifiers actually handed to the
//! encoder, so this module records exactly that: an ordered `encode_ledger`, appended at the
//! encode call site from the SAME window slice the encoder is handed, plus the number of
//! encoder invocations.
//!
//! The ledger is deliberately NOT rebuilt from the selection afterwards. A ledger derived
//! from the selection's own id list would agree with the selection by construction and would
//! therefore be evidence about nothing; it is written from the window that is about to be
//! encoded, so a defect in the windowing appears in the ledger and in the embeddings
//! together.
//!
//! # The encode runs with the mechanism PROVEN engaged, not asserted
//!
//! Eval mode, `no_grad` and `detach` are all in the code path below, and all three are
//! OBSERVED while they run rather than described afterwards: [`EncodeWitness`] records the
//! encoder's own `training()` flag sampled inside every window, the `requires_grad_enabled()`
//! flag of every detached embedding tensor, and the autograd tape length immediately before
//! the first encode and immediately after the last.
//!
//! The witness is then CHECKED, on the shipped path, before the input is returned. An
//! observation nothing acts on is a comment with a struct around it, and it would leave the
//! reproducibility of the head's embeddings resting on a test that a future encoder change
//! could quietly stop covering. A test additionally compares the tape length and the gradient
//! of every trainable parameter across the whole call.
//!
//! # There is no multiplicity-shaped input here
//!
//! Nothing in this module's signatures can express "encode this row n times". The window
//! composition is a consecutive chunking of the selection's own order, and the only count in
//! the file is the batch size. That is the structural half of TRN-05; the adversarial half
//! lives in `negative.rs`.

use aprender::autograd;
use aprender::classification::{
    HeadFitReport, MultinomialLogisticRegression, Regularization, DEFAULT_MAX_ITER,
};
use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;

use super::config::{HeadRegularization, ResolvedSetFitConfig};
use super::tune::selection_texts;
use super::SetFitTrainError;

/// What was OBSERVED while the encode ran.
///
/// Every field is sampled inside the encode loop. Nothing here is a restatement of what the
/// code was written to do: a witness that recorded intent would be exactly the false green
/// this type exists to remove.
#[derive(Debug, Clone)]
pub(crate) struct EncodeWitness {
    /// `encoder.training()`, OR-ed over every window. `false` iff eval mode held throughout.
    pub(crate) training_observed: bool,
    /// `requires_grad_enabled()` of every detached embedding tensor, OR-ed.
    pub(crate) requires_grad_observed: bool,
    /// Autograd tape length immediately before the first encode.
    pub(crate) tape_before: usize,
    /// Autograd tape length immediately after the last encode.
    pub(crate) tape_after: usize,
}

impl EncodeWitness {
    /// Refuse the input unless all three mechanisms actually held.
    ///
    /// The witness is CHECKED on the shipped path, not merely recorded for a test to look at.
    /// An observation nothing acts on is a comment with a struct around it: if a future
    /// encoder change starts recording under `no_grad`, or a mode flip is added above the
    /// encode, the head's input silently stops being reproducible and every provenance claim
    /// built on it becomes false. This makes that a typed refusal instead.
    ///
    /// # Errors
    ///
    /// [`SetFitTrainError::HeadEncodeNotIsolated`], naming which of the three failed.
    fn require_isolated(&self) -> Result<(), SetFitTrainError> {
        if self.training_observed
            || self.requires_grad_observed
            || self.tape_after != self.tape_before
        {
            return Err(SetFitTrainError::HeadEncodeNotIsolated {
                training_observed: self.training_observed,
                requires_grad_observed: self.requires_grad_observed,
                // MAGNITUDE, not `saturating_sub`. A tape that SHRANK across the encode is
                // just as much a lost-isolation signal as one that grew, and a saturating
                // subtraction reported it as `0` — producing a refusal whose three payload
                // fields all read clean (`false`, `false`, `0`) while the run was rejected,
                // which is a diagnosis nobody can act on.
                tape_growth: self.tape_after.abs_diff(self.tape_before),
            });
        }
        Ok(())
    }
}

/// The head's fitting input: one embedding row per unique selected row, in selection order.
#[derive(Debug)]
pub(crate) struct HeadDataset {
    embeddings: Vec<Vec<f32>>,
    class_indices: Vec<usize>,
    ordered_labels: Vec<String>,
    encode_ledger: Vec<String>,
    encode_call_count: usize,
    witness: EncodeWitness,
}

impl HeadDataset {
    /// The embedding rows, aligned with the selection's order.
    pub(crate) fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// The class index of each row.
    pub(crate) fn class_indices(&self) -> &[usize] {
        &self.class_indices
    }

    /// The declared label map, in label order — from `PreparedDataset::label_names`.
    pub(crate) fn ordered_labels(&self) -> &[String] {
        &self.ordered_labels
    }

    /// The UNIQUE row count. The only `n` the head's lambda may be resolved against.
    ///
    /// Read off `embeddings` rather than stored beside it. A second copy of this number
    /// would be state a future edit could desynchronise from the vector it counts, and —
    /// unlike `encode_ledger` and `encode_call_count`, which are independent observations —
    /// it proves nothing the vector does not already say.
    pub(crate) fn n(&self) -> usize {
        self.embeddings.len()
    }

    /// The ordered identifiers actually handed to the encoder.
    ///
    /// `#[cfg(test)]`: the shipped path MOVES the ledger out through
    /// [`HeadDataset::into_evidence_parts`] rather than borrowing it, so this borrow exists
    /// only for the assertions that read the ledger off an intermediate `HeadDataset`.
    #[cfg(test)]
    pub(crate) fn encode_ledger(&self) -> &[String] {
        &self.encode_ledger
    }

    /// How many times the encoder was invoked. `#[cfg(test)]` for the same reason as above.
    #[cfg(test)]
    pub(crate) fn encode_call_count(&self) -> usize {
        self.encode_call_count
    }

    /// What was observed while the encode ran.
    pub(crate) fn witness(&self) -> &EncodeWitness {
        &self.witness
    }

    /// Hand the probe's two halves to the caller by MOVE: embeddings and the RECORDED ledger.
    ///
    /// A separate destructor from [`Self::into_evidence_parts`] because the two callers want
    /// different halves — stage two wants the labels and the ledger and drops the embeddings,
    /// 03-08's verify probe wants the embeddings and the ledger and already has the labels.
    /// One destructor returning all four would hand each caller a `Vec` it has to remember to
    /// ignore.
    ///
    /// The ids come from HERE rather than from `selection.ordered_ids()` on purpose. The
    /// ledger was written window by window as the encoder was fed; a list rebuilt from the
    /// selection would agree with the selection by construction and could not see a windowing
    /// defect at all, which is the whole reason the ledger is recorded in band.
    pub(crate) fn into_probe_parts(self) -> (Vec<Vec<f32>>, Vec<String>) {
        (self.embeddings, self.encode_ledger)
    }

    /// Hand the evidence-bound parts to the caller by MOVE.
    ///
    /// `fit_head` drops this dataset on the next line, so cloning the labels and the ledger
    /// out of it allocates one `String` per selected row only to free the original.
    pub(crate) fn into_evidence_parts(self) -> (Vec<String>, Vec<String>, usize) {
        (self.ordered_labels, self.encode_ledger, self.encode_call_count)
    }
}

/// Build the head's input: each unique selected row encoded exactly once.
///
/// # Errors
///
/// [`SetFitTrainError::HeadEncodeBatchSizeZero`] for a zero window size,
/// [`SetFitTrainError::SelectionLabelOutOfRange`] for a class index outside the declared
/// label map, [`SetFitTrainError::SelectionRowMissing`] for a selected id that names no row
/// of `dataset.train()`, and [`SetFitTrainError::Encoder`] for anything the encoder rejects.
pub(crate) fn head_dataset(
    encoder: &mut SetFitMiniLm,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    batch_size: u32,
) -> Result<HeadDataset, SetFitTrainError> {
    let batch = window_size(batch_size)?;

    let ordered_labels = dataset.label_names().to_vec();
    let class_indices = class_indices_of(&ordered_labels, selection)?;

    // The id and the text of each selected row, in selection order, held together in ONE
    // vector. The windows below are slices of THIS vector, and both the ledger entry and the
    // encoder's input come from the same slice — so a windowing defect that duplicates one
    // row and drops another shows up in the ledger and in the embeddings together. A ledger
    // rebuilt afterwards from `selection.ordered_ids()` would agree with the selection by
    // construction and could not see that defect at all.
    let texts = selection_texts(dataset, selection)?;
    let rows: Vec<(&str, &str)> =
        selection.ordered_ids().into_iter().zip(texts.iter().map(String::as_str)).collect();
    debug_assert_eq!(rows.len(), selection.len(), "one row per selected example");

    // Eval mode BEFORE anything is encoded. The transition owns the encoder's mode; it is
    // left in eval on the way out, which is the state every later read expects.
    encoder.set_training(false);
    let encoded = encode_once(encoder, &rows, batch)?;

    let built = HeadDataset {
        embeddings: encoded.embeddings,
        class_indices,
        ordered_labels,
        encode_ledger: encoded.encode_ledger,
        encode_call_count: encoded.encode_call_count,
        witness: encoded.witness,
    };
    // Read back through the accessor the tests read, so the gate and the assertions are
    // looking at the same value rather than at two copies of it.
    built.witness().require_isolated()?;
    Ok(built)
}

/// The window size a `u32` knob names, refusing zero BEFORE any `chunks` call.
///
/// One function rather than a copy in each caller: `chunks(0)` panics, so this refusal is the
/// only thing standing between a zero knob and an abort, and a guard that has to be re-typed at
/// every new encode door is a guard that will eventually be missing from one of them. It stays
/// at the CALLER rather than inside [`encode_once`] so the refusal keeps its precedence — a
/// zero window is reported before the selection and label-map work each caller does first.
fn window_size(batch_size: u32) -> Result<usize, SetFitTrainError> {
    let batch = batch_size as usize;
    if batch == 0 {
        return Err(SetFitTrainError::HeadEncodeBatchSizeZero);
    }
    Ok(batch)
}

/// The encode's outputs and the observations taken while it ran.
struct Encoded {
    embeddings: Vec<Vec<f32>>,
    encode_ledger: Vec<String>,
    encode_call_count: usize,
    witness: EncodeWitness,
}

/// One encoder invocation per consecutive window, inside `no_grad`, detaching every result.
fn encode_once(
    encoder: &SetFitMiniLm,
    rows: &[(&str, &str)],
    batch: usize,
) -> Result<Encoded, SetFitTrainError> {
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
    let mut encode_ledger: Vec<String> = Vec::with_capacity(rows.len());
    let mut encode_call_count = 0_usize;
    let mut training_observed = false;
    let mut requires_grad_observed = false;

    // One buffer for every window, refilled in place. The encoder wants a contiguous
    // `&[&str]`, but it does not want a fresh allocation per batch.
    let mut window_texts: Vec<&str> = Vec::with_capacity(batch.min(rows.len().max(1)));

    let tape_before = autograd::graph_tape_len();
    let result = autograd::no_grad(|| -> Result<(), SetFitTrainError> {
        for window in rows.chunks(batch) {
            // The ledger is written HERE, from the window that is about to be encoded.
            for (id, _) in window {
                encode_ledger.push((*id).to_string());
            }
            training_observed |= encoder.training();
            encode_call_count += 1;
            // Same slice the ledger was just written from — the property the module doc
            // rests on is that these two never come from different sources.
            window_texts.clear();
            window_texts.extend(window.iter().map(|&(_, text)| text));
            let embedded = encoder
                .encode_texts(&window_texts)
                .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;
            // Detach before anything is stored, so no graph node survives into the head's
            // input even if a future encoder change starts recording under `no_grad`.
            let detached = embedded.detach();
            requires_grad_observed |= detached.requires_grad_enabled();
            push_rows(&mut embeddings, &detached, window.len())?;
        }
        Ok(())
    });
    result?;
    let tape_after = autograd::graph_tape_len();

    Ok(Encoded {
        embeddings,
        encode_ledger,
        encode_call_count,
        witness: EncodeWitness {
            training_observed,
            requires_grad_observed,
            tape_before,
            tape_after,
        },
    })
}

/// Encode arbitrary `(id, text)` rows through the SAME encode-once path, at a SHARED borrow.
///
/// # Why this door exists rather than a direct `encode_texts` call at the use site
///
/// Plan 03-09's validation evaluator needs embeddings for the canonical validation split, and
/// verify.rs already records the reason a second encode path is unacceptable: it would measure
/// a model the trainer never ran. Everything that makes the trainer's encode reproducible —
/// windowing, `no_grad`, the `detach` before anything is stored, the shape check in
/// [`push_rows`], and the isolation witness — lives inside [`encode_once`], and this is how a
/// caller outside `head_dataset` reaches it.
///
/// # A SHARED borrow, and what that costs
///
/// [`head_dataset`] takes `&mut SetFitMiniLm` because the TRANSITION owns the encoder's mode
/// and sets eval before encoding. The evaluator holds a finished run and only has `&`, so this
/// function cannot set the mode — it OBSERVES it instead. The witness is checked exactly as
/// `head_dataset` checks it, so an encoder that arrived in training mode is a typed
/// [`SetFitTrainError::HeadEncodeNotIsolated`] rather than a quietly irreproducible metric.
/// That is fail-closed in the direction that matters: the verify transition leaves the encoder
/// in eval, so the shipped path passes and a future change that stops doing so is red.
///
/// # Errors
///
/// [`SetFitTrainError::HeadEncodeBatchSizeZero`] for a zero window size,
/// [`SetFitTrainError::HeadEncodeNotIsolated`] when the encode was not isolated, and
/// [`SetFitTrainError::Encoder`] for anything the encoder rejects.
pub(crate) fn encode_eval_rows(
    encoder: &SetFitMiniLm,
    rows: &[(&str, &str)],
    batch_size: u32,
) -> Result<Vec<Vec<f32>>, SetFitTrainError> {
    let encoded = encode_once(encoder, rows, window_size(batch_size)?)?;
    encoded.witness.require_isolated()?;
    Ok(encoded.embeddings)
}

/// Split a `[B, H]` embedding tensor into `expected` owned rows.
///
/// The shape is CHECKED rather than indexed. A `[B, H]` that arrived with the wrong `B` would
/// otherwise silently push a different number of rows than the ledger recorded, which is the
/// one way the two could disagree without any windowing defect.
///
/// `pub(crate)` so the two OTHER eval-mode encode windows in this module tree — the tuning
/// loop's baseline encode and the frozen probe — split their `[B, H]` through the same checked
/// function. They previously indexed `shape()[1]` directly, which PANICS on a non-2-D return
/// and silently accepts a `B` that disagrees with the window; one guard reached from three
/// call sites is the only way the check cannot be present at one and absent at the others.
pub(crate) fn push_rows(
    out: &mut Vec<Vec<f32>>,
    embedded: &autograd::Tensor,
    expected: usize,
) -> Result<(), SetFitTrainError> {
    let shape = embedded.shape();
    let malformed = |reason: String| SetFitTrainError::Encoder { reason };
    if shape.len() != 2 {
        return Err(malformed(format!("expected a [B, H] embedding, got shape {shape:?}")));
    }
    let (rows, hidden) = (shape[0], shape[1]);
    if rows != expected || hidden == 0 {
        return Err(malformed(format!(
            "expected {expected} embedding rows of non-zero width, got shape {shape:?}"
        )));
    }
    for row in embedded.data().chunks(hidden) {
        out.push(row.to_vec());
    }
    Ok(())
}

// ===========================================================================================
// The head's objective: ONE lambda, resolved against the UNIQUE row count
// ===========================================================================================

/// THE single resolution of the head's L2 coefficient.
///
/// One function, called by the trusted fit and by `negative.rs`'s adversary alike. A second
/// resolution site is how the adversarial control silently stops isolating multiplicity: an
/// adversary that re-resolved `SklearnEquivalentC` against its own DUPLICATED row count would
/// be minimizing a different objective, and the difference it reported would be a lambda
/// shift wearing multiplicity's name. That is why the adversary takes a plain `f64`.
///
/// The `lambda = 1 / (2 * C * n)` arithmetic is NOT restated here — it is
/// `Regularization::resolve_lambda` in `aprender-core`, where the half-constant, the
/// sum-versus-mean convention and the unpenalized intercept are contracted together. What
/// this function owns, and the only thing it owns, is the choice of `n`: the UNIQUE selected
/// row count, never a pair count and never a duplicated row count.
pub(crate) fn resolve_lambda(regularization: &HeadRegularization, unique_rows: usize) -> f64 {
    core_regularization(regularization).resolve_lambda(unique_rows)
}

/// The core head's request form for a configured knob.
fn core_regularization(regularization: &HeadRegularization) -> Regularization {
    match *regularization {
        HeadRegularization::Lambda(lambda) => Regularization::Lambda(lambda),
        HeadRegularization::SklearnEquivalentC { c } => Regularization::SklearnEquivalentC { c },
    }
}

/// The head's iteration budget on the shipped path — scikit-learn's own default.
pub(crate) const HEAD_MAX_ITER: usize = DEFAULT_MAX_ITER;

/// Stage two's product, before the typestate wraps it.
#[derive(Debug)]
pub(crate) struct FittedHead {
    /// The fitted head itself.
    pub(crate) head: MultinomialLogisticRegression,
    /// The optimizer's deterministic record.
    pub(crate) report: HeadFitReport,
    /// The L2 coefficient the fit actually minimized under.
    pub(crate) lambda: f64,
    /// The encode-once input it was fitted on.
    pub(crate) input: HeadDataset,
}

/// Encode the selection exactly once and fit the multiclass head on it.
///
/// `pub(crate)` and separate from the transition on purpose. `fit_head` cannot be called
/// twice with the same tuned encoder — it consumes the run — so an invariance claim of the
/// form "changing knob X leaves the head unchanged" is only expressible against this
/// function. Sharing one body means the property is asserted about the code the transition
/// actually runs, not about a second implementation written for the test.
///
/// # Errors
///
/// Anything [`head_dataset`] rejects, plus [`SetFitTrainError::HeadFit`] carrying the head's
/// own typed failure.
pub(crate) fn fit_on_selection(
    encoder: &mut SetFitMiniLm,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
    max_iter: usize,
) -> Result<FittedHead, SetFitTrainError> {
    let requested = config.requested();
    let input = head_dataset(encoder, dataset, selection, requested.batch_size())?;
    // `input.n()` is the encode-once row count and nothing else can be substituted for it
    // here: the budget lives on `requested`, but this function never reads it.
    let lambda = resolve_lambda(&requested.head_regularization(), input.n());
    let mut head =
        MultinomialLogisticRegression::new(input.ordered_labels().len()).with_max_iter(max_iter);
    let report = head
        .fit(
            input.embeddings(),
            input.class_indices(),
            input.ordered_labels(),
            Regularization::Lambda(lambda),
        )
        .map_err(SetFitTrainError::HeadFit)?;
    Ok(FittedHead { head, report, lambda, input })
}

/// The class index of every selected row, checked against the DECLARED label map.
fn class_indices_of(
    ordered_labels: &[String],
    selection: &Selection,
) -> Result<Vec<usize>, SetFitTrainError> {
    let classes = ordered_labels.len();
    selection
        .examples()
        .iter()
        .map(|example| {
            if example.label < classes {
                Ok(example.label)
            } else {
                Err(SetFitTrainError::SelectionLabelOutOfRange { label: example.label, classes })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use aprender::setfit::pair_cosine_mse;
    use aprender_contrastive_data::ledger::AccessLedger;
    use aprender_contrastive_data::prepared::CanonicalDeclarations;
    use aprender_contrastive_data::schema::LabeledExample;
    use aprender_contrastive_data::split::SplitDeclaration;

    use crate::train::setfit::test_fixtures as fx;

    use super::*;

    /// The fixture cell every test here builds from: 3 classes x 8 shots = 24 unique rows.
    const SHOTS: u32 = 8;
    const SELECTED: usize = 24;

    /// The fixture's dataset and its 24-row selection, with nothing tuned.
    ///
    /// Stage two's input does not depend on stage one having run — that is the point of a
    /// transition whose only inputs are the dataset and the selection — so these tests use a
    /// fresh encoder and skip the tuning loop entirely.
    fn parts() -> (SetFitMiniLm, PreparedDataset<Canonical>, Selection) {
        let dataset = fx::fixture_dataset();
        let selection = fx::fixture_selection(fx::FIXTURE_SEED, SHOTS);
        (fx::slice_encoder(fx::FIXTURE_SEED), dataset, selection)
    }

    /// A canonical dataset none of the fixture selection's identifiers name.
    fn dataset_with_foreign_ids() -> PreparedDataset<Canonical> {
        let label_names: Vec<String> =
            ["alpha", "beta", "gamma"].iter().map(|s| (*s).to_string()).collect();
        let rows = |role: &str, per_class: usize| -> Vec<LabeledExample> {
            (0..label_names.len())
                .flat_map(|label| {
                    (0..per_class).map(move |index| LabeledExample {
                        id: format!("foreign-{role}:{label}-{index}"),
                        input: format!("the {role} row {label} number {index} ."),
                        label,
                        label_text: ["alpha", "beta", "gamma"][label].to_string(),
                        source_split: role.to_string(),
                    })
                })
                .collect()
        };
        let decl = |per_class: usize| SplitDeclaration {
            expected_class_counts: vec![per_class; label_names.len()],
            label_names: label_names.clone(),
        };
        let mut ledger = AccessLedger::new();
        PreparedDataset::<Canonical>::from_labeled_rows(
            rows("train", 8),
            rows("validation", 1),
            rows("test", 1),
            &CanonicalDeclarations {
                train: decl(8),
                validation: decl(1),
                test: decl(1),
                label_names,
            },
            &mut ledger,
        )
        .expect("the foreign corpus is a valid canonical dataset")
    }

    // -------------------------------------------------------------------------------------
    // The exactly-once proof
    // -------------------------------------------------------------------------------------

    /// THE exactly-once assertion: the ledger's MULTISET is the selection's id set, once each.
    ///
    /// A duplicate-plus-omission defect keeps the row count, keeps the order of what remains
    /// and stays perfectly deterministic. Only the multiset sees it.
    #[test]
    fn head_input_ledger_is_the_selection_multiset_exactly_once() {
        let (mut encoder, dataset, selection) = parts();
        let built = head_dataset(&mut encoder, &dataset, &selection, 4)
            .expect("the fixture selection must encode");

        // Vacuity guard first: an empty ledger satisfies almost any relation over it.
        assert_eq!(selection.len(), SELECTED, "the fixture cell must draw 24 rows");
        assert_eq!(
            built.encode_ledger().len(),
            selection.len(),
            "(i) one ledger entry per selected row",
        );

        let mut ledger_sorted: Vec<&str> =
            built.encode_ledger().iter().map(String::as_str).collect();
        ledger_sorted.sort_unstable();
        let mut selected_sorted = selection.ordered_ids();
        selected_sorted.sort_unstable();
        assert_eq!(
            ledger_sorted, selected_sorted,
            "(ii) the MULTISET handed to the encoder must be the selection's id set with \
             multiplicity exactly one — a row encoded twice against another omitted keeps the \
             count and fails only here",
        );

        let ledger_order: Vec<&str> = built.encode_ledger().iter().map(String::as_str).collect();
        assert_eq!(ledger_order, selection.ordered_ids(), "(iii) ledger order is selection order");
    }

    /// One encoder invocation per pinned window — no retries, no per-row fallback.
    #[test]
    fn head_input_encode_call_count_is_one_per_pinned_window() {
        let (mut encoder, dataset, selection) = parts();
        for batch in [1_u32, 3, 4, 7, 24] {
            let built = head_dataset(&mut encoder, &dataset, &selection, batch)
                .expect("every window size must encode");
            let expected = selection.len().div_ceil(batch as usize);
            assert_eq!(
                built.encode_call_count(),
                expected,
                "batch {batch}: expected ceil({}/{batch}) = {expected} encoder invocations",
                selection.len(),
            );
            assert_eq!(built.encode_ledger().len(), selection.len(), "batch {batch}: ledger");
        }
    }

    /// A window wider than the selection is a single short batch, still exactly once.
    #[test]
    fn head_input_a_batch_larger_than_the_row_count_still_encodes_every_row_once() {
        let (mut encoder, dataset, selection) = parts();
        let built = head_dataset(&mut encoder, &dataset, &selection, 4096)
            .expect("an oversized window must encode");
        assert_eq!(built.encode_call_count(), 1, "one short batch");
        assert_eq!(built.n(), SELECTED);

        let mut sorted: Vec<&str> = built.encode_ledger().iter().map(String::as_str).collect();
        sorted.sort_unstable();
        let mut expected = selection.ordered_ids();
        expected.sort_unstable();
        assert_eq!(sorted, expected, "the multiset is unchanged by the window width");
    }

    /// Distinct texts must produce distinct embeddings.
    ///
    /// Without this, a defect that encoded ONE row twenty-four times would satisfy every
    /// count-based check above with twenty-four identical rows.
    #[test]
    fn head_input_rows_from_distinct_texts_have_distinct_embeddings() {
        let (mut encoder, dataset, selection) = parts();
        let built = head_dataset(&mut encoder, &dataset, &selection, 4)
            .expect("the fixture selection must encode");
        let rows = built.embeddings();
        assert_eq!(rows.len(), SELECTED);
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                assert_ne!(
                    rows[i], rows[j],
                    "rows {i} and {j} are bitwise identical embeddings, but the fixture's texts \
                     are all distinct — a row was encoded twice",
                );
            }
        }
    }

    // -------------------------------------------------------------------------------------
    // The mechanism, proven engaged
    // -------------------------------------------------------------------------------------

    /// `no_grad` is PROVEN engaged: no tape growth, and not one gradient disturbed.
    ///
    /// The gradients are read with `autograd::get_grad(param.id())`, not `param.grad()`.
    /// `ComputationGraph::backward` writes into the graph's own registry copies, so
    /// `param.grad()` is structurally `None` on this path and asserting it unchanged would be
    /// asserting `None == None` — theater, not evidence (the same trap `tune.rs`'s module doc
    /// records as PF-001 in a new costume).
    #[test]
    fn head_input_encode_builds_no_graph_and_touches_no_gradient() {
        let (mut encoder, dataset, selection) = parts();

        // Seed real gradients and a non-empty tape, so "unchanged" is not vacuous.
        encoder.set_training(true);
        let texts: Vec<&str> =
            dataset.train().rows().iter().take(2).map(|r| r.input.as_str()).collect();
        let za = encoder.encode_texts(&texts).expect("branch a encodes");
        let zb = encoder.encode_texts(&[texts[1], texts[0]]).expect("branch b encodes");
        pair_cosine_mse(&za, &zb, &[1.0, 0.0]).expect("the pairwise loss is defined").backward();

        let snapshot = |enc: &mut SetFitMiniLm| -> BTreeMap<String, Option<Vec<f32>>> {
            enc.trainable_parameters_mut()
                .into_iter()
                .map(|(name, t)| (name, autograd::get_grad(t.id()).map(|g| g.data().to_vec())))
                .collect()
        };
        let before = snapshot(&mut encoder);
        assert!(
            before.values().any(Option::is_some),
            "the seeding backward must leave at least one real gradient, or the assertion \
             below compares None against None",
        );
        let tape_before = autograd::graph_tape_len();
        assert!(tape_before > 0, "the seeding forward must leave a non-empty tape");

        let built = head_dataset(&mut encoder, &dataset, &selection, 4)
            .expect("the fixture selection must encode");

        assert_eq!(autograd::graph_tape_len(), tape_before, "the encode recorded operations");
        assert_eq!(
            built.witness().tape_before,
            built.witness().tape_after,
            "the in-band witness must agree: the tape did not grow across the encode",
        );
        assert_eq!(snapshot(&mut encoder), before, "the encode disturbed a gradient");
    }

    /// The stored rows come off a DETACHED tensor: nothing graph-connected survives.
    #[test]
    fn head_input_detached_embeddings_do_not_require_grad() {
        let (mut encoder, dataset, selection) = parts();
        let built = head_dataset(&mut encoder, &dataset, &selection, 4)
            .expect("the fixture selection must encode");
        assert!(
            !built.witness().requires_grad_observed,
            "every embedding tensor the encode produced must report requires_grad_enabled() \
             == false after detach",
        );
    }

    /// Eval mode is forced by the transition, and observed WHILE the encode runs.
    #[test]
    fn head_input_encodes_in_eval_mode_even_from_a_training_encoder() {
        let (mut training_encoder, dataset, selection) = parts();
        training_encoder.set_training(true);
        let from_training = head_dataset(&mut training_encoder, &dataset, &selection, 4)
            .expect("a training-mode encoder must still produce the head's input");

        assert!(
            !from_training.witness().training_observed,
            "the encoder reported training() == true inside an encode window",
        );
        assert!(!training_encoder.training(), "the encoder is left in eval mode");

        // A FRESH encoder, which is the property this comparison rests on — `parts()`
        // would also rebuild a dataset and a selection only to discard both.
        let mut eval_encoder = fx::slice_encoder(fx::FIXTURE_SEED);
        let from_eval = head_dataset(&mut eval_encoder, &dataset, &selection, 4)
            .expect("an eval-mode encoder must produce the head's input");
        assert_eq!(
            from_training.embeddings(),
            from_eval.embeddings(),
            "the two must agree BITWISE; the slice's dropout probability is 0.1, so a \
             training-mode encode would differ",
        );
    }

    // -------------------------------------------------------------------------------------
    // Determinism, labels, and the typed refusals
    // -------------------------------------------------------------------------------------

    /// Two builds are bitwise identical — embeddings AND ledger.
    #[test]
    fn head_input_two_builds_are_bitwise_identical() {
        let (mut first, dataset, selection) = parts();
        let mut second = fx::slice_encoder(fx::FIXTURE_SEED);
        let a = head_dataset(&mut first, &dataset, &selection, 4).expect("first build");
        let b = head_dataset(&mut second, &dataset, &selection, 4).expect("second build");
        assert_eq!(a.embeddings(), b.embeddings(), "pinned windows must be bitwise reproducible");
        assert_eq!(a.encode_ledger(), b.encode_ledger());
        assert_eq!(a.encode_call_count(), b.encode_call_count());
    }

    /// Label order comes from the DECLARED map, pinned against the fixture's own labels.
    #[test]
    fn head_input_label_order_comes_from_the_declared_label_map() {
        let (mut encoder, dataset, selection) = parts();
        let built = head_dataset(&mut encoder, &dataset, &selection, 4).expect("build");
        assert_eq!(built.ordered_labels(), ["alpha", "beta", "gamma"]);
        assert_eq!(
            built.ordered_labels(),
            dataset.label_names(),
            "the order is the dataset's declared map, never an incidental class-iteration order",
        );
        for (row, &class) in built.class_indices().iter().enumerate() {
            assert!(class < built.ordered_labels().len(), "row {row} names an undeclared class");
        }
    }

    /// A selected id that names no row of the dataset is a typed refusal, never a silent skip.
    #[test]
    fn head_input_a_selected_row_absent_from_the_dataset_is_a_typed_error() {
        let (mut encoder, _, selection) = parts();
        let foreign = dataset_with_foreign_ids();
        match head_dataset(&mut encoder, &foreign, &selection, 4) {
            Err(SetFitTrainError::SelectionRowMissing { id }) => {
                assert!(
                    selection.ordered_ids().contains(&id.as_str()),
                    "the refusal must name a SELECTED id, got `{id}`",
                );
            }
            other => panic!(
                "a selection whose ids name no row must fail closed; a silent skip is exactly \
                 the omission the ledger exists to catch. Got {other:?}",
            ),
        }
    }

    /// The witness GATE fires on each of the three defects, and only on those.
    ///
    /// The shipped encode cannot produce any of them, so the gate would otherwise only ever
    /// be observed passing — which is the same "never seen red" problem the in-band negative
    /// discipline exists for. The case table is run both ways.
    #[test]
    fn head_input_the_encode_witness_refuses_a_non_isolated_encode() {
        let clean = EncodeWitness {
            training_observed: false,
            requires_grad_observed: false,
            tape_before: 7,
            tape_after: 7,
        };
        assert!(clean.require_isolated().is_ok(), "an isolated encode must be accepted");

        let cases = [
            ("training mode", EncodeWitness { training_observed: true, ..clean.clone() }),
            ("live gradients", EncodeWitness { requires_grad_observed: true, ..clean.clone() }),
            ("tape growth", EncodeWitness { tape_after: 9, ..clean.clone() }),
        ];
        for (name, witness) in cases {
            match witness.require_isolated() {
                Err(SetFitTrainError::HeadEncodeNotIsolated { tape_growth, .. }) => {
                    if name == "tape growth" {
                        assert_eq!(tape_growth, 2, "the refusal must report the growth");
                    }
                }
                other => panic!("`{name}` must be refused, got {other:?}"),
            }
        }
    }

    /// A zero window size is a typed refusal, not a `chunks(0)` panic.
    #[test]
    fn head_input_a_zero_batch_size_is_a_typed_error() {
        let (mut encoder, dataset, selection) = parts();
        assert!(matches!(
            head_dataset(&mut encoder, &dataset, &selection, 0),
            Err(SetFitTrainError::HeadEncodeBatchSizeZero),
        ));
    }

    /// This module's own source, with its test module removed.
    ///
    /// The cut is what makes the scan below evidence. A guard that reads the file it lives in
    /// finds every needle IN ITSELF: `text.contains("no_grad")` is satisfied by the assertion
    /// that spells it, and would stay green if the mechanism were deleted from the code. The
    /// header is assembled at runtime for the same reason.
    fn module_source() -> String {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/train/setfit/head_input.rs");
        std::fs::read_to_string(&path).expect("this module is readable")
    }

    /// The shipped half of `whole`, cut at the test module's header.
    fn shipped_source(whole: &str) -> &str {
        let header = format!("\nmod {} {{", "tests");
        let cut = whole.find(&header).expect("this module ends with its test module");
        &whole[..cut]
    }

    /// The module names no multiplicity-shaped input anywhere, and the mechanism is present.
    #[test]
    fn head_input_names_no_multiplicity_shaped_parameter() {
        // The pair-type scan covers the WHOLE file, tests included: there is no reason for
        // this module to mention the type at all. Assembled so it cannot match itself.
        let whole = module_source();
        let needle = format!("{}{}", "P", "air");
        assert!(
            !whole.contains(&needle),
            "head_input.rs mentions the pair type at all; stage two's input must be \
             expressible only from the selection",
        );

        let shipped = shipped_source(&whole);
        for mechanism in ["no_grad", "detach", "label_names", "set_training(false)"] {
            assert!(
                shipped.contains(mechanism),
                "`{mechanism}` must be in the SHIPPED code path, not only in a test that \
                 mentions it",
            );
        }
        // And exactly one lambda resolution lives here (03-07 task 2 shares it with the
        // adversary; a second site is how the control stops isolating multiplicity).
        let resolver = format!("{}{}", "fn resolve_", "lambda");
        assert_eq!(shipped.matches(&resolver).count(), 1);
    }
}
