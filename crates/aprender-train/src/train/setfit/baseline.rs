//! `FrozenProbeRun` — the SAFE-03 baseline that structurally cannot claim SetFit.
//!
//! # Why a separate TYPE and not a flag
//!
//! A frozen-encoder linear probe and a SetFit run produce the same SHAPE of artifact: an
//! encoder, a fitted multiclass head, and a report. The difference is that one tuned the
//! encoder and the other did not — and that difference is exactly what a benchmark table is
//! asking about when it puts the two side by side. A boolean field, or a `kind` string on a
//! shared type, makes mislabelling a one-character edit that no gate would catch.
//!
//! So this is a DIFFERENT TYPE. It is not a [`super::SetFitRun`] state, there is no `From` or
//! `Into` between them, and its report's `kind` is the literal `"frozen_linear_probe"`. There
//! is no value of this type that can be turned into a `SetFitRun<_>` by any public path, which
//! is what makes SAFE-03 structural rather than a naming convention.
//!
//! # What it does
//!
//! Encodes the selected rows ONCE, in eval mode, inside [`aprender::autograd::no_grad`], with
//! no tuning step of any kind, and fits [`MultinomialLogisticRegression`] on those frozen
//! embeddings. It binds `contracts/linear-probe-classifier-v1.yaml`, whose invariants are
//! precisely "frozen encoder weights do not receive gradients" and "only W and b are updated".
//!
//! Phase 5 runs its baselines through this door.

use aprender::autograd;
use aprender::classification::multinomial::{MultinomialLogisticRegression, Regularization};
use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;

use super::SetFitTrainError;

/// The literal this baseline reports itself as. NEVER "setfit".
pub const FROZEN_PROBE_KIND: &str = "frozen_linear_probe";

/// The encode window this probe uses.
///
/// # KNOWN LIMIT, recorded rather than left as a bare `8`
///
/// The SetFit path encodes its head input in `config.batch_size()`-wide windows (the reference
/// default is 16), and this type carries no configuration to read that from. The two paths
/// therefore pad to different per-window sequence lengths, so the embeddings a benchmark puts
/// side by side are not guaranteed to agree bitwise even for identical text in identical
/// order — which is weaker than the comparability this module's header claims. Closing it
/// needs the window size threaded in from the run's configuration; until then the number is
/// named, so a reader can see it is a decision rather than an accident.
const PROBE_ENCODE_WINDOW: usize = 8;

/// What a frozen-probe run reports.
///
/// Its own type, so it cannot be confused with a SetFit run's evidence at a call site or in a
/// serialized benchmark row.
#[derive(Debug, Clone, PartialEq)]
pub struct FrozenProbeReport {
    /// Always [`FROZEN_PROBE_KIND`]. Present so a serialized row is self-describing.
    kind: &'static str,
    /// Rows encoded and fitted on.
    row_count: usize,
    /// Embedding width.
    hidden: usize,
    /// The ordered label names the head was fitted against.
    ordered_labels: Vec<String>,
    /// Autograd tape length before and after the encode.
    ///
    /// Recorded as a PAIR rather than asserted internally, so a test can check the DELTA. The
    /// absolute value is not this type's to claim: the model loader leaves entries on the tape
    /// at load time (D-ITEM-06), so an absolute assertion here would be a test of the loader
    /// wearing this type's name.
    tape_len_around_encode: (usize, usize),
}

impl FrozenProbeReport {
    /// The run kind. Always [`FROZEN_PROBE_KIND`].
    #[must_use]
    pub fn kind(&self) -> &'static str {
        self.kind
    }

    /// Rows encoded and fitted on.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Embedding width.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The ordered label names.
    #[must_use]
    pub fn ordered_labels(&self) -> &[String] {
        &self.ordered_labels
    }

    /// `(before, after)` autograd tape lengths around the encode.
    #[must_use]
    pub fn tape_len_around_encode(&self) -> (usize, usize) {
        self.tape_len_around_encode
    }
}

/// A frozen-encoder linear-probe run.
///
/// Deliberately NOT convertible to [`super::SetFitRun`] in either direction.
#[derive(Debug)]
pub struct FrozenProbeRun {
    encoder: SetFitMiniLm,
    dataset: PreparedDataset<Canonical>,
    selection: Selection,
    regularization: Regularization,
}

impl FrozenProbeRun {
    /// Build a probe run over a selection.
    #[must_use]
    pub fn new(
        encoder: SetFitMiniLm,
        dataset: PreparedDataset<Canonical>,
        selection: Selection,
        regularization: Regularization,
    ) -> Self {
        Self { encoder, dataset, selection, regularization }
    }

    /// Encode the selected rows with a FROZEN encoder and fit the head on the result.
    ///
    /// The encoder is put in eval mode and the encode runs inside `no_grad`, so no graph is
    /// built and no gradient can reach an encoder parameter — the invariant
    /// `linear-probe-classifier-v1` states as "frozen encoder weights do not receive
    /// gradients". There is no optimizer, no backward call and no step anywhere in this
    /// function; the ONLY parameters that change are the head's `W` and `b`.
    ///
    /// # Errors
    ///
    /// [`SetFitTrainError::Encoder`] if the encoder rejects a row, and
    /// [`SetFitTrainError::HeadFit`] carrying the head's OWN typed failure if the fit is
    /// rejected. Not `Evidence { reason: String }`: TRN-04's whole point is that a caller can
    /// tell "your data was bad" from "the optimizer ran out of budget" without matching on
    /// message text, and a baseline that renders the head's error to a string is a baseline
    /// whose failures cannot be compared with the SetFit path's.
    #[cfg_attr(
        feature = "setfit",
        provable_contracts_macros::contract(
            "linear-probe-classifier-v1",
            equation = "linear_probe"
        )
    )]
    pub fn fit(
        mut self,
    ) -> Result<(MultinomialLogisticRegression, FrozenProbeReport), SetFitTrainError> {
        // The SAME text resolution the tuning loop uses, so a probe and a SetFit run encode
        // identical strings in identical order — which is what makes the two comparable at all.
        let owned = super::tune::selection_texts(&self.dataset, &self.selection)?;
        let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
        let class_indices: Vec<usize> = self.selection.examples().iter().map(|e| e.label).collect();

        // Eval mode + no_grad. The tape lengths are recorded so a test can assert the DELTA.
        self.encoder.set_training(false);
        let before = autograd::graph_tape_len();
        let mut features: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let encoded = autograd::no_grad(|| -> Result<(), SetFitTrainError> {
            for window in texts.chunks(PROBE_ENCODE_WINDOW) {
                let embedded = self
                    .encoder
                    .encode_texts(window)
                    .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;
                // The SAME checked split the head's encode-once input uses: a non-2-D return
                // or a `B` that disagrees with the window is a typed refusal, not a panic and
                // not a silently short row.
                super::head_input::push_rows(&mut features, &embedded, window.len())?;
            }
            Ok(())
        });
        encoded?;
        let after = autograd::graph_tape_len();

        let ordered_labels: Vec<String> = self.dataset.label_names().to_vec();
        let hidden = features.first().map_or(0, Vec::len);
        let mut head = MultinomialLogisticRegression::new(ordered_labels.len());
        head.fit(&features, &class_indices, &ordered_labels, self.regularization)
            .map_err(SetFitTrainError::HeadFit)?;

        let report = FrozenProbeReport {
            kind: FROZEN_PROBE_KIND,
            row_count: features.len(),
            hidden,
            ordered_labels,
            tape_len_around_encode: (before, after),
        };
        Ok((head, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::setfit::test_fixtures as fx;

    fn probe() -> FrozenProbeRun {
        let (encoder, dataset, selection, config) =
            fx::prepared_run(fx::default_variant(), None).into_parts();
        let _ = config;
        FrozenProbeRun::new(
            encoder,
            dataset,
            selection,
            Regularization::SklearnEquivalentC { c: 1.0 },
        )
    }

    /// The probe produces a WORKING classifier: finite probabilities that sum to one.
    #[test]
    fn baseline_frozen_probe_fits_a_working_classifier() {
        let (head, report) = probe().fit().expect("the frozen probe must fit");
        assert!(report.row_count() > 0, "non-vacuity: rows were actually encoded");
        assert!(report.hidden() > 0);
        assert_eq!(report.ordered_labels().len(), 3, "the fixture declares three classes");

        let features = vec![vec![0.0_f32; report.hidden()]];
        let probs = head.predict_proba(&features).expect("predict_proba");
        let row = probs.first().expect("one row in, one row out");
        assert_eq!(row.len(), 3);
        let total: f64 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "probabilities must sum to 1, got {total}");
        assert!(row.iter().all(|p| p.is_finite() && *p > 0.0), "and be finite and positive");
    }

    /// The report NEVER says "setfit".
    #[test]
    fn baseline_report_kind_is_frozen_linear_probe() {
        let (_, report) = probe().fit().expect("fit");
        assert_eq!(report.kind(), "frozen_linear_probe");
        assert_eq!(report.kind(), FROZEN_PROBE_KIND);
        assert!(!report.kind().contains("setfit"), "a baseline must never claim SetFit");
    }

    /// The frozen probe BUILDS NO GRAPH — the property its name claims.
    ///
    /// Asserted as a DELTA, with a non-vacuity check, because the absolute tape length is
    /// owned by the model loader (D-ITEM-06) and not by this function.
    #[test]
    fn baseline_encode_builds_no_graph() {
        let (_, report) = probe().fit().expect("fit");
        let (before, after) = report.tape_len_around_encode();
        assert_eq!(
            before, after,
            "the frozen probe's encode must record nothing on the autograd tape; it grew from \
             {before} to {after}, so a gradient path exists where the type claims none",
        );
        // Non-vacuity: if the loader is ever fixed so `before` is 0, this test still compares
        // something real only because the encode itself is asserted to have happened.
        assert!(report.row_count() > 0);
    }

    /// SOURCE ASSERTION: this module never constructs a `SetFitRun`.
    ///
    /// The type-level guarantee is that no `From`/`Into` exists, but a future edit could add
    /// one here without any other test noticing. Reading the source is the cheapest way to
    /// keep the structural claim honest.
    #[test]
    fn baseline_never_constructs_a_setfit_run() {
        let source = include_str!("baseline.rs");
        let code_only: String = source
            .lines()
            .filter(|line| {
                let t = line.trim_start();
                !t.starts_with("//") && !t.starts_with("///") && !t.starts_with("//!")
            })
            .collect::<Vec<_>>()
            .join("\n");
        // The needle is ASSEMBLED rather than written, because a literal here appears in
        // this file and the search would find its own assertion. That self-reference has now
        // bitten four separate provenance greps in this phase; building the token from parts
        // is the fix that does not depend on anyone remembering the trap.
        let needle = format!("{}{}", "SetFitRun", "<");
        assert_eq!(
            code_only.matches(needle.as_str()).count(),
            0,
            "baseline.rs must not construct or name a {needle} outside comments",
        );
        // Non-vacuity: the filter kept real code rather than emptying the buffer.
        assert!(code_only.contains("FrozenProbeRun"), "the code filter must retain the source");
        assert!(code_only.contains("frozen_linear_probe"));
    }
}
