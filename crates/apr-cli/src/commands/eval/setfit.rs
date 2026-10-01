//! `apr eval` for a `setfit-apr-v1` artifact — the durable selection-lock workflow (D-16, TRN-07).
//!
//! # The workflow is TWO invocations with a FILE between them
//!
//! Review finding B4: the previous design had validation and test evaluation in one
//! conceptual flow — the test command would create the lock, mint the token and grant access.
//! That is a contradiction. A separate `apr eval --split test` process has no lock input, and
//! creating the lock inside the test command defeats the entire requirement, which is that the
//! selection be COMMITTED BEFORE test access is taken.
//!
//! So the lock is a durable artifact:
//!
//! ```text
//! apr eval M.apr --task classify --data D --selection S --split validation --lock-out L
//! apr eval M.apr --task classify --data D --selection S --split test       --selection-lock L
//! ```
//!
//! The first writes `L`. The second reads it, reconstructs it through
//! `SelectionLock::from_canonical_bytes`, and lets the Phase 3 doors refuse an artifact or a
//! dataset that does not match. Absent, stale and mismatched locks are three different typed
//! refusals.
//!
//! # This adapter adds NO gating of its own, and bypasses NONE
//!
//! Every check that decides whether canonical test rows may be read lives in `aprender-train`:
//! `reload_verified_run_from_apr`'s three provenance identifiers,
//! `SelectionLock::verify_integrity`, `mint_test_token`'s artifact comparison, and
//! `CanonicalTestAccess::grant`'s artifact AND dataset comparisons. This file's job is to
//! surface those refusals with the flag names the library cannot know. It defines no gating
//! type, and there is no path here to the test split that does not pass through `grant`.
//!
//! # Bounds
//!
//! Artifact bytes come through `setfit_io::read_setfit_apr_file_bounded` — the ONE bounded
//! artifact door. The lock file is small and is refused above `MAX_SELECTION_LOCK_BYTES` from
//! its stat'd length, BEFORE it is read, on the same reasoning: a bound applied after the
//! allocation is not a bound on the work an attacker can request.

use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;
use colored::Colorize;
use entrenar::train::setfit::apr_evaluate::evaluate_validation_from_artifact;
use entrenar::train::setfit::apr_reload::{reload_verified_run_from_apr, ReloadedSetFitCredential};
use entrenar::train::setfit::evaluate::{ValidationEvaluation, ValidationMetricKind};
use entrenar::train::setfit::lock::{
    create_selection_lock, CanonicalTestAccess, SelectionCandidate, SelectionLock, SelectionRule,
    MAX_SELECTION_LOCK_BYTES,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::commands::data_contrastive;
use crate::error::{CliError, Result};
use crate::output;
use crate::setfit_io;

/// Which split this invocation measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Split {
    /// The canonical validation split. May COMMIT a selection by writing a lock.
    Validation,
    /// The canonical test split. Requires a lock committed by a PRIOR invocation.
    Test,
}

impl std::str::FromStr for Split {
    type Err = CliError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "validation" => Ok(Self::Validation),
            "test" => Ok(Self::Test),
            other => Err(CliError::ValidationFailed(format!(
                "--split must be `validation` or `test`, got `{other}`. There is no third \
                 split a SetFit evaluation may read: the train split is what the model was \
                 fitted on, and reading it as an evaluation would report memorisation as skill."
            ))),
        }
    }
}

/// Everything the two flag sets carry, resolved.
#[derive(Debug, Clone)]
pub(crate) struct SetFitEvalArgs<'a> {
    /// The `setfit-apr-v1` artifact under evaluation.
    pub(crate) artifact: &'a Path,
    /// The Phase 2 prepared dataset directory (`--data`).
    pub(crate) data: Option<&'a Path>,
    /// The Phase 2 selection manifest (`--selection`).
    pub(crate) selection: Option<&'a Path>,
    /// Which split (`--split`).
    pub(crate) split: Split,
    /// Where to write the durable lock (`--lock-out`, validation only).
    pub(crate) lock_out: Option<&'a Path>,
    /// The lock a prior validation run committed (`--selection-lock`, test only).
    pub(crate) selection_lock: Option<&'a Path>,
    /// Additional artifacts to consider (`--candidate`, validation only).
    pub(crate) candidates: &'a [PathBuf],
    /// Replace an existing `--lock-out`.
    pub(crate) force: bool,
    /// The global `--json`.
    pub(crate) json: bool,
}

/// The machine-readable row Phase 5 consumes (EVAL-03 shape).
#[derive(Debug, Serialize)]
struct EvalRow {
    /// `validation` or `test`.
    split: &'static str,
    /// The metric measured.
    metric: String,
    /// Its value.
    value: f64,
    /// Its IEEE-754 bits, because the lock hashes bits and a decimal is a rendering.
    value_bits: u64,
    /// How many rows it was computed over.
    n_rows: usize,
    /// The artifact this measurement is about.
    artifact_sha256: String,
    /// The whole dataset's fingerprint.
    dataset_fingerprint: String,
    /// The validation split's own fingerprint.
    validation_split_fingerprint: String,
    /// The labels the head indexes by, in row order.
    ordered_labels: Vec<String>,
    /// The evidence table hash carried by the artifact.
    evidence_table_hash: Option<String>,
    /// The selection's root seed, from the artifact's provenance.
    selection_root_seed: Option<u64>,
    /// How each candidate's `config_hash` was derived, so Phase 5 can reproduce it.
    config_hash_derivation: &'static str,
    /// The candidates considered, in the order they were supplied.
    candidates: Vec<CandidateRow>,
    /// The lock this run wrote or consumed.
    lock: Option<LockRow>,
    /// Anything the operator must know that the numbers do not say.
    notes: Vec<String>,
}

/// One candidate, as recorded.
#[derive(Debug, Serialize)]
struct CandidateRow {
    /// The candidate artifact's SHA-256.
    artifact_sha256: String,
    /// The hex SHA-256 over its canonical `requested_config` sub-document.
    config_hash: String,
    /// Its measured value.
    value: f64,
    /// Its measured value's bits.
    value_bits: u64,
}

/// The lock, as written or consumed.
#[derive(Debug, Serialize)]
struct LockRow {
    /// Where it lives.
    path: String,
    /// Its recorded hash.
    lock_hash: String,
    /// The artifact it names as chosen.
    chosen_artifact_sha256: String,
    /// The rule that chose it.
    rule: String,
    /// `written` or `consumed`.
    role: &'static str,
}

/// How `config_hash` is derived, stated in the report so Phase 5 can reproduce it.
const CONFIG_HASH_DERIVATION: &str =
    "sha256(canonical JSON bytes of the artifact document's `requested_config` sub-document)";

/// The metric this command measures.
///
/// Fixed rather than a flag, deliberately: `SelectionLock::from_candidates` refuses a candidate
/// set whose metric kinds disagree, so a per-invocation choice would let an operator build a
/// lock that cannot be created and discover it only at the end. Accuracy is the contract's
/// default; widening this is a deliberate change with a flag AND a candidate-set rule, not a
/// convenience.
const EVAL_METRIC: ValidationMetricKind = ValidationMetricKind::Accuracy;

/// The selection rule this command commits under.
const EVAL_RULE: SelectionRule = SelectionRule::MaxMetricLowestIndexTieBreak;

/// Run a SetFit evaluation.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a missing or contradictory flag set and for every typed
/// library refusal (a selection or dataset that is not the artifact's, a stale lock, a token
/// whose dataset does not match); [`CliError::InvalidFormat`] for an over-cap lock file;
/// [`CliError::ModelLoadFailed`] for an artifact the load ladder rejects.
pub(crate) fn run(args: &SetFitEvalArgs<'_>) -> Result<()> {
    // (1) THE FLAGS AND THE OUTPUT PATH FIRST, before a byte of the dataset or the artifact is
    //     read. A run that spends a minute loading a corpus and then says "--selection-lock is
    //     required" has told the operator something it knew before it started — and the same is
    //     true of the destination it was asked to write. That second half was MISSING (WR-10):
    //     the no-clobber check lived only at the write, so the whole multi-candidate sweep ran
    //     first. `setfit_train.rs`'s module header states this discipline as four checks and
    //     says refusing the output path is the one that stops a long run from ending in "I will
    //     not overwrite that file"; this is that check, in the command that lacked it.
    let data = args.data.ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <DIR> is required to evaluate a setfit-apr-v1 artifact: the canonical \
             splits live in the Phase 2 prepared dataset directory, and the artifact records \
             which corpus it was trained on but does not carry the rows."
                .to_string(),
        )
    })?;
    let selection = args.selection.ok_or_else(|| {
        CliError::ValidationFailed(
            "--selection <FILE> is required: the artifact records the selection's hashes, and \
             the reload door compares them against the selection-manifest.json `apr data \
             select` wrote. Without it there is nothing to compare."
                .to_string(),
        )
    })?;
    check_split_flags(args)?;
    if let Some(destination) = args.lock_out {
        refuse_existing_lock(destination, args.force)?;
    }

    // (2) THE PHASE 2 INPUTS, through the doors `data_contrastive` already owns.
    let mut ledger = AccessLedger::new();
    let dataset = data_contrastive::read_attested_canonical(data, &mut ledger)?;
    let manifest = data_contrastive::read_selection_manifest(selection)?;
    let replayed = Selection::replay(&manifest, &dataset, &mut ledger).map_err(|error| {
        CliError::ValidationFailed(format!(
            "--selection {} does not replay against --data {}: {error}",
            selection.display(),
            data.display()
        ))
    })?;

    // (3) THE TRUSTED DOOR. Its typed refusals — a selection or dataset that is not this
    //     artifact's — surface as ValidationFailed naming the flags. This adapter adds no
    //     gating of its own.
    let credential = reload_artifact(args.artifact, &dataset, &replayed)?;

    match args.split {
        Split::Validation => run_validation(args, &credential, &dataset, &replayed),
        Split::Test => run_test(args, &credential, &dataset),
    }
}

/// Refuse a flag set that belongs to the other split.
///
/// A flag silently ignored is worse than a flag refused: an operator who passes `--lock-out` to
/// a test run and sees a green report has every reason to believe a lock was written.
fn check_split_flags(args: &SetFitEvalArgs<'_>) -> Result<()> {
    match args.split {
        Split::Validation => {
            if args.selection_lock.is_some() {
                return Err(CliError::ValidationFailed(
                    "--selection-lock belongs to `--split test`: a validation run COMMITS a \
                     selection, it does not consume one. Use --lock-out <FILE> to write the \
                     lock this run's decision produces."
                        .to_string(),
                ));
            }
            Ok(())
        }
        Split::Test => {
            if args.lock_out.is_some() {
                return Err(CliError::ValidationFailed(
                    "--lock-out belongs to `--split validation`. A test run that could write \
                     its own lock would defeat the requirement the lock exists for: that the \
                     selection be committed BEFORE canonical test access is taken."
                        .to_string(),
                ));
            }
            if !args.candidates.is_empty() {
                return Err(CliError::ValidationFailed(
                    "--candidate belongs to `--split validation`. The candidate set is the \
                     selection DECISION, and it was fixed when the lock was written."
                        .to_string(),
                ));
            }
            if args.selection_lock.is_none() {
                return Err(CliError::ValidationFailed(
                    "--selection-lock <FILE> is REQUIRED for `--split test`. Canonical test \
                     rows are reachable only through a selection lock that was committed \
                     BEFORE this command ran. Write one first:\n  apr eval <ARTIFACT> --task \
                     classify --data <DIR> --selection <FILE> --split validation --lock-out \
                     <FILE>\nthen pass that file here as --selection-lock."
                        .to_string(),
                ));
            }
            Ok(())
        }
    }
}

/// Read one artifact and reload it against the supplied inputs.
fn reload_artifact(
    path: &Path,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
) -> Result<ReloadedSetFitCredential> {
    let bytes = setfit_io::read_setfit_apr_file_bounded(path)?;
    reload_verified_run_from_apr(&bytes, dataset, selection).map_err(|error| {
        CliError::ValidationFailed(format!(
            "{}: {error}\nThe artifact, --data and --selection must be the three the run was \
             produced from. Check that --data names the same prepared dataset directory and \
             --selection the same selection-manifest.json that `apr setfit train` consumed.",
            path.display()
        ))
    })
}

/// `--split validation`: measure, optionally COMMIT the decision as a durable lock.
fn run_validation(
    args: &SetFitEvalArgs<'_>,
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
) -> Result<()> {
    // The evaluated artifact is always candidate ZERO, so the run that commits the lock is
    // always in the set it commits over — `create_selection_lock` refuses a lock whose
    // candidates do not contain the creating model, and discovering that at the end of a
    // multi-artifact sweep would be a wasted sweep.
    let mut candidate_rows = Vec::with_capacity(args.candidates.len() + 1);
    let mut candidates = Vec::with_capacity(args.candidates.len() + 1);

    let own = evaluate_one(credential, dataset)?;
    let own_config_hash = config_hash_of(credential);
    candidate_rows.push(candidate_row(&own_config_hash, &own));
    candidates.push(SelectionCandidate::from_evaluation(
        &own_config_hash,
        own.clone(),
    ));

    for path in args.candidates {
        // Each candidate goes through the SAME door: the full load ladder plus the three
        // provenance identifiers. A candidate that was not trained on this corpus and this
        // selection is refused here rather than silently compared against ones that were.
        let other = reload_artifact(path, dataset, selection)?;
        let evaluation = evaluate_one(&other, dataset)?;
        let config_hash = config_hash_of(&other);
        candidate_rows.push(candidate_row(&config_hash, &evaluation));
        candidates.push(SelectionCandidate::from_evaluation(
            &config_hash,
            evaluation,
        ));
    }

    let mut notes = Vec::new();
    let lock_row = match args.lock_out {
        Some(destination) => {
            let lock =
                create_selection_lock(credential, candidates, EVAL_RULE).map_err(|error| {
                    CliError::ValidationFailed(format!(
                        "the selection lock could not be committed: {error}"
                    ))
                })?;
            write_lock(destination, &lock, args.force)?;
            Some(LockRow {
                path: destination.display().to_string(),
                lock_hash: lock.lock_hash().to_string(),
                chosen_artifact_sha256: lock.chosen_artifact_hash().to_string(),
                rule: lock.rule().to_string(),
                role: "written",
            })
        }
        None => {
            notes.push(
                "no --lock-out was given, so this run measured but COMMITTED NOTHING. A \
                 validation run that did not commit a selection cannot later unlock canonical \
                 test access: `apr eval --split test` requires a lock file written by a prior \
                 validation run."
                    .to_string(),
            );
            None
        }
    };

    let row = EvalRow {
        split: "validation",
        metric: EVAL_METRIC.tag().to_string(),
        value: own.value(),
        value_bits: own.value_bits(),
        n_rows: own.n_rows(),
        artifact_sha256: own.artifact_hash().to_string(),
        dataset_fingerprint: own.dataset_fingerprint().to_string(),
        validation_split_fingerprint: own.validation_split_fingerprint().to_string(),
        ordered_labels: credential.model().ordered_labels().to_vec(),
        evidence_table_hash: evidence_table_hash(credential),
        selection_root_seed: selection_root_seed(credential),
        config_hash_derivation: CONFIG_HASH_DERIVATION,
        candidates: candidate_rows,
        lock: lock_row,
        notes,
    };
    report(args.json, &row)
}

/// `--split test`: consume a PRIOR lock, mint, grant, measure.
fn run_test(
    args: &SetFitEvalArgs<'_>,
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
) -> Result<()> {
    let lock_path = args
        .selection_lock
        .ok_or_else(|| CliError::ValidationFailed("--selection-lock is required".to_string()))?;
    let lock = read_lock(lock_path)?;

    // THE THREE DOORS, in order, each refusing on its own terms. Nothing here re-checks what
    // they check, and nothing here can reach `dataset.test()` around them: the split comes out
    // of the grant.
    let token = lock.mint_test_token(credential).map_err(|error| {
        CliError::ValidationFailed(format!(
            "the selection lock {} does not admit this artifact: {error}\nA lock names ONE \
             chosen artifact. If you continued tuning after committing the selection, the lock \
             is stale — re-run `--split validation --lock-out` to commit the new decision \
             before taking test access.",
            lock_path.display()
        ))
    })?;
    let grant = CanonicalTestAccess::grant(token, credential, dataset).map_err(|error| {
        CliError::ValidationFailed(format!(
            "canonical test access was refused: {error}\nThe --data directory must be the \
             corpus the lock was taken over."
        ))
    })?;

    // THE LABEL MAP — the fourth door, and the one the three above deliberately do not cover.
    // `mint_test_token` compares the artifact hash; `grant` compares artifact + dataset
    // fingerprint; the reload gate compares corpus/ledger/draw. NONE of them compares the two
    // LABEL ORDERINGS. `evaluate_test` scores `position(result.label())` — an index into the
    // ARTIFACT's head — against `row.label`, an index into the DATASET. If the two orderings
    // disagree, every comparison is between unrelated indices and the command reports a
    // confidently wrong accuracy instead of an error.
    //
    // The validation evaluator already refuses exactly this by name
    // (`aprender-train/src/train/setfit/apr_evaluate.rs`, `LabelMapMismatch`); the test path
    // shipped without it. Read the labels OFF THE REBUILT HEAD, not off the document's copy,
    // which is what a classification actually indexes into.
    let artifact_labels: Vec<String> = credential.model().ordered_labels().to_vec();
    let dataset_labels: Vec<String> = dataset.label_names().to_vec();
    if artifact_labels != dataset_labels {
        return Err(CliError::ValidationFailed(format!(
            "label map mismatch: the artifact's labels are {artifact_labels:?} but the dataset's \
             are {dataset_labels:?}.\nIndex i of one is not index i of the other, so scoring them \
             against each other would produce a confidently wrong number rather than an error. \
             The --data directory must be the corpus this artifact was trained over."
        )));
    }

    let evaluation = evaluate_test(credential, grant.test())?;

    let row = EvalRow {
        split: "test",
        metric: EVAL_METRIC.tag().to_string(),
        value: evaluation.value,
        value_bits: evaluation.value.to_bits(),
        n_rows: evaluation.n_rows,
        artifact_sha256: grant.artifact_hash().to_string(),
        dataset_fingerprint: dataset.validation_witness().dataset_fingerprint_hex(),
        validation_split_fingerprint: dataset.validation_witness().fingerprint_hex(),
        ordered_labels: credential.model().ordered_labels().to_vec(),
        evidence_table_hash: evidence_table_hash(credential),
        selection_root_seed: selection_root_seed(credential),
        config_hash_derivation: CONFIG_HASH_DERIVATION,
        candidates: Vec::new(),
        lock: Some(LockRow {
            path: lock_path.display().to_string(),
            lock_hash: grant.lock_hash().to_string(),
            chosen_artifact_sha256: lock.chosen_artifact_hash().to_string(),
            rule: lock.rule().to_string(),
            role: "consumed",
        }),
        notes: vec![
            "canonical TEST measurement, taken under a selection lock committed by a prior \
             invocation. This number is reportable exactly once per committed selection: \
             measuring it again after further tuning requires a NEW validation run and a NEW \
             lock."
                .to_string(),
        ],
    };
    report(args.json, &row)
}

/// One artifact's validation measurement, through the library's fresh-process evaluator.
fn evaluate_one(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
) -> Result<ValidationEvaluation> {
    evaluate_validation_from_artifact(credential, dataset, EVAL_METRIC)
        .map_err(|error| CliError::ValidationFailed(error.to_string()))
}

/// A test-split measurement, computed over the rows the GRANT admitted.
///
/// There is no library evaluator for the test split, and that is deliberate on the library's
/// part: `ValidationEvaluation` is evidence that feeds selection, and a test measurement must
/// never be able to. So this is a plain accuracy over `grant.test()`, reported and never
/// convertible into a candidate — the type system enforces that, because nothing here
/// constructs a `ValidationEvaluation`.
struct TestMeasurement {
    value: f64,
    n_rows: usize,
}

fn evaluate_test(
    credential: &ReloadedSetFitCredential,
    split: &aprender_contrastive_data::split::Split<aprender_contrastive_data::split::Test>,
) -> Result<TestMeasurement> {
    use aprender::setfit::{ClassifyRequestDocument, MAX_BATCH_TEXTS};

    let rows = split.rows();
    if rows.is_empty() {
        return Err(CliError::ValidationFailed(
            "the canonical test split has no rows to measure".to_string(),
        ));
    }
    let labels = credential.model().ordered_labels().to_vec();
    let mut correct = 0_usize;
    let mut seen = 0_usize;
    for chunk in rows.chunks(MAX_BATCH_TEXTS) {
        let request = ClassifyRequestDocument::new(chunk.iter().map(|row| row.input.clone()));
        let response = credential
            .model()
            .classify(&request)
            .map_err(|error| CliError::InferenceFailed(error.to_string()))?;
        for (row, result) in chunk.iter().zip(response.results()) {
            let predicted = labels.iter().position(|label| label == result.label());
            if predicted == Some(row.label) {
                correct += 1;
            }
            seen += 1;
        }
    }
    if seen != rows.len() {
        return Err(CliError::InferenceFailed(format!(
            "the classifier returned {seen} results for {} test rows",
            rows.len()
        )));
    }
    #[allow(clippy::cast_precision_loss)]
    let value = correct as f64 / seen as f64;
    Ok(TestMeasurement {
        value,
        n_rows: seen,
    })
}

/// One candidate's row, carrying the SAME config hash the candidate was built with.
///
/// Passed in rather than recomputed here: two derivations of one value is two values that can
/// disagree, and the one the lock records is the one the report must show.
fn candidate_row(config_hash: &str, evaluation: &ValidationEvaluation) -> CandidateRow {
    CandidateRow {
        artifact_sha256: evaluation.artifact_hash().to_string(),
        config_hash: config_hash.to_string(),
        value: evaluation.value(),
        value_bits: evaluation.value_bits(),
    }
}

/// The hex SHA-256 over the artifact document's canonical `requested_config` sub-document.
///
/// Stated in the report as [`CONFIG_HASH_DERIVATION`] so Phase 5 can reproduce it from the
/// artifact alone. `serde_json::to_vec` over the recovered `Value` is canonical for this
/// purpose because the value came out of a `deny_unknown_fields` parse of the artifact's own
/// bytes and `serde_json::Map` preserves object order as read.
fn config_hash_of(credential: &ReloadedSetFitCredential) -> String {
    let requested = &credential.model().doc_view().requested_config;
    let bytes = serde_json::to_vec(requested).unwrap_or_default();
    aprender_contrastive_data::hash::hex(&Sha256::digest(&bytes).into())
}

/// The evidence table's hash, read straight off the artifact document.
///
/// Was a generic path-walker over `&[&str]`; it had exactly two callers with constant paths, a
/// silently-unreachable arm for any other root, and it cloned the found `Value`. Two direct
/// readers say the same thing with neither cost.
fn evidence_table_hash(credential: &ReloadedSetFitCredential) -> Option<String> {
    credential
        .model()
        .doc_view()
        .evidence
        .get("table_hash")?
        .as_str()
        .map(str::to_string)
}

/// The selection's root seed, read straight off the artifact document.
fn selection_root_seed(credential: &ReloadedSetFitCredential) -> Option<u64> {
    credential
        .model()
        .doc_view()
        .provenance
        .get("selection_root_seed")?
        .as_u64()
}

/// Refuse an occupied `--lock-out` destination — ONE decision, ONE wording, TWO call sites.
///
/// # The decision is not re-implemented here
///
/// `setfit_train::refuse_existing_output` is the crate's standalone no-clobber gate, and its
/// own doc says it exists to be callable before the work. This function delegates the DECISION
/// to it and replaces only the MESSAGE. That gate has exactly one failure mode, documented on
/// it, so nothing is lost by re-stating it.
///
/// # Why a bespoke message at all
///
/// The generic refusal is "Refusing to replace existing file {path} (pass --force)". True, and
/// it does not tell an operator what replacing THIS file costs: the lock is the record that a
/// selection was committed BEFORE canonical test access was taken, so replacing one silently
/// invalidates every test measurement taken under the old one while leaving a file that looks
/// current. That is worth spelling out, and worth spelling out in exactly ONE place — the
/// pre-flight and the write-time refusal being two wordings for one refusal is precisely how
/// they would drift.
fn refuse_existing_lock(destination: &Path, force: bool) -> Result<()> {
    crate::commands::setfit_train::refuse_existing_output(destination, force).map_err(|_| {
        CliError::ValidationFailed(format!(
            "{} already exists. A selection lock is a COMMITMENT, so overwriting one is never \
             implicit: pass --force if you intend to replace the committed decision, and be \
             aware that any test measurement taken under the old lock no longer describes the \
             selection this file records.",
            destination.display()
        ))
    })
}

/// Write the lock atomically, refusing to clobber without `--force`.
///
/// Temp file in the DESTINATION directory, `sync_all`, one `rename` — the 04-06 precedent, for
/// the same reason: a lock half-written by an interrupted run would be a file that looks like a
/// committed selection and is not one.
fn write_lock(destination: &Path, lock: &SelectionLock, force: bool) -> Result<()> {
    refuse_existing_lock(destination, force)?;
    // THE WRITE goes through the crate's ONE atomic writer, not a fourth hand-roll.
    //
    // THREE ORDERED CHECKS STAND HERE, and none is redundant with the others. They differ in
    // WHEN they run, which is the whole of their value:
    //
    //   1. the PRE-FLIGHT in `run`, before any input is read — so an occupied destination
    //      costs the operator nothing but the message. This was the missing one (WR-10): the
    //      refusal used to arrive after the corpus read, the load ladder and a classify pass
    //      over the validation split for every candidate.
    //   2. THIS call, before any bytes are produced — it covers a destination that appeared
    //      between the pre-flight and the decision, which on a long sweep is a real interval.
    //   3. `atomic_write`'s own `refuse_existing_output`, immediately before the rename — the
    //      narrowest window, and the reason `force` is still passed through.
    //
    // Deleting (3) as "now redundant" is the regression this arrangement invites, so it is
    // pinned: `write_lock_still_refuses_a_destination_that_appeared_mid_run` turns red when
    // that line is removed. What NONE of the three closes is the check-to-`rename` window
    // inside `atomic_write` itself (WR-01) — `rename` replaces unconditionally, so this path
    // is narrower than it was and is not race-free.
    //
    // The hand-rolled copy this replaces had drifted from the shared writer in two ways that
    // mattered: its temp was `.{name}.tmp` with no pid or ordinal, so two concurrent
    // `--lock-out` runs into one directory clobbered each other's scratch file, and it opened
    // with `create(true).truncate(true)` rather than `create_new(true)`, silently reusing a
    // crashed run's leftover instead of reporting it. `temp_path` + `fill_and_sync` get both
    // right, and they are proven once instead of three times.
    crate::commands::setfit_train::atomic_write(destination, &lock.to_canonical_bytes(), force)
}

/// Read a lock file, BOUNDED BEFORE THE READ, and reconstruct it.
pub(crate) fn read_lock(path: &Path) -> Result<SelectionLock> {
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::ValidationFailed(format!(
                "{} not found. A selection lock is written by `apr eval --split validation \
                 --lock-out <FILE>`; canonical test access is not reachable without one.",
                path.display()
            ))
        } else {
            CliError::Io(error)
        }
    })?;
    if !metadata.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() > MAX_SELECTION_LOCK_BYTES {
        return Err(CliError::InvalidFormat(format!(
            "{}: the selection lock declares {} bytes against the contracted bound of \
             {MAX_SELECTION_LOCK_BYTES}; refused from its declared length, before a byte is \
             read, because a bound applied after the allocation is not a bound.",
            path.display(),
            metadata.len()
        )));
    }
    let file = fs::File::open(path).map_err(CliError::Io)?;
    let mut bytes = Vec::new();
    // `+ 1` so a length that lied is DETECTED rather than silently truncated into a payload
    // that happens to parse.
    file.take(MAX_SELECTION_LOCK_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(CliError::Io)?;
    if bytes.len() as u64 > MAX_SELECTION_LOCK_BYTES {
        return Err(CliError::InvalidFormat(format!(
            "{}: the selection lock's stream exceeded {MAX_SELECTION_LOCK_BYTES} bytes, so its \
             declared length lied",
            path.display()
        )));
    }
    SelectionLock::from_canonical_bytes(&bytes).map_err(|error| {
        CliError::ValidationFailed(format!(
            "{} is not a usable selection lock: {error}",
            path.display()
        ))
    })
}

/// Emit the row, machine-readable or human.
fn report(json: bool, row: &EvalRow) -> Result<()> {
    if json {
        let rendered = serde_json::to_string_pretty(row).map_err(|error| {
            CliError::Aprender(format!("the evaluation row did not serialize: {error}"))
        })?;
        println!("{rendered}");
        return Ok(());
    }

    output::header(&format!("SetFit evaluation ({})", row.split));
    println!(
        "{}",
        output::kv_table(&[
            ("Metric", format!("{} = {:.6}", row.metric, row.value)),
            ("Rows", row.n_rows.to_string()),
            ("Artifact", row.artifact_sha256.clone()),
            ("Dataset", row.dataset_fingerprint.clone()),
            ("Validation split", row.validation_split_fingerprint.clone()),
            ("Labels", row.ordered_labels.join(", ")),
            (
                "Evidence table",
                row.evidence_table_hash
                    .clone()
                    .unwrap_or_else(|| "(absent)".to_string()),
            ),
            (
                "Selection seed",
                row.selection_root_seed
                    .map_or_else(|| "(absent)".to_string(), |s| s.to_string()),
            ),
        ])
    );
    if let Some(lock) = &row.lock {
        output::subheader(&format!("Selection lock ({})", lock.role));
        println!(
            "{}",
            output::kv_table(&[
                ("Path", lock.path.clone()),
                ("Lock hash", lock.lock_hash.clone()),
                ("Chosen artifact", lock.chosen_artifact_sha256.clone()),
                ("Rule", lock.rule.clone()),
            ])
        );
    }
    if !row.candidates.is_empty() {
        output::subheader("Candidates");
        let rows: Vec<Vec<String>> = row
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                vec![
                    i.to_string(),
                    c.artifact_sha256.clone(),
                    format!("{:.6}", c.value),
                ]
            })
            .collect();
        println!("{}", output::table(&["#", "Artifact", "Value"], &rows));
    }
    for note in &row.notes {
        println!("  {} {note}", "note:".yellow());
    }
    Ok(())
}

#[cfg(test)]
#[path = "setfit_tests.rs"]
mod setfit_tests;
