//! `apr setfit train` — the user surface of the SetFit training lifecycle (D-05).
//!
//! # This file is a FILESYSTEM ADAPTER and nothing else
//!
//! It reads a configuration file, hands the bytes to `aprender-train`'s single
//! validating constructor, replays Phase 2's artifacts through the doors
//! `data_contrastive` already owns, drives the shipped lifecycle transitions, and
//! writes the bytes the library hands back. Nothing here decides a knob's legality,
//! samples a pair, tunes an encoder, fits a head or judges a verification — every
//! one of those lives behind `entrenar::train::setfit`'s public API.
//!
//! # Fail on the REQUEST before blaming the data
//!
//! The order of the first four steps is deliberate and is asserted by tests:
//! parse the config, merge the overrides, resolve the device, refuse an existing
//! output — all four BEFORE a single row of `--data` is read and long before the
//! encoder is loaded. A bad seed is not a problem with the dataset, and reporting it
//! as one sends the operator to the wrong place. Refusing the output path last of
//! the four is what stops a twenty-minute run from ending in "I will not overwrite
//! that file".
//!
//! # The override merge goes through the PUBLIC validated door
//!
//! `--seed` and `--device` are applied to the REQUEST that `SetFitTrainConfig::to_request`
//! hands back, and the result goes through `SetFitTrainConfig::new`. The merged
//! configuration is therefore validated AS A WHOLE by the same single implementation
//! that validated the file. Mutating an already-validated config would let an invalid
//! merge through; deserializing into the library's private wire struct is not possible
//! from this crate at all, which is the review finding plan 04-14 closed by adding
//! `to_request`.
//!
//! # Every artifact goes through ONE writer
//!
//! [`atomic_write`] is the only function here that creates a file: temp file in the
//! DESTINATION directory, `write_all`, `sync_all`, `rename`, no-clobber unless
//! `--force`, and temp cleanup on every error path. There is exactly one `fs::rename`
//! site and no `File::create` outside it, and both facts are asserted on this file's
//! own source.
//!
//! # Not the network
//!
//! No `--offline` parameter, for the reason `dispatch_setfit_command` records: this
//! command opens no socket. `--model-dir` names a checkout the operator obtained
//! beforehand.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::hash::hex;
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::select::Selection;
use colored::Colorize;
use entrenar::train::device::{resolve_device, Device, DeviceError};
use entrenar::train::setfit::apr_codec::AprCodec;
use entrenar::train::setfit::config::SetFitTrainConfig;
use entrenar::train::setfit::{SetFitRun, SetFitTrainError};
use serde::Serialize;

use crate::commands::data_contrastive;
use crate::error::{CliError, Result};
use crate::output;

/// The two configuration encodings this command reads, for error messages.
const ACCEPTED_CONFIG_EXTENSIONS: &str = "`.toml` or `.json`";

/// What to do about a missing or unusable `--model-dir`.
///
/// The library cannot name a CLI flag, and it cannot know that this command never
/// downloads. Both facts belong in the adapter's message.
const MODEL_DIR_REMEDY: &str = "Point --model-dir at a BERT checkout containing \
     tokenizer.json and the encoder weights — the pinned all-MiniLM-L6-v2, which is held to \
     its tokenizer digest, or any other BERT, which is validated structurally. This command \
     NEVER downloads: obtain the checkout beforehand (for example with `batuta hf pull`) and \
     pass the directory.";

// ==========================================================================================
// CLOSED GAP (04-17 G1): the verified artifact's bytes ARE now reachable
// ==========================================================================================
//
// This block used to hold `ARTIFACT_BYTES_GAP` and a `verified_artifact_bytes` that always
// returned a typed refusal. `verify::run_verify_policy` dropped the artifact buffer and kept
// only `bytes.len()`, so this command could report an artifact's SHA-256 and could not write
// the file that digest was of. Both of the ways around it were worse than the gap and neither
// was taken: re-serializing here would be a SECOND implementation of the bundle -> artifact
// mapping whose output is not what was verified, and widening the library was outside 04-06's
// wave-5 file ownership.
//
// Plan 04-17 landed the door in the crate that owns it:
// `SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes(self) -> Vec<u8>`, handing
// back the exact `Vec` the policy hashed and closed — `verify_into_artifact_bytes_are_the_
// hashed_bytes` re-hashes it and requires the digest to equal `artifact_hash()`. So the bytes
// written below and the digest printed beside them cannot disagree.
//
// The door CONSUMES the run, which is the one thing a caller has to arrange for: every value
// the completion report needs is read before the write. There is no local wrapper — this
// module calls the library door exactly once, and
// `setfit_train_e2e_records_the_blocker_that_stops_short_of_an_artifact` asserts that.

// ==========================================================================================
// Test-only fault-injection seam (the shape `data_contrastive` established)
// ==========================================================================================

// When set, `atomic_write` fails AFTER the temp file is written and synced but BEFORE the
// rename — the one window in which a partial artifact could exist.
//
// A `thread_local` rather than a global: cargo runs each test on its own thread, so two
// tests cannot see each other's injection.
#[cfg(test)]
thread_local! {
    static FAIL_BEFORE_RENAME: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether an induced pre-rename failure is armed on this thread.
#[cfg(test)]
fn induced_prerename_failure() -> bool {
    FAIL_BEFORE_RENAME.with(std::cell::Cell::get)
}

/// Production builds have no seam at all.
#[cfg(not(test))]
const fn induced_prerename_failure() -> bool {
    false
}

// ==========================================================================================
// The single writer
// ==========================================================================================

/// A temp name inside the DESTINATION directory, unique per process and per call.
fn temp_path(target: &Path) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ordinal = COUNTER.fetch_add(1, Ordering::Relaxed);
    let stem = target.file_name().map_or_else(
        || "artifact".to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!(".{stem}.tmp.{}.{ordinal}", std::process::id()))
}

/// Fill the temp file and get it onto the platter, then offer the injection point.
fn fill_and_sync(temp: &Path, bytes: &[u8]) -> Result<()> {
    // `create_new` on the TEMP as well: two concurrent runs must not share one scratch
    // file, and a leftover scratch file from a crashed run is a diagnosable error rather
    // than silent reuse.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    if induced_prerename_failure() {
        return Err(CliError::Io(std::io::Error::other(
            "induced pre-rename failure (test seam)",
        )));
    }
    Ok(())
}

/// Produce `target` atomically, with no-clobber by default.
///
/// The temp file lives in the destination directory because `rename` is only atomic
/// within one filesystem; a temp in `/tmp` would silently degrade to a copy across a
/// mount point, which is exactly the partial-write window this exists to close.
///
/// Every failure path removes the temp, so an interrupted write leaves neither a partial
/// artifact nor a stray file for the next `--force`-less run to trip over — and the
/// cleanup's own failure never replaces the reason the write failed.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] when `target` exists and `force` is false;
/// [`CliError::Io`] for any create, write, sync or rename failure.
pub(crate) fn atomic_write(target: &Path, bytes: &[u8], force: bool) -> Result<()> {
    refuse_existing_output(target, force)?;
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;

    let temp = temp_path(target);
    let result =
        fill_and_sync(&temp, bytes).and_then(|()| fs::rename(&temp, target).map_err(CliError::Io));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// The no-clobber gate, callable on its own so it can run BEFORE the work.
///
/// # Three call sites in two commands, and the order is the point
///
/// This was `fn`, and the doc above already said it exists to be callable before training —
/// a stated purpose the visibility prevented any other module from using. `apr eval
/// --lock-out` therefore ran its ENTIRE multi-candidate sweep before discovering that the
/// lock file it was asked to write already existed (finding WR-10), which is exactly the
/// twenty-minute run ending in "I will not overwrite that file" that this module's header
/// says the ordering discipline exists to prevent. It is `pub(crate)` now, and the three
/// call sites are ordered:
///
/// 1. `apr setfit train`'s pre-flight — before a row of `--data` is read.
/// 2. `apr eval --lock-out`'s pre-flight — same rule, one command over, reached through
///    `eval::setfit::refuse_existing_lock`, which re-states this refusal in the wording a
///    selection lock needs rather than re-implementing the decision.
/// 3. [`atomic_write`]'s own check — immediately before the rename.
///
/// The third is NOT made redundant by the first two, and deleting it as such would be the
/// regression `write_lock_still_refuses_a_destination_that_appeared_mid_run` exists to
/// catch: a pre-flight closes the window in which a long run ends by refusing to write,
/// while the write-time check is what makes the guarantee true for a file that appeared
/// WHILE the run was going.
///
/// # What this still does not close
///
/// The check-to-`fs::rename` window inside [`atomic_write`] remains open (WR-01): `rename`
/// replaces its destination unconditionally, so a file created after this check and before
/// that rename is still destroyed without `--force`. Refusing earlier narrows the window; it
/// does not make the write path race-free, and closing it needs the destination taken with
/// `O_CREAT|O_EXCL`.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] naming the path and `--force`. That is the ONLY failure
/// mode, which is what lets a caller replace the message without losing a distinction.
pub(crate) fn refuse_existing_output(target: &Path, force: bool) -> Result<()> {
    if !force && target.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Refusing to replace existing file {} (pass --force to replace it)",
            target.display()
        )));
    }
    Ok(())
}

// ==========================================================================================
// Configuration: file first, then a validated merge
// ==========================================================================================

/// Parse `--config` by extension, straight through the library's validating deserializer.
///
/// Deserialization IS validation here: `SetFitTrainConfig` carries
/// `#[serde(try_from = ...)]` onto its single constructor, so an unknown key is refused
/// by `deny_unknown_fields` and an invalid VALUE is refused by the knob table — both
/// before this function returns, and therefore before anything is read from `--data`.
///
/// # Errors
///
/// [`CliError::InvalidFormat`] for an extension this command does not read;
/// [`CliError::ValidationFailed`] for any parse or knob rejection, prefixed so the
/// message says which file the operator has to edit.
fn parse_config(path: &Path) -> Result<SetFitTrainConfig> {
    let bytes = fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::FileNotFound(path.to_path_buf())
        } else {
            CliError::Io(error)
        }
    })?;

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("toml") => toml::from_str(&bytes).map_err(|error| config_error(path, &error)),
        Some("json") => serde_json::from_str(&bytes).map_err(|error| config_error(path, &error)),
        // Sniffing the content instead would make the same file mean different things on
        // different days. The extension is the declaration.
        other => Err(CliError::InvalidFormat(format!(
            "--config {}: unsupported extension {}; expected {ACCEPTED_CONFIG_EXTENSIONS}",
            path.display(),
            other.map_or_else(|| "<none>".to_string(), |ext| format!("`.{ext}`")),
        ))),
    }
}

/// One rendering for both deserializers, so a TOML and a JSON rejection read alike.
fn config_error(path: &Path, error: &impl std::fmt::Display) -> CliError {
    CliError::ValidationFailed(format!("--config {}: {error}", path.display()))
}

/// Apply `--seed` / `--device` through the PUBLIC validated merge door.
///
/// Read the twelve knobs back out as a REQUEST, override at most two of them, and send
/// the whole thing back through the single validating constructor. The order matters:
/// the merged result is validated AS A WHOLE, so an override cannot arrive at a value the
/// file itself would have been refused for. Overriding a field on the already-validated
/// config would be the opposite — the value would never meet the knob table, and the
/// type's promise that every value of it has been validated for its whole life would
/// become false.
///
/// Note that `SetFitTrainConfig::new` normalizes `pair_config.root_seed` to the top-level
/// seed, so `--seed` reseeds the pair stream too. That is the documented behaviour of the
/// door, not an accident of this call site.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] naming the flag whose value the merged configuration
/// was refused for.
fn merge_overrides(
    config: &SetFitTrainConfig,
    seed: Option<u64>,
    device: Option<&str>,
) -> Result<SetFitTrainConfig> {
    let mut request = config.to_request();
    if let Some(seed) = seed {
        request.root_seed = seed;
    }
    if let Some(device) = device {
        request.device = device.to_string();
    }
    SetFitTrainConfig::new(request).map_err(|error| {
        CliError::ValidationFailed(format!(
            "the merged configuration was refused: {error} — check --config together with \
             {}",
            overridden_flags(seed, device)
        ))
    })
}

/// Resolve a training configuration for a caller that may or may not have a file.
///
/// # The ONE config door, shared with `apr setfit bench run`
///
/// A benchmark cell's `--config` is optional and its seed is the CELL's, not the file's, so
/// the bench adapter needs "parse if present, else the frozen published defaults, then apply
/// this seed". Every one of those steps already exists here, and re-deriving them in
/// `setfit_bench.rs` would put a second definition of "the default configuration" in the
/// repository — which is precisely the knob a benchmark must not have two of.
///
/// `SetFitTrainConfig::reference_defaults` is the library's own frozen recipe. It is not
/// copied here; it is called.
///
/// # Errors
///
/// Everything [`parse_config`] and [`merge_overrides`] return.
pub(crate) fn resolve_config(config_path: Option<&Path>, seed: u64) -> Result<SetFitTrainConfig> {
    let base = match config_path {
        Some(path) => parse_config(path)?,
        None => SetFitTrainConfig::reference_defaults(seed),
    };
    // The seed goes through the validated merge even on the defaults path, so the merged
    // configuration is validated AS A WHOLE by the same single implementation in both cases
    // and `pair_config.root_seed` is reseeded with it.
    merge_overrides(&base, Some(seed), None)
}

/// Name the flags that participated in the merge, so the message points somewhere.
fn overridden_flags(seed: Option<u64>, device: Option<&str>) -> String {
    let mut flags = Vec::new();
    if seed.is_some() {
        flags.push("--seed");
    }
    if device.is_some() {
        flags.push("--device");
    }
    if flags.is_empty() {
        return "no overrides (the file alone was refused)".to_string();
    }
    flags.join(" and ")
}

/// Probe the host for the MERGED device request, failing closed (OPS-06).
///
/// This runs before any data is read and long before the encoder is loaded: an explicit
/// device this host cannot provide is a fact about the request, and discovering it after
/// twenty minutes of training would be useless. `resolve_device` is the single definition
/// of both the grammar and the no-silent-fallback rule; this only translates the refusal
/// and appends the flag the library cannot name.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for an unavailable or unparseable device — nonzero exit
/// code 5, never a silent fall back to CPU.
fn resolve_requested_device(config: &SetFitTrainConfig) -> Result<Device> {
    let requested = config.device().as_str();
    resolve_device(requested).map_err(|error| {
        let remedy = match error {
            DeviceError::CudaNotAvailable { .. } => {
                " — this host has no usable CUDA runtime. Pass --device cpu to opt in to the \
                 CPU path explicitly; there is deliberately no silent fallback, because a \
                 benchmark number produced on a device nobody asked for is worse than no \
                 number"
            }
            DeviceError::InvalidSpec(_) => {
                " — set `device` in --config, or pass --device, to one of cpu, cuda, cuda:N \
                 (N in 0..=15) or auto"
            }
        };
        CliError::ValidationFailed(format!("device: {error}{remedy}"))
    })
}

// ==========================================================================================
// Phase 2 ingest — through the doors `apr data` already owns
// ==========================================================================================

/// What the selection says about where this run's data came from.
///
/// Captured BEFORE the selection is moved into the lifecycle, because the report needs it
/// on both the dry-run path (where no run exists) and the training path.
#[derive(Serialize)]
struct Provenance {
    dataset_fingerprint: String,
    validation_split_fingerprint: String,
    selection_semantic_hash: String,
    selection_ledger_hash: String,
    selection_root_seed: u64,
    shots_per_class: u32,
}

impl Provenance {
    /// Read every value off the replayed selection. Nothing here is recomputed.
    fn of(selection: &Selection) -> Self {
        Self {
            dataset_fingerprint: selection.dataset_fingerprint_hex().to_string(),
            validation_split_fingerprint: selection.validation_fingerprint_hex().to_string(),
            selection_semantic_hash: hex(&selection.semantic_hash()),
            selection_ledger_hash: hex(&selection.ledger_hash()),
            selection_root_seed: selection.root_seed(),
            shots_per_class: selection.shots_per_class(),
        }
    }
}

/// Everything the two Phase 2 artifacts produce, replayed against each other.
struct Phase2Inputs {
    dataset: aprender_contrastive_data::prepared::PreparedDataset<
        aprender_contrastive_data::prepared::Canonical,
    >,
    selection: Selection,
    provenance: Provenance,
}

/// Read `--data` and `--selection` through the attested doors, strictly.
///
/// Both doors belong to `data_contrastive`: one reader of `benchmark-manifest.json` and
/// one reader of `selection-manifest.json` for the whole CLI. `Selection::replay` is the
/// only route from manifest bytes back to a `Selection` — this command cannot construct
/// one, because the crate's constructor is private.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for any attested-ingest or replay rejection;
/// [`CliError::Io`] for a read failure.
fn read_phase2_inputs(data: &Path, selection_path: &Path) -> Result<Phase2Inputs> {
    let mut ledger = AccessLedger::new();
    let dataset = data_contrastive::read_attested_canonical(data, &mut ledger)?;
    let manifest = data_contrastive::read_selection_manifest(selection_path)?;
    let selection = Selection::replay(&manifest, &dataset, &mut ledger).map_err(|error| {
        CliError::ValidationFailed(format!(
            "selection {} does not replay against --data {}: {error}",
            selection_path.display(),
            data.display()
        ))
    })?;
    let provenance = Provenance::of(&selection);
    Ok(Phase2Inputs {
        dataset,
        selection,
        provenance,
    })
}

/// Map a lifecycle failure onto the CLI surface, adding the knob the library cannot name.
fn train_error(error: &SetFitTrainError) -> CliError {
    let remedy = match error {
        SetFitTrainError::UncalibratedRegime { .. } => {
            " — the thresholds this gate applies were MEASURED in another regime, and an \
             epsilon measured on one architecture is not evidence about another. Widening the \
             calibrated set is a deliberate contract edit (Phase 3 D-10(c)), never an inline \
             change to unblock a run"
        }
        SetFitTrainError::UnsupportedDeviceForPhase3 { .. } => {
            " — pass --device cpu, or set `device` in --config"
        }
        SetFitTrainError::SelectionRowMissing { .. }
        | SetFitTrainError::SelectionRowContentMismatch { .. } => {
            " — --selection and --data disagree about the rows. Re-run `apr data select` \
             against THIS dataset directory"
        }
        _ => "",
    };
    CliError::ValidationFailed(format!("setfit training: {error}{remedy}"))
}

// ==========================================================================================
// Reports
// ==========================================================================================

/// The MERGED configuration, exactly as validated, plus the probe's answer.
///
/// This is what makes a Phase 5 cell reproducible from the report alone (D-07): the twelve
/// knobs printed here are the merged ones, not the file's, so a reader never has to know
/// which flags were passed to reconstruct the run.
#[derive(Serialize)]
struct ResolvedConfigReport<'a> {
    requested: &'a SetFitTrainConfig,
    resolved_device: String,
}

/// `apr setfit train --json`.
#[derive(Serialize)]
struct TrainReport<'a> {
    command: &'static str,
    output: String,
    artifact_sha256: String,
    artifact_format_id: &'a str,
    evidence_table_hash: &'a str,
    provenance: &'a Provenance,
    resolved: ResolvedConfigReport<'a>,
}

/// `apr setfit train --dry-run [--json]`.
///
/// The `checks` list is explicit about what a dry run does NOT do. A pre-flight that lets
/// a reader assume it validated the encoder would be worse than no pre-flight.
#[derive(Serialize)]
struct DryRunReport<'a> {
    command: &'static str,
    dry_run: bool,
    output: String,
    model_dir: String,
    provenance: &'a Provenance,
    resolved: ResolvedConfigReport<'a>,
    checks_performed: [&'static str; 4],
    checks_skipped: [&'static str; 3],
}

/// The four things a dry run really does check.
const DRY_RUN_PERFORMED: [&str; 4] = [
    "config parsed and validated as a whole, including --seed/--device overrides",
    "device resolved on this host (no silent fallback)",
    "--output refused if it exists without --force",
    "--data and --selection replayed strictly against each other",
];

/// And the three it does not.
const DRY_RUN_SKIPPED: [&str; 3] = [
    "--model-dir was NOT opened (a multi-hundred-megabyte read is not a pre-flight)",
    "no encoder tuning, head fit or artifact verification ran",
    "nothing was written",
];

/// Human-readable dry-run output.
fn render_dry_run_human(report: &DryRunReport) {
    output::section("SetFit Training — Dry Run");
    println!();
    output::kv("Output", &report.output);
    output::kv("Model dir", &report.model_dir);
    output::kv("Device", &report.resolved.resolved_device);
    output::kv("Root seed", report.resolved.requested.root_seed());
    output::kv("Epochs", report.resolved.requested.epochs());
    output::kv("Batch size", report.resolved.requested.batch_size());
    output::kv("Encoder LR", report.resolved.requested.encoder_lr());
    output::kv(
        "Dataset fingerprint",
        &report.provenance.dataset_fingerprint,
    );
    output::kv(
        "Selection",
        format!(
            "{} ({} shot(s) per class, seed {})",
            report.provenance.selection_semantic_hash,
            report.provenance.shots_per_class,
            report.provenance.selection_root_seed
        ),
    );
    println!();
    println!("{} Checked:", "OK".green());
    for line in report.checks_performed {
        println!("  - {line}");
    }
    println!();
    println!("Not checked by a dry run:");
    for line in report.checks_skipped {
        println!("  - {line}");
    }
}

/// Human-readable completion output.
fn render_train_human(report: &TrainReport) {
    output::section("SetFit Training");
    println!();
    output::kv("Artifact", &report.output);
    output::kv("Format", report.artifact_format_id);
    output::kv("SHA-256", &report.artifact_sha256);
    output::kv("Evidence table", report.evidence_table_hash);
    output::kv("Device", &report.resolved.resolved_device);
    output::kv("Root seed", report.resolved.requested.root_seed());
    output::kv(
        "Dataset fingerprint",
        &report.provenance.dataset_fingerprint,
    );
    println!();
    println!(
        "{} The written artifact was reloaded from its own bytes and re-predicted before \
         this line printed.",
        "OK".green()
    );
}

/// Serialize a report, or say why it could not be serialized.
fn report_json(report: &impl Serialize) -> Result<String> {
    serde_json::to_string_pretty(report).map_err(|error| {
        CliError::ValidationFailed(format!("Failed to encode the training report: {error}"))
    })
}

// ==========================================================================================
// The command
// ==========================================================================================

/// Train a SetFit classifier from Phase 2 artifacts and write a verified APR.
///
/// See the module header for the ordering rules this body implements.
///
/// # Errors
///
/// [`CliError::FileNotFound`] for an absent `--config`; [`CliError::InvalidFormat`] for a
/// config extension this command does not read; [`CliError::ValidationFailed`] for any
/// knob, merge, device, ingest, replay or lifecycle rejection, and for an existing
/// `--output` without `--force`; [`CliError::ModelLoadFailed`] for an unusable
/// `--model-dir`; [`CliError::Io`] for a read or write failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    config_path: &Path,
    data: &Path,
    selection_path: &Path,
    model_dir: &Path,
    output_path: &Path,
    seed: Option<u64>,
    device: Option<&str>,
    force: bool,
    dry_run: bool,
    json_output: bool,
) -> Result<()> {
    // (1) The REQUEST, in full, before anything is read from --data.
    let file_config = parse_config(config_path)?;
    let merged = merge_overrides(&file_config, seed, device)?;
    let resolved_device = resolve_requested_device(&merged)?;
    // (2) And the one filesystem fact that is also a property of the request.
    refuse_existing_output(output_path, force)?;

    // (3) Phase 2's artifacts, replayed strictly against each other.
    let inputs = read_phase2_inputs(data, selection_path)?;

    let resolved = ResolvedConfigReport {
        requested: &merged,
        resolved_device: resolved_device.tag(),
    };

    if dry_run {
        let report = DryRunReport {
            command: "setfit-train",
            dry_run: true,
            output: output_path.display().to_string(),
            model_dir: model_dir.display().to_string(),
            provenance: &inputs.provenance,
            resolved,
            checks_performed: DRY_RUN_PERFORMED,
            checks_skipped: DRY_RUN_SKIPPED,
        };
        if json_output {
            println!("{}", report_json(&report)?);
        } else {
            render_dry_run_human(&report);
        }
        return Ok(());
    }

    // (4) The encoder. Deliberately after everything above: it is the expensive read.
    let encoder =
        SetFitMiniLm::from_pretrained_dir(model_dir, merged.root_seed()).map_err(|error| {
            CliError::ModelLoadFailed(format!(
                "--model-dir {}: {error}. {MODEL_DIR_REMEDY}",
                model_dir.display()
            ))
        })?;

    // (5) The shipped lifecycle, in the only order the typestate permits. Every gate —
    //     the device probe, the pair budget, the selection/dataset agreement, the
    //     calibration regime, the evidence thresholds, the head fit and the artifact
    //     round trip — is inside these four calls.
    let prepared = SetFitRun::prepare(encoder, inputs.dataset, inputs.selection, merged.clone())
        .map_err(|e| train_error(&e))?;
    let tuned = prepared.tune_encoder().map_err(|e| train_error(&e))?;
    let fitted = tuned.fit_head().map_err(|e| train_error(&e))?;
    let verified = fitted
        .verify_artifact(&AprCodec::new())
        .map_err(|e| train_error(&e))?;

    // (6) The report's three recorded values, read BEFORE the write — `into_artifact_bytes`
    //     consumes the run, deliberately (a borrowing door would let a caller hold ~90 MB of
    //     artifact and the whole live model at once). Reading them here is not a workaround:
    //     they are recorded facts about a run that has finished, and the file about to be
    //     written is the artifact they describe.
    let artifact_sha256 = verified.artifact_hash();
    let artifact_format_id = verified.artifact_format_id().to_string();
    let evidence_table_hash = verified.evidence_table_hash().to_string();

    // (7) The write. One rename site, no-clobber, temp in the destination directory. These
    //     are the bytes the trusted policy hashed and round-trip-closed, not a
    //     re-serialization of them, so `artifact_sha256` above is genuinely this file's
    //     digest.
    let bytes = verified.into_artifact_bytes();
    atomic_write(output_path, &bytes, force)?;

    // Advisory, on stderr, in BOTH modes. Suppressing it under --json would hide
    // it from the one consumer that can act on it — a packaging pipeline — while
    // leaving the JSON document on stdout untouched either way.
    //
    // Both numbers in the sentence are MiB (2^20), which is the unit the quota is
    // actually expressed in: AWS_LAMBDA_ZIP_LIMIT_BYTES is 262_144_000 — the exact
    // figure Lambda's InvalidParameterValueException names — and that is 250 MiB
    // exactly, which is what "250 MB" in the AWS quota tables means. Dividing it by
    // 10^6 printed "262 MB", a third threshold appearing in no AWS document and
    // matching neither the quota page (250) nor the API error (262144000); an
    // operator checking the advisory against AWS found no such number. Printing MiB
    // next to a decimal-MB threshold is how "256.0 MB exceeds the 250 MB limit"
    // comes to be a comparison of two different units, so BOTH sides use MiB.
    let limit = aprender::setfit::AWS_LAMBDA_ZIP_LIMIT_BYTES;
    if bytes.len() as u64 > limit {
        let size_mib = bytes.len() as f64 / 1_048_576.0;
        let limit_mib = limit as f64 / 1_048_576.0;
        eprintln!(
            "  {} Artifact size ({size_mib:.1} MiB) exceeds the {limit_mib:.0} MiB unzipped limit for standard AWS Lambda zip deployment. Deploy via container image (Docker) or other compute targets (ECS, EC2, Kubernetes).",
            output::badge_info("NOTE")
        );
    }

    // (8) The report, from the run's own recorded values.
    let report = TrainReport {
        command: "setfit-train",
        output: output_path.display().to_string(),
        artifact_sha256,
        artifact_format_id: &artifact_format_id,
        evidence_table_hash: &evidence_table_hash,
        provenance: &inputs.provenance,
        resolved,
    };
    if json_output {
        println!("{}", report_json(&report)?);
    } else {
        render_train_human(&report);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::data_tweeteval::{self, fixtures, CANONICAL_REVISION};
    use crate::TweetEvalStanceProfile;
    use aprender::setfit::MAX_SEQUENCE_LENGTH;
    use std::collections::BTreeSet;
    use tempfile::TempDir;

    /// This module's own source, for the source assertions.
    const SETFIT_TRAIN_SOURCE: &str = include_str!("setfit_train.rs");

    /// Assemble a search needle from fragments at RUNTIME.
    ///
    /// These scans read the file they live in, so a whole literal would appear IN the
    /// scanned source and inflate the count — the self-match hazard 04-14 committed once
    /// inside the very helper that warns about it. This doc comment therefore describes
    /// the needles without spelling any of them.
    fn needle(fragments: &[&str]) -> String {
        fragments.concat()
    }

    /// The contracted benchmark seed every fixture here uses.
    const FIXTURE_SEED: u64 = 13;

    /// A hand-authored TOML config carrying all twelve knobs.
    ///
    /// Hand-authored on purpose: this is the file shape an operator types, and the test
    /// below asserts it lands on exactly the value the library's own
    /// `SetFitTrainConfig::reference_defaults` produces. `budget` and `hard_cap` are
    /// omitted rather than written as nulls, which TOML cannot express — the library's
    /// wire form makes both optional, and the omission is the shape a real file has.
    fn reference_toml(seed: u64) -> String {
        format!(
            "encoder_lr = 2e-5\n\
             epochs = 1\n\
             batch_size = 16\n\
             warmup_ratio = 0.1\n\
             grad_clip_max_norm = 1.0\n\
             max_length = {max_length}\n\
             freeze_policy = []\n\
             root_seed = {seed}\n\
             device = \"cpu\"\n\
             lr_schedule = \"warmup_linear_decay\"\n\
             \n\
             [pair_config]\n\
             strategy = \"oversampling\"\n\
             singleton_policy = \"negatives_only\"\n\
             \n\
             [head_regularization]\n\
             kind = \"sklearn_equivalent_c\"\n\
             c = 1.0\n",
            max_length = MAX_SEQUENCE_LENGTH,
        )
    }

    fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, contents).expect("fixture file is writable");
        path
    }

    /// A config file this command accepts, at the given extension.
    fn config_file(dir: &Path, name: &str) -> PathBuf {
        write(dir, name, &reference_toml(FIXTURE_SEED))
    }

    /// A real Phase 2 benchmark directory plus its selection manifest.
    ///
    /// Both are produced by RUNNING the shipped commands over a synthetic source tree —
    /// `apr data tweet-eval-stance` then `apr data select` — exactly as
    /// `data_contrastive`'s own tests do. Nothing here fabricates an attestation or a
    /// manifest by hand: a hand-built one would only prove this reader accepts what this
    /// test module thinks the writer emits.
    fn phase2_artifacts(root: &Path) -> (PathBuf, PathBuf) {
        let source = root.join("source");
        fs::create_dir_all(&source).expect("fixture source directory is creatable");
        fixtures::write_canonical_fixture_tagged(&source, fixtures::DEFAULT_TAG);
        let data = root.join("benchmark");
        data_tweeteval::run(
            &data,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            true,
        )
        .expect("the synthetic canonical fixture prepares cleanly");
        data_contrastive::run_select(&data, 8, FIXTURE_SEED, false, None, false, true)
            .expect("selection succeeds against the prepared fixture");
        let selection = data.join("selection-manifest.json");
        assert!(selection.is_file(), "apr data select wrote its manifest");
        (data, selection)
    }

    /// Every path under `root`, so "nothing was written" is a set comparison rather than
    /// a spot check on the one filename the test happened to think of.
    fn listing(root: &Path) -> BTreeSet<PathBuf> {
        fn walk(dir: &Path, into: &mut BTreeSet<PathBuf>) {
            let Ok(entries) = fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, into);
                }
                into.insert(path);
            }
        }
        let mut out = BTreeSet::new();
        walk(root, &mut out);
        out
    }

    // --------------------------------------------------------------------------------
    // Configuration: both encodings, and every rejection BEFORE training
    // --------------------------------------------------------------------------------

    #[test]
    fn setfit_train_accepts_a_toml_config() {
        let temp = TempDir::new().expect("tempdir");
        let path = config_file(temp.path(), "train.toml");

        let parsed = parse_config(&path).expect("a well-formed TOML config is accepted");
        assert_eq!(
            parsed,
            SetFitTrainConfig::reference_defaults(FIXTURE_SEED),
            "a hand-authored TOML file must land on exactly the value the library's own \
             reference recipe produces — otherwise the documented file shape and the \
             documented defaults describe two different runs"
        );
    }

    #[test]
    fn setfit_train_accepts_a_json_config() {
        let temp = TempDir::new().expect("tempdir");
        // Derived from the library rather than hand-authored, so this leg proves the JSON
        // round trip through the SAME validating constructor.
        let encoded =
            serde_json::to_string_pretty(&SetFitTrainConfig::reference_defaults(FIXTURE_SEED))
                .expect("the reference config serializes");
        let path = write(temp.path(), "train.json", &encoded);

        let parsed = parse_config(&path).expect("a well-formed JSON config is accepted");
        assert_eq!(parsed, SetFitTrainConfig::reference_defaults(FIXTURE_SEED));
    }

    #[test]
    fn setfit_train_refuses_an_unknown_config_field() {
        let temp = TempDir::new().expect("tempdir");
        let mut contents = reference_toml(FIXTURE_SEED);
        contents.push_str("mystery_knob = 7\n");
        let path = write(temp.path(), "train.toml", &contents);

        let error = parse_config(&path).expect_err("an unknown key must be refused");
        assert!(
            matches!(error, CliError::ValidationFailed(_)),
            "an unknown key is a validation failure (exit 5); got: {error}"
        );
        assert!(
            error.to_string().contains("mystery_knob"),
            "the refusal must name the offending key; got: {error}"
        );
    }

    #[test]
    fn setfit_train_refuses_an_invalid_knob_value() {
        let temp = TempDir::new().expect("tempdir");
        // A structurally perfect file with one value the knob table forbids. This is the
        // case `deny_unknown_fields` cannot catch and the validating constructor must.
        let contents = reference_toml(FIXTURE_SEED).replace("epochs = 1", "epochs = 0");
        let path = write(temp.path(), "train.toml", &contents);

        let error = parse_config(&path).expect_err("a zero epoch count must be refused");
        assert!(
            matches!(error, CliError::ValidationFailed(_)),
            "got: {error}"
        );
        assert!(
            error.to_string().contains("epochs"),
            "the refusal must name the knob, not merely say the config was invalid; got: {error}"
        );
    }

    #[test]
    fn setfit_train_refuses_a_config_extension_it_does_not_read() {
        let temp = TempDir::new().expect("tempdir");
        let path = write(temp.path(), "train.yaml", &reference_toml(FIXTURE_SEED));

        let error = parse_config(&path).expect_err("an unread extension must be refused");
        assert!(
            matches!(error, CliError::InvalidFormat(_)),
            "a wrong extension is a format problem (exit 4), not a validation one; got: {error}"
        );
        assert!(
            error.to_string().contains(".toml") && error.to_string().contains(".json"),
            "the refusal must name what IS accepted; got: {error}"
        );
    }

    // --------------------------------------------------------------------------------
    // The merge, through the public door
    // --------------------------------------------------------------------------------

    #[test]
    fn setfit_train_seed_override_is_reflected_in_the_merged_config() {
        let base = SetFitTrainConfig::reference_defaults(FIXTURE_SEED);
        let merged = merge_overrides(&base, Some(29), None).expect("a contracted seed merges");

        assert_eq!(merged.root_seed(), 29, "the override must reach the config");
        assert_eq!(
            merged.pair_config().root_seed,
            29,
            "the merge door reseeds the pair stream too — a run whose pair sampler kept the \
             file's seed would replay a stream the reported seed does not describe"
        );
        assert_eq!(
            base.root_seed(),
            FIXTURE_SEED,
            "the merge must not mutate its input"
        );
    }

    #[test]
    fn setfit_train_refuses_an_invalid_device_override_at_the_merge() {
        let base = SetFitTrainConfig::reference_defaults(FIXTURE_SEED);
        let error = merge_overrides(&base, None, Some("gpu"))
            .expect_err("a device outside the grammar must be refused by the merge");

        assert!(
            matches!(error, CliError::ValidationFailed(_)),
            "got: {error}"
        );
        let rendered = error.to_string();
        assert!(
            rendered.contains("--device"),
            "the refusal must name the flag that carried the bad value; got: {rendered}"
        );
    }

    #[test]
    fn setfit_train_device_cuda_fails_closed_with_a_nonzero_exit_code() {
        // The grammar accepts `cuda`, so this passes the merge and is refused by the PROBE.
        // Splitting the two is what makes "no silent fallback" testable on a CPU host: the
        // request is well-formed and is still refused.
        let base = SetFitTrainConfig::reference_defaults(FIXTURE_SEED);
        let merged = merge_overrides(&base, None, Some("cuda")).expect("`cuda` parses");

        let error = resolve_requested_device(&merged)
            .expect_err("an explicit CUDA request on a CPU-only host must fail closed");
        let rendered = error.to_string();
        assert!(
            rendered.contains("CUDA"),
            "the refusal must say what was unavailable; got: {rendered}"
        );
        assert!(
            rendered.contains("--device cpu"),
            "and it must name the opt-in, because the absent behaviour is a silent fallback; \
             got: {rendered}"
        );
        assert_ne!(
            format!("{:?}", error.exit_code()),
            format!("{:?}", std::process::ExitCode::SUCCESS),
            "OPS-06 requires a NONZERO exit code, not merely a message on stderr"
        );
    }

    // --------------------------------------------------------------------------------
    // Output safety
    // --------------------------------------------------------------------------------

    #[test]
    fn setfit_train_refuses_an_existing_output_before_it_reads_the_data() {
        let temp = TempDir::new().expect("tempdir");
        let config = config_file(temp.path(), "train.toml");
        let output = write(temp.path(), "model.apr", "previous artifact");

        // --data does not exist. If the output check ran after ingest, THAT is the error
        // this would report, so the assertion below pins the ORDER and not just the check.
        let error = run(
            &config,
            &temp.path().join("absent-data"),
            &temp.path().join("absent-selection.json"),
            &temp.path().join("absent-model-dir"),
            &output,
            None,
            None,
            false,
            false,
            true,
        )
        .expect_err("an existing --output without --force must be refused");

        let rendered = error.to_string();
        assert!(
            rendered.contains("model.apr") && rendered.contains("--force"),
            "the refusal must name the file and the flag; got: {rendered}"
        );
        assert_eq!(
            fs::read_to_string(&output).expect("the existing file is still readable"),
            "previous artifact",
            "a refused run must not have touched the file it refused to replace"
        );
    }

    #[test]
    fn setfit_train_a_failed_write_leaves_no_partial_file() {
        let temp = TempDir::new().expect("tempdir");
        let target = temp.path().join("model.apr");
        let before = listing(temp.path());

        FAIL_BEFORE_RENAME.with(|cell| cell.set(true));
        let error = atomic_write(&target, b"bytes that never land", false)
            .expect_err("the injected pre-rename failure propagates");
        FAIL_BEFORE_RENAME.with(|cell| cell.set(false));

        assert!(matches!(error, CliError::Io(_)), "got: {error}");
        assert!(!target.exists(), "no artifact at the target path");
        assert_eq!(
            listing(temp.path()),
            before,
            "and no temp file left behind either — a stray scratch file would make the \
             next --force-less run fail for the wrong reason"
        );
    }

    // --------------------------------------------------------------------------------
    // The dry run, over REAL Phase 2 artifacts
    // --------------------------------------------------------------------------------

    #[test]
    fn setfit_train_dry_run_validates_the_real_inputs_and_writes_nothing() {
        let temp = TempDir::new().expect("tempdir");
        let (data, selection) = phase2_artifacts(temp.path());
        let config = config_file(temp.path(), "train.toml");
        let output = temp.path().join("model.apr");
        let before = listing(temp.path());

        run(
            &config,
            &data,
            &selection,
            // Never opened on this path, which is the point: a dry run must not cost a
            // multi-hundred-megabyte read. An absent directory proves it was not opened.
            &temp.path().join("absent-model-dir"),
            &output,
            Some(29),
            Some("cpu"),
            false,
            true,
            true,
        )
        .expect("a dry run over real Phase 2 artifacts succeeds");

        assert!(!output.exists(), "a dry run writes no artifact");
        assert_eq!(
            listing(temp.path()),
            before,
            "a dry run writes NOTHING — not the artifact, not a temp file, not a log"
        );
    }

    // --------------------------------------------------------------------------------
    // End to end over REAL Phase 2 artifacts (OPS-02's train leg, tier3 weight)
    //
    // These are `#[ignore]`d because they build a whole benchmark directory and a whole
    // selection before they start. 04-10 runs them as their OWN invocation —
    //
    //     cargo test -p apr-cli --features setfit --lib setfit_train -- --ignored
    //
    // — because libtest takes ONE positional filter, so this cannot be combined with
    // another filter in a single command.
    //
    // # What they can and cannot assert today
    //
    // They drive `run` through every stage the shipped libraries can serve WITH THE
    // CONFORMANCE SLICE as `--model-dir`, and stop at the first one that directory cannot
    // pass. 04-06 recorded TWO independent, measured, phase-level blockers between this
    // command and a written artifact. **Both are now closed.**
    //
    // 1. ~~No encoder can both pass the calibration gate AND carry an artifact~~
    //    (orchestrator note F-10, measured by 04-05). Before Phase 5's 05-03 calibration
    //    edit (commit a63bb130b), `tune_encoder` judged
    //    `encoder.architecture_fingerprint()` against a calibrated set whose only entry
    //    was the phase-3 slice — and the slice cannot compute two of the artifact's
    //    contract-resident probes (a 97-row vocab closure; 64 position rows against a
    //    256-token probe) — while the production pin computed every probe and returned
    //    `UncalibratedRegime`. CLOSED: a63bb130b added a second, MEASURED regime entry for
    //    the production checkout, additively, leaving the fixture entry byte-untouched.
    //    `apr setfit train --model-dir <production checkout>` now exits 0 and writes a real
    //    artifact — measured at 90,777,156 bytes for the s8/seed-13 cell by 05-07's spawned
    //    ladder (`crates/apr-cli/tests/setfit_cli_lifecycle.rs`,
    //    `setfit_cli_production_chain_completes_after_the_calibration_edit`).
    // 2. ~~The verified artifact's bytes are not reachable out-of-crate.~~ CLOSED by
    //    04-17 G1: `SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes`. The
    //    second test below now asserts the door LANDED and is called exactly once, which
    //    is the successor of the assertion that used to name the gap.
    //
    // What still stops the test below short of an artifact is neither of those: it is the
    // `--model-dir` it passes. The conformance slice is a FIXTURE, not a
    // `from_pretrained_dir` checkout (no `config.json`), and the production checkout is an
    // 86.7 MB offline prerequisite that a `--lib` suite may not require. So these tests
    // assert the stages that DO run and the typed refusal at the first that does not; the
    // positive end-to-end rungs live in the spawned ladder named above, where the checkout
    // can be env-gated.
    // --------------------------------------------------------------------------------

    /// The in-repo conformance slice, which is a real directory and is NOT a pinned
    /// checkout. Resolved from this crate's manifest rather than from the process's
    /// working directory, which `cargo test` does not guarantee.
    fn slice_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../aprender-core/tests/fixtures/setfit")
    }

    /// A complete, accepted `--config` for `apr setfit train`.
    ///
    /// Twelve knobs with no defaults means the first question anyone running this
    /// command hits is "what does a valid config even look like" — this is that
    /// answer, and it is the exact file used to train a real `setfit-apr-v1`
    /// end-to-end (seed 17, 8 shots, TweetEval stance).
    fn reference_config_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/setfit/train-config.json")
    }

    /// The reference config must survive the PRODUCTION door, not a re-parse.
    ///
    /// `parse_config` is what `--config` actually calls, and for this type
    /// deserialization IS validation: the wire form routes through
    /// `SetFitTrainConfig::new`, so an unknown key or an out-of-range knob is
    /// rejected here rather than surfacing mid-training. A fixture nothing
    /// executes rots into a plausible-looking file that no longer loads; this
    /// test is what stops that. It caught `max_length: 128` on first authoring —
    /// the pinned tokenizer truncates at 256 and takes no max-length parameter
    /// (contract setfit-train-lifecycle-v1, TRN-02).
    #[test]
    fn reference_train_config_fixture_is_accepted_by_the_real_parser() {
        let path = reference_config_path();
        assert!(
            path.exists(),
            "reference config fixture missing at {}",
            path.display()
        );
        let config = parse_config(&path).unwrap_or_else(|error| {
            panic!(
                "reference config fixture must parse and validate through parse_config, \
                 but was rejected: {error}"
            )
        });
        // Spot-check the knobs a reader is most likely to copy wrong. `max_length`
        // is the one the pinned tokenizer constrains; `root_seed` must be a
        // contracted benchmark seed so the fixture doubles as a runnable example.
        assert_eq!(config.max_length() as usize, MAX_SEQUENCE_LENGTH);
        assert_eq!(config.root_seed(), 17);
    }

    #[test]
    #[ignore = "integration weight: builds a real benchmark directory and selection. Run as \
                its own invocation — `cargo test -p apr-cli --features setfit --lib \
                setfit_train -- --ignored` (04-10's setfit-cli-tests leg). It stops short of \
                a written artifact because of the `--model-dir` it passes: the conformance \
                SLICE is a fixture with no config.json, not a from_pretrained_dir checkout. \
                That is a property of the fixture, not of this command — the production \
                checkout DOES produce an artifact since 05-03's calibration edit \
                (a63bb130b), proven by the spawned ladder in \
                crates/apr-cli/tests/setfit_cli_lifecycle.rs"]
    fn setfit_train_e2e_clears_every_stage_up_to_the_encoder_over_real_phase_two_artifacts() {
        let temp = TempDir::new().expect("tempdir");
        let (data, selection) = phase2_artifacts(temp.path());
        let config = config_file(temp.path(), "train.toml");
        let output = temp.path().join("model.apr");
        let model_dir = slice_fixture_dir();
        assert!(
            model_dir.is_dir(),
            "the conformance slice fixture must exist at {} — if the fixture estate moved, \
             this test should be updated rather than deleted",
            model_dir.display()
        );
        let before = listing(temp.path());

        let error = run(
            &config,
            &data,
            &selection,
            &model_dir,
            &output,
            Some(29),
            Some("cpu"),
            false,
            false,
            true,
        )
        .expect_err(
            "a training run cannot complete today — see the two blockers in this section's \
             header. If this line starts returning Ok, one of them has been fixed and the \
             assertions below are what should be rewritten",
        );

        // The refusal is at the ENCODER door, which is the evidence that everything before
        // it ran: the config parsed and merged, `--device cpu` resolved, `--output` was
        // clear, the benchmark directory passed the attested boundary, and the selection
        // manifest replayed strictly against it. Any of those failing would have produced
        // a DIFFERENT error, which is what makes this a stage assertion and not a smoke
        // test.
        let rendered = error.to_string();
        assert!(
            matches!(error, CliError::ModelLoadFailed(_)),
            "the first unservable stage is the encoder load; got: {error}"
        );
        assert!(
            rendered.contains("--model-dir"),
            "the refusal must name the flag the operator has to change; got: {rendered}"
        );
        assert!(
            rendered.contains("NEVER downloads"),
            "and it must state the offline prerequisite (A5), because the obvious next \
             assumption is that the command will fetch the pin itself; got: {rendered}"
        );
        // The library refused on `config.json`, which is AFTER the tokenizer load — so the
        // slice's tokenizer.json satisfied the pinned-digest check and the gap really is
        // the encoder weights, not a near-miss tokenizer. Pinning the observed file name
        // is what turns "it failed somewhere in the loader" into a stage measurement.
        assert!(
            rendered.contains("config.json"),
            "the refusal must name the first missing pin file — observed at authoring time: \
             SetFitError::ImportIo(config.json: No such file or directory); got: {rendered}"
        );
        assert_eq!(
            listing(temp.path()),
            before,
            "a run that failed after ingest must leave the working tree exactly as it \
             found it — no artifact, no temp file"
        );
    }

    /// Why the e2e above stops short of an artifact, plus the proof that 04-17's door landed.
    ///
    /// # Un-ignored by 04-17, and it was never the heavy one
    ///
    /// It was `#[ignore]`d only because it was authored beside the e2e above and inherited
    /// its reason. It builds no benchmark directory and no selection: it stats three files
    /// and scans this module's own source, so running it in the default invocation costs
    /// nothing and buys a gate on both halves.
    ///
    /// # What changed, and what deliberately did not
    ///
    /// The FIRST half is what 04-06 wrote, with its CAUSE corrected by 05-07. 04-06 read the
    /// slice's three fixture files as evidence of F-10 — "no encoder both passes
    /// CALIBRATED_REGIMES and computes the probes". Since Phase 5's 05-03 calibration edit
    /// (commit `a63bb130b`) that reading is false: the production checkout passes both. What
    /// the three files still prove, and all they ever proved from HERE, is that this
    /// directory is a SLICE — which is why the e2e above stops. Not one assertion moved; only
    /// the sentence saying what they mean.
    ///
    /// The SECOND half was `ARTIFACT_BYTES_GAP.contains("into_artifact_bytes")` — "the gap
    /// must keep naming the exact door that closes it". That door landed, so the assertion's
    /// successor is that the door is CALLED, exactly once, and that no refusal constant
    /// survived it. The claim was not weakened to make the test green; it advanced to the
    /// next thing that can go wrong.
    #[test]
    fn setfit_train_e2e_records_the_blocker_that_stops_short_of_an_artifact() {
        // THE SLICE, asserted structurally rather than described. The slice fixture is a
        // real MiniLM slice — the pinned tokenizer plus a carved-down encoder — and the
        // two files below are what make it a SLICE and not a pin. `from_pretrained_dir`
        // needs the pin; `from_slice_fixture` reads these, and is `conformance-fixtures`
        // gated and unavailable to a shipped CLI.
        let dir = slice_fixture_dir();
        assert!(
            dir.join("tokenizer.json").is_file(),
            "the slice carries the PINNED tokenizer, which is why the failure above is \
             about the encoder and not about the tokenizer"
        );
        assert!(
            dir.join("slice_config.json").is_file() && dir.join("vocab_remap.json").is_file(),
            "and these two are what make it a slice: a 97-row vocabulary closure and the \
             reduced dimensions. If they ever disappear the fixture has become something \
             else and this whole section needs re-measuring"
        );
        assert!(
            !dir.join("model.safetensors").is_file(),
            "there is deliberately no pinned encoder weight file in the repository — the \
             90 MB pin is an offline prerequisite, not a committed artifact"
        );

        // BLOCKER 2 IS CLOSED (04-17 G1), and this is what keeps it closed.
        //
        // The library door is called EXACTLY ONCE. More than one call site would mean more
        // than one path from a verified run to a file, and the phase's whole claim — the
        // served model is the evaluated model — rests on there being one.
        let door = needle(&["verified.into_artifact_", "bytes()"]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&door).count(),
            1,
            "exactly one call to the library's bytes door, so there is exactly one path \
             from a verified run to a file on disk"
        );

        // And it is what gets written: the write takes the bytes that call returned, so a
        // future edit that re-serialized instead would have to change this line too.
        let write_site = needle(&["atomic_write(output_path, &", "bytes, force)?"]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&write_site).count(),
            1,
            "and those bytes are the ones written"
        );

        // No refusal constant survived. A stale gap message left in the source would tell
        // a later reader the door is still missing, which is now false.
        let stale_gap = needle(&["ARTIFACT_BYTES_", "GAP:"]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&stale_gap).count(),
            0,
            "the gap constant must be gone, not merely unused"
        );
    }

    #[test]
    fn setfit_train_the_writer_and_the_bounded_reader_compose_and_a_forced_rewrite_is_identical() {
        // The write half of OPS-02's train leg, exercised on the only bytes available
        // today. It is NOT the artifact-level determinism witness the plan asked for —
        // that needs a real artifact, which this `--lib` suite has no `--model-dir` for —
        // but it does prove
        // the two halves of this CLI's artifact file I/O compose: what `atomic_write`
        // lands is byte-for-byte what `setfit_io`'s bounded door reads back, and a
        // second `--force` run over the same input produces the same file rather than a
        // file that merely parses the same.
        let temp = TempDir::new().expect("tempdir");
        let target = temp.path().join("model.apr");
        let payload: Vec<u8> = (0..=255_u8).cycle().take(9001).collect();

        atomic_write(&target, &payload, false).expect("the first write lands");
        let first = crate::setfit_io::read_setfit_apr_file_bounded(&target)
            .expect("the bounded door reads what the writer wrote");
        assert_eq!(first, payload, "write then read is the identity");

        atomic_write(&target, &payload, true).expect("a --force rewrite lands");
        let second =
            crate::setfit_io::read_setfit_apr_file_bounded(&target).expect("and reads back");
        assert_eq!(
            first, second,
            "a second identical run with --force produces a byte-identical file"
        );
    }

    // --------------------------------------------------------------------------------
    // Source assertions: the shape the review findings require
    // --------------------------------------------------------------------------------

    #[test]
    fn setfit_train_merges_through_the_public_door_and_names_no_wire_type() {
        let door = needle(&["config.to_", "request()"]);
        assert!(
            SETFIT_TRAIN_SOURCE.contains(&door),
            "the override merge must go through the public read-back door; the private wire \
             struct this plan was originally written against cannot be named from apr-cli at \
             all, which is the finding 04-14 closed"
        );

        let wire = needle(&["Wire", " {"]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&wire).count(),
            0,
            "no wire type may be constructed here"
        );
        let wire_path = needle(&["Config", "Wire"]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&wire_path).count(),
            0,
            "and none may be named here either"
        );
    }

    #[test]
    fn setfit_train_has_one_rename_site_and_creates_files_nowhere_else() {
        let rename = needle(&["fs::", "rename("]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&rename).count(),
            1,
            "exactly ONE rename site, so the three write-safety properties are proven once \
             rather than once per caller (Phase 2 plan 02-09's discipline)"
        );

        let create = needle(&["File::", "create("]);
        assert_eq!(
            SETFIT_TRAIN_SOURCE.matches(&create).count(),
            0,
            "and no unconditional file creation outside the atomic helper, which opens with \
             create_new so a leftover scratch file is diagnosed rather than reused"
        );
    }
}
