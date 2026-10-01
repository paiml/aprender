//! `apr data select` and `apr data pairs` — the user surface of the deterministic
//! contrastive data protocol.
//!
//! # This file is a FILESYSTEM ADAPTER and nothing else (D-05)
//!
//! It reads bytes, hands them to `aprender-contrastive-data`, and writes the bytes the
//! crate hands back. No parsing, hashing, selection, sampling, budget resolution or
//! manifest composition happens here — all of it lives behind the crate's public API,
//! which is bytes-in / typed-values-out by construction (D-04, enforced by
//! `make contrastive-data-boundary`).
//!
//! # Neither command touches the network
//!
//! There is no `--offline` parameter on either entry point. The crate opens no socket and
//! this module only reads and writes local files, so an offline switch would advertise a
//! capability that does not exist.
//!
//! # Every artifact goes through ONE writer
//!
//! `atomic_write` is the only function here that creates a file: temp file in the
//! DESTINATION directory, `write_all`, `sync_all`, `rename`, no-clobber unless `--force`,
//! and temp cleanup on every error path. `apr data pairs --dump` calls the same helper, so
//! the three write-safety properties are proven once instead of once per artifact.
//!
//! # `run_pairs`
//!
//! Implemented by plan 02-09 Task 3 at the signature Task 1 declared: bounded pair
//! generation with strict replay and a streaming `--dump`. Every error path returns a
//! structured `CliError` rather than panicking — the std placeholder macros abort the
//! process and are banned by repo policy, so no code path here may reintroduce them.

use crate::commands::data_tweeteval;
use crate::error::{CliError, Result};
use crate::output;
use aprender_contrastive_data::attestation::DatasetAttestation;
use aprender_contrastive_data::dedup::ExclusionRecord;
use aprender_contrastive_data::error::ContrastiveDataError;
use aprender_contrastive_data::hash::hex;
use aprender_contrastive_data::ledger::{AccessLedger, AccessRecord};
use aprender_contrastive_data::manifest::{
    dump_pairs, pair_manifest_hash, PairReplayRecord, SelectedExampleRecord, SelectionManifest,
};
use aprender_contrastive_data::pairs::{EmittedKinds, PairConfig, PairSampler};
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};
use aprender_contrastive_data::split::{CompatibilityTest, SplitRole, Test, Train, Validation};
use colored::Colorize;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The file `apr data select` writes and `apr data pairs` reads.
pub(crate) const SELECTION_MANIFEST_FILE: &str = "selection-manifest.json";

/// Wire name for a seed drawn from the ten contracted benchmark seeds.
const SEED_MODE_CONTRACTED: &str = "contracted";

/// Wire name for a seed accepted only because `--any-seed` was passed.
const SEED_MODE_UNCONTRACTED: &str = "uncontracted";

/// What to do when a benchmark directory is missing or unreadable. An error that names a
/// missing file without naming the command that produces it is diagnosable but not
/// actionable.
const PREPARE_REMEDY: &str = "Prepare one with \
     `apr data tweet-eval-stance --output <DIR>` (canonical profile), then point --data at \
     that directory.";

// ==========================================================================================
// Test-only fault-injection seam
// ==========================================================================================

// When set, `atomic_write` fails AFTER the temp file is written and synced but BEFORE the
// rename — the one window in which a partial artifact could exist.
//
// A `thread_local` rather than a global: cargo runs each test on its own thread, so two
// tests cannot see each other's injection.
//
// Plain `//` rather than `///`: rustdoc generates nothing for a macro invocation, and
// `-D warnings` rejects the doc form (`unused_doc_comments`).
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
// The shared atomic writer — the only thing in this module that creates a file
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
///
/// `create_new` on the TEMP as well: two concurrent runs must not share one scratch file,
/// and a leftover scratch file from a crashed run is a diagnosable error rather than
/// silent reuse.
fn fill_and_sync<F>(temp: &Path, fill: F) -> Result<()>
where
    F: FnOnce(&mut fs::File) -> Result<()>,
{
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp)?;
    fill(&mut file)?;
    file.sync_all()?;
    if induced_prerename_failure() {
        return Err(CliError::Io(std::io::Error::other(
            "induced pre-rename failure (test seam)",
        )));
    }
    Ok(())
}

/// Produce `target` atomically from a streaming writer, with no-clobber by default.
///
/// The temp file lives in the destination directory because `rename` is only atomic within
/// one filesystem; a temp in `/tmp` would silently degrade to a copy across a mount point,
/// which is exactly the partial-write window this exists to close.
///
/// Every failure path removes the temp, so an interrupted write leaves neither a partial
/// artifact nor a stray file for the next `--force`-less run to trip over.
///
/// `fill` takes the open file rather than a byte slice so `--dump` can stream a
/// million-pair audit dump through `dump_pairs` without ever holding it in memory.
/// Accumulating the WHOLE dump before writing would reintroduce the `O(budget)` allocation
/// the protocol exists to avoid, in the one command whose job is to demonstrate its absence.
/// A fixed-size `BufWriter` is a different thing and is used at the `--dump` call site: its
/// buffer does not grow with the budget, so it coalesces syscalls without retaining pairs.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] when `target` exists and `force` is false;
/// [`CliError::Io`] for any create, write, sync or rename failure; whatever `fill` raises.
fn atomic_write_with<F>(target: &Path, force: bool, fill: F) -> Result<()>
where
    F: FnOnce(&mut fs::File) -> Result<()>,
{
    if !force && target.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Refusing to replace existing file {} (pass --force to replace it)",
            target.display()
        )));
    }
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;

    let temp = temp_path(target);
    let result =
        fill_and_sync(&temp, fill).and_then(|()| fs::rename(&temp, target).map_err(CliError::Io));
    if result.is_err() {
        // Best-effort: the write has already failed, and a cleanup failure must not
        // replace the reason it failed.
        let _ = fs::remove_file(&temp);
    }
    result
}

/// The byte form of [`atomic_write_with`], for artifacts small enough to already be in
/// memory (every manifest is).
///
/// # Errors
///
/// The same set as [`atomic_write_with`].
fn atomic_write(target: &Path, bytes: &[u8], force: bool) -> Result<()> {
    atomic_write_with(target, force, |file| {
        file.write_all(bytes).map_err(CliError::Io)
    })
}

// ==========================================================================================
// Attested ingest — the ONLY door to a canonical dataset (D-16 / review finding F16)
// ==========================================================================================

/// Map a crate failure onto the CLI surface without losing detail. The crate's messages
/// already name the split, the field, and both the expected and observed value.
fn dataset_error(error: &ContrastiveDataError) -> CliError {
    CliError::ValidationFailed(format!("contrastive data: {error}"))
}

/// Read one required file, with a message that says what to do about it.
fn read_required(path: &Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::ValidationFailed(format!("{} not found. {PREPARE_REMEDY}", path.display()))
        } else {
            CliError::Io(error)
        }
    })
}

/// Split ROLE -> the file that holds it inside a prepared benchmark directory.
///
/// Canonical roles are `{role}.jsonl`; the compatibility profile writes its
/// `compatibility_test` role to `test.jsonl`, which is the filename the SetFit wrapper
/// expects. Handling that case here is what lets a compatibility directory be read far
/// enough to be refused by PROFILE — "no such file: compatibility_test.jsonl" would be a
/// true statement about the wrong problem.
///
/// # An ALLOWLIST, not a sanitizer
///
/// The role reaching this function comes from `dataset_attestation.splits`, which is
/// attacker-controlled bytes until the crate has verified them — and the caller joins the
/// result onto `--data`. `Path::join` REPLACES the base when handed an absolute component,
/// so a manifest declaring `"role": "/etc/passwd"` would have read `/etc/passwd.jsonl`, and
/// `"../../.."` would have escaped the directory: a file-existence oracle, and an unbounded
/// read if the target is a FIFO.
///
/// Enumerating the four legal roles is what closes that, rather than stripping bad
/// characters. A sanitizer has to anticipate every encoding of "leave this directory"; an
/// allowlist only has to know the roles this protocol defines, and it fails closed on
/// anything it has never heard of.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] naming the role, when it is not one of the four defined
/// by the split protocol.
fn role_file(role: &str) -> Result<String> {
    if role == CompatibilityTest::ROLE {
        return Ok("test.jsonl".to_string());
    }
    if role == Train::ROLE || role == Validation::ROLE || role == Test::ROLE {
        return Ok(format!("{role}.jsonl"));
    }
    Err(CliError::ValidationFailed(format!(
        "attestation declares split role {role:?}, which is not one of \
         {:?}, {:?}, {:?}, {:?}. Refusing to derive a path from it.",
        Train::ROLE,
        Validation::ROLE,
        Test::ROLE,
        CompatibilityTest::ROLE,
    )))
}

/// Open a prepared benchmark directory through the crate's attested boundary.
///
/// The CLI supplies bytes and nothing else. Which files to read comes from the
/// attestation's own split roles rather than from a hardcoded canonical triple, so a
/// compatibility, mixed, stale or forged directory reaches the boundary and is refused
/// there by profile, schema version, per-split digest, class counts, exclusion digest or
/// fingerprint — instead of dying earlier on a missing filename.
///
/// Only the CANONICAL constructor is ever called. A compatibility attestation is
/// `ProfileMismatch`, and the type system independently forbids a
/// `PreparedDataset<Compatibility>` from reaching selection at all (D-19).
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a missing/unreadable file, a manifest this build
/// cannot read, or any attested-boundary rejection.
///
/// # `pub(crate)` so `apr setfit train` reads through THIS door (04-06)
///
/// The alternative was a second ingest sequence in `commands/setfit_train.rs`, which
/// would have made two readers of `benchmark-manifest.json` — the exact defect the
/// comment inside this function exists to prevent, and the one that would let a
/// training run accept a directory `apr data select` refuses. Widening the visibility
/// keeps the attested boundary a single implementation with a single set of
/// rejections.
pub(crate) fn read_attested_canonical(
    data_dir: &Path,
    ledger: &mut AccessLedger,
) -> Result<PreparedDataset<Canonical>> {
    let manifest_bytes = read_required(&data_dir.join(data_tweeteval::MANIFEST_FILE))?;
    // The schema-version gate and the attestation extraction belong to `data_tweeteval`,
    // so there is ONE reader of `benchmark-manifest.json` rather than two that can drift.
    let attestation_bytes = data_tweeteval::attestation_bytes_from_manifest(&manifest_bytes)?;
    let attestation =
        DatasetAttestation::from_bytes(&attestation_bytes).map_err(|e| dataset_error(&e))?;

    let mut buffers = BTreeMap::new();
    for role in attestation.splits.keys() {
        buffers.insert(
            role.clone(),
            read_required(&data_dir.join(role_file(role)?))?,
        );
    }
    PreparedDataset::<Canonical>::from_attested_bytes(&attestation_bytes, &buffers, ledger)
        .map_err(|e| dataset_error(&e))
}

// ==========================================================================================
// Request pre-flight: fail on a bad REQUEST before blaming the data
// ==========================================================================================

/// The contracted few-shot sizes, rendered for an error message.
fn allowed_shots_text() -> String {
    let sizes: Vec<String> = data_tweeteval::FEW_SHOT_SIZES
        .iter()
        .map(usize::to_string)
        .collect();
    format!("{{{}}}", sizes.join(", "))
}

/// Reject a shot count outside the contracted set, BEFORE anything is read.
///
/// The list is `data_tweeteval::FEW_SHOT_SIZES` — the same array written into every
/// benchmark manifest's `few_shot` section — not a third copy of the four literals. The
/// crate re-validates independently inside `FewShotSelector::select`, so if the two ever
/// disagreed the result would be a typed `InvalidShots` error, never an off-protocol
/// selection.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] naming the offending value and the allowed set.
fn validate_shots(shots: u32) -> Result<()> {
    // `usize::try_from` rather than a cast: on a 16-bit target a `u32` shot count does not
    // fit, and such a value is not in the list anyway.
    if usize::try_from(shots).is_ok_and(|shots| data_tweeteval::FEW_SHOT_SIZES.contains(&shots)) {
        return Ok(());
    }
    Err(CliError::ValidationFailed(format!(
        "--shots {shots} is not a contracted few-shot size; expected one of {}",
        allowed_shots_text()
    )))
}

/// Resolve the seed mode, refusing an uncontracted seed unless `--any-seed` was passed.
///
/// The ten seeds are `data_tweeteval::BENCHMARK_SEEDS`
/// (`crates/apr-cli/src/commands/data_tweeteval.rs`), the same array written into every
/// benchmark manifest's `few_shot.seeds`. **42 is not among them**, which is why `--seed`
/// has no default.
///
/// The returned mode is REPORTED, not stored as a separate manifest field: the manifest
/// records `root_seed`, and the mode is a total function of it. A parallel `seed_mode`
/// field could disagree with the seed printed beside it, which is strictly worse than
/// deriving it.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] naming the seed, the ten contracted seeds and the
/// `--any-seed` escape hatch.
fn resolve_seed_mode(seed: u64, any_seed: bool) -> Result<&'static str> {
    if data_tweeteval::BENCHMARK_SEEDS.contains(&seed) {
        return Ok(SEED_MODE_CONTRACTED);
    }
    if any_seed {
        return Ok(SEED_MODE_UNCONTRACTED);
    }
    let seeds: Vec<String> = data_tweeteval::BENCHMARK_SEEDS
        .iter()
        .map(u64::to_string)
        .collect();
    Err(CliError::ValidationFailed(format!(
        "--seed {seed} is not one of the ten contracted benchmark seeds [{}]. Pass \
         --any-seed to draw an experimental selection anyway; the seed itself is recorded \
         in the manifest, so a reader can always tell which of the two you did",
        seeds.join(", ")
    )))
}

// ==========================================================================================
// Machine-readable reports
// ==========================================================================================

/// `apr data select --json`.
///
/// Every field is read off the manifest the crate produced; the CLI adds only the two
/// paths and the derived seed mode.
#[derive(Serialize)]
struct SelectReport<'a> {
    command: &'static str,
    data: String,
    manifest: String,
    profile: &'a str,
    root_seed: u64,
    /// Derived from `root_seed`, not stored beside it — see `seed_mode_of`.
    seed_mode: &'static str,
    shots_per_class: u32,
    selected: usize,
    semantic_hash: &'a str,
    ledger_hash: &'a str,
    ordered_examples: &'a [SelectedExampleRecord],
    exclusions: &'a ExclusionRecord,
    access_ledger: &'a [AccessRecord],
}

/// Render the `--json` report for a written selection manifest.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] if the report cannot be serialized.
fn select_report_json(
    data_dir: &Path,
    manifest_path: &Path,
    manifest: &SelectionManifest,
    seed_mode: &'static str,
) -> Result<String> {
    let payload = &manifest.payload;
    let report = SelectReport {
        command: "data-select",
        data: data_dir.display().to_string(),
        manifest: manifest_path.display().to_string(),
        profile: &payload.profile,
        root_seed: payload.root_seed,
        seed_mode,
        shots_per_class: payload.shots_per_class,
        selected: payload.ordered_examples.len(),
        semantic_hash: &manifest.semantic_hash,
        ledger_hash: &payload.ledger_hash,
        ordered_examples: &payload.ordered_examples,
        exclusions: &payload.exclusions,
        access_ledger: &payload.access_ledger,
    };
    serde_json::to_string_pretty(&report).map_err(|error| {
        CliError::ValidationFailed(format!("Failed to encode the selection report: {error}"))
    })
}

/// Human-readable `apr data select` output.
fn render_select_human(
    data_dir: &Path,
    manifest_path: &Path,
    manifest: &SelectionManifest,
    seed_mode: &str,
) {
    let payload = &manifest.payload;
    output::section("Few-Shot Selection");
    println!();
    output::kv("Data", data_dir.display());
    output::kv("Profile", &payload.profile);
    output::kv("Shots per class", payload.shots_per_class);
    output::kv("Root seed", format!("{} ({seed_mode})", payload.root_seed));
    output::kv(
        "Selected",
        format!(
            "{} example(s) across {} class(es)",
            payload.ordered_examples.len(),
            payload.label_names.len()
        ),
    );
    output::kv("Semantic hash", &manifest.semantic_hash);
    output::kv("Ledger hash", &payload.ledger_hash);
    output::kv(
        "Excluded",
        format!(
            "{} training row(s) removed from the pool before selection",
            payload.exclusions.excluded_train_ids().len()
        ),
    );
    output::kv("Manifest", manifest_path.display());
    println!();
    println!(
        "{} Selection manifest written. Replay it with:",
        "OK".green()
    );
    println!(
        "  apr data pairs --selection {} --data {}",
        manifest_path.display(),
        data_dir.display()
    );
}

// ==========================================================================================
// Commands
// ==========================================================================================

/// Select `shots` examples per class from an attested canonical benchmark directory and
/// write the replayable selection manifest.
///
/// The whole body is adapter work: pre-flight the REQUEST, read bytes, call the crate,
/// write the crate's bytes back verbatim, report. Nothing here decides which rows are
/// selected or what the manifest looks like.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for an invalid shot count, an uncontracted seed without
/// `--any-seed`, any attested-ingest rejection, or an existing output file without
/// `--force`; [`CliError::Io`] for a read or write failure.
pub(crate) fn run_select(
    data: &Path,
    shots: u32,
    seed: u64,
    any_seed: bool,
    output: Option<&Path>,
    force: bool,
    json_output: bool,
) -> Result<()> {
    // Fail-closed on the REQUEST first. A bad shot count or an off-protocol seed is not a
    // problem with the data, and reporting it as one sends the user to the wrong place.
    validate_shots(shots)?;
    let seed_mode = resolve_seed_mode(seed, any_seed)?;

    let mut ledger = AccessLedger::new();
    let dataset = read_attested_canonical(data, &mut ledger)?;

    let cfg = SelectionConfig {
        root_seed: seed,
        shots_per_class: shots,
    };
    let selection =
        FewShotSelector::select(&dataset, &cfg, &mut ledger).map_err(|e| dataset_error(&e))?;
    // NOTHING may touch the ledger between these two calls: `from_selection` refuses a
    // ledger that has grown since `select` returned, because the payload's embedded
    // records would no longer describe it.
    let mut manifest =
        SelectionManifest::from_selection(&selection, &ledger).map_err(|e| dataset_error(&e))?;
    // The crate reads no clock for the same reason it opens no file. The field is outside
    // the hashed region, so filling it changes no digest.
    manifest.volatile.created_at = chrono::Utc::now().to_rfc3339();

    let manifest_path = output.unwrap_or(data).join(SELECTION_MANIFEST_FILE);
    let bytes = manifest.to_file_bytes().map_err(|e| dataset_error(&e))?;
    atomic_write(&manifest_path, &bytes, force)?;

    if json_output {
        println!(
            "{}",
            select_report_json(data, &manifest_path, &manifest, seed_mode)?
        );
    } else {
        render_select_human(data, &manifest_path, &manifest, seed_mode);
    }
    Ok(())
}

// ==========================================================================================
// apr data pairs
// ==========================================================================================

/// What one `apr data pairs` run produced. Small by construction: the pair STREAM is never
/// held, only summarized.
struct PairsOutcome {
    record: PairReplayRecord,
    manifest_hash: String,
    hard_cap: u64,
    positives: u64,
    negatives: u64,
}

/// `apr data pairs --json`.
#[derive(Serialize)]
struct PairsReport<'a> {
    command: &'static str,
    selection: String,
    data: String,
    selection_hash: &'a str,
    pair_manifest_hash: &'a str,
    root_seed: u64,
    strategy: &'a str,
    strategy_version: u32,
    budget: u64,
    hard_cap: u64,
    default_was_clamped: bool,
    emitted_kinds: &'a str,
    positives: u64,
    negatives: u64,
    singleton_policy: &'a str,
    singleton_policy_version: u32,
    degenerate_policy_version: u32,
    affected_singleton_classes: u64,
    /// The three declared deviation clauses, verbatim from the contract via the crate.
    deviation: &'a [String; 3],
    dump: Option<String>,
}

/// Map a pair-configuration failure, adding the FLAG the user has to change.
///
/// The crate cannot name a CLI flag, and an error that states a constraint without naming
/// the knob that satisfies it is diagnosable but not actionable. The constraint itself is
/// still entirely the crate's: `resolve_budget` decides, this only translates.
fn pair_config_error(error: &ContrastiveDataError) -> CliError {
    let remedy = match error {
        ContrastiveDataError::BudgetExceedsHardCap { .. } => {
            " — raise --hard-cap or lower --budget. The cap BINDS an explicit budget rather \
             than clamping it, so that a run can never quietly emit fewer pairs than asked for"
        }
        ContrastiveDataError::ZeroBudget => {
            " — pass --budget <N> with N >= 1, or omit --budget for the contracted default"
        }
        ContrastiveDataError::ZeroHardCap => {
            " — pass --hard-cap <N> with N >= 1, or omit --hard-cap for the contracted default"
        }
        _ => "",
    };
    CliError::ValidationFailed(format!("contrastive data: {error}{remedy}"))
}

/// Read the selection manifest, verifying its envelope digest before it can be used.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] when the file is absent or its digest disagrees with its
/// payload; [`CliError::Io`] for any other read failure.
///
/// `pub(crate)` for the same reason as [`read_attested_canonical`]: `apr setfit train`
/// consumes the manifest `apr data select` wrote, and a second reader of that file is
/// a second place for the envelope digest check to be omitted.
pub(crate) fn read_selection_manifest(path: &Path) -> Result<SelectionManifest> {
    let bytes = fs::read(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::ValidationFailed(format!(
                "{} not found. Write one with `apr data select --data <DIR> --shots <N> \
                 --seed <SEED>`.",
                path.display()
            ))
        } else {
            CliError::Io(error)
        }
    })?;
    SelectionManifest::from_bytes(&bytes).map_err(|e| dataset_error(&e))
}

/// The two pair-kind counts, from the layout rather than from a third pass over the stream.
///
/// `emitted_kinds` plus the resolved budget determine the split exactly: the interleave is
/// `2t -> positive`, `2t + 1 -> negative` in the `both` case, and every ordinal is one kind
/// in the two degenerate cases. A positive draw puts both endpoints in ONE class and a
/// negative draw's second endpoint skips the first endpoint's contiguous block, so the
/// derived `1.0`/`0.0` target is the kind — counting it by streaming re-derived the whole
/// pair space to learn `ceil(budget/2)`. At the default hard cap that was 1,048,576 wasted
/// Philox draws on top of the passes `pair_manifest_hash` and `--dump` already make.
///
/// `EmittedKinds` is `#[non_exhaustive]`, so a variant this build does not know falls back
/// to the streaming count rather than to a guess.
///
/// # Errors
///
/// Whatever `pair_at` raises on the fallback path. It cannot raise for an ordinal below the
/// resolved budget, but the error is propagated rather than asserted away.
fn count_kinds(sampler: &PairSampler<'_>) -> Result<(u64, u64)> {
    let budget = sampler.budget();
    match sampler.layout().emitted_kinds() {
        EmittedKinds::Both => Ok((budget.div_ceil(2), budget / 2)),
        EmittedKinds::PositivesOnly => Ok((budget, 0)),
        EmittedKinds::NegativesOnly => Ok((0, budget)),
        _ => count_kinds_by_streaming(sampler),
    }
}

/// The fallback count, used only for an `EmittedKinds` variant this build predates.
///
/// Two `u64`s and nothing else: collecting the stream to count it would reintroduce the
/// `O(budget)` memory in the command whose job is to show it is absent.
fn count_kinds_by_streaming(sampler: &PairSampler<'_>) -> Result<(u64, u64)> {
    let mut positives = 0_u64;
    let mut negatives = 0_u64;
    for ordinal in 0..sampler.budget() {
        let labeled = sampler.pair_at(ordinal).map_err(|e| dataset_error(&e))?;
        // The target is exactly 1.0 or 0.0, derived by the crate from class identity;
        // the midpoint comparison avoids a float equality lint without changing meaning.
        if labeled.target > 0.5 {
            positives += 1;
        } else {
            negatives += 1;
        }
    }
    Ok((positives, negatives))
}

/// Replay the selection strictly, build the bounded stream, and summarize it.
///
/// This is the whole of `apr data pairs` except the reporting, and every step is a crate
/// call: `SelectionManifest::from_bytes` (digest), `from_attested_bytes` (dataset
/// identity), `Selection::replay` (the strict ladder ending in a full recomputation),
/// `PairSampler::new` (which applies `resolve_budget`, so the zero and over-cap cases
/// surface from ONE place), `pair_manifest_hash`, `dump_pairs`.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for any of those rejections or an existing dump file
/// without `--force`; [`CliError::Io`] for a read or write failure.
fn pairs_outcome(
    selection_path: &Path,
    data: &Path,
    budget: Option<u64>,
    hard_cap: Option<u64>,
    dump: Option<&Path>,
    force: bool,
) -> Result<PairsOutcome> {
    let manifest = read_selection_manifest(selection_path)?;
    let mut ledger = AccessLedger::new();
    let dataset = read_attested_canonical(data, &mut ledger)?;
    // STRICT replay is the only route from manifest bytes back to a `Selection`. The CLI
    // never constructs one — it cannot: `Selection::assemble` is crate-private.
    let selection =
        Selection::replay(&manifest, &dataset, &mut ledger).map_err(|e| dataset_error(&e))?;

    let cfg = PairConfig {
        budget,
        hard_cap,
        ..PairConfig::new(selection.root_seed())
    };
    let sampler = PairSampler::new(&selection, &cfg).map_err(|e| pair_config_error(&e))?;

    let record = PairReplayRecord::from_sampler(&sampler);
    let digest = pair_manifest_hash(&sampler, &record).map_err(|e| dataset_error(&e))?;
    let (positives, negatives) = count_kinds(&sampler)?;

    if let Some(path) = dump {
        atomic_write_with(path, force, |file| {
            // A FIXED-SIZE buffer, not an O(budget) one: without it every pair is its own
            // `write(2)`, which is ~24.6k syscalls at the default cell and ~326k across the
            // benchmark grid. `dump_pairs` flushes explicitly before returning, and
            // `fill_and_sync` calls `sync_all` only after this closure returns, so the
            // ordering that makes the write atomic is unchanged.
            let mut buffered = BufWriter::new(&mut *file);
            dump_pairs(&sampler, &mut buffered).map_err(|e| dataset_error(&e))
        })?;
    }

    Ok(PairsOutcome {
        manifest_hash: hex(&digest),
        hard_cap: cfg.resolved_hard_cap(),
        positives,
        negatives,
        record,
    })
}

/// Render the `--json` report for a pair run.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] if the report cannot be serialized.
fn pairs_report_json(
    selection_path: &Path,
    data: &Path,
    outcome: &PairsOutcome,
    dump: Option<&Path>,
) -> Result<String> {
    let record = &outcome.record;
    let report = PairsReport {
        command: "data-pairs",
        selection: selection_path.display().to_string(),
        data: data.display().to_string(),
        selection_hash: &record.selection_hash,
        pair_manifest_hash: &outcome.manifest_hash,
        root_seed: record.root_seed,
        strategy: &record.strategy,
        strategy_version: record.strategy_version,
        budget: outcome.record.budget,
        hard_cap: outcome.hard_cap,
        default_was_clamped: record.default_was_clamped,
        emitted_kinds: &record.emitted_kinds,
        positives: outcome.positives,
        negatives: outcome.negatives,
        singleton_policy: &record.singleton_policy,
        singleton_policy_version: record.singleton_policy_version,
        degenerate_policy_version: record.degenerate_policy_version,
        affected_singleton_classes: record.affected_singleton_classes,
        deviation: &record.deviation,
        dump: dump.map(|path| path.display().to_string()),
    };
    serde_json::to_string_pretty(&report).map_err(|error| {
        CliError::ValidationFailed(format!("Failed to encode the pair report: {error}"))
    })
}

/// Human-readable `apr data pairs` output.
fn render_pairs_human(
    selection_path: &Path,
    data: &Path,
    outcome: &PairsOutcome,
    dump: Option<&Path>,
) {
    let record = &outcome.record;
    output::section("Contrastive Pair Stream");
    println!();
    output::kv("Selection", selection_path.display());
    output::kv("Data", data.display());
    output::kv("Selection hash", &record.selection_hash);
    output::kv("Pair manifest hash", &outcome.manifest_hash);
    output::kv("Root seed", record.root_seed);
    output::kv(
        "Strategy",
        format!("{} (v{})", record.strategy, record.strategy_version),
    );
    output::kv(
        "Budget",
        format!(
            "{} pair(s) per epoch, hard cap {}{}",
            outcome.record.budget,
            outcome.hard_cap,
            if record.default_was_clamped {
                " (the DEFAULT budget was clamped by the cap)"
            } else {
                ""
            }
        ),
    );
    output::kv("Emitted kinds", &record.emitted_kinds);
    output::kv(
        "Counts",
        format!(
            "{} positive, {} negative",
            outcome.positives, outcome.negatives
        ),
    );
    output::kv(
        "Singleton policy",
        format!(
            "{} (v{}), {} affected class(es)",
            record.singleton_policy,
            record.singleton_policy_version,
            record.affected_singleton_classes
        ),
    );
    output::kv(
        "Degenerate policy",
        format!("v{}", record.degenerate_policy_version),
    );
    if let Some(path) = dump {
        output::kv("Dump", path.display());
    }
    println!();
    println!("Declared deviations from the pinned SetFit reference (Aprender policy):");
    for (index, clause) in record.deviation.iter().enumerate() {
        println!("  {}. {clause}", index + 1);
    }
    println!();
    println!(
        "{} Pairs are REPLAYED from this tuple, not stored; only the hash above is persisted.",
        "OK".green()
    );
}

/// Replay a selection manifest against its dataset and report the bounded pair stream.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a manifest whose digest, provenance or ordered list
/// does not survive strict replay, for a budget/hard-cap combination the crate refuses, or
/// for an existing dump file without `--force`; [`CliError::Io`] for a read or write
/// failure.
pub(crate) fn run_pairs(
    selection: &Path,
    data: &Path,
    budget: Option<u64>,
    hard_cap: Option<u64>,
    dump: Option<&Path>,
    force: bool,
    json_output: bool,
) -> Result<()> {
    let outcome = pairs_outcome(selection, data, budget, hard_cap, dump, force)?;
    if json_output {
        println!("{}", pairs_report_json(selection, data, &outcome, dump)?);
    } else {
        render_pairs_human(selection, data, &outcome, dump);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::data_tweeteval::{
        self, fixtures, CANONICAL_REVISION, LABEL_NAMES, MANIFEST_FILE,
    };
    use crate::TweetEvalStanceProfile;
    use aprender_contrastive_data::hash::{exact_hash, hex, normalized_hash};
    use aprender_contrastive_data::manifest::SelectedExampleRecord;
    use aprender_contrastive_data::pairs::{parse_pair_dump, validate_pair_records};
    use aprender_contrastive_data::select::Selection;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // --------------------------------------------------------------------------------
    // Fixtures. Every benchmark directory here is produced by RUNNING
    // `apr data tweet-eval-stance` over a synthetic source tree — nothing fabricates an
    // attestation by hand, because a hand-built one would only prove the reader accepts
    // what this test module thinks the writer emits.
    // --------------------------------------------------------------------------------

    fn prepare(root: &Path, name: &str, tag: &str, profile: TweetEvalStanceProfile) -> PathBuf {
        let source = root.join(format!("{name}-source"));
        fs::create_dir_all(&source).expect("fixture source directory is creatable");
        fixtures::write_canonical_fixture_tagged(&source, tag);
        let out = root.join(name);
        data_tweeteval::run(
            &out,
            profile,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            true,
        )
        .expect("the synthetic canonical fixture prepares cleanly");
        out
    }

    fn canonical_dir(root: &Path) -> PathBuf {
        prepare(
            root,
            "canonical",
            fixtures::DEFAULT_TAG,
            TweetEvalStanceProfile::Canonical,
        )
    }

    fn compatibility_dir(root: &Path) -> PathBuf {
        prepare(
            root,
            "setfit",
            fixtures::DEFAULT_TAG,
            TweetEvalStanceProfile::Setfit,
        )
    }

    /// A directory whose `train.jsonl` comes from one preparation and whose
    /// `validation.jsonl` comes from a DIFFERENT one. Both halves are individually
    /// well-formed; only the attested per-split digest catches it.
    fn mixed_dir(root: &Path) -> PathBuf {
        let first = canonical_dir(root);
        let second = prepare(
            root,
            "second",
            "second preparation",
            TweetEvalStanceProfile::Canonical,
        );
        let foreign = fs::read(second.join("validation.jsonl")).expect("second validation split");
        let mine = fs::read(first.join("validation.jsonl")).expect("first validation split");
        assert_ne!(
            foreign, mine,
            "the two preparations must differ, or the mixed fixture proves nothing"
        );
        fs::write(first.join("validation.jsonl"), foreign).expect("validation split is writable");
        first
    }

    fn edit_manifest(dir: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
        let path = dir.join(MANIFEST_FILE);
        let bytes = fs::read(&path).expect("benchmark manifest is readable");
        let mut value: serde_json::Value =
            serde_json::from_slice(&bytes).expect("benchmark manifest is JSON");
        mutate(&mut value);
        let mut out = serde_json::to_vec_pretty(&value).expect("manifest re-encodes");
        out.push(b'\n');
        fs::write(&path, out).expect("benchmark manifest is writable");
    }

    fn read_manifest(dir: &Path) -> SelectionManifest {
        let bytes = fs::read(dir.join(SELECTION_MANIFEST_FILE)).expect("selection manifest exists");
        SelectionManifest::from_bytes(&bytes).expect("selection manifest parses and verifies")
    }

    fn message(error: &CliError) -> String {
        error.to_string()
    }

    // --------------------------------------------------------------------------------
    // Happy paths
    // --------------------------------------------------------------------------------

    #[test]
    fn select_writes_a_manifest_whose_ordered_examples_carry_ids_and_labels() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());

        run_select(&data, 8, 13, false, None, false, false).expect("select succeeds");

        let manifest = read_manifest(&data);
        assert_eq!(manifest.payload.root_seed, 13);
        assert_eq!(manifest.payload.shots_per_class, 8);
        assert_eq!(manifest.payload.profile, "canonical");
        assert_eq!(
            manifest.payload.ordered_examples.len(),
            8 * LABEL_NAMES.len(),
            "8 shots for each of the three declared classes"
        );
        // Labels, not just ids: a manifest of bare ids would force every later consumer to
        // re-derive the class from the dataset.
        let labels: Vec<usize> = manifest
            .payload
            .ordered_examples
            .iter()
            .map(|row| row.label)
            .collect();
        assert_eq!(labels.first(), Some(&0));
        assert_eq!(labels.last(), Some(&2));
        assert!(
            manifest
                .payload
                .ordered_examples
                .iter()
                .all(|row| row.id.starts_with("train:")),
            "every selected row comes from the training split"
        );
        // The persisted access ledger is the D-19 evidence Phase 5's selection lock reads.
        assert!(
            manifest
                .payload
                .access_ledger
                .iter()
                .any(|record| record.purpose == "select"),
            "the selection's own access record is persisted in the manifest"
        );
        assert!(!manifest.payload.ledger_hash.is_empty());
    }

    #[test]
    fn two_runs_produce_identical_hashed_payloads() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());

        run_select(&data, 8, 13, false, None, false, false).expect("first select");
        let first = read_manifest(&data);
        run_select(&data, 8, 13, false, None, true, false).expect("second select with --force");
        let second = read_manifest(&data);

        assert_eq!(first.semantic_hash, second.semantic_hash);
        assert_eq!(
            first
                .payload
                .to_canonical_bytes()
                .expect("payload serializes"),
            second
                .payload
                .to_canonical_bytes()
                .expect("payload serializes"),
            "the hashed payload is byte-identical across runs"
        );
        assert!(
            !first.volatile.tool_version.is_empty(),
            "the volatile block is populated even though it is never hashed"
        );
    }

    #[test]
    fn the_written_manifest_is_the_verbatim_crate_serialization() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        run_select(&data, 8, 17, false, None, false, false).expect("select succeeds");

        let on_disk = fs::read(data.join(SELECTION_MANIFEST_FILE)).expect("manifest readable");
        let parsed = SelectionManifest::from_bytes(&on_disk).expect("round trip");
        assert_eq!(
            parsed.to_file_bytes().expect("re-serialize"),
            on_disk,
            "the CLI writes to_file_bytes() verbatim and composes no JSON around it"
        );
    }

    #[test]
    fn the_output_directory_can_differ_from_the_data_directory() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        let out = temp.path().join("elsewhere");

        run_select(&data, 8, 13, false, Some(&out), false, false).expect("select succeeds");

        assert!(out.join(SELECTION_MANIFEST_FILE).is_file());
        assert!(
            !data.join(SELECTION_MANIFEST_FILE).exists(),
            "--output redirects the write; it does not duplicate it"
        );
    }

    #[test]
    fn the_json_report_carries_ids_labels_hash_ledger_and_reduced_pools() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        run_select(&data, 8, 13, false, None, false, false).expect("select succeeds");
        let manifest = read_manifest(&data);
        let path = data.join(SELECTION_MANIFEST_FILE);

        let json = select_report_json(&data, &path, &manifest, SEED_MODE_CONTRACTED)
            .expect("report serializes");

        for needle in [
            "\"semantic_hash\"",
            "\"ordered_examples\"",
            "\"label\"",
            "\"access_ledger\"",
            "\"reduced_pools\"",
            "\"excluded_train_ids\"",
            "\"seed_mode\"",
            "\"contracted\"",
            "\"shots_per_class\"",
        ] {
            assert!(
                json.contains(needle),
                "the --json report must carry {needle}"
            );
        }
        assert!(json.contains(&manifest.semantic_hash));
        assert!(json.contains(&manifest.payload.ordered_examples[0].id));
    }

    // --------------------------------------------------------------------------------
    // Rejections
    // --------------------------------------------------------------------------------

    #[test]
    fn nine_shots_is_refused_before_the_filesystem_is_touched() {
        let temp = TempDir::new().expect("tempdir");
        // A directory that does NOT exist. If the shot count were validated after ingest
        // this test would report a missing-manifest error instead, so the assertion below
        // is about ORDER as well as about the message.
        let absent = temp.path().join("no-such-directory");

        let error = run_select(&absent, 9, 13, false, None, false, false)
            .expect_err("nine shots is not a contracted few-shot size");
        let text = message(&error);
        assert!(text.contains("8"), "{text}");
        assert!(text.contains("16"), "{text}");
        assert!(text.contains("32"), "{text}");
        assert!(text.contains("64"), "{text}");
        assert!(text.contains('9'), "the offending value is named: {text}");
        assert!(
            !text.contains("benchmark-manifest"),
            "the shot count is checked BEFORE anything is read: {text}"
        );
    }

    #[test]
    fn an_uncontracted_seed_names_the_ten_seeds_and_the_escape_hatch() {
        let temp = TempDir::new().expect("tempdir");
        let absent = temp.path().join("no-such-directory");

        let error = run_select(&absent, 8, 42, false, None, false, false)
            .expect_err("42 is not one of the ten contracted benchmark seeds");
        let text = message(&error);
        assert!(text.contains("42"), "{text}");
        assert!(text.contains("--any-seed"), "{text}");
        for seed in [13, 17, 23, 29, 31, 37, 41, 43, 47, 53] {
            assert!(
                text.contains(&seed.to_string()),
                "seed {seed} named in: {text}"
            );
        }
        assert!(
            !text.contains("benchmark-manifest"),
            "the seed policy is checked BEFORE anything is read: {text}"
        );
    }

    #[test]
    fn any_seed_accepts_an_uncontracted_seed_and_the_manifest_records_it() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());

        run_select(&data, 8, 42, true, None, false, false).expect("--any-seed accepts seed 42");

        let manifest = read_manifest(&data);
        assert_eq!(
            manifest.payload.root_seed, 42,
            "the seed itself is recorded, so the mode is a total function of the manifest"
        );
        let json = select_report_json(
            &data,
            &data.join(SELECTION_MANIFEST_FILE),
            &manifest,
            SEED_MODE_UNCONTRACTED,
        )
        .expect("report serializes");
        assert!(json.contains(SEED_MODE_UNCONTRACTED), "{json}");
    }

    #[test]
    fn a_compatibility_profile_directory_is_a_profile_mismatch() {
        let temp = TempDir::new().expect("tempdir");
        let data = compatibility_dir(temp.path());

        let error = run_select(&data, 8, 13, false, None, false, false)
            .expect_err("a setfit compatibility directory may never be selected from");
        let text = message(&error);
        assert!(text.contains("profile"), "{text}");
        // The attestation's profile tag is the CRATE's `Compatibility::PROFILE`
        // ("compatibility"), not the CLI's `--profile setfit` spelling. Asserting the
        // crate's word is the point: it is the value the boundary actually compared.
        assert!(text.contains("canonical"), "{text}");
        assert!(text.contains("compatibility"), "{text}");
    }

    #[test]
    fn a_mixed_directory_is_a_split_hash_mismatch_naming_the_split() {
        let temp = TempDir::new().expect("tempdir");
        let data = mixed_dir(temp.path());

        let error = run_select(&data, 8, 13, false, None, false, false)
            .expect_err("a directory assembled from two preparations is refused");
        let text = message(&error);
        assert!(text.contains("validation"), "the split is named: {text}");
        assert!(text.contains("hash") || text.contains("digest"), "{text}");
    }

    #[test]
    fn a_schema_version_one_directory_names_the_supported_set_and_the_remedy() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        edit_manifest(&data, |value| {
            // `Value::from` rather than the json! macro: this plan's acceptance grep
            // requires the file to contain no `json!`, and a needle that matches a test
            // helper is the self-needle problem 02-08 had to fix twice.
            value["schema_version"] = serde_json::Value::from(1_u64);
        });

        let error = run_select(&data, 8, 13, false, None, false, false)
            .expect_err("a version-1 benchmark directory has no migration");
        let text = message(&error);
        assert!(text.contains('1'), "{text}");
        assert!(text.contains('2'), "the supported set is named: {text}");
        assert!(
            text.contains("apr data tweet-eval-stance"),
            "the remedy is actionable: {text}"
        );
    }

    #[test]
    fn a_tampered_fingerprint_is_a_fingerprint_mismatch() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        edit_manifest(&data, |value| {
            value["dataset_attestation"]["dataset_fingerprint"] =
                serde_json::Value::String("0".repeat(64));
        });

        let error = run_select(&data, 8, 13, false, None, false, false)
            .expect_err("an edited fingerprint is refused");
        let text = message(&error);
        assert!(text.contains("fingerprint"), "{text}");
    }

    #[test]
    fn a_tampered_exclusion_hash_is_an_exclusion_record_mismatch() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        edit_manifest(&data, |value| {
            value["dataset_attestation"]["exclusion_hash"] =
                serde_json::Value::String("f".repeat(64));
        });

        let error = run_select(&data, 8, 13, false, None, false, false)
            .expect_err("an edited exclusion digest is refused");
        let text = message(&error);
        assert!(text.contains("exclusion"), "{text}");
    }

    #[test]
    fn a_missing_data_directory_names_the_file_it_could_not_find() {
        let temp = TempDir::new().expect("tempdir");
        let absent = temp.path().join("no-such-directory");

        let error = run_select(&absent, 8, 13, false, None, false, false)
            .expect_err("there is nothing to select from");
        let text = message(&error);
        assert!(text.contains("benchmark-manifest.json"), "{text}");
        assert!(text.contains("no-such-directory"), "{text}");
        assert!(
            text.contains("apr data tweet-eval-stance"),
            "an actionable message points at the command that creates one: {text}"
        );
    }

    // --------------------------------------------------------------------------------
    // Write safety — proven here once, for the helper every artifact goes through
    // --------------------------------------------------------------------------------

    #[test]
    fn an_existing_manifest_is_not_replaced_without_force_and_stays_byte_identical() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        run_select(&data, 8, 13, false, None, false, false).expect("first select");
        let before = fs::read(data.join(SELECTION_MANIFEST_FILE)).expect("manifest readable");

        let error = run_select(&data, 16, 17, false, None, false, false)
            .expect_err("no-clobber is the default");
        let text = message(&error);
        assert!(text.contains(SELECTION_MANIFEST_FILE), "{text}");
        assert!(text.contains("--force"), "{text}");

        let after = fs::read(data.join(SELECTION_MANIFEST_FILE)).expect("manifest readable");
        assert_eq!(before, after, "the refused write changed nothing");
    }

    #[test]
    fn force_replaces_the_manifest_completely_rather_than_overwriting_a_prefix() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        // Longer than any manifest, so a prefix-only write would leave trailing garbage
        // and the round trip below would fail.
        let filler = vec![b'x'; 4 * 1024 * 1024];
        fs::write(data.join(SELECTION_MANIFEST_FILE), &filler).expect("filler written");

        run_select(&data, 8, 13, false, None, true, false).expect("--force replaces it");

        let bytes = fs::read(data.join(SELECTION_MANIFEST_FILE)).expect("manifest readable");
        assert!(
            bytes.len() < filler.len(),
            "the file was replaced, not patched"
        );
        SelectionManifest::from_bytes(&bytes).expect("the replacement is a complete manifest");
    }

    #[test]
    fn an_induced_mid_write_failure_leaves_no_partial_artifact_and_no_temp_file() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        let before: Vec<PathBuf> = fs::read_dir(&data)
            .expect("data dir")
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .collect();

        FAIL_BEFORE_RENAME.with(|flag| flag.set(true));
        let error = run_select(&data, 8, 13, false, None, false, false)
            .expect_err("the induced failure must surface");
        FAIL_BEFORE_RENAME.with(|flag| flag.set(false));
        assert!(message(&error).contains("induced"), "{}", message(&error));

        assert!(
            !data.join(SELECTION_MANIFEST_FILE).exists(),
            "no partial manifest survives"
        );
        let after: Vec<PathBuf> = fs::read_dir(&data)
            .expect("data dir")
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .collect();
        assert_eq!(
            before.len(),
            after.len(),
            "no temp file was left behind: {after:?}"
        );
    }

    #[test]
    fn an_induced_mid_write_failure_leaves_a_pre_existing_manifest_byte_unchanged() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        run_select(&data, 8, 13, false, None, false, false).expect("first select");
        let before = fs::read(data.join(SELECTION_MANIFEST_FILE)).expect("manifest readable");
        let entries_before = fs::read_dir(&data).expect("data dir").count();

        FAIL_BEFORE_RENAME.with(|flag| flag.set(true));
        let error = run_select(&data, 16, 17, false, None, true, false)
            .expect_err("the induced failure must surface even under --force");
        FAIL_BEFORE_RENAME.with(|flag| flag.set(false));
        assert!(message(&error).contains("induced"), "{}", message(&error));

        let after = fs::read(data.join(SELECTION_MANIFEST_FILE)).expect("manifest readable");
        assert_eq!(before, after, "--force did not truncate the existing file");
        assert_eq!(
            entries_before,
            fs::read_dir(&data).expect("data dir").count(),
            "no temp file was left behind"
        );
    }

    // --------------------------------------------------------------------------------
    // Structural: the adapter must not reimplement the crate
    // --------------------------------------------------------------------------------

    #[test]
    fn the_selected_ids_are_the_crates_own_and_not_a_cli_reordering() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        run_select(&data, 8, 13, false, None, false, false).expect("select succeeds");
        let manifest = read_manifest(&data);

        // The mirror: rebuild the selection through the crate directly and require the
        // written manifest to agree exactly. Without this, "a manifest was written" would
        // be satisfied by a CLI that selected rows itself.
        let attestation_bytes = data_tweeteval::attestation_bytes_from_manifest(
            &fs::read(data.join(MANIFEST_FILE)).expect("benchmark manifest"),
        )
        .expect("attestation bytes");
        let mut buffers = std::collections::BTreeMap::new();
        for role in ["train", "validation", "test"] {
            buffers.insert(
                role.to_string(),
                fs::read(data.join(format!("{role}.jsonl"))).expect("split readable"),
            );
        }
        let mut ledger = aprender_contrastive_data::ledger::AccessLedger::new();
        let dataset = aprender_contrastive_data::prepared::PreparedDataset::<
            aprender_contrastive_data::prepared::Canonical,
        >::from_attested_bytes(&attestation_bytes, &buffers, &mut ledger)
        .expect("the fixture directory is attested");
        let selection = aprender_contrastive_data::select::FewShotSelector::select(
            &dataset,
            &aprender_contrastive_data::select::SelectionConfig {
                root_seed: 13,
                shots_per_class: 8,
            },
            &mut ledger,
        )
        .expect("selection succeeds");

        let written: Vec<&str> = manifest
            .payload
            .ordered_examples
            .iter()
            .map(|row| row.id.as_str())
            .collect();
        assert_eq!(written, selection.ordered_ids());
    }

    // --------------------------------------------------------------------------------
    // `apr data pairs`
    // --------------------------------------------------------------------------------

    /// A selected canonical directory plus the manifest `apr data select` wrote into it.
    fn selected(root: &Path) -> (PathBuf, PathBuf) {
        let data = canonical_dir(root);
        run_select(&data, 8, 13, false, None, false, false).expect("select succeeds");
        let manifest = data.join(SELECTION_MANIFEST_FILE);
        (data, manifest)
    }

    /// Rewrite a selection manifest with a RESEALED envelope digest.
    ///
    /// Resealing matters: `SelectionManifest::from_bytes` verifies the digest before it
    /// returns, so an un-resealed edit never reaches `Selection::replay` at all and would
    /// test the parser instead of the replay ladder. The digest is recomputed with the
    /// CRATE's own `hash::exact_hash` — `semantic_hash` is SHA-256 over
    /// `payload.to_canonical_bytes()` — so this module still implements no hashing.
    fn reseal(path: &Path, mutate: impl FnOnce(&mut SelectionManifest)) {
        let bytes = fs::read(path).expect("selection manifest readable");
        let mut manifest =
            SelectionManifest::from_bytes(&bytes).expect("the honest manifest parses");
        mutate(&mut manifest);
        let payload = manifest
            .payload
            .to_canonical_bytes()
            .expect("payload serializes");
        let text = core::str::from_utf8(&payload).expect("canonical payload is UTF-8");
        manifest.semantic_hash = hex(&exact_hash(text));
        fs::write(path, manifest.to_file_bytes().expect("envelope serializes"))
            .expect("selection manifest writable");
        // The forgery must be internally consistent, or every test below would be
        // measuring the parser rather than the ladder it targets.
        let reread = fs::read(path).expect("re-read");
        SelectionManifest::from_bytes(&reread).expect("the resealed forgery parses cleanly");
    }

    /// A training row of class `label` that the selection did NOT pick, with its real
    /// content hashes read out of `train.jsonl`.
    fn unselected_row(
        data: &Path,
        manifest: &SelectionManifest,
        label: usize,
    ) -> SelectedExampleRecord {
        let taken: Vec<&str> = manifest
            .payload
            .ordered_examples
            .iter()
            .map(|row| row.id.as_str())
            .collect();
        let jsonl = fs::read_to_string(data.join("train.jsonl")).expect("train split readable");
        for line in jsonl.lines() {
            let row: serde_json::Value = serde_json::from_str(line).expect("train row is JSON");
            let id = row["id"].as_str().expect("row id");
            let row_label =
                usize::try_from(row["label"].as_u64().expect("row label")).expect("label fits");
            let input = row["input"].as_str().expect("row input");
            if row_label == label && !taken.contains(&id) {
                return SelectedExampleRecord {
                    id: id.to_string(),
                    label,
                    exact_hash: hex(&exact_hash(input)),
                    normalized_hash: hex(&normalized_hash(input)),
                };
            }
        }
        panic!("the fixture must hold an unselected row of class {label}");
    }

    fn replayed_selection(data: &Path, manifest_path: &Path) -> Selection {
        let bytes = fs::read(manifest_path).expect("selection manifest readable");
        let manifest = SelectionManifest::from_bytes(&bytes).expect("manifest parses");
        let mut ledger = AccessLedger::new();
        let dataset = read_attested_canonical(data, &mut ledger).expect("attested directory");
        Selection::replay(&manifest, &dataset, &mut ledger).expect("strict replay succeeds")
    }

    #[test]
    fn pairs_reports_a_stable_hash_and_the_two_kind_counts() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());

        let first =
            pairs_outcome(&manifest, &data, None, None, None, false).expect("pairs succeeds");
        let second =
            pairs_outcome(&manifest, &data, None, None, None, false).expect("pairs succeeds again");

        assert_eq!(
            first.manifest_hash, second.manifest_hash,
            "two runs over the same inputs commit to the same tuple"
        );
        assert_eq!(first.manifest_hash.len(), 64, "the hash is rendered as hex");
        assert!(first.record.budget > 0);
        assert_eq!(
            first.positives + first.negatives,
            first.record.budget,
            "every emitted pair is counted exactly once"
        );
        assert!(
            first.positives > 0 && first.negatives > 0,
            "3x8 emits both kinds"
        );
        assert_eq!(first.record.emitted_kinds, "both");
        assert_eq!(first.record.deviation.len(), 3);
    }

    #[test]
    fn a_different_seed_commits_to_a_different_pair_manifest_hash() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());
        run_select(&data, 8, 13, false, None, false, false).expect("select at 13");
        let thirteen = pairs_outcome(
            &data.join(SELECTION_MANIFEST_FILE),
            &data,
            None,
            None,
            None,
            false,
        )
        .expect("pairs at 13");
        run_select(&data, 8, 17, false, None, true, false).expect("select at 17");
        let seventeen = pairs_outcome(
            &data.join(SELECTION_MANIFEST_FILE),
            &data,
            None,
            None,
            None,
            false,
        )
        .expect("pairs at 17");

        assert_ne!(
            thirteen.manifest_hash, seventeen.manifest_hash,
            "the selection hash is inside the digest, so a different selection cannot \
             collide with it"
        );
    }

    #[test]
    fn an_odd_budget_splits_the_two_kinds_by_exactly_one() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());

        let outcome =
            pairs_outcome(&manifest, &data, Some(255), None, None, false).expect("pairs succeeds");

        assert_eq!(outcome.record.budget, 255);
        let delta = outcome.positives.abs_diff(outcome.negatives);
        assert_eq!(delta, 1, "an odd budget cannot be split evenly");
    }

    #[test]
    fn the_dump_round_trips_through_parse_pair_dump_and_validate_pair_records() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let dump = temp.path().join("pairs.jsonl");

        run_pairs(&manifest, &data, Some(64), None, Some(&dump), false, false)
            .expect("pairs with --dump succeeds");

        let bytes = fs::read(&dump).expect("dump readable");
        assert_eq!(
            bytes.iter().filter(|b| **b == b'\n').count(),
            64,
            "one JSON line per pair, in stream order"
        );
        let records = parse_pair_dump(&bytes).expect("the dump parses as untrusted records");
        let selection = replayed_selection(&data, &manifest);
        let validated = validate_pair_records(&records, &selection)
            .expect("every endpoint is in the selection");
        assert_eq!(validated.len(), 64);

        // The mirror: the validated pairs ARE the sampler's own stream, pair for pair.
        // Without this, "the dump parses" would be satisfied by a dump of anything.
        let cfg = PairConfig {
            budget: Some(64),
            ..PairConfig::new(selection.root_seed())
        };
        let sampler = PairSampler::new(&selection, &cfg).expect("sampler");
        let mine: Vec<_> = (0..64)
            .map(|ordinal| sampler.pair_at(ordinal).expect("pair"))
            .collect();
        assert_eq!(validated, mine);
    }

    /// A role from an untrusted attestation can never become a path outside `--data`.
    ///
    /// `Path::join` REPLACES the base when the component is absolute, so before the
    /// allowlist a manifest declaring `"role": "/etc/passwd"` read `/etc/passwd.jsonl`.
    /// The mirror matters as much as the rejections: without it, a `role_file` that
    /// refused everything would pass this test while breaking every real directory.
    #[test]
    fn role_file_refuses_any_role_that_could_escape_the_data_directory() {
        let data_dir = Path::new("/benchmark/data");

        for hostile in [
            "/etc/passwd",
            "../../../etc/passwd",
            "..",
            "train/../../escape",
            "sub/dir",
            "",
            "TRAIN",
            "train ",
        ] {
            let refused = role_file(hostile);
            assert!(
                refused.is_err(),
                "role {hostile:?} was accepted and would resolve to {:?}",
                refused.map(|f| data_dir.join(f))
            );
        }

        // The mirror: every role the protocol actually defines still resolves, and stays
        // a single component inside the data directory.
        for (role, want) in [
            (Train::ROLE, "train.jsonl"),
            (Validation::ROLE, "validation.jsonl"),
            (Test::ROLE, "test.jsonl"),
            (CompatibilityTest::ROLE, "test.jsonl"),
        ] {
            let file = role_file(role).expect("a protocol role resolves");
            assert_eq!(file, want, "role {role:?}");
            assert_eq!(
                data_dir.join(&file).parent(),
                Some(data_dir),
                "role {role:?} escaped to {:?}",
                data_dir.join(&file)
            );
        }
    }

    /// The DERIVED kind counts must equal the STREAMED ones they replaced.
    ///
    /// `count_kinds` stopped measuring the stream and started deriving from `emitted_kinds`
    /// plus the budget. That is only sound while the derivation agrees with the interleave
    /// it assumes (`2t -> positive`, `2t + 1 -> negative`); nothing else in the tree pins
    /// the two together, so a change to `pair_at`'s ordering would silently make every
    /// reported `positives`/`negatives` wrong while the whole suite stayed green.
    ///
    /// This is the mirror for that optimization: `count_kinds_by_streaming` is still the
    /// ground truth, and this test is what keeps the fast path honest. Odd budgets are
    /// included deliberately — `div_ceil` and `/` differ exactly there, so an even-only
    /// test would pass under a swapped pair of expressions.
    #[test]
    fn the_derived_kind_counts_equal_the_streamed_ones() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let selection = replayed_selection(&data, &manifest);

        for budget in [1_u64, 2, 3, 7, 64, 255] {
            let cfg = PairConfig {
                budget: Some(budget),
                ..PairConfig::new(selection.root_seed())
            };
            let sampler = PairSampler::new(&selection, &cfg).expect("sampler");

            let derived = count_kinds(&sampler).expect("derived counts");
            let streamed = count_kinds_by_streaming(&sampler).expect("streamed counts");
            assert_eq!(
                derived, streamed,
                "budget {budget}: derived {derived:?} but the stream says {streamed:?}"
            );
            assert_eq!(
                derived.0 + derived.1,
                budget,
                "budget {budget}: the two kinds must account for every ordinal"
            );
        }
    }

    #[test]
    fn the_json_report_carries_the_hash_the_budget_the_policies_and_the_deviation() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let outcome =
            pairs_outcome(&manifest, &data, None, None, None, false).expect("pairs succeeds");

        let json = pairs_report_json(&manifest, &data, &outcome, None).expect("report serializes");
        for needle in [
            "\"pair_manifest_hash\"",
            "\"budget\"",
            "\"default_was_clamped\"",
            "\"hard_cap\"",
            "\"emitted_kinds\"",
            "\"positives\"",
            "\"negatives\"",
            "\"singleton_policy\"",
            "\"singleton_policy_version\"",
            "\"degenerate_policy_version\"",
            "\"deviation\"",
            "\"selection_hash\"",
        ] {
            assert!(
                json.contains(needle),
                "the --json report must carry {needle}"
            );
        }
        assert!(json.contains(&outcome.manifest_hash));
        assert!(
            json.contains("SELF-PAIRS ARE EXCLUDED"),
            "the three deviation clauses are copied verbatim, not summarized"
        );
    }

    // ---- rejections -----------------------------------------------------------------

    #[test]
    fn a_hand_edited_manifest_is_refused_before_any_replay_work() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let bytes = fs::read(&manifest).expect("manifest readable");
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).expect("manifest JSON");
        value["payload"]["root_seed"] = serde_json::Value::from(17_u64);
        fs::write(
            &manifest,
            serde_json::to_vec_pretty(&value).expect("re-encode"),
        )
        .expect("manifest writable");

        let error = run_pairs(&manifest, &data, None, None, None, false, false)
            .expect_err("an edited manifest is not a manifest");
        let text = message(&error);
        assert!(
            text.contains("semantic hash") || text.contains("hash mismatch"),
            "{text}"
        );
    }

    #[test]
    fn a_resealed_manifest_naming_an_absent_id_is_refused_and_the_id_is_named() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        reseal(&manifest, |m| {
            m.payload.ordered_examples[0].id = "train:999999".to_string();
        });

        let error = run_pairs(&manifest, &data, None, None, None, false, false)
            .expect_err("an id outside the selection pool is refused");
        let text = message(&error);
        assert!(text.contains("train:999999"), "the id is named: {text}");
    }

    #[test]
    fn a_resealed_manifest_with_a_wrong_row_hash_is_refused_and_the_id_is_named() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let victim = {
            let bytes = fs::read(&manifest).expect("readable");
            SelectionManifest::from_bytes(&bytes)
                .expect("parses")
                .payload
                .ordered_examples[0]
                .id
                .clone()
        };
        reseal(&manifest, |m| {
            m.payload.ordered_examples[0].exact_hash = "0".repeat(64);
        });

        let error = run_pairs(&manifest, &data, None, None, None, false, false)
            .expect_err("a recorded row hash that disagrees with the split is refused");
        let text = message(&error);
        assert!(text.contains(&victim), "the row is named: {text}");
        assert!(text.contains("hash"), "{text}");
    }

    #[test]
    fn a_resealed_manifest_with_a_wrong_ordered_list_is_caught_only_by_recomputation() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());

        // The CONTROL first: the untouched manifest replays. Without it, "the forgery is
        // refused" would also be satisfied by a fixture that had become unusable.
        pairs_outcome(&manifest, &data, None, None, None, false)
            .expect("the honest manifest replays");

        let substitute = {
            let bytes = fs::read(&manifest).expect("readable");
            let parsed = SelectionManifest::from_bytes(&bytes).expect("parses");
            unselected_row(&data, &parsed, 0)
        };
        reseal(&manifest, |m| {
            // Same class, same count, same ordering, REAL row hashes, resealed envelope.
            // Every static rung of the ladder accepts this; only recomputing the ordered
            // list from the seed rejects it.
            m.payload.ordered_examples[0] = substitute.clone();
        });

        let error = run_pairs(&manifest, &data, None, None, None, false, false)
            .expect_err("a selection no seed could have produced is refused");
        let text = message(&error);
        assert!(
            text.contains("ordered_examples") || text.contains("replay"),
            "the recomputation rung is what fired: {text}"
        );
    }

    #[test]
    fn replaying_against_a_different_dataset_directory_is_a_fingerprint_mismatch() {
        let temp = TempDir::new().expect("tempdir");
        let (_data, manifest) = selected(temp.path());
        let other = prepare(
            temp.path(),
            "other",
            "a different preparation",
            TweetEvalStanceProfile::Canonical,
        );

        let error = run_pairs(&manifest, &other, None, None, None, false, false)
            .expect_err("a selection may only be replayed against the dataset it was drawn from");
        let text = message(&error);
        assert!(text.contains("fingerprint"), "{text}");
    }

    #[test]
    fn a_compatibility_directory_fails_at_attested_ingest_for_pairs_too() {
        let temp = TempDir::new().expect("tempdir");
        let (_data, manifest) = selected(temp.path());
        let compat = compatibility_dir(temp.path());

        let error = run_pairs(&manifest, &compat, None, None, None, false, false)
            .expect_err("the pairs path uses the same attested door");
        let text = message(&error);
        assert!(text.contains("profile"), "{text}");
        assert!(text.contains("compatibility"), "{text}");
    }

    #[test]
    fn a_stale_schema_directory_fails_at_attested_ingest_for_pairs_too() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        edit_manifest(&data, |value| {
            value["schema_version"] = serde_json::Value::from(1_u64);
        });

        let error = run_pairs(&manifest, &data, None, None, None, false, false)
            .expect_err("a version-1 directory has no migration on either command");
        let text = message(&error);
        assert!(text.contains("apr data tweet-eval-stance"), "{text}");
    }

    #[test]
    fn a_zero_budget_and_a_zero_hard_cap_are_distinct_typed_errors() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());

        let zero_budget = message(
            &run_pairs(&manifest, &data, Some(0), None, None, false, false)
                .expect_err("a zero budget emits nothing"),
        );
        assert!(zero_budget.contains("budget"), "{zero_budget}");

        let zero_cap = message(
            &run_pairs(&manifest, &data, None, Some(0), None, false, false)
                .expect_err("a zero cap can never be satisfied"),
        );
        assert!(
            zero_cap.contains("hard_cap") || zero_cap.contains("hard-cap"),
            "{zero_cap}"
        );
        assert_ne!(
            zero_budget, zero_cap,
            "a request defect and a configuration defect are different messages"
        );
    }

    #[test]
    fn a_budget_above_the_hard_cap_names_both_numbers_and_the_flag_to_raise() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());

        let error = run_pairs(&manifest, &data, Some(500), Some(100), None, false, false)
            .expect_err("the cap BINDS an explicit budget; it does not clamp it");
        let text = message(&error);
        assert!(text.contains("500"), "{text}");
        assert!(text.contains("100"), "{text}");
        assert!(
            text.contains("--hard-cap"),
            "the remedy names the flag: {text}"
        );
    }

    #[test]
    fn a_missing_selection_manifest_names_the_command_that_writes_one() {
        let temp = TempDir::new().expect("tempdir");
        let data = canonical_dir(temp.path());

        let error = run_pairs(
            &data.join(SELECTION_MANIFEST_FILE),
            &data,
            None,
            None,
            None,
            false,
            false,
        )
        .expect_err("there is no selection to replay");
        let text = message(&error);
        assert!(text.contains(SELECTION_MANIFEST_FILE), "{text}");
        assert!(text.contains("apr data select"), "actionable: {text}");
    }

    // ---- the dump inherits Task 2's write safety ------------------------------------

    #[test]
    fn the_dump_is_not_clobbered_without_force_and_force_replaces_it() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let dump = temp.path().join("pairs.jsonl");
        fs::write(&dump, b"pre-existing\n").expect("dump seeded");

        let error = run_pairs(&manifest, &data, Some(8), None, Some(&dump), false, false)
            .expect_err("no-clobber is the default for the dump too");
        assert!(message(&error).contains("--force"), "{}", message(&error));
        assert_eq!(
            fs::read(&dump).expect("dump readable"),
            b"pre-existing\n",
            "the refused write changed nothing"
        );

        run_pairs(&manifest, &data, Some(8), None, Some(&dump), true, false)
            .expect("--force replaces the dump");
        assert_eq!(
            fs::read_to_string(&dump)
                .expect("dump readable")
                .lines()
                .count(),
            8
        );
    }

    #[test]
    fn an_induced_mid_write_failure_on_the_dump_leaves_no_artifact_and_no_temp_file() {
        let temp = TempDir::new().expect("tempdir");
        let (data, manifest) = selected(temp.path());
        let dumps = temp.path().join("dumps");
        fs::create_dir_all(&dumps).expect("dump dir");
        let dump = dumps.join("pairs.jsonl");

        FAIL_BEFORE_RENAME.with(|flag| flag.set(true));
        let error = run_pairs(&manifest, &data, Some(8), None, Some(&dump), false, false)
            .expect_err("the induced failure must surface on the dump path too");
        FAIL_BEFORE_RENAME.with(|flag| flag.set(false));

        assert!(message(&error).contains("induced"), "{}", message(&error));
        assert!(!dump.exists(), "no partial dump survives");
        assert_eq!(
            fs::read_dir(&dumps).expect("dump dir").count(),
            0,
            "no temp file was left behind"
        );
    }
}
