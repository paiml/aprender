//! `apr setfit bench run` — ONE benchmark cell in, ONE verified row out (D-13, EVAL-03/EVAL-05).
//!
//! # This file is a FILESYSTEM ADAPTER, exactly like `setfit_train.rs`
//!
//! Every semantic decision lives behind a library door. The row schema and its digest
//! discipline are `entrenar::train::setfit::bench_row`; the metric assembly is
//! `bench_metrics::assemble_quality_block`; the predictions are
//! `apr_evaluate::evaluate_rows_from_artifact`; the training lifecycle is `SetFitRun`'s four
//! transitions; the test-split gate is Phase 3's `create_selection_lock` -> `mint_test_token`
//! -> `CanonicalTestAccess::grant` chain, consumed here and reimplemented nowhere. This module
//! reads files, spawns one child, times things, and writes files.
//!
//! # The three modes, and why they are three
//!
//! 1. **EXECUTION.** Train, write, RELOAD, measure, evaluate, emit. One process per cell.
//! 2. **RECORD** (`--record`). Ingest a row file that a DIFFERENT host executed, verifying its
//!    digest, schema, cell identity and filename agreement before recording it. Nothing is
//!    executed. This is the D-09 transport path: the LoRA cells run on a GPU host and their
//!    rows travel back as bytes.
//! 3. **COLD PROBE** (`--cold-probe`). A dedicated fresh child whose entire job is to load one
//!    artifact, classify once, print `COLD_LATENCY_MS=<f64>` and exit.
//!
//! # Why the cold probe is a CHILD and not the first classify here
//!
//! The contract's resource protocol forbids measuring cold latency in the process that just
//! finished training: that process's page cache is warm, its allocator arenas are populated
//! and the encoder is already resident, so its "first" classify is operationally WARM. The
//! same argument applies to peak RSS, and more sharply — a kernel high-water mark is process
//! CUMULATIVE, so `VmHWM` read in a process that trained and then inferred reports the
//! TRAINING peak while claiming to report the inference peak. Hence two fields with two
//! mechanisms, never one pooled number.
//!
//! The child is spawned under `/usr/bin/time` (`-l` on macOS, `-v` on Linux) and the parent
//! reads the child's TRUE kernel high-water mark out of that output. Both platforms therefore
//! report an exact HWM of an equivalently-scoped process, which is what makes a macOS SetFit
//! row and a Linux LoRA row comparable at all. `mach_task_basic_info` was proposed in review
//! and REJECTED: this workspace sets `unsafe_code = "forbid"`.
//!
//! # Bounds
//!
//! Every file this module reads is bounded from its stat'd length BEFORE the read
//! (T-05-09-05), on the reasoning `eval/setfit.rs` states: a bound applied after the
//! allocation is not a bound on the work an attacker can request.

use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use entrenar::train::setfit::bench_row::{
    sha256_hex, BenchRow, CellKey, Method, RecordOutcome, RunManifest, BENCH_SEEDS, BENCH_SHOTS,
    CLAIMS_CONTRACT_ID,
};

use crate::commands::setfit_train::{atomic_write, refuse_existing_output};
use crate::error::{CliError, Result};

// ==========================================================================================
// Bounds and layout
// ==========================================================================================

/// The house cap for every small file this module reads (04-18's 16 MiB pattern).
///
/// The `setfit-apr-v1` artifact is NOT read through this: it goes through
/// `setfit_io::read_setfit_apr_file_bounded`, the ONE bounded artifact door.
pub(crate) const MAX_BENCH_FILE_BYTES: u64 = 16 * 1024 * 1024;

// The directory layout is the LIBRARY's, not this adapter's. `bench report` (05-10) resolves
// rows, locks and ledgers by these same names, and two spellings of a path are two paths: a
// drift between the writer here and the reader there would make every cell look un-run while
// reporting a missing-file error naming a path the writer never used. Re-exported rather than
// restated so there is exactly one definition.
pub(crate) use entrenar::train::setfit::bench_gate::{
    LEDGER_DIR, LOCKS_DIR, ROWS_DIR, RUN_MANIFEST_FILE,
};

/// The machine-readable line the cold-probe child prints, and its parent parses.
///
/// A CONTRACT between two processes, so it is a constant rather than a format string typed
/// twice: the child writes `COLD_LATENCY_MS=<f64>` and the parent looks for exactly this
/// prefix. A drift between the two spellings would surface as "the probe produced no
/// latency", which reads like a probe failure and is not one.
pub(crate) const COLD_LATENCY_PREFIX: &str = "COLD_LATENCY_MS=";

// ==========================================================================================
// The request
// ==========================================================================================

/// Everything the three modes carry, resolved from clap.
#[derive(Debug, Clone)]
pub(crate) struct BenchRunArgs<'a> {
    /// `setfit` or `lora` (execution mode).
    pub(crate) method: Option<&'a str>,
    /// Examples per class (execution mode).
    pub(crate) shots: Option<u32>,
    /// The contracted seed (execution mode).
    pub(crate) seed: Option<u32>,
    /// The attested prepared dataset directory (execution mode).
    pub(crate) data: Option<&'a Path>,
    /// The selection manifest — the pairing key (execution mode).
    pub(crate) selection: Option<&'a Path>,
    /// Where rows, locks, ledgers and the run manifest live.
    pub(crate) bench_dir: Option<&'a Path>,
    /// The pinned encoder checkout (execution mode, `setfit`).
    pub(crate) model_dir: Option<&'a Path>,
    /// The base model the adapter applies to (execution mode, `lora`).
    pub(crate) base_model: Option<&'a Path>,
    /// Optional training configuration (execution mode).
    pub(crate) config: Option<&'a Path>,
    /// Replace an existing row file or ledger.
    pub(crate) force: bool,
    /// Ingest a transported row file — no execution.
    pub(crate) record: Option<&'a Path>,
    /// The artifact the cold-probe child measures.
    pub(crate) cold_probe: Option<&'a Path>,
    /// The LoRA base model, when the cold probe measures a LoRA cell.
    pub(crate) cold_probe_base: Option<&'a Path>,
    /// The single text the cold probe classifies.
    pub(crate) probe_text: Option<&'a Path>,
    /// The global `--json`.
    pub(crate) json: bool,
}

/// Run one benchmark-matrix command.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a missing or contradictory flag set, an uncontracted
/// cell, an existing row file without `--force`, and every typed library refusal;
/// [`CliError::InvalidFormat`] for an over-cap file; [`CliError::Io`] for a read or write
/// failure.
pub(crate) fn run(args: &BenchRunArgs<'_>) -> Result<()> {
    // The COLD PROBE first, because it is the mode with the fewest obligations: it takes no
    // bench directory, writes nothing, and must not pay for any check the other two need.
    if let Some(artifact) = args.cold_probe {
        return cold_probe::run(artifact, args.cold_probe_base, args.probe_text);
    }

    // Both remaining modes need a bench directory. Checked here rather than in each, so the
    // message is one message.
    let bench_dir = args.bench_dir.ok_or_else(|| {
        CliError::ValidationFailed(
            "--bench-dir <DIR> is required. Rows are not a directory listing: completeness is \
             defined by the run manifest that lives there, and a listing can only report what \
             is present, never what is missing."
                .to_string(),
        )
    })?;

    if let Some(row_file) = args.record {
        return record_mode(bench_dir, row_file, args.force, args.json);
    }

    execute_mode(bench_dir, args)
}

// ==========================================================================================
// Cell identity — a non-contracted cell is a TYPED REFUSAL, never a run
// ==========================================================================================

/// Resolve `(method, shots, seed)` into a contracted cell, or refuse naming the contract.
///
/// The three components are checked SEPARATELY even though `CellKey::is_contracted` would
/// answer all three at once: an operator who typed `--seed 42` needs to be told that 42 is not
/// a contracted seed, not that "the cell is outside the matrix". The library's own combined
/// refusal remains the backstop at [`BenchRow::from_bytes`].
///
/// # Errors
///
/// [`CliError::ValidationFailed`] naming [`CLAIMS_CONTRACT_ID`] and the accepted values.
pub(crate) fn resolve_cell(
    method: Option<&str>,
    shots: Option<u32>,
    seed: Option<u32>,
) -> Result<CellKey> {
    let method_tag = method.ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "--method <setfit|lora> is required by {CLAIMS_CONTRACT_ID}"
        ))
    })?;
    let method = Method::from_tag(method_tag).ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "--method `{method_tag}` is not one of the two methods {CLAIMS_CONTRACT_ID} \
             compares: `setfit` or `lora`."
        ))
    })?;

    let shots = shots.ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "--shots <SHOTS> is required; {CLAIMS_CONTRACT_ID} contracts {BENCH_SHOTS:?}"
        ))
    })?;
    if !BENCH_SHOTS.contains(&shots) {
        return Err(CliError::ValidationFailed(format!(
            "--shots {shots} is not a contracted shot count. {CLAIMS_CONTRACT_ID} declares \
             exactly {BENCH_SHOTS:?}, and a cell outside that set is not part of the claim the \
             40-cell matrix makes — running it would produce a row the report has no slot for."
        )));
    }

    let seed = seed.ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "--seed <SEED> is required; {CLAIMS_CONTRACT_ID} contracts {BENCH_SEEDS:?}"
        ))
    })?;
    if !BENCH_SEEDS.contains(&seed) {
        return Err(CliError::ValidationFailed(format!(
            "--seed {seed} is not a contracted seed. {CLAIMS_CONTRACT_ID} declares exactly \
             {BENCH_SEEDS:?}. Note that 42 is deliberately NOT among them: a tool that defaults \
             a seed to 42 samples outside the protocol while appearing to honour it."
        )));
    }

    Ok(CellKey::new(method, shots, seed))
}

/// The row filename grammar, produced by ONE function — the LIBRARY's.
///
/// `{method}-s{shots}-seed{seed}.json`. Two spellings of a filename are two filenames, and both
/// the resume path here and `bench report`'s gate compare a recorded digest against the file
/// this name resolves to — so a drift between two copies would silently make every cell look
/// un-run. Delegated rather than restated.
pub(crate) use entrenar::train::setfit::bench_gate::row_file_name;

/// The committed SetFit lock record's path, RELATIVE to the bench directory.
#[must_use]
pub(crate) fn lock_relative_path(cell: CellKey) -> String {
    format!(
        "{LOCKS_DIR}/{}-s{}-seed{}.lock.json",
        cell.method.tag(),
        cell.shots,
        cell.seed
    )
}

/// The LoRA candidate ledger's path, RELATIVE to the bench directory.
#[must_use]
pub(crate) fn ledger_relative_path(cell: CellKey) -> String {
    format!(
        "{LEDGER_DIR}/{}-s{}-seed{}.jsonl",
        cell.method.tag(),
        cell.shots,
        cell.seed
    )
}

// ==========================================================================================
// Bounded reads
// ==========================================================================================

/// Read a small file, refused from its DECLARED length before a byte is read.
///
/// # Errors
///
/// [`CliError::FileNotFound`], [`CliError::NotAFile`], [`CliError::InvalidFormat`] above the
/// cap, or [`CliError::Io`].
pub(crate) fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            CliError::FileNotFound(path.to_path_buf())
        } else {
            CliError::Io(error)
        }
    })?;
    if !metadata.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    if metadata.len() > MAX_BENCH_FILE_BYTES {
        return Err(CliError::InvalidFormat(format!(
            "{}: declares {} bytes against the bench cap of {MAX_BENCH_FILE_BYTES}; refused \
             from its declared length, before a byte is read.",
            path.display(),
            metadata.len()
        )));
    }
    let file = fs::File::open(path).map_err(CliError::Io)?;
    let mut bytes = Vec::new();
    // `+ 1` so a length that LIED is detected rather than silently truncated into a payload
    // that happens to parse.
    file.take(MAX_BENCH_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(CliError::Io)?;
    if bytes.len() as u64 > MAX_BENCH_FILE_BYTES {
        return Err(CliError::InvalidFormat(format!(
            "{}: the stream exceeded {MAX_BENCH_FILE_BYTES} bytes, so its declared length lied",
            path.display()
        )));
    }
    Ok(bytes)
}

// ==========================================================================================
// The run manifest — read, initialise-on-first-touch, record, write back
// ==========================================================================================

/// Load the bench directory's run manifest, DECLARING it on first touch.
///
/// Declaring rather than erroring is deliberate: the expectation set is derived in code from
/// the contract's constants, so a fresh directory has exactly one correct manifest and asking
/// the operator to run a separate `--declare` step would only create a way to skip it.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a manifest whose digest, schema or expectation set the
/// library refuses; [`CliError::Io`] or [`CliError::InvalidFormat`] for the read.
pub(crate) fn load_or_declare_manifest(bench_dir: &Path) -> Result<RunManifest> {
    let path = bench_dir.join(RUN_MANIFEST_FILE);
    if !path.exists() {
        return Ok(RunManifest::declare());
    }
    let bytes = read_bounded(&path)?;
    RunManifest::from_bytes(&bytes).map_err(|error| {
        CliError::ValidationFailed(format!(
            "{}: {error}\nThe run manifest defines what a complete run IS. It is not \
             regenerated on a mismatch, because regenerating it would erase the pre-declared \
             expectation set that makes an omitted cell visible at all.",
            path.display()
        ))
    })
}

/// Record `row_sha256` against `cell` and persist the manifest.
///
/// `RunManifest::record` is idempotent on an identical digest (the resume path after a crash
/// at cell 35) and a typed error on a differing one (the collision). Neither decision is made
/// here; this only surfaces the refusal with the path the library cannot know.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for an unknown cell or a differing re-record;
/// [`CliError::Io`] for the write.
pub(crate) fn record_in_manifest(
    bench_dir: &Path,
    cell: CellKey,
    row_sha256: &str,
) -> Result<RecordOutcome> {
    let mut manifest = load_or_declare_manifest(bench_dir)?;
    let outcome = manifest.record(cell, row_sha256).map_err(|error| {
        CliError::ValidationFailed(format!(
            "{}: {error}",
            bench_dir.join(RUN_MANIFEST_FILE).display()
        ))
    })?;
    let bytes = manifest.to_file_bytes().map_err(|error| {
        CliError::ValidationFailed(format!("the run manifest did not serialize: {error}"))
    })?;
    // `force = true`: the manifest is a LIVE accumulator that every cell updates, so refusing
    // to replace it would make the second cell of any run fail. The no-clobber discipline
    // belongs to the ROW files, which are write-once evidence.
    atomic_write(&bench_dir.join(RUN_MANIFEST_FILE), &bytes, true)?;
    Ok(outcome)
}

// ==========================================================================================
// The resource protocol (EVAL-05) — three distinct measurement surfaces
// ==========================================================================================

/// The contracted resource measurements and the mechanisms that produced them.
///
/// Read `contracts/setfit-benchmark-claims-v1.yaml`'s `resource_protocol` beside this module:
/// every boundary here is implemented from that text, and the mechanism strings are mandatory
/// fields rather than a line in a methods section because a number without its measurement
/// boundary is not a measurement.
pub(crate) mod resource {
    use std::path::Path;
    use std::process::Command;
    use std::time::Instant;

    use entrenar::train::setfit::bench_row::{
        MECHANISM_CHILD_MAX_RSS_TIME_L, MECHANISM_CHILD_MAX_RSS_VM_HWM,
    };

    use crate::error::{CliError, Result};

    use super::COLD_LATENCY_PREFIX;

    // ---- The four mechanism strings, all named in this file --------------------------------

    /// Linux `/proc/self/status` `VmHWM`, read INSIDE the measured process.
    ///
    /// This value is labelled TRAIN peak and nothing else. `VmHWM` is process-CUMULATIVE, so
    /// in a process that trains, then reloads, then infers it reports the TRAINING peak.
    pub(crate) const MECHANISM_VM_HWM: &str = "vm_hwm";

    /// The sampled FALLBACK's prefix; the ACTUAL measured rate is appended.
    ///
    /// A poll at any finite rate can miss the peak entirely and the bias is one-directional
    /// (it can only understate), so the contract labels this a sampled LOWER BOUND and forbids
    /// comparing it against a `child_max_rss_*` figure.
    pub(crate) const MECHANISM_SAMPLED_PREFIX: &str = "sysinfo_sampled_";

    /// The two exact-kernel-high-water-mark mechanisms, re-exported so all four strings this
    /// module can emit are visible in one place.
    pub(crate) const MECHANISM_CHILD_TIME_L: &str = MECHANISM_CHILD_MAX_RSS_TIME_L;
    /// See [`MECHANISM_CHILD_TIME_L`].
    pub(crate) const MECHANISM_CHILD_VM_HWM: &str = MECHANISM_CHILD_MAX_RSS_VM_HWM;

    /// The warm measurement's sample count, after the contracted warmups.
    pub(crate) const WARM_MEASURED_COUNT: usize = 10;

    /// The batch size the throughput pass declares.
    ///
    /// RECORDED on the row because throughput without a batch size is not a comparable number
    /// (PF-008 warning sign 4).
    pub(crate) const THROUGHPUT_BATCH_SIZE: u32 = 32;

    /// The sampled fallback's target poll rate, in hertz.
    pub(crate) const SAMPLE_TARGET_HZ: u32 = 20;

    // ---- Peak RSS --------------------------------------------------------------------------

    /// One peak-RSS measurement with its boundary.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct PeakRss {
        /// The measured high-water mark, in bytes.
        pub(crate) bytes: u64,
        /// Which mechanism produced it.
        pub(crate) mechanism: String,
        /// Present exactly when `mechanism` is a sampled one.
        pub(crate) sample_interval_hz: Option<u32>,
    }

    /// Parse Linux `/proc/self/status`'s `VmHWM` line into BYTES.
    ///
    /// The kernel reports kilobytes; the row records bytes. Returns `None` when the field is
    /// absent or unparseable, which is a real state on a kernel that does not export it — and
    /// a caller that silently substituted zero would publish "this run used no memory".
    #[must_use]
    pub(crate) fn parse_vm_hwm_bytes(status_text: &str) -> Option<u64> {
        for line in status_text.lines() {
            // `strip_prefix` on the TRIMMED line, not `contains`: `VmHWMX:` and a line
            // mentioning VmHWM in prose must not match, and `contains` would accept both.
            let Some(rest) = line.trim_start().strip_prefix("VmHWM:") else {
                continue;
            };
            let mut fields = rest.split_whitespace();
            let value: u64 = fields.next()?.parse().ok()?;
            // The unit is part of the fact. A kernel that started reporting something other
            // than kB would otherwise be misread by a factor of 1024 in silence.
            if fields.next()? != "kB" {
                return None;
            }
            return value.checked_mul(1024);
        }
        None
    }

    /// Parse a `/usr/bin/time` block into `(bytes, mechanism)`.
    ///
    /// # The two platforms report DIFFERENT UNITS, and that is the whole hazard
    ///
    /// macOS `/usr/bin/time -l` prints `<value>  maximum resident set size` in **BYTES**.
    /// GNU `/usr/bin/time -v` prints `Maximum resident set size (kbytes): <value>` in
    /// **KILOBYTES**. The identical numeral therefore means two different quantities depending
    /// on which block it came from, so the mechanism is DERIVED FROM THE FORM THAT PARSED
    /// rather than from a `cfg!` — a parent that assumed its own platform would mislabel every
    /// number the moment the two ever ran on different hosts, which is exactly the D-09
    /// arrangement this benchmark uses.
    ///
    /// This parser ships a must-match / must-not-match case table (CLAUDE.md rule 7). The
    /// neighbouring lines in both blocks are the trap: macOS's `average shared memory size`
    /// and GNU's `Average resident set size (kbytes)` are one word away from matching.
    #[must_use]
    pub(crate) fn parse_child_max_rss(time_output: &str) -> Option<(u64, &'static str)> {
        for line in time_output.lines() {
            let trimmed = line.trim();
            // GNU form FIRST. It is prefix-anchored and unit-explicit, so it cannot be
            // confused with the macOS form (which ends with the label, not the number).
            if let Some(rest) = trimmed.strip_prefix("Maximum resident set size (kbytes):") {
                let kib: u64 = rest.trim().parse().ok()?;
                return kib.checked_mul(1024).map(|b| (b, MECHANISM_CHILD_VM_HWM));
            }
            // macOS form: the value PRECEDES a lowercase label that ends the line.
            if let Some(value) = trimmed.strip_suffix("maximum resident set size") {
                let bytes: u64 = value.trim().parse().ok()?;
                return Some((bytes, MECHANISM_CHILD_TIME_L));
            }
        }
        None
    }

    /// Parse the cold-probe child's one machine-readable line.
    #[must_use]
    pub(crate) fn parse_cold_latency_ms(stdout: &str) -> Option<f64> {
        stdout
            .lines()
            .find_map(|line| line.trim().strip_prefix(COLD_LATENCY_PREFIX))
            .and_then(|value| value.trim().parse().ok())
    }

    /// The median of a sample. Even counts average the two middles.
    ///
    /// Takes the samples by value and sorts in place: a median that mutated its caller's vector
    /// would reorder the latency series a reader might also want to print.
    #[must_use]
    pub(crate) fn median(mut samples: Vec<f64>) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(f64::total_cmp);
        let mid = samples.len() / 2;
        if samples.len() % 2 == 1 {
            Some(samples[mid])
        } else {
            Some((samples[mid - 1] + samples[mid]) / 2.0)
        }
    }

    // ---- The TRAIN peak, measured inside THIS process ---------------------------------------

    /// A running peak-RSS observation over the training phase.
    ///
    /// On Linux this is a no-op that reads `VmHWM` at the end — the kernel already keeps the
    /// high-water mark, and a sampler would only produce a worse estimate of a number that is
    /// exact. Everywhere else a background thread polls `sysinfo` and reports the maximum it
    /// SAW, together with the rate it actually achieved.
    pub(crate) struct TrainRssSampler {
        #[cfg(not(target_os = "linux"))]
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        #[cfg(not(target_os = "linux"))]
        handle: Option<std::thread::JoinHandle<(u64, u32)>>,
    }

    impl TrainRssSampler {
        /// Begin observing. Call BEFORE the training invocation.
        #[cfg(target_os = "linux")]
        #[must_use]
        pub(crate) fn start() -> Self {
            Self {}
        }

        /// Begin observing. Call BEFORE the training invocation.
        #[cfg(not(target_os = "linux"))]
        #[must_use]
        pub(crate) fn start() -> Self {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

                let interval =
                    std::time::Duration::from_micros(1_000_000 / u64::from(SAMPLE_TARGET_HZ));
                let pid = Pid::from_u32(std::process::id());
                let mut system = System::new();
                let mut peak = 0_u64;
                let mut samples = 0_u64;
                let started = Instant::now();
                while !flag.load(Ordering::Relaxed) {
                    system.refresh_processes_specifics(
                        ProcessesToUpdate::Some(&[pid]),
                        true,
                        ProcessRefreshKind::new().with_memory(),
                    );
                    if let Some(process) = system.process(pid) {
                        peak = peak.max(process.memory());
                    }
                    samples += 1;
                    std::thread::sleep(interval);
                }
                // The ACTUAL achieved rate, not the requested one. A thread that was starved
                // and polled at 3 Hz must not report 20: the mechanism string is the reader's
                // only handle on how much of the peak the sampler could have seen.
                let elapsed = started.elapsed().as_secs_f64();
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let achieved_hz = if elapsed > 0.0 {
                    ((samples as f64 / elapsed).round() as u32).max(1)
                } else {
                    SAMPLE_TARGET_HZ
                };
                (peak, achieved_hz)
            });
            Self {
                stop,
                handle: Some(handle),
            }
        }

        /// Stop observing and report the peak with its mechanism.
        #[cfg(target_os = "linux")]
        #[must_use]
        pub(crate) fn finish(self) -> PeakRss {
            let bytes = std::fs::read_to_string("/proc/self/status")
                .ok()
                .and_then(|text| parse_vm_hwm_bytes(&text))
                .unwrap_or(0);
            PeakRss {
                bytes,
                mechanism: MECHANISM_VM_HWM.to_string(),
                sample_interval_hz: None,
            }
        }

        /// Stop observing and report the peak with its mechanism.
        #[cfg(not(target_os = "linux"))]
        #[must_use]
        pub(crate) fn finish(mut self) -> PeakRss {
            use std::sync::atomic::Ordering;

            self.stop.store(true, Ordering::Relaxed);
            let (bytes, hz) = self
                .handle
                .take()
                .and_then(|handle| handle.join().ok())
                .unwrap_or((0, SAMPLE_TARGET_HZ));
            PeakRss {
                bytes,
                // The interval travels IN the mechanism string as well as in its own field, so
                // a reader looking at either one sees the boundary.
                mechanism: format!("{MECHANISM_SAMPLED_PREFIX}{hz}"),
                sample_interval_hz: Some(hz),
            }
        }
    }

    // ---- COLD latency + INFERENCE peak, in a dedicated fresh child ---------------------------

    /// What the cold-measurement child produced.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct ColdMeasurement {
        /// ONE classify, in a process that did nothing else.
        pub(crate) cold_latency_ms: f64,
        /// That process's TRUE kernel high-water mark.
        pub(crate) peak_rss_bytes: u64,
        /// Which of the two exact mechanisms produced it.
        pub(crate) peak_rss_mechanism: &'static str,
    }

    /// The `/usr/bin/time` invocation for this host.
    ///
    /// Absolute, never a bare `time`: `time` is a shell BUILTIN in bash and zsh and the
    /// builtin does not implement `-l` or `-v` at all. Resolving it through `PATH` would find
    /// whichever `time` a developer's environment happened to expose.
    const TIME_BIN: &str = "/usr/bin/time";

    /// `-l` reports the max RSS on macOS; `-v` does on GNU coreutils.
    #[cfg(target_os = "macos")]
    const TIME_FLAG: &str = "-l";
    /// See the macOS twin.
    #[cfg(not(target_os = "macos"))]
    const TIME_FLAG: &str = "-v";

    /// Spawn the dedicated cold-measurement child and read both numbers off it.
    ///
    /// `artifact` is the written production artifact; `base` is the LoRA base model when this
    /// is a LoRA cell and `None` for a standalone `setfit-apr-v1`.
    ///
    /// # Errors
    ///
    /// [`CliError::InferenceFailed`] when the child cannot be spawned, exits non-zero, or
    /// produces output neither parser recognises.
    pub(crate) fn measure_cold(
        artifact: &Path,
        base: Option<&Path>,
        probe_text: &Path,
    ) -> Result<ColdMeasurement> {
        // The child is THIS binary, resolved through `current_exe` — never a bare `apr`.
        // CLAUDE.md Verification Discipline rule 3: four `apr` binaries once coexisted on the
        // dev box and a bare `apr` resolved to a 26-day-old one. A benchmark row measured
        // against a different build than the one under test is worse than no row.
        let exe = std::env::current_exe().map_err(|error| {
            CliError::InferenceFailed(format!(
                "the cold-probe child could not be resolved: {error}. `current_exe` is the only \
                 spelling of \"this binary\" that cannot resolve to another build."
            ))
        })?;

        let mut command = Command::new(TIME_BIN);
        command
            .arg(TIME_FLAG)
            .arg(&exe)
            .arg("setfit")
            .arg("bench")
            .arg("run")
            .arg("--cold-probe")
            .arg(artifact);
        if let Some(base) = base {
            command.arg("--cold-probe-base").arg(base);
        }
        command.arg("--probe-text").arg(probe_text);

        let output = command.output().map_err(|error| {
            CliError::InferenceFailed(format!(
                "the cold-probe child could not be spawned under {TIME_BIN} {TIME_FLAG}: \
                 {error}"
            ))
        })?;
        // The status is read off the reaped `Output` ON ITS OWN LINE. CLAUDE.md Verification
        // Discipline rule 1: a status taken through a pipe is the LAST command's status, and
        // that defect has shipped twice in this repository.
        let status = output.status;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if !status.success() {
            return Err(CliError::InferenceFailed(format!(
                "the cold-probe child exited with {status}. Its output follows, because the \
                 child's refusal is the finding:\n{stderr}"
            )));
        }

        let cold_latency_ms = parse_cold_latency_ms(&stdout).ok_or_else(|| {
            CliError::InferenceFailed(format!(
                "the cold-probe child exited 0 but printed no `{COLD_LATENCY_PREFIX}` line. A \
                 missing measurement is a failure, not a zero.\nstdout:\n{stdout}"
            ))
        })?;
        // `/usr/bin/time` writes its report to STDERR on both platforms, which is why the two
        // streams are parsed separately rather than merged.
        let (peak_rss_bytes, peak_rss_mechanism) =
            parse_child_max_rss(&stderr).ok_or_else(|| {
                CliError::InferenceFailed(format!(
                    "{TIME_BIN} {TIME_FLAG} reported no maximum resident set size, so the child's \
                 true high-water mark is unknown. Emitting a sampled figure in its place would \
                 put a lower bound in a field the contract declares exact.\nstderr:\n{stderr}"
                ))
            })?;

        Ok(ColdMeasurement {
            cold_latency_ms,
            peak_rss_bytes,
            peak_rss_mechanism,
        })
    }

    // ---- WARM latency + throughput, against the RELOADED model in this process ---------------

    /// Median of [`WARM_MEASURED_COUNT`] classifies after the contracted warmups.
    ///
    /// `classify_one` must perform ONE single-text classification against the RELOADED model.
    /// The warmups are discarded, which is the boundary `apr bench --warmup` already uses.
    ///
    /// # Errors
    ///
    /// Whatever `classify_one` returns.
    pub(crate) fn warm_latency_ms_median<F>(warmups: u32, mut classify_one: F) -> Result<f64>
    where
        F: FnMut() -> Result<()>,
    {
        for _ in 0..warmups {
            classify_one()?;
        }
        let mut samples = Vec::with_capacity(WARM_MEASURED_COUNT);
        for _ in 0..WARM_MEASURED_COUNT {
            let started = Instant::now();
            classify_one()?;
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        median(samples).ok_or_else(|| {
            CliError::InferenceFailed(
                "the warm-latency sample was empty, which cannot happen with a positive \
                 measured count — treat this as a defect rather than a zero"
                    .to_string(),
            )
        })
    }

    /// Rows per second over ONE full pass of the test split at the declared batch size.
    ///
    /// `classify_batch` is handed each batch's row count and returns how many rows it actually
    /// classified; the mismatch check belongs to the caller's door, not here.
    ///
    /// # Errors
    ///
    /// Whatever `classify_batch` returns.
    pub(crate) fn throughput_rows_per_sec<F>(
        n_rows: usize,
        batch_size: u32,
        mut classify_batch: F,
    ) -> Result<f64>
    where
        F: FnMut(usize, usize) -> Result<()>,
    {
        let batch = batch_size.max(1) as usize;
        let started = Instant::now();
        let mut offset = 0;
        while offset < n_rows {
            let end = (offset + batch).min(n_rows);
            classify_batch(offset, end)?;
            offset = end;
        }
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            // A pass that measured zero wall time is a timer resolution artefact, not infinite
            // throughput. Reporting `inf` would render as `null` through serde_json.
            return Err(CliError::InferenceFailed(
                "the throughput pass measured zero wall time; the number this would produce is \
                 not a throughput"
                    .to_string(),
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        Ok(n_rows as f64 / elapsed)
    }
}

// ==========================================================================================
// The cold-probe child
// ==========================================================================================

/// The dedicated fresh child: load one artifact, classify once, print, exit.
mod cold_probe {
    use std::path::Path;
    use std::time::Instant;

    use crate::error::{CliError, Result};

    use super::{read_bounded, COLD_LATENCY_PREFIX};

    /// Run the probe.
    ///
    /// This process does NOTHING else. It has not trained, its page cache is cold, its
    /// allocator arenas are empty — which is what makes the one classify it performs a COLD
    /// measurement rather than a warm one wearing the word.
    ///
    /// # Errors
    ///
    /// [`CliError::ValidationFailed`] for a missing `--probe-text`;
    /// [`CliError::ModelLoadFailed`] for an artifact the load ladder rejects;
    /// [`CliError::InferenceFailed`] for a classify failure.
    pub(super) fn run(
        artifact: &Path,
        base: Option<&Path>,
        probe_text: Option<&Path>,
    ) -> Result<()> {
        let text_path = probe_text.ok_or_else(|| {
            CliError::ValidationFailed(
                "--probe-text <FILE> is required with --cold-probe: the probe classifies ONE \
                 text, and which text it is belongs to the measurement."
                    .to_string(),
            )
        })?;
        let text = String::from_utf8(read_bounded(text_path)?).map_err(|error| {
            CliError::ValidationFailed(format!(
                "--probe-text {}: not UTF-8 ({error})",
                text_path.display()
            ))
        })?;
        let text = text.trim_end_matches(['\n', '\r']).to_string();

        let cold_latency_ms = match base {
            Some(base) => lora_probe(artifact, base, &text)?,
            None => setfit_probe(artifact, &text)?,
        };

        // ONE machine-readable line on stdout. `/usr/bin/time`'s report goes to stderr, so the
        // parent parses two streams and neither can swallow the other.
        println!("{COLD_LATENCY_PREFIX}{cold_latency_ms}");
        Ok(())
    }

    /// A standalone `setfit-apr-v1`: the production loader, then one classify.
    fn setfit_probe(artifact: &Path, text: &str) -> Result<f64> {
        use aprender::setfit::{load_setfit_apr, ClassifyRequestDocument};

        // The timer opens BEFORE the read, because "cold" includes getting the artifact off
        // the platter and through the load ladder. A cold latency that excluded loading would
        // be a warm latency with extra steps.
        let started = Instant::now();
        let bytes = crate::setfit_io::read_setfit_apr_file_bounded(artifact)?;
        let model = load_setfit_apr(&bytes).map_err(|error| {
            CliError::ModelLoadFailed(format!("{}: {error}", artifact.display()))
        })?;
        let request = ClassifyRequestDocument::new(std::iter::once(text.to_string()));
        model
            .classify(&request)
            .map_err(|error| CliError::InferenceFailed(error.to_string()))?;
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }

    /// A LoRA cell: the base model plus the trained adapter, through the route 05-06 Task 3
    /// PROVED — `ClassifyPipeline::load_adapter` then `predict_proba_tokenized`.
    ///
    /// Named by those two symbols deliberately. 05-06's preflight recorded that no adapter
    /// load route existed before it, that `ClassifyPipeline::from_apr` builds FRESH LoRA
    /// layers, and that `resume_from_apr_checkpoint` installs tensors through `if let Ok(..)`
    /// — a silent partial load by construction. Inventing a reload here would re-open all of
    /// that against 40 remote 9B cells.
    fn lora_probe(adapter: &Path, base: &Path, text: &str) -> Result<f64> {
        let started = Instant::now();
        let mut pipeline = super::lora::reload_base_and_adapter(base, adapter)?;
        let token_ids = super::lora::tokenize_one(&pipeline, text)?;
        let probabilities = pipeline.predict_proba_tokenized(&token_ids);
        if probabilities.is_empty() {
            return Err(CliError::InferenceFailed(
                "the reloaded LoRA pipeline returned an empty probability vector".to_string(),
            ));
        }
        Ok(started.elapsed().as_secs_f64() * 1000.0)
    }
}

// ==========================================================================================
// `--record`: ingest a row another host executed. NO EXECUTION.
// ==========================================================================================

/// Ingest a transported row file into this bench directory.
///
/// # The verification order is the property
///
/// `BenchRow::from_bytes` verifies schema, digest, contracted cell and evidence-tag agreement
/// BEFORE returning a value, so nothing downstream can hold a row whose digest disagrees with
/// its payload. This function adds exactly one check the library cannot make: that the
/// FILENAME the operator handed over names the same cell the payload declares. A row whose
/// name and content disagree is a bookkeeping error that would otherwise land silently in the
/// slot named by whichever of the two the reader happened to trust.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a digest, schema, cell or filename disagreement, and for
/// a differing re-record; [`CliError::Io`] for the copy.
fn record_mode(bench_dir: &Path, row_file: &Path, force: bool, json: bool) -> Result<()> {
    let bytes = read_bounded(row_file)?;
    let row = BenchRow::from_bytes(&bytes).map_err(|error| {
        CliError::ValidationFailed(format!(
            "{}: {error}\nThese bytes are not recorded. Re-export the row on the host that \
             produced it rather than editing it here — the digest is over the payload, so an \
             edit that makes the row parse also makes it a different measurement.",
            row_file.display()
        ))
    })?;

    let cell = row.payload.cell();
    let expected_name = row_file_name(cell);
    let observed_name = row_file
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    if observed_name != expected_name {
        return Err(CliError::ValidationFailed(format!(
            "{}: the payload declares cell {cell} whose row file is `{expected_name}`, but the \
             file handed over is named `{observed_name}`. The name and the content must agree: \
             a row filed under another cell's name would be counted as that cell by every \
             later reader.",
            row_file.display()
        )));
    }

    let row_sha256 = sha256_hex(&row.payload.to_canonical_bytes().map_err(|error| {
        CliError::ValidationFailed(format!("the row payload did not re-serialize: {error}"))
    })?);

    // RECORD FIRST, COPY SECOND. `RunManifest::record` is the door that refuses a differing
    // re-record, and refusing AFTER the copy would leave the differing row on disk beside a
    // manifest that never accepted it.
    let outcome = record_in_manifest(bench_dir, cell, &row_sha256)?;

    let destination = bench_dir.join(ROWS_DIR).join(&expected_name);
    match outcome {
        // The idempotent path: the manifest already carries THIS digest, so the file on disk
        // is the same measurement. Re-writing it is permitted without `--force` precisely
        // because nothing can change — this is the resume-after-a-dropped-ssh case.
        RecordOutcome::AlreadyRecorded => atomic_write(&destination, &bytes, true)?,
        RecordOutcome::Recorded => atomic_write(&destination, &bytes, force)?,
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "command": "setfit-bench-record",
                "cell": cell.render(),
                "row": destination.display().to_string(),
                "row_sha256": row_sha256,
                "outcome": match outcome {
                    RecordOutcome::Recorded => "recorded",
                    RecordOutcome::AlreadyRecorded => "already_recorded",
                },
                "executed": false,
            })
        );
    } else {
        println!(
            "recorded {cell} -> {} ({})",
            destination.display(),
            match outcome {
                RecordOutcome::Recorded => "new",
                RecordOutcome::AlreadyRecorded => "identical digest, idempotent",
            }
        );
    }
    Ok(())
}

// ==========================================================================================
// Execution — the per-method cell paths
// ==========================================================================================

/// Execute one cell.
///
/// The check order follows `setfit_train.rs`'s module header: the whole REQUEST first (the
/// cell key, the required flags), then the output refusal, then the inputs. A run that spends
/// a training pass and then says "I will not overwrite that row" has told the operator
/// something it knew before it started.
fn execute_mode(bench_dir: &Path, args: &BenchRunArgs<'_>) -> Result<()> {
    let cell = resolve_cell(args.method, args.shots, args.seed)?;

    let data = args.data.ok_or_else(|| {
        CliError::ValidationFailed(
            "--data <DIR> is required: the canonical splits live in the Phase 2 prepared \
             dataset directory."
                .to_string(),
        )
    })?;
    let selection = args.selection.ok_or_else(|| {
        CliError::ValidationFailed(
            "--selection <FILE> is required. It is the PAIRING KEY: EVAL-02's identical-\
             sampled-ID guarantee is that both methods consumed the same manifest for this \
             (shots, seed), and the row records its hash."
                .to_string(),
        )
    })?;

    // The output refusal, before any input is read.
    let row_path = bench_dir.join(ROWS_DIR).join(row_file_name(cell));
    refuse_existing_output(&row_path, args.force).map_err(|_| {
        CliError::ValidationFailed(format!(
            "{} already exists, so cell {cell} has already been executed. Re-running a \
             completed cell is idempotent-or-refused, never silently duplicated: pass --force \
             only if you intend to replace the run that produced the published number.",
            row_path.display()
        ))
    })?;

    let request = CellRequest {
        cell,
        data,
        selection,
        bench_dir,
        model_dir: args.model_dir,
        base_model: args.base_model,
        config: args.config,
        force: args.force,
        json: args.json,
    };

    match cell.method {
        Method::Setfit => setfit_cell::execute(&request),
        Method::Lora => lora::execute(&request),
    }
}

/// One resolved cell-execution request, shared by both method paths.
pub(crate) struct CellRequest<'a> {
    /// The contracted cell.
    pub(crate) cell: CellKey,
    /// The attested prepared dataset directory.
    pub(crate) data: &'a Path,
    /// The selection manifest — the pairing key.
    pub(crate) selection: &'a Path,
    /// Where rows, locks, ledgers and the run manifest live.
    pub(crate) bench_dir: &'a Path,
    /// The pinned encoder checkout (`setfit` only).
    pub(crate) model_dir: Option<&'a Path>,
    /// The base model the adapter applies to (`lora` only).
    pub(crate) base_model: Option<&'a Path>,
    /// Optional training configuration.
    pub(crate) config: Option<&'a Path>,
    /// Replace an existing row file or ledger.
    pub(crate) force: bool,
    /// The global `--json`.
    pub(crate) json: bool,
}

/// Everything the Phase 2 ingest produces, replayed against itself and against the cell.
pub(crate) struct Phase2 {
    /// The attested canonical dataset.
    pub(crate) dataset: aprender_contrastive_data::prepared::PreparedDataset<
        aprender_contrastive_data::prepared::Canonical,
    >,
    /// The replayed selection.
    pub(crate) selection: aprender_contrastive_data::select::Selection,
    /// The manifest's own digest — THE PAIRING KEY both methods record.
    pub(crate) manifest_semantic_hash: String,
    /// The pinned upstream revision the directory was prepared from.
    pub(crate) dataset_revision: String,
}

/// Read `--data` and `--selection` through the doors `data_contrastive` owns.
///
/// # ONE ingest for BOTH methods, which is the whole of EVAL-02
///
/// The same three calls `commands/eval/setfit.rs` and `apr finetune --selection-manifest`
/// make — `read_attested_canonical`, `read_selection_manifest`, `Selection::replay`. A
/// per-method ingest would make "identical sampled IDs" rest on two readers agreeing, which
/// is the exporter-correctness argument the phase deliberately rejected.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for any attested-ingest or replay rejection, and for a
/// selection whose draw does not describe the cell being run.
pub(crate) fn read_phase2(request: &CellRequest<'_>) -> Result<Phase2> {
    use aprender_contrastive_data::ledger::AccessLedger;
    use aprender_contrastive_data::select::Selection;

    use crate::commands::{data_contrastive, data_tweeteval};

    let mut ledger = AccessLedger::new();
    let dataset = data_contrastive::read_attested_canonical(request.data, &mut ledger)?;
    let manifest = data_contrastive::read_selection_manifest(request.selection)?;
    let selection = Selection::replay(&manifest, &dataset, &mut ledger).map_err(|error| {
        CliError::ValidationFailed(format!(
            "--selection {} does not replay against --data {}: {error}",
            request.selection.display(),
            request.data.display()
        ))
    })?;

    // THE CELL KEY AND THE SELECTION MUST DESCRIBE THE SAME DRAW. Nothing else checks this:
    // `Selection::replay` proves the manifest describes THIS dataset, and the row records
    // `shots`/`seed` from the FLAGS. A cell run with `--seed 13` against a manifest drawn at
    // seed 17 would publish a row filed under seed 13 whose rows are seed 17's, and every
    // paired delta in the report would be comparing two different draws while looking
    // correctly paired — which is precisely the comparison PF-007 exists to forbid.
    if selection.root_seed() != u64::from(request.cell.seed) {
        return Err(CliError::ValidationFailed(format!(
            "--seed {} disagrees with the selection manifest, which was drawn at root seed {}. \
             The row would be filed under a seed its rows do not come from.",
            request.cell.seed,
            selection.root_seed()
        )));
    }
    if selection.shots_per_class() != request.cell.shots {
        return Err(CliError::ValidationFailed(format!(
            "--shots {} disagrees with the selection manifest, which carries {} per class.",
            request.cell.shots,
            selection.shots_per_class()
        )));
    }

    let manifest_path = request.data.join(data_tweeteval::MANIFEST_FILE);
    let manifest_bytes = read_bounded(&manifest_path)?;
    let dataset_revision = data_tweeteval::dataset_revision_from_manifest(&manifest_bytes)?;

    // THE PAIRING KEY, derived from the REPLAYED selection and cross-checked against the
    // manifest's own envelope. `apr finetune --selection-manifest` records
    // `hex(replayed.semantic_hash())` (05-06), so deriving it the same way here is what makes
    // the two methods' rows pair at all. The equality check is the two-sided half: if the
    // envelope and the replay ever disagreed, one method's rows would pair on a value the
    // other's never carried, and the report would silently drop every delta.
    let manifest_semantic_hash = aprender_contrastive_data::hash::hex(&selection.semantic_hash());
    if manifest_semantic_hash != manifest.semantic_hash {
        return Err(CliError::ValidationFailed(format!(
            "the selection manifest's envelope declares {} but the replayed selection hashes \
             to {manifest_semantic_hash}. The pairing key is ambiguous, so no row may be \
             written from it.",
            manifest.semantic_hash
        )));
    }

    Ok(Phase2 {
        dataset,
        selection,
        manifest_semantic_hash,
        dataset_revision,
    })
}

/// Where the cell ran, read from the host rather than declared.
#[must_use]
pub(crate) fn host_identity() -> entrenar::train::setfit::bench_row::HostIdentity {
    entrenar::train::setfit::bench_row::HostIdentity {
        hostname: sysinfo::System::host_name().unwrap_or_else(|| "unknown".to_string()),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

/// Seal a payload into a row and land both the row file and the manifest entry.
///
/// Atomic rename, so an interrupted cell leaves either NO row or one complete digest-valid
/// row — never a partial. The manifest is updated AFTER the row lands, so a manifest entry
/// always has a file behind it.
///
/// # Errors
///
/// [`CliError::ValidationFailed`] for a serialization or manifest refusal; [`CliError::Io`]
/// for the write.
pub(crate) fn emit_row(
    bench_dir: &Path,
    payload: entrenar::train::setfit::bench_row::BenchRowPayload,
    force: bool,
) -> Result<(PathBuf, String)> {
    let cell = payload.cell();
    let row = BenchRow::new(payload);
    let file_bytes = row.to_file_bytes().map_err(|error| {
        CliError::ValidationFailed(format!("the benchmark row did not serialize: {error}"))
    })?;
    let path = bench_dir.join(ROWS_DIR).join(row_file_name(cell));
    atomic_write(&path, &file_bytes, force)?;
    record_in_manifest(bench_dir, cell, &row.semantic_hash)?;
    Ok((path, row.semantic_hash))
}

/// Write the single probe text this cell's cold measurement classifies.
///
/// Taken from the test split's first row rather than invented, so the cold measurement is over
/// the same kind of input the throughput pass sees.
///
/// # Errors
///
/// [`CliError::Io`] for the write.
pub(crate) fn write_probe_text(bench_dir: &Path, cell: CellKey, text: &str) -> Result<PathBuf> {
    let path = bench_dir
        .join("probe")
        .join(format!("{}.txt", row_file_name(cell).replace(".json", "")));
    atomic_write(&path, text.as_bytes(), true)?;
    Ok(path)
}

// ==========================================================================================
// The SetFit cell path
// ==========================================================================================

/// The SetFit cell: train, write, RELOAD, measure, lock, evaluate, emit.
///
/// # Pure library orchestration
///
/// Not one semantic decision is made in this module. `SetFitRun`'s four transitions carry
/// every training gate; `load_setfit_apr` (reached through `reload_verified_run_from_apr`'s
/// load ladder) is the production loader, so only a VERIFIED artifact proceeds;
/// `evaluate_rows_from_artifact` is the ONE prediction door a benchmark cell may call;
/// `create_selection_lock` -> `mint_test_token` -> `CanonicalTestAccess::grant` is the D-16
/// workflow `apr eval` uses, consumed rather than reimplemented; `assemble_quality_block`
/// computes every published number. This module reads files, times things and writes files.
mod setfit_cell {
    use std::path::{Path, PathBuf};

    use aprender::setfit::{ClassifyRequestDocument, PINNED_REVISION};
    use aprender_contrastive_data::ledger::AccessLedger;
    use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
    use aprender_contrastive_data::select::Selection;
    use entrenar::train::setfit::apr_codec::AprCodec;
    use entrenar::train::setfit::apr_evaluate::{
        evaluate_rows_from_artifact, evaluate_validation_from_artifact, EvaluatedSplit,
    };
    use entrenar::train::setfit::apr_reload::{
        reload_verified_run_from_apr, ReloadedSetFitCredential,
    };
    use entrenar::train::setfit::bench_metrics::assemble_quality_block;
    use entrenar::train::setfit::bench_row::{
        sha256_hex, BenchLockRef, BenchRowPayload, MethodEvidence, ResourceBlock, SetfitEvidence,
        BENCH_ROW_SCHEMA_VERSION, CLAIMS_CONTRACT_ID, WARMUP_COUNT,
    };
    use entrenar::train::setfit::evaluate::ValidationMetricKind;
    use entrenar::train::setfit::lock::{
        create_selection_lock, CanonicalTestAccess, SelectionCandidate, SelectionRule,
    };
    use entrenar::train::setfit::SetFitRun;

    use crate::commands::{data_contrastive, data_tweeteval, setfit_train};

    use super::resource::{self, TrainRssSampler, THROUGHPUT_BATCH_SIZE};
    use super::{
        emit_row, host_identity, lock_relative_path, read_bounded, write_probe_text, CellRequest,
        CliError, Result,
    };

    /// The metric the lock orders candidates by — the SAME fixed choice `apr eval` makes.
    ///
    /// Fixed rather than a flag for the reason `eval/setfit.rs` records: a candidate set whose
    /// metric kinds disagree cannot form a lock, so a per-invocation choice would let an
    /// operator build a lock that cannot be created and discover it at the end.
    const EVAL_METRIC: ValidationMetricKind = ValidationMetricKind::Accuracy;

    /// The selection rule this command commits under — again `apr eval`'s.
    const EVAL_RULE: SelectionRule = SelectionRule::MaxMetricLowestIndexTieBreak;

    use super::{read_phase2, Phase2};

    /// Execute one SetFit cell.
    #[allow(clippy::too_many_lines)]
    pub(super) fn execute(request: &CellRequest<'_>) -> Result<()> {
        let model_dir = request.model_dir.ok_or_else(|| {
            CliError::ValidationFailed(
                "--model-dir <DIR> is required for a setfit cell: it names the pinned \
                 all-MiniLM-L6-v2 checkout. This command NEVER downloads."
                    .to_string(),
            )
        })?;

        // (1) THE REQUEST, in full, before anything expensive. The seed is the CELL's, and it
        //     goes through the library's validated merge door whether or not a file was given.
        let config = setfit_train::resolve_config(request.config, u64::from(request.cell.seed))?;

        // (2) Phase 2's artifacts, replayed strictly against each other and against the cell.
        let training_inputs = read_phase2(request)?;
        let dataset_revision = training_inputs.dataset_revision.clone();
        let selection_manifest_hash = training_inputs.manifest_semantic_hash.clone();
        let dataset_fingerprint = training_inputs
            .dataset
            .validation_witness()
            .dataset_fingerprint_hex();

        // (3) THE ENCODER, then the shipped lifecycle. Every gate — device probe, pair budget,
        //     selection/dataset agreement, calibration regime, evidence thresholds, head fit,
        //     artifact round trip — is inside those four transitions.
        let encoder =
            aprender::setfit::SetFitMiniLm::from_pretrained_dir(model_dir, config.root_seed())
                .map_err(|error| {
                    CliError::ModelLoadFailed(format!(
                        "--model-dir {}: {error}",
                        model_dir.display()
                    ))
                })?;

        // The TRAIN peak sampler opens here and closes the moment training ends, so what it
        // observes is the training process's footprint and not the reload's.
        let sampler = TrainRssSampler::start();
        let started = std::time::Instant::now();
        let prepared = SetFitRun::prepare(
            encoder,
            training_inputs.dataset,
            training_inputs.selection,
            config,
        )
        .map_err(train_error)?;
        let tuned = prepared.tune_encoder().map_err(train_error)?;
        let fitted = tuned.fit_head().map_err(train_error)?;
        let verified = fitted
            .verify_artifact(&AprCodec::new())
            .map_err(train_error)?;
        #[allow(clippy::cast_possible_truncation)]
        let train_wall_ms = started.elapsed().as_millis() as u64;
        let train_peak = sampler.finish();

        let evidence_table_hash = verified.evidence_table_hash().to_string();
        let apr_artifact_sha256 = verified.artifact_hash();

        // (4) THE WRITE. These are the bytes the trusted policy hashed and round-trip-closed,
        //     so `artifact_bytes` below counts the artifact `apr_artifact_sha256` describes.
        //     SetFit ships ONE file, so `deployable_total_bytes == artifact_bytes` — stated
        //     rather than assumed, because on the LoRA side they deliberately differ.
        let bytes = verified.into_artifact_bytes();
        let artifact_bytes = bytes.len() as u64;
        let artifact_path = artifact_path(request.bench_dir, request);
        super::atomic_write(&artifact_path, &bytes, true)?;
        drop(bytes);

        // (5) THE RELOAD. Through the production door, so only a VERIFIED artifact proceeds —
        //     and against a FRESH Phase 2 ingest, because the lifecycle consumed the first.
        //     Reading the directory twice is the same discipline `apr eval` runs under: the
        //     evaluation's inputs pass the attested boundary in their own right.
        let eval_inputs = read_phase2(request)?;
        let artifact_file_bytes = crate::setfit_io::read_setfit_apr_file_bounded(&artifact_path)?;
        let credential = reload_verified_run_from_apr(
            &artifact_file_bytes,
            &eval_inputs.dataset,
            &eval_inputs.selection,
        )
        .map_err(|error| {
            CliError::ValidationFailed(format!(
                "{}: {error}\nThe artifact this cell just wrote did not reload against the \
                 inputs it was trained on, so no number measured from it would describe the \
                 run that produced it.",
                artifact_path.display()
            ))
        })?;
        drop(artifact_file_bytes);

        // (6) COLD LATENCY + INFERENCE PEAK, in a dedicated fresh child. NEVER the first
        //     classify in this process: it has just trained and is operationally warm.
        let probe_text = eval_inputs
            .dataset
            .test()
            .rows()
            .first()
            .map(|row| row.input.clone())
            .ok_or_else(|| {
                CliError::ValidationFailed(
                    "the canonical test split has no rows, so there is nothing to probe with"
                        .to_string(),
                )
            })?;
        let probe_path = write_probe_text(request.bench_dir, request.cell, &probe_text)?;
        let cold = resource::measure_cold(&artifact_path, None, &probe_path)?;

        // (7) WARM + THROUGHPUT, against the RELOADED model in this process.
        let warm_request = ClassifyRequestDocument::new(std::iter::once(probe_text.clone()));
        let mut backend_identity = String::new();
        let warm_latency_ms_median = resource::warm_latency_ms_median(WARMUP_COUNT, || {
            let response = credential
                .model()
                .classify(&warm_request)
                .map_err(|error| CliError::InferenceFailed(error.to_string()))?;
            // BACKEND IDENTITY, READ FROM EXECUTION (Ph4 D-12). `ClassifyResponse::backend`
            // is `ExecutionBackend::identity` called on the value the encode invocation
            // RETURNED — there is no parameter, no setter and no configuration path that
            // reaches it. Echoing a device string from `--config` here would produce a row
            // that says GPU because somebody typed GPU.
            backend_identity = response.backend().to_string();
            Ok(())
        })?;

        let test_rows_data = eval_inputs.dataset.test().rows();
        let throughput_rows_per_sec = resource::throughput_rows_per_sec(
            test_rows_data.len(),
            THROUGHPUT_BATCH_SIZE,
            |from, to| {
                let batch = ClassifyRequestDocument::new(
                    test_rows_data[from..to].iter().map(|row| row.input.clone()),
                );
                credential
                    .model()
                    .classify(&batch)
                    .map_err(|error| CliError::InferenceFailed(error.to_string()))?;
                Ok(())
            },
        )?;

        // (8) THE EVALUATION, through the Phase 3 lock chain. Validation FIRST (it is what a
        //     selection may be made on), then the lock is COMMITTED TO DISK, then the token,
        //     then the grant, then the test rows.
        let validation_rows = evaluate_rows_from_artifact(
            &credential,
            &eval_inputs.dataset,
            EvaluatedSplit::Validation,
        )
        .map_err(|error| CliError::ValidationFailed(error.to_string()))?;

        let scalar =
            evaluate_validation_from_artifact(&credential, &eval_inputs.dataset, EVAL_METRIC)
                .map_err(|error| CliError::ValidationFailed(error.to_string()))?;
        let config_hash = config_hash_of(&credential);
        let lock = create_selection_lock(
            &credential,
            vec![SelectionCandidate::from_evaluation(&config_hash, scalar)],
            EVAL_RULE,
        )
        .map_err(|error| {
            CliError::ValidationFailed(format!(
                "the selection lock could not be committed: {error}"
            ))
        })?;

        // COMMIT THE LOCK BYTES. The row's `lock_hash` is a CLAIM; this file is the evidence,
        // and 05-10's gate recomputes the digest from these bytes rather than trusting the
        // field. `force = true` because the row file is the write-once artifact — a re-run
        // that got past the row's no-clobber gate is entitled to rewrite its own lock.
        let lock_bytes = lock.to_canonical_bytes();
        let lock_rel = lock_relative_path(request.cell);
        super::atomic_write(&request.bench_dir.join(&lock_rel), &lock_bytes, true)?;
        let committed_lock_hash = sha256_hex(&lock_bytes);

        let token = lock.mint_test_token(&credential).map_err(|error| {
            CliError::ValidationFailed(format!(
                "the selection lock does not admit this artifact: {error}"
            ))
        })?;
        let grant = CanonicalTestAccess::grant(token, &credential, &eval_inputs.dataset).map_err(
            |error| {
                CliError::ValidationFailed(format!("canonical test access was refused: {error}"))
            },
        )?;
        let test_rows = evaluate_rows_from_artifact(
            &credential,
            &eval_inputs.dataset,
            EvaluatedSplit::Test(&grant),
        )
        .map_err(|error| CliError::ValidationFailed(error.to_string()))?;

        // (9) THE QUALITY BLOCK. Test rows supply the accuracy family, validation rows the
        //     calibration diagnostics — SEPARATE PARAMETERS, so there is no argument order
        //     that feeds test probabilities to the calibration functions (D-07).
        let ordered_labels: Vec<String> = credential.model().ordered_labels().to_vec();
        let quality = assemble_quality_block(&test_rows, &validation_rows, &ordered_labels)
            .map_err(|error| CliError::ValidationFailed(error.to_string()))?;

        // (10) THE ROW.
        let payload = BenchRowPayload {
            schema_version: BENCH_ROW_SCHEMA_VERSION,
            contract_id: CLAIMS_CONTRACT_ID.to_string(),
            method: request.cell.method,
            shots: request.cell.shots,
            seed: request.cell.seed,
            dataset_revision,
            dataset_fingerprint,
            model_revision: PINNED_REVISION.to_string(),
            selection_manifest_hash,
            backend_identity,
            host: host_identity(),
            quality,
            resource: ResourceBlock {
                train_wall_ms,
                cold_latency_ms: cold.cold_latency_ms,
                // Always true on a row this adapter writes: `measure_cold` has no in-process
                // path, so there is no branch here that could set it false.
                cold_measured_in_child_process: true,
                warm_latency_ms_median,
                throughput_rows_per_sec,
                throughput_batch_size: THROUGHPUT_BATCH_SIZE,
                warmup_count: WARMUP_COUNT,
                train_peak_rss_bytes: train_peak.bytes,
                train_peak_rss_mechanism: train_peak.mechanism.clone(),
                inference_peak_rss_bytes: cold.peak_rss_bytes,
                inference_peak_rss_mechanism: cold.peak_rss_mechanism.to_string(),
                peak_rss_sample_interval_hz: train_peak.sample_interval_hz,
                artifact_bytes,
                // SetFit ships ONE standalone file: the encoder, the tokenizer identity, the
                // pooling policy and the head are all inside it. The equality is therefore a
                // FACT about the format, not a copy-paste.
                deployable_total_bytes: artifact_bytes,
            },
            evidence: MethodEvidence::Setfit(SetfitEvidence {
                evidence_table_hash,
                apr_artifact_sha256,
                lock: BenchLockRef {
                    lock_hash: committed_lock_hash,
                    role: "written".to_string(),
                    rule: lock.rule().to_string(),
                    lock_record_path: lock_rel,
                },
            }),
        };

        let (row_path, row_hash) = emit_row(request.bench_dir, payload, request.force)?;
        report(request.json, request.cell, &row_path, &row_hash)
    }

    /// Where this cell's artifact lands inside the bench directory.
    fn artifact_path(bench_dir: &Path, request: &CellRequest<'_>) -> PathBuf {
        bench_dir.join("artifacts").join(format!(
            "{}-s{}-seed{}.apr",
            request.cell.method.tag(),
            request.cell.shots,
            request.cell.seed
        ))
    }

    /// The hex SHA-256 over the artifact document's canonical `requested_config` sub-document.
    ///
    /// The SAME derivation `apr eval` records as `CONFIG_HASH_DERIVATION`, so a lock this
    /// command writes and a lock `apr eval` writes carry comparable candidate identities.
    fn config_hash_of(credential: &ReloadedSetFitCredential) -> String {
        use sha2::Digest as _;
        let requested = &credential.model().doc_view().requested_config;
        let bytes = serde_json::to_vec(requested).unwrap_or_default();
        aprender_contrastive_data::hash::hex(&sha2::Sha256::digest(&bytes).into())
    }

    /// Map a lifecycle failure onto the CLI surface.
    fn train_error(error: entrenar::train::setfit::SetFitTrainError) -> CliError {
        CliError::ValidationFailed(format!("setfit bench cell: {error}"))
    }

    /// What the command says when a cell lands.
    fn report(
        json: bool,
        cell: entrenar::train::setfit::bench_row::CellKey,
        row_path: &Path,
        row_hash: &str,
    ) -> Result<()> {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "command": "setfit-bench-run",
                    "cell": cell.render(),
                    "row": row_path.display().to_string(),
                    "row_sha256": row_hash,
                    "executed": true,
                })
            );
        } else {
            println!("{cell} -> {} ({row_hash})", row_path.display());
        }
        Ok(())
    }
}

// ==========================================================================================
// The LoRA cell path
// ==========================================================================================

/// The 9B LoRA baseline cell.
///
/// # The reload route is 05-06 Task 3's, called BY NAME
///
/// `ClassifyPipeline::load_adapter` and `ClassifyPipeline::predict_proba_tokenized` are the
/// two entries that plan's preflight PROVED, in a fresh process, with a two-sided control
/// (fresh-process vs in-process max |diff| 0.000000000; with-adapter vs without 0.194929659).
/// Before it, no adapter load route existed: `ClassifyPipeline::from_apr` builds FRESH LoRA
/// layers and `resume_from_apr_checkpoint` installs tensors through `if let Ok(..)`, a silent
/// partial load by construction. Forty remote 9B cells against an improvised reload is the
/// failure mode that ordering exists to prevent, so this module invents nothing.
///
/// # The training call is `run_classify_core`, not a second trainer
///
/// One implementation drives both `apr finetune --task classify` and this cell. The
/// `TrainingConfig` is constructed here EXPLICITLY — contracted seed, `val_split 0.0`,
/// `early_stopping_patience 0` — so no default literal reaches the benchmark path.
pub(crate) mod lora {
    use std::io::Write as _;
    use std::path::{Path, PathBuf};

    use entrenar::finetune::{ClassifyConfig, ClassifyPipeline, SafetySample};
    use entrenar::train::setfit::bench_metrics::assemble_quality_block;
    use entrenar::train::setfit::bench_row::{
        sha256_hex, BenchRowPayload, LoraEvidence, MethodEvidence, ResourceBlock,
        BENCH_ROW_SCHEMA_VERSION, CLAIMS_CONTRACT_ID, WARMUP_COUNT,
    };
    use entrenar::transformer::{Transformer, TransformerConfig};

    use crate::commands::finetune::{ClassifyOutcome, ClassifyRun, ClassifySelection};

    use super::resource::{self, TrainRssSampler, THROUGHPUT_BATCH_SIZE};
    use super::{
        emit_row, host_identity, ledger_relative_path, read_bounded, read_phase2, write_probe_text,
        CellRequest, CliError, Result,
    };

    // ---- The FROZEN published defaults (D-07). No per-cell knob exists. ---------------------

    /// The three TweetEval stance classes.
    const NUM_CLASSES: usize = 3;
    /// Published `apr finetune --task classify` default.
    const LORA_RANK: u32 = 16;
    /// Published default.
    const LEARNING_RATE: f64 = 1e-4;
    /// Published default.
    const EPOCHS: u32 = 3;
    /// Published default.
    const MAX_SEQ_LEN: usize = 512;
    /// No validation is carved, so there is nothing to select an epoch on.
    const VAL_SPLIT: f32 = 0.0;
    /// `0` means DISABLED — the meaning 05-06 corrected. It used to mean "stop after one
    /// epoch", which is the opposite of the intent.
    const EARLY_STOPPING_PATIENCE: usize = 0;
    /// The checkpoint format `apr finetune` writes.
    const CHECKPOINT_FORMAT: &str = "apr";
    /// The adapter file `ClassifyTrainer::save_adapter_apr` writes into each epoch directory.
    const ADAPTER_FILE: &str = "model.adapter.apr";

    /// Execute one LoRA cell.
    #[allow(clippy::too_many_lines)]
    pub(super) fn execute(request: &CellRequest<'_>) -> Result<()> {
        let base_model = request.base_model.ok_or_else(|| {
            CliError::ValidationFailed(
                "--base-model <FILE> is required for a lora cell. An adapter alone is not \
                 deployable, so the row records base_model_bytes and deployable_total_bytes = \
                 base + adapter; neither is computable without the base."
                    .to_string(),
            )
        })?;
        if request.config.is_some() {
            return Err(CliError::ValidationFailed(
                "--config belongs to the setfit method. The LoRA cell runs the FROZEN published \
                 `apr finetune --task classify` defaults, which is what makes the comparison a \
                 comparison of methods rather than of tuning effort."
                    .to_string(),
            ));
        }

        // (1) Phase 2's artifacts, through the SAME door the setfit cell uses.
        let phase2 = read_phase2(request)?;
        let selection = ClassifySelection {
            samples: selected_samples(&phase2)?,
            semantic_hash: phase2.manifest_semantic_hash.clone(),
        };

        // (2) THE CANDIDATE LEDGER, appended BEFORE training starts and therefore before any
        //     test access. `no_selection_attestation` claims no uncontracted model selection
        //     happened; a claim nothing can contradict is not evidence. The ledger is what
        //     makes it checkable: 05-10 recomputes its digest and requires exactly one line.
        let ledger_rel = ledger_relative_path(request.cell);
        let ledger_path = request.bench_dir.join(&ledger_rel);
        let config_hash = config_hash(base_model);
        append_candidate(
            &ledger_path,
            request.cell,
            request.force,
            &phase2.manifest_semantic_hash,
            &config_hash,
        )?;

        // (3) THE TRAINING CALL. Explicit configuration, no literal from a default table.
        let checkpoint_dir = request.bench_dir.join("lora").join(format!(
            "{}-s{}-seed{}",
            request.cell.method.tag(),
            request.cell.shots,
            request.cell.seed
        ));
        let training = entrenar::finetune::TrainingConfig {
            epochs: EPOCHS as usize,
            val_split: VAL_SPLIT,
            save_every: EPOCHS as usize,
            early_stopping_patience: EARLY_STOPPING_PATIENCE,
            checkpoint_dir: checkpoint_dir.clone(),
            seed: u64::from(request.cell.seed),
            log_interval: 1,
            distributed: None,
            ..entrenar::finetune::TrainingConfig::default()
        };
        let run = ClassifyRun {
            model_path: Some(base_model),
            model_size: None,
            data_path: None,
            output_path: Some(&checkpoint_dir),
            num_classes: NUM_CLASSES,
            rank: LORA_RANK,
            epochs: EPOCHS,
            learning_rate: LEARNING_RATE,
            plan_only: false,
            checkpoint_format: CHECKPOINT_FORMAT,
            oversample: false,
            max_seq_len: Some(MAX_SEQ_LEN),
            quantize_nf4: false,
            gpus: None,
            gpu_backend: "auto",
            role: None,
            bind: None,
            coordinator: None,
            expect_workers: None,
            json_output: request.json,
        };

        let sampler = TrainRssSampler::start();
        let outcome =
            crate::commands::finetune::run_classify_core(&run, training, Some(&selection))?
                .ok_or_else(|| {
                    CliError::ValidationFailed(
                        "the classify run returned without training. A benchmark cell has no \
                     non-training path — this is a defect, not an empty result."
                            .to_string(),
                    )
                })?;
        let train_peak = sampler.finish();

        // (4) THE ATTESTATION'S OWN PRECONDITION. `epochs_completed == epochs_requested` is
        //     what makes "no epoch was selected on a metric" checkable from the row alone.
        if outcome.epochs_completed != outcome.epochs_requested || outcome.stopped_early {
            return Err(CliError::ValidationFailed(format!(
                "the cell requested {} epochs and completed {} (stopped_early={}). A row \
                 attesting no selection cannot be written from a run that ended somewhere the \
                 configuration did not ask for.",
                outcome.epochs_requested, outcome.epochs_completed, outcome.stopped_early
            )));
        }
        // `best/` is the ACTUAL model-selection surface (05-06's finding). Its existence would
        // mean an epoch was chosen on a metric, which is exactly what the attestation denies.
        let best_dir = outcome.checkpoint_dir.join("best");
        if best_dir.exists() {
            return Err(CliError::ValidationFailed(format!(
                "{} exists, so an epoch was selected on a validation metric. \
                 no_selection_attestation cannot be written over that.",
                best_dir.display()
            )));
        }

        // (5) THE WRITTEN ADAPTER. Resolved by NAME from the epoch the run says it completed,
        //     never by "whatever the newest directory is": `save_checkpoint`'s result is
        //     discarded inside the trainer, so a failed final write would otherwise leave an
        //     EARLIER epoch's adapter to be attested as the trained model.
        let final_epoch = outcome.epochs_completed.saturating_sub(1);
        let adapter_path = outcome
            .checkpoint_dir
            .join(format!("epoch-{final_epoch}"))
            .join(ADAPTER_FILE);
        if !adapter_path.is_file() {
            return Err(CliError::ValidationFailed(format!(
                "{} is missing. The run reported {} completed epochs, so this is the adapter \
                 the row would attest — and the trainer discards its own checkpoint-write \
                 result, so an absent file here means the final write failed silently.",
                adapter_path.display(),
                outcome.epochs_completed
            )));
        }

        let adapter_bytes = std::fs::metadata(&adapter_path)
            .map_err(CliError::Io)?
            .len();
        let base_bytes = std::fs::metadata(base_model).map_err(CliError::Io)?.len();
        let adapter_sha256 = sha256_of_file(&adapter_path)?;
        let base_model_sha256 = base_model_digest(base_model)?;

        // (6) THE RELOAD, then the measurements — mirroring the SetFit path's discipline.
        //     Never the in-memory final-epoch training state: EVAL-05's numbers come from the
        //     production artifacts a user would actually ship.
        let probe_text = phase2
            .dataset
            .test()
            .rows()
            .first()
            .map(|row| row.input.clone())
            .ok_or_else(|| {
                CliError::ValidationFailed(
                    "the canonical test split has no rows, so there is nothing to probe with"
                        .to_string(),
                )
            })?;
        let probe_path = write_probe_text(request.bench_dir, request.cell, &probe_text)?;
        let cold = resource::measure_cold(&adapter_path, Some(base_model), &probe_path)?;

        let mut pipeline = reload(
            base_model,
            &adapter_path,
            &outcome.model_config,
            outcome.classify_config.clone(),
        )?;

        let warm_tokens = tokenize_one(&pipeline, &probe_text)?;
        let warm_latency_ms_median = resource::warm_latency_ms_median(WARMUP_COUNT, || {
            let probabilities = pipeline.predict_proba_tokenized(&warm_tokens);
            if probabilities.len() != NUM_CLASSES {
                return Err(CliError::InferenceFailed(format!(
                    "the reloaded pipeline returned {} probabilities for {NUM_CLASSES} classes",
                    probabilities.len()
                )));
            }
            Ok(())
        })?;

        // (7) THE TEST-SPLIT PASS. One loop produces BOTH the throughput measurement and the
        //     per-row probability vectors, so the numbers a row publishes and the pass they
        //     were timed over cannot be two different passes.
        let test_rows = phase2.dataset.test().rows();
        let tokenized: Vec<Vec<u32>> = test_rows
            .iter()
            .map(|row| tokenize_one(&pipeline, &row.input))
            .collect::<Result<_>>()?;
        let mut probabilities: Vec<Vec<f64>> = Vec::with_capacity(test_rows.len());
        let throughput_rows_per_sec = resource::throughput_rows_per_sec(
            test_rows.len(),
            THROUGHPUT_BATCH_SIZE,
            |from, to| {
                for tokens in &tokenized[from..to] {
                    let row = pipeline.predict_proba_tokenized(tokens);
                    if row.len() != NUM_CLASSES {
                        return Err(CliError::InferenceFailed(format!(
                            "the reloaded pipeline returned {} probabilities for {NUM_CLASSES} \
                             classes",
                            row.len()
                        )));
                    }
                    probabilities.push(row.into_iter().map(f64::from).collect());
                }
                Ok(())
            },
        )?;

        // (8) THE QUALITY BLOCK — assembled by the SAME method-agnostic library function the
        //     SetFit cell uses. Two assemblies would be two definitions of `F_avg`.
        let ordered_labels: Vec<String> = phase2.dataset.label_names().to_vec();
        let truth: Vec<usize> = test_rows.iter().map(|row| row.label).collect();
        let test_predictions = row_predictions_from_probabilities(
            &probabilities,
            &truth,
            &ordered_labels,
            &adapter_sha256,
            "test",
        )?;
        // Calibration is validation-only (D-07), so the validation split gets its own pass.
        let validation_rows = phase2.dataset.validation().rows();
        let validation_predictions = {
            let mut rows: Vec<Vec<f64>> = Vec::with_capacity(validation_rows.len());
            for row in validation_rows {
                let tokens = tokenize_one(&pipeline, &row.input)?;
                rows.push(
                    pipeline
                        .predict_proba_tokenized(&tokens)
                        .into_iter()
                        .map(f64::from)
                        .collect(),
                );
            }
            let truth: Vec<usize> = validation_rows.iter().map(|row| row.label).collect();
            row_predictions_from_probabilities(
                &rows,
                &truth,
                &ordered_labels,
                &adapter_sha256,
                "validation",
            )?
        };
        let quality =
            assemble_quality_block(&test_predictions, &validation_predictions, &ordered_labels)
                .map_err(|error| CliError::ValidationFailed(error.to_string()))?;

        // (9) THE LEDGER'S FINAL STATE, digested AFTER the run.
        let ledger_bytes = read_bounded(&ledger_path)?;
        let candidates_trained = u32::try_from(
            ledger_bytes
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .count(),
        )
        .unwrap_or(u32::MAX);
        if candidates_trained != 1 {
            return Err(CliError::ValidationFailed(format!(
                "{} carries {candidates_trained} candidate lines; the contract requires exactly \
                 1. More than one candidate for a cell IS the uncontracted model selection the \
                 attestation says did not happen.",
                ledger_path.display()
            )));
        }

        let payload = BenchRowPayload {
            schema_version: BENCH_ROW_SCHEMA_VERSION,
            contract_id: CLAIMS_CONTRACT_ID.to_string(),
            method: request.cell.method,
            shots: request.cell.shots,
            seed: request.cell.seed,
            dataset_revision: phase2.dataset_revision.clone(),
            dataset_fingerprint: phase2
                .dataset
                .validation_witness()
                .dataset_fingerprint_hex(),
            model_revision: base_model_sha256.clone(),
            selection_manifest_hash: phase2.manifest_semantic_hash.clone(),
            // BACKEND IDENTITY FROM EXECUTION (Ph4 D-12). `outcome.gpu` is what the PIPELINE
            // reported; the cpu form cannot fabricate a GPU name because there is no branch
            // here that reads a device flag. Verification Discipline rule 2: if the row says
            // GPU, a pipeline-read GPU name exists.
            backend_identity: backend_identity(outcome.gpu.as_ref()),
            host: host_identity(),
            quality,
            resource: ResourceBlock {
                train_wall_ms: outcome.total_time_ms,
                cold_latency_ms: cold.cold_latency_ms,
                cold_measured_in_child_process: true,
                warm_latency_ms_median,
                throughput_rows_per_sec,
                throughput_batch_size: THROUGHPUT_BATCH_SIZE,
                warmup_count: WARMUP_COUNT,
                train_peak_rss_bytes: train_peak.bytes,
                train_peak_rss_mechanism: train_peak.mechanism.clone(),
                inference_peak_rss_bytes: cold.peak_rss_bytes,
                inference_peak_rss_mechanism: cold.peak_rss_mechanism.to_string(),
                peak_rss_sample_interval_hz: train_peak.sample_interval_hz,
                // THE SIZE SPLIT (review consensus item 8). `artifact_bytes` is the ADAPTER
                // ALONE, which is the honest answer to "what did this method produce"; the
                // deployable size is base + adapter, which is the only field a cross-method
                // size claim may be built on. An adapter-only figure standing in for a
                // deployable size would understate LoRA by three orders of magnitude.
                artifact_bytes: adapter_bytes,
                deployable_total_bytes: base_bytes.saturating_add(adapter_bytes),
            },
            evidence: MethodEvidence::Lora(LoraEvidence {
                base_model_sha256,
                base_model_bytes: base_bytes,
                adapter_sha256,
                epochs_requested: outcome.epochs_requested,
                epochs_completed: outcome.epochs_completed,
                early_stopping_disabled: EARLY_STOPPING_PATIENCE == 0,
                val_split: f64::from(VAL_SPLIT),
                no_selection_attestation: true,
                candidate_ledger_sha256: sha256_hex(&ledger_bytes),
                candidates_trained,
                candidate_ledger_path: ledger_rel,
            }),
        };

        let (row_path, row_hash) = emit_row(request.bench_dir, payload, request.force)?;
        if request.json {
            println!(
                "{}",
                serde_json::json!({
                    "command": "setfit-bench-run",
                    "cell": request.cell.render(),
                    "row": row_path.display().to_string(),
                    "row_sha256": row_hash,
                    "executed": true,
                })
            );
        } else {
            println!("{} -> {} ({row_hash})", request.cell, row_path.display());
        }
        Ok(())
    }

    /// The replayed rows, as the trainer's own sample type.
    ///
    /// Through `finetune::resolve_selected_samples`, which is the mapping the CLI flag path
    /// already uses: the selection's contracted order (class ascending, draw order within a
    /// class) is a property of that function, and a second mapping here would be a second
    /// order that could disagree with the one `apr finetune` trains on.
    fn selected_samples(phase2: &super::Phase2) -> Result<Vec<SafetySample>> {
        let ordered: Vec<(&str, usize)> = phase2
            .selection
            .examples()
            .iter()
            .map(|example| (example.id.as_str(), example.label))
            .collect();
        crate::commands::finetune::resolve_selected_samples(&ordered, phase2.dataset.train().rows())
    }

    /// Append EXACTLY ONE candidate line, refusing a second without `--force`.
    ///
    /// Opened with `append(true)`, never truncate and never rewrite: the ledger's value is
    /// that it accumulates. A writer that could shorten it could erase the second candidate
    /// it is here to reveal.
    /// Takes plain values rather than a `&Phase2` so it is directly unit-testable: a
    /// `PreparedDataset` cannot be built in a unit test, and a refusal that can only be
    /// exercised through a training run is a refusal nothing checks.
    pub(super) fn append_candidate(
        ledger_path: &Path,
        cell: entrenar::train::setfit::bench_row::CellKey,
        force: bool,
        selection_manifest_hash: &str,
        config_hash: &str,
    ) -> Result<()> {
        if ledger_path.exists() {
            let existing = read_bounded(ledger_path)?;
            let lines = existing
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
                .count();
            if lines > 0 && !force {
                return Err(CliError::ValidationFailed(format!(
                    "{} already records {lines} candidate(s) for cell {cell}. A SECOND \
                     candidate for one cell is exactly the uncontracted model selection this \
                     cell's attestation says did not happen — pass --force only if you are \
                     deliberately replacing the run that produced the published number.",
                    ledger_path.display(),
                )));
            }
        }
        if let Some(parent) = ledger_path.parent() {
            std::fs::create_dir_all(parent).map_err(CliError::Io)?;
        }
        // `--force` starts a FRESH ledger rather than appending to the old one, so
        // `candidates_trained == 1` stays a true statement about the run being recorded rather
        // than a count of every run this directory has ever seen.
        let line = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "selection_manifest_hash": selection_manifest_hash,
            "epochs_requested": EPOCHS,
            "seed": cell.seed,
            "config_hash": config_hash,
        });
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(!force)
            .write(true)
            .truncate(force)
            .open(ledger_path)
            .map_err(CliError::Io)?;
        writeln!(file, "{line}").map_err(CliError::Io)?;
        file.sync_all().map_err(CliError::Io)?;
        Ok(())
    }

    /// The reloaded base + adapter pipeline, through the route 05-06 Task 3 proved.
    pub(crate) fn reload_base_and_adapter(base: &Path, adapter: &Path) -> Result<ClassifyPipeline> {
        // The cold probe reaches this without an outcome in hand, so the architecture comes
        // from the base artifact's own metadata — the same resolver `apr finetune` uses.
        let model_config =
            crate::commands::model_config::resolve_transformer_config(Some(base), None)?;
        let classify_config = ClassifyConfig {
            num_classes: NUM_CLASSES,
            lora_rank: LORA_RANK as usize,
            lora_alpha: LORA_RANK as f32,
            #[allow(clippy::cast_possible_truncation)]
            learning_rate: LEARNING_RATE as f32,
            epochs: EPOCHS as usize,
            max_seq_len: MAX_SEQ_LEN,
            ..ClassifyConfig::default()
        };
        reload(base, adapter, &model_config, classify_config)
    }

    /// Build the pipeline and install the written adapter into it.
    fn reload(
        base: &Path,
        adapter: &Path,
        model_config: &TransformerConfig,
        classify_config: ClassifyConfig,
    ) -> Result<ClassifyPipeline> {
        let transformer = Transformer::from_apr(base, model_config)
            .map_err(|error| CliError::ModelLoadFailed(format!("{}: {error}", base.display())))?;
        let mut pipeline = ClassifyPipeline::from_model(transformer, model_config, classify_config);
        // `load_adapter` reads and shape-checks EVERY tensor before installing any, so a
        // refusal leaves the pipeline untouched. A partial adapter classifies confidently and
        // is not the model that was trained.
        pipeline.load_adapter(adapter).map_err(|error| {
            CliError::ModelLoadFailed(format!("{}: {error}", adapter.display()))
        })?;
        Ok(pipeline)
    }

    /// Tokenize one text through the pipeline's OWN tokenizer.
    ///
    /// `pre_tokenize` is the pipeline's single tokenization implementation — the same one
    /// training used. Reproducing its byte-level fallback here would be a second tokenizer,
    /// and a benchmark whose evaluation tokenizes differently from its training is measuring
    /// something other than the model.
    pub(crate) fn tokenize_one(pipeline: &ClassifyPipeline, text: &str) -> Result<Vec<u32>> {
        let sample = SafetySample {
            input: text.to_string(),
            label: 0,
        };
        pipeline
            .pre_tokenize(std::slice::from_ref(&sample))
            .into_iter()
            .next()
            .map(|tokenized| tokenized.token_ids)
            .ok_or_else(|| {
                CliError::InferenceFailed(
                    "pre_tokenize returned no sample for one input".to_string(),
                )
            })
    }

    /// `<device>:<implementation>:<kernel>`, three segments like the SetFit side's.
    ///
    /// The cpu arm cannot fabricate a GPU string: it is reached exactly when the pipeline
    /// reported no GPU, and it names no device it did not observe.
    fn backend_identity(gpu: Option<&(String, usize)>) -> String {
        match gpu {
            Some((name, _)) => format!("gpu({name}):entrenar-classify:lora"),
            None => "cpu:entrenar-classify:lora".to_string(),
        }
    }

    /// SHA-256 over a file's bytes, streamed.
    fn sha256_of_file(path: &Path) -> Result<String> {
        use sha2::Digest as _;
        let mut file = std::fs::File::open(path).map_err(CliError::Io)?;
        let mut hasher = sha2::Sha256::new();
        std::io::copy(&mut file, &mut hasher).map_err(CliError::Io)?;
        let digest: [u8; 32] = hasher.finalize().into();
        Ok(aprender_contrastive_data::hash::hex(&digest))
    }

    /// The base model's digest, CACHED beside the weights.
    ///
    /// A 9B base is hashed once per host rather than once per cell: forty cells x a
    /// multi-gigabyte read is an hour of I/O that measures nothing. The cache is keyed by the
    /// file's own path and is re-derived whenever it is absent, so a stale cache is a missing
    /// cache rather than a wrong digest.
    fn base_model_digest(base: &Path) -> Result<String> {
        let cache = base.with_extension("sha256");
        if cache.is_file() {
            let cached = String::from_utf8(read_bounded(&cache)?)
                .map_err(|error| CliError::InvalidFormat(error.to_string()))?;
            let cached = cached.trim().to_string();
            if cached.len() == 64 && cached.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Ok(cached);
            }
        }
        let digest = sha256_of_file(base)?;
        // Best effort, and through the SHARED atomic writer like every other file this module
        // creates: an unwritable directory must not fail the cell (the digest is already
        // computed and correct), but a half-written cache would be read back as a corrupt
        // digest on the next cell and refused there instead of here.
        let _ = super::atomic_write(&cache, format!("{digest}\n").as_bytes(), true);
        Ok(digest)
    }

    /// A stable digest over the knobs this cell ran, for the ledger line.
    fn config_hash(base_model: &Path) -> String {
        let document = serde_json::json!({
            "base_model": base_model.display().to_string(),
            "num_classes": NUM_CLASSES,
            "lora_rank": LORA_RANK,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "max_seq_len": MAX_SEQ_LEN,
            "val_split": VAL_SPLIT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        });
        sha256_hex(document.to_string().as_bytes())
    }

    /// Wrap per-row probabilities in the library's evidence type.
    ///
    /// `RowPredictions` has no public constructor by design — a value of it is EVIDENCE that a
    /// measurement happened through the credentialed SetFit door. The LoRA method has no such
    /// door and never will: its artifacts are not `setfit-apr-v1`. So the vectors are handed
    /// to the assembly through the library's declared LoRA entry rather than by forging the
    /// SetFit witness.
    fn row_predictions_from_probabilities(
        probabilities: &[Vec<f64>],
        truth: &[usize],
        ordered_labels: &[String],
        artifact_hash: &str,
        split_tag: &'static str,
    ) -> Result<entrenar::train::setfit::apr_evaluate::RowPredictions> {
        entrenar::train::setfit::apr_evaluate::row_predictions_from_lora(
            probabilities,
            truth,
            ordered_labels,
            artifact_hash,
            split_tag,
        )
        .map_err(|error| CliError::ValidationFailed(error.to_string()))
    }

    /// The checkpoint directory layout this module resolves against, named for the tests.
    #[must_use]
    pub(crate) fn adapter_path_for(checkpoint_dir: &Path, epochs_completed: u32) -> PathBuf {
        checkpoint_dir
            .join(format!("epoch-{}", epochs_completed.saturating_sub(1)))
            .join(ADAPTER_FILE)
    }
}

// ==========================================================================================
// `apr setfit bench report` — verified data only, estimation-first (D-13/D-15, EVAL-04)
// ==========================================================================================

/// The report: verify the whole directory, then render what was verified and nothing else.
///
/// # This module renders. It decides nothing.
///
/// Completeness, pairing, provenance recomputation and every statistic live in
/// `entrenar::train::setfit::bench_gate`. `aggregate` takes a `VerifiedRunSet`, which has no
/// public constructor, so this adapter CANNOT print a partial table even by mistake: there is
/// no value it could compute one from.
///
/// # The two things a renderer can get wrong that no arithmetic check would catch
///
/// 1. **A technically-true table that reads as a like-for-like benchmark.** Every resource
///    column therefore carries its mechanism string and its measurement scope, and a row that
///    puts a `sysinfo_sampled_*` LOWER BOUND beside a `child_max_rss_*` exact kernel high-water
///    mark prints [`INCOMPARABLE_NOTE`] on that row. The note is proven TWO-SIDED — a
///    same-mechanism control asserts its absence — so it is not unconditional boilerplate a
///    reader learns to skip.
/// 2. **An adapter's bytes standing in for a deployable model size.** The cross-method size
///    table uses `deployable_total_bytes` only. LoRA's adapter-only `artifact_bytes` appears in
///    the per-method detail, labelled, and never in the comparison.
pub(crate) mod report {
    use std::path::Path;

    use entrenar::train::setfit::bench_gate::{
        aggregate, mechanisms_are_comparable, verify_run, MethodShotQuality, MethodShotResource,
        RunAggregate, SeriesSummary, ShotDelta,
    };
    use entrenar::train::setfit::bench_row::{Method, RunManifest, BENCH_METHODS, BENCH_SHOTS};

    use super::{read_bounded, RUN_MANIFEST_FILE};
    use crate::commands::setfit_train::atomic_write;
    use crate::error::{CliError, Result};

    /// Everything `bench report` carries, resolved from clap.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct BenchReportArgs<'a> {
        /// Where rows, locks, ledgers and the run manifest live.
        pub(crate) bench_dir: &'a Path,
        /// Emit the machine-readable detail on stdout instead of the human tables.
        pub(crate) json: bool,
        /// Also write the machine-readable detail to this file.
        pub(crate) out: Option<&'a Path>,
    }

    // --- Pinned strings. Constants rather than inline literals so the tests assert the SHIPPED
    // --- text rather than a copy of it that can drift.

    /// The quality table's header.
    pub(crate) const QUALITY_TABLE_HEADER: &str =
        "QUALITY - official F_avg, mean +/- (n-1) std [min, max]";

    /// The paired-delta table's header.
    ///
    /// RETAINED AND DEFERRED. Never emitted under the active scope, and pinned as a
    /// MUST-NOT-MATCH literal by the rendering case table — the ACTUAL string the two-method
    /// renderer emitted, not an invented near-miss.
    pub(crate) const DELTA_TABLE_HEADER: &str =
        "PAIRED DELTA - F_avg(setfit) - F_avg(lora), same selection manifest, 95% CI";

    /// The per-method resource detail's header.
    pub(crate) const RESOURCE_TABLE_HEADER: &str =
        "RESOURCE - per method and host, every figure with its measurement boundary";

    /// The cross-method resource comparison's header.
    ///
    /// RETAINED AND DEFERRED. Same rule as [`DELTA_TABLE_HEADER`].
    pub(crate) const RESOURCE_COMPARISON_HEADER: &str =
        "RESOURCE COMPARISON - setfit beside lora, mechanism-labelled at the point of comparison";

    /// The distinctive fragment of one cross-method comparison ROW.
    ///
    /// Copied out of `comparison_row`'s own format string, so the case table asserts the
    /// SHIPPED format rather than a copy of it that can drift.
    pub(crate) const COMPARISON_ROW_MARKER: &str = "  |  lora ";

    /// The two-method report's title line. Retained, deferred, a MUST-NOT-MATCH literal.
    pub(crate) const TWO_METHOD_TITLE: &str = "SetFit vs LoRA - benchmark claims report";

    /// The ACTIVE report's title line. One method, and it says so in the first line.
    pub(crate) const SINGLE_METHOD_TITLE: &str = "SetFit - benchmark claims report";

    /// The cross-method size table's header. It NAMES `deployable`, because the field it uses is
    /// the claim.
    pub(crate) const SIZE_TABLE_HEADER: &str =
        "MODEL SIZE - deployable_total_bytes, what a user must ship to serve this";

    /// The D-09 framing line the TWO-METHOD resource sections carried.
    ///
    /// RETAINED AND DEFERRED (2.0.0, D-19). It describes a two-host design that the active
    /// scope does not have, so the active report no longer emits it: a framing line describing
    /// a design that was not run is a claim about the report. Kept, not deleted, because
    /// D-ITEM-05-15 restores exactly the design it describes.
    pub(crate) const PER_HOST_FRAMING_TWO_HOST: &str =
        "as-deployed method costs; hosts differ by design and are never averaged together";

    /// The framing line the ACTIVE, single-method resource section carries.
    pub(crate) const PER_HOST_FRAMING: &str =
        "as-deployed costs on ONE host for ONE method; every figure carries its measurement \
         boundary, and no figure here is averaged across hosts";

    /// The label a sampled peak-RSS figure carries EVERYWHERE it is rendered.
    ///
    /// A poll at any finite rate can miss the peak entirely, the bias is one-directional, and
    /// its magnitude depends on the allocation pattern — so a reader who sees the number must
    /// see the bound, in the per-shot summary and not only in a methods paragraph.
    pub(crate) const SAMPLED_LOWER_BOUND_LABEL: &str = "LOWER BOUND (sampled; can only understate)";

    /// The within-row mechanism-asymmetry note the active resource section carries.
    pub(crate) const WITHIN_ROW_ASYMMETRY_NOTE: &str =
        "The two figures above are TWO metrics from TWO processes, and here they come from two \
         different mechanism CLASSES. They are never added, never averaged, and never reduced \
         to a single figure: a kernel high-water mark is process-cumulative, so the training \
         process's peak is not the inference peak.";

    /// The note a mixed-mechanism comparison row carries.
    pub(crate) const INCOMPARABLE_NOTE: &str =
        "mechanisms differ: sampled lower bound vs exact kernel high-water mark - not a \
         like-for-like comparison";

    /// What a degenerate paired interval renders as.
    pub(crate) const CI_UNAVAILABLE: &str = "CI unavailable (zero variance)";

    /// The estimation-first statement the TWO-METHOD report carried.
    ///
    /// RETAINED AND DEFERRED (2.0.0, D-19). It advertises PAIRED intervals, which the active
    /// scope does not produce — and a note promising a statistic the report does not contain
    /// is itself a claim about the report.
    pub(crate) const ESTIMATION_FIRST_NOTE_PAIRED: &str =
        "Estimation-first (D-08): point estimates, dispersion and paired 95% CIs only. No \
         binary verdict is printed; p-values live in the --json detail.";

    /// The estimation-first statement (D-08), for the ACTIVE single-method scope.
    ///
    /// It deliberately does NOT contain the word a verdict would use. A report that prints a
    /// binary better/worse is precisely where few-shot seed sensitivity hides: rankings that
    /// reverse across seeds become one word. It advertises exactly the interval the active
    /// scope computes and no other.
    pub(crate) const ESTIMATION_FIRST_NOTE: &str =
        "Estimation-first (D-08): point estimates, dispersion and 95% seed-dispersion \
         intervals only. No binary verdict is printed.";

    /// How the seed-dispersion interval is labelled AT THE POINT OF PRESENTATION.
    ///
    /// A bare interval beside a mean reads as a comparison. This says what it actually is.
    pub(crate) const SEED_CI_LABEL: &str =
        "95% CI over the ten contracted seeds at fixed data and protocol - a seed-dispersion \
         interval, not a population interval";

    /// The single-method note.
    ///
    /// ITS VOCABULARY IS CONSTRAINED, AND NOT FOR STYLE. Plan 05-13 gates the committed report
    /// against the must-not-match literals this module's tests pin, and a note written with
    /// those words would trip a gate mandated by the same plan — which would pressure someone
    /// into either weakening the gate or gutting the note. So it refers to the deferred arm
    /// ONLY by the decision id, the ticket id, and the phrase "a second method". It does not
    /// name the deferred method, uses no comparative connective, does not use the phrase for a
    /// paired difference, and does not use the word for a statistical verdict.
    pub(crate) const SINGLE_METHOD_NOTE: &str =
        "SCOPE - ONE METHOD WAS MEASURED. This report covers SetFit alone. A second method was \
         planned for this matrix and was not run; the reason is recorded as decision D-19 and \
         the restoration path as ticket D-ITEM-05-15. Nothing here states or implies any \
         result about a second method. Read the absence as absence.";

    /// How LoRA's adapter-only figure is labelled where it IS shown.
    pub(crate) const ADAPTER_ONLY_LABEL: &str = "adapter only";

    /// The per-method artifact-bytes ROW LABEL the two-method detail carried.
    ///
    /// RETAINED AND DEFERRED (2.0.0, D-19). It names the deferred method in a label that is
    /// printed on every SetFit row, so under the active scope it puts the second method's name
    /// beside a SetFit number in the one place a reader is looking at a number. A label is
    /// prose, and the prohibition is on prose too.
    pub(crate) const ARTIFACT_BYTES_LABEL_TWO_METHOD: &str =
        "artifact bytes (adapter only for lora)";

    /// The ACTIVE, single-method artifact-bytes row label.
    ///
    /// It states which question the figure answers — what this method WROTE — because that is a
    /// different question from what a user must ship, which is the size table's column.
    pub(crate) const ARTIFACT_BYTES_LABEL: &str = "artifact bytes (what this method wrote)";

    /// The size table's CROSS-METHOD footnote.
    ///
    /// RETAINED AND DEFERRED (2.0.0, D-19). Every sentence in it is about a comparison between
    /// two methods and about the size of a base model that was never loaded here, so under the
    /// active scope it is a paragraph of claim language about a run that did not happen.
    pub(crate) const SIZE_TABLE_FOOTNOTE_TWO_METHOD: &str =
        "A cross-method size claim uses THIS column and no other. LoRA's adapter-only\n\
         artifact bytes are in the per-method detail above, labelled; an adapter is not a\n\
         deployable model, and presenting it beside SetFit's standalone APR would understate\n\
         LoRA by the size of its base model.\n";

    /// The size table's ACTIVE footnote.
    ///
    /// It keeps the RULE — a size claim uses this column and no other — and drops the
    /// comparison the rule was written to protect, because the comparison was not run.
    pub(crate) const SIZE_TABLE_FOOTNOTE: &str =
        "A size claim uses THIS column and no other. The artifact bytes in the per-method\n\
         detail above are what this method WROTE, which is a different question from what a\n\
         user must ship to serve it.\n";

    /// The provenance sources the header names — TWO-METHOD.
    ///
    /// RETAINED AND DEFERRED (2.0.0, D-19). A candidate ledger is the second method's selection
    /// evidence. Under the active scope there is no ledger to recompute from, so naming one is
    /// a claim about a verification this run did not perform.
    pub(crate) const PROVENANCE_SOURCES_TWO_METHOD: &str =
        "the committed lock and ledger bytes, the committed selection manifest,\n\
         \x20         and the row's own confusion matrix";

    /// The evidence the header names as RECOMPUTED — ACTIVE.
    ///
    /// Extended by 05-16 (the selection manifest, opened at a gate-derived path the row cannot
    /// choose) and 05-17 (the row's own confusion matrix, from which every published accuracy
    /// figure is recomputed in closed form). A header that named fewer sources than the run
    /// actually recomputes teaches a reader to trust the report LESS than the evidence warrants,
    /// which is the same defect as over-claiming with the sign flipped.
    pub(crate) const PROVENANCE_SOURCES: &str =
        "the committed lock bytes, the committed selection manifest, and\n\
         \x20         the row's own confusion matrix";

    /// What the report still CANNOT rule out, stated rather than hidden — ACTIVE and deferred.
    ///
    /// 05-17 RETIRED the previous sentence, which read: "a producer holding both the rows and
    /// those files could still emit a mutually consistent forgery. This report proves
    /// consistency, not truth." That is no longer true as an unqualified claim — a forged
    /// quality metric is now refused by the closed-form cross-check, a forged evidence path by
    /// containment, and a forged pairing key by the manifest recomputation.
    ///
    /// THREE residuals remain, and all three are named here because an understated disclosure
    /// is as much a defect as an overstated one:
    /// 1. the cross-check proves the metrics agree with the recorded matrix, NOT that the matrix
    ///    is the one the model produced;
    /// 2. the two calibration diagnostics are not recomputable at all — no committed file
    ///    carries the per-row probabilities they need;
    /// 3. `evidence_table_hash` and `apr_artifact_sha256` are claims about artifacts this index
    ///    deliberately does not carry.
    ///
    /// It must stay byte-identical in meaning to `bench_gate`'s own residual section and to
    /// `selection_safety_evidence.residual_risk` in the claims contract; the three are one
    /// statement written down three times, and gap 1 was findable precisely because three
    /// statements of one fact had drifted apart.
    pub(crate) const RESIDUAL_DISCLOSURE: &str =
        "every published accuracy figure is recomputed from the row's OWN confusion\n\
         \x20         matrix, so a doctored figure is refused — but the matrix itself is\n\
         \x20         producer-written, and a producer who edits it and recomputes the\n\
         \x20         figures from it emits a set this report cannot distinguish from a\n\
         \x20         measurement. The two calibration diagnostics are not recomputable at\n\
         \x20         all: no committed file carries the per-row probabilities they need.\n\
         \x20         `evidence_table_hash` and `apr_artifact_sha256` stay claims about\n\
         \x20         artifacts this index does not carry.";

    /// The machine-readable payload's schema tag.
    pub(crate) const REPORT_PAYLOAD_SCHEMA: &str = "setfit-bench-report-v1";

    /// The `--json` / `--out` payload.
    ///
    /// The aggregate sits under `detail` rather than at the top level because that is what it
    /// IS: the machine-readable detail, and the ONLY place a p-value appears. D-08 permits
    /// p-values in the detail and forbids them in claim language, and a key named `detail` is a
    /// structural statement of that boundary rather than a convention a future renderer can
    /// forget.
    #[derive(Debug, serde::Serialize)]
    pub(crate) struct ReportPayload<'a> {
        /// The payload schema.
        pub(crate) schema: &'static str,
        /// The claims contract the numbers were computed under.
        pub(crate) contract_id: &'a str,
        /// Per-seed deltas, p-values, every mechanism string, and both size fields.
        pub(crate) detail: &'a RunAggregate,
    }

    impl<'a> ReportPayload<'a> {
        /// Wrap an aggregate.
        pub(crate) fn new(report: &'a RunAggregate) -> Self {
            Self {
                schema: REPORT_PAYLOAD_SCHEMA,
                contract_id: &report.contract_id,
                detail: report,
            }
        }
    }

    /// What to do about a refusal, appended to every typed gate error.
    const REMEDY: &str =
        "The report has no partial-data mode: a missing, substituted, unmatched or \
         post-test-selected cell invalidates the whole run (EVAL-04). Re-run the named cell \
         with `apr setfit bench run --method <M> --shots <S> --seed <D> --bench-dir <DIR>`, or, \
         if it ran on the other host, ingest its row with `--record <ROW_FILE>`.";

    /// Read the run manifest. It must EXIST — a report over a declared-on-the-spot manifest
    /// would be a report over eighty pending cells, which is a confusing way to say "there is
    /// no run here".
    fn load_manifest_required(bench_dir: &Path) -> Result<RunManifest> {
        let path = bench_dir.join(RUN_MANIFEST_FILE);
        if !path.exists() {
            return Err(CliError::ValidationFailed(format!(
                "{}: no run manifest. Completeness is defined by that file and the contract, \
                 never by a directory listing — a listing can only report what is present, and \
                 cannot report what is missing. Run at least one cell with `apr setfit bench \
                 run --bench-dir {}` to declare it.",
                path.display(),
                bench_dir.display()
            )));
        }
        let bytes = read_bounded(&path)?;
        RunManifest::from_bytes(&bytes)
            .map_err(|error| CliError::ValidationFailed(format!("{}: {error}", path.display())))
    }

    /// Verify the directory and aggregate it, or return the typed refusal.
    ///
    /// Separated from [`run`] so a test can assert that a refused run set produces an error and
    /// NO rendered table — the rendering functions are simply never reached, which is a stronger
    /// statement than "the output happened not to contain a header".
    pub(crate) fn verified_aggregate(bench_dir: &Path) -> Result<RunAggregate> {
        let manifest = load_manifest_required(bench_dir)?;
        let verified = verify_run(&manifest, bench_dir)
            .map_err(|error| CliError::ValidationFailed(format!("{error}\n\n{REMEDY}")))?;
        Ok(aggregate(&verified))
    }

    /// Render a verified benchmark directory.
    ///
    /// # Errors
    ///
    /// [`CliError::ValidationFailed`] for any gate refusal (naming the cell and, for provenance,
    /// the file), a missing manifest, or a serialization failure; [`CliError::Io`] for the
    /// optional `--out` write.
    pub(crate) fn run(args: &BenchReportArgs<'_>) -> Result<()> {
        let report = verified_aggregate(args.bench_dir)?;
        let payload = ReportPayload::new(&report);

        if let Some(out) = args.out {
            let mut bytes = serde_json::to_vec_pretty(&payload).map_err(|error| {
                CliError::ValidationFailed(format!("the report did not serialize: {error}"))
            })?;
            bytes.push(b'\n');
            atomic_write(out, &bytes, true)?;
        }

        if args.json {
            let rendered = serde_json::to_string_pretty(&payload).map_err(|error| {
                CliError::ValidationFailed(format!("the report did not serialize: {error}"))
            })?;
            println!("{rendered}");
        } else {
            print!("{}", render_human(&report));
        }
        Ok(())
    }

    // --- Rendering -------------------------------------------------------------------------

    /// One summary as `mean +/- std [min, max]`.
    fn summary_cell(summary: &SeriesSummary) -> String {
        format!(
            "{:>10.4} {:>9.4} {:>10.4} {:>10.4}",
            summary.mean, summary.std, summary.min, summary.max
        )
    }

    /// The header block: what was verified, and what a reader may conclude from it.
    fn render_header(report: &RunAggregate) -> String {
        let cross_method = report.methods.len() >= 2;
        let title = if cross_method {
            TWO_METHOD_TITLE
        } else {
            SINGLE_METHOD_TITLE
        };
        // THE HEADER MAY ONLY NAME EVIDENCE THIS RUN ACTUALLY RECOMPUTED. A candidate ledger
        // belongs to the deferred arm; under the active scope there is none, and a header that
        // names one describes a check that did not run.
        let sources = if cross_method {
            PROVENANCE_SOURCES_TWO_METHOD
        } else {
            PROVENANCE_SOURCES
        };
        format!(
            "{title}\n\
             contract: {}\n\
             design:   {} seeds per cell, df = {}, 95% CI uses the frozen t = {:.15}\n\
             verified: every cell of the contracted matrix. A missing, substituted, unmatched or\n\
             \x20         post-test-selected cell would have REFUSED this report rather than\n\
             \x20         shrunk it, and the evidence below was RECOMPUTED from\n\
             \x20         {sources}\n\
             \x20         rather than read off the rows.\n\
             residual: {RESIDUAL_DISCLOSURE}\n\n",
            report.contract_id, report.n_seeds, report.degrees_of_freedom, report.t_crit_975_df9
        )
    }

    /// The quality table: per `(method, shots)`, the headline and its dispersion.
    pub(crate) fn render_quality(quality: &[MethodShotQuality]) -> String {
        let mut out = String::from(QUALITY_TABLE_HEADER);
        out.push('\n');
        out.push_str(
            "method   shots       mean       std        min        max      95% CI (seeds)   \
             (macro F1 mean / MCC mean)\n",
        );
        for group in quality {
            // THE INTERVAL IS RENDERED BESIDE THE MEAN IT BELONGS TO, and a degenerate one is
            // a NAMED state rather than a blank cell — never a `null` a reader takes for a
            // missing measurement (CR-03).
            let interval = match (group.f_avg_seed_ci95.low, group.f_avg_seed_ci95.high) {
                (Some(low), Some(high)) => format!("[{low:.4}, {high:.4}]"),
                _ => CI_UNAVAILABLE.to_string(),
            };
            out.push_str(&format!(
                "{:<8} {:>5} {} {:>18} {:>12.4} {:>8.4}\n",
                group.method.tag(),
                group.shots,
                summary_cell(&group.f_avg),
                interval,
                group.macro_f1.mean,
                group.mcc.mean
            ));
        }
        out.push_str(
            "F_avg = (F1_against + F1_favor) / 2, the official TweetEval stance metric. Macro F1\n\
             is published BESIDE it and never instead of it.\n",
        );
        // THE LABEL TRAVELS WITH THE NUMBER. A bare interval beside a mean reads as a
        // comparison; this states what it actually measures, in the same block.
        out.push_str(SEED_CI_LABEL);
        out.push_str(".\n");
        out
    }

    /// The paired-delta table, with the degenerate case stated rather than blanked.
    pub(crate) fn render_deltas(deltas: &[ShotDelta]) -> String {
        let mut out = String::from(DELTA_TABLE_HEADER);
        out.push('\n');
        out.push_str("shots   mean delta   95% CI\n");
        for delta in deltas {
            let interval = match (delta.ci95.low, delta.ci95.high) {
                (Some(low), Some(high)) => format!("[{low:.4}, {high:.4}]"),
                // NEVER a blank cell and never a `null`: a degenerate interval is a visible,
                // named state, and its point estimate is still reported because it is well
                // defined and it is what a reader wants.
                _ => CI_UNAVAILABLE.to_string(),
            };
            out.push_str(&format!(
                "{:>5} {:>12.4}   {interval}\n",
                delta.shots, delta.mean_delta
            ));
        }
        out.push_str(ESTIMATION_FIRST_NOTE_PAIRED);
        out.push('\n');
        out
    }

    /// Does this resource set cover more than one method?
    ///
    /// DERIVED FROM THE DATA, never from a flag a caller can get wrong. The two renderers below
    /// take only a resource slice, so the scope they render under has to come from the slice
    /// itself — and a set holding one method's groups is, definitionally, a single-method
    /// report. This is the same predicate `render_human` applies via `report.methods`.
    fn resource_is_cross_method(resource: &[MethodShotResource]) -> bool {
        let Some(first) = resource.first() else {
            return false;
        };
        resource.iter().any(|g| g.method != first.method)
    }

    /// The per-method resource detail. Every figure carries its measurement boundary.
    pub(crate) fn render_resource_detail(resource: &[MethodShotResource]) -> String {
        let cross_method = resource_is_cross_method(resource);
        let mut out = String::from(RESOURCE_TABLE_HEADER);
        out.push('\n');
        out.push_str(PER_HOST_FRAMING);
        out.push_str("\n\n");
        for group in resource {
            out.push_str(&format!(
                "{} / s{}  host: {}  backend: {}\n",
                group.method.tag(),
                group.shots,
                group.hosts.join(", "),
                group.backends.join(", ")
            ));
            out.push_str(&format!(
                "  train wall                            {:>12.1} ms\n",
                group.train_wall_ms.mean
            ));
            out.push_str(&format!(
                "  cold latency (fresh child)            {:>12.1} ms\n",
                group.cold_latency_ms.mean
            ));
            out.push_str(&format!(
                "  warm latency (median of 10 after 3)   {:>12.1} ms\n",
                group.warm_latency_ms_median.mean
            ));
            out.push_str(&format!(
                "  throughput                            {:>12.1} rows/s @ batch {}\n",
                group.throughput_rows_per_sec.mean,
                join_u32(&group.throughput_batch_sizes)
            ));
            // TWO PEAKS, TWO ROWS, TWO MECHANISM STRINGS — never one column and never one
            // combined figure. A kernel high-water mark is process-cumulative, so the training
            // process's peak is not the inference peak, and on this host the two fields can
            // carry different mechanism CLASSES within the SAME row. A sampled figure carries
            // its lower-bound label HERE, beside the value.
            out.push_str(&format!(
                "  train peak RSS (training process)     {:>12.0} B   mechanism: {} [{}]{}\n",
                group.train_peak_rss_bytes.mean,
                group.train_peak_rss_mechanisms.join(", "),
                class_tags(group.train_peak_rss_mechanism_classes.as_slice()),
                lower_bound_suffix(group.train_peak_rss_mechanism_classes.as_slice()),
            ));
            out.push_str(&format!(
                "  inference peak RSS (cold child)       {:>12.0} B   mechanism: {} [{}]{}\n",
                group.inference_peak_rss_bytes.mean,
                group.inference_peak_rss_mechanisms.join(", "),
                class_tags(group.inference_peak_rss_mechanism_classes.as_slice()),
                lower_bound_suffix(group.inference_peak_rss_mechanism_classes.as_slice()),
            ));
            // THE WITHIN-ROW ASYMMETRY, SURFACED WHERE IT HAPPENS. With no second method to
            // compare against, one row's two peaks are the ONLY place it can be seen.
            if group.train_peak_rss_mechanism_classes != group.inference_peak_rss_mechanism_classes
            {
                out.push_str(&format!("  ^ {WITHIN_ROW_ASYMMETRY_NOTE}\n"));
            }
            // THE ARTIFACT-BYTES FIGURE, LABELLED, AND ONLY HERE. It answers "what did this
            // method write", which is a different question from "what must a user ship".
            // The label is SCOPE-AWARE: the two-method form names the deferred method, and a
            // label naming a method that was not run is a claim about the report (D-19).
            out.push_str(&format!(
                "  {:<38}{:>12.0} B\n",
                if cross_method {
                    ARTIFACT_BYTES_LABEL_TWO_METHOD
                } else {
                    ARTIFACT_BYTES_LABEL
                },
                group.artifact_bytes.mean
            ));
            out.push('\n');
        }
        out
    }

    /// Render the batch sizes a group used.
    ///
    /// A LIST rather than one number: throughput without a batch size is not a comparable
    /// figure (PF-008 warning sign 4), and if a group's ten cells somehow ran at two batch
    /// sizes a reader must see that rather than see the first one.
    fn join_u32(values: &[u32]) -> String {
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The lower-bound label, when ANY of a group's mechanisms is a sampled one.
    ///
    /// A suffix, so it renders on the same line as the number it qualifies.
    fn lower_bound_suffix(
        classes: &[entrenar::train::setfit::bench_gate::MechanismClass],
    ) -> String {
        use entrenar::train::setfit::bench_gate::MechanismClass;
        if classes
            .iter()
            .any(|c| *c == MechanismClass::SampledLowerBound)
        {
            format!("  {SAMPLED_LOWER_BOUND_LABEL}")
        } else {
            String::new()
        }
    }

    /// Render a list of mechanism classes as their tags.
    fn class_tags(classes: &[entrenar::train::setfit::bench_gate::MechanismClass]) -> String {
        classes
            .iter()
            .map(|class| class.tag())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The cross-method resource comparison — where the mechanism labels EARN their place.
    ///
    /// This is the only table that puts a SetFit figure beside a LoRA one, so it is the only
    /// place a like-for-like reading can be created by accident. Both mechanism strings are
    /// printed on every row, and a row whose two sides come from different mechanism CLASSES
    /// carries [`INCOMPARABLE_NOTE`].
    pub(crate) fn render_resource_comparison(resource: &[MethodShotResource]) -> String {
        let mut out = String::from(RESOURCE_COMPARISON_HEADER);
        out.push('\n');
        out.push_str(PER_HOST_FRAMING_TWO_HOST);
        out.push_str("\n\n");
        for shots in BENCH_SHOTS {
            let setfit = resource
                .iter()
                .find(|g| g.method == Method::Setfit && g.shots == shots);
            let lora = resource
                .iter()
                .find(|g| g.method == Method::Lora && g.shots == shots);
            let (Some(setfit), Some(lora)) = (setfit, lora) else {
                continue;
            };
            out.push_str(&format!("s{shots}\n"));
            out.push_str(&comparison_row(
                "train peak RSS",
                setfit.train_peak_rss_bytes.mean,
                &setfit.train_peak_rss_mechanisms,
                lora.train_peak_rss_bytes.mean,
                &lora.train_peak_rss_mechanisms,
            ));
            out.push_str(&comparison_row(
                "inference peak RSS",
                setfit.inference_peak_rss_bytes.mean,
                &setfit.inference_peak_rss_mechanisms,
                lora.inference_peak_rss_bytes.mean,
                &lora.inference_peak_rss_mechanisms,
            ));
            out.push('\n');
        }
        out
    }

    /// One comparison row, with both mechanisms and the note when they are not comparable.
    fn comparison_row(
        label: &str,
        setfit_value: f64,
        setfit_mechanisms: &[String],
        lora_value: f64,
        lora_mechanisms: &[String],
    ) -> String {
        let setfit_mechanism = setfit_mechanisms.join(", ");
        let lora_mechanism = lora_mechanisms.join(", ");
        let mut row = format!(
            "  {label:<20} setfit {setfit_value:>14.0} B ({setfit_mechanism})  |  lora \
             {lora_value:>14.0} B ({lora_mechanism})\n"
        );
        // COMPARABLE ONLY IF EVERY PAIRING IS. A group whose ten cells used two mechanisms is
        // labelled by the strictest of them, because a table cannot be half comparable.
        let comparable = !setfit_mechanisms.is_empty()
            && !lora_mechanisms.is_empty()
            && setfit_mechanisms.iter().all(|s| {
                lora_mechanisms
                    .iter()
                    .all(|l| mechanisms_are_comparable(s, l))
            });
        if !comparable {
            row.push_str(&format!("    ^ {INCOMPARABLE_NOTE}\n"));
        }
        row
    }

    /// The cross-method size table. `deployable_total_bytes` ONLY.
    ///
    /// An adapter of a few tens of megabytes beside a 90 MB standalone classifier reads as
    /// parity, while the deployable figures differ by the size of a 9B base model. That is the
    /// most natural chart to draw from the row schema, which is exactly why the field this
    /// table may use is named in the contract and enforced here.
    pub(crate) fn render_sizes(resource: &[MethodShotResource]) -> String {
        let cross_method = resource_is_cross_method(resource);
        let mut out = String::from(SIZE_TABLE_HEADER);
        out.push('\n');
        out.push_str("method   shots   deployable_total_bytes\n");
        for method in BENCH_METHODS {
            for shots in BENCH_SHOTS {
                let Some(group) = resource
                    .iter()
                    .find(|g| g.method == method && g.shots == shots)
                else {
                    continue;
                };
                out.push_str(&format!(
                    "{:<8} {:>5} {:>24.0}\n",
                    method.tag(),
                    shots,
                    group.deployable_total_bytes.mean
                ));
            }
        }
        out.push_str(if cross_method {
            SIZE_TABLE_FOOTNOTE_TWO_METHOD
        } else {
            SIZE_TABLE_FOOTNOTE
        });
        out
    }

    /// The whole human report.
    #[must_use]
    pub(crate) fn render_human(report: &RunAggregate) -> String {
        // SCOPE-AWARE, NOT DEFENSIVE. Under the active scope the cross-method sections are
        // REMOVED, not emitted empty: a section header with no rows still tells a reader a
        // comparison was attempted, which is the implication D-19 forbids.
        let cross_method = report.methods.len() >= 2;

        let mut out = render_header(report);
        if !cross_method {
            out.push_str(SINGLE_METHOD_NOTE);
            out.push_str("\n\n");
        }
        out.push_str(&render_quality(&report.quality));
        out.push('\n');
        if cross_method {
            out.push_str(&render_deltas(&report.deltas));
            out.push('\n');
        }
        out.push_str(&render_resource_detail(&report.resource));
        if cross_method {
            out.push_str(&render_resource_comparison(&report.resource));
        }
        out.push_str(&render_sizes(&report.resource));
        if !cross_method {
            // The estimation-first note lived inside the delta table, which the active scope
            // does not render. It is not dropped with it — it is a statement about how every
            // number in this report is presented, not about the deltas.
            out.push('\n');
            out.push_str(ESTIMATION_FIRST_NOTE);
            out.push('\n');
        }
        out
    }
}

/// `apr setfit bench verify-cell` — the SINGLE-CELL verification door (plan 05-11 task 3).
///
/// # Which steps it applies, and which it deliberately excludes
///
/// It is `verify_run`'s steps 1 + 4 + 6 over ONE declared cell: the manifest's own digest,
/// step 3's per-entry rule applied to THAT entry only, the row's file / schema / envelope
/// digest / manifest-digest / slot checks, and provenance recomputed from committed bytes.
///
/// It does NOT apply step 2 (expectation-set equality), step 3's SWEEP over every entry, step 5
/// (pairing) or step 7. The reason is not tidiness: at pilot time the manifest declares 40 cells
/// with 39 still `pending`, which is exactly the state step 3's sweep refuses on — so a door
/// that inherited the set-level checks could never pass on the cell it exists to check. And
/// `bench report` over a copy holding one row refuses at completeness BEFORE the row loop, so
/// the pilot row's own bytes are never read at all. This door reads them.
///
/// # It emits no statistic
///
/// No mean, no dispersion, no interval, no aggregate — the library door it calls returns `()`,
/// so there is nothing to print even by accident. `bench report` is the only door that emits
/// numbers; a per-cell door that printed statistics would be a partial-data report under
/// another name, which the claims contract forbids.
pub(crate) mod verify_cell {
    use std::path::Path;

    use entrenar::train::setfit::bench_gate::verify_cell as verify_one_cell;
    use entrenar::train::setfit::bench_row::{
        CellKey, Method, RunManifest, BENCH_SEEDS, BENCH_SHOTS,
    };

    use super::{read_bounded, RUN_MANIFEST_FILE};
    use crate::error::{CliError, Result};

    /// Everything `bench verify-cell` carries, resolved from clap.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct BenchVerifyCellArgs<'a> {
        /// Where rows, locks, ledgers and the run manifest live.
        pub(crate) bench_dir: &'a Path,
        /// `setfit` or `lora` — the row-validity vocabulary, not the active scope.
        pub(crate) method: &'a str,
        /// Examples per class.
        pub(crate) shots: u32,
        /// One of the ten contracted seeds.
        pub(crate) seed: u32,
    }

    /// What a passing door prints. ONE line, and not a statistic in it.
    pub(crate) const PASS_LINE_PREFIX: &str = "VERIFIED (single cell): ";

    /// The scope statement the pass line carries, so a reader cannot take it for a report.
    ///
    /// CORRECTED BY 05-17 (finding T-05-16-05). This enumeration had fallen BEHIND the door: it
    /// named provenance alone while step 6 had grown the selection binding (05-16) and the
    /// closed-form quality cross-check (05-17). Under-claiming is the safe direction and is
    /// still a defect — a door whose own printed account of its coverage is incomplete teaches
    /// a reader to trust it less than the evidence warrants, and it is the same class of
    /// artifact this phase exists to prevent.
    pub(crate) const DOOR_SCOPE_NOTE: &str =
        "scope: steps 1+4+6 of verify_run over ONE cell - manifest digest, this entry's own \
         completeness, the row file/schema/envelope digest/manifest digest/slot, provenance \
         recomputed from the committed lock bytes at a path that had to pass containment, the \
         selection manifest recomputed at a path derived from the cell key, and every published \
         accuracy figure recomputed from the row's own confusion matrix. NOT the set-level \
         steps (expectation-set equality, the all-entries sweep, pairing, attestation), which \
         is what lets this pass while other cells are still pending. No statistic is emitted; \
         `bench report` is the only door that publishes numbers.";

    /// Verify one cell's own evidence.
    ///
    /// # Errors
    ///
    /// [`CliError::ValidationFailed`] when the cell key is not contracted, when the manifest
    /// cannot be read or parsed, or when the gate refuses the cell.
    pub(crate) fn run(args: &BenchVerifyCellArgs<'_>) -> Result<()> {
        let method = Method::from_tag(args.method).ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "--method must be `setfit` or `lora`, got `{}`",
                args.method
            ))
        })?;
        if !BENCH_SHOTS.contains(&args.shots) {
            return Err(CliError::ValidationFailed(format!(
                "--shots {} is not contracted; the matrix is exactly {BENCH_SHOTS:?}",
                args.shots
            )));
        }
        if !BENCH_SEEDS.contains(&args.seed) {
            return Err(CliError::ValidationFailed(format!(
                "--seed {} is not contracted; the ten contracted seeds are {BENCH_SEEDS:?}, and \
                 42 is deliberately NOT among them",
                args.seed
            )));
        }
        let cell = CellKey::new(method, args.shots, args.seed);

        let manifest_path = args.bench_dir.join(RUN_MANIFEST_FILE);
        let bytes = read_bounded(&manifest_path)?;
        let manifest = RunManifest::from_bytes(&bytes).map_err(|error| {
            CliError::ValidationFailed(format!(
                "the run manifest at {} was refused: {error}",
                manifest_path.display()
            ))
        })?;

        verify_one_cell(&manifest, args.bench_dir, cell).map_err(|error| {
            CliError::ValidationFailed(format!("cell {} REFUSED: {error}", cell.render()))
        })?;

        println!("{PASS_LINE_PREFIX}{}", cell.render());
        println!("{DOOR_SCOPE_NOTE}");
        Ok(())
    }
}

#[cfg(test)]
#[path = "setfit_bench_tests.rs"]
mod setfit_bench_tests;
