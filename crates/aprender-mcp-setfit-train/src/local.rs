//! The LOCAL dispatcher: run the training child in this process.
//!
//! This is the whole of "training happens here" — the pinned `apr` binary, the
//! operator-provisioned paths, single-flight admission, the child supervisor
//! and the terminal write. It is what a long-lived server (stdio, or streamable
//! HTTP on a host that outlives its requests) uses.
//!
//! # Why this is a MODULE and not the server
//!
//! The serverless request Lambda dispatches to a Step Functions execution and
//! never trains anything itself: it has no `apr`, no dataset and no 88 MB
//! encoder checkout, and it must not, because the pmcp.run deployment package
//! has a 250 MB ceiling that the platform's own backend has already OOM-killed
//! once. Keeping every local concern behind [`Dispatcher`] is what lets the
//! request server be built without any of it.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use pmcp::async_trait;
use pmcp::types::TaskStatus;
use tokio::sync::Notify;

use crate::task_store::{AprenderTaskStore, CancelSink};
use crate::{
    run_payload, terminal_result, DatasetUpload, Dispatcher, DRY_RUN_TIMEOUT_SECS,
    OUTPUT_TAIL_BYTES, TOOL_STATUS, TOOL_UPLOAD,
};

/// The selection manifest's name inside a dataset directory — where
/// `apr data select` writes it, and the one file this crate looks for by name.
/// Everything else in the directory is the CLI's to validate.
pub const SELECTION_MANIFEST: &str = "selection-manifest.json";

/// Everything the operator provisions; nothing here comes from the client.
///
/// The client varies the training CONFIG; the server owns which dataset, which
/// selection, which encoder checkout and where artifacts land — the thin-server
/// philosophy applied to training.
#[derive(Debug, Clone)]
pub struct TrainerPaths {
    /// The pinned `apr` binary (must carry the `setfit` feature).
    pub apr_bin: PathBuf,
    /// Attested benchmark directory, as written by `apr data tweet-eval-stance`.
    pub data: PathBuf,
    /// The selection manifest, as written by `apr data select`.
    pub selection: PathBuf,
    /// Pinned all-MiniLM-L6-v2 checkout (the CLI never downloads).
    pub model_dir: PathBuf,
    /// Where per-job config files and trained artifacts are written.
    pub output_dir: PathBuf,
}

/// The env var naming the pinned `apr`.
pub const ENV_APR_BIN: &str = "APRENDER_SETFIT_TRAIN_APR_BIN";
/// The env var naming the attested benchmark directory.
pub const ENV_DATA: &str = "APRENDER_SETFIT_TRAIN_DATA";
/// The env var naming the selection manifest.
pub const ENV_SELECTION: &str = "APRENDER_SETFIT_TRAIN_SELECTION";
/// The env var naming the pinned encoder checkout.
pub const ENV_MODEL_DIR: &str = "APRENDER_SETFIT_TRAIN_MODEL_DIR";
/// The env var naming where configs and artifacts are written.
pub const ENV_OUTPUT_DIR: &str = "APRENDER_SETFIT_TRAIN_OUTPUT_DIR";

impl TrainerPaths {
    /// Read every path from the environment.
    ///
    /// The Lambda worker has no argv to configure — the CDK stack sets these
    /// five variables — so this is its only door. The stdio/HTTP runner accepts
    /// flags that fall back to the SAME constants, which is why they are
    /// constants rather than literals repeated in two places.
    ///
    /// # Errors
    ///
    /// Names the first unset variable, in the same voice [`Self::validate`]
    /// uses for the first missing path.
    pub fn from_env() -> Result<Self, String> {
        fn var(name: &str) -> Result<PathBuf, String> {
            std::env::var_os(name)
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} is unset"))
        }
        Ok(Self {
            apr_bin: var(ENV_APR_BIN)?,
            data: var(ENV_DATA)?,
            selection: var(ENV_SELECTION)?,
            model_dir: var(ENV_MODEL_DIR)?,
            output_dir: var(ENV_OUTPUT_DIR)?,
        })
    }

    /// Refuse a misconfigured server at STARTUP, naming the first missing
    /// piece — not at the first submit, minutes into someone's workflow.
    ///
    /// # Errors
    ///
    /// A human-readable message naming the path and what was expected of it.
    pub fn validate(&self) -> Result<(), String> {
        if !self.apr_bin.is_file() {
            return Err(format!(
                "--apr-bin (APRENDER_SETFIT_TRAIN_APR_BIN) {} is not a file; point it at a \
                 pinned, setfit-featured `apr` (the $APR that scripts/apr_bin.sh exports)",
                self.apr_bin.display()
            ));
        }
        if !self.data.is_dir() {
            return Err(format!(
                "--data (APRENDER_SETFIT_TRAIN_DATA) {} is not a directory; expected an \
                 attested benchmark dir (benchmark-manifest.json + split JSONL)",
                self.data.display()
            ));
        }
        if !self.selection.is_file() {
            return Err(format!(
                "--selection (APRENDER_SETFIT_TRAIN_SELECTION) {} is not a file; expected a \
                 selection-manifest.json",
                self.selection.display()
            ));
        }
        if !self.model_dir.is_dir() {
            return Err(format!(
                "--model-dir (APRENDER_SETFIT_TRAIN_MODEL_DIR) {} is not a directory; expected \
                 a pinned all-MiniLM-L6-v2 checkout",
                self.model_dir.display()
            ));
        }
        std::fs::create_dir_all(&self.output_dir).map_err(|e| {
            format!(
                "--output-dir (APRENDER_SETFIT_TRAIN_OUTPUT_DIR) {} cannot be created: {e}",
                self.output_dir.display()
            )
        })?;
        self.probe_setfit_subcommand()
    }

    /// `is_file()` is not enough: `setfit` is deliberately NOT in apr-cli's
    /// `default` feature set, so the `apr` a plain `cargo build --release`
    /// produces — the command the repo's own docs give — has no `setfit`
    /// subcommand at all. That binary passes every check above, the server
    /// starts, advertises `train`, and then dies on EVERY submit inside the
    /// pre-flight with clap's `unrecognized subcommand`.
    ///
    /// This function's whole promise is "name the first missing piece at
    /// STARTUP"; the likeliest missing piece was the one it did not look for.
    /// `--help` is free — no model load, no I/O.
    fn probe_setfit_subcommand(&self) -> Result<(), String> {
        let probe = std::process::Command::new(&self.apr_bin)
            .args(["setfit", "train", "--help"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output();
        match probe {
            Err(e) => Err(format!(
                "--apr-bin (APRENDER_SETFIT_TRAIN_APR_BIN) {} could not be executed: {e}",
                self.apr_bin.display()
            )),
            Ok(out) if !out.status.success() => Err(format!(
                "--apr-bin (APRENDER_SETFIT_TRAIN_APR_BIN) {} does not answer \
                 `setfit train --help` ({}): build it with `--features setfit` — setfit is NOT \
                 in apr-cli's default feature set, so a plain `cargo build --release` produces \
                 an apr this server cannot use.\n{}",
                self.apr_bin.display(),
                out.status,
                tail(&out.stderr, OUTPUT_TAIL_BYTES)
            )),
            Ok(_) => Ok(()),
        }
    }
}

/// The one training run this process will admit at a time, and the handle that
/// stops it.
#[derive(Debug)]
struct RunningJob {
    task_id: String,
    cancel: Arc<Notify>,
}

/// Single-flight admission plus the cancel relay.
///
/// Per-process by design: it is about this container's CPU, not about task
/// state. A second container admitting its own job is correct behaviour, not a
/// bug — which is exactly why task state lives in the store instead.
#[derive(Debug, Default)]
pub struct RunningJobs {
    current: Mutex<Option<RunningJob>>,
}

impl RunningJobs {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn guard(&self) -> MutexGuard<'_, Option<RunningJob>> {
        self.current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Admit `task_id`, or refuse naming the job already running.
    fn try_admit(&self, task_id: &str) -> Result<Arc<Notify>, String> {
        let mut current = self.guard();
        if let Some(running) = current.as_ref() {
            return Err(format!(
                "a training job is already running ({}); this server trains one model at a \
                 time — poll `{TOOL_STATUS}` and resubmit when it finishes",
                running.task_id
            ));
        }
        let cancel = Arc::new(Notify::new());
        *current = Some(RunningJob {
            task_id: task_id.to_string(),
            cancel: Arc::clone(&cancel),
        });
        Ok(cancel)
    }

    /// Release the slot, but only if `task_id` still owns it.
    fn release(&self, task_id: &str) {
        let mut current = self.guard();
        if current.as_ref().is_some_and(|r| r.task_id == task_id) {
            *current = None;
        }
    }
}

impl CancelSink for RunningJobs {
    fn cancel(&self, task_id: &str) {
        let current = self.guard();
        if let Some(running) = current.as_ref().filter(|r| r.task_id == task_id) {
            // `notify_one`, NOT `notify_waiters`: the latter wakes only waiters
            // already registered and stores no permit, and `run_child` registers
            // on the first poll of its `select!` — after the spawn is scheduled.
            // A cancel landing in that window would be silently dropped.
            running.cancel.notify_one();
        }
    }
}

/// The base `apr setfit train` invocation. One builder, so the dry run and the
/// real run cannot drift.
fn train_command(paths: &TrainerPaths, run: &PreparedRun) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(&paths.apr_bin);
    cmd.arg("setfit")
        .arg("train")
        .arg("--config")
        .arg(&run.config_path)
        .arg("--data")
        .arg(&run.data)
        .arg("--selection")
        .arg(&run.selection)
        .arg("--model-dir")
        .arg(&paths.model_dir)
        .arg("--output")
        .arg(&run.artifact_path)
        .arg("--json")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // A dying server must not orphan a CPU-saturating trainer. This is also
        // what makes cancellation-by-drop safe.
        .kill_on_drop(true);
    cmd
}

/// `apr setfit train --json` writes ONE pretty-printed report and nothing else
/// to stdout, so the whole buffer is the document. If stdout ever gains a
/// second writer this must fail loudly rather than quietly find something else
/// that parses.
fn parse_json_report(stdout: &[u8]) -> Option<serde_json::Value> {
    serde_json::from_slice(stdout).ok()
}

/// At most the last `max` bytes of a captured stream, on a char boundary.
fn tail(stream: &[u8], max: usize) -> String {
    let text = String::from_utf8_lossy(stream);
    let text = text.trim();
    if text.len() <= max {
        return text.to_string();
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &text[start..])
}

/// What a failing child said — BOTH streams. Keeping only stderr is defect
/// #2418, which `aprender-mcp`'s subprocess module already paid for once: a
/// failing `apr` can still have written its `--json` report to stdout.
fn failure_detail(stdout: &[u8], stderr: &[u8]) -> String {
    let err = tail(stderr, OUTPUT_TAIL_BYTES);
    let out = tail(stdout, OUTPUT_TAIL_BYTES);
    match (err.is_empty(), out.is_empty()) {
        (false, false) => format!("{err}\n--- stdout ---\n{out}"),
        (false, true) => err,
        (true, false) => out,
        (true, true) => "the trainer produced no output".to_string(),
    }
}

/// A job's terminal verdict.
#[derive(Debug)]
pub enum Outcome {
    /// The trainer exited 0 and printed a parseable `--json` report.
    Completed(serde_json::Value),
    /// Anything else: a refused request, a non-zero exit, an unreadable report.
    Failed(String),
    /// A `tasks/cancel` reached the running child.
    Cancelled,
}

impl Outcome {
    /// The `phase` string this verdict reports in the status payload.
    #[must_use]
    pub const fn phase(&self) -> &'static str {
        match self {
            Self::Completed(_) => "completed",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    /// The MCP task status a verdict maps onto. `Failed` is a real status the
    /// SDK's state machine accepts, so a task client can tell a failed run from
    /// a successful one by STATUS and need not parse the payload.
    #[must_use]
    pub const fn task_status(&self) -> TaskStatus {
        match self {
            Self::Completed(_) => TaskStatus::Completed,
            Self::Failed(_) => TaskStatus::Failed,
            Self::Cancelled => TaskStatus::Cancelled,
        }
    }
}

/// A run that has passed the CLI's own pre-flight and is ready to execute.
///
/// Produced by [`prepare_run`], consumed by [`execute_run`]. The split is not
/// decoration: the in-process dispatcher must answer the MCP call the moment
/// pre-flight passes and run the child afterwards, while the Lambda worker does
/// both back to back. One preparation, two schedules.
#[derive(Debug, Clone)]
pub struct PreparedRun {
    /// Where this run's config JSON was written.
    pub config_path: PathBuf,
    /// Where the trainer will write the `.apr` artifact.
    pub artifact_path: PathBuf,
    /// The attested benchmark directory this run trains on: the operator's
    /// packaged default, or a directory the client supplied.
    pub data: PathBuf,
    /// Its selection manifest.
    pub selection: PathBuf,
}

/// Which dataset a run trains on: the packaged default, or a client-supplied
/// directory that must at least contain a selection manifest.
///
/// Only the manifest's PRESENCE is checked here, because it is the one file
/// this crate has to find by name. Whether the directory is a valid attested
/// benchmark — manifest digests, split fingerprints, replayable selection — is
/// the CLI's judgement, delivered by the pre-flight in its own words, and a
/// second judge here could only disagree with it.
///
/// # Errors
///
/// When the supplied directory has no selection manifest.
pub fn run_inputs(
    paths: &TrainerPaths,
    dataset: Option<&Path>,
) -> Result<(PathBuf, PathBuf), String> {
    match dataset {
        None => Ok((paths.data.clone(), paths.selection.clone())),
        Some(dir) => {
            let selection = dir.join(SELECTION_MANIFEST);
            if !selection.is_file() {
                return Err(format!(
                    "dataset {} has no {SELECTION_MANIFEST}; a dataset is an attested benchmark \
                     directory as `apr data tweet-eval-stance` and `apr data select` write it",
                    dir.display()
                ));
            }
            Ok((dir.to_path_buf(), selection))
        }
    }
}

/// Materialize a run's config and put it through `apr setfit train --dry-run`.
///
/// The ONE door to "is this request legal", shared by the in-process dispatcher
/// and the Lambda worker — the CLI's single validating constructor decides, and
/// neither caller re-implements it.
///
/// # Errors
///
/// The CLI's own refusal text, or whatever stopped the config being written.
pub async fn prepare_run(
    paths: &TrainerPaths,
    task_id: &str,
    config: &serde_json::Value,
    dataset: Option<&Path>,
) -> Result<PreparedRun, String> {
    let (data, selection) = run_inputs(paths, dataset)?;
    let artifact_path = paths.output_dir.join(format!("{task_id}.apr"));
    let config_path = paths.output_dir.join(format!("{task_id}.config.json"));
    let bytes =
        serde_json::to_vec_pretty(config).map_err(|e| format!("config serialization: {e}"))?;
    tokio::fs::write(&config_path, &bytes)
        .await
        .map_err(|e| format!("cannot write {}: {e}", config_path.display()))?;
    let prepared = PreparedRun {
        config_path,
        artifact_path,
        data,
        selection,
    };
    if let Some(reason) = preflight(paths, &prepared).await {
        return Err(reason);
    }
    Ok(prepared)
}

/// Supervise a prepared run to its terminal verdict.
///
/// `cancel` is the relay a `tasks/cancel` fires into. A caller with no way to
/// be cancelled (the Lambda worker: an invocation in flight cannot be
/// interrupted from outside) passes a `Notify` nobody notifies, which is honest
/// rather than a special case — the guarded terminal write is what makes a
/// cancel that lands mid-run correct there.
pub async fn execute_run(paths: &TrainerPaths, prepared: &PreparedRun, cancel: &Notify) -> Outcome {
    run_child(paths, prepared, cancel).await
}

/// The CLI's own request checks, run synchronously. `None` means accepted.
async fn preflight(paths: &TrainerPaths, run: &PreparedRun) -> Option<String> {
    let mut dry = train_command(paths, run);
    dry.arg("--dry-run");
    match tokio::time::timeout(Duration::from_secs(DRY_RUN_TIMEOUT_SECS), dry.output()).await {
        Err(_) => Some(format!(
            "pre-flight (--dry-run) exceeded {DRY_RUN_TIMEOUT_SECS}s"
        )),
        Ok(Err(e)) => Some(format!("cannot spawn {}: {e}", paths.apr_bin.display())),
        // BOTH streams: `Command::output()` re-pipes stdout regardless of what
        // the builder asked for, and a signal kill leaves stderr empty — which
        // rendered as a refusal with no reason at all.
        Ok(Ok(out)) if !out.status.success() => Some(failure_detail(&out.stdout, &out.stderr)),
        Ok(Ok(_)) => None,
    }
}

/// Supervise one training child to its terminal state.
///
/// `wait_with_output` keeps both pipes draining while the child runs — a child
/// that fills a pipe nobody reads deadlocks, and this makes that tokio's
/// guarantee rather than a local invariant. Cancellation drops that future and
/// `kill_on_drop` reaps the child.
async fn run_child(paths: &TrainerPaths, run: &PreparedRun, cancel: &Notify) -> Outcome {
    let mut cmd = train_command(paths, run);
    let child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => return Outcome::Failed(format!("cannot spawn {}: {e}", paths.apr_bin.display())),
    };
    let output = tokio::select! {
        output = child.wait_with_output() => output,
        () = cancel.notified() => return Outcome::Cancelled,
    };
    match output {
        Ok(out) if out.status.success() => parse_json_report(&out.stdout).map_or_else(
            || {
                Outcome::Failed(
                    "the trainer exited 0 but printed no parseable --json report".to_string(),
                )
            },
            Outcome::Completed,
        ),
        Ok(out) => Outcome::Failed(format!(
            "the trainer exited with {}: {}",
            out.status,
            failure_detail(&out.stdout, &out.stderr)
        )),
        Err(e) => Outcome::Failed(format!("waiting on the trainer failed: {e}")),
    }
}

/// Runs the training child in THIS process.
///
/// Holds everything the request side must not need: the pinned binary, the
/// operator's paths, and the single-flight guard. The store comes along because
/// the waiter performs the terminal write itself — there is no finalizer on
/// this path, and nothing else would ever move the task off `working`.
pub struct LocalDispatcher {
    paths: Arc<TrainerPaths>,
    running: Arc<RunningJobs>,
    store: Arc<AprenderTaskStore>,
}

impl std::fmt::Debug for LocalDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalDispatcher")
            .field("paths", &self.paths)
            .finish_non_exhaustive()
    }
}

impl LocalDispatcher {
    #[must_use]
    pub fn new(
        paths: Arc<TrainerPaths>,
        running: Arc<RunningJobs>,
        store: Arc<AprenderTaskStore>,
    ) -> Self {
        Self {
            paths,
            running,
            store,
        }
    }
}

#[async_trait]
impl Dispatcher for LocalDispatcher {
    fn artifact_uri(&self, task_id: &str) -> String {
        self.paths
            .output_dir
            .join(format!("{task_id}.apr"))
            .display()
            .to_string()
    }

    async fn dataset_upload(&self) -> Result<DatasetUpload, String> {
        Err(format!(
            "this server runs locally and takes no uploads: pass `dataset_uri` to `train` as the \
             path of an attested benchmark directory on this machine. `{TOOL_UPLOAD}` is how the \
             cloud deployment takes a dataset."
        ))
    }

    async fn artifact_download_url(&self, _artifact_uri: &str) -> Option<String> {
        // The artifact is a local file the client can already read; there is
        // nothing to sign and nothing to expire.
        None
    }

    async fn dispatch(
        &self,
        task_id: &str,
        owner: &str,
        config: &serde_json::Value,
        dataset_uri: Option<&str>,
    ) -> Result<(), String> {
        // Locally a dataset URI is a directory path. Refuse anything else up
        // front, in the same voice the cloud dispatcher uses for a URI it did
        // not issue, so a client learns the rule from either deployment.
        let dataset = dataset_uri.map(Path::new);
        if let Some(dir) = dataset {
            if !dir.is_dir() {
                return Err(format!(
                    "dataset_uri {} is not a directory on this machine; this server takes a local \
                     attested benchmark directory, not a URL",
                    dir.display()
                ));
            }
        }

        // Single-flight FIRST: refusing before any file is written keeps a
        // refused submit from leaving a config behind.
        let cancel = self.running.try_admit(task_id)?;

        // The CLI's own request checks, in its own words, before the task is
        // allowed to look like a running job. A refusal releases the slot.
        let prepared = match prepare_run(&self.paths, task_id, config, dataset).await {
            Ok(prepared) => prepared,
            Err(reason) => {
                self.running.release(task_id);
                return Err(reason);
            }
        };

        let paths = Arc::clone(&self.paths);
        let running = Arc::clone(&self.running);
        let store = Arc::clone(&self.store);
        let task_id = task_id.to_string();
        let owner = owner.to_string();
        let artifact_display = prepared.artifact_path.display().to_string();
        tokio::spawn(async move {
            let outcome = execute_run(&paths, &prepared, &cancel).await;
            running.release(&task_id);
            let (report, error) = match &outcome {
                Outcome::Completed(report) => (Some(report), None),
                Outcome::Failed(message) => (None, Some(message.as_str())),
                Outcome::Cancelled => (None, Some("cancelled by the client (tasks/cancel)")),
            };
            let payload = run_payload(
                &task_id,
                &artifact_display,
                outcome.phase(),
                report,
                error,
                None,
            );
            let failed = !matches!(outcome, Outcome::Completed(_));
            // Tolerated, not unwrapped: the task can have expired or already
            // been cancelled, and `finish` no-ops on a second terminal write.
            let _ = store
                .finish(
                    &task_id,
                    &owner,
                    outcome.task_status(),
                    terminal_result(&payload, failed),
                )
                .await;
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pretty_printed_report_is_the_whole_document() {
        let report = parse_json_report(b"{\n  \"command\": \"setfit train\"\n}\n")
            .expect("pretty report parses");
        assert_eq!(report["command"], "setfit train");
        assert!(
            parse_json_report(b"warming up\n{\"command\":\"x\"}\n").is_none(),
            "a second writer on stdout must fail loudly, not be scanned around"
        );
    }

    #[test]
    fn a_failure_keeps_both_streams() {
        let detail = failure_detail(b"{\"partial\":true}", b"error: refused");
        assert!(detail.contains("error: refused"), "{detail}");
        assert!(
            detail.contains("{\"partial\":true}"),
            "stdout must survive a failure (#2418): {detail}"
        );
        assert_eq!(failure_detail(b"", b""), "the trainer produced no output");
    }

    #[test]
    fn tail_clips_on_char_boundaries() {
        let clipped = tail("héllo wörld".repeat(400).as_bytes(), 64);
        assert!(clipped.starts_with('…'));
        assert!(clipped.len() <= 64 + '…'.len_utf8());
    }

    #[test]
    fn single_flight_refuses_a_second_admit_and_names_the_first() {
        let running = RunningJobs::new();
        running.try_admit("task-a").expect("first admit");
        let refusal = running.try_admit("task-b").expect_err("second refused");
        assert!(refusal.contains("task-a"), "{refusal}");
        running.release("task-a");
        running.try_admit("task-b").expect("admit after release");
    }

    #[test]
    fn release_only_frees_the_slot_its_own_task_holds() {
        let running = RunningJobs::new();
        running.try_admit("task-a").expect("admit");
        // A straggler from a previous run must not free the current job's slot.
        running.release("task-stale");
        assert!(
            running.try_admit("task-b").is_err(),
            "task-a still holds the slot"
        );
        running.release("task-a");
        running.try_admit("task-b").expect("now free");
    }

    fn paths_in(root: &Path) -> TrainerPaths {
        TrainerPaths {
            apr_bin: root.join("apr"),
            data: root.join("default-data"),
            selection: root.join("default-data").join(SELECTION_MANIFEST),
            model_dir: root.join("encoder"),
            output_dir: root.join("out"),
        }
    }

    fn scratch(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!("setfit-train-{label}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn no_dataset_means_the_packaged_default() {
        let root = scratch("default");
        let paths = paths_in(&root);
        let (data, selection) = run_inputs(&paths, None).expect("defaults");
        assert_eq!(data, paths.data);
        assert_eq!(selection, paths.selection);
    }

    #[test]
    fn a_supplied_dataset_routes_both_paths_through_it() {
        let root = scratch("supplied");
        let paths = paths_in(&root);
        let dataset = root.join("mine");
        std::fs::create_dir_all(&dataset).expect("dataset dir");
        std::fs::write(dataset.join(SELECTION_MANIFEST), b"{}").expect("manifest");
        let (data, selection) = run_inputs(&paths, Some(&dataset)).expect("supplied");
        assert_eq!(data, dataset);
        assert_eq!(selection, dataset.join(SELECTION_MANIFEST));
        // And the command the CLI sees names the supplied paths, not the
        // packaged ones — this is the whole point of the override.
        let run = PreparedRun {
            config_path: root.join("c.json"),
            artifact_path: root.join("a.apr"),
            data: data.clone(),
            selection: selection.clone(),
        };
        let cmd = train_command(&paths, &run);
        let args: Vec<String> = cmd
            .as_std()
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let after = |flag: &str| {
            let i = args.iter().position(|a| a == flag).expect(flag);
            args[i + 1].clone()
        };
        assert_eq!(after("--data"), data.display().to_string());
        assert_eq!(after("--selection"), selection.display().to_string());
        assert!(!args.iter().any(|a| a.contains("default-data")), "{args:?}");
    }

    #[test]
    fn a_dataset_without_a_selection_manifest_is_refused_by_name() {
        let root = scratch("nomanifest");
        let paths = paths_in(&root);
        let dataset = root.join("bare");
        std::fs::create_dir_all(&dataset).expect("dataset dir");
        let err = run_inputs(&paths, Some(&dataset)).expect_err("must refuse");
        assert!(err.contains(SELECTION_MANIFEST), "{err}");
        assert!(err.contains("bare"), "must name the directory: {err}");
    }

    #[test]
    fn every_verdict_maps_to_a_distinct_task_status() {
        assert_eq!(
            Outcome::Completed(serde_json::Value::Null).task_status(),
            TaskStatus::Completed
        );
        assert_eq!(
            Outcome::Failed(String::new()).task_status(),
            TaskStatus::Failed,
            "a failed run must be distinguishable by STATUS, not only by payload"
        );
        assert_eq!(Outcome::Cancelled.task_status(), TaskStatus::Cancelled);
    }
}
