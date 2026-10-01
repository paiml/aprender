//! `aprender-setfit-trainer` — the heavy half of the deployment.
//!
//! Invoked asynchronously by the request function with nothing but a task id
//! and an owner. Everything else it needs — the config, where the artifact goes
//! — it reads from the task's envelope, which is why the invoke payload can
//! stay two strings.
//!
//! ```text
//!   read envelope -> fetch dataset -> pre-flight -> run apr -> upload -> terminal write
//!        |               |               |            |          |
//!        +---------------+---------------+------------+----------+--> any failure is a
//!                                                                     `failed` task with
//!                                                                     the reason in it
//! ```
//!
//! "fetch dataset" is a no-op for a run on the packaged dataset. When the
//! envelope names one the client uploaded, it is brought down and unpacked
//! under `/tmp` first, and the CLI's pre-flight then judges it exactly as it
//! judges the packaged one — this binary looks for one file by name and
//! validates nothing.
//!
//! The invariant that makes this safe to retry, race or interrupt is upstream:
//! `AprenderTaskStore::finish` is guarded, so the FIRST terminal write wins and
//! a straggler is a no-op success. This binary never has to reason about who
//! else might be writing.
//!
//! # Cold start does the proving
//!
//! `TrainerPaths::validate` runs at container init, and it probes
//! `apr setfit train --help`. On Lambda that is not a formality: it is the one
//! cheap check that the cross-compiled aarch64 `apr` in the package actually
//! executes on this runtime and carries the `setfit` feature. A packaging
//! mistake surfaces as an init failure in the logs rather than as a training
//! run that dies two minutes in.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use aprender_mcp_setfit_train::{
    execute_run, prepare_run, run_payload, terminal_result, AprenderTaskStore, Outcome,
    TrainerPaths,
};
use aprender_setfit_train_lambda::{
    fetch_dataset, upload_artifact, DynamoDbTaskBackend, TrainingJob,
};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use pmcp::server::task_store::{TaskStore, TaskStoreError};
use pmcp::types::TaskStatus;
use tokio::sync::Notify;
use tracing_subscriber::EnvFilter;

/// How often a running job checks whether it has been cancelled.
///
/// Without this the deployment would accept `tasks/cancel`, mark the record
/// cancelled, and keep a 6 GB function burning CPU for up to fifteen minutes —
/// a cancel that costs money and changes nothing. Lambda gives no way to
/// interrupt an invocation from outside, so the invocation has to ask. Fifteen
/// seconds is ~1% of the measured 127-second run and 60 consistent reads at the
/// 900-second ceiling.
const CANCEL_POLL: Duration = Duration::from_secs(15);

/// Everything resolved once per container.
struct Worker {
    paths: Arc<TrainerPaths>,
    store: Arc<AprenderTaskStore>,
    s3: aws_sdk_s3::Client,
}

impl Worker {
    async fn init() -> Result<Self, String> {
        let paths = TrainerPaths::from_env()?;
        // Names the first missing piece, and proves the packaged `apr` runs.
        paths.validate()?;
        let backend = DynamoDbTaskBackend::from_env().await?;
        let aws = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Ok(Self {
            paths: Arc::new(paths),
            store: Arc::new(AprenderTaskStore::new(Arc::new(backend))),
            s3: aws_sdk_s3::Client::new(&aws),
        })
    }

    /// Record a verdict against the task. Failures to record are logged, not
    /// propagated: the task may have expired or already been cancelled, and
    /// neither is this binary's problem to solve.
    async fn record(&self, job: &TrainingJob, artifact_uri: &str, outcome: &Outcome) {
        let (report, error) = match outcome {
            Outcome::Completed(report) => (Some(report), None),
            Outcome::Failed(message) => (None, Some(message.as_str())),
            Outcome::Cancelled => (None, Some("cancelled by the client (tasks/cancel)")),
        };
        let payload = run_payload(
            &job.task_id,
            artifact_uri,
            outcome.phase(),
            report,
            error,
            None,
        );
        let failed = !matches!(outcome, Outcome::Completed(_));
        if let Err(e) = self
            .store
            .finish(
                &job.task_id,
                &job.owner,
                outcome.task_status(),
                terminal_result(&payload, failed),
            )
            .await
        {
            tracing::warn!("task {} could not be finished: {e}", job.task_id);
        }
    }
}

/// Watch the record for a cancel and fire the relay when one lands.
///
/// Returns a handle the caller aborts once the run is over, so a finished job
/// stops polling immediately rather than at the next tick.
fn watch_for_cancel(
    store: Arc<AprenderTaskStore>,
    job: TrainingJob,
    cancel: Arc<Notify>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(CANCEL_POLL).await;
            match store.get(&job.task_id, &job.owner).await {
                Ok(task) if task.status == TaskStatus::Cancelled => {
                    tracing::info!("task {} was cancelled; stopping the trainer", job.task_id);
                    // `notify_one` stores a permit if nobody is waiting yet —
                    // the same reason `RunningJobs` uses it rather than
                    // `notify_waiters`.
                    cancel.notify_one();
                    return;
                }
                // A VANISHED record means the task expired mid-run: there is
                // nobody left to hand a result to, so stop the work too.
                Err(TaskStoreError::NotFound { .. }) => {
                    tracing::info!("task {} is gone; stopping the trainer", job.task_id);
                    cancel.notify_one();
                    return;
                }
                // Anything else is the STORE failing, not the task ending. A
                // throttle or a transient network error must not kill a healthy
                // 127-second run — it just means this tick learned nothing.
                Err(e) => {
                    tracing::warn!("cancel poll for {} failed, retrying: {e}", job.task_id);
                }
                Ok(_) => {}
            }
        }
    })
}

/// Best-effort removal of a run's scratch files.
///
/// A warm container is reused, `/tmp` is 2 GB and one artifact is ~87 MB, so
/// roughly twenty runs on one container would fill it. The failure that causes
/// is a training run that dies writing its output, which reads as a trainer
/// bug rather than as housekeeping nobody did.
async fn clean_up(paths: &[&Path]) {
    for path in paths {
        if let Err(e) = tokio::fs::remove_file(path).await {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!("could not remove {}: {e}", path.display());
            }
        }
    }
}

/// Run one job to a terminal task state.
async fn run_job(worker: &Worker, job: TrainingJob) {
    let envelope = match worker.store.get_envelope(&job.task_id, &job.owner).await {
        Ok(Some(envelope)) => envelope,
        // No envelope means no task: expired, or never written. There is
        // nothing to train and nothing to write a verdict to.
        Ok(None) | Err(_) => {
            tracing::warn!(
                "task {} has no run envelope; it expired or was never written",
                job.task_id
            );
            return;
        }
    };

    let artifact_uri = envelope
        .get("artifact_uri")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let Some(config) = envelope.get("config") else {
        worker
            .record(
                &job,
                &artifact_uri,
                &Outcome::Failed("the run envelope carries no `config`".to_string()),
            )
            .await;
        return;
    };

    // A client-supplied dataset comes down first. Its scratch directory is
    // removed with the run's other files, whatever the outcome.
    let dataset_dir = worker
        .paths
        .output_dir
        .join(format!("{}-dataset", job.task_id));
    let dataset = match envelope.get("dataset_uri").and_then(|v| v.as_str()) {
        None => None,
        Some(uri) => match fetch_dataset(&worker.s3, uri, &dataset_dir).await {
            Ok(dir) => {
                tracing::info!(
                    "task {}: dataset {uri} unpacked to {}",
                    job.task_id,
                    dir.display()
                );
                Some(dir)
            }
            Err(reason) => {
                worker
                    .record(&job, &artifact_uri, &Outcome::Failed(reason))
                    .await;
                let _ = tokio::fs::remove_dir_all(&dataset_dir).await;
                return;
            }
        },
    };

    let prepared = match prepare_run(&worker.paths, &job.task_id, config, dataset.as_deref()).await
    {
        Ok(prepared) => {
            tracing::info!(
                "task {}: pre-flight passed on {}; training",
                job.task_id,
                prepared.data.display()
            );
            prepared
        }
        // The CLI's own refusal, in its own words — the same text a local
        // submit would have returned synchronously.
        Err(reason) => {
            worker
                .record(&job, &artifact_uri, &Outcome::Failed(reason))
                .await;
            let _ = tokio::fs::remove_dir_all(&dataset_dir).await;
            return;
        }
    };

    let cancel = Arc::new(Notify::new());
    let watcher = watch_for_cancel(Arc::clone(&worker.store), job.clone(), Arc::clone(&cancel));
    let outcome = execute_run(&worker.paths, &prepared, &cancel).await;
    watcher.abort();

    // Only a completed run has an artifact to publish. Upload BEFORE the
    // terminal write: a client that reads `completed` must be able to fetch
    // what the payload names, and the reverse order makes that a race.
    let outcome = match outcome {
        Outcome::Completed(report) => {
            match upload_artifact(&worker.s3, &artifact_uri, &prepared.artifact_path).await {
                Ok(()) => Outcome::Completed(report),
                Err(reason) => Outcome::Failed(format!(
                    "the model trained but could not be published: {reason}"
                )),
            }
        }
        other => other,
    };

    worker.record(&job, &artifact_uri, &outcome).await;
    clean_up(&[&prepared.artifact_path, &prepared.config_path]).await;
    let _ = tokio::fs::remove_dir_all(&dataset_dir).await;
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_ansi(false)
        .try_init();

    // Init failures must kill the container, not every invocation separately:
    // a package missing its `apr` is a deployment fault, and Lambda reports an
    // init error far more legibly than 900 seconds of the same handler error.
    let worker = Worker::init().await.map_err(Error::from)?;

    lambda_runtime::run(service_fn(|event: LambdaEvent<TrainingJob>| {
        let worker = &worker;
        async move {
            run_job(worker, event.payload).await;
            // The task record is the outcome channel, so a training FAILURE is
            // a successful invocation that recorded one. Returning an error
            // here would only add a Lambda error metric for a run whose verdict
            // is already durable and already visible to the client.
            Ok::<(), Error>(())
        }
    }))
    .await
}
