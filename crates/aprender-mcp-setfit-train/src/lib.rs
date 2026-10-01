//! Thin single-algorithm MCP TRAINING server: SetFit few-shot training exposed
//! as an async MCP Task (spec 2025-11-25), plus a polling tool for clients that
//! do not speak tasks.
//!
//! # Training goes through the CLI door, not a third ingest sequence
//!
//! The predict server (`aprender-mcp-setfit`) calls core IN-PROCESS because
//! predict's one door is a public library API (`VerifiedSetFitModel::classify`,
//! D-09/OPS-03). Training's one door is different: it is the `apr setfit train`
//! ADAPTER, whose attested-ingest sequence is deliberately `pub(crate)` in
//! apr-cli — `read_attested_canonical`'s own doc says widening it was rejected
//! precisely so there can never be "two readers of benchmark-manifest.json".
//! An in-process trainer here would be that second (third) reader. So this
//! server SUPERVISES a pinned `apr` binary as a child process, and the binary
//! is PINNED (an explicit path, probed at startup), never resolved from `PATH`.
//!
//! # The task lifecycle follows chess-mcp's model
//!
//! [`task_store`] carries the full rationale; the shape here is its consumer:
//!
//! 1. the handler clears the handoff, resolves the owner, and MINTS the task
//!    itself via `mint_for_request` — so it holds the canonical id BEFORE it
//!    dispatches anything;
//! 2. the run's inputs go into the task's envelope, not into the dispatch
//!    payload;
//! 3. the work is dispatched and the handler returns a task-shaped `working`
//!    value; pmcp's create gate consults the armed handoff and returns THIS
//!    task rather than minting a second one;
//! 4. whoever finishes the work performs the terminal write
//!    ([`AprenderTaskStore::finish`]), guarded so a straggler cannot overwrite
//!    a verdict that already landed;
//! 5. a dispatch that fails compensates immediately, so a task never wedges in
//!    `working` waiting for work that was never started.
//!
//! Today "dispatch" is a spawned in-process waiter, which is what a long-lived
//! stdio server needs. The serverless deployment replaces step 3's dispatch
//! with a Step Functions execution and step 4's writer with a finalizer Lambda,
//! and swaps [`InMemoryTaskBackend`] for a DynamoDB backend behind the same
//! seam. Steps 1, 2 and 5 do not change — which is the point of doing it this
//! way now rather than later.
//!
//! # One job at a time
//!
//! [`RunningJobs`] refuses a second submit while one runs. This is a RESOURCE
//! policy about THIS process's CPU — training saturates it at ~4 GB RSS — so
//! per-process state is the CORRECT scope for it, unlike task state, which
//! must be durable and shared. It carries no correctness weight for the task
//! pairing: the handoff does that.
//!
//! # Fail on the REQUEST before blaming the run
//!
//! Submit runs `apr setfit train --dry-run` synchronously first — the CLI's own
//! pre-flight — so a bad request is refused at the MCP boundary in seconds
//! instead of surfacing minutes later as a failed job. Note the cost: pmcp
//! dispatches requests through a single worker
//! (`Server::spawn_request_worker` — "request handling stays serialized"), so
//! this window blocks every other request too, which is why the budget is
//! deliberately tight.

mod local;
mod task_store;

pub use local::{
    execute_run, prepare_run, run_inputs, LocalDispatcher, Outcome, PreparedRun, RunningJobs,
    TrainerPaths, ENV_APR_BIN, ENV_DATA, ENV_MODEL_DIR, ENV_OUTPUT_DIR, ENV_SELECTION,
    SELECTION_MANIFEST,
};
pub use task_store::{
    mint_id, AprenderTaskStore, BackendError, CancelSink, InMemoryTaskBackend, StoredTask,
    TaskBackend,
};

use std::sync::Arc;

use pmcp::server::typed_tool::TypedTool;
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::types::{CallToolResult, Content, TaskStatus, TaskSupport, ToolExecution};
use pmcp::RequestHandlerExtra;
use pmcp::Server;

pub use args::{StatusArgs, TrainArgs, UploadArgs};

/// The server identity both the stdio runner and any transport wrapper report.
pub const SERVER_NAME: &str = "aprender-setfit-train";

/// The one training tool.
pub const TOOL_TRAIN: &str = "train";

/// The polling companion for clients without MCP Tasks support.
pub const TOOL_STATUS: &str = "train_status";

/// Mints a slot a client can upload a dataset into, for `train`'s `dataset_uri`.
pub const TOOL_UPLOAD: &str = "dataset_upload_url";

/// What an uploaded dataset must be. Stated once, served to clients verbatim
/// from `dataset_upload_url`, and the only layout the worker looks for.
pub const DATASET_FORMAT: &str = "a .tar.gz of an attested benchmark directory: \
benchmark-manifest.json, selection-manifest.json and the train/validation/test JSONL at the \
archive root (or all inside one top-level directory) — exactly what `apr data \
tweet-eval-stance` and `apr data select` write";

/// The owner bucket an UNAUTHENTICATED request binds to **when the transport
/// supplies no auth context at all** — which is stdio, and only stdio.
///
/// This mirrors pmcp's private `V1_UNAUTHENTICATED_OWNER`, and it has to: the
/// handler mints under this owner and pmcp's create gate looks the handoff up
/// under whatever ITS `resolve_owner` returned. If the two ever disagree, the
/// gate mints a second task and the client polls an id nobody updates — which
/// the E2E's task-id correlation assertion is what would catch.
///
/// MEASURED on the deployed server (2026-09-04): pmcp.run supplies an auth
/// context even with `[auth] enabled = false`, whose subject is the literal
/// `"unknown"`. So [`resolve_owner`] takes its `Some` arm and every task on
/// that deployment is owned by `unknown`, never by this constant. Nothing
/// breaks — every door resolves the owner the same way, which is the whole
/// reason the owner is derived per request rather than assumed — but a query
/// written against `local` finds nothing there, and this comment used to imply
/// it would.
pub const UNAUTHENTICATED_OWNER: &str = "local";

/// TTL requested for minted training tasks: generous next to the measured
/// envelope (127 s wall for the 8-shot reference train on an M-series host),
/// because an expired task discards a finished artifact's result.
const TASK_TTL_MS: u64 = 3_600_000;

/// Budget for the synchronous `--dry-run` pre-flight — also the worst case for
/// how long this server can answer nothing else. Tight on purpose.
const DRY_RUN_TIMEOUT_SECS: u64 = 30;

/// Envelope version for every JSON payload this server emits.
const STATUS_SCHEMA_VERSION: u64 = 1;

/// The bound this reading surface owes on a client-supplied document, mirroring
/// the predict sibling's `MAX_REQUEST_BODY_BYTES`. `config` is a
/// `serde_json::Value`, so `deny_unknown_fields` cannot reach inside it and
/// nothing else would stop a caller handing us a 500 MB object to parse and
/// forward.
pub const MAX_CONFIG_BYTES: usize = 1_048_576;

/// Cap on each captured output stream in a failure message.
const OUTPUT_TAIL_BYTES: usize = 2_000;

mod args {
    // The JsonSchema derive expands serde_json::json!, which expands to
    // .unwrap() internally — kept this narrow so the ban covers everything else.
    #![allow(clippy::disallowed_methods)]

    use schemars::JsonSchema;
    use serde::Deserialize;

    /// Arguments for [`super::TOOL_TRAIN`].
    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct TrainArgs {
        /// The full twelve-knob SetFit training configuration, passed VERBATIM
        /// to `apr setfit train --config`. This server validates nothing about
        /// it on purpose: the CLI's single validating constructor is the one
        /// implementation of config legality, and a second validator here could
        /// only drift from it. Its SIZE is bounded (see `MAX_CONFIG_BYTES`) —
        /// that is a transport bound, a different question from legality.
        pub config: serde_json::Value,
        /// Which dataset to train on. Absent, the server's packaged dataset.
        /// On the cloud deployment this is the `dataset_uri` that
        /// `dataset_upload_url` returned, after the upload; on a local server
        /// it is the path of an attested benchmark directory on that machine.
        #[serde(default)]
        pub dataset_uri: Option<String>,
    }

    /// Arguments for [`super::TOOL_UPLOAD`]: there are none. An empty object
    /// with `deny_unknown_fields`, so a client that passes a filename or a
    /// size here learns immediately that the upload happens OUT OF BAND, at
    /// the URL this tool returns, not through this call.
    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct UploadArgs {}

    /// Arguments for [`super::TOOL_STATUS`].
    #[derive(Debug, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct StatusArgs {
        /// The `task_id` a `train` call returned.
        pub task_id: String,
    }
}

/// Where the training work actually happens.
///
/// The request side mints the task, writes the envelope and answers `working`;
/// WHO does the training and WHERE is behind this trait. That is the seam that
/// makes the serverless deployment possible at all: the request Lambda has no
/// `apr`, no dataset and no 88 MB encoder checkout, because
/// [`local::LocalDispatcher`] — which needs all three — is not what it holds.
///
/// A dispatcher that returns `Err` has started nothing, and the caller
/// compensates the task to `failed` rather than leaving a client polling work
/// that will never run.
#[pmcp::async_trait]
pub trait Dispatcher: Send + Sync {
    /// Where this task's artifact will land, for the status payload. A local
    /// path today; an `s3://` URI under Step Functions.
    fn artifact_uri(&self, task_id: &str) -> String;

    /// Mint somewhere a client can put a dataset, and the URI to name it by.
    ///
    /// MCP has no file-upload primitive (2025-11-25): tools take JSON. So the
    /// upload happens OUT OF BAND — this hands back a time-limited URL to PUT
    /// the archive at, and the `dataset_uri` the client then passes to `train`.
    /// A dispatcher with no upload story says so in its error.
    async fn dataset_upload(&self) -> Result<DatasetUpload, String>;

    /// A time-limited URL a client can download a COMPLETED artifact from.
    ///
    /// Minted at read time, never stored: a presigned URL expires, and the
    /// terminal payload in the task store must stay true for the task's whole
    /// TTL. `None` where the artifact needs no signing (a local file).
    async fn artifact_download_url(&self, artifact_uri: &str) -> Option<String>;

    /// Start the work, or refuse it. Refusing is expected — the local
    /// dispatcher runs the CLI's own pre-flight here, so a bad config is a tool
    /// error in seconds rather than a failed job minutes later — and a
    /// `dataset_uri` this dispatcher did not issue, or that was never uploaded,
    /// is refused here too rather than by the worker minutes later.
    async fn dispatch(
        &self,
        task_id: &str,
        owner: &str,
        config: &serde_json::Value,
        dataset_uri: Option<&str>,
    ) -> Result<(), String>;
}

/// What [`Dispatcher::dataset_upload`] hands a client.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetUpload {
    /// Pass this to `train` as `dataset_uri` once the upload has completed.
    pub dataset_uri: String,
    /// PUT the archive here. Expires; the dataset behind it does not.
    pub upload_url: String,
    /// How long `upload_url` stays valid.
    pub expires_in_secs: u64,
    /// The HTTP method `upload_url` accepts.
    pub method: &'static str,
    /// What the archive must contain — [`DATASET_FORMAT`], verbatim.
    pub format: &'static str,
}

/// The owner this request's task belongs to.
///
/// Derived from the request rather than assumed, because the owner is the
/// output of a per-era decision, not a constant: the authenticated subject when
/// there is one, [`UNAUTHENTICATED_OWNER`] otherwise. Assuming a constant is
/// how a store silently matches nothing the moment a client authenticates.
fn resolve_owner(extra: &RequestHandlerExtra) -> String {
    extra.auth_context().map_or_else(
        || UNAUTHENTICATED_OWNER.to_string(),
        |ctx| ctx.subject.clone(),
    )
}

/// The ONE status shape every surface serves — `train_status`'s result and the
/// terminal task result alike — so no two doors can tell different stories.
///
/// Public because the terminal writer is not always in this process: under the
/// Lambda deployment a worker on another host performs it minutes later, and it
/// must write the SAME shape the request side would have. A second formatter
/// there is how two doors start telling different stories.
///
/// `artifact_url` is the one field that is NOT part of the stored verdict: it
/// is a time-limited download link, filled in by `train_status` at read time
/// and always `null` in what the worker persists. The key is present either
/// way so the shape is one shape.
#[must_use]
pub fn run_payload(
    task_id: &str,
    artifact_path: &str,
    phase: &str,
    report: Option<&serde_json::Value>,
    error: Option<&str>,
    artifact_url: Option<&str>,
) -> serde_json::Value {
    #[derive(serde::Serialize)]
    struct Payload<'a> {
        schema_version: u64,
        task_id: &'a str,
        phase: &'a str,
        artifact_path: &'a str,
        artifact_url: Option<&'a str>,
        report: Option<&'a serde_json::Value>,
        error: Option<&'a str>,
    }
    serde_json::to_value(Payload {
        schema_version: STATUS_SCHEMA_VERSION,
        task_id,
        phase,
        artifact_path,
        artifact_url,
        report,
        error,
    })
    .unwrap_or(serde_json::Value::Null)
}

/// What a run needs handed to whoever finishes it.
///
/// The finisher of a serverless run is a different process on a different host
/// minutes later — it has the task id and nothing else, so the CONFIG itself
/// travels here rather than a path only this process can read. Owner-scoped and
/// TTL-bounded, and deliberately off the `TaskStore` trait so the SDK can never
/// serve it to a client.
///
/// `dataset_uri` travels here too — a short handle to an object in a bucket the
/// worker's role can read, never the dataset itself. The envelope lives in a
/// DynamoDB item with a 400 KB ceiling; a dataset does not fit and must not.
fn run_envelope(
    config: &serde_json::Value,
    artifact_uri: &str,
    dataset_uri: Option<&str>,
) -> serde_json::Value {
    #[derive(serde::Serialize)]
    struct Envelope<'a> {
        config: &'a serde_json::Value,
        artifact_uri: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        dataset_uri: Option<&'a str>,
    }
    serde_json::to_value(Envelope {
        config,
        artifact_uri,
        dataset_uri,
    })
    .unwrap_or(serde_json::Value::Null)
}

/// Wrap a terminal payload as the tool result the store persists.
///
/// Public for the same reason as [`run_payload`]: the out-of-process worker
/// performs the terminal write.
#[must_use]
pub fn terminal_result(payload: &serde_json::Value, failed: bool) -> CallToolResult {
    let content = vec![Content::Text {
        text: payload.to_string(),
    }];
    if failed {
        CallToolResult::error(content)
    } else {
        CallToolResult::new(content)
    }
}

/// Turn a dispatch failure into a FAILED task, then report it to the caller.
///
/// A task minted for work that was never started must never be left `working`:
/// the client would poll a task nobody will ever finish. Mirrors chess's
/// `compensate_dispatch_failure`.
async fn compensate(
    store: &AprenderTaskStore,
    task_id: &str,
    owner: &str,
    artifact_path: &str,
    reason: String,
) -> pmcp::Error {
    let payload = run_payload(task_id, artifact_path, "failed", None, Some(&reason), None);
    let _ = store
        .finish(
            task_id,
            owner,
            TaskStatus::Failed,
            terminal_result(&payload, true),
        )
        .await;
    pmcp::Error::validation(format!("training request refused: {reason}"))
}

/// Mint, stash, dispatch, answer — the handler.
async fn submit(
    store: Arc<AprenderTaskStore>,
    dispatcher: Arc<dyn Dispatcher>,
    extra: &RequestHandlerExtra,
    config: serde_json::Value,
    dataset_uri: Option<String>,
) -> pmcp::Result<serde_json::Value> {
    // (a) Drop any arm left by an earlier call that never opened the create
    // gate — a plain, non-task-augmented submit does exactly that.
    store.clear_handoff();

    // (b) The transport bound, before anything is forwarded or written.
    let config_bytes = serde_json::to_vec(&config)
        .map_err(|e| pmcp::Error::internal(format!("config serialization: {e}")))?;
    if config_bytes.len() > MAX_CONFIG_BYTES {
        return Err(pmcp::Error::validation(format!(
            "config is {} bytes; this server accepts at most {MAX_CONFIG_BYTES}",
            config_bytes.len()
        )));
    }

    let owner = resolve_owner(extra);

    // (c) Mint FIRST: the id keys the artifact, the envelope and the dispatch.
    let task = store
        .mint_for_request(&owner, Some(TASK_TTL_MS))
        .await
        .map_err(|e| pmcp::Error::internal(format!("cannot mint a training task: {e}")))?;
    let task_id = task.task_id.clone();
    let artifact_uri = dispatcher.artifact_uri(&task_id);

    // (d) The envelope carries this run's inputs to whoever finishes it —
    // owner-scoped and TTL-bounded, never the dispatch payload.
    if let Err(e) = store
        .put_envelope(
            &task_id,
            &owner,
            run_envelope(&config, &artifact_uri, dataset_uri.as_deref()),
        )
        .await
    {
        let message = format!("cannot record the run envelope: {e}");
        return Err(compensate(&store, &task_id, &owner, &artifact_uri, message).await);
    }

    // (e) Dispatch. A refusal here compensates, so the task never wedges.
    if let Err(reason) = dispatcher
        .dispatch(&task_id, &owner, &config, dataset_uri.as_deref())
        .await
    {
        return Err(compensate(&store, &task_id, &owner, &artifact_uri, reason).await);
    }

    // (f) The task-shaped answer. `taskId` + `status` open pmcp's create gate;
    // the armed handoff makes the store return THIS task rather than minting a
    // second one. A plain call gets this value verbatim and polls
    // `train_status` with the same id.
    let mut value = run_payload(&task_id, &artifact_uri, "working", None, None, None);
    if let Some(object) = value.as_object_mut() {
        object.insert("taskId".to_string(), serde_json::Value::String(task_id));
        object.insert(
            "status".to_string(),
            serde_json::Value::String("working".to_string()),
        );
        object.insert("ttl".to_string(), serde_json::Value::from(TASK_TTL_MS));
    }
    Ok(value)
}

/// Read a run's status back out of the task store.
///
/// This door — not `tasks/result` — is where a completed artifact's download
/// link appears. The SDK serves `tasks/result` straight from the store, and
/// the store holds the verdict the worker wrote, which is deliberately free of
/// anything that expires. The link is minted here, at read time, every time.
async fn status(
    store: &AprenderTaskStore,
    dispatcher: &dyn Dispatcher,
    extra: &RequestHandlerExtra,
    task_id: &str,
) -> pmcp::Result<serde_json::Value> {
    use pmcp::server::task_store::TaskStore;
    let owner = resolve_owner(extra);
    let task = store
        .get(task_id, &owner)
        .await
        .map_err(|e| pmcp::Error::validation(format!("no task {task_id} on this server: {e}")))?;
    // A terminal task's payload is the result the worker wrote — one shape,
    // both doors. A working task has no result yet, so synthesize the same
    // shape from the envelope, which holds the artifact path.
    if let Ok(result) = store.get_result(task_id, &owner).await {
        if let Some(Content::Text { text }) = result.content.first() {
            if let Ok(mut payload) = serde_json::from_str::<serde_json::Value>(text) {
                let completed = payload.get("phase").and_then(|p| p.as_str()) == Some("completed");
                if completed {
                    let artifact = payload
                        .get("artifact_path")
                        .and_then(|p| p.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let url = dispatcher.artifact_download_url(&artifact).await;
                    if let Some(object) = payload.as_object_mut() {
                        object.insert(
                            "artifact_url".to_string(),
                            url.map_or(serde_json::Value::Null, serde_json::Value::String),
                        );
                    }
                }
                return Ok(payload);
            }
        }
    }
    let artifact = store
        .get_envelope(task_id, &owner)
        .await
        .ok()
        .flatten()
        .and_then(|envelope| {
            envelope
                .get("artifact_uri")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default();
    let phase = if task.status.is_terminal() {
        // Terminal with no readable result: report the status honestly rather
        // than claiming `working` forever.
        format!("{:?}", task.status).to_lowercase()
    } else {
        "working".to_string()
    };
    Ok(run_payload(task_id, &artifact, &phase, None, None, None))
}

/// Hand a client somewhere to put a dataset.
async fn upload(dispatcher: &dyn Dispatcher) -> pmcp::Result<serde_json::Value> {
    let slot = dispatcher
        .dataset_upload()
        .await
        .map_err(pmcp::Error::validation)?;
    serde_json::to_value(slot).map_err(|e| pmcp::Error::internal(e.to_string()))
}

/// The streamable-HTTP config this server serves.
///
/// `max_request_bytes` is narrowed from pmcp's 4 MB default to
/// [`MAX_CONFIG_BYTES`], because the config object is the only large thing a
/// client sends and pmcp rejects an oversized body with HTTP 413 BEFORE any
/// JSON parsing. The in-handler check stays: it is the only bound on stdio,
/// which has no HTTP body to reject.
///
/// Sessions are LEFT ON (the default). A standalone remote server outlives its
/// requests, so a session is meaningful; the serverless wrapper is the thing
/// that must use `stateless()`, because API Gateway routes successive requests
/// to different containers each with a fresh in-memory session map.
#[must_use]
pub fn http_config() -> pmcp::server::streamable_http_server::StreamableHttpServerConfig {
    pmcp::server::streamable_http_server::StreamableHttpServerConfig {
        max_request_bytes: MAX_CONFIG_BYTES,
        ..Default::default()
    }
}

/// Serve `server` over streamable HTTP until the task ends.
///
/// Returns the bound address alongside the join handle so a caller that asked
/// for port 0 can report where it actually landed — which is what makes an
/// E2E able to drive this transport without guessing a free port.
///
/// # Errors
///
/// Whatever binding or serving reports.
pub async fn serve_http(
    server: Server,
    addr: std::net::SocketAddr,
) -> pmcp::Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    // The SDK takes the server behind ONE `Arc<tokio::sync::Mutex<_>>` shared
    // by every session, and `dispatch_public_request` holds that lock across
    // the whole of `handle_request_with_context`. That is what keeps the mint
    // handoff's single slot correct on this transport: the tool handler and the
    // create gate that consumes its arm cannot be interleaved with another
    // request's. See `task_store`'s note on the same invariant.
    let server = Arc::new(tokio::sync::Mutex::new(server));
    let http = pmcp::server::streamable_http_server::StreamableHttpServer::with_config(
        addr,
        server,
        http_config(),
    );
    http.start().await
}

/// Assemble the server: three tools and the task store, nothing else.
///
/// # Errors
///
/// `pmcp::Error` if the builder refuses the configuration.
pub fn build_server(
    store: Arc<AprenderTaskStore>,
    dispatcher: Arc<dyn Dispatcher>,
    name: &str,
    version: &str,
) -> pmcp::Result<Server> {
    let train_store = Arc::clone(&store);
    let train_dispatcher = Arc::clone(&dispatcher);
    // `TypedTool::new` derives the schema (with `$ref`s inlined) and
    // deserializes the arguments itself — the same pipeline
    // `tool_typed_with_description` uses for `train_status`, so the two tools
    // cannot advertise schemas built different ways. The explicit registration
    // stays because only `TypedTool` carries `with_execution`.
    let train_tool = TypedTool::new(TOOL_TRAIN, move |args: TrainArgs, extra| {
        let store = Arc::clone(&train_store);
        let dispatcher = Arc::clone(&train_dispatcher);
        Box::pin(
            async move { submit(store, dispatcher, &extra, args.config, args.dataset_uri).await },
        )
    })
    .with_description(
        "Start a SetFit few-shot training run with this server's encoder, on its packaged \
         dataset or on one you uploaded (pass the dataset_uri that dataset_upload_url \
         returned). Returns an MCP task: poll tasks/get for status and tasks/result for \
         the trainer's report, or poll train_status with the same task_id — train_status \
         is also where a completed artifact's download link appears. One job runs at a \
         time.",
    )
    .with_execution(ToolExecution::new().with_task_support(TaskSupport::Optional));

    let status_store = Arc::clone(&store);
    let status_dispatcher = Arc::clone(&dispatcher);
    let upload_dispatcher = Arc::clone(&dispatcher);
    let server = Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool(TOOL_TRAIN, train_tool)
        .tool_typed_with_description::<StatusArgs, _, _>(
            TOOL_STATUS,
            "Report a training task's phase, and on completion the trainer's full \
             --json report (artifact path + sha256, provenance, resolved config) plus \
             artifact_url, a time-limited link to download the .apr. Takes the task_id \
             that `train` returned.",
            move |args: StatusArgs, extra| {
                let store = Arc::clone(&status_store);
                let dispatcher = Arc::clone(&status_dispatcher);
                async move { status(&store, dispatcher.as_ref(), &extra, &args.task_id).await }
            },
        )
        .tool_typed_with_description::<UploadArgs, _, _>(
            TOOL_UPLOAD,
            "Mint an upload slot for a training dataset. Returns a time-limited URL to PUT \
             a .tar.gz of an attested benchmark directory (benchmark-manifest.json, \
             selection-manifest.json and the train/validation/test JSONL — what `apr data` \
             writes), and the dataset_uri to pass to `train` once the upload completes. \
             Takes no arguments; the upload itself happens at the returned URL, not here.",
            move |_args: UploadArgs, _extra| {
                let dispatcher = Arc::clone(&upload_dispatcher);
                async move { upload(dispatcher.as_ref()).await }
            },
        )
        .task_store(store)
        .build()?;
    Ok(server)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally
mod tests {
    use super::*;

    #[test]
    fn train_args_refuse_unknown_keys() {
        let err =
            serde_json::from_value::<TrainArgs>(serde_json::json!({ "config": {}, "shots": 8 }))
                .expect_err("unknown key must be refused");
        assert!(err.to_string().contains("shots"), "{err}");
    }

    #[test]
    fn train_args_take_an_optional_dataset() {
        let plain: TrainArgs =
            serde_json::from_value(serde_json::json!({ "config": {} })).expect("no dataset");
        assert!(plain.dataset_uri.is_none());
        let with: TrainArgs = serde_json::from_value(
            serde_json::json!({ "config": {}, "dataset_uri": "s3://b/datasets/x.tar.gz" }),
        )
        .expect("with dataset");
        assert_eq!(
            with.dataset_uri.as_deref(),
            Some("s3://b/datasets/x.tar.gz")
        );
    }

    #[test]
    fn upload_args_refuse_anything_at_all() {
        serde_json::from_value::<UploadArgs>(serde_json::json!({})).expect("empty is the shape");
        // A filename here means the client thinks the upload goes THROUGH the
        // tool. It does not, and the refusal is where they learn that.
        serde_json::from_value::<UploadArgs>(serde_json::json!({ "file": "data.tar.gz" }))
            .expect_err("an argument must be refused");
    }

    #[test]
    fn the_envelope_carries_the_dataset_handle_only_when_there_is_one() {
        let bare = run_envelope(&serde_json::json!({}), "s3://b/tasks/t.apr", None);
        assert!(bare.get("dataset_uri").is_none(), "{bare}");
        let with = run_envelope(
            &serde_json::json!({}),
            "s3://b/tasks/t.apr",
            Some("s3://b/datasets/d.tar.gz"),
        );
        assert_eq!(with["dataset_uri"], "s3://b/datasets/d.tar.gz");
    }

    #[test]
    fn status_args_require_a_task_id() {
        serde_json::from_value::<StatusArgs>(serde_json::json!({}))
            .expect_err("task_id is not optional — there is no 'latest' across containers");
    }

    #[test]
    fn the_payload_shape_is_one_shape() {
        let report = serde_json::json!({ "artifact_sha256": "ab" });
        let done = run_payload("t-1", "/tmp/x.apr", "completed", Some(&report), None, None);
        assert_eq!(done["schema_version"], 1);
        assert_eq!(done["task_id"], "t-1");
        assert_eq!(done["report"]["artifact_sha256"], "ab");
        assert!(done["error"].is_null());
        assert!(
            done["artifact_url"].is_null(),
            "never stored, minted at read time"
        );
        let working = run_payload("t-1", "/tmp/x.apr", "working", None, None, None);
        // Same keys, whichever door serves it.
        let (mut a, mut b): (Vec<_>, Vec<_>) = (
            done.as_object().expect("obj").keys().collect(),
            working.as_object().expect("obj").keys().collect(),
        );
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn a_failed_run_is_an_error_result() {
        let payload = run_payload("t-1", "/tmp/x.apr", "failed", None, Some("boom"), None);
        assert!(terminal_result(&payload, true).is_error);
        assert!(!terminal_result(&payload, false).is_error);
    }
}
