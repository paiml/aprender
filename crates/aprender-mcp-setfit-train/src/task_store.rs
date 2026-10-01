//! The training task store: a `pmcp::TaskStore` over a [`TaskBackend`] seam,
//! plus the mint handoff that makes the TOOL HANDLER the task-id authority.
//!
//! Ported from `chess-mcp`'s `game_analysis::task_store` (phases 14–15), which
//! is the in-house model for MCP Tasks on a serverless MCP server. The
//! decisions below are theirs; the comments record why each survives the port.
//!
//! # SDK-core `TaskStore` directly, NOT the `pmcp-tasks` crate (chess D-14-A)
//!
//! `pmcp-tasks` defines its OWN `TaskStore` trait, distinct from
//! `pmcp::server::task_store::TaskStore` — and `ServerBuilder::task_store`
//! wants the latter. Depending on both puts two same-named traits in play and
//! buys an adapter. Implementing core's trait directly is one trait, no
//! adapter, and it is what chess concluded independently.
//!
//! # The [`TaskBackend`] seam (chess D-14-B)
//!
//! [`AprenderTaskStore`] holds an `Arc<dyn TaskBackend>` rather than a
//! DynamoDB client, so the whole `TaskStore` contract is exercised with no AWS
//! credentials and no network. [`InMemoryTaskBackend`] backs the stdio server
//! and every test here; a DynamoDB backend drops in behind the same seam for
//! the Lambda deployment without touching the store, the tools, or the server.
//!
//! # [`BackendError::Conflict`] (chess D-14-E)
//!
//! The SDK's [`TaskStoreError`] has exactly four variants — `NotFound`,
//! `InvalidTransition`, `Expired`, `Internal` — and none of them means
//! "someone else wrote first". Without a typed conflict signal a
//! compare-and-swap retry policy could only be written by matching on a
//! formatted string, which this repo's error discipline forbids.
//! [`BackendError`] adds `Conflict`; the `From` impl collapses it onto
//! `Internal` at the trait boundary, so the SDK-facing surface is unchanged.
//!
//! # The mint handoff (chess D-17) — why the handler must mint
//!
//! pmcp's rule is `D-STORE-MINTS-ID`: with a store configured, the STORE mints
//! the id that reaches the wire, and it does so AFTER the tool handler has
//! returned. That is fine when the work is an in-process future the handler
//! spawned. It is fatal the moment the work is dispatched OUT OF BAND — to a
//! Step Functions execution, a second Lambda, or anything else that must be
//! handed an id — because a handler that mints its own id A and then lets the
//! SDK mint a second id B leaves the client polling B while the worker updates
//! A.
//!
//! [`RequestMintHandoff`] closes that: the handler calls
//! [`AprenderTaskStore::mint_for_request`], which mints AND arms the handoff,
//! and [`TaskStore::create`] consults [`RequestMintHandoff::take_matching`]
//! FIRST — returning the handler's task instead of minting a second one.
//!
//! An earlier revision of this crate solved the same problem with a store
//! DECORATOR that observed `create` and reported the minted id back. That only
//! works because its waiter was in-process and could be told the id late; it
//! cannot hand an id to a dispatch that has already happened. The handoff is
//! the more general mechanism, so the decorator is gone.
//!
//! # The envelope side channel (chess T-14-27)
//!
//! [`AprenderTaskStore::put_envelope`] / [`get_envelope`](AprenderTaskStore::get_envelope)
//! are deliberately NOT on the `TaskStore` trait: the SDK must never serve an
//! envelope to a client. They carry a run's inputs from the request handler to
//! whatever finishes the work — owner-scoped and TTL-bounded — rather than
//! through a dispatch payload. Chess's reason is worth keeping verbatim: a
//! Step Functions execution input is retained in execution history and
//! readable by anyone holding `states:GetExecutionHistory`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use pmcp::async_trait;
use pmcp::server::task_store::{StoreConfig, TaskStore, TaskStoreError};
use pmcp::types::tasks::Task;
use pmcp::types::{CallToolResult, TaskStatus};

/// How long an armed mint stays consumable. The handler arms it as its last
/// step and pmcp's create gate runs microseconds later in the same request
/// future; five seconds is chess's bound and is enormous next to that.
pub const HANDOFF_MAX_AGE_MS: u64 = 5_000;

/// Fallback TTL when neither the caller nor `StoreConfig` names one — the
/// SDK's `default_ttl_ms` is itself an `Option`, so something has to decide.
pub const DEFAULT_TTL_MS: u64 = 3_600_000;

/// The full record a [`TaskBackend`] persists per `(owner_id, task_id)`.
///
/// `envelope` is the side channel — opaque to the trait, never on the wire.
/// `version` is the CAS token every mutation consults.
#[derive(Debug, Clone)]
pub struct StoredTask {
    pub task: Task,
    pub result: Option<CallToolResult>,
    pub envelope: Option<serde_json::Value>,
    pub version: u64,
    /// Epoch-seconds expiry. `None` means no TTL.
    pub expires_at_secs: Option<u64>,
}

/// Error surface for [`TaskBackend`] operations.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// A compare-and-swap write lost a race against the record's version.
    #[error("task record conflict (concurrent write)")]
    Conflict,
    /// Every other backend failure.
    #[error(transparent)]
    Store(#[from] TaskStoreError),
}

/// The ONE place `Conflict` collapses onto the SDK's error type — at the trait
/// boundary, so the SDK-facing surface keeps its four variants.
impl From<BackendError> for TaskStoreError {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::Conflict => Self::Internal {
                message: "task record conflict (concurrent write)".to_string(),
            },
            BackendError::Store(inner) => inner,
        }
    }
}

/// Persistence for task records. In-memory here; DynamoDB for the deployment.
#[async_trait]
pub trait TaskBackend: Send + Sync {
    /// Insert a brand-new record. Callers mint a fresh id first, so a
    /// collision at `(owner_id, task_id)` is an id-generation bug, not a
    /// normal path — implementors MUST answer [`BackendError::Conflict`],
    /// which is the natural mapping for a DynamoDB conditional `PutItem`
    /// failing with `ConditionalCheckFailedException`.
    async fn put_new(&self, owner_id: &str, record: StoredTask) -> Result<(), BackendError>;

    /// Fetch one LIVE record. `Ok(None)` when absent OR expired.
    async fn get(&self, owner_id: &str, task_id: &str) -> Result<Option<StoredTask>, BackendError>;

    /// Compare-and-swap on `version`. [`BackendError::Conflict`] on mismatch.
    async fn cas(
        &self,
        owner_id: &str,
        task_id: &str,
        expected_version: u64,
        record: StoredTask,
    ) -> Result<(), BackendError>;

    /// An owner's LIVE records, with an opaque backend-defined cursor.
    async fn list(
        &self,
        owner_id: &str,
        cursor: Option<&str>,
    ) -> Result<(Vec<StoredTask>, Option<String>), BackendError>;

    /// Drop expired records, returning how many. The in-memory backend needs
    /// this because nothing else reclaims; DynamoDB has native TTL and can
    /// answer 0.
    async fn sweep_expired(&self) -> Result<usize, BackendError>;
}

/// The process-local backend: what the stdio server and every test here use.
#[derive(Debug, Default)]
pub struct InMemoryTaskBackend {
    /// `owner_id -> task_id -> record`.
    records: Mutex<HashMap<String, HashMap<String, StoredTask>>>,
}

impl InMemoryTaskBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn guard(&self) -> MutexGuard<'_, HashMap<String, HashMap<String, StoredTask>>> {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A record is live when it has no expiry or its expiry is still ahead.
fn is_live(record: &StoredTask, now_secs: u64) -> bool {
    record.expires_at_secs.is_none_or(|e| now_secs <= e)
}

#[async_trait]
impl TaskBackend for InMemoryTaskBackend {
    async fn put_new(&self, owner_id: &str, record: StoredTask) -> Result<(), BackendError> {
        let mut records = self.guard();
        let owned = records.entry(owner_id.to_string()).or_default();
        if owned.contains_key(&record.task.task_id) {
            return Err(BackendError::Conflict);
        }
        owned.insert(record.task.task_id.clone(), record);
        Ok(())
    }

    async fn get(&self, owner_id: &str, task_id: &str) -> Result<Option<StoredTask>, BackendError> {
        let now = now_epoch_secs();
        Ok(self
            .guard()
            .get(owner_id)
            .and_then(|owned| owned.get(task_id))
            .filter(|record| is_live(record, now))
            .cloned())
    }

    async fn cas(
        &self,
        owner_id: &str,
        task_id: &str,
        expected_version: u64,
        record: StoredTask,
    ) -> Result<(), BackendError> {
        let mut records = self.guard();
        let current = records
            .get_mut(owner_id)
            .and_then(|owned| owned.get_mut(task_id))
            .ok_or(BackendError::Store(TaskStoreError::NotFound {
                task_id: task_id.to_string(),
            }))?;
        if current.version != expected_version {
            return Err(BackendError::Conflict);
        }
        *current = record;
        Ok(())
    }

    async fn list(
        &self,
        owner_id: &str,
        _cursor: Option<&str>,
    ) -> Result<(Vec<StoredTask>, Option<String>), BackendError> {
        let now = now_epoch_secs();
        let mut owned: Vec<StoredTask> = self
            .guard()
            .get(owner_id)
            .map(|owned| {
                owned
                    .values()
                    .filter(|record| is_live(record, now))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        // Newest first, matching the SDK's own in-memory store.
        owned.sort_by(|a, b| b.task.created_at.cmp(&a.task.created_at));
        Ok((owned, None))
    }

    async fn sweep_expired(&self) -> Result<usize, BackendError> {
        let now = now_epoch_secs();
        let mut records = self.guard();
        let mut removed = 0;
        for owned in records.values_mut() {
            let before = owned.len();
            owned.retain(|_, record| is_live(record, now));
            removed += before - owned.len();
        }
        Ok(removed)
    }
}

/// A single request's armed (handler-minted, not-yet-consumed) task.
struct PendingMint {
    task: Task,
    owner_id: String,
    armed_at_ms: u64,
}

/// The mint-handoff primitive (chess D-17): see the module doc.
///
/// # Why one slot is enough
///
/// pmcp serializes request handling through a single worker
/// (`Server::spawn_request_worker` — "request handling stays serialized"), so
/// the handler body and the `store.create()` that follows it run in ONE
/// critical section per process: at most one legitimate armed mint is in
/// flight. Chess reaches the same conclusion from its
/// `Arc<tokio::sync::Mutex<Server>>` wrapping. Any change that lets tool
/// handlers run concurrently invalidates this argument and must re-run
/// the handoff tests below.
#[derive(Default)]
pub struct RequestMintHandoff {
    pending: Mutex<Option<PendingMint>>,
}

impl RequestMintHandoff {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn guard(&self) -> MutexGuard<'_, Option<PendingMint>> {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Drop any armed mint. Every minting handler calls this ON ENTRY, so a
    /// mint armed by an earlier call that never opened the create gate — a
    /// plain, non-task-augmented call, for instance — cannot be consumed by a
    /// LATER unrelated request.
    pub fn clear(&self) {
        *self.guard() = None;
    }

    /// Record the task this request minted, as the handler's last step.
    pub fn arm(&self, owner_id: &str, task: &Task) {
        *self.guard() = Some(PendingMint {
            task: task.clone(),
            owner_id: owner_id.to_string(),
            armed_at_ms: now_epoch_millis(),
        });
    }

    /// Consume-and-clear. Returns the armed task ONLY when the owner matches
    /// byte-for-byte AND it is younger than [`HANDOFF_MAX_AGE_MS`]. A present
    /// but non-matching arm is DROPPED rather than returned, so a stale or
    /// owner-mismatched arm can never be handed to the wrong caller and a miss
    /// never leaves a dangling arm behind.
    pub fn take_matching(&self, owner_id: &str, now_ms: u64) -> Option<Task> {
        let pending = self.guard().take()?;
        let fresh = now_ms.saturating_sub(pending.armed_at_ms) <= HANDOFF_MAX_AGE_MS;
        (pending.owner_id == owner_id && fresh).then_some(pending.task)
    }
}

#[must_use]
pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[must_use]
pub fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn rfc3339_now() -> String {
    // Second precision is what the SDK's own store emits, and the field is
    // documentation for clients rather than an ordering key.
    let secs = now_epoch_secs();
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (y, mo, d) = civil_from_days(i64::try_from(days).unwrap_or(0));
    format!(
        "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's `civil_from_days`, so a timestamp needs no date crate.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Where a successful `tasks/cancel` is relayed so the work actually stops.
///
/// The store owns the RECORD; something else owns the running child. Cancelling
/// the record without stopping the work leaves a CPU-saturating trainer alive
/// and single-flight refusing every later submit, so the two must be wired
/// together — but the store must not know what a training child is.
pub trait CancelSink: Send + Sync {
    /// Stop the work behind `task_id`, if this process is running it.
    fn cancel(&self, task_id: &str);
}

/// The training server's task store.
pub struct AprenderTaskStore {
    backend: Arc<dyn TaskBackend>,
    handoff: RequestMintHandoff,
    config: StoreConfig,
    cancel_sink: Option<Arc<dyn CancelSink>>,
}

impl std::fmt::Debug for AprenderTaskStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AprenderTaskStore").finish_non_exhaustive()
    }
}

impl AprenderTaskStore {
    #[must_use]
    pub fn new(backend: Arc<dyn TaskBackend>) -> Self {
        Self {
            backend,
            handoff: RequestMintHandoff::new(),
            config: StoreConfig::default(),
            cancel_sink: None,
        }
    }

    /// Wire the relay a successful `tasks/cancel` fires into.
    #[must_use]
    pub fn with_cancel_sink(mut self, sink: Arc<dyn CancelSink>) -> Self {
        self.cancel_sink = Some(sink);
        self
    }

    /// Drop any armed mint — every minting handler's first step.
    pub fn clear_handoff(&self) {
        self.handoff.clear();
    }

    fn ttl_ms(&self, requested: Option<u64>) -> u64 {
        requested
            .or(self.config.default_ttl_ms)
            .unwrap_or(DEFAULT_TTL_MS)
    }

    /// Mint a `Working` task and persist it. Does NOT arm the handoff.
    async fn mint(&self, owner_id: &str, ttl_ms: Option<u64>) -> Result<Task, TaskStoreError> {
        let ttl = self.ttl_ms(ttl_ms);
        let now = rfc3339_now();
        // `Task` is #[non_exhaustive]: the builder is the only construction
        // path that survives the SDK adding a field.
        let task = Task::new(mint_id(), TaskStatus::Working)
            .with_ttl(ttl)
            .with_timestamps(now.clone(), now);
        let task = task.with_poll_interval(self.config.default_poll_interval_ms);
        let record = StoredTask {
            task: task.clone(),
            result: None,
            envelope: None,
            version: 1,
            expires_at_secs: Some(now_epoch_secs() + ttl / 1000),
        };
        self.backend.put_new(owner_id, record).await?;
        Ok(task)
    }

    /// The handler-facing mint: mint, then ARM the handoff. This is the only
    /// method that arms for a fresh mint.
    ///
    /// # Errors
    ///
    /// Whatever the backend's insert reports.
    pub async fn mint_for_request(
        &self,
        owner_id: &str,
        ttl_ms: Option<u64>,
    ) -> Result<Task, TaskStoreError> {
        let task = self.mint(owner_id, ttl_ms).await?;
        self.handoff.arm(owner_id, &task);
        Ok(task)
    }

    /// The ONE place a missing / expired / wrong-owner record becomes
    /// `NotFound`. Every read routes through here, so liveness is enforced by
    /// the store on top of whatever the backend filtered — DynamoDB's TTL
    /// deletion is asynchronous, so an expired item really can come back from
    /// a query, and this makes that harmless.
    async fn load(&self, owner_id: &str, task_id: &str) -> Result<StoredTask, TaskStoreError> {
        self.backend
            .get(owner_id, task_id)
            .await?
            .ok_or_else(|| TaskStoreError::NotFound {
                task_id: task_id.to_string(),
            })
    }

    /// Read-modify-write under the record's CAS token, retried once on a lost
    /// race. One retry is enough because the only concurrent writers are the
    /// worker's terminal write and a client's cancel.
    async fn cas_update<F>(
        &self,
        owner_id: &str,
        task_id: &str,
        mutate: F,
    ) -> Result<StoredTask, TaskStoreError>
    where
        F: Fn(&mut StoredTask) -> Result<(), TaskStoreError>,
    {
        for _ in 0..2 {
            let mut record = self.load(owner_id, task_id).await?;
            let expected = record.version;
            mutate(&mut record)?;
            record.version = expected + 1;
            record.task.last_updated_at = rfc3339_now();
            match self
                .backend
                .cas(owner_id, task_id, expected, record.clone())
                .await
            {
                Ok(()) => return Ok(record),
                Err(BackendError::Conflict) => continue,
                Err(other) => return Err(other.into()),
            }
        }
        Err(TaskStoreError::Internal {
            message: "task record conflict (concurrent write)".to_string(),
        })
    }

    /// Stash this run's inputs for whoever finishes the work. OFF the trait on
    /// purpose — see the module doc.
    ///
    /// # Errors
    ///
    /// `NotFound` when the task is gone; otherwise the backend's error.
    pub async fn put_envelope(
        &self,
        task_id: &str,
        owner_id: &str,
        envelope: serde_json::Value,
    ) -> Result<(), TaskStoreError> {
        self.cas_update(owner_id, task_id, |record| {
            record.envelope = Some(envelope.clone());
            Ok(())
        })
        .await
        .map(|_| ())
    }

    /// Read back what [`Self::put_envelope`] stored.
    ///
    /// # Errors
    ///
    /// `NotFound` when the task is gone or expired.
    pub async fn get_envelope(
        &self,
        task_id: &str,
        owner_id: &str,
    ) -> Result<Option<serde_json::Value>, TaskStoreError> {
        Ok(self.load(owner_id, task_id).await?.envelope)
    }

    /// The terminal write, as ONE guarded step: persist the result and move to
    /// a terminal status together.
    ///
    /// `Working -> terminal` proceeds. `terminal -> a different terminal` is
    /// suppressed as a no-op SUCCESS rather than overwriting — that is a second
    /// writer racing the first (a straggling worker against a cancel), and the
    /// first terminal write is the one that happened.
    ///
    /// # Errors
    ///
    /// `NotFound` when the task is gone or expired.
    pub async fn finish(
        &self,
        task_id: &str,
        owner_id: &str,
        status: TaskStatus,
        result: CallToolResult,
    ) -> Result<(), TaskStoreError> {
        self.cas_update(owner_id, task_id, |record| {
            if record.task.status.is_terminal() {
                return Ok(());
            }
            record.result = Some(result.clone());
            record.task.status = status;
            Ok(())
        })
        .await
        .map(|_| ())
    }
}

/// A v4-shaped random id, without a uuid dependency: 122 random bits in the
/// canonical layout. Collisions are not a practical concern and `put_new`
/// answers `Conflict` if one ever happened.
///
/// Public because it is the ONE id shape this server mints — tasks here, and
/// dataset upload slots in the cloud dispatcher — so a client sees one kind of
/// handle rather than two.
#[must_use]
pub fn mint_id() -> String {
    let mut bytes = [0u8; 16];
    // Two independent entropy sources xor'd per byte: the OS-seeded hasher
    // state and the monotonic clock. Neither is a CSPRNG on its own; a task id
    // is an unguessable handle, not a secret, and this keeps the crate free of
    // a rand dependency for one call site.
    for (i, byte) in bytes.iter_mut().enumerate() {
        let h = std::collections::hash_map::RandomState::new();
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = h.build_hasher();
        hasher.write_usize(i);
        hasher.write_u64(now_epoch_millis());
        *byte = u8::try_from(hasher.finish() & 0xff).unwrap_or(0);
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

#[async_trait]
impl TaskStore for AprenderTaskStore {
    /// The create gate. Consults the handoff FIRST so a handler-minted task is
    /// returned rather than a second one being minted (chess D-17).
    async fn create(&self, owner_id: &str, ttl: Option<u64>) -> Result<Task, TaskStoreError> {
        if let Some(armed) = self.handoff.take_matching(owner_id, now_epoch_millis()) {
            // Prefer the STORED record: the handler may have moved the task on
            // (a dispatch failure compensating to `failed`) between arming and
            // the gate, and the client must see that, not the armed snapshot.
            if let Ok(Some(record)) = self.backend.get(owner_id, &armed.task_id).await {
                return Ok(record.task);
            }
            return Ok(armed);
        }
        self.mint(owner_id, ttl).await
    }

    async fn get(&self, task_id: &str, owner_id: &str) -> Result<Task, TaskStoreError> {
        Ok(self.load(owner_id, task_id).await?.task)
    }

    async fn update_status(
        &self,
        task_id: &str,
        owner_id: &str,
        status: TaskStatus,
        message: Option<String>,
    ) -> Result<Task, TaskStoreError> {
        let record = self
            .cas_update(owner_id, task_id, |record| {
                // Same guard as `finish`: never overwrite a terminal verdict
                // with a different one.
                if record.task.status.is_terminal() && status.is_terminal() {
                    return Ok(());
                }
                record.task.status = status;
                record.task.status_message.clone_from(&message);
                Ok(())
            })
            .await?;
        Ok(record.task)
    }

    async fn list(
        &self,
        owner_id: &str,
        cursor: Option<&str>,
    ) -> Result<(Vec<Task>, Option<String>), TaskStoreError> {
        let (records, next) = self.backend.list(owner_id, cursor).await?;
        Ok((records.into_iter().map(|r| r.task).collect(), next))
    }

    async fn cancel(&self, task_id: &str, owner_id: &str) -> Result<Task, TaskStoreError> {
        let record = self
            .cas_update(owner_id, task_id, |record| {
                if record.task.status.is_terminal() {
                    return Err(TaskStoreError::InvalidTransition {
                        task_id: task_id.to_string(),
                        from: record.task.status,
                        to: TaskStatus::Cancelled,
                    });
                }
                record.task.status = TaskStatus::Cancelled;
                Ok(())
            })
            .await?;
        // Only after the record is durably Cancelled: a relay fired before the
        // write could stop the work for a cancel that then failed to persist.
        if let Some(sink) = self.cancel_sink.as_ref() {
            sink.cancel(task_id);
        }
        Ok(record.task)
    }

    async fn cleanup_expired(&self) -> Result<usize, TaskStoreError> {
        Ok(self.backend.sweep_expired().await?)
    }

    fn config(&self) -> &StoreConfig {
        &self.config
    }

    async fn set_result(
        &self,
        task_id: &str,
        owner_id: &str,
        result: CallToolResult,
    ) -> Result<(), TaskStoreError> {
        self.cas_update(owner_id, task_id, |record| {
            record.result = Some(result.clone());
            Ok(())
        })
        .await
        .map(|_| ())
    }

    async fn get_result(
        &self,
        task_id: &str,
        owner_id: &str,
    ) -> Result<CallToolResult, TaskStoreError> {
        self.load(owner_id, task_id)
            .await?
            .result
            .ok_or_else(|| TaskStoreError::NotFound {
                task_id: task_id.to_string(),
            })
    }

    fn supports_results(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally
mod tests {
    use super::*;

    fn store() -> AprenderTaskStore {
        AprenderTaskStore::new(Arc::new(InMemoryTaskBackend::new()))
    }

    fn result(text: &str) -> CallToolResult {
        CallToolResult::new(vec![pmcp::types::Content::Text {
            text: text.to_string(),
        }])
    }

    #[tokio::test]
    async fn the_handoff_makes_create_return_the_handler_minted_task() {
        let store = store();
        store.clear_handoff();
        let minted = store.mint_for_request("owner-1", None).await.expect("mint");
        // This is what pmcp's create gate does after the handler returns.
        let created = store.create("owner-1", None).await.expect("create");
        assert_eq!(
            created.task_id, minted.task_id,
            "the store must not mint a SECOND id — the client would poll one \
             id while the worker updates the other"
        );
    }

    #[tokio::test]
    async fn the_handoff_is_consumed_once_and_a_later_create_mints_fresh() {
        let store = store();
        let minted = store.mint_for_request("owner-1", None).await.expect("mint");
        let first = store.create("owner-1", None).await.expect("first");
        assert_eq!(first.task_id, minted.task_id);
        let second = store.create("owner-1", None).await.expect("second");
        assert_ne!(
            second.task_id, minted.task_id,
            "an unrelated later request must get its own task"
        );
    }

    #[tokio::test]
    async fn an_owner_mismatch_drops_the_arm_rather_than_handing_it_over() {
        let store = store();
        let minted = store.mint_for_request("owner-1", None).await.expect("mint");
        let other = store.create("owner-2", None).await.expect("other owner");
        assert_ne!(other.task_id, minted.task_id, "never hand across owners");
        // ...and the dropped arm must not linger for owner-1 either.
        let later = store.create("owner-1", None).await.expect("later");
        assert_ne!(
            later.task_id, minted.task_id,
            "a miss leaves no dangling arm"
        );
    }

    #[tokio::test]
    async fn clear_stops_a_plain_calls_mint_being_adopted_by_a_later_request() {
        let store = store();
        // A plain (non-task-augmented) call mints and arms, but no create gate
        // ever runs for it.
        let plain = store.mint_for_request("owner-1", None).await.expect("mint");
        // The next minting handler clears on entry.
        store.clear_handoff();
        let next = store.mint_for_request("owner-1", None).await.expect("mint");
        let created = store.create("owner-1", None).await.expect("create");
        assert_eq!(created.task_id, next.task_id);
        assert_ne!(
            created.task_id, plain.task_id,
            "the stale arm from the plain call must not be adopted"
        );
    }

    #[tokio::test]
    async fn a_stale_arm_is_not_consumed() {
        let store = store();
        let minted = store.mint_for_request("owner-1", None).await.expect("mint");
        // Ask as if the gate ran long after the arm.
        let taken = store
            .handoff
            .take_matching("owner-1", now_epoch_millis() + HANDOFF_MAX_AGE_MS + 1);
        assert!(taken.is_none(), "a stale arm must not be handed over");
        let created = store.create("owner-1", None).await.expect("create");
        assert_ne!(created.task_id, minted.task_id);
    }

    #[tokio::test]
    async fn the_envelope_never_reaches_the_wire_but_round_trips_for_the_worker() {
        let store = store();
        let task = store.mint_for_request("owner-1", None).await.expect("mint");
        store
            .put_envelope(&task.task_id, "owner-1", serde_json::json!({"seed": 17}))
            .await
            .expect("put envelope");
        let back = store
            .get_envelope(&task.task_id, "owner-1")
            .await
            .expect("get envelope")
            .expect("some envelope");
        assert_eq!(back["seed"], 17);
        // The trait surface a client reaches carries no envelope.
        let served = store.get(&task.task_id, "owner-1").await.expect("get");
        let json = serde_json::to_value(&served).expect("task serializes");
        assert!(
            json.get("envelope").is_none(),
            "the SDK must never serve an envelope: {json}"
        );
    }

    #[tokio::test]
    async fn a_terminal_write_wins_and_a_second_one_does_not_overwrite_it() {
        let store = store();
        let task = store.mint_for_request("owner-1", None).await.expect("mint");
        store
            .finish(
                &task.task_id,
                "owner-1",
                TaskStatus::Completed,
                result("first"),
            )
            .await
            .expect("first terminal write");
        // A straggling worker racing a cancel must not rewrite the verdict.
        store
            .finish(
                &task.task_id,
                "owner-1",
                TaskStatus::Failed,
                result("second"),
            )
            .await
            .expect("second write is a no-op success");
        let served = store.get(&task.task_id, "owner-1").await.expect("get");
        assert_eq!(served.status, TaskStatus::Completed);
        let got = store
            .get_result(&task.task_id, "owner-1")
            .await
            .expect("result");
        match got.content.first().expect("content") {
            pmcp::types::Content::Text { text } => assert_eq!(text, "first"),
            other => panic!("expected text: {other:?}"),
        }
    }

    #[tokio::test]
    async fn owner_scoping_hides_another_owners_task() {
        let store = store();
        let task = store.mint_for_request("owner-1", None).await.expect("mint");
        let err = store
            .get(&task.task_id, "owner-2")
            .await
            .expect_err("another owner must not read it");
        assert!(matches!(err, TaskStoreError::NotFound { .. }), "{err:?}");
    }

    #[tokio::test]
    async fn cancel_refuses_a_task_that_already_finished() {
        let store = store();
        let task = store.mint_for_request("owner-1", None).await.expect("mint");
        store
            .finish(&task.task_id, "owner-1", TaskStatus::Completed, result("x"))
            .await
            .expect("finish");
        let err = store
            .cancel(&task.task_id, "owner-1")
            .await
            .expect_err("cancelling a finished task is a transition error");
        assert!(
            matches!(err, TaskStoreError::InvalidTransition { .. }),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn an_expired_record_reads_as_not_found_and_sweeps() {
        let backend = Arc::new(InMemoryTaskBackend::new());
        let store = AprenderTaskStore::new(backend.clone());
        // A 1 ms TTL is already in the past by the time we read.
        let task = store
            .mint_for_request("owner-1", Some(1))
            .await
            .expect("mint");
        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        let err = store
            .get(&task.task_id, "owner-1")
            .await
            .expect_err("an expired task is NotFound");
        assert!(matches!(err, TaskStoreError::NotFound { .. }), "{err:?}");
        assert_eq!(store.cleanup_expired().await.expect("sweep"), 1);
    }

    #[test]
    fn minted_ids_are_distinct_and_v4_shaped() {
        let a = mint_id();
        let b = mint_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), 36, "{a}");
        assert_eq!(a.as_bytes()[14], b'4', "version nibble: {a}");
        assert!(matches!(a.as_bytes()[19], b'8' | b'9' | b'a' | b'b'), "{a}");
    }

    #[test]
    fn timestamps_are_rfc3339_utc() {
        let ts = rfc3339_now();
        assert_eq!(ts.len(), 20, "{ts}");
        assert!(ts.ends_with('Z'), "{ts}");
        // 2026-xx and not 1970-xx: the epoch maths actually ran.
        assert!(ts.starts_with("202"), "{ts}");
    }
}
