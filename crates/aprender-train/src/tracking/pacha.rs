//! `PachaBackend`: tracking runs recorded in the pacha registry (EXT-04, aprender#4386).
//!
//! EXT-001 §3.1: pacha is the system of record for runs, and the entrenar
//! SQLite file is the per-step metrics writer only. A run is therefore split:
//! - a **pointer row** in pacha's `runs` table: the ULID id, status, params,
//!   and a `run_json` of `{schema, run_id, metrics_db, summary, record}`;
//! - its **metric series** in the host metrics DB (`tracking_metrics`), keyed
//!   by the same ULID and stored as exact `f64` bits, so NaN and -0.0 survive.
//!
//! I-6 (referential integrity): every pointer resolves, i.e. its `metrics_db`
//! exists and holds its `run_id`. `save_run` commits the metrics before the
//! pointer and `delete_run` drops the pointer first, so a crash can orphan
//! metrics but never leave a dangling pointer. [`PachaBackend::fsck`] checks it.
//!
//! Both files run in WAL mode with a `busy_timeout`, and every write is a
//! `BEGIN IMMEDIATE` transaction, so concurrent writers wait instead of
//! failing a lock upgrade.
//!
//! Pointer rows have no recipe, so pacha's recipe queries never return them;
//! pacha's `get_run` does not parse them (they are not `ExperimentRun`s).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};

use super::storage::{Result, RunRecord, TrackingBackend, TrackingStorageError};
use super::ulid::is_ulid;
use super::{Run, RunStatus};

/// The `run_json.schema` tag of a tracking pointer row.
pub const POINTER_SCHEMA: &str = "entrenar.tracking.run/v1";

const BUSY_TIMEOUT_MS: u32 = 5000;

/// pacha's `runs` table, as `aprender-registry` creates it.
const REGISTRY_RUNS_DDL: &str = "CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    recipe_name TEXT,
    recipe_version TEXT,
    hyperparameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    run_json TEXT NOT NULL
);";

const METRICS_DDL: &str = "CREATE TABLE IF NOT EXISTS tracking_runs (run_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS tracking_metrics (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    seq INTEGER NOT NULL,
    step INTEGER NOT NULL,
    bits INTEGER NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, key, seq)
);";

/// The `run_json` of a pointer row.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Pointer {
    schema: String,
    run_id: String,
    metrics_db: PathBuf,
    /// Last value of each metric, for reading the registry without the metrics DB.
    summary: HashMap<String, f64>,
    /// The run with each metric's key but not its points.
    record: RunRecord,
}

/// Why a pointer row fails I-6.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DanglingReason {
    /// `run_json` is not JSON.
    Unparseable,
    /// The row id is not a ULID, or differs from `run_json.run_id`.
    BadId,
    /// `run_json.metrics_db` does not exist.
    MissingMetricsDb,
    /// The metrics DB exists but does not hold this run.
    MissingRun,
}

/// One pointer row that does not resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dangling {
    pub row_id: String,
    pub metrics_db: Option<PathBuf>,
    pub reason: DanglingReason,
}

/// What `fsck` found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FsckReport {
    /// Tracking pointer rows checked.
    pub pointers: usize,
    /// Registry rows that are not tracking pointers (pacha `ExperimentRun`s).
    pub other_rows: usize,
    pub dangling: Vec<Dangling>,
}

impl FsckReport {
    /// I-6 holds: no pointer dangles.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.dangling.is_empty()
    }
}

/// Tracking backend over a pacha registry plus a host metrics DB.
#[derive(Debug)]
pub struct PachaBackend {
    registry: Connection,
    metrics: Connection,
    metrics_path: PathBuf,
}

fn open_wal(path: &Path) -> Result<Connection> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(std::time::Duration::from_millis(u64::from(BUSY_TIMEOUT_MS)))?;
    conn.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))?;
    Ok(conn)
}

fn status_str(s: RunStatus) -> &'static str {
    match s {
        RunStatus::Active => "running",
        RunStatus::Completed => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
}

fn rfc3339(ms: Option<u64>) -> Option<String> {
    let ms = i64::try_from(ms?).ok()?;
    DateTime::<Utc>::from_timestamp_millis(ms).map(|t| t.to_rfc3339())
}

impl PachaBackend {
    /// Open (creating if absent) the registry at `registry_db` and the
    /// metrics DB at `metrics_db`. Both are switched to WAL mode.
    pub fn open(registry_db: impl AsRef<Path>, metrics_db: impl AsRef<Path>) -> Result<Self> {
        let registry = open_wal(registry_db.as_ref())?;
        registry.execute_batch(REGISTRY_RUNS_DDL)?;
        let metrics_path = std::path::absolute(metrics_db.as_ref())?;
        let metrics = open_wal(&metrics_path)?;
        metrics.execute_batch(METRICS_DDL)?;
        Ok(Self { registry, metrics, metrics_path })
    }

    /// The metrics DB this backend writes.
    #[must_use]
    pub fn metrics_path(&self) -> &Path {
        &self.metrics_path
    }

    fn write_metrics(&mut self, run: &Run) -> Result<()> {
        let tx = self.metrics.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM tracking_metrics WHERE run_id = ?1", [&run.run_id])?;
        tx.execute("INSERT OR IGNORE INTO tracking_runs (run_id) VALUES (?1)", [&run.run_id])?;
        {
            let mut ins = tx.prepare(
                "INSERT INTO tracking_metrics (run_id, key, seq, step, bits, value) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (key, series) in &run.metrics {
                for (seq, (value, step)) in series.iter().enumerate() {
                    ins.execute(params![
                        run.run_id,
                        key,
                        seq as i64,
                        *step as i64,
                        value.to_bits() as i64,
                        value
                    ])?;
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn write_pointer(&mut self, run: &Run) -> Result<()> {
        let mut record = RunRecord::from(run);
        // The series live in the metrics DB; the keys stay here, so a metric
        // declared with no points yet survives the round-trip.
        for series in record.metrics.values_mut() {
            series.clear();
        }
        let summary = run
            .metrics
            .iter()
            .filter_map(|(k, v)| v.last().map(|(val, _)| (k.clone(), *val)))
            .collect();
        let pointer = Pointer {
            schema: POINTER_SCHEMA.to_string(),
            run_id: run.run_id.clone(),
            metrics_db: self.metrics_path.clone(),
            summary,
            record,
        };
        let tx = self.registry.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT OR REPLACE INTO runs (id, recipe_name, recipe_version, hyperparameters_json, \
             status, started_at, finished_at, run_json) VALUES (?1, NULL, NULL, ?2, ?3, ?4, ?5, ?6)",
            params![
                run.run_id,
                serde_json::to_string(&run.params)?,
                status_str(run.status),
                rfc3339(run.start_time_ms).unwrap_or_default(),
                rfc3339(run.end_time_ms),
                serde_json::to_string(&pointer)?,
            ],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn pointer(&self, run_id: &str) -> Result<Pointer> {
        let json: Option<String> = self
            .registry
            .query_row("SELECT run_json FROM runs WHERE id = ?1", [run_id], |r| r.get(0))
            .optional()?;
        let json = json.ok_or_else(|| TrackingStorageError::RunNotFound(run_id.to_string()))?;
        let pointer: Pointer = serde_json::from_str(&json)?;
        if pointer.schema != POINTER_SCHEMA {
            return Err(TrackingStorageError::RunNotFound(run_id.to_string()));
        }
        Ok(pointer)
    }

    fn read_series(conn: &Connection, run_id: &str) -> Result<HashMap<String, Vec<(f64, u64)>>> {
        let mut stmt = conn.prepare(
            "SELECT key, step, bits FROM tracking_metrics WHERE run_id = ?1 ORDER BY key, seq",
        )?;
        let mut out: HashMap<String, Vec<(f64, u64)>> = HashMap::new();
        let rows = stmt.query_map([run_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))
        })?;
        for row in rows {
            let (key, step, bits) = row?;
            out.entry(key).or_default().push((f64::from_bits(bits as u64), step as u64));
        }
        Ok(out)
    }

    /// Check I-6 over every row of the registry: each tracking pointer names
    /// a metrics DB that exists and holds its run.
    pub fn fsck(&self) -> Result<FsckReport> {
        fsck_registry(&self.registry)
    }
}

/// `fsck` of the registry at `path`, opened read-only (`apr runs fsck`).
pub fn fsck_path(path: &Path) -> Result<FsckReport> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    conn.busy_timeout(std::time::Duration::from_millis(u64::from(BUSY_TIMEOUT_MS)))?;
    fsck_registry(&conn)
}

fn fsck_registry(registry: &Connection) -> Result<FsckReport> {
    let mut report = FsckReport::default();
    let mut stmt = registry.prepare("SELECT id, run_json FROM runs ORDER BY id")?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for (row_id, json) in rows {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&json) else {
            report.dangling.push(Dangling {
                row_id,
                metrics_db: None,
                reason: DanglingReason::Unparseable,
            });
            continue;
        };
        if value.get("schema").and_then(serde_json::Value::as_str) != Some(POINTER_SCHEMA) {
            report.other_rows += 1;
            continue;
        }
        report.pointers += 1;
        let pointer: Option<Pointer> = serde_json::from_value(value).ok();
        let reason = match &pointer {
            None => Some(DanglingReason::Unparseable),
            Some(p) if p.run_id != row_id || !is_ulid(&row_id) => Some(DanglingReason::BadId),
            Some(p) => resolve(&p.metrics_db, &p.run_id),
        };
        if let Some(reason) = reason {
            report.dangling.push(Dangling {
                row_id,
                metrics_db: pointer.map(|p| p.metrics_db),
                reason,
            });
        }
    }
    Ok(report)
}

/// `None` if `metrics_db` holds `run_id`, else why not.
fn resolve(metrics_db: &Path, run_id: &str) -> Option<DanglingReason> {
    if !metrics_db.is_file() {
        return Some(DanglingReason::MissingMetricsDb);
    }
    let held =
        Connection::open_with_flags(metrics_db, OpenFlags::SQLITE_OPEN_READ_ONLY).and_then(|c| {
            c.query_row("SELECT 1 FROM tracking_runs WHERE run_id = ?1", [run_id], |_| Ok(()))
                .optional()
        });
    match held {
        Ok(Some(())) => None,
        // No row, or no `tracking_runs` table at all: the run is not there.
        _ => Some(DanglingReason::MissingRun),
    }
}

impl TrackingBackend for PachaBackend {
    fn save_run(&mut self, run: &Run) -> Result<()> {
        if !is_ulid(&run.run_id) {
            return Err(TrackingStorageError::InvalidRunId(run.run_id.clone()));
        }
        // Metrics before pointer: a crash between the two leaves no dangling pointer.
        self.write_metrics(run)?;
        self.write_pointer(run)
    }

    fn load_run(&self, run_id: &str) -> Result<Run> {
        let pointer = self.pointer(run_id)?;
        let metrics = if pointer.metrics_db == self.metrics_path {
            Self::read_series(&self.metrics, run_id)?
        } else {
            let conn =
                Connection::open_with_flags(&pointer.metrics_db, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            Self::read_series(&conn, run_id)?
        };
        let mut run = pointer.record.into_run();
        run.metrics.extend(metrics);
        Ok(run)
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        let ids: Vec<String> = {
            let mut stmt = self
                .registry
                .prepare("SELECT id, run_json FROM runs WHERE recipe_name IS NULL ORDER BY id")?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows.into_iter()
                .filter(|(_, json)| {
                    serde_json::from_str::<serde_json::Value>(json).ok().and_then(|v| {
                        v.get("schema").and_then(serde_json::Value::as_str).map(str::to_string)
                    }) == Some(POINTER_SCHEMA.to_string())
                })
                .map(|(id, _)| id)
                .collect()
        };
        ids.iter().map(|id| self.load_run(id)).collect()
    }

    fn delete_run(&mut self, run_id: &str) -> Result<()> {
        self.pointer(run_id)?;
        // Pointer first: a crash between the two leaves orphan metrics, not a dangling pointer.
        self.registry.execute("DELETE FROM runs WHERE id = ?1", [run_id])?;
        let tx = self.metrics.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM tracking_metrics WHERE run_id = ?1", [run_id])?;
        tx.execute("DELETE FROM tracking_runs WHERE run_id = ?1", [run_id])?;
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "pacha_tests.rs"]
mod tests;
