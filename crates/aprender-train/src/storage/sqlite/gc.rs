//! Run reaper: `apr runs gc` (EXT-03, aprender#4385).
//!
//! Two kinds of garbage, planned together and counted as one union:
//! - **tmp**: experiments named `.tmp*`. They are the file name of a test's
//!   `TempDir` output dir, leaked into the store by tests. Their runs are
//!   deleted along with their params, metrics, artifacts and span ids.
//! - **orphans**: `running` rows whose starting process is gone. They are
//!   marked `failed`, not deleted: the metrics of a crashed run are still
//!   evidence.
//!
//! A run in both sets counts once (it is deleted). A live run is never
//! touched, even inside a `.tmp*` experiment, and a row from another host is
//! spared because this host cannot see that host's pids.

use super::backend::SqliteBackend;
use super::liveness;
use crate::storage::{Result, StorageError};
use chrono::Utc;
use rusqlite::params;
use std::collections::BTreeSet;
use std::path::Path;

/// The liveness tuple a run recorded at `start_run` (all `None` before EXT-03).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordedIdentity {
    pub host: Option<String>,
    pub boot_id: Option<String>,
    pub pid: Option<u32>,
    pub start_ticks: Option<u64>,
}

/// What liveness says about one `running` row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// The recorded 4-tuple matches a process that exists now.
    Live,
    /// Recorded on another host: not judgeable here, so spared.
    Foreign,
    /// Same host, but the boot, the pid or the pid's start time differs.
    Dead,
    /// No tuple recorded (written before EXT-03): nothing can prove it live.
    Unrecorded,
}

/// Judge a recorded identity against this host.
///
/// `probe(pid)` returns the start ticks of `pid` now, or `None` if it is gone.
pub(crate) fn classify(
    rec: &RecordedIdentity,
    here_host: &str,
    here_boot: &str,
    probe: &dyn Fn(u32) -> Option<u64>,
) -> Verdict {
    let (Some(host), Some(boot), Some(pid), Some(ticks)) =
        (&rec.host, &rec.boot_id, rec.pid, rec.start_ticks)
    else {
        return Verdict::Unrecorded;
    };
    if host != here_host {
        return Verdict::Foreign;
    }
    if boot == here_boot && probe(pid) == Some(ticks) {
        Verdict::Live
    } else {
        Verdict::Dead
    }
}

/// A dry-run: what `gc` would do, computed without writing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GcPlan {
    /// Runs of `.tmp*` experiments that are not live or foreign: deleted.
    pub tmp_runs: BTreeSet<String>,
    /// Every non-live `running` row, including any inside a `.tmp*` experiment.
    pub orphan_runs: BTreeSet<String>,
    /// `.tmp*` experiments left with no run after the deletion: deleted.
    pub tmp_experiments: BTreeSet<String>,
    /// `running` rows proven live: untouched.
    pub spared_live: BTreeSet<String>,
    /// `running` rows recorded on another host: untouched.
    pub spared_foreign: BTreeSet<String>,
}

impl GcPlan {
    /// `tmp ∪ orphans`, each run once.
    pub fn union(&self) -> BTreeSet<&str> {
        self.tmp_runs.iter().chain(&self.orphan_runs).map(String::as_str).collect()
    }

    /// Orphans that are not also deleted as tmp: these get marked `failed`.
    pub fn orphans_to_fail(&self) -> impl Iterator<Item = &String> {
        self.orphan_runs.iter().filter(|r| !self.tmp_runs.contains(*r))
    }
}

/// What `gc_apply` changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GcReport {
    pub runs_deleted: usize,
    pub runs_failed: usize,
    pub experiments_deleted: usize,
}

struct RunRow {
    id: String,
    experiment_id: String,
    exp_name: String,
    status: String,
    rec: RecordedIdentity,
}

impl SqliteBackend {
    /// Plan a gc against this host's live processes.
    pub fn gc_plan(&self) -> Result<GcPlan> {
        let host = liveness::host_name().unwrap_or_default();
        let boot = liveness::boot_id().unwrap_or_default();
        self.gc_plan_with(&host, &boot, &|pid| {
            liveness::ProcIdentity::of_pid(pid).map(|p| p.start_ticks)
        })
    }

    /// Plan a gc with an explicit host, boot and pid probe (tests plant these).
    pub(crate) fn gc_plan_with(
        &self,
        here_host: &str,
        here_boot: &str,
        probe: &dyn Fn(u32) -> Option<u64>,
    ) -> Result<GcPlan> {
        let rows = self.gc_rows()?;
        let mut plan = GcPlan::default();
        let mut tmp_exps: BTreeSet<String> = BTreeSet::new();
        let mut kept_in_tmp: BTreeSet<String> = BTreeSet::new();

        for row in &rows {
            let is_tmp = row.exp_name.starts_with(".tmp");
            if is_tmp {
                tmp_exps.insert(row.experiment_id.clone());
            }
            let verdict =
                (row.status == "running").then(|| classify(&row.rec, here_host, here_boot, probe));
            match verdict {
                Some(Verdict::Live) => {
                    plan.spared_live.insert(row.id.clone());
                }
                Some(Verdict::Foreign) => {
                    plan.spared_foreign.insert(row.id.clone());
                }
                Some(Verdict::Dead | Verdict::Unrecorded) => {
                    plan.orphan_runs.insert(row.id.clone());
                }
                None => {}
            }
            let spared = matches!(verdict, Some(Verdict::Live | Verdict::Foreign));
            if is_tmp && !spared {
                plan.tmp_runs.insert(row.id.clone());
            } else if is_tmp {
                kept_in_tmp.insert(row.experiment_id.clone());
            }
        }

        // `.tmp*` experiments with no runs at all are garbage too.
        for (id, name) in self.gc_experiments()? {
            if name.starts_with(".tmp") {
                tmp_exps.insert(id);
            }
        }
        plan.tmp_experiments = tmp_exps.difference(&kept_in_tmp).cloned().collect();
        Ok(plan)
    }

    /// Apply a plan in one transaction. Orphans are re-checked as still
    /// `running`, so a run that finished since the plan is left as it is.
    pub fn gc_apply(&self, plan: &GcPlan) -> Result<GcReport> {
        let conn = self.lock_conn()?;
        let tx = conn.unchecked_transaction().map_err(backend_err)?;
        let mut report = GcReport::default();

        for run in &plan.tmp_runs {
            for table in ["params", "metrics", "artifacts", "span_ids"] {
                tx.execute(&format!("DELETE FROM {table} WHERE run_id = ?1"), [run])
                    .map_err(backend_err)?;
            }
            report.runs_deleted +=
                tx.execute("DELETE FROM runs WHERE id = ?1", [run]).map_err(backend_err)?;
        }
        let now = Utc::now().to_rfc3339();
        for run in plan.orphans_to_fail() {
            report.runs_failed += tx
                .execute(
                    "UPDATE runs SET status = 'failed', end_time = ?1 WHERE id = ?2 AND status = 'running'",
                    params![now, run],
                )
                .map_err(backend_err)?;
        }
        for exp in &plan.tmp_experiments {
            report.experiments_deleted += tx
                .execute(
                    "DELETE FROM experiments WHERE id = ?1 \
                     AND NOT EXISTS (SELECT 1 FROM runs WHERE experiment_id = ?1)",
                    [exp],
                )
                .map_err(backend_err)?;
        }
        tx.commit().map_err(backend_err)?;
        Ok(report)
    }

    /// Write a consistent copy of the database to `dest` (`VACUUM INTO`).
    /// `dest` must not exist.
    pub fn backup_into(&self, dest: &Path) -> Result<()> {
        if dest.exists() {
            return Err(StorageError::Backend(format!(
                "backup target already exists: {}",
                dest.display()
            )));
        }
        let conn = self.lock_conn()?;
        conn.execute("VACUUM INTO ?1", [dest.to_string_lossy()]).map_err(backend_err)?;
        Ok(())
    }

    fn gc_rows(&self) -> Result<Vec<RunRow>> {
        let conn = self.lock_conn()?;
        let mut stmt = conn
            .prepare(
                "SELECT r.id, r.experiment_id, e.name, r.status, \
                 r.host, r.boot_id, r.pid, r.proc_start_time \
                 FROM runs r JOIN experiments e ON e.id = r.experiment_id",
            )
            .map_err(backend_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok(RunRow {
                    id: row.get(0)?,
                    experiment_id: row.get(1)?,
                    exp_name: row.get(2)?,
                    status: row.get(3)?,
                    rec: RecordedIdentity {
                        host: row.get(4)?,
                        boot_id: row.get(5)?,
                        pid: row.get::<_, Option<i64>>(6)?.and_then(|p| u32::try_from(p).ok()),
                        start_ticks: row
                            .get::<_, Option<i64>>(7)?
                            .and_then(|t| u64::try_from(t).ok()),
                    },
                })
            })
            .map_err(backend_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(backend_err)?;
        Ok(rows)
    }

    fn gc_experiments(&self) -> Result<Vec<(String, String)>> {
        let conn = self.lock_conn()?;
        let mut stmt = conn.prepare("SELECT id, name FROM experiments").map_err(backend_err)?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(backend_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(backend_err)?;
        Ok(rows)
    }

    /// Overwrite a run's recorded identity (tests plant dead and reused pids).
    #[cfg(test)]
    pub(crate) fn plant_identity(&self, run_id: &str, rec: &RecordedIdentity) -> Result<()> {
        let conn = self.lock_conn()?;
        conn.execute(
            "UPDATE runs SET host = ?1, boot_id = ?2, pid = ?3, proc_start_time = ?4 WHERE id = ?5",
            params![
                rec.host,
                rec.boot_id,
                rec.pid.map(i64::from),
                rec.start_ticks.and_then(|t| i64::try_from(t).ok()),
                run_id
            ],
        )
        .map_err(backend_err)?;
        Ok(())
    }
}

fn backend_err(e: rusqlite::Error) -> StorageError {
    StorageError::Backend(format!("gc: {e}"))
}
