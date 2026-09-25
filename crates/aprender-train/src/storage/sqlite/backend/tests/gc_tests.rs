//! `apr runs gc` reaper tests (EXT-03, aprender#4385; FALSIFY-EXT-002).

use crate::storage::sqlite::backend::SqliteBackend;
use crate::storage::sqlite::gc::{classify, Verdict};
use crate::storage::sqlite::liveness::ProcIdentity;
use crate::storage::sqlite::RecordedIdentity;
use crate::storage::{ExperimentStorage, RunStatus};

fn rec(host: &str, boot: &str, pid: u32, ticks: u64) -> RecordedIdentity {
    RecordedIdentity {
        host: Some(host.into()),
        boot_id: Some(boot.into()),
        pid: Some(pid),
        start_ticks: Some(ticks),
    }
}

fn started(b: &mut SqliteBackend, exp: &str) -> String {
    let run = b.create_run(exp).expect("create_run");
    b.start_run(&run).expect("start_run");
    run
}

#[test]
fn classify_keys_on_the_full_tuple() {
    let probe = |pid: u32| (pid == 7).then_some(100);
    let v = |r: &RecordedIdentity| classify(r, "h", "b", &probe);
    assert_eq!(v(&rec("h", "b", 7, 100)), Verdict::Live);
    assert_eq!(v(&rec("h", "b", 7, 99)), Verdict::Dead, "pid reused within the boot");
    assert_eq!(v(&rec("h", "b0", 7, 100)), Verdict::Dead, "same pid+ticks, earlier boot");
    assert_eq!(v(&rec("h", "b", 8, 100)), Verdict::Dead, "pid gone");
    assert_eq!(v(&rec("other", "b", 8, 1)), Verdict::Foreign);
    assert_eq!(v(&RecordedIdentity::default()), Verdict::Unrecorded);
    let partial = RecordedIdentity { pid: Some(7), ..RecordedIdentity::default() };
    assert_eq!(v(&partial), Verdict::Unrecorded, "pid alone is not an identity");
}

/// FALSIFY-EXT-002. The planted live run holds a pid that a dead run also
/// recorded (the pid was reused); a reaper keyed on the pid alone either
/// spares the dead run or reaps the live one. The 4-tuple does neither.
#[cfg(target_os = "linux")]
#[test]
fn falsify_ext_002_gc_spares_live_and_reused_pid() {
    let mut b = SqliteBackend::open_in_memory().expect("open");
    let me = ProcIdentity::current().expect("linux has /proc");

    let train = b.create_experiment("train-a", None).expect("exp");
    let live = started(&mut b, &train);
    let reused = started(&mut b, &train);
    b.plant_identity(&reused, &rec(&me.host, &me.boot_id, me.pid, me.start_ticks + 1))
        .expect("plant");
    let old_boot = started(&mut b, &train);
    b.plant_identity(&old_boot, &rec(&me.host, "previous-boot", me.pid, me.start_ticks))
        .expect("plant");
    let foreign = started(&mut b, &train);
    b.plant_identity(&foreign, &rec("some-other-host", "x", 1, 1)).expect("plant");
    let legacy = started(&mut b, &train);
    b.plant_identity(&legacy, &RecordedIdentity::default()).expect("plant");
    let done = started(&mut b, &train);
    b.complete_run(&done, RunStatus::Success).expect("complete");

    let tmp = b.create_experiment(".tmpAB12cd", None).expect("exp");
    let tmp_dead = started(&mut b, &tmp);
    b.plant_identity(&tmp_dead, &RecordedIdentity::default()).expect("plant");
    let tmp_done = started(&mut b, &tmp);
    b.complete_run(&tmp_done, RunStatus::Success).expect("complete");
    b.log_metric(&tmp_done, "loss", 0, 1.0).expect("metric");
    let tmp_live = started(&mut b, &tmp);
    let tmp_empty = b.create_experiment(".tmpEMPTY0", None).expect("exp");

    let plan = b.gc_plan().expect("plan");
    let set = |v: &[&String]| v.iter().map(|s| (*s).clone()).collect();
    assert_eq!(plan.spared_live, set(&[&live, &tmp_live]));
    assert_eq!(plan.spared_foreign, set(&[&foreign]));
    assert_eq!(plan.orphan_runs, set(&[&reused, &old_boot, &legacy, &tmp_dead]));
    assert_eq!(plan.tmp_runs, set(&[&tmp_dead, &tmp_done]));
    assert_eq!(plan.tmp_experiments, set(&[&tmp_empty]), ".tmpAB12cd still holds a live run");
    // tmp_dead is in both sets: 2 tmp + 4 orphans = 6 memberships, 5 runs.
    assert_eq!(plan.union().len(), 5, "union counted once");

    let report = b.gc_apply(&plan).expect("apply");
    assert_eq!((report.runs_deleted, report.runs_failed, report.experiments_deleted), (2, 3, 1));

    for run in [&live, &tmp_live, &foreign] {
        assert_eq!(b.get_run_status(run).expect("kept"), RunStatus::Running, "{run} spared");
    }
    for run in [&reused, &old_boot, &legacy] {
        assert_eq!(b.get_run_status(run).expect("kept"), RunStatus::Failed, "{run} reaped");
    }
    assert_eq!(b.get_run_status(&done).expect("kept"), RunStatus::Success);
    assert!(b.get_run_status(&tmp_dead).is_err() && b.get_run_status(&tmp_done).is_err());
    let orphaned_metrics: i64 = b
        .lock_conn()
        .expect("lock")
        .query_row("SELECT COUNT(*) FROM metrics WHERE run_id = ?1", [&tmp_done], |r| r.get(0))
        .expect("count");
    assert_eq!(orphaned_metrics, 0, "children deleted with the run");

    let again = b.gc_plan().expect("replan");
    assert!(again.orphan_runs.is_empty(), "0 non-live running rows after gc");
    assert!(again.union().is_empty());
}

#[test]
fn gc_on_a_clean_store_plans_nothing() {
    let b = SqliteBackend::open_in_memory().expect("open");
    let plan = b.gc_plan().expect("plan");
    assert!(plan.union().is_empty() && plan.tmp_experiments.is_empty());
    assert_eq!(b.gc_apply(&plan).expect("apply"), Default::default());
}

#[test]
fn backup_into_copies_and_refuses_to_overwrite() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut b = SqliteBackend::open(dir.path().join("e.db")).expect("open");
    let exp = b.create_experiment("keep", None).expect("exp");
    let dest = dir.path().join("backup.db");
    b.backup_into(&dest).expect("backup");
    let copy = SqliteBackend::open(&dest).expect("open copy");
    assert_eq!(copy.list_experiments().expect("list").len(), 1, "{exp} copied");
    assert!(b.backup_into(&dest).is_err(), "existing backup is never overwritten");
}

#[test]
fn a_pre_ext03_database_gains_the_liveness_columns() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("old.db");
    {
        let conn = rusqlite::Connection::open(&path).expect("open");
        conn.execute_batch(
            "CREATE TABLE experiments (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT,
               config TEXT, tags TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now')),
               updated_at TEXT NOT NULL DEFAULT (datetime('now')));
             CREATE TABLE runs (id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending', start_time TEXT, end_time TEXT, tags TEXT);
             INSERT INTO experiments (id, name) VALUES ('e1', 'old');
             INSERT INTO runs (id, experiment_id, status) VALUES ('r1', 'e1', 'running');",
        )
        .expect("old schema");
    }
    let b = SqliteBackend::open(&path).expect("upgrade on open");
    let plan = b.gc_plan().expect("plan");
    assert!(plan.orphan_runs.contains("r1"), "an unrecorded running row is an orphan");
    drop(b);
    SqliteBackend::open(&path).expect("second open is idempotent");
}
