//! EXT-10 (aprender#4392): FALSIFY-EXT-008 — backfill never fabricates.

use super::*;
use tempfile::TempDir;

fn write(path: &Path, body: &str) {
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(path, body).expect("write");
}

/// Four legacy shapes seen under /mnt/nvme-raid0/runs, plus an empty dir.
fn root() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    let r = dir.path();
    // A checkpoint sidecar with some fields, no r.json.
    write(&r.join("ckpt-only/ckpt/epoch-000.apr"), "w0");
    write(
        &r.join("ckpt-only/ckpt/epoch-000.metadata.json"),
        r#"{"epoch":0,"train_loss":10.25,"val_loss":10.0}"#,
    );
    write(
        &r.join("ckpt-only/ckpt/epoch-001.metadata.json"),
        r#"{"epoch":1,"train_loss":9.5,"note":"text is not a metric"}"#,
    );
    // A result file that records success and per-step rows.
    write(
        &r.join("with-result/r.json"),
        r#"{"status":"OK","final_val_loss":10.08,"per_step_metrics":[{"step":0,"train_loss":11.0,"lr":0.0},{"step":1,"train_loss":10.5}]}"#,
    );
    write(&r.join("with-result/e.log"), "log");
    // A result that records a failure verbatim.
    write(
        &r.join("failed/r.json"),
        r#"{"status":"DIVERGED","per_step_metrics":[{"step":0,"train_loss":1e9}]}"#,
    );
    // A log only; a student model with a log.
    write(&r.join("log-only/launch.log"), "started");
    write(&r.join("distill/student.apr"), "s");
    std::fs::create_dir_all(r.join("empty")).expect("mkdir");
    dir
}

fn import(p: &[Planned], name: &str) -> Run {
    match &p
        .iter()
        .find(|p| p.source.ends_with(name))
        .expect(name)
        .outcome
    {
        Outcome::Import(run) => (**run).clone(),
        Outcome::Skip(reason) => panic!("{name} skipped: {reason}"),
    }
}

/// FALSIFY-EXT-008: every imported run is `provenance=backfilled`, with no
/// time, no hash and no status the source did not hold; metrics are exactly
/// the numeric fields on disk; and nothing is dropped without a reason.
#[test]
fn falsify_ext_008_backfill_never_fabricates() {
    let dir = root();
    let planned = plan(dir.path(), None).expect("plan");
    assert_eq!(planned.len(), 6, "every dir is accounted for");

    for p in &planned {
        if let Outcome::Import(run) = &p.outcome {
            assert_eq!(run.tags["provenance"], "backfilled");
            assert_eq!((run.start_time_ms, run.end_time_ms), (None, None));
            assert!(
                run.params
                    .keys()
                    .all(|k| !k.contains("sha") && !k.contains("hash")),
                "{:?}",
                run.params
            );
        }
    }

    let ckpt = import(&planned, "ckpt-only");
    assert_eq!(ckpt.status, RunStatus::Unknown, "no r.json, no status");
    let mut keys: Vec<_> = ckpt.metrics.keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["epoch.train_loss", "epoch.val_loss"]);
    assert_eq!(ckpt.metrics["epoch.train_loss"], [(10.25, 0), (9.5, 1)]);
    assert_eq!(ckpt.artifacts.len(), 1);

    let ok = import(&planned, "with-result");
    assert_eq!(ok.status, RunStatus::Completed);
    assert_eq!(ok.metrics["train_loss"], [(11.0, 0), (10.5, 1)]);
    assert_eq!(ok.metrics["lr"], [(0.0, 0)]);
    assert_eq!(ok.metrics["final_val_loss"], [(10.08, 0)]);

    let failed = import(&planned, "failed");
    assert_eq!(failed.status, RunStatus::Unknown, "only OK is mapped");
    assert_eq!(failed.params["recorded_status"], "DIVERGED");

    assert_eq!(import(&planned, "distill").artifacts.len(), 1);
    for (name, why) in [
        ("log-only", "nothing structured"),
        ("empty", "empty directory"),
    ] {
        match &planned
            .iter()
            .find(|p| p.source.ends_with(name))
            .expect(name)
            .outcome
        {
            Outcome::Skip(reason) => assert!(reason.contains(why), "{name}: {reason}"),
            Outcome::Import(_) => panic!("{name} must be skipped"),
        }
    }
}

/// A metadata file without an `epoch` field is unreadable, not guessed from
/// the file name.
#[test]
fn an_epoch_is_never_guessed_from_the_file_name() {
    let dir = TempDir::new().expect("tempdir");
    write(
        &dir.path().join("r1/ckpt/epoch-007.metadata.json"),
        r#"{"train_loss":1.0}"#,
    );
    write(&dir.path().join("r1/ckpt/epoch-007.apr"), "w");
    let Outcome::Import(run) = plan_dir(&dir.path().join("r1")) else {
        panic!("has a model file");
    };
    assert!(run.metrics.is_empty(), "{:?}", run.metrics);
    assert_eq!(run.params["unreadable"], "epoch-007.metadata.json");
}

/// Applying writes one backfilled run per import with NULL times in the
/// stored row, and a second import of the same root adds nothing.
#[test]
fn apply_is_idempotent_and_stores_nulls() {
    let dir = root();
    let home = dir.path().join("pacha-home");
    let planned = plan(dir.path(), None).expect("plan");
    let ids = apply(&home, &planned).expect("apply");
    assert_eq!(ids.len(), 4);

    let backend = open_backend(&home).expect("backend");
    let stored = backend.load_run(&ids[0]).expect("load");
    assert_eq!(stored.tags["provenance"], "backfilled");
    assert_eq!((stored.start_time_ms, stored.end_time_ms), (None, None));
    // The registry column is NOT NULL, so an unknown start is the empty
    // string there: never a time.
    let row = rusqlite_row_times(&home, &ids[0]);
    assert_eq!(row, (String::new(), None), "registry row has no time");

    let again = plan(dir.path(), Some(&backend)).expect("replan");
    assert!(again.iter().all(|p| matches!(p.outcome, Outcome::Skip(_))));
    assert_eq!(
        again
            .iter()
            .filter(|p| matches!(&p.outcome, Outcome::Skip(r) if r.starts_with("already imported")))
            .count(),
        4
    );
}

/// `(started_at, finished_at)` of run `id` in the pacha registry.
fn rusqlite_row_times(home: &Path, id: &str) -> (String, Option<String>) {
    let conn = rusqlite::Connection::open(RegistryConfig::new(home).db_path()).expect("open");
    conn.query_row(
        "SELECT started_at, finished_at FROM runs WHERE id = ?1",
        [id],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?)),
    )
    .expect("row")
}
