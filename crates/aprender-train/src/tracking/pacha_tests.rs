//! EXT-04 (aprender#4386): `PachaBackend` round-trip, concurrent writers, and
//! FALSIFY-EXT-003 (a dangling metrics pointer is caught by `fsck`).

use super::*;
use crate::tracking::ulid::new_ulid;
use proptest::prelude::*;
use tempfile::TempDir;

fn paths(dir: &TempDir) -> (PathBuf, PathBuf) {
    (dir.path().join("pacha/registry.db"), dir.path().join("cache/metrics.db"))
}

fn sample_run(id: &str) -> Run {
    let mut run = Run::new(id.to_string(), Some("r".into()), "exp".into());
    run.params.insert("lr".into(), "6e-4".into());
    run.metrics.insert("loss".into(), vec![(10.5, 1), (f64::NAN, 2), (-0.0, 3)]);
    run.artifacts.push("model.apr".into());
    run.tags.insert("host".into(), "yoga".into());
    run.status = RunStatus::Completed;
    run.end_time_ms = run.start_time_ms.map(|t| t + 1);
    run
}

/// Runs compare by value, with metric values compared bit-for-bit.
fn assert_same(a: &Run, b: &Run) {
    let bits = |r: &Run| -> Vec<(String, Vec<(u64, u64)>)> {
        let mut m: Vec<_> = r
            .metrics
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|(x, s)| (x.to_bits(), *s)).collect()))
            .collect();
        m.sort();
        m
    };
    assert_eq!(bits(a), bits(b), "metric series differ");
    let strip = |r: &Run| Run { metrics: HashMap::new(), ..r.clone() };
    assert_eq!(format!("{:?}", strip(a).status), format!("{:?}", strip(b).status));
    assert_eq!(
        (&a.run_id, &a.run_name, &a.experiment_name, &a.params, &a.artifacts, &a.tags),
        (&b.run_id, &b.run_name, &b.experiment_name, &b.params, &b.artifacts, &b.tags)
    );
    assert_eq!((a.start_time_ms, a.end_time_ms), (b.start_time_ms, b.end_time_ms));
}

#[test]
fn round_trip_keeps_nan_and_negative_zero_and_lists_only_pointers() {
    let dir = TempDir::new().expect("tempdir");
    let (reg, met) = paths(&dir);
    let mut b = PachaBackend::open(&reg, &met).expect("open");
    let run = sample_run(&new_ulid());
    b.save_run(&run).expect("save");
    assert_same(&b.load_run(&run.run_id).expect("load"), &run);

    // A native pacha row (an `ExperimentRun`, with a recipe) is not a tracking run.
    b.registry
        .execute(
            "INSERT INTO runs VALUES ('pacha-native', 'recipe', '1', '{}', 'completed', '', NULL, '{\"id\":\"x\"}')",
            [],
        )
        .expect("plant native row");
    let listed = b.list_runs().expect("list");
    assert_eq!(listed.len(), 1);
    assert_same(&listed[0], &run);
    let report = b.fsck().expect("fsck");
    assert_eq!((report.pointers, report.other_rows), (1, 1));
    assert!(report.is_clean(), "{report:?}");

    b.delete_run(&run.run_id).expect("delete");
    assert!(matches!(b.load_run(&run.run_id), Err(TrackingStorageError::RunNotFound(_))));
    let orphans: i64 = b
        .metrics
        .query_row("SELECT COUNT(*) FROM tracking_metrics", [], |r| r.get(0))
        .expect("count");
    assert_eq!(orphans, 0, "delete must drop the metric series too");
}

#[test]
fn a_host_local_run_id_never_reaches_the_registry() {
    let dir = TempDir::new().expect("tempdir");
    let (reg, met) = paths(&dir);
    let mut b = PachaBackend::open(&reg, &met).expect("open");
    let err = b.save_run(&sample_run("run-1")).expect_err("run-1 is host-local");
    assert!(matches!(err, TrackingStorageError::InvalidRunId(ref id) if id == "run-1"), "{err}");
    assert_eq!(b.fsck().expect("fsck").pointers, 0);
}

#[test]
fn both_stores_are_wal() {
    let dir = TempDir::new().expect("tempdir");
    let (reg, met) = paths(&dir);
    let b = PachaBackend::open(&reg, &met).expect("open");
    for conn in [&b.registry, &b.metrics] {
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0)).expect("pragma");
        assert_eq!(mode, "wal");
    }
}

/// The DDL here must stay the DDL pacha creates, or the pointer rows break
/// pacha's own table.
#[test]
fn registry_ddl_matches_pacha() {
    let pacha = include_str!("../../../aprender-registry/src/registry/database.rs");
    let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let ours = squash(REGISTRY_RUNS_DDL);
    let body = ours.trim_end_matches(';');
    assert!(squash(pacha).contains(body), "pacha's runs DDL drifted from PachaBackend's:\n{ours}");
}

#[test]
fn two_concurrent_writers_lose_no_rows() {
    const PER_WRITER: usize = 40;
    let dir = TempDir::new().expect("tempdir");
    let (reg, met) = paths(&dir);
    // Create the schema once so the writers race on rows, not on DDL.
    drop(PachaBackend::open(&reg, &met).expect("open"));
    let written: Vec<Vec<String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let (reg, met) = (reg.clone(), met.clone());
                s.spawn(move || {
                    let mut b = PachaBackend::open(&reg, &met).expect("open");
                    (0..PER_WRITER)
                        .map(|_| {
                            let run = sample_run(&new_ulid());
                            b.save_run(&run).expect("concurrent save must not fail");
                            run.run_id
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("writer")).collect()
    });
    let want: std::collections::BTreeSet<String> = written.into_iter().flatten().collect();
    assert_eq!(want.len(), 3 * PER_WRITER);
    let b = PachaBackend::open(&reg, &met).expect("open");
    let got: std::collections::BTreeSet<String> =
        b.list_runs().expect("list").into_iter().map(|r| r.run_id).collect();
    assert_eq!(got, want, "a concurrent writer lost rows");
    assert!(b.fsck().expect("fsck").is_clean());
}

/// FALSIFY-EXT-003: each way a pointer can dangle is reported, and the live
/// store is clean.
#[test]
fn falsify_ext_003_fsck_dangling_pointer() {
    let dir = TempDir::new().expect("tempdir");
    let (reg, met) = paths(&dir);
    let mut b = PachaBackend::open(&reg, &met).expect("open");
    let ok = sample_run(&new_ulid());
    b.save_run(&ok).expect("save");
    assert!(b.fsck().expect("fsck").is_clean());

    // (1) metrics_db points at a file that does not exist.
    let gone = sample_run(&new_ulid());
    b.save_run(&gone).expect("save");
    b.registry
        .execute(
            "UPDATE runs SET run_json = replace(run_json, ?1, ?2) WHERE id = ?3",
            params![
                met.to_string_lossy(),
                dir.path().join("nowhere.db").to_string_lossy(),
                gone.run_id
            ],
        )
        .expect("plant missing db");
    // (2) the metrics DB exists but lost the run.
    let lost = sample_run(&new_ulid());
    b.save_run(&lost).expect("save");
    b.metrics.execute("DELETE FROM tracking_runs WHERE run_id = ?1", [&lost.run_id]).expect("lose");
    // (3) the row id and the pointer disagree.
    let moved = sample_run(&new_ulid());
    b.save_run(&moved).expect("save");
    b.registry
        .execute("UPDATE runs SET id = ?1 WHERE id = ?2", params![new_ulid(), moved.run_id])
        .expect("plant id mismatch");

    let report = b.fsck().expect("fsck");
    let reasons: Vec<&DanglingReason> = report.dangling.iter().map(|d| &d.reason).collect();
    assert_eq!(report.pointers, 4);
    assert_eq!(report.dangling.len(), 3, "{report:?}");
    for want in
        [DanglingReason::MissingMetricsDb, DanglingReason::MissingRun, DanglingReason::BadId]
    {
        assert!(reasons.contains(&&want), "{want:?} not reported: {report:?}");
    }
    assert!(!report.dangling.iter().any(|d| d.row_id == ok.run_id), "the good run is not dangling");
}

fn arb_run() -> impl Strategy<Value = Run> {
    let kv = prop::collection::hash_map("[a-z_]{1,8}", ".{0,12}", 0..4);
    let series = prop::collection::vec((any::<f64>(), any::<u64>()), 0..6);
    (
        kv.clone(),
        kv,
        prop::collection::hash_map("[a-z_./]{1,10}", series, 0..4),
        prop::collection::vec(".{0,16}", 0..3),
        prop::option::of(".{0,10}"),
        prop::sample::select(vec![
            RunStatus::Active,
            RunStatus::Completed,
            RunStatus::Failed,
            RunStatus::Cancelled,
        ]),
        prop::option::of(0u64..(1 << 47)),
    )
        .prop_map(|(params, tags, metrics, artifacts, name, status, end)| {
            let mut run = Run::new(new_ulid(), name, "prop".into());
            run.params = params;
            run.tags = tags;
            run.metrics = metrics;
            run.artifacts = artifacts;
            run.status = status;
            run.end_time_ms = end;
            run
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Round-trip: every run saved is loaded back unchanged, including after
    /// an overwrite with a shorter metric series.
    #[test]
    fn prop_round_trip(run in arb_run(), again in arb_run()) {
        let dir = TempDir::new().expect("tempdir");
        let (reg, met) = paths(&dir);
        let mut b = PachaBackend::open(&reg, &met).expect("open");
        b.save_run(&run).expect("save");
        assert_same(&b.load_run(&run.run_id).expect("load"), &run);
        let again = Run { run_id: run.run_id.clone(), ..again };
        b.save_run(&again).expect("overwrite");
        assert_same(&b.load_run(&run.run_id).expect("load"), &again);
        prop_assert!(b.fsck().expect("fsck").is_clean());
    }
}
