//! EXT-22 (aprender#4404): `apr runs export` against a real pacha registry.

use super::*;
use entrenar::tracking::pacha::PachaBackend;
use entrenar::tracking::{ExperimentTracker, RunStatus};
use pacha::{Registry, RegistryConfig};
use tempfile::TempDir;

/// A pacha home with one finished tracking run and a lineage edge into it.
fn home() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let reg = dir.path().join("pacha").join("registry.db");
    let met = dir.path().join("pacha").join("tracking-metrics.db");
    let mut tracker =
        ExperimentTracker::new("ext-22", PachaBackend::open(&reg, &met).expect("open"));
    let id = tracker.start_run(None).expect("start");
    tracker.log_metric(&id, "loss", 1.5, 1).expect("metric");
    tracker.end_run(&id, RunStatus::Completed).expect("end");
    drop(tracker);
    let r = Registry::open(RegistryConfig::new(dir.path().join("pacha"))).expect("registry");
    r.add_lineage_edge("blake3:00", &id, "consumed", None)
        .expect("edge");
    (dir, reg)
}

/// FALSIFY-EXT-010: re-export of an unchanged host is byte-identical and leaves the file
/// alone; deleting one export row turns the ledger check RED, naming the row.
#[test]
fn falsify_ext_010_deleted_export_row_red() {
    let (dir, reg) = home();
    let out = dir.path().join("raid").join("lambda").join("pacha.jsonl");
    run_export(Some(&reg), &out, false, true).expect("export");
    let first = std::fs::read(&out).expect("read");
    assert_eq!(String::from_utf8_lossy(&first).lines().count(), 2);
    let mtime = std::fs::metadata(&out)
        .and_then(|m| m.modified())
        .expect("mtime");

    run_export(Some(&reg), &out, false, true).expect("re-export");
    assert_eq!(
        std::fs::read(&out).expect("read"),
        first,
        "I-7: byte-identical"
    );
    let again = std::fs::metadata(&out)
        .and_then(|m| m.modified())
        .expect("mtime");
    assert_eq!(mtime, again, "an unchanged export is not rewritten");
    run_export(Some(&reg), &out, true, true).expect("the ledger check passes on the export");

    // Plant: delete one row.
    let text = String::from_utf8(first).expect("utf8");
    let kept: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    std::fs::write(&out, &kept).expect("plant");
    let err = run_export(Some(&reg), &out, true, true).expect_err("a deleted row must be RED");
    assert!(
        err.to_string().contains("1 row(s) missing, 0 extra"),
        "{err}"
    );

    // Plant: a row the registry does not hold (a stale or foreign row).
    let stale = text.replacen("\"consumed\"", "\"produced\"", 1);
    assert_ne!(stale, text, "the plant must change a row");
    let row = stale
        .lines()
        .find(|l| !text.lines().any(|t| t == *l))
        .expect("changed row");
    std::fs::write(&out, format!("{text}{row}\n")).expect("plant");
    let err = run_export(Some(&reg), &out, true, true).expect_err("an extra row must be RED");
    assert!(
        err.to_string().contains("0 row(s) missing, 1 extra"),
        "{err}"
    );

    // Same rows, other order: still RED — only the canonical bytes pass.
    let mut rev: Vec<&str> = text.lines().collect();
    rev.reverse();
    std::fs::write(&out, rev.join("\n") + "\n").expect("plant");
    let err = run_export(Some(&reg), &out, true, true).expect_err("reordered must be RED");
    assert!(
        err.to_string().contains("same rows, different bytes"),
        "{err}"
    );

    // A write repairs it.
    run_export(Some(&reg), &out, false, true).expect("export");
    assert_eq!(std::fs::read_to_string(&out).expect("read"), text);
}

#[test]
fn ext_22_export_refuses_a_missing_registry_and_a_missing_file() {
    let (dir, reg) = home();
    let out = dir.path().join("pacha.jsonl");
    let absent = dir.path().join("absent.db");
    assert!(run_export(Some(&absent), &out, false, true).is_err());
    assert!(!absent.exists(), "export must not create a registry");
    assert!(
        run_export(Some(&reg), &out, true, true).is_err(),
        "--check with no file"
    );
}
