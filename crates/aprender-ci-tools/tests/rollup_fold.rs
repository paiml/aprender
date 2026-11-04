//! aprender#6313: `aprender-rollup fold <base.yaml> <entries-dir>` prints the aggregate on stdout, rc 0,
//! and rc 2 (not measured) when it cannot fold. The committed roadmap IS the fold of its base.
use std::path::PathBuf;
use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_aprender-rollup"))
}

fn roadmaps() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/roadmaps")
}

#[test]
fn fold_of_the_committed_base_is_the_committed_roadmap_byte_for_byte() {
    let rm = roadmaps();
    let o = bin()
        .arg("fold")
        .arg(rm.join("roadmap-base.yaml"))
        .arg(rm.join("entries"))
        .output()
        .expect("spawn");
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let want = std::fs::read(rm.join("roadmap.yaml")).expect("roadmap.yaml");
    assert!(
        o.stdout == want,
        "fold(roadmap-base.yaml, entries/) != roadmap.yaml"
    );
}

#[test]
fn a_missing_base_is_not_measured_rc_2_with_empty_stdout() {
    let o = bin()
        .args(["fold", "/nonexistent/base.yaml"])
        .arg(roadmaps().join("entries"))
        .output()
        .expect("spawn");
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(String::from_utf8_lossy(&o.stderr).contains("not_measured"));
}

#[test]
fn a_wrong_argument_count_is_not_measured_rc_2() {
    let o = bin().args(["fold", "only-one"]).output().expect("spawn");
    assert_eq!(o.status.code(), Some(2));
}
