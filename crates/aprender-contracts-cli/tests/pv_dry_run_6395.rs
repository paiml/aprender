//! `pv --dry-run` on generate / book / unlock (paiml/aprender#6395, contract pv-dry-run-preview-v1).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn pv(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pv"))
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("spawn pv")
}

fn contract() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/pv-query-diff-info-v1.yaml")
}

fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(base: &Path, d: &Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(d).expect("read_dir").flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else {
                out.push(p.strip_prefix(base).expect("prefix").display().to_string());
            }
        }
    }
    if dir.exists() {
        walk(dir, dir, &mut out);
    }
    out.sort();
    out
}

#[test]
fn generate_dry_run_leaves_dir_empty_and_names_every_real_path() {
    let c = contract();
    let cwd = tempfile::tempdir().expect("tmp");
    let dry = pv(
        &["generate", c.to_str().expect("ok"), "--dry-run"],
        cwd.path(),
    );
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert!(files_under(cwd.path()).is_empty(), "dry run wrote files");
    assert!(
        !cwd.path().join("generated").exists(),
        "dry run created output dir"
    );
    let plan = String::from_utf8_lossy(&dry.stdout).to_string();

    let real_cwd = tempfile::tempdir().expect("tmp");
    let real = pv(&["generate", c.to_str().expect("ok")], real_cwd.path());
    assert!(real.status.success());
    let written = files_under(&real_cwd.path().join("generated"));
    assert!(!written.is_empty());
    for f in &written {
        assert!(plan.contains(f.as_str()), "plan misses {f}:\n{plan}");
    }
    assert_eq!(
        plan.matches("would create").count(),
        written.len(),
        "plan count differs from real run:\n{plan}"
    );
}

#[test]
fn generate_dry_run_reports_overwrite_for_existing_files() {
    let c = contract();
    let cwd = tempfile::tempdir().expect("tmp");
    assert!(pv(&["generate", c.to_str().expect("ok")], cwd.path())
        .status
        .success());
    let before = files_under(cwd.path());
    let dry = pv(
        &["generate", c.to_str().expect("ok"), "--dry-run"],
        cwd.path(),
    );
    assert!(String::from_utf8_lossy(&dry.stdout).contains("would overwrite"));
    assert_eq!(before, files_under(cwd.path()));
}

#[test]
fn book_dry_run_writes_no_pages_and_names_them() {
    let cwd = tempfile::tempdir().expect("tmp");
    let src = cwd.path().join("contracts");
    std::fs::create_dir(&src).expect("ok");
    std::fs::copy(contract(), src.join("pv-query-diff-info-v1.yaml")).expect("ok");
    let dry = pv(&["book", "contracts", "--dry-run", "-o", "out"], cwd.path());
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert!(
        !cwd.path().join("out").exists(),
        "dry run created output dir"
    );
    assert!(String::from_utf8_lossy(&dry.stdout).contains("pv-query-diff-info-v1.md"));
}

#[test]
fn unlock_dry_run_keeps_bytes_and_shows_the_removed_line() {
    let cwd = tempfile::tempdir().expect("tmp");
    let p = cwd.path().join("c.yaml");
    let text = "metadata:\n  version: 1.0.0\n  locked_level: 3\n";
    std::fs::write(&p, text).expect("ok");
    let dry = pv(
        &[
            "unlock",
            p.to_str().expect("ok"),
            "--reason",
            "t",
            "--dry-run",
        ],
        cwd.path(),
    );
    assert!(
        dry.status.success(),
        "{}",
        String::from_utf8_lossy(&dry.stderr)
    );
    assert_eq!(std::fs::read_to_string(&p).expect("ok"), text);
    assert!(String::from_utf8_lossy(&dry.stdout).contains("-  locked_level: 3"));
}

fn tree_snapshot(dir: &Path) -> Vec<String> {
    files_under(dir)
}

#[cfg(unix)]
#[test]
fn generate_dry_run_creates_nothing_anywhere_even_with_a_read_only_tmpdir() {
    use std::os::unix::fs::PermissionsExt;
    let c = contract();
    let cwd = tempfile::tempdir().expect("tmp");
    let ro = tempfile::tempdir().expect("ro tmp");
    std::fs::set_permissions(ro.path(), std::fs::Permissions::from_mode(0o555)).expect("chmod");
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .args(["generate", c.to_str().expect("utf8"), "--dry-run"])
        .env("TMPDIR", ro.path())
        .current_dir(cwd.path())
        .output()
        .expect("spawn pv");
    let ro_left = tree_snapshot(ro.path());
    std::fs::set_permissions(ro.path(), std::fs::Permissions::from_mode(0o755)).expect("chmod");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(ro_left.is_empty(), "dry run touched TMPDIR: {ro_left:?}");
    assert!(tree_snapshot(cwd.path()).is_empty(), "dry run wrote to cwd");
    assert_eq!(
        std::fs::read_dir(ro.path()).expect("ls").count(),
        0,
        "dry run left an entry in TMPDIR"
    );
}

#[test]
fn book_dry_run_update_summary_errors_when_summary_is_missing() {
    let cwd = tempfile::tempdir().expect("tmp");
    let src = cwd.path().join("contracts");
    std::fs::create_dir(&src).expect("mkdir");
    std::fs::copy(contract(), src.join("pv-query-diff-info-v1.yaml")).expect("copy");
    let args = ["book", "contracts", "-o", "out", "--update-summary"];
    let real = pv(&args, cwd.path());
    let _ = std::fs::remove_dir_all(cwd.path().join("out"));
    let mut dry_args = args.to_vec();
    dry_args.push("--dry-run");
    let dry = pv(&dry_args, cwd.path());
    assert!(
        !real.status.success(),
        "real run should fail without SUMMARY.md"
    );
    assert!(!dry.status.success(), "dry run must fail like the real run");
    assert!(!cwd.path().join("out").exists());
}
