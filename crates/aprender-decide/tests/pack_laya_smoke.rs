//! `pack_laya` with no arguments runs its smoke path and exits 0 (#5138).
//!
//! The examples-nightly lane (`scripts/dogfood_examples.sh`) runs every workspace example
//! with no arguments. `pack_laya` used to print its usage line and exit 2 there, which read
//! as a failed example. This target runs the example the same way, twice, and requires
//! exit 0, the golden `laya_tiny` sha256 on the `SMOKE` line, an `inspect` line with no
//! eligibility field, and the same stdout bytes from both runs.
//!
//! This integration target runs in CI: the fragment
//! `ci/explicit-test-commands.d/803-aprender-decide-pack-laya-smoke.cmd` carries
//! `cargo test -p aprender-decide --test pack_laya_smoke`.

use std::path::Path;
use std::process::Command;

/// Runs `pack_laya` with no arguments through the same cargo that runs this test.
fn run_no_args() -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO"))
        .args([
            "run",
            "--quiet",
            "-p",
            "aprender-decide",
            "--example",
            "pack_laya",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo runs");
    let stdout = String::from_utf8(out.stdout).expect("stdout is UTF-8");
    assert!(
        out.status.success(),
        "pack_laya with no arguments exited {:?}\nstdout:\n{stdout}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.code(), stdout)
}

#[test]
fn pack_laya_no_args_is_a_deterministic_smoke() {
    let golden = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya_tiny.apr.sha256"),
    )
    .expect("golden sha");
    let golden = golden.split_whitespace().next().expect("golden hex");

    let (code, first) = run_no_args();
    assert_eq!(code, Some(0));
    let lines: Vec<&str> = first.lines().collect();
    assert_eq!(lines.len(), 2, "two lines expected:\n{first}");
    assert!(
        lines[0].starts_with("SMOKE pack-fixture laya_tiny ")
            && lines[0].contains(&format!("sha256={golden} "))
            && lines[0].contains("golden=match"),
        "{}",
        lines[0]
    );
    let inspect: serde_json::Value = serde_json::from_str(lines[1]).expect("inspect line is JSON");
    assert_eq!(inspect["artifact_sha256"], golden);
    assert!(
        inspect.get("deploy_eligible").is_none(),
        "inspect makes no eligibility claim: {}",
        lines[1]
    );
    assert!(!first.contains("REFUSED"), "{first}");

    let (_, second) = run_no_args();
    assert_eq!(
        first, second,
        "two no-argument runs printed different bytes"
    );
}
