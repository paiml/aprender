//! `pv coverage` report rendering (#5621), driven through
//! the binary on a two-contract corpus.

#![allow(clippy::disallowed_methods)]

use std::path::Path;
use std::process::{Command, Output};

fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::copy(
        fx.join("relu-kernel-v1.yaml"),
        dir.path().join("relu-kernel-v1.yaml"),
    )
    .expect("copy relu");
    let dot = std::fs::read_to_string(fx.join("dot-kernel-v1.yaml")).expect("read dot");
    let dot = dot.replacen(
        "metadata:\n",
        "metadata:\n  depends_on: [\"relu-kernel-v1\"]\n",
        1,
    );
    std::fs::write(dir.path().join("dot-kernel-v1.yaml"), dot).expect("write dot");
    dir
}

fn pv(args: &[&str], dir: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pv"))
        .arg(args[0])
        .arg(dir)
        .args(&args[1..])
        .output()
        .expect("run pv")
}

fn stdout(o: &Output) -> String {
    assert!(
        o.status.success(),
        "pv failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn coverage_report_prints_per_contract_rows_totals_and_percentage() {
    let d = corpus();
    let out = stdout(&pv(&["coverage"], d.path()));
    assert!(
        out.starts_with("Obligation Coverage Report\n=========================="),
        "{out}"
    );
    assert!(
        out.contains("dot-kernel-v1") && out.contains("relu-kernel-v1"),
        "{out}"
    );
    assert!(out.contains("Contracts:            2"), "{out}");
    assert!(out.contains("Overall obligation coverage: "), "{out}");
    assert!(
        out.contains("Falsification tests:") && out.contains("Kani harnesses:"),
        "{out}"
    );
    assert!(
        !out.contains("Binding implemented"),
        "binding lines only with --binding: {out}"
    );
}

#[test]
fn coverage_with_binding_adds_binding_totals() {
    let d = corpus();
    let b = d.path().join("binding.yaml");
    std::fs::write(
        &b,
        "version: 1.0.0\ntarget_crate: x\nbindings:\n- contract: dot-kernel-v1.yaml\n  equation: dot\n  \
         module_path: x::d\n  function: d\n  signature: 'fn d()'\n  status: implemented\n",
    )
    .expect("binding");
    let out = stdout(&pv(
        &["coverage", "--binding", b.to_str().expect("utf8")],
        d.path(),
    ));
    assert!(out.contains("Binding implemented:  1"), "{out}");
    assert!(out.contains("Binding partial:      0"), "{out}");
    assert!(out.contains("impl=1/1"), "{out}");
}
