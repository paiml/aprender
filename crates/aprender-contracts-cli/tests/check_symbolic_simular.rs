//! `pv check-symbolic <dir>` over `contracts/simular/` (aprender#7305).
//!
//! The directory form walks every contract file, names what it skips, and is strict: a violated OR an
//! unparsed invariant anywhere in the directory exits non-zero. Each test runs the built `pv` binary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const FILES: [&str; 4] = [
    "f64-dense-solve-v1.yaml",
    "sim-determinism-core-v1.yaml",
    "simulation-determinism-v1.yaml",
    "simulation-step-v1.yaml",
];

fn simular_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/simular")
}

/// (exit code, stdout, stderr) of `pv check-symbolic <args>`.
fn check_symbolic(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .arg("check-symbolic")
        .args(args)
        .output()
        .expect("run pv");
    (
        out.status.code().expect("pv exit code"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A temp copy of `contracts/simular/` with `from` replaced by `to` in `file` (exactly one site).
fn planted(file: &str, from: &str, to: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("tempdir");
    for entry in fs::read_dir(simular_dir()).expect("read simular dir") {
        let p = entry.expect("dir entry").path();
        if !p.is_file() {
            continue; // pv lint leaves a `.pv/` cache directory beside the contracts
        }
        let name = p.file_name().expect("file name").to_owned();
        let mut text = fs::read_to_string(&p).expect("read contract");
        if name.to_str() == Some(file) {
            assert_eq!(text.matches(from).count(), 1, "{file}: site '{from}'");
            text = text.replacen(from, to, 1);
        }
        fs::write(tmp.path().join(name), text).expect("write copy");
    }
    tmp
}

fn json_report(dir: &Path) -> (i32, serde_json::Value) {
    let (code, out, err) = check_symbolic(&[dir.to_str().expect("utf-8 path"), "--json"]);
    let v = serde_json::from_str(&out).unwrap_or_else(|e| panic!("json: {e}\n{out}\n{err}"));
    (code, v)
}

#[test]
fn directory_is_walked_and_binding_is_named() {
    let dir = simular_dir();
    let (code, out, err) = check_symbolic(&[dir.to_str().expect("utf-8 path")]);
    for f in FILES {
        assert!(out.contains(f), "{f} not walked\n{out}\n{err}");
    }
    let skip = out
        .lines()
        .find(|l| l.contains("binding.yaml"))
        .unwrap_or_else(|| panic!("binding.yaml not named\n{out}\n{err}"));
    assert!(skip.contains("skipped"), "{skip}");
    assert_eq!(code, 0, "{out}\n{err}");
}

#[test]
fn directory_has_zero_unparsed_and_zero_violated() {
    let (code, v) = json_report(&simular_dir());
    assert_eq!(v["files"].as_array().map(Vec::len), Some(4), "{v}");
    assert_eq!(v["skipped"].as_array().map(Vec::len), Some(1), "{v}");
    assert_eq!(v["unparsed"], 0, "{v}");
    assert_eq!(v["violated"], 0, "{v}");
    // Zero unparsed must not be reached by emptying the invariants: at least the 28 minus the non-numeric ones.
    assert!(v["holds"].as_u64() >= Some(20), "{v}");
    assert_eq!(code, 0);
}

#[test]
fn planted_wrong_formula_is_violated_with_counterexample() {
    for (file, from, to) in [
        (
            "simulation-step-v1.yaml",
            "step_forward(pos) = pos + 1",
            "step_forward(pos) = pos + 2",
        ),
        (
            "simulation-determinism-v1.yaml",
            "energy(x, v) = 0.5 * (v^2 + w^2 * x^2)",
            "energy(x, v) = 0.5 * (v^2 - w^2 * x^2)",
        ),
        (
            "f64-dense-solve-v1.yaml",
            "l21(a, b) = b / l11(a)",
            "l21(a, b) = b / a",
        ),
        (
            "sim-determinism-core-v1.yaml",
            "t(k) = k * dt_ns",
            "t(k) = k + dt_ns",
        ),
    ] {
        let tmp = planted(file, from, to);
        let (code, out, err) = check_symbolic(&[tmp.path().to_str().expect("utf-8 path")]);
        assert_ne!(code, 0, "{file}: planted formula passed\n{out}");
        let i = out
            .find("VIOLATED")
            .unwrap_or_else(|| panic!("{file}: no VIOLATED\n{out}"));
        assert!(
            out[i..].contains(" at "),
            "{file}: no counterexample\n{out}"
        );
        assert!(err.contains("violated"), "{file}: {err}");
    }
}

#[test]
fn planted_unparsable_invariant_is_unparsed_and_fails_the_directory() {
    let tmp = planted(
        "simulation-step-v1.yaml",
        "'time(n+1) > time(n)'",
        "'Time strictly increases per step'",
    );
    let (code, v) = json_report(tmp.path());
    assert_eq!(v["unparsed"], 1, "{v}");
    assert_eq!(v["violated"], 0, "{v}");
    assert_ne!(code, 0, "unparsed is never a pass");
    let (code, out, err) = check_symbolic(&[tmp.path().to_str().expect("utf-8 path")]);
    assert_ne!(code, 0);
    assert!(
        out.contains("unparsed  [step_monotonicity] Time strictly increases"),
        "{out}"
    );
    assert!(err.contains("unparsed"), "{err}");
}

#[test]
fn unreadable_contract_and_empty_directory_are_not_a_pass() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = tmp.path().to_str().expect("utf-8 path");
    let (code, _, err) = check_symbolic(&[dir]);
    assert_ne!(code, 0, "an empty directory checked nothing");
    assert!(err.contains("no contract file"), "{err}");
    fs::write(
        tmp.path().join("broken-v1.yaml"),
        "metadata: [not, a, contract\n",
    )
    .expect("write");
    let (code, _, err) = check_symbolic(&[dir]);
    assert_ne!(code, 0);
    assert!(err.contains("broken-v1.yaml"), "{err}");
}

#[test]
fn single_file_form_is_unchanged() {
    let file = simular_dir().join("simulation-step-v1.yaml");
    let (code, out, _) = check_symbolic(&[file.to_str().expect("utf-8 path")]);
    assert_eq!(code, 0, "{out}");
    assert!(out.trim_end().ends_with("unparsed 0"), "{out}");
}
