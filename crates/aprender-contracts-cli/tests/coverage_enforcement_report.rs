//! `pv coverage --enforcement`: the E0/E1/E2 enforcement-quality report.
//!
//! `print_enforcement_report` is private and prints to stdout, so each case runs the
//! `pv` binary against a throwaway crate and asserts on the exact report lines.
//!
//! Refs: #5721

use std::path::Path;
use std::process::{Command, Output};

const BINDING: &str = "version: 1.0.0
target_crate: fixture
bindings:
- contract: relu-kernel-v1.yaml
  equation: relu
  status: implemented
- contract: relu-kernel-v1.yaml
  equation: relu_two
  status: implemented
";

/// `alpha` has a domain pre-check and a post macro (E2), `beta` a pre-check only (E1),
/// `gamma` no definition at all (E0).
const GENERATED: &str = "macro_rules! contract_pre_alpha {
    ($x:expr) => { assert!($x.is_finite()) };
}
macro_rules! contract_post_alpha {
    ($x:expr) => {};
}
macro_rules! contract_pre_beta {
    ($x:expr) => { assert!($x.len() > 0) };
}
// contract_pre_decoy! lives in generated_contracts.rs and must not be counted.
";

const LIB: &str = "fn a() { contract_pre_alpha!(x); }
fn b() { contract_pre_beta!(x); }
fn c() { contract_pre_gamma!(x); }
";

/// Lay out `<tmp>/{contracts,crate/src,binding.yaml}`; returns the `pv coverage` output.
fn run_pv(src_files: &[(&str, &str)], with_src: bool, with_binding_flag: bool) -> Output {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    std::fs::create_dir_all(root.join("contracts")).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/relu-kernel-v1.yaml"),
        root.join("contracts/relu-kernel-v1.yaml"),
    )
    .unwrap();
    std::fs::write(root.join("binding.yaml"), BINDING).unwrap();
    std::fs::create_dir_all(root.join("crate")).unwrap();
    if with_src {
        std::fs::create_dir_all(root.join("crate/src")).unwrap();
        for (name, body) in src_files {
            std::fs::write(root.join("crate/src").join(name), body).unwrap();
        }
    }
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_pv"));
    cmd.arg("coverage").arg(root.join("contracts"));
    if with_binding_flag {
        cmd.arg("--binding").arg(root.join("binding.yaml"));
    }
    cmd.arg("--enforcement").arg(root.join("crate"));
    cmd.output().expect("run pv")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn report_counts_levels_and_scores_them() {
    let o = run_pv(
        &[("lib.rs", LIB), ("generated_contracts.rs", GENERATED)],
        true,
        true,
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let out = stdout(&o);
    for line in [
        "Enforcement Quality Report",
        "  Bindings declared:    2",
        "  Call sites found:     3",
        "  Penetration:          150.0% (3/2)",
        "  E0 (generic !is_empty):  1",
        "  E1 (domain pre-checks):  1",
        "  E2 (pre + post checks):  1",
        "  Quality score:           0.53 (E0=0.1, E1=0.5, E2=1.0)",
        "  Enforcement score:       0.8000 (penetration × quality)",
    ] {
        assert!(out.contains(line), "missing {line:?} in:\n{out}");
    }
}

#[test]
fn call_site_lines_carry_level_line_number_and_macro() {
    let o = run_pv(
        &[("lib.rs", LIB), ("generated_contracts.rs", GENERATED)],
        true,
        true,
    );
    let out = stdout(&o);
    assert!(out.contains("  Call sites:"), "{out}");
    assert!(out.contains("lib.rs:1 — contract_pre_alpha"), "{out}");
    assert!(out.contains("[E2] "), "{out}");
    assert!(out.contains("lib.rs:2 — contract_pre_beta"), "{out}");
    assert!(out.contains("[E1] "), "{out}");
    assert!(out.contains("lib.rs:3 — contract_pre_gamma"), "{out}");
    assert!(out.contains("[E0] "), "{out}");
    assert!(
        !out.contains("contract_pre_decoy"),
        "generated_contracts.rs must be skipped:\n{out}"
    );
}

#[test]
fn a_crate_without_call_sites_scores_zero_and_prints_no_site_list() {
    let o = run_pv(&[("lib.rs", "fn nothing() {}\n")], true, true);
    let out = stdout(&o);
    assert!(out.contains("  Call sites found:     0"), "{out}");
    assert!(out.contains("  Penetration:          0.0% (0/2)"), "{out}");
    assert!(out.contains("  Quality score:           0.00"), "{out}");
    assert!(out.contains("  Enforcement score:       0.0000"), "{out}");
    assert!(!out.contains("Call sites:\n"), "{out}");
}

#[test]
fn a_crate_without_a_src_dir_warns_and_prints_no_report() {
    let o = run_pv(&[], false, true);
    let out = stdout(&o);
    assert!(!out.contains("Enforcement Quality Report"), "{out}");
    assert!(String::from_utf8_lossy(&o.stderr).contains("/src not found"));
}

#[test]
fn enforcement_without_binding_is_refused() {
    let o = run_pv(&[("lib.rs", LIB)], true, false);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--enforcement requires --binding"));
}
