//! #5690: behaviour of `run_certification`, driven through `apr-qa certify`.
//!
//! Each case runs in a scratch directory holding its own `docs/certifications/models.csv`,
//! because the function reads that path relative to the working directory.

use std::path::{Path, PathBuf};
use std::process::Command;

const QWEN: &str = "Qwen/Qwen2.5-Coder-1.5B-Instruct";

const CSV: &str = "model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4\n\
Qwen/Qwen2.5-Coder-1.5B-Instruct,qwen2,1.5B,small,PENDING,0,-,none,,false,false,false,false\n\
meta-llama/Llama-3.2-1B-Instruct,llama,1B,small,PENDING,0,-,none,,false,false,false,false\n";

fn scratch(tag: &str, csv: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-qa-cert-5690-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("docs/certifications")).expect("mkdir");
    if let Some(body) = csv {
        std::fs::write(dir.join("docs/certifications/models.csv"), body).expect("write csv");
    }
    dir
}

/// (exit code, stdout, stderr) with ANSI colour stripped.
fn certify(dir: &Path, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_apr-qa"))
        .current_dir(dir)
        .arg("certify")
        .args(args)
        .output()
        .expect("spawn apr-qa");
    let strip = |b: &[u8]| {
        let s = String::from_utf8_lossy(b).into_owned();
        let mut o = String::new();
        let mut it = s.chars();
        while let Some(c) = it.next() {
            if c == '\u{1b}' {
                for d in it.by_ref() {
                    if d == 'm' {
                        break;
                    }
                }
            } else {
                o.push(c);
            }
        }
        o
    };
    (
        out.status.code().unwrap_or(-1),
        strip(&out.stdout),
        strip(&out.stderr),
    )
}

#[test]
fn dry_run_all_lists_every_model_with_its_tiered_playbook() {
    let d = scratch("all", Some(CSV));
    let (rc, out, _) = certify(&d, &["--all", "--dry-run"]);
    assert_eq!(rc, 0, "{out}");
    assert!(out.contains("Models to certify: 2"), "{out}");
    assert!(out.contains(&format!("Would certify: {QWEN}")), "{out}");
    assert!(
        out.contains("playbooks/models/qwen2.5-coder-1.5b-quick.playbook.yaml"),
        "{out}"
    );
    assert!(
        out.contains("playbooks/models/llama-3.2-1b-quick.playbook.yaml"),
        "{out}"
    );
}

#[test]
fn dry_run_family_filter_and_tier_select_one_model_and_suffix() {
    let d = scratch("fam", Some(CSV));
    let (rc, out, _) = certify(&d, &["--family", "llama", "--dry-run", "--tier", "mvp"]);
    assert_eq!(rc, 0, "{out}");
    assert!(out.contains("Tier: mvp"), "{out}");
    assert!(out.contains("Models to certify: 1"), "{out}");
    assert!(out.contains("llama-3.2-1b-mvp.playbook.yaml"), "{out}");
    assert!(!out.contains(QWEN), "{out}");
}

#[test]
fn dry_run_header_reports_fail_fast_and_model_cache() {
    let d = scratch("hdr", Some(CSV));
    let (rc, out, err) = certify(
        &d,
        &[QWEN, "--dry-run", "--fail-fast", "--model-cache", "/x/y"],
    );
    assert_eq!(rc, 0, "{out}{err}");
    assert!(out.contains("Fail-fast: true"), "{out}");
    assert!(out.contains("Model cache: /x/y"), "{out}");
    assert!(
        err.contains("[ENVIRONMENT] === Diagnostic Context ==="),
        "{err}"
    );
}

#[test]
fn unknown_family_exits_1_naming_the_family() {
    let d = scratch("nofam", Some(CSV));
    let (rc, out, err) = certify(&d, &["--family", "nosuch", "--dry-run"]);
    assert_eq!(rc, 1, "{out}{err}");
    assert!(out.contains("Models to certify: 0"), "{out}");
    assert!(err.contains("No models matched the given filter."), "{err}");
    assert!(
        err.contains("Family 'nosuch' not found in models.csv."),
        "{err}"
    );
}

#[test]
fn unknown_tier_exits_1_before_printing_the_header() {
    let d = scratch("tier", Some(CSV));
    let (rc, out, err) = certify(&d, &["--all", "--dry-run", "--tier", "bogus"]);
    assert_eq!(rc, 1, "{out}{err}");
    assert!(err.contains("Unknown tier: bogus"), "{err}");
    assert!(!out.contains("APR Model Certification"), "{out}");
}

#[test]
fn unknown_kernel_class_exits_1() {
    let d = scratch("kc", Some(CSV));
    let (rc, _, err) = certify(&d, &["--all", "--dry-run", "--kernel-class", "bogus"]);
    assert_eq!(rc, 1, "{err}");
    assert!(err.contains("Unknown kernel class: bogus"), "{err}");
}

#[test]
fn missing_models_csv_exits_1() {
    let d = scratch("nocsv", None);
    let (rc, _, err) = certify(&d, &["--all", "--dry-run"]);
    assert_eq!(rc, 1, "{err}");
    assert!(err.contains("Error reading models.csv"), "{err}");
}

#[test]
fn malformed_models_csv_exits_1() {
    let d = scratch("badcsv", Some("model_id,family\nx,y\n"));
    let (rc, _, err) = certify(&d, &["--all", "--dry-run"]);
    assert_eq!(rc, 1, "{err}");
    assert!(err.contains("Error parsing models.csv"), "{err}");
}

#[test]
fn unwritable_output_dir_exits_1_after_listing_models() {
    let d = scratch("out", Some(CSV));
    std::fs::write(d.join("blocker"), b"").expect("blocker file");
    let (rc, out, err) = certify(&d, &["--all", "--output", "blocker/sub"]);
    assert_eq!(rc, 1, "{out}{err}");
    assert!(out.contains("Models to certify: 2"), "{out}");
    assert!(err.contains("Error creating output directory"), "{err}");
}
