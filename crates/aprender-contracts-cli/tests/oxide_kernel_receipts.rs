//! OXIDE-001 O-2 (aprender#3522) — `ont:Kernel` and the kernel shapes, on the CLI (FALSIFY-KRC-002).
//!
//! Every case is the COMMITTED kernel tree — `contracts/kernel-receipt-v1.yaml`, `evidence/kernels/**` and the
//! pilot source the manifest names — copied into a tempdir with ONE edit planted. A tempdir carries no
//! `lint-baseline.json`, so every shape is armed there and a violation is a real Fail (exit 1), while on `main`
//! the same shapes are only reported.
//!
//! | edit | exit | shape that must say so |
//! |---|---|---|
//! | none | 0 | — (and the kernel node is counted) |
//! | a required host's receipt deleted | 1 | kernel-parity `missingReceipt` |
//! | one row's parity `pass` flipped | 1 | kernel-parity `parityPass` |
//! | one row measured on a dirty tree | 1 | kernel-parity `treeDirtyPaths` |
//! | one row's timing `pass` flipped | 1 | kernel-timing `timingPass` |
//! | a foreign GPU process on one row | 1 | kernel-timing `foreignGpuProcs` |
//! | `unsafe` planted in the device module's HELPER | 1 | kernel-safety `unsafeSite` |
//! | authoring ptx with no exemption | 1 | kernel-ptx-exemption `exemptionReceipt` |
//! | a foreign file under evidence/kernels | 2 | WrongCorpus, named |
//! | EXPECTED_KERNELS drifted | 2 | WrongCorpus, both numbers |
//!
//! Both directions: the untouched copy must PASS, so a build that rejects every kernel fails this file.

use std::path::{Path, PathBuf};
use std::process::Command;

const KERNEL_DIR: &str = "evidence/kernels/gdn_gated_rmsnorm";
const SOURCE: &str = "experiments/cuda-oxide/gated-rmsnorm/src/main.rs";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn copy_in(dst: &Path, rel: &str) {
    let to = dst.join(rel);
    std::fs::create_dir_all(to.parent().expect("parent")).expect("mkdir");
    std::fs::copy(repo_root().join(rel), &to).unwrap_or_else(|e| panic!("copy {rel}: {e}"));
}

/// The number of manifests the committed tree declares.
fn expected_kernels() -> usize {
    let text = std::fs::read_to_string(repo_root().join("evidence/kernels/EXPECTED_KERNELS"))
        .expect("read");
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .and_then(|l| l.parse().ok())
        .expect("a count")
}

/// The committed kernel tree, in a tempdir laid out like the repo: EVERY kernel directory and the source
/// each manifest names, so the copy agrees with EXPECTED_KERNELS. The planted edits all go to
/// `gdn_gated_rmsnorm`; the other kernels ride along untouched.
fn committed_copy() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    copy_in(d.path(), "contracts/kernel-receipt-v1.yaml");
    copy_in(d.path(), "evidence/kernels/EXPECTED_KERNELS");
    for k in std::fs::read_dir(repo_root().join("evidence/kernels")).expect("evidence/kernels") {
        let k = k.expect("entry").path();
        let manifest = k.join("kernel.json");
        if !manifest.is_file() {
            continue;
        }
        let rel = format!(
            "evidence/kernels/{}",
            k.file_name().expect("name").to_string_lossy()
        );
        for e in std::fs::read_dir(&k).expect("kernel dir") {
            let name = e.expect("entry").file_name();
            copy_in(d.path(), &format!("{rel}/{}", name.to_string_lossy()));
        }
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest).expect("read")).expect("json");
        copy_in(d.path(), m["source"].as_str().expect("source"));
    }
    d
}

fn edit_json(d: &Path, rel: &str, f: impl FnOnce(&mut serde_json::Value)) {
    let p = d.join(rel);
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&p).expect("read")).expect("json");
    f(&mut v);
    std::fs::write(&p, v.to_string()).expect("write");
}

fn yoga_row0(d: &Path, f: impl FnOnce(&mut serde_json::Value)) {
    edit_json(d, &format!("{KERNEL_DIR}/yoga.json"), |v| {
        f(&mut v["receipts"][0])
    });
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn all(&self) -> String {
        format!(
            "exit {}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }
    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.stdout).unwrap_or_default()
    }
}

fn shapes_on(d: &Path) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .args([
            "lint",
            d.join("contracts").to_str().expect("utf-8"),
            "--gate",
            "shapes",
            "--format",
            "json",
        ])
        .output()
        .expect("spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// One planted edit must fail exactly the named shape, naming the property.
fn assert_fails(d: &Path, shape: &str, property: &str) {
    let r = shapes_on(d);
    assert_eq!(r.code, 1, "{}", r.all());
    assert!(
        r.stdout.contains(shape),
        "{shape} must be named\n{}",
        r.all()
    );
    assert!(
        r.stdout.contains(property),
        "{property} must be named\n{}",
        r.all()
    );
}

#[test]
fn the_committed_kernel_tree_passes_every_armed_kernel_shape() {
    let d = committed_copy();
    let r = shapes_on(d.path());
    assert_eq!(r.code, 0, "{}", r.all());
    let v = r.json();
    assert_eq!(v["extra"]["violations"], 0, "{}", r.all());
    assert_eq!(
        v["extra"]["by_entity_type"]["kernel-receipt"],
        expected_kernels(),
        "{}",
        r.all()
    );
    assert_eq!(
        v["extra"]["pc_extract"]["kernel-receipt"],
        "fired",
        "{}",
        r.all()
    );
}

#[test]
fn a_required_host_with_no_receipt_fails_parity() {
    let d = committed_copy();
    std::fs::remove_file(d.path().join(KERNEL_DIR).join("yoga.json")).expect("rm");
    assert_fails(d.path(), "kernel-parity", "missingReceipt");
}

#[test]
fn a_failed_parity_row_fails_parity() {
    let d = committed_copy();
    yoga_row0(d.path(), |r| r["parity"]["pass"] = false.into());
    assert_fails(d.path(), "kernel-parity", "parityPass");
}

#[test]
fn a_row_measured_on_a_dirty_tree_fails_parity() {
    let d = committed_copy();
    yoga_row0(d.path(), |r| r["tree_dirty_paths"] = 3.into());
    assert_fails(d.path(), "kernel-parity", "treeDirtyPaths");
}

#[test]
fn a_lost_timing_row_fails_timing() {
    let d = committed_copy();
    yoga_row0(d.path(), |r| r["timing"]["pass"] = false.into());
    assert_fails(d.path(), "kernel-timing", "timingPass");
}

#[test]
fn a_row_measured_beside_a_foreign_gpu_process_fails_timing() {
    // The O-1 gx10 contamination: an unlocked `apr` process moved the hand-PTX time 4.1 → 7.8 µs.
    let d = committed_copy();
    yoga_row0(d.path(), |r| {
        r["foreign_gpu_procs"] = "4242, apr, 1328 MiB;".into()
    });
    assert_fails(d.path(), "kernel-timing", "foreignGpuProcs");
}

#[test]
fn unsafe_in_a_device_helper_fails_safety() {
    let d = committed_copy();
    let p = d.path().join(SOURCE);
    let src = std::fs::read_to_string(&p).expect("read source");
    let planted = src.replacen(
        "let v = input[base + k];",
        "let v = unsafe { *input.as_ptr().add(base + k) };",
        1,
    );
    assert_ne!(planted, src, "the plant must land in warp_sum_sq");
    std::fs::write(&p, planted).expect("write");
    assert_fails(d.path(), "kernel-safety", "unsafeSite");
}

#[test]
fn a_ptx_kernel_without_an_exemption_fails_and_with_one_passes() {
    let d = committed_copy();
    let manifest = format!("{KERNEL_DIR}/kernel.json");
    edit_json(d.path(), &manifest, |m| m["authoring"] = "ptx".into());
    assert_fails(d.path(), "kernel-ptx-exemption", "exemptionReceipt");

    // The receipt that would justify hand PTX: here, an existing host receipt stands in for it.
    edit_json(d.path(), &manifest, |m| {
        m["ptx_exemption"]["receipt"] = format!("{KERNEL_DIR}/yoga.json").into();
    });
    let r = shapes_on(d.path());
    assert_eq!(r.code, 0, "a resolving exemption passes\n{}", r.all());
}

#[test]
fn a_foreign_file_under_evidence_kernels_declines_naming_it() {
    let d = committed_copy();
    std::fs::write(d.path().join(KERNEL_DIR).join("notes.json"), "{}").expect("write");
    let r = shapes_on(d.path());
    assert_eq!(r.code, 2, "{}", r.all());
    assert!(r.all().contains("notes.json"), "{}", r.all());
}

#[test]
fn a_kernel_added_without_bumping_the_denominator_declines() {
    let d = committed_copy();
    std::fs::write(
        d.path().join("evidence/kernels/EXPECTED_KERNELS"),
        format!("{}\n", expected_kernels() + 1),
    )
    .expect("write");
    let r = shapes_on(d.path());
    assert_eq!(r.code, 2, "{}", r.all());
}

#[test]
fn the_committed_tree_agrees_with_its_own_denominator() {
    // Independent of the extractor: count the manifests on disk.
    let manifests = std::fs::read_dir(repo_root().join("evidence/kernels"))
        .expect("evidence/kernels")
        .filter_map(Result::ok)
        .filter(|e| e.path().join("kernel.json").is_file())
        .count();
    assert_eq!(manifests, expected_kernels());
}
