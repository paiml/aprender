//! FALSIFY-DECIDE-APR-002/-003/-011: packing is byte-deterministic across processes
//! and serde_json map backings, and the manifest stores only Python probe values.
//!
//! The dual-backing proof is the verify, not one test: this module runs standalone
//! (`preserve_order=OFF`) and with the test-only `serde-preserve-order` feature
//! (`serde_json/preserve_order`, so `=ON`, selected explicitly rather than by feature
//! unification with another crate), and `golden_sha` must pass in both against the same
//! committed hash. `backing_canary` prints which backing each run compiled.

use super::tests::{fixture_probes, pack_tiny};
use super::{artifact_sha256_hex, inspect_manifest, PROBE_INPUTS, PROBE_TASK};
use crate::test_support::{fixture_apr, fixture_task, load_laya};
use crate::Task;
use std::path::PathBuf;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/laya_tiny.apr.sha256")
}

/// Which serde_json map backing this test binary compiled (the 08-04 canary).
#[test]
fn backing_canary() {
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(r#"{"b":1,"a":2}"#).expect("parse canary map");
    let insertion_ordered = map.keys().next().map(String::as_str) == Some("b");
    println!(
        "serde_json backing: preserve_order={}",
        if insertion_ordered { "ON" } else { "OFF" }
    );
}

#[test]
fn same_process_twice() {
    let a = pack_tiny();
    let b = pack_tiny();
    assert_eq!(a.len(), b.len());
    assert!(a == b, "two packs in one process differ");
}

/// The packed tiny fixture's sha256 equals the committed golden. With
/// `DECIDE_BLESS_GOLDEN=1` it writes the golden instead and FAILS the run, so a bless
/// can never pass silently.
#[test]
fn golden_sha() {
    let sha = artifact_sha256_hex(&pack_tiny());
    let path = golden_path();
    if std::env::var("DECIDE_BLESS_GOLDEN").as_deref() == Ok("1") {
        std::fs::write(&path, format!("{sha}\n")).expect("write the golden");
        panic!(
            "blessed {} = {sha} on ARCH={}; re-run without DECIDE_BLESS_GOLDEN",
            path.display(),
            std::env::consts::ARCH
        );
    }
    let golden = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "golden {} unreadable ({e}); bless with DECIDE_BLESS_GOLDEN=1",
            path.display()
        )
    });
    println!(
        "golden_sha: packed {sha}, golden {}, ARCH={}",
        golden.trim(),
        std::env::consts::ARCH
    );
    assert_eq!(sha, golden.trim(), "the packed tiny fixture moved");
}

/// The manifest's probe expectations are probes.json's hex strings byte for byte, and
/// wherever the Rust probe's bits differ from Python's, the stored bits are Python's.
#[test]
fn no_rust_floats_in_manifest() {
    let manifest = inspect_manifest(&pack_tiny()).expect("manifest");
    let python = fixture_probes();
    assert_eq!(manifest.probes, python, "stored probes == probes.json");

    let laya = load_laya(&fixture_apr(), fixture_task()).expect("fixture loads");
    let task = Task::from_slice(PROBE_TASK.as_bytes()).expect("probe task");
    let texts: Vec<String> = PROBE_INPUTS.iter().map(|s| (*s).to_string()).collect();
    let rust = laya.classify_for_task(&task, &texts).expect("rust probes");
    let (mut same, mut differ) = (0usize, 0usize);
    for (stored, r) in manifest.probes.iter().zip(&rust) {
        for (h, p) in stored.probabilities_f32_hex.iter().zip(&r.probabilities) {
            let rust_hex = format!("{:08x}", p.to_bits());
            if &rust_hex == h {
                same += 1;
            } else {
                differ += 1;
            }
        }
    }
    println!(
        "no_rust_floats_in_manifest: {same} probe values bit-equal to Rust, {differ} differ \
         (stored values are Python's in both cases), ARCH={}",
        std::env::consts::ARCH
    );
}
