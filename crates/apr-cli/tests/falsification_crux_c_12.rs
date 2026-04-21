//! E2E falsification tests for `apr llava-lint` (CRUX-C-12).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #978: exercise the CLI surface
//! end-to-end on captured JSON observations and assert the classifier
//! verdicts + non-zero exit codes on known-bad input.

use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_tmp_json(name: &str, body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix(name)
        .suffix(".json")
        .tempfile()
        .expect("create tempfile");
    f.write_all(body.as_bytes()).expect("write tempfile");
    f.flush().expect("flush tempfile");
    f
}

// ---- help surface (g2 proof) ----------------------------------------------

#[test]
fn falsify_crux_c_12_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["llava-lint", "--help"])
        .output()
        .expect("run apr llava-lint --help");
    assert!(out.status.success(), "apr llava-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_12_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("llava-lint")
        .output()
        .expect("run apr llava-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr llava-lint` must exit non-zero"
    );
}

// ---- image_tokens gate (FALSIFY-CRUX-C-12-001) ----------------------------

#[test]
fn falsify_crux_c_12_001_image_tokens_ok_on_llava15() {
    let tmp = write_tmp_json(
        "llava-imgtok-ok",
        r#"{ "image_tokens": { "arch": "llava15", "got": 576 } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(
        out.status.success(),
        "LLaVA-1.5 576 must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_12_001_image_tokens_rejects_mismatch() {
    let tmp = write_tmp_json(
        "llava-imgtok-mm",
        r#"{ "image_tokens": { "arch": "llava15", "got": 729 } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success(), "LLaVA-1.5 with SigLIP count must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-C-12-001"),
        "stderr must cite FALSIFY-CRUX-C-12-001; got:\n{stderr}"
    );
}

// ---- caption parity gate (FALSIFY-CRUX-C-12-002) --------------------------

#[test]
fn falsify_crux_c_12_002_caption_ok_on_byte_identical() {
    let tmp = write_tmp_json(
        "llava-capt-ok",
        r#"{ "caption": { "apr": "a cat", "golden": "a cat" } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_12_002_caption_rejects_byte_divergence() {
    let tmp = write_tmp_json(
        "llava-capt-div",
        r#"{ "caption": { "apr": "a cat", "golden": "a bat" } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-12-002"));
}

// ---- mmproj compatibility gate (FALSIFY-CRUX-C-12-003) --------------------

#[test]
fn falsify_crux_c_12_003_mmproj_ok_on_clip_matching_dim() {
    let tmp = write_tmp_json(
        "llava-mmproj-ok",
        r#"{ "mmproj": { "arch": "clip", "projection_dim": 4096, "hidden_size": 4096 } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_12_003_mmproj_rejects_dim_mismatch() {
    let tmp = write_tmp_json(
        "llava-mmproj-mm",
        r#"{ "mmproj": { "arch": "clip", "projection_dim": 1024, "hidden_size": 4096 } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-12-003"));
}

#[test]
fn falsify_crux_c_12_003_mmproj_rejects_unknown_arch() {
    let tmp = write_tmp_json(
        "llava-mmproj-badarch",
        r#"{ "mmproj": { "arch": "dinov2", "projection_dim": 4096, "hidden_size": 4096 } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-12-003"));
}

// ---- image format gate (FALSIFY-CRUX-C-12-004) ----------------------------

#[test]
fn falsify_crux_c_12_004_image_format_ok_on_png() {
    let tmp = write_tmp_json(
        "llava-fmt-ok",
        r#"{ "image_format": { "filename": "/tmp/photo.png" } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_12_004_image_format_rejects_unsupported_extension() {
    let tmp = write_tmp_json(
        "llava-fmt-mp4",
        r#"{ "image_format": { "filename": "clip.mp4" } }"#,
    );
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-12-004"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_12_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("llava-empty", "");
    let out = apr_binary()
        .args(["llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_12_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "llava-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr llava-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_12_json_output_shape() {
    let tmp = write_tmp_json(
        "llava-json-shape",
        r#"{
          "image_tokens": { "arch": "siglip", "got": 729 },
          "caption":      { "apr": "hi", "golden": "hi" },
          "mmproj":       { "arch": "clip", "projection_dim": 4096, "hidden_size": 4096 },
          "image_format": { "filename": "a.jpg" }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "llava-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr llava-lint --json");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"image_tokens\""),
        "--json must emit image_tokens key"
    );
    assert!(stdout.contains("\"caption\""));
    assert!(stdout.contains("\"mmproj\""));
    assert!(stdout.contains("\"image_format\""));
}
