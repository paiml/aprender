//! CRUX-F-18 — end-to-end falsification harness for `apr embed-viz-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-18-{001,003} gate
//! the classifier discharges has a matching captured CSV body that the
//! binary must classify exactly as the harness expects.

use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_csv(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-18-")
        .suffix(".csv")
        .tempfile()
        .expect("tempfile");
    f.write_all(body.as_bytes()).expect("write");
    f.flush().expect("flush");
    f
}

fn good_body() -> &'static str {
    "token_id,token_str,x,y\n0,<pad>,0.1,0.2\n1,<unk>,-0.5,1.5\n2,hello,3.14,-2.71\n"
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_f_18_cli_help_advertises_flags() {
    let out = apr_binary().args(["embed-viz-lint", "--help"]).output().expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    for flag in ["--csv-file", "--expected-vocab-size", "--csv-file-b"] {
        assert!(stdout.contains(flag), "--help must advertise {flag}; got:\n{stdout}");
    }
}

#[test]
fn falsify_crux_f_18_cli_missing_file_fails() {
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file", "/nonexistent/crux-f-18-missing.csv"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_18_001_schema_ok_on_good_body() {
    let f = write_csv(good_body());
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good body must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_18_001_schema_rejects_missing_column() {
    let body = "token_id,token_str,x\n0,<pad>,0.1\n";
    let f = write_csv(body);
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing y column must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingColumn") || stderr.contains("MissingHeader"),
        "stderr must name MissingColumn/Header; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_18_001_schema_rejects_nonfinite_coord() {
    let body = "token_id,token_str,x,y\n0,<pad>,nan,0.2\n";
    let f = write_csv(body);
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "nan coord must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("CoordNotFiniteFloat"),
        "stderr must name CoordNotFiniteFloat; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_18_001_row_count_ok_when_matches_vocab() {
    let f = write_csv(good_body());
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f.path())
        .args(["--expected-vocab-size", "3"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "row-count match must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_18_001_row_count_rejects_mismatch() {
    let f = write_csv(good_body());
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f.path())
        .args(["--expected-vocab-size", "100"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "row-count mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Mismatch"),
        "stderr must name Mismatch; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_18_003_determinism_ok_on_byte_identical() {
    let f1 = write_csv(good_body());
    let f2 = write_csv(good_body());
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f1.path())
        .arg("--csv-file-b")
        .arg(f2.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "byte-identical CSVs must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_18_003_determinism_rejects_diverging_csvs() {
    let f1 = write_csv(good_body());
    let body2 = "token_id,token_str,x,y\n0,<pad>,0.1,0.2\n1,<unk>,-0.5,1.5\n2,hello,3.99,-2.71\n";
    let f2 = write_csv(body2);
    let out = apr_binary()
        .args(["embed-viz-lint", "--csv-file"])
        .arg(f1.path())
        .arg("--csv-file-b")
        .arg(f2.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "diverging CSVs must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FirstDiffAtByte") || stderr.contains("LengthDiffers"),
        "stderr must name FirstDiffAtByte or LengthDiffers; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_18_json_output_contains_outcomes() {
    let f1 = write_csv(good_body());
    let f2 = write_csv(good_body());
    let out = apr_binary()
        .args(["--json", "embed-viz-lint", "--csv-file"])
        .arg(f1.path())
        .args(["--expected-vocab-size", "3"])
        .arg("--csv-file-b")
        .arg(f2.path())
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good bodies must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["schema"].as_str().expect("schema").contains("Ok"));
    assert!(parsed["row_count"].as_str().expect("row_count").contains("Ok"));
    assert!(parsed["determinism"].as_str().expect("determinism").contains("Ok"));
}
