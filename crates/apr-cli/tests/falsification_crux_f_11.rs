//! CRUX-F-11 — end-to-end falsification harness for `apr check-finite-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-11-{002,003} gate the
//! classifier discharges has a matching captured JSON body that the binary
//! must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_json(body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-11-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(serde_json::to_vec_pretty(body).expect("serialize").as_slice())
        .expect("write");
    f.flush().expect("flush");
    f
}

fn good_error_json() -> serde_json::Value {
    json!({
        "error": "non_finite",
        "layer": "blk.0.ffn_up",
        "shape": [1, 4096],
        "first_bad_index": 0,
        "value": "nan",
        "op": "ffn_up"
    })
}

fn good_layer_list(n_blocks: usize) -> serde_json::Value {
    let mut layers: Vec<serde_json::Value> = Vec::new();
    for b in 0..n_blocks {
        for op in &[
            "attention_q",
            "attention_k",
            "attention_v",
            "attention_out",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
            "layernorm_in",
            "layernorm_post",
        ] {
            layers.push(json!({"name": format!("blk.{b}.{op}"), "shape": [1, 4096]}));
        }
    }
    layers.push(json!({"name": "embed_tokens"}));
    layers.push(json!({"name": "output_norm"}));
    layers.push(json!({"name": "lm_head"}));
    json!({"layers": layers})
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_f_11_cli_help_advertises_flags() {
    let out = apr_binary().args(["check-finite-lint", "--help"]).output().expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--error-file"), "--help must advertise --error-file; got:\n{stdout}");
    assert!(stdout.contains("--list-file"), "--help must advertise --list-file; got:\n{stdout}");
    assert!(stdout.contains("--min-layers"), "--help must advertise --min-layers; got:\n{stdout}");
}

#[test]
fn falsify_crux_f_11_cli_requires_at_least_one_file() {
    let out = apr_binary().args(["check-finite-lint"]).output().expect("run");
    assert!(!out.status.success(), "bare invocation must not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("at least one of --error-file or --list-file"),
        "stderr must explain the requirement; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_11_cli_missing_error_file_fails() {
    let out = apr_binary()
        .args(["check-finite-lint", "--error-file", "/nonexistent/crux-f-11-missing.json"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing error-file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_11_002_error_json_ok_on_well_formed() {
    let f = write_json(&good_error_json());
    let out = apr_binary()
        .args(["check-finite-lint", "--error-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good error JSON must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_11_002_error_json_rejects_missing_layer() {
    let mut body = good_error_json();
    body.as_object_mut().expect("obj").remove("layer");
    let f = write_json(&body);
    let out = apr_binary()
        .args(["check-finite-lint", "--error-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing layer key must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingKey") && stderr.contains("layer"),
        "stderr must name MissingKey(layer); got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_11_002_error_json_rejects_wrong_error_tag() {
    let mut body = good_error_json();
    body["error"] = json!("weird_other");
    let f = write_json(&body);
    let out = apr_binary()
        .args(["check-finite-lint", "--error-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "wrong error tag must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ErrorTagWrong"), "stderr must name ErrorTagWrong; got:\n{stderr}");
}

#[test]
fn falsify_crux_f_11_002_error_json_rejects_value_out_of_set() {
    let mut body = good_error_json();
    body["value"] = json!("banana");
    let f = write_json(&body);
    let out = apr_binary()
        .args(["check-finite-lint", "--error-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "value outside set must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ValueOutOfSet"), "stderr must name ValueOutOfSet; got:\n{stderr}");
}

#[test]
fn falsify_crux_f_11_003_layer_coverage_ok_on_well_formed() {
    let f = write_json(&good_layer_list(28));
    let out = apr_binary()
        .args(["check-finite-lint", "--list-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good layer list must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_11_003_layer_coverage_rejects_too_few() {
    let f = write_json(&good_layer_list(2));
    let out = apr_binary()
        .args(["check-finite-lint", "--list-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "too-few-layers must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("TooFewLayers"), "stderr must name TooFewLayers; got:\n{stderr}");
}

#[test]
fn falsify_crux_f_11_003_layer_coverage_rejects_missing_op_prefixes() {
    // 100 attention-only layers; FFN/LN absent.
    let mut layers: Vec<serde_json::Value> = Vec::new();
    for b in 0..20 {
        for op in &["attention_q", "attention_k", "attention_v", "attention_out"] {
            layers.push(json!({"name": format!("blk.{b}.{op}")}));
        }
        for k in 0..5 {
            layers.push(json!({"name": format!("blk.{b}.extra_{k}")}));
        }
    }
    let f = write_json(&json!({"layers": layers}));
    let out = apr_binary()
        .args(["check-finite-lint", "--list-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing op prefixes must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingOpPrefixes") && stderr.contains("ffn_gate"),
        "stderr must name MissingOpPrefixes with ffn_gate; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_11_min_layers_threshold_is_configurable() {
    // Set --min-layers 10; the 2-block list (21 layers) should now pass on count.
    let f = write_json(&good_layer_list(2));
    let out = apr_binary()
        .args(["check-finite-lint", "--list-file"])
        .arg(f.path())
        .args(["--min-layers", "10"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "10-layer floor must accept 21-layer list; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_11_json_output_contains_outcomes() {
    let ef = write_json(&good_error_json());
    let lf = write_json(&good_layer_list(28));
    let out = apr_binary()
        .args(["--json", "check-finite-lint", "--error-file"])
        .arg(ef.path())
        .arg("--list-file")
        .arg(lf.path())
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good bodies must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["error_json"].as_str().expect("error_json").contains("Ok"));
    assert!(parsed["layer_coverage"].as_str().expect("layer_coverage").contains("Ok"));
}
