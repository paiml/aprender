//! CRUX-B-02 — end-to-end falsification harness for `apr gguf-safetensors-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-B-02-{001,003,004}
//! gate the classifier discharges has a matching e2e JSON observation
//! that the binary must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_obs(json_body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-b-02-obs-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(serde_json::to_vec_pretty(json_body).unwrap().as_slice())
        .expect("write obs");
    f.flush().expect("flush");
    f
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_b_02_cli_help_advertises_observation_file() {
    let out = apr_binary()
        .args(["gguf-safetensors-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_b_02_cli_bare_invocation_is_usage_error() {
    let out = apr_binary()
        .arg("gguf-safetensors-lint")
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "bare invocation must not exit 0 — missing required --observation-file"
    );
}

#[test]
fn falsify_crux_b_02_cli_missing_file_fails_with_stamp() {
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            "/nonexistent/crux-b-02-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "nonexistent file must not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-B-02") || stderr.contains("not found"),
        "stderr must stamp FALSIFY-CRUX-B-02; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_b_02_cli_empty_file_fails() {
    let tmp = tempfile::Builder::new()
        .prefix("crux-b-02-empty-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "empty observation must not exit 0");
}

#[test]
fn falsify_crux_b_02_cli_invalid_json_fails() {
    let mut tmp = tempfile::Builder::new()
        .prefix("crux-b-02-bad-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    tmp.write_all(b"{not json").unwrap();
    tmp.flush().unwrap();
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "invalid JSON must not exit 0");
}

#[test]
fn falsify_crux_b_02_cli_no_gates_present_fails() {
    let obs = json!({ "unrelated": "field" });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "no known gates must not exit 0");
}

// ===== FALSIFY-CRUX-B-02-001 layout =====

#[test]
fn falsify_crux_b_02_001_layout_complete_passes() {
    let obs = json!({
        "layout": {
            "listing": ["model.safetensors", "config.json", "tokenizer.json", "generation_config.json"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "complete layout must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] layout"), "expected layout PASS");
}

#[test]
fn falsify_crux_b_02_001_layout_missing_safetensors_fails() {
    let obs = json!({
        "layout": {
            "listing": ["config.json", "tokenizer.json"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing safetensors must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-B-02-001"),
        "stderr must stamp FALSIFY-CRUX-B-02-001; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_b_02_001_layout_empty_listing_fails_all() {
    let obs = json!({ "layout": { "listing": [] } });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "empty listing must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-02-001"));
}

// ===== FALSIFY-CRUX-B-02-003 metadata =====

#[test]
fn falsify_crux_b_02_003_metadata_full_passes() {
    let obs = json!({
        "metadata": {
            "kv": {
                "general.architecture":       { "str": "llama" },
                "llama.embedding_length":     { "u32": 4096 },
                "llama.block_count":          { "u32": 32 },
                "llama.attention.head_count": { "u32": 32 }
            },
            "expected_outcome": "ok"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "full metadata must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_02_003_metadata_missing_architecture_expected_missing_key_passes() {
    // Absence-of-key IS expected — classifier reports missing_key, observer
    // pre-declared missing_key, gate PASSES.
    let obs = json!({
        "metadata": {
            "kv": {
                "llama.embedding_length":     { "u32": 4096 },
                "llama.block_count":          { "u32": 32 },
                "llama.attention.head_count": { "u32": 32 }
            },
            "expected_outcome": "missing_key"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "expected missing_key must pass when key is missing; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_02_003_metadata_missing_key_with_wrong_expectation_fails() {
    // Classifier reports missing_key, observer expected ok — FAIL.
    let obs = json!({
        "metadata": {
            "kv": {
                "llama.embedding_length":     { "u32": 4096 },
                "llama.block_count":          { "u32": 32 },
                "llama.attention.head_count": { "u32": 32 }
            },
            "expected_outcome": "ok"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "expected ok with missing key must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-02-003"));
}

#[test]
fn falsify_crux_b_02_003_metadata_wrong_type_expected_wrong_type_passes() {
    let obs = json!({
        "metadata": {
            "kv": {
                "general.architecture":       { "u32": 7 },  // should be str
                "llama.embedding_length":     { "u32": 4096 },
                "llama.block_count":          { "u32": 32 },
                "llama.attention.head_count": { "u32": 32 }
            },
            "expected_outcome": "wrong_type"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "expected wrong_type with u32 for architecture must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_02_003_metadata_string_for_embedding_length_is_wrong_type() {
    let obs = json!({
        "metadata": {
            "kv": {
                "general.architecture":       { "str": "llama" },
                "llama.embedding_length":     { "str": "4096" },  // should be u32
                "llama.block_count":          { "u32": 32 },
                "llama.attention.head_count": { "u32": 32 }
            },
            "expected_outcome": "wrong_type"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "str for u32 field must classify as wrong_type; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== FALSIFY-CRUX-B-02-004 peft =====

#[test]
fn falsify_crux_b_02_004_peft_default_targets_resolve() {
    let obs = json!({
        "peft": {
            "tensor_names": [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.mlp.up_proj.weight"
            ],
            "target_modules":   ["q_proj", "v_proj"],
            "expected_outcome": "resolved"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "q_proj+v_proj over llama layout must resolve; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_02_004_peft_unknown_target_is_unresolved() {
    // Classifier reports unresolved, observer pre-declared unresolved — PASS.
    let obs = json!({
        "peft": {
            "tensor_names":     ["model.layers.0.self_attn.q_proj.weight"],
            "target_modules":   ["q_proj", "missing_module"],
            "expected_outcome": "unresolved"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "unknown target module with expected=unresolved must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_02_004_peft_unknown_target_with_wrong_expectation_fails() {
    let obs = json!({
        "peft": {
            "tensor_names":     ["model.layers.0.self_attn.q_proj.weight"],
            "target_modules":   ["q_proj", "missing_module"],
            "expected_outcome": "resolved"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "unresolved with expected=resolved must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-02-004"));
}

#[test]
fn falsify_crux_b_02_004_peft_empty_target_list_trivially_resolves() {
    let obs = json!({
        "peft": {
            "tensor_names":     ["anything"],
            "target_modules":   [],
            "expected_outcome": "resolved"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "empty target_modules must resolve vacuously; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== Multi-gate + JSON shape =====

#[test]
fn falsify_crux_b_02_multi_gate_all_pass() {
    let obs = json!({
        "layout": {
            "listing": ["model.safetensors", "config.json", "tokenizer.json"]
        },
        "metadata": {
            "kv": {
                "general.architecture":       { "str": "llama" },
                "llama.embedding_length":     { "u32": 4096 },
                "llama.block_count":          { "u32": 32 },
                "llama.attention.head_count": { "u32": 32 }
            },
            "expected_outcome": "ok"
        },
        "peft": {
            "tensor_names":     ["model.layers.0.self_attn.q_proj.weight"],
            "target_modules":   ["q_proj"],
            "expected_outcome": "resolved"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "all three gates green must exit 0; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] layout"));
    assert!(stdout.contains("[PASS] metadata"));
    assert!(stdout.contains("[PASS] peft"));
}

#[test]
fn falsify_crux_b_02_json_output_shape() {
    let obs = json!({
        "layout": { "listing": ["model.safetensors", "config.json", "tokenizer.json"] }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "--json",
            "gguf-safetensors-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json must emit valid JSON");
    assert_eq!(parsed["contract"], "CRUX-B-02");
    assert!(parsed["gates"].is_array());
    assert_eq!(parsed["gates"][0]["gate"], "layout");
    assert_eq!(parsed["gates"][0]["falsify_id"], "FALSIFY-CRUX-B-02-001");
    assert_eq!(parsed["gates"][0]["passed"], true);
}
