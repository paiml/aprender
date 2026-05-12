//! Regression test for `apr trace --json --payload` output schema.
//!
//! Locks in the JSON schema shape that satisfies the M34 FAST PATH Step
//! 2 exit criterion (companion `paiml/claude-code-parity-apr`
//! docs/specifications/claude-code-parity-apr-poc.md § "M32d FAST PATH"):
//!
//!   "apr trace --json --payload <gguf> --prompt 'What is 2+2?' returns
//!    non-null output_stats for every transformer_block_N entry, with
//!    finite L2 norms."
//!
//! Skipped when the cached Qwen3-Coder GGUF is absent (fixture-absent ≠
//! defect, per M32c.2.2.2.1.4 convention).

use std::path::Path;
use std::process::Command;

const CANONICAL_QWEN3_CODER_GGUF_PATHS: &[&str] = &[
    "/home/noah/.cache/pacha/models/2b88b180a790988f.gguf",
    "/mnt/nvme-raid0/models/qwen3-coder-30b-q4k.gguf",
];

const APR_BIN_PATHS: &[&str] = &[
    "/mnt/nvme-raid0/targets/aprender/release/apr",
    "target/release/apr",
];

#[test]
fn f_apr_trace_json_payload_schema_001() {
    let Some(gguf_path) = CANONICAL_QWEN3_CODER_GGUF_PATHS
        .iter()
        .find(|p| Path::new(p).exists())
    else {
        eprintln!("F-APR-TRACE-JSON-PAYLOAD-001: skipped — no cached Qwen3-Coder GGUF");
        return;
    };
    let Some(apr_bin) = APR_BIN_PATHS.iter().find(|p| Path::new(p).exists()) else {
        eprintln!("F-APR-TRACE-JSON-PAYLOAD-001: skipped — apr binary not found");
        return;
    };

    eprintln!("F-APR-TRACE-JSON-PAYLOAD-001: running {apr_bin} trace --json --payload {gguf_path}");
    let start = std::time::Instant::now();
    let output = Command::new(apr_bin)
        .args(["trace", "--json", "--payload", gguf_path])
        .output()
        .expect("apr trace command");
    let elapsed = start.elapsed();

    assert!(
        output.status.success(),
        "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: apr exited {:?}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);

    // FAST PATH Step 2 exit criterion: stdout must be valid JSON with no
    // preamble lines (so `apr trace --json --payload | jq` works).
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!(
            "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: stdout is not valid JSON: {e}\n\
             First 500 chars of stdout:\n{}",
            &stdout[..stdout.len().min(500)]
        );
    });

    // Top-level required fields
    let required_top = [
        "format",
        "architecture",
        "num_layers",
        "hidden_dim",
        "vocab_size",
        "prompt",
        "encoded_tokens",
        "embedding",
        "layers",
        "final_norm",
        "logits",
    ];
    for field in &required_top {
        assert!(
            parsed.get(field).is_some(),
            "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: top-level field `{field}` missing"
        );
    }

    // 48-layer count for Qwen3-Coder-30B-A3B-Instruct
    let layers = parsed["layers"].as_array().expect("layers must be array");
    assert_eq!(
        layers.len(),
        48,
        "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: expected 48 layers (Qwen3-Coder-30B), got {}",
        layers.len()
    );

    // Per-layer required fields + finite stats
    let required_layer = [
        "layer_idx",
        "attn_norm",
        "qkv",
        "attn_out",
        "ffn_norm",
        "ffn_out",
        "output",
    ];
    for (i, layer) in layers.iter().enumerate() {
        for field in &required_layer {
            assert!(
                layer.get(field).is_some(),
                "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: layer[{i}] field `{field}` missing"
            );
        }
        let output = &layer["output"];
        let std_dev = output["std_dev"].as_f64().unwrap_or(f64::NAN);
        assert!(
            std_dev.is_finite(),
            "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: layer[{i}].output.std_dev = {std_dev} (not finite)"
        );
        let nan_count = output["nan_count"].as_u64().unwrap_or(0);
        let inf_count = output["inf_count"].as_u64().unwrap_or(0);
        assert_eq!(
            nan_count, 0,
            "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: layer[{i}].output.nan_count = {nan_count}"
        );
        assert_eq!(
            inf_count, 0,
            "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: layer[{i}].output.inf_count = {inf_count}"
        );
    }

    // logits.l2_norm must be finite
    let l2 = parsed["logits"]["l2_norm"]
        .as_f64()
        .expect("logits.l2_norm must be a number");
    assert!(
        l2.is_finite() && l2 > 0.0,
        "F-APR-TRACE-JSON-PAYLOAD-001 FAIL: logits.l2_norm = {l2} (not finite-positive)"
    );

    eprintln!(
        "F-APR-TRACE-JSON-PAYLOAD-001: PASS\n  elapsed = {elapsed:?}\n  layers = {} (all finite)\n  logits.l2_norm = {l2:.4}",
        layers.len()
    );
}
