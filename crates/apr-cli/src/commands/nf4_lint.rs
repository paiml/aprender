//! CRUX-B-10 — `apr nf4-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the four NF4 classifiers in `nf4_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "codebook": {
//!     "expected": [ -1.0, -0.6961..., ..., 1.0 ]   // 16 entries, bnb canonical
//!   },
//!   "roundtrip": {
//!     "weights":    [f32, ...],
//!     "max_rel_l2": 0.15
//!   },
//!   "storage": {
//!     "n_weights":                    1_100_000_000,
//!     "block_size":                   64,
//!     "double_quant":                 false,
//!     "expected_min_bytes_per_weight": 0.50,
//!     "expected_max_bytes_per_weight": 0.65
//!   },
//!   "parity": {
//!     "target":         0.0,
//!     "expected_index": 7
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-10
//! stderr stamp on any failing gate.

use crate::commands::nf4_classifier::{
    expected_nf4_storage_bytes, nearest_codebook_index, nf4_dequantize_block, nf4_quantize_block,
    rel_l2_error, NF4_CODEBOOK, NF4_DEFAULT_BLOCK_SIZE, NF4_MAX_REL_L2_ERROR_SYNTHETIC,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Nf4LintArgs {
    pub observation_file: String,
    pub json: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct GateReport {
    gate: &'static str,
    falsify_id: &'static str,
    outcome: String,
    passed: bool,
}

pub fn run(args: Nf4LintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-10: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-10: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-10: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-10: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(cb) = obs.get("codebook") {
        let (report, err) = run_codebook_gate(cb);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(rt) = obs.get("roundtrip") {
        let (report, err) = run_roundtrip_gate(rt);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(st) = obs.get("storage") {
        let (report, err) = run_storage_gate(st);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(p) = obs.get("parity") {
        let (report, err) = run_parity_gate(p);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(
            "FALSIFY-CRUX-B-10: observation has none of codebook/roundtrip/storage/parity".into(),
        );
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-10",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            let tag = if r.passed { "PASS" } else { "FAIL" };
            println!("[{tag}] {} ({}): {}", r.gate, r.falsify_id, r.outcome);
        }
    }

    if !failures.is_empty() {
        return Err(failures.join("\n"));
    }
    Ok(())
}

fn read_f32_array(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default()
}

fn run_codebook_gate(v: &Value) -> (GateReport, Option<String>) {
    let expected = v.get("expected").map(read_f32_array).unwrap_or_default();
    let passed = if expected.is_empty() {
        NF4_CODEBOOK.len() == 16
    } else if expected.len() != NF4_CODEBOOK.len() {
        false
    } else {
        expected
            .iter()
            .zip(NF4_CODEBOOK.iter())
            .all(|(e, c)| (*e - *c).abs() < 1e-6)
    };
    let desc = if passed {
        format!("codebook matches ({} entries)", NF4_CODEBOOK.len())
    } else {
        format!(
            "codebook divergence (expected_len={}, got_len={})",
            expected.len(),
            NF4_CODEBOOK.len()
        )
    };
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-10-001 codebook gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "codebook",
            falsify_id: "FALSIFY-CRUX-B-10-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_roundtrip_gate(v: &Value) -> (GateReport, Option<String>) {
    let weights = v.get("weights").map(read_f32_array).unwrap_or_default();
    let max_rel_l2 = v
        .get("max_rel_l2")
        .and_then(|x| x.as_f64())
        .unwrap_or(NF4_MAX_REL_L2_ERROR_SYNTHETIC);

    if weights.is_empty() {
        let desc = "weights array is empty".to_string();
        return (
            GateReport {
                gate: "roundtrip",
                falsify_id: "FALSIFY-CRUX-B-10-003",
                outcome: desc.clone(),
                passed: false,
            },
            Some(format!(
                "FALSIFY-CRUX-B-10-003 roundtrip gate failed: {desc}"
            )),
        );
    }

    let (idx, scale) = nf4_quantize_block(&weights);
    let recon = nf4_dequantize_block(&idx, scale);
    let err = rel_l2_error(&weights, &recon);
    let passed = err.is_finite() && err <= max_rel_l2;
    let desc = format!("rel_l2={err:.6} (max={max_rel_l2})");
    let fail = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-10-003 roundtrip gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "roundtrip",
            falsify_id: "FALSIFY-CRUX-B-10-003",
            outcome: desc,
            passed,
        },
        fail,
    )
}

fn run_storage_gate(v: &Value) -> (GateReport, Option<String>) {
    let n_weights = v.get("n_weights").and_then(|x| x.as_u64()).unwrap_or(0);
    let block_size = v
        .get("block_size")
        .and_then(|x| x.as_u64())
        .unwrap_or(NF4_DEFAULT_BLOCK_SIZE as u64);
    let dq = v
        .get("double_quant")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let min_bpw = v
        .get("expected_min_bytes_per_weight")
        .and_then(|x| x.as_f64())
        .unwrap_or(0.50);
    let max_bpw = v
        .get("expected_max_bytes_per_weight")
        .and_then(|x| x.as_f64())
        .unwrap_or(0.65);

    if n_weights == 0 || block_size == 0 {
        let desc = format!("invalid n_weights={n_weights} block_size={block_size}");
        return (
            GateReport {
                gate: "storage",
                falsify_id: "FALSIFY-CRUX-B-10-002",
                outcome: desc.clone(),
                passed: false,
            },
            Some(format!("FALSIFY-CRUX-B-10-002 storage gate failed: {desc}")),
        );
    }

    let bytes = expected_nf4_storage_bytes(n_weights, block_size, dq);
    let bpw = (bytes as f64) / (n_weights as f64);
    let passed = bpw >= min_bpw && bpw <= max_bpw;
    let desc = format!("bytes={bytes} bpw={bpw:.4} (envelope [{min_bpw},{max_bpw}], dq={dq})");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-10-002 storage gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "storage",
            falsify_id: "FALSIFY-CRUX-B-10-002",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_parity_gate(v: &Value) -> (GateReport, Option<String>) {
    let target = v
        .get("target")
        .and_then(|x| x.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.0);
    let expected = v
        .get("expected_index")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u8;
    let got = nearest_codebook_index(target);
    let passed = got == expected;
    let desc = format!("target={target} expected_index={expected} got={got}");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-10-004 parity gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "parity",
            falsify_id: "FALSIFY-CRUX-B-10-004",
            outcome: desc,
            passed,
        },
        err,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_obs(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn args_for(f: &NamedTempFile) -> Nf4LintArgs {
        Nf4LintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = Nf4LintArgs {
            observation_file: "/nonexistent/path/nf4.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-10"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("   \n  ");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("{ this is not json ");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("none of codebook/roundtrip/storage/parity"));
    }

    #[test]
    fn codebook_default_passes_when_empty_expected() {
        // Empty `expected` → gate checks NF4_CODEBOOK.len()==16, which passes.
        let f = write_obs(r#"{"codebook": {}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn codebook_matching_expected_passes() {
        let expected: Vec<f32> = NF4_CODEBOOK.to_vec();
        let obs = serde_json::json!({ "codebook": { "expected": expected } });
        let f = write_obs(&obs.to_string());
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn codebook_wrong_length_fails() {
        let f = write_obs(r#"{"codebook": {"expected": [0.0, 1.0]}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-10-001"));
    }

    #[test]
    fn roundtrip_empty_weights_fails() {
        let f = write_obs(r#"{"roundtrip": {"weights": []}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-10-003"));
    }

    #[test]
    fn roundtrip_well_scaled_weights_pass() {
        // Weights whose extremes hit codebook values roundtrip with low error.
        let f = write_obs(
            r#"{"roundtrip": {"weights": [1.0, -1.0, 0.0, 0.5, -0.5], "max_rel_l2": 0.5}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn storage_invalid_dimensions_fail() {
        let f = write_obs(r#"{"storage": {"n_weights": 0, "block_size": 64}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-10-002"));
    }

    #[test]
    fn storage_envelope_passes_on_large_tensor() {
        let obs = serde_json::json!({
            "storage": {
                "n_weights": 1_000_000_000u64,
                "block_size": 64,
                "double_quant": false,
                "expected_min_bytes_per_weight": 0.50,
                "expected_max_bytes_per_weight": 0.65
            }
        });
        let f = write_obs(&obs.to_string());
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn parity_matching_index_passes() {
        let f = write_obs(r#"{"parity": {"target": 0.0, "expected_index": 7}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn parity_wrong_index_fails() {
        let f = write_obs(r#"{"parity": {"target": 0.0, "expected_index": 3}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-10-004"));
    }

    #[test]
    fn json_mode_renders_all_gates_ok() {
        let obs = serde_json::json!({
            "codebook": {},
            "parity": { "target": 1.0, "expected_index": 15 }
        });
        let f = write_obs(&obs.to_string());
        let args = Nf4LintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
