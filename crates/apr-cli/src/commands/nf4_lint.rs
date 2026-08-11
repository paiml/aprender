//! CRUX-B-10 — `apr nf4-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the four NF4 classifiers in `nf4_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "codebook": {
//!     "expected": [ -1.0, -0.6961..., ..., 1.0 ]   // REQUIRED: 16 entries, bnb canonical
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

use super::lint_error::{load_json_observation, LintError};
use crate::commands::lint_vacuity::{json_type, verdict_tag, Verdict};
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

pub fn run(args: Nf4LintArgs) -> Result<(), LintError> {
    let obs: Value = load_json_observation(&args.observation_file, "FALSIFY-CRUX-B-10")?;

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
        return Err(LintError::unusable(
            "FALSIFY-CRUX-B-10: observation has none of codebook/roundtrip/storage/parity",
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-10",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            let tag = verdict_tag(r.passed, &r.outcome);
            println!("[{tag}] {} ({}): {}", r.gate, r.falsify_id, r.outcome);
        }
    }

    if !failures.is_empty() {
        return Err(LintError::gate_failed(failures.join("\n")));
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

/// Read the observation's `expected` codebook.
///
/// 0.63.0 funnelled "absent", "empty", "wrong type" and "misspelled key" into
/// one empty `Vec`, whose branch then asserted `NF4_CODEBOOK.len() == 16` — a
/// tautology over a compile-time constant. The gate printed "codebook matches
/// (16 entries)" having compared nothing.
fn parse_expected_codebook(v: &Value) -> Result<Vec<f32>, String> {
    let obj = v.as_object().ok_or_else(|| {
        format!(
            "codebook section must be a JSON object, got {}",
            json_type(v)
        )
    })?;
    let Some(expected) = obj.get("expected").filter(|e| !e.is_null()) else {
        let keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        return Err(format!(
            "VACUOUS: codebook section supplies no \"expected\" array (keys present: {keys:?}), \
             so the NF4 codebook was compared against nothing"
        ));
    };
    let arr = expected.as_array().ok_or_else(|| {
        format!(
            "codebook.expected must be an array of {} numbers, got {}",
            NF4_CODEBOOK.len(),
            json_type(expected)
        )
    })?;
    if arr.is_empty() {
        return Err(
            "VACUOUS: codebook.expected is an empty array, so no entry was compared".to_string(),
        );
    }
    arr.iter()
        .enumerate()
        .map(|(i, n)| {
            n.as_f64().map(|f| f as f32).ok_or_else(|| {
                format!(
                    "codebook.expected[{i}] must be a number, got {}",
                    json_type(n)
                )
            })
        })
        .collect()
}

fn compare_codebook(v: &Value) -> Result<String, String> {
    let expected = parse_expected_codebook(v)?;
    if expected.len() != NF4_CODEBOOK.len() {
        return Err(format!(
            "codebook divergence (expected_len={}, got_len={})",
            expected.len(),
            NF4_CODEBOOK.len()
        ));
    }
    if let Some((i, (e, c))) = expected
        .iter()
        .zip(NF4_CODEBOOK.iter())
        .enumerate()
        .find(|(_, (e, c))| (**e - **c).abs() >= 1e-6)
        .map(|(i, pair)| (i, pair))
    {
        return Err(format!(
            "codebook divergence at index {i}: expected {e}, got {c}"
        ));
    }
    Ok(format!(
        "supplied codebook compared entry-by-entry against the built-in NF4 table and matched \
         ({} entries)",
        NF4_CODEBOOK.len()
    ))
}

fn run_codebook_gate(v: &Value) -> (GateReport, Option<String>) {
    let result = compare_codebook(v);
    let verdict = Verdict::of(&result);
    let desc = match result {
        Ok(msg) | Err(msg) => msg,
    };
    let err = if verdict == Verdict::Pass {
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
            passed: verdict == Verdict::Pass,
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
        let err = run(args).unwrap_err().to_string();
        // The whole *-lint family reports a missing input identically:
        // "File not found: <path>" with exit 3 (commands::lint_error).
        assert!(err.contains("File not found"), "got: {err}");
        assert!(err.contains("/nonexistent/path/nf4.json"), "got: {err}");
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("   \n  ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("{ this is not json ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("none of codebook/roundtrip/storage/parity"));
    }

    /// Was `codebook_default_passes_when_empty_expected`, asserting `is_ok()`
    /// on `{"codebook": {}}` with the comment "gate checks
    /// NF4_CODEBOOK.len()==16, which passes" — the test encoded the tautology
    /// as intended behaviour and so held the defect in place.
    #[test]
    fn falsifier_codebook_with_nothing_supplied_is_vacuous() {
        for body in [
            r#"{"codebook": {}}"#,                     // section present, empty
            r#"{"codebook": {"expected": []}}"#,       // empty array
            r#"{"codebook": {"expcted": [0.0,1.0]}}"#, // key misspelled
        ] {
            let f = write_obs(body);
            let err = run(args_for(&f)).unwrap_err().to_string();
            assert!(err.contains("VACUOUS"), "{body}: {err}");
            assert!(err.contains("FALSIFY-CRUX-B-10-001"), "{body}: {err}");
            assert!(
                !err.contains("codebook matches"),
                "{body}: must not claim a comparison happened: {err}"
            );
        }
    }

    #[test]
    fn falsifier_wrong_typed_codebook_is_a_schema_error() {
        for (body, want) in [
            (
                r#"{"codebook": {"expected": "oops"}}"#,
                "codebook.expected must be an array",
            ),
            (
                r#"{"codebook": {"expected": [0.0, "x"]}}"#,
                "codebook.expected[1] must be a number",
            ),
            (
                r#"{"codebook": "oops"}"#,
                "codebook section must be a JSON object",
            ),
        ] {
            let f = write_obs(body);
            let err = run(args_for(&f)).unwrap_err().to_string();
            assert!(err.contains(want), "{body}: expected {want:?}, got {err}");
        }
    }

    /// A 16-entry codebook that differs in one entry must be caught — the
    /// length check alone would wave it through.
    #[test]
    fn falsifier_codebook_with_one_wrong_entry_fails() {
        let mut expected: Vec<f32> = NF4_CODEBOOK.to_vec();
        expected[7] += 0.25;
        let obs = serde_json::json!({ "codebook": { "expected": expected } });
        let f = write_obs(&obs.to_string());
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("codebook divergence at index 7"), "{err}");
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
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-10-001"));
    }

    #[test]
    fn roundtrip_empty_weights_fails() {
        let f = write_obs(r#"{"roundtrip": {"weights": []}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-10-004"));
    }

    #[test]
    fn json_mode_renders_all_gates_ok() {
        let obs = serde_json::json!({
            "codebook": { "expected": NF4_CODEBOOK.to_vec() },
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
