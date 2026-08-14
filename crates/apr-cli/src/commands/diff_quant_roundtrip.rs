//! CRUX-B-20 — `apr diff --quant-roundtrip` per-tensor quant error report.
//!
//! Loads two model files (a high-precision reference and a quantized variant),
//! compares each tensor pairwise on RMSE / cosine / max-abs error, sorts by
//! RMSE descending, and emits a JSON or TSV report.
//!
//! Verdict bucketing (per contract `contracts/crux-B-20-v1.yaml`
//! `equations.per_tensor_error_metrics`):
//! - cosine ≥ 0.999 → `green`
//! - cosine ≥ 0.99  → `yellow`
//! - else           → `red`
//!
//! Threshold gate (contract invariant
//! "exit code ≠ 0 if any tensor cosine < 0.95 (unless --no-threshold)"):
//! - default threshold 0.95
//! - when ANY tensor cosine < threshold and `no_threshold` is false, the
//!   caller sets a non-zero exit code (caller wires this via `CliError`).

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use crate::error::{CliError, Result};

#[derive(Debug, Clone, Serialize)]
pub struct TensorRoundtripReport {
    pub tensor: String,
    pub qtype: String,
    pub numel: usize,
    pub rmse: f32,
    pub cosine: f32,
    pub max_abs: f32,
    pub verdict: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantRoundtripReport {
    pub reference: String,
    pub quantized: String,
    pub threshold: f32,
    pub tensors: Vec<TensorRoundtripReport>,
    pub any_below_threshold: bool,
}

/// Verdict bucket per the B-20 spec.
pub(crate) fn verdict_for(cosine: f32) -> &'static str {
    if cosine >= 0.999 {
        "green"
    } else if cosine >= 0.99 {
        "yellow"
    } else {
        "red"
    }
}

/// Pairwise error metrics — pure math, easy to unit-test.
///
/// Returns (rmse, cosine, max_abs). For empty input or zero-norm vectors,
/// returns sentinels so the caller can still emit a row instead of crashing:
/// - empty → (0.0, 1.0, 0.0)
/// - either norm == 0 but lengths match → (rmse, 0.0, max_abs)
pub(crate) fn error_metrics(reference: &[f32], quantized: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(reference.len(), quantized.len(), "length mismatch");
    if reference.is_empty() {
        return (0.0, 1.0, 0.0);
    }
    let n = reference.len() as f64;
    let mut sum_sq_err = 0.0f64;
    let mut dot = 0.0f64;
    let mut nr2 = 0.0f64;
    let mut nq2 = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&r, &q) in reference.iter().zip(quantized) {
        let err = r - q;
        sum_sq_err += f64::from(err) * f64::from(err);
        dot += f64::from(r) * f64::from(q);
        nr2 += f64::from(r) * f64::from(r);
        nq2 += f64::from(q) * f64::from(q);
        let abs = err.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let rmse = (sum_sq_err / n).sqrt() as f32;
    let cosine = if nr2 == 0.0 || nq2 == 0.0 {
        0.0
    } else {
        (dot / (nr2.sqrt() * nq2.sqrt())) as f32
    };
    (rmse, cosine, max_abs)
}

/// Build a roundtrip report from two paths.
///
/// The current implementation supports SafeTensors fp16/fp32 reference and
/// SafeTensors quantized variants (the only pair we can reliably read with
/// pure Rust dependencies). GGUF Q4_K_M ground truth lives in
/// `format::gguf::dequant` and a follow-up PR will wire it in — the contract
/// allows partial_algorithm_level shipping with one format pair.
pub fn build_report(
    reference_path: &Path,
    quantized_path: &Path,
    threshold: f32,
) -> Result<QuantRoundtripReport> {
    let reference = load_tensors_f32(reference_path)?;
    let quantized = load_tensors_f32(quantized_path)?;

    let mut rows: Vec<TensorRoundtripReport> = Vec::new();
    let mut any_below_threshold = false;

    for (name, (ref_data, ref_dtype)) in &reference {
        let Some((q_data, q_dtype)) = quantized.get(name) else {
            continue; // tensor absent in quantized side — skip
        };
        if ref_data.len() != q_data.len() {
            // Shape mismatch — surface as a red row with cosine=0 so the
            // threshold gate trips, but don't abort the whole report.
            rows.push(TensorRoundtripReport {
                tensor: name.clone(),
                qtype: format!("{}↔{}", ref_dtype, q_dtype),
                numel: ref_data.len(),
                rmse: f32::INFINITY,
                cosine: 0.0,
                max_abs: f32::INFINITY,
                verdict: "red",
            });
            any_below_threshold = true;
            continue;
        }
        let (rmse, cosine, max_abs) = error_metrics(ref_data, q_data);
        if cosine < threshold {
            any_below_threshold = true;
        }
        rows.push(TensorRoundtripReport {
            tensor: name.clone(),
            qtype: q_dtype.clone(),
            numel: ref_data.len(),
            rmse,
            cosine,
            max_abs,
            verdict: verdict_for(cosine),
        });
    }

    // Sort by rmse DESC per contract invariant.
    rows.sort_by(|a, b| {
        b.rmse
            .partial_cmp(&a.rmse)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(QuantRoundtripReport {
        reference: reference_path.display().to_string(),
        quantized: quantized_path.display().to_string(),
        threshold,
        tensors: rows,
        any_below_threshold,
    })
}

/// Load a model's tensors as `BTreeMap<name, (Vec<f32>, dtype_label)>`.
///
/// Supports SafeTensors (F32, F16, BF16) via the `safetensors` crate.
fn load_tensors_f32(path: &Path) -> Result<BTreeMap<String, (Vec<f32>, String)>> {
    let bytes = std::fs::read(path)
        .map_err(|e| CliError::ValidationFailed(format!("read {}: {}", path.display(), e)))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| {
        CliError::ValidationFailed(format!("parse safetensors {}: {}", path.display(), e))
    })?;

    let mut out = BTreeMap::new();
    for name in st.names() {
        let view = st.tensor(name).map_err(|e| {
            CliError::ValidationFailed(format!("tensor '{name}' in {}: {}", path.display(), e))
        })?;
        let dtype = view.dtype();
        let dtype_label = format!("{dtype:?}");
        let data = match dtype {
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                let mut v = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                v
            }
            safetensors::Dtype::F16 => {
                let bytes = view.data();
                let mut v = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    v.push(half::f16::from_bits(bits).to_f32());
                }
                v
            }
            safetensors::Dtype::BF16 => {
                let bytes = view.data();
                let mut v = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    v.push(half::bf16::from_bits(bits).to_f32());
                }
                v
            }
            other => {
                return Err(CliError::ValidationFailed(format!(
                    "unsupported dtype {other:?} for tensor '{name}' in {} — supported: F32, F16, BF16",
                    path.display()
                )));
            }
        };
        // safetensors 0.7: `names()` yields `&str`, so `.clone()` would keep a borrow.
        out.insert(name.to_string(), (data, dtype_label));
    }
    Ok(out)
}

/// Render the report as TSV (header + one row per tensor).
pub(crate) fn render_tsv(report: &QuantRoundtripReport) -> String {
    let mut out = String::new();
    out.push_str("tensor\tqtype\tnumel\trmse\tcosine\tmax_abs\tverdict\n");
    for r in &report.tensors {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.6e}\t{:.6}\t{:.6e}\t{}\n",
            r.tensor, r.qtype, r.numel, r.rmse, r.cosine, r.max_abs, r.verdict
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_identity() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let (rmse, cosine, max_abs) = error_metrics(&a, &a);
        assert_eq!(rmse, 0.0);
        assert!((cosine - 1.0).abs() < 1e-6);
        assert_eq!(max_abs, 0.0);
    }

    #[test]
    fn metrics_anti_parallel() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![-1.0f32, -2.0, -3.0];
        let (_rmse, cosine, _max_abs) = error_metrics(&a, &b);
        assert!((cosine - -1.0).abs() < 1e-6);
    }

    #[test]
    fn metrics_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let (_rmse, cosine, _max_abs) = error_metrics(&a, &b);
        assert!(cosine.abs() < 1e-6);
    }

    #[test]
    fn metrics_empty() {
        let (rmse, cosine, max_abs) = error_metrics(&[], &[]);
        assert_eq!(rmse, 0.0);
        assert_eq!(cosine, 1.0);
        assert_eq!(max_abs, 0.0);
    }

    #[test]
    fn metrics_small_error_high_cosine() {
        // ε perturbation should give very high cosine, very low rmse.
        let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = a.iter().map(|&x| x + 0.0005).collect();
        let (rmse, cosine, max_abs) = error_metrics(&a, &b);
        assert!(rmse < 0.001);
        assert!(cosine > 0.999, "cosine={cosine}");
        assert!((max_abs - 0.0005).abs() < 1e-4);
    }

    #[test]
    fn verdict_buckets() {
        assert_eq!(verdict_for(1.0), "green");
        assert_eq!(verdict_for(0.9995), "green");
        assert_eq!(verdict_for(0.999), "green");
        assert_eq!(verdict_for(0.9989), "yellow");
        assert_eq!(verdict_for(0.99), "yellow");
        assert_eq!(verdict_for(0.989), "red");
        assert_eq!(verdict_for(0.0), "red");
    }

    // -----------------------------------------------------------------
    // Integration falsifiers (build real safetensors pairs, end-to-end
    // through build_report).
    // -----------------------------------------------------------------

    use safetensors::tensor::{Dtype, TensorView};
    use std::fs;
    use tempfile::TempDir;

    fn write_f32_safetensors(path: &std::path::Path, tensors: &[(&str, &[u8], Vec<usize>)]) {
        let views: Vec<(&str, TensorView<'_>)> = tensors
            .iter()
            .map(|(name, bytes, shape)| {
                (
                    *name,
                    TensorView::new(Dtype::F32, shape.clone(), bytes).expect("TensorView"),
                )
            })
            .collect();
        let serialized = safetensors::serialize(views, None).expect("serialize");
        fs::write(path, serialized).expect("write");
    }

    fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// FALSIFY-CRUX-B-20-001 — rows sorted by rmse DESC; every row carries
    /// rmse, cosine, qtype, numel, max_abs, verdict.
    #[test]
    fn falsify_crux_b_20_001_rows_sorted_by_rmse_desc() {
        let tmp = TempDir::new().expect("tempdir");
        let ref_path = tmp.path().join("ref.safetensors");
        let q_path = tmp.path().join("quant.safetensors");

        let elems = 256;
        let base: Vec<f32> = (0..elems).map(|i| (i as f32 * 0.1).sin()).collect();
        let near: Vec<f32> = base.iter().map(|x| x + 0.0005).collect();
        let mid: Vec<f32> = base.iter().map(|x| x + 0.02).collect();
        let far: Vec<f32> = base.iter().map(|x| x + 0.5).collect();

        let ref_bytes = f32_to_bytes(&base);
        let near_bytes = f32_to_bytes(&near);
        let mid_bytes = f32_to_bytes(&mid);
        let far_bytes = f32_to_bytes(&far);

        write_f32_safetensors(
            &ref_path,
            &[
                ("a_near", &ref_bytes, vec![elems]),
                ("b_mid", &ref_bytes, vec![elems]),
                ("c_far", &ref_bytes, vec![elems]),
            ],
        );
        write_f32_safetensors(
            &q_path,
            &[
                ("a_near", &near_bytes, vec![elems]),
                ("b_mid", &mid_bytes, vec![elems]),
                ("c_far", &far_bytes, vec![elems]),
            ],
        );

        let report = build_report(&ref_path, &q_path, 0.95).expect("build_report");
        assert_eq!(report.tensors.len(), 3, "expected 3 tensor rows");

        // sorted by rmse DESC: c_far > b_mid > a_near
        assert_eq!(report.tensors[0].tensor, "c_far");
        assert_eq!(report.tensors[1].tensor, "b_mid");
        assert_eq!(report.tensors[2].tensor, "a_near");

        // monotone decreasing rmse
        for w in report.tensors.windows(2) {
            assert!(
                w[0].rmse >= w[1].rmse,
                "rmse must be non-increasing: {} >= {}",
                w[0].rmse,
                w[1].rmse,
            );
        }

        // schema check — every row has all 7 fields populated meaningfully
        for r in &report.tensors {
            assert_eq!(r.qtype, "F32");
            assert_eq!(r.numel, elems);
            assert!(r.rmse >= 0.0);
            assert!(r.cosine >= -1.0 && r.cosine <= 1.0);
            assert!(r.max_abs >= 0.0);
            assert!(matches!(r.verdict, "green" | "yellow" | "red"));
        }
    }

    /// FALSIFY-CRUX-B-20-002 — `any_below_threshold` flips when ANY tensor's
    /// cosine drops below the threshold; clean when all are above.
    #[test]
    fn falsify_crux_b_20_002_threshold_gate_flips_on_low_cosine() {
        let tmp = TempDir::new().expect("tempdir");
        let ref_path = tmp.path().join("ref.safetensors");
        let q_path = tmp.path().join("quant.safetensors");

        let elems = 256;
        let base: Vec<f32> = (0..elems).map(|i| (i as f32 * 0.1).cos()).collect();
        let near: Vec<f32> = base.iter().map(|x| x + 0.0005).collect();

        let ref_bytes = f32_to_bytes(&base);
        let near_bytes = f32_to_bytes(&near);

        write_f32_safetensors(&ref_path, &[("only.weight", &ref_bytes, vec![elems])]);
        write_f32_safetensors(&q_path, &[("only.weight", &near_bytes, vec![elems])]);

        // Tiny perturbation → high cosine → does NOT flip (default 0.95).
        let clean = build_report(&ref_path, &q_path, 0.95).expect("build");
        assert!(
            !clean.any_below_threshold,
            "high-cosine case should NOT flip"
        );
        assert_eq!(clean.tensors[0].verdict, "green");

        // Sign-flip half the values — cosine drops sharply.
        let bad: Vec<f32> = base
            .iter()
            .enumerate()
            .map(|(i, &x)| if i % 2 == 0 { x } else { -x })
            .collect();
        let bad_bytes = f32_to_bytes(&bad);
        write_f32_safetensors(&q_path, &[("only.weight", &bad_bytes, vec![elems])]);

        let dirty = build_report(&ref_path, &q_path, 0.95).expect("build");
        assert!(dirty.any_below_threshold, "low-cosine case MUST flip");
        assert!(
            dirty.tensors[0].cosine < 0.95,
            "expected cosine < 0.95, got {}",
            dirty.tensors[0].cosine
        );
    }
}
