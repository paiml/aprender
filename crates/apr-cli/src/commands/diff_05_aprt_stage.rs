// SHIP-007 PR-D: APRT stage tensor diff for `apr diff --values`.
//
// Contract: contracts/apr-cli-trace-save-tensor-v1.yaml
//   `apr_diff_values_compat` invariant — `apr diff --values` MUST recognize
//   the 12-byte APRT header and read the f32 LE body so stage tensors
//   captured by `apr trace --save-tensor` can be compared element-wise
//   without external metadata.
//
// NOTE: this file is `include!()`-ed by diff.rs; do NOT add `use` for
// crate::error::CliError, std::path::Path, colored::Colorize — diff.rs
// already imports them. Add only NEW imports here.

/// Detect whether `path` opens to a save-tensor file produced by
/// `apr trace --save-tensor` (magic bytes `APRT` at offset 0).
///
/// Returns `false` on any I/O error or non-matching magic — callers
/// fall through to the existing RosettaStone path.
fn is_aprt_stage_file(path: &Path) -> bool {
    use std::io::Read;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() {
        return false;
    }
    magic == *b"APRT"
}

/// Element-wise stats for two equally-sized f32 vectors.
struct AprtStageStats {
    dim_product: usize,
    layer: u32,
    max_abs_diff: f32,
    max_abs_diff_index: usize,
    rms_diff: f32,
    cosine_sim: f64,
    top_diffs: Vec<(usize, f32, f32, f32)>,
}

fn compute_aprt_stage_stats(
    layer: u32,
    a: &[f32],
    b: &[f32],
    top_k: usize,
) -> AprtStageStats {
    let n = a.len();
    let mut max_abs_diff = 0.0f32;
    let mut max_idx = 0usize;
    let mut sum_sq_diff = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut diffs: Vec<(usize, f32, f32, f32)> = Vec::with_capacity(n);
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        sum_sq_diff += f64::from(d) * f64::from(d);
        if d > max_abs_diff {
            max_abs_diff = d;
            max_idx = i;
        }
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
        diffs.push((i, x, y, d));
    }
    let rms = if n == 0 {
        0.0
    } else {
        ((sum_sq_diff / n as f64).sqrt()) as f32
    };
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    diffs.sort_by(|x, y| {
        y.3.partial_cmp(&x.3).unwrap_or(std::cmp::Ordering::Equal)
    });
    diffs.truncate(top_k.max(1));
    AprtStageStats {
        dim_product: n,
        layer,
        max_abs_diff,
        max_abs_diff_index: max_idx,
        rms_diff: rms,
        cosine_sim: cosine,
        top_diffs: diffs,
    }
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn run_aprt_stage_diff(
    path1: &Path,
    path2: &Path,
    limit: usize,
    json_output: bool,
) -> Result<(), CliError> {
    use realizar::inference_trace::save_tensor::read_tensor_file;

    let mut f1 = std::fs::File::open(path1).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot open {}: {e}", path1.display()))
    })?;
    let (h1, vals1) = read_tensor_file(&mut f1).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot read APRT stage tensor {}: {e}",
            path1.display()
        ))
    })?;

    let mut f2 = std::fs::File::open(path2).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot open {}: {e}", path2.display()))
    })?;
    let (h2, vals2) = read_tensor_file(&mut f2).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot read APRT stage tensor {}: {e}",
            path2.display()
        ))
    })?;

    if h1.dim_product != h2.dim_product {
        return Err(CliError::ValidationFailed(format!(
            "APRT dim_product mismatch: {} has {} f32s, {} has {} f32s — \
             cannot compare stage tensors of different sizes",
            path1.display(),
            h1.dim_product,
            path2.display(),
            h2.dim_product
        )));
    }
    if h1.layer != h2.layer {
        return Err(CliError::ValidationFailed(format!(
            "APRT layer mismatch: {} is layer {}, {} is layer {} — \
             stages from different layers cannot be compared element-wise",
            path1.display(),
            h1.layer,
            path2.display(),
            h2.layer
        )));
    }

    let stats = compute_aprt_stage_stats(h1.layer, &vals1, &vals2, limit);

    if json_output {
        let top: Vec<serde_json::Value> = stats
            .top_diffs
            .iter()
            .map(|(idx, a, b, d)| {
                serde_json::json!({
                    "index": idx,
                    "a": a,
                    "b": b,
                    "abs_diff": d,
                })
            })
            .collect();
        let layer_label = if stats.layer == 0xFFFF_FFFF {
            serde_json::Value::String("whole-model".to_string())
        } else {
            serde_json::Value::Number(stats.layer.into())
        };
        let json = serde_json::json!({
            "schema": "apr-stage-diff-v1",
            "path_a": path1.display().to_string(),
            "path_b": path2.display().to_string(),
            "layer": layer_label,
            "dim_product": stats.dim_product,
            "max_abs_diff": stats.max_abs_diff,
            "max_abs_diff_index": stats.max_abs_diff_index,
            "rms_diff": stats.rms_diff,
            "cosine_sim": stats.cosine_sim,
            "top_diffs": top,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════════╗".cyan()
        );
        println!(
            "{}",
            "║           APR STAGE TENSOR DIFF (APRT, element-wise)                         ║".cyan()
        );
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
        );
        let layer_disp = if stats.layer == 0xFFFF_FFFF {
            "whole-model".to_string()
        } else {
            format!("{}", stats.layer)
        };
        println!("║ A:           {:<64}║", path1.display().to_string());
        println!("║ B:           {:<64}║", path2.display().to_string());
        println!("║ Layer:       {:<64}║", layer_disp);
        println!("║ Elements:    {:<64}║", stats.dim_product);
        println!(
            "║ max|diff|:   {:<64}║",
            format!("{:.6e} (at index {})", stats.max_abs_diff, stats.max_abs_diff_index)
        );
        println!("║ RMS diff:    {:<64.6e}║", stats.rms_diff);
        println!("║ cosine sim:  {:<64.10}║", stats.cosine_sim);
        println!(
            "{}",
            "╠══════════════════════════════════════════════════════════════════════════════╣".cyan()
        );
        println!("║ Top divergences (by |a - b|):                                                ║");
        for (idx, a, b, d) in &stats.top_diffs {
            println!(
                "║   [{:>8}] a={:>+13.6e}  b={:>+13.6e}  |Δ|={:.6e}                ║",
                idx, a, b, d
            );
        }
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════════╝".cyan()
        );
    }
    Ok(())
}

#[cfg(test)]
mod aprt_stage_diff_tests {
    use super::*;
    use std::io::Write;

    fn write_aprt_file(path: &Path, layer: u32, values: &[f32]) {
        let mut f = std::fs::File::create(path).unwrap();
        // 12-byte header: magic + layer + dim_product
        f.write_all(b"APRT").unwrap();
        f.write_all(&layer.to_le_bytes()).unwrap();
        let dim = u32::try_from(values.len()).unwrap();
        f.write_all(&dim.to_le_bytes()).unwrap();
        for v in values {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn provenance_pin_pr_d_rev1() {
        // Drift guard for SHIP-007 PR-D PARTIAL_ALGORITHM_LEVEL discharge.
        // If anyone moves this file or renames `is_aprt_stage_file` /
        // `run_aprt_stage_diff`, the include!() in diff.rs and these unit
        // tests will fail — forcing a contract bump in
        // `apr-cli-trace-save-tensor-v1.yaml`.
        let p = std::path::PathBuf::from("/nonexistent/aprt-pin.bin");
        assert!(!is_aprt_stage_file(&p));
    }

    #[test]
    fn is_aprt_stage_file_detects_magic() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("ok.bin");
        write_aprt_file(&p, 0, &[1.0, 2.0, 3.0]);
        assert!(is_aprt_stage_file(&p));
    }

    #[test]
    fn is_aprt_stage_file_rejects_non_aprt() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("not.bin");
        std::fs::write(&p, b"GGUF\x00\x00\x00\x00").unwrap();
        assert!(!is_aprt_stage_file(&p));
    }

    #[test]
    fn is_aprt_stage_file_rejects_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("trunc.bin");
        std::fs::write(&p, b"AP").unwrap();
        assert!(!is_aprt_stage_file(&p));
    }

    #[test]
    fn is_aprt_stage_file_rejects_missing() {
        let p = std::path::PathBuf::from("/nonexistent/path/to/aprt-file.bin");
        assert!(!is_aprt_stage_file(&p));
    }

    #[test]
    fn compute_stats_identical_inputs_zero_diff() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let stats = compute_aprt_stage_stats(0, &a, &a, 3);
        assert_eq!(stats.dim_product, 4);
        assert_eq!(stats.max_abs_diff, 0.0);
        assert_eq!(stats.rms_diff, 0.0);
        assert!((stats.cosine_sim - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compute_stats_known_max_and_rms() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![1.0_f32, 2.5, 3.0, 4.0];
        let stats = compute_aprt_stage_stats(0, &a, &b, 5);
        // max |diff| = |2.0-2.5| = 0.5 at index 1
        assert_eq!(stats.max_abs_diff_index, 1);
        assert!((stats.max_abs_diff - 0.5).abs() < 1e-6);
        // RMS = sqrt((0 + 0.25 + 0 + 0)/4) = 0.25
        assert!((stats.rms_diff - 0.25).abs() < 1e-6);
        // top 1 diff is index 1
        assert_eq!(stats.top_diffs[0].0, 1);
    }

    #[test]
    fn compute_stats_top_k_sorted_descending_by_abs_diff() {
        let a = vec![0.0_f32, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.1_f32, 0.5, -0.3, 0.05, 0.2];
        let stats = compute_aprt_stage_stats(0, &a, &b, 3);
        assert_eq!(stats.top_diffs.len(), 3);
        // expected order by |b|: 0.5, 0.3, 0.2
        assert_eq!(stats.top_diffs[0].0, 1);
        assert_eq!(stats.top_diffs[1].0, 2);
        assert_eq!(stats.top_diffs[2].0, 4);
    }

    #[test]
    fn run_diff_dim_product_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let p1 = tmp.path().join("a.bin");
        let p2 = tmp.path().join("b.bin");
        write_aprt_file(&p1, 0, &[1.0, 2.0]);
        write_aprt_file(&p2, 0, &[1.0, 2.0, 3.0]);
        let r = run_aprt_stage_diff(&p1, &p2, 5, true);
        assert!(r.is_err(), "expected dim_product mismatch error");
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("dim_product mismatch"), "msg was: {msg}");
    }

    #[test]
    fn run_diff_layer_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let p1 = tmp.path().join("a.bin");
        let p2 = tmp.path().join("b.bin");
        write_aprt_file(&p1, 0, &[1.0, 2.0]);
        write_aprt_file(&p2, 7, &[1.0, 2.0]);
        let r = run_aprt_stage_diff(&p1, &p2, 5, true);
        assert!(r.is_err(), "expected layer mismatch error");
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("layer mismatch"), "msg was: {msg}");
    }

    #[test]
    fn run_diff_identical_files_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let p1 = tmp.path().join("a.bin");
        let p2 = tmp.path().join("b.bin");
        write_aprt_file(&p1, 3, &[1.0, 2.0, 3.0]);
        write_aprt_file(&p2, 3, &[1.0, 2.0, 3.0]);
        let r = run_aprt_stage_diff(&p1, &p2, 5, true);
        assert!(r.is_ok());
    }
}
