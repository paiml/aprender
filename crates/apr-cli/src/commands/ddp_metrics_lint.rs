//! `apr ddp-metrics-lint` — CRUX-D-11 DDP multi-GPU metrics gate.
//!
//! Reads two captured `apr finetune --parallel ddp --json` outputs (N=1 and
//! N=k) and dispatches the pure classifiers in `ddp_metrics_classifier`.
//! Exits non-zero on any failure.
//!
//! Spec: `contracts/crux-D-11-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::ddp_metrics_classifier::{
    classify_allreduce_bandwidth, classify_loss_parity, classify_scaling_efficiency,
    DdpAllreduceOutcome, DdpLossParityOutcome, DdpScalingOutcome, D11_DEFAULT_LOSS_TOLERANCE,
    D11_DEFAULT_SCALING_FLOOR,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    metrics_1gpu_file: &Path,
    metrics_ngpu_file: &Path,
    world_size: i64,
    scaling_floor: f64,
    loss_tolerance: f64,
    json: bool,
) -> Result<()> {
    let body_1 = load_json(metrics_1gpu_file)?;
    let body_n = load_json(metrics_ngpu_file)?;

    let scaling = classify_scaling_efficiency(&body_1, &body_n, world_size, scaling_floor);
    let parity = classify_loss_parity(&body_1, &body_n, loss_tolerance);
    let allreduce = classify_allreduce_bandwidth(&body_n);

    print_report(
        metrics_1gpu_file,
        metrics_ngpu_file,
        &scaling,
        &parity,
        &allreduce,
        json,
    );

    if !matches!(scaling, DdpScalingOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "ddp-metrics-lint scaling-efficiency gate rejected: {scaling:?}"
        )));
    }
    if !matches!(parity, DdpLossParityOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "ddp-metrics-lint loss-parity gate rejected: {parity:?}"
        )));
    }
    if !matches!(allreduce, DdpAllreduceOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "ddp-metrics-lint allreduce-bandwidth gate rejected: {allreduce:?}"
        )));
    }
    Ok(())
}

fn load_json(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(path)));
    }
    let body_text = std::fs::read_to_string(path)?;
    serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr ddp-metrics-lint: failed to parse JSON from {}: {e}",
            path.display()
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    file1: &Path,
    file_n: &Path,
    scaling: &DdpScalingOutcome,
    parity: &DdpLossParityOutcome,
    allreduce: &DdpAllreduceOutcome,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "metrics_1gpu_file": file1.display().to_string(),
            "metrics_ngpu_file": file_n.display().to_string(),
            "scaling_efficiency": format!("{scaling:?}"),
            "loss_parity": format!("{parity:?}"),
            "allreduce_bandwidth": format!("{allreduce:?}"),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("ddp-metrics-lint report");
    println!("  metrics_1gpu_file   : {}", file1.display());
    println!("  metrics_ngpu_file   : {}", file_n.display());
    println!("  scaling_efficiency  : {scaling:?}");
    println!("  loss_parity         : {parity:?}");
    println!("  allreduce_bandwidth : {allreduce:?}");
}

/// Default constants re-exported so CLI defaults stay in sync with the
/// classifier module.
pub const DDP_METRICS_DEFAULT_SCALING_FLOOR: f64 = D11_DEFAULT_SCALING_FLOOR;
pub const DDP_METRICS_DEFAULT_LOSS_TOLERANCE: f64 = D11_DEFAULT_LOSS_TOLERANCE;

#[cfg(test)]
mod cov_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    fn w(s: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }
    #[test]
    fn missing_first_file_is_file_not_found() {
        let b = w("{}");
        let err = run(
            Path::new("/no/such/1gpu.json"),
            b.path(),
            2,
            DDP_METRICS_DEFAULT_SCALING_FLOOR,
            DDP_METRICS_DEFAULT_LOSS_TOLERANCE,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn missing_second_file_is_file_not_found() {
        let a = w("{}");
        let err = run(
            a.path(),
            Path::new("/no/such/ngpu.json"),
            2,
            DDP_METRICS_DEFAULT_SCALING_FLOOR,
            DDP_METRICS_DEFAULT_LOSS_TOLERANCE,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn invalid_json_is_invalid_format() {
        let a = w("xx");
        let b = w("{}");
        let err = run(
            a.path(),
            b.path(),
            2,
            DDP_METRICS_DEFAULT_SCALING_FLOOR,
            DDP_METRICS_DEFAULT_LOSS_TOLERANCE,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }
}
