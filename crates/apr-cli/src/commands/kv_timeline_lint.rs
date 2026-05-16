//! `apr kv-timeline-lint` — CRUX-F-06 KV-cache utilization timeline gate.
//!
//! Reads an already-captured `apr profile --kv-timeline --json` body and
//! dispatches the pure classifiers in `kv_timeline_classifier`. Exits
//! non-zero on any failure.
//!
//! Spec: `contracts/crux-F-06-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::kv_timeline_classifier::{
    classify_block_conservation, classify_peak_consistency, classify_preemption_trigger,
    classify_schema, classify_used_pct_arithmetic, KvBlockConservationOutcome, KvPeakOutcome,
    KvPreemptionOutcome, KvSchemaOutcome, KvUsedPctOutcome, F06_DEFAULT_PREEMPT_THRESHOLD,
};
use crate::error::{CliError, Result};

pub(crate) fn run(timeline_file: &Path, preempt_threshold: f64, json: bool) -> Result<()> {
    if !timeline_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(timeline_file)));
    }
    let body_text = std::fs::read_to_string(timeline_file)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr kv-timeline-lint: failed to parse JSON from {}: {e}",
            timeline_file.display()
        ))
    })?;

    let schema = classify_schema(&body);
    // Downstream classifiers assume the schema gate passed; if it failed, we
    // still surface the schema outcome but skip the dependent gates.
    let (block, used_pct, peak, preempt) = if matches!(schema, KvSchemaOutcome::Ok) {
        (
            classify_block_conservation(&body),
            classify_used_pct_arithmetic(&body),
            classify_peak_consistency(&body),
            classify_preemption_trigger(&body, preempt_threshold),
        )
    } else {
        (
            KvBlockConservationOutcome::Ok,
            KvUsedPctOutcome::Ok,
            KvPeakOutcome::Ok,
            KvPreemptionOutcome::Ok,
        )
    };

    print_report(
        timeline_file,
        &schema,
        &block,
        &used_pct,
        &peak,
        &preempt,
        json,
    );

    if !matches!(schema, KvSchemaOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "kv-timeline-lint schema gate rejected body: {schema:?}"
        )));
    }
    if !matches!(block, KvBlockConservationOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "kv-timeline-lint block-conservation gate rejected body: {block:?}"
        )));
    }
    if !matches!(used_pct, KvUsedPctOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "kv-timeline-lint used-pct gate rejected body: {used_pct:?}"
        )));
    }
    if !matches!(peak, KvPeakOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "kv-timeline-lint peak-consistency gate rejected body: {peak:?}"
        )));
    }
    if !matches!(preempt, KvPreemptionOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "kv-timeline-lint preemption-trigger gate rejected body: {preempt:?}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &Path,
    schema: &KvSchemaOutcome,
    block: &KvBlockConservationOutcome,
    used_pct: &KvUsedPctOutcome,
    peak: &KvPeakOutcome,
    preempt: &KvPreemptionOutcome,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "schema": format!("{schema:?}"),
            "block_conservation": format!("{block:?}"),
            "used_pct_arithmetic": format!("{used_pct:?}"),
            "peak_consistency": format!("{peak:?}"),
            "preemption_trigger": format!("{preempt:?}"),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("kv-timeline-lint report for {}", path.display());
    println!("  schema             : {schema:?}");
    println!("  block_conservation : {block:?}");
    println!("  used_pct_arithmetic: {used_pct:?}");
    println!("  peak_consistency   : {peak:?}");
    println!("  preemption_trigger : {preempt:?}");
}

/// Expose the default for tests that want the canonical vLLM threshold.
pub const KV_TIMELINE_DEFAULT_THRESHOLD: f64 = F06_DEFAULT_PREEMPT_THRESHOLD;
