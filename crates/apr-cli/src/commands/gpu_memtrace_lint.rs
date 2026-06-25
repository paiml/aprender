//! `apr gpu-memtrace-lint` — CRUX-F-07 GPU memory Chrome-Trace gate.
//!
//! Reads an already-captured `apr profile --gpu-memory-trace` output (or any
//! Chrome Trace Event Format JSON describing GPU alloc/free events) and
//! dispatches the pure classifiers in `gpu_memtrace_classifier`. Exits
//! non-zero on any failure.
//!
//! Spec: `contracts/crux-F-07-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::gpu_memtrace_classifier::{
    classify_alloc_free_pairing, classify_monotonic_timestamps, classify_schema,
    AllocFreePairingOutcome, ChromeTraceSchemaOutcome, MonotonicTimestampsOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(trace_file: &Path, json: bool) -> Result<()> {
    if !trace_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(trace_file)));
    }
    let body_text = std::fs::read_to_string(trace_file)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr gpu-memtrace-lint: failed to parse JSON from {}: {e}",
            trace_file.display()
        ))
    })?;

    let schema = classify_schema(&body);
    let (pairing, ts) = if matches!(schema, ChromeTraceSchemaOutcome::Ok { .. }) {
        (
            classify_alloc_free_pairing(&body),
            classify_monotonic_timestamps(&body),
        )
    } else {
        (AllocFreePairingOutcome::Ok, MonotonicTimestampsOutcome::Ok)
    };

    print_report(trace_file, &schema, &pairing, &ts, json);

    if !matches!(schema, ChromeTraceSchemaOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "gpu-memtrace-lint schema gate rejected body: {schema:?}"
        )));
    }
    if !matches!(pairing, AllocFreePairingOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "gpu-memtrace-lint alloc/free-pairing gate rejected body: {pairing:?}"
        )));
    }
    if !matches!(ts, MonotonicTimestampsOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "gpu-memtrace-lint monotonic-timestamps gate rejected body: {ts:?}"
        )));
    }
    Ok(())
}

fn print_report(
    path: &Path,
    schema: &ChromeTraceSchemaOutcome,
    pairing: &AllocFreePairingOutcome,
    ts: &MonotonicTimestampsOutcome,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "schema": format!("{schema:?}"),
            "alloc_free_pairing": format!("{pairing:?}"),
            "monotonic_timestamps": format!("{ts:?}"),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("gpu-memtrace-lint report for {}", path.display());
    println!("  schema              : {schema:?}");
    println!("  alloc_free_pairing  : {pairing:?}");
    println!("  monotonic_timestamps: {ts:?}");
}

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
    fn missing_file_is_file_not_found() {
        let err = run(Path::new("/no/such/gpu.json"), false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn invalid_json_is_invalid_format() {
        let f = w("nope");
        let err = run(f.path(), false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }
    #[test]
    fn empty_object_runs() {
        let f = w("{}");
        let _ = run(f.path(), true);
    }
}
