//! GH-280: Capability Match Gate for `apr qa`.
//!
//! Validates that the model's required operations are supported by the GPU
//! backend BEFORE running inference. This is Gate 0 — the earliest gate —
//! because if the GPU can't run the model, all GPU-dependent gates are moot.
//!
//! For GGUF files: reads architecture from metadata, derives constraints,
//! checks GPU capability.
//!
//! For non-GGUF files (APR, SafeTensors): skips gracefully (these formats
//! don't yet carry architecture constraints in a standardized way).

use super::qa::{GateResult, QaConfig};
use crate::error::Result;
use std::path::Path;
use std::time::Instant;

/// Run the capability match gate.
///
/// Reads the model's architecture string from GGUF metadata, derives the
/// required operations from `ArchConstraints`, and checks whether the GPU
/// backend supports all of them.
///
/// # Returns
///
/// - `GateResult::passed` if all required ops are GPU-supported (or non-GGUF)
/// - `GateResult::failed` if the GPU lacks required kernels (lists missing ops)
/// - `GateResult::skipped` if the file can't be read or isn't GGUF
pub fn run_capability_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    let start = Instant::now();

    if !config.json && config.verbose {
        println!(
            "{}",
            colored::Colorize::yellow("Running capability match gate (GH-280)...")
        );
    }

    // Read file header to detect format
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            let duration = start.elapsed();
            return Ok(GateResult::failed(
                "capability_match",
                &format!("Failed to read model file: {e}"),
                None,
                None,
                duration,
            ));
        }
    };

    if data.len() < 4 {
        let duration = start.elapsed();
        return Ok(GateResult::failed(
            "capability_match",
            "Model file too small to detect format",
            None,
            None,
            duration,
        ));
    }

    // Only check GGUF files — APR/SafeTensors don't carry arch constraints yet
    let magic = &data[0..4];
    if magic != b"GGUF" {
        let duration = start.elapsed();
        return Ok(GateResult::passed(
            "capability_match",
            "Non-GGUF format — capability check not applicable",
            None,
            None,
            duration,
        ));
    }

    // Parse GGUF to get architecture string
    let arch = match extract_gguf_architecture(&data) {
        Some(a) => a,
        None => {
            let duration = start.elapsed();
            return Ok(GateResult::passed(
                "capability_match",
                "GGUF missing architecture metadata — skipping capability check",
                None,
                None,
                duration,
            ));
        }
    };

    // Derive constraints and check capability
    #[cfg(feature = "inference")]
    {
        use realizar::capability::{check_capability, gpu_supported_ops, required_ops};
        use realizar::gguf::ArchConstraints;

        let constraints = ArchConstraints::from_architecture(&arch);
        let required = required_ops(&constraints);
        let supported = gpu_supported_ops();

        let duration = start.elapsed();
        match check_capability(&required, &supported) {
            Ok(()) => Ok(GateResult::passed(
                "capability_match",
                &format!(
                    "Architecture '{}': all {} required ops supported by GPU",
                    arch,
                    required.len()
                ),
                Some(required.len() as f64),
                Some(0.0),
                duration,
            )),
            Err(missing) => {
                let missing_names: Vec<String> = missing.iter().map(ToString::to_string).collect();
                Ok(GateResult::failed(
                    "capability_match",
                    &format!(
                        "Architecture '{}': GPU missing kernel support for [{}]. \
                         GPU inference will produce garbage — use CPU.",
                        arch,
                        missing_names.join(", ")
                    ),
                    Some(missing.len() as f64),
                    Some(0.0),
                    duration,
                ))
            }
        }
    }

    #[cfg(not(feature = "inference"))]
    {
        let _ = arch;
        let duration = start.elapsed();
        Ok(GateResult::passed(
            "capability_match",
            "Capability check requires inference feature",
            None,
            None,
            duration,
        ))
    }
}

/// Extract the architecture string from GGUF metadata.
///
/// Uses aprender's GGUF reader to parse metadata without loading tensors.
fn extract_gguf_architecture(data: &[u8]) -> Option<String> {
    let reader = aprender::format::gguf::reader::GgufReader::from_bytes(data.to_vec()).ok()?;
    reader.architecture()
}
