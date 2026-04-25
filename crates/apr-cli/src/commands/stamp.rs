//! `apr stamp` — write provenance fields onto an existing APR v2 file.
//!
//! Wraps `aprender::format::v2::stamp_provenance_bytes` (PR #1050) so the
//! shipped MODEL-1 teacher (and any other pre-`GATE-APR-PROV-001..003`
//! `.apr`) can have its `license` / `data_source` / `data_license`
//! populated post-hoc, unblocking SHIP-009 full discharge.
//!
//! Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
//! §v2.52.0 atomic next action (2) "Teacher provenance gap".
//! Helper: `aprender::format::v2::stamp_provenance_bytes`.

use crate::error::{CliError, Result};
use aprender::format::v2::{stamp_provenance_bytes, AprV2Reader, ProvenancePatch};
use std::fs;
use std::path::Path;

/// Run the stamp command — read input `.apr`, patch the three provenance
/// fields if any are provided, write to output, then verify by re-reading.
///
/// At least one of `license` / `data_source` / `data_license` must be
/// `Some(...)`; the helper rejects an empty patch on its own, but we
/// also surface a clear CLI error message to keep the failure mode
/// human-readable.
pub(crate) fn run(
    file: &Path,
    license: Option<&str>,
    data_source: Option<&str>,
    data_license: Option<&str>,
    output: &Path,
    force: bool,
    json_output: bool,
) -> Result<()> {
    if license.is_none() && data_source.is_none() && data_license.is_none() {
        return Err(CliError::ValidationFailed(
            "apr stamp: at least one of --license, --data-source, --data-license \
             must be specified — refusing to rewrite without changes"
                .to_string(),
        ));
    }

    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }
    if output.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output.display()
        )));
    }

    if !json_output {
        eprintln!("Reading {}", file.display());
    }
    let input = fs::read(file)
        .map_err(|e| CliError::ValidationFailed(format!("read failed: {e}")))?;

    let patch = ProvenancePatch {
        license: license.map(str::to_string),
        data_source: data_source.map(str::to_string),
        data_license: data_license.map(str::to_string),
    };

    let stamped = stamp_provenance_bytes(&input, &patch)
        .map_err(|e| CliError::ValidationFailed(format!("stamp failed: {e:?}")))?;

    fs::write(output, &stamped)
        .map_err(|e| CliError::ValidationFailed(format!("write failed: {e}")))?;

    // Re-read to confirm round-trip succeeded — a stamp that produces a
    // file that doesn't parse back is a hard ship-blocker, fail fast.
    let verify_reader = AprV2Reader::from_bytes(&stamped)
        .map_err(|e| CliError::ValidationFailed(format!("post-stamp verify failed: {e:?}")))?;

    if json_output {
        let summary = serde_json::json!({
            "command":      "stamp",
            "input":        file.display().to_string(),
            "output":       output.display().to_string(),
            "input_bytes":  input.len(),
            "output_bytes": stamped.len(),
            "tensor_count": verify_reader.tensor_names().len(),
            "stamped":      {
                "license":      verify_reader.metadata().license,
                "data_source":  verify_reader.metadata().data_source,
                "data_license": verify_reader.metadata().data_license,
            },
            "header_flags_bits": verify_reader.header().flags.bits(),
        });
        println!("{}", serde_json::to_string_pretty(&summary).unwrap_or_default());
    } else {
        println!(
            "✓ Stamped {} → {} ({} tensors, {} → {} bytes)",
            file.display(),
            output.display(),
            verify_reader.tensor_names().len(),
            input.len(),
            stamped.len(),
        );
        println!("  license:      {:?}", verify_reader.metadata().license);
        println!("  data_source:  {:?}", verify_reader.metadata().data_source);
        println!("  data_license: {:?}", verify_reader.metadata().data_license);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};
    use tempfile::TempDir;

    /// Build a minimal valid APR v2 file at `path` with no provenance fields set.
    fn write_unpopulated_apr(path: &Path) {
        let metadata = AprV2Metadata::new("stamp-cli-test");
        let mut writer = AprV2Writer::new(metadata);
        writer.add_tensor(
            "weight",
            TensorDType::F32,
            vec![2, 3],
            vec![0u8; 24],
        );
        let bytes = writer.write().expect("write test apr");
        fs::write(path, &bytes).expect("write test apr to disk");
    }

    #[test]
    fn stamp_cli_populates_all_three_fields() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);

        let result = run(
            &input,
            Some("Apache-2.0"),
            Some("huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct"),
            Some("Apache-2.0"),
            &output,
            false,
            true, // json_output to keep stdout structured
        );
        assert!(result.is_ok(), "stamp run must succeed: {result:?}");

        let bytes = fs::read(&output).unwrap();
        let reader = AprV2Reader::from_bytes(&bytes).unwrap();
        assert_eq!(reader.metadata().license.as_deref(), Some("Apache-2.0"));
        assert_eq!(
            reader.metadata().data_source.as_deref(),
            Some("huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct")
        );
        assert_eq!(
            reader.metadata().data_license.as_deref(),
            Some("Apache-2.0")
        );
    }

    #[test]
    fn stamp_cli_rejects_empty_patch() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);

        let result = run(&input, None, None, None, &output, false, true);
        let err = result.unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("at least one"),
            "empty-patch CLI error must be explicit: {msg}"
        );
        // Output file must NOT have been written.
        assert!(
            !output.exists(),
            "rejected stamp must not create the output file"
        );
    }

    #[test]
    fn stamp_cli_rejects_missing_input() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("does-not-exist.apr");
        let output = dir.path().join("output.apr");

        let result = run(
            &input,
            Some("Apache-2.0"),
            None,
            None,
            &output,
            false,
            true,
        );
        let err = result.unwrap_err();
        // CliError::FileNotFound — exact variant, not just substring match.
        assert!(
            matches!(err, CliError::FileNotFound(_)),
            "missing-input must surface FileNotFound, got: {err:?}"
        );
    }

    #[test]
    fn stamp_cli_rejects_existing_output_without_force() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);
        fs::write(&output, b"pre-existing").unwrap();

        let result = run(
            &input,
            Some("Apache-2.0"),
            None,
            None,
            &output,
            false, // force=false
            true,
        );
        let err = result.unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("already exists") && msg.contains("--force"),
            "existing-output error must mention --force: {msg}"
        );
        // Pre-existing content must be untouched.
        let still_there = fs::read(&output).unwrap();
        assert_eq!(still_there, b"pre-existing");
    }

    #[test]
    fn stamp_cli_overwrites_existing_output_with_force() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);
        fs::write(&output, b"pre-existing").unwrap();

        let result = run(
            &input,
            Some("MIT"),
            None,
            None,
            &output,
            true, // force=true
            true,
        );
        assert!(result.is_ok(), "stamp with --force must succeed: {result:?}");

        // Output must now be a valid APR file with the patched license.
        let bytes = fs::read(&output).unwrap();
        let reader = AprV2Reader::from_bytes(&bytes).expect("force-overwritten file must parse");
        assert_eq!(reader.metadata().license.as_deref(), Some("MIT"));
    }
}
