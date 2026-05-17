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
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    file: &Path,
    license: Option<&str>,
    data_source: Option<&str>,
    data_license: Option<&str>,
    hf_architecture: Option<&str>,
    hf_model_type: Option<&str>,
    architecture: Option<&str>,
    output: &Path,
    force: bool,
    json_output: bool,
) -> Result<()> {
    if license.is_none()
        && data_source.is_none()
        && data_license.is_none()
        && hf_architecture.is_none()
        && hf_model_type.is_none()
        && architecture.is_none()
    {
        return Err(CliError::ValidationFailed(
            "apr stamp: at least one of --license, --data-source, --data-license, \
             --hf-architecture, --hf-model-type, --architecture must be specified \
             — refusing to rewrite without changes"
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
    let input =
        fs::read(file).map_err(|e| CliError::ValidationFailed(format!("read failed: {e}")))?;

    let patch = ProvenancePatch {
        license: license.map(str::to_string),
        data_source: data_source.map(str::to_string),
        data_license: data_license.map(str::to_string),
        hf_architecture: hf_architecture.map(str::to_string),
        hf_model_type: hf_model_type.map(str::to_string),
        architecture: architecture.map(str::to_string),
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
                "license":         verify_reader.metadata().license,
                "data_source":     verify_reader.metadata().data_source,
                "data_license":    verify_reader.metadata().data_license,
                "hf_architecture": verify_reader.metadata().hf_architecture,
                "hf_model_type":   verify_reader.metadata().hf_model_type,
                "architecture":    verify_reader.metadata().architecture,
            },
            "header_flags_bits": verify_reader.header().flags.bits(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).unwrap_or_default()
        );
    } else {
        println!(
            "✓ Stamped {} → {} ({} tensors, {} → {} bytes)",
            file.display(),
            output.display(),
            verify_reader.tensor_names().len(),
            input.len(),
            stamped.len(),
        );
        println!("  license:         {:?}", verify_reader.metadata().license);
        println!(
            "  data_source:     {:?}",
            verify_reader.metadata().data_source
        );
        println!(
            "  data_license:    {:?}",
            verify_reader.metadata().data_license
        );
        println!(
            "  hf_architecture: {:?}",
            verify_reader.metadata().hf_architecture
        );
        println!(
            "  hf_model_type:   {:?}",
            verify_reader.metadata().hf_model_type
        );
        println!(
            "  architecture:    {:?}",
            verify_reader.metadata().architecture
        );
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
        writer.add_tensor("weight", TensorDType::F32, vec![2, 3], vec![0u8; 24]);
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
            None, // hf_architecture
            None, // hf_model_type
            None, // architecture
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

        let result = run(
            &input, None, None, None, None, None, None, &output, false, true,
        );
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
            None,
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
            None, // hf_architecture
            None, // hf_model_type
            None, // architecture
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
            None, // hf_architecture
            None, // hf_model_type
            None, // architecture
            &output,
            true, // force=true
            true,
        );
        assert!(
            result.is_ok(),
            "stamp with --force must succeed: {result:?}"
        );

        // Output must now be a valid APR file with the patched license.
        let bytes = fs::read(&output).unwrap();
        let reader = AprV2Reader::from_bytes(&bytes).expect("force-overwritten file must parse");
        assert_eq!(reader.metadata().license.as_deref(), Some("MIT"));
    }

    // ========================================================================
    // PMAT-690 P0-K extension (SPEC §86) — HF identity + architecture
    // family stamping for in-place pre-P0-K APR salvage
    // ========================================================================

    /// SPEC §86 use case: a pre-P0-K APR has `architecture = "LlamaForCausalLM"`
    /// (the P0-H fallback) but the actual tensors are Qwen2. `apr stamp
    /// --hf-architecture Qwen2ForCausalLM --hf-model-type qwen2 --architecture qwen2`
    /// MUST patch all three fields so the resulting APR is loadable as a
    /// proper Qwen2 init for `apr pretrain --init`.
    #[test]
    fn stamp_p0k_recovers_pre_p0k_apr_identity() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        // Pre-P0-K state: arch=LlamaForCausalLM (wrong), no hf_architecture.
        let metadata = AprV2Metadata {
            architecture: Some("LlamaForCausalLM".to_string()),
            hf_architecture: None,
            hf_model_type: None,
            ..AprV2Metadata::new("p0k-stamp-test")
        };
        let mut writer = AprV2Writer::new(metadata);
        writer.add_tensor(
            "model.embed_tokens.weight",
            TensorDType::F32,
            vec![128, 64],
            vec![0u8; 128 * 64 * 4],
        );
        let bytes = writer.write().expect("write pre-P0-K test apr");
        fs::write(&input, &bytes).expect("write test apr to disk");

        let result = run(
            &input,
            None,
            None,
            None,
            Some("Qwen2ForCausalLM"),
            Some("qwen2"),
            Some("qwen2"),
            &output,
            false,
            true,
        );
        assert!(result.is_ok(), "stamp run must succeed: {result:?}");

        let out_bytes = fs::read(&output).unwrap();
        let reader = AprV2Reader::from_bytes(&out_bytes).unwrap();
        assert_eq!(
            reader.metadata().hf_architecture.as_deref(),
            Some("Qwen2ForCausalLM"),
            "hf_architecture must be patched"
        );
        assert_eq!(
            reader.metadata().hf_model_type.as_deref(),
            Some("qwen2"),
            "hf_model_type must be patched"
        );
        assert_eq!(
            reader.metadata().architecture.as_deref(),
            Some("qwen2"),
            "architecture (family slug) must be patched away from the wrong P0-H fallback"
        );
    }

    /// SPEC §86 partial stamp: an operator who only knows the HF class
    /// name can patch hf_architecture alone without touching the family slug.
    /// Verifies the stamp is field-independent.
    #[test]
    fn stamp_p0k_partial_hf_architecture_only() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);

        let result = run(
            &input,
            None,
            None,
            None,
            Some("Qwen2ForCausalLM"),
            None,
            None,
            &output,
            false,
            true,
        );
        assert!(result.is_ok(), "partial stamp must succeed: {result:?}");

        let out_bytes = fs::read(&output).unwrap();
        let reader = AprV2Reader::from_bytes(&out_bytes).unwrap();
        assert_eq!(
            reader.metadata().hf_architecture.as_deref(),
            Some("Qwen2ForCausalLM")
        );
        assert_eq!(
            reader.metadata().hf_model_type,
            None,
            "unpatched field must remain None"
        );
    }
}
