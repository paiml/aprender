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
/// The provenance license fields are the same ones `apr validate-manifest`
/// enforces SPDX on (FALSIFY-PM-004). Validate BEFORE writing: a bad identifier
/// baked into an artifact is only discovered at publish time, by which point the
/// .apr is already distributed (issue #2391).
///
/// Extracted rather than inlined into `run`: as a nested `for`/`if let`/`if let`
/// it pushed `run`'s cognitive complexity from 19 to 29, over the repo's
/// threshold of 25.
fn reject_non_spdx_licenses(license: Option<&str>, data_license: Option<&str>) -> Result<()> {
    for (flag, value) in [("--license", license), ("--data-license", data_license)] {
        let Some(v) = value else { continue };
        if let Some(why) = crate::commands::spdx::reject_reason(flag, v) {
            return Err(CliError::ValidationFailed(format!("apr stamp: {why}")));
        }
    }
    Ok(())
}

pub(crate) fn run(
    file: &Path,
    license: Option<&str>,
    data_source: Option<&str>,
    data_license: Option<&str>,
    hf_architecture: Option<&str>,
    hf_model_type: Option<&str>,
    architecture: Option<&str>,
    tokenizer_dir: Option<&Path>,
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
        && tokenizer_dir.is_none()
    {
        return Err(CliError::ValidationFailed(
            "apr stamp: at least one of --license, --data-source, --data-license, \
             --hf-architecture, --hf-model-type, --architecture, --tokenizer must \
             be specified — refusing to rewrite without changes"
                .to_string(),
        ));
    }

    reject_non_spdx_licenses(license, data_license)?;

    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }
    if output.exists() && !force {
        return Err(CliError::ValidationFailed(format!(
            "Output file '{}' already exists. Use --force to overwrite.",
            output.display()
        )));
    }

    // PMAT-690 P3-C-prep defect 1 (2026-05-17): load tokenizer files if
    // --tokenizer <DIR> is provided. Supports vocab.json + merges.txt
    // (HF GPT-2/BPE format) and tokenizer.json (HF unified format).
    let (tok_vocab, tok_merges, tok_model_type) = if let Some(dir) = tokenizer_dir {
        load_tokenizer_files(dir)?
    } else {
        (None, None, None)
    };

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
        tokenizer_vocab: tok_vocab,
        tokenizer_merges: tok_merges,
        tokenizer_model_type: tok_model_type,
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
        // `{:?}` on Option<String> printed `Some("Apache-2.0")` at users.
        let m = verify_reader.metadata();
        println!("  license:         {}", show(m.license.as_deref()));
        println!("  data_source:     {}", show(m.data_source.as_deref()));
        println!("  data_license:    {}", show(m.data_license.as_deref()));
        println!("  hf_architecture: {}", show(m.hf_architecture.as_deref()));
        println!("  hf_model_type:   {}", show(m.hf_model_type.as_deref()));
        println!("  architecture:    {}", show(m.architecture.as_deref()));
    }

    Ok(())
}

/// Render an optional stamped field: the value, or `(unset)`.
fn show(v: Option<&str>) -> String {
    v.map_or_else(|| "(unset)".to_string(), ToString::to_string)
}

/// PMAT-690 P3-C-prep defect 1: load tokenizer files from a directory.
///
/// Accepts two input shapes:
///
/// - **`<dir>/vocab.json` + `<dir>/merges.txt`** — HF GPT-2 / Qwen BPE
///   format. `vocab.json` is `{"token": id, ...}` mapping; we sort by
///   id and extract the string array. `merges.txt` is one merge per
///   line (e.g. `"Ä t"`); we skip the optional `#version:` header.
/// - **`<dir>/tokenizer.json`** — HF unified format. Parsed via the
///   same path `apr convert` uses (`load_tokenizer_from_explicit_path`
///   in aprender-core). Returns vocab + merges from the unified JSON.
///
/// Returns `(vocab, merges, model_type)`. `model_type` is `Some("BPE")`
/// when merges are present (the typical case for Qwen / GPT-2 family).
///
/// # Errors
///
/// Returns Err when the directory has neither a `tokenizer.json` nor
/// a `vocab.json + merges.txt` pair. Empty vocabulary is also an error
/// — operator explicitly requested embedded tokenizer.
fn load_tokenizer_files(
    dir: &Path,
) -> Result<(Option<Vec<String>>, Option<Vec<String>>, Option<String>)> {
    if !dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: {} is not a directory",
            dir.display()
        )));
    }

    // Preferred: unified tokenizer.json (reuse aprender-core's loader)
    let unified = dir.join("tokenizer.json");
    if unified.is_file() {
        return load_unified_tokenizer(&unified);
    }

    // Fallback: vocab.json + merges.txt (the Qwen-coder pretrain default)
    let vocab_path = dir.join("vocab.json");
    let merges_path = dir.join("merges.txt");
    if !vocab_path.is_file() {
        return Err(CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: neither tokenizer.json nor vocab.json found in {}",
            dir.display()
        )));
    }
    let vocab_str = fs::read_to_string(&vocab_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: read vocab.json failed: {e}"
        ))
    })?;
    let vocab_map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&vocab_str)
        .map_err(|e| {
            CliError::ValidationFailed(format!(
                "apr stamp --tokenizer: vocab.json is not a valid JSON object: {e}"
            ))
        })?;
    // Sort by id (the value) to produce a position-indexed vector
    let mut pairs: Vec<(u64, String)> = vocab_map
        .iter()
        .filter_map(|(tok, id)| id.as_u64().map(|n| (n, tok.clone())))
        .collect();
    pairs.sort_by_key(|(id, _)| *id);
    let vocab: Vec<String> = pairs.into_iter().map(|(_, tok)| tok).collect();
    if vocab.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: vocab.json in {} has no entries",
            dir.display()
        )));
    }

    // merges.txt: one merge per line; skip "#version: ..." header if present
    let merges: Option<Vec<String>> = if merges_path.is_file() {
        let merges_str = fs::read_to_string(&merges_path).map_err(|e| {
            CliError::ValidationFailed(format!(
                "apr stamp --tokenizer: read merges.txt failed: {e}"
            ))
        })?;
        let m: Vec<String> = merges_str
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_string)
            .collect();
        if m.is_empty() {
            None
        } else {
            Some(m)
        }
    } else {
        None
    };

    let model_type = if merges.is_some() {
        Some("BPE".to_string())
    } else {
        None
    };
    Ok((Some(vocab), merges, model_type))
}

/// PMAT-690 P3-C-prep defect 1: parse a unified `tokenizer.json` file.
/// The format embeds vocab + merges in one object — extract both.
fn load_unified_tokenizer(
    path: &Path,
) -> Result<(Option<Vec<String>>, Option<Vec<String>>, Option<String>)> {
    let content = fs::read_to_string(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: read {} failed: {e}",
            path.display()
        ))
    })?;
    let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: {} is not valid JSON: {e}",
            path.display()
        ))
    })?;
    let model = json.get("model").ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: {} missing `model` field",
            path.display()
        ))
    })?;
    let model_type = model
        .get("type")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);
    let vocab_obj = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "apr stamp --tokenizer: {} missing `model.vocab`",
                path.display()
            ))
        })?;
    let mut pairs: Vec<(u64, String)> = vocab_obj
        .iter()
        .filter_map(|(tok, id)| id.as_u64().map(|n| (n, tok.clone())))
        .collect();
    pairs.sort_by_key(|(id, _)| *id);
    let vocab: Vec<String> = pairs.into_iter().map(|(_, tok)| tok).collect();
    if vocab.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "apr stamp --tokenizer: {} has empty vocab",
            path.display()
        )));
    }
    // merges may be either ["a b", "c d"] (string form) or [["a","b"],["c","d"]] (array form)
    let merges: Option<Vec<String>> = model.get("merges").and_then(|v| v.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|m| match m {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Array(parts) if parts.len() == 2 => {
                    let a = parts[0].as_str()?;
                    let b = parts[1].as_str()?;
                    Some(format!("{a} {b}"))
                }
                _ => None,
            })
            .collect()
    });
    let merges = merges.filter(|m| !m.is_empty());
    Ok((Some(vocab), merges, model_type))
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
            None, // tokenizer_dir
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

    /// FALSIFY-STAMP-SPDX-001 — `apr stamp` must not bake a license into an
    /// artifact that `apr validate-manifest` (FALSIFY-PM-004) then rejects.
    /// 0.63.0 accepted `NOT-A-LICENSE-!!` and exited 0 (issue #2391).
    #[test]
    fn stamp_rejects_non_spdx_license_and_writes_nothing() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);

        for bad in ["NOT-A-LICENSE-!!", "Apache2", "MIT License", "whatever"] {
            let err = run(
                &input,
                Some(bad),
                None,
                None,
                None,
                None,
                None,
                None,
                &output,
                true,
                true,
            )
            .expect_err(&format!("--license {bad:?} must be rejected"));
            assert!(
                format!("{err}").contains("SPDX"),
                "rejection must say why: {err}"
            );
            assert!(
                !output.exists(),
                "a rejected --license {bad:?} still wrote {}",
                output.display()
            );
        }
    }

    #[test]
    fn stamp_rejects_non_spdx_data_license() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);
        let err = run(
            &input,
            None,
            None,
            Some("NOT-A-LICENSE"),
            None,
            None,
            None,
            None,
            &output,
            true,
            true,
        )
        .expect_err("--data-license must be validated too");
        assert!(format!("{err}").contains("SPDX"), "got: {err}");
    }

    /// Control: the identifiers `validate-manifest` accepts must still stamp,
    /// including the lower-case spelling the Hub uses.
    #[test]
    fn stamp_still_accepts_every_valid_spdx_identifier() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        write_unpopulated_apr(&input);
        for ok in ["Apache-2.0", "mit", "CC-BY-4.0", "llama3.1"] {
            let output = dir.path().join(format!("out-{ok}.apr"));
            let r = run(
                &input,
                Some(ok),
                None,
                None,
                None,
                None,
                None,
                None,
                &output,
                true,
                true,
            );
            assert!(r.is_ok(), "{ok} must still be stampable: {r:?}");
        }
    }

    /// The report line printed `license: Some("Apache-2.0")` — Rust Debug on
    /// `Option<String>` reaching a user-facing report.
    #[test]
    fn show_renders_value_or_unset_without_debug_option() {
        assert_eq!(show(Some("Apache-2.0")), "Apache-2.0");
        assert_eq!(show(None), "(unset)");
        assert!(!show(Some("MIT")).contains("Some("));
    }

    #[test]
    fn stamp_cli_rejects_empty_patch() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);

        let result = run(
            &input, None, None, None, None, None, None, None, &output, false, true,
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
            None, // tokenizer_dir
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
            None, // tokenizer_dir
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
            None, // tokenizer_dir
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
            None, // tokenizer_dir
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

    /// PMAT-690 P3-C-prep defect 1: --tokenizer <DIR> with vocab.json
    /// + merges.txt embeds the vocab + merges into AprV2Metadata.custom
    /// AND sets the HAS_VOCAB header flag. Required for `apr run` to
    /// accept the stamped APR for inference (the §86 publish-readiness
    /// preflight surfaced this gap on P2-E ep49).
    #[test]
    fn stamp_p3c_defect1_embeds_tokenizer_from_vocab_merges() {
        use aprender::format::v2::AprV2Flags;
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);

        // Stage a vocab.json + merges.txt pair (Qwen-coder pretrain format)
        let tok_dir = dir.path().join("tokenizer");
        fs::create_dir_all(&tok_dir).unwrap();
        // vocab.json: {"<unk>": 0, "Ġ": 1, "the": 2}
        let vocab_json = r#"{"<unk>": 0, "Ġ": 1, "the": 2}"#;
        fs::write(tok_dir.join("vocab.json"), vocab_json).unwrap();
        // merges.txt
        fs::write(
            tok_dir.join("merges.txt"),
            "#version: 0.2\nĠ t\nh e\nĠt he\n",
        )
        .unwrap();

        let result = run(
            &input,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&tok_dir),
            &output,
            false,
            true,
        );
        assert!(
            result.is_ok(),
            "stamp with --tokenizer must succeed: {result:?}"
        );

        let bytes = fs::read(&output).unwrap();
        let reader = AprV2Reader::from_bytes(&bytes).unwrap();

        // HAS_VOCAB flag MUST be set (the apr run gate per PMAT-172)
        assert!(
            reader.header().flags.contains(AprV2Flags::HAS_VOCAB),
            "HAS_VOCAB header flag must be set after --tokenizer stamp \
             (the load-bearing check for apr run inference)"
        );

        // custom.tokenizer.vocabulary present + has 3 entries
        let vocab = reader
            .metadata()
            .custom
            .get("tokenizer.vocabulary")
            .and_then(|v| v.as_array())
            .expect("tokenizer.vocabulary must be set");
        assert_eq!(vocab.len(), 3);
        // Sorted by id: <unk>=0, Ġ=1, the=2
        assert_eq!(vocab[0].as_str(), Some("<unk>"));
        assert_eq!(vocab[2].as_str(), Some("the"));

        // custom.tokenizer.merges present + has 3 entries (header skipped)
        let merges = reader
            .metadata()
            .custom
            .get("tokenizer.merges")
            .and_then(|v| v.as_array())
            .expect("tokenizer.merges must be set");
        assert_eq!(merges.len(), 3);

        // model_type=BPE inferred from presence of merges
        assert_eq!(
            reader
                .metadata()
                .custom
                .get("tokenizer.model_type")
                .and_then(|v| v.as_str()),
            Some("BPE")
        );
    }

    /// PMAT-690 P3-C-prep defect 1: --tokenizer flag alone (no other
    /// patches) must satisfy the has_any() gate.
    #[test]
    fn stamp_p3c_defect1_tokenizer_alone_passes_has_any_gate() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);
        let tok_dir = dir.path().join("tokenizer");
        fs::create_dir_all(&tok_dir).unwrap();
        fs::write(tok_dir.join("vocab.json"), r#"{"a": 0}"#).unwrap();

        let result = run(
            &input,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&tok_dir),
            &output,
            false,
            true,
        );
        assert!(
            result.is_ok(),
            "stamp with --tokenizer alone must succeed: {result:?}"
        );
    }

    /// PMAT-690 P3-C-prep defect 1: --tokenizer with neither
    /// tokenizer.json nor vocab.json present → clear error.
    #[test]
    fn stamp_p3c_defect1_tokenizer_dir_without_files_errors() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("input.apr");
        let output = dir.path().join("output.apr");
        write_unpopulated_apr(&input);
        let empty_tok = dir.path().join("empty-tokenizer");
        fs::create_dir_all(&empty_tok).unwrap();

        let result = run(
            &input,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(&empty_tok),
            &output,
            false,
            true,
        );
        let err = result.unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("neither tokenizer.json nor vocab.json found"),
            "expected clear missing-files error, got: {msg}"
        );
    }
}
