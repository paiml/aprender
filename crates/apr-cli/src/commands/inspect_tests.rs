// PMAT-540 Phase 5: Tests for inspect command helper functions

#[cfg(test)]
mod inspect_tests {
    use super::*;
    use std::path::Path;

    // ========================================================================
    // validate_path
    // ========================================================================

    #[test]
    fn validate_path_nonexistent() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::FileNotFound(_) => {}
            e => panic!("Expected FileNotFound, got {e:?}"),
        }
    }

    #[test]
    fn validate_path_directory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::NotAFile(_) => {}
            e => panic!("Expected NotAFile, got {e:?}"),
        }
    }

    #[test]
    fn validate_path_valid_file() {
        let file = tempfile::NamedTempFile::new().expect("create temp file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // InspectResult JSON serialization
    // ========================================================================

    #[test]
    fn inspect_result_json_serialization() {
        let result = InspectResult {
            file: "model.apr".to_string(),
            valid: true,
            format: "APR v2".to_string(),
            version: "2.0".to_string(),
            tensor_count: 100,
            size_bytes: 1_000_000,
            checksum_valid: true,
            architecture: Some("llama".to_string()),
            num_layers: Some(32),
            num_heads: Some(32),
            hidden_size: Some(4096),
            vocab_size: Some(128256),
            flags: FlagsInfo {
                lz4_compressed: false,
                zstd_compressed: false,
                encrypted: false,
                signed: false,
                sharded: false,
                quantized: true,
                has_vocab: true,
            },
            metadata: MetadataInfo {
                architecture: Some("llama".to_string()),
                ..MetadataInfo::default()
            },
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("model.apr"));
        assert!(json.contains("\"valid\":true"));
        assert!(json.contains("\"tensor_count\":100"));
        assert!(json.contains("\"architecture\":\"llama\""));
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn inspect_result_skips_none_fields() {
        let result = InspectResult {
            file: "test.apr".to_string(),
            valid: false,
            format: "unknown".to_string(),
            version: "0".to_string(),
            tensor_count: 0,
            size_bytes: 0,
            checksum_valid: false,
            architecture: None,
            num_layers: None,
            num_heads: None,
            hidden_size: None,
            vocab_size: None,
            flags: FlagsInfo {
                lz4_compressed: false,
                zstd_compressed: false,
                encrypted: false,
                signed: false,
                sharded: false,
                quantized: false,
                has_vocab: false,
            },
            metadata: MetadataInfo::default(),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        // Top-level architecture (on InspectResult) has skip_serializing_if
        // but metadata.architecture does NOT — it serializes as null
        assert!(!json.contains("\"num_layers\""), "None num_layers should be skipped");
        assert!(!json.contains("\"hidden_size\""), "None hidden_size should be skipped");
    }

    // ========================================================================
    // FlagsInfo
    // ========================================================================

    #[test]
    fn flags_info_all_false() {
        let flags = FlagsInfo {
            lz4_compressed: false,
            zstd_compressed: false,
            encrypted: false,
            signed: false,
            sharded: false,
            quantized: false,
            has_vocab: false,
        };
        let json = serde_json::to_string(&flags).expect("serialize");
        assert!(json.contains("\"lz4_compressed\":false"));
    }

    // ========================================================================
    // C-APR-PROVENANCE / AC-SHIP2-012 / FALSIFY-SHIP-022
    // ========================================================================

    /// GATE-APR-PROV-002 (JSON half) / INV-APR-PROV-002 / FM-APR-PROV-SILENT-SKIP:
    /// MetadataInfo JSON serialization MUST contain the three provenance keys
    /// with `null` value when they are None, never silently skip them via
    /// `skip_serializing_if`.
    #[test]
    fn falsify_ship_022_inspect_emits_provenance_keys() {
        let meta = MetadataInfo::default();
        let json = serde_json::to_string(&meta).expect("serialize MetadataInfo");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");
        let obj = parsed.as_object().expect("JSON object at top level");

        for key in ["license", "data_source", "data_license"] {
            assert!(
                obj.contains_key(key),
                "MetadataInfo JSON must emit key `{key}` even when None \
                 (no skip_serializing_if); violating this hides provenance \
                 from auditors (FM-APR-PROV-SILENT-SKIP)"
            );
            assert!(
                obj[key].is_null(),
                "key `{key}` must serialize as null when None, got {:?}",
                obj[key]
            );
        }
    }

    /// GATE-APR-PROV-002 (text half) / INV-APR-PROV-002: text rendering
    /// MUST emit each provenance key as the literal "(missing)" when the
    /// field is None, rather than silently omitting the line.
    #[test]
    fn falsify_ship_022_inspect_missing_renders_as_missing() {
        let meta = MetadataInfo::default();
        let rendered = format_provenance_block(&meta);

        assert!(
            rendered.contains("Provenance:"),
            "text output must contain a 'Provenance:' block header; got:\n{rendered}"
        );
        for key in ["license", "data_source", "data_license"] {
            assert!(
                rendered.contains(&format!("{key}: (missing)")),
                "text output must render absent `{key}` as `(missing)`; got:\n{rendered}"
            );
        }
    }

    /// GATE-APR-PROV-002 (text half, populated variant): when provenance
    /// fields are populated, text rendering MUST emit the actual values
    /// (not "(missing)").
    #[test]
    fn falsify_ship_022_inspect_populated_renders_values() {
        let meta = MetadataInfo {
            license: Some("Apache-2.0".to_string()),
            data_source: Some("teacher-only".to_string()),
            data_license: Some("Apache-2.0".to_string()),
            ..Default::default()
        };
        let rendered = format_provenance_block(&meta);

        assert!(rendered.contains("license: Apache-2.0"));
        assert!(rendered.contains("data_source: teacher-only"));
        assert!(rendered.contains("data_license: Apache-2.0"));
        assert!(
            !rendered.contains("(missing)"),
            "populated provenance must not render `(missing)`; got:\n{rendered}"
        );
    }

    // ========================================================================
    // PMAT-690 P0-K — apr inspect surfaces hf_architecture + hf_model_type
    // ========================================================================

    /// PMAT-690 P0-K: `apr inspect --json` MUST emit `hf_architecture` and
    /// `hf_model_type` keys (null when None). Operators query
    /// `apr inspect --json | jq .metadata.hf_architecture` to verify that
    /// the upstream `apr convert` stamping worked. Silently skipping the
    /// keys hides the upstream-producer defect that this contract was
    /// authored to prevent (see memory/feedback_upstream_metadata_masquerade.md).
    #[test]
    fn pmat_690_p0k_inspect_emits_hf_arch_keys_when_none() {
        let meta = MetadataInfo::default();
        let json = serde_json::to_string(&meta).expect("serialize MetadataInfo");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");
        let obj = parsed.as_object().expect("JSON object at top level");

        // Both keys MUST be present and null (not skipped via
        // skip_serializing_if). The contract on apr inspect is that an
        // operator can grep for `"hf_architecture"` in any output and
        // distinguish "stamped" from "missing".
        for key in ["hf_architecture", "hf_model_type"] {
            assert!(
                obj.contains_key(key),
                "MetadataInfo JSON must emit key `{key}` even when None \
                 (no skip_serializing_if). Auditing the import→pretrain→export \
                 chain requires both keys to be grep-checkable."
            );
            assert!(
                obj[key].is_null(),
                "key `{key}` must serialize as null when None, got {:?}",
                obj[key]
            );
        }
    }

    /// PMAT-690 P0-K: when hf_architecture / hf_model_type are populated,
    /// `apr inspect --json` renders the actual values (not null, not the
    /// architecture-family-lowercase string).
    #[test]
    fn pmat_690_p0k_inspect_emits_hf_arch_values_when_populated() {
        let meta = MetadataInfo {
            architecture: Some("qwen2".to_string()),
            hf_architecture: Some("Qwen2ForCausalLM".to_string()),
            hf_model_type: Some("qwen2".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&meta).expect("serialize MetadataInfo");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("parse JSON");
        let obj = parsed.as_object().expect("JSON object");

        assert_eq!(
            obj.get("architecture").and_then(|v| v.as_str()),
            Some("qwen2"),
            "architecture (family) must render unchanged"
        );
        assert_eq!(
            obj.get("hf_architecture").and_then(|v| v.as_str()),
            Some("Qwen2ForCausalLM"),
            "hf_architecture (HF class name) must render the canonical string"
        );
        assert_eq!(
            obj.get("hf_model_type").and_then(|v| v.as_str()),
            Some("qwen2"),
            "hf_model_type must render the config.json::model_type value"
        );
    }

    // ========================================================================
    // PMAT-690 P3-A — `apr inspect --quality` scorer (AC-SHIP2-007 ≥ 90)
    // ========================================================================

    fn mk_header(checksum_valid: bool) -> HeaderData {
        HeaderData {
            version: (2, 0),
            flags: AprV2Flags::default(),
            tensor_count: 0,
            metadata_offset: 0,
            metadata_size: 0,
            tensor_index_offset: 0,
            data_offset: 0,
            checksum_valid,
        }
    }

    /// PMAT-690 P3-A INV-001: a ship-ready model (all 5 sub-scores
    /// populated) scores ≥ 90.
    #[test]
    fn pmat_690_p3a_ship_ready_model_scores_at_least_90() {
        let meta = MetadataInfo {
            architecture: Some("qwen2".to_string()),
            hf_architecture: Some("Qwen2ForCausalLM".to_string()),
            hf_model_type: Some("qwen2".to_string()),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            license: Some("Apache-2.0".to_string()),
            data_source: Some("teacher-only".to_string()),
            data_license: Some("Apache-2.0".to_string()),
            ..Default::default()
        };
        let mut header = mk_header(true);
        header.flags = header.flags.with(AprV2Flags::HAS_VOCAB);
        let q = compute_quality_score(&meta, &header);
        assert!(
            q.ship_ready,
            "ship-ready model must score ≥ 90 per AC-SHIP2-007 (got {})",
            q.score
        );
        assert!(q.score >= 90, "got {}", q.score);
    }

    /// PMAT-690 P3-A INV-002: a model missing both HF identity AND
    /// provenance scores well below 90 — the §81-§83 cascade scenario.
    #[test]
    fn pmat_690_p3a_no_hf_no_provenance_scores_below_55() {
        let meta = MetadataInfo {
            architecture: Some("qwen2".to_string()),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            // No hf_architecture, no hf_model_type, no provenance.
            ..Default::default()
        };
        let mut header = mk_header(true);
        header.flags = header.flags.with(AprV2Flags::HAS_VOCAB);
        let q = compute_quality_score(&meta, &header);
        assert!(
            !q.ship_ready,
            "pre-§84 cascade model must NOT be ship-ready (got {})",
            q.score
        );
        // physics(20) + structural(20) + tokenizer(15) = 55 max without
        // provenance + hf_identity.
        assert!(
            q.score <= 55,
            "no provenance + no hf identity must cap at 55 (got {})",
            q.score
        );
    }

    /// PMAT-690 P3-A INV-003: invalid checksum drops the physics
    /// sub-score to 0 and pulls the total below 90.
    #[test]
    fn pmat_690_p3a_invalid_checksum_blocks_ship() {
        let meta = MetadataInfo {
            architecture: Some("qwen2".to_string()),
            hf_architecture: Some("Qwen2ForCausalLM".to_string()),
            hf_model_type: Some("qwen2".to_string()),
            hidden_size: Some(896),
            num_layers: Some(24),
            num_heads: Some(14),
            license: Some("Apache-2.0".to_string()),
            data_source: Some("teacher-only".to_string()),
            data_license: Some("Apache-2.0".to_string()),
            ..Default::default()
        };
        let mut header = mk_header(false); // ← invalid
        header.flags = header.flags.with(AprV2Flags::HAS_VOCAB);
        let q = compute_quality_score(&meta, &header);
        assert_eq!(
            q.physics, 0,
            "invalid checksum must zero physics sub-score"
        );
        assert!(
            !q.ship_ready,
            "model with invalid checksum must NOT be ship-ready (got score {})",
            q.score
        );
    }

    /// PMAT-690 P3-A INV-004: QualityReport JSON contains the
    /// breakdown fields operators need to debug a sub-90 score.
    #[test]
    fn pmat_690_p3a_quality_json_emits_breakdown() {
        let meta = MetadataInfo::default();
        let header = mk_header(true);
        let q = compute_quality_score(&meta, &header);
        let json = q.to_json();
        let obj = json.as_object().expect("JSON object");
        assert!(obj.contains_key("score"));
        assert!(obj.contains_key("ship_ready"));
        assert!(obj.contains_key("threshold"));
        assert_eq!(obj["threshold"].as_i64(), Some(90));
        let breakdown = obj
            .get("breakdown")
            .and_then(|v| v.as_object())
            .expect("breakdown");
        for key in [
            "physics",
            "structural",
            "provenance",
            "hf_identity",
            "tokenizer",
        ] {
            assert!(
                breakdown.contains_key(key),
                "breakdown MUST include `{key}` so operators can debug a sub-90 score"
            );
        }
    }
}
