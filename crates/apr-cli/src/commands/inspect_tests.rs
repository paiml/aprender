// PMAT-540 Phase 5: Tests for inspect command helper functions

#[cfg(test)]
mod inspect_tests {
    use super::*;
    use std::path::Path;

    // ========================================================================
    // rosetta_json (#3733): --json shows every metadata key, unfiltered
    // ========================================================================

    #[test]
    fn rosetta_json_emits_every_metadata_key() {
        use aprender::format::rosetta::{FormatType, InspectionReport};
        let keys = [
            "general.architecture",
            "qwen35.context_length",
            "qwen35.full_attention_interval",
            "quantize.imatrix.file",
            "zzz_never_heard_of.depth",
            "tokenizer.ggml.model",
        ];
        let report = InspectionReport {
            format: FormatType::Gguf,
            file_size: 1,
            metadata: keys.iter().map(|k| ((*k).to_string(), "v".to_string())).collect(),
            tensors: vec![],
            total_params: 0,
            quantization: None,
            architecture: Some("qwen35".to_string()),
        };
        let v = rosetta_json(Path::new("m.gguf"), &report);
        let shown = v["metadata"].as_object().expect("metadata object");
        assert_eq!(shown.len(), keys.len());
        for k in keys {
            assert!(shown.contains_key(k), "--json dropped {k}");
        }
    }

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
            setfit: None,
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
            setfit: None,
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

    // ========================================================================
    // WR-09 / T-04-70: read_metadata's ABSOLUTE cap
    // ========================================================================

    /// A file whose length satisfies the file-length bound, so the CAP is what fires.
    ///
    /// `set_len` and not a zero buffer: the extension is sparse, so it is instant and
    /// costs no disk even at 16 MiB + 1. A test that actually wrote those bytes would
    /// put a 16 MiB write in every `cargo test` for no additional evidence — the
    /// declared size is the whole subject, and nothing reads past it.
    fn write_sparse_file_of_len(dir: &Path, name: &str, len: u64) -> std::path::PathBuf {
        let path = dir.join(name);
        let file = std::fs::File::create(&path).expect("the sparse fixture is creatable");
        file.set_len(len).expect("sparse extension needs no buffer");
        file.sync_all().expect("the sparse fixture syncs");
        path
    }

    fn header_declaring(metadata_size: u64) -> HeaderData {
        HeaderData {
            version: (2, 0),
            flags: AprV2Flags::default(),
            tensor_count: 0,
            metadata_offset: HEADER_SIZE_V2 as u64,
            metadata_size: u32::try_from(metadata_size)
                .expect("a declared size under test must fit the u32 header field"),
            tensor_index_offset: 0,
            data_offset: 0,
            checksum_valid: true,
        }
    }

    /// T-04-70. A hostile header must be refused BEFORE the allocation it asks for.
    ///
    /// The declared size is `MAX_TAG_METADATA_BYTES + 1` and NOT `0xFFFF_FFFF`, which
    /// is the only value that exercises the cap. Measured at HEAD: the pre-existing
    /// file-length check runs FIRST, so a 4 GiB declaration on any reasonably-sized
    /// temp file is refused by the OLD bound and the cap is never reached — that test
    /// would have passed before the fix and proved nothing. So the file is padded past
    /// `metadata_offset + 16 MiB` to make the length check pass, leaving the cap as
    /// the only thing that can refuse it.
    #[test]
    fn read_metadata_refuses_a_block_over_the_absolute_cap() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let declared = crate::setfit_tag::MAX_TAG_METADATA_BYTES + 1;
        let path = write_sparse_file_of_len(
            dir.path(),
            "hostile.apr",
            HEADER_SIZE_V2 as u64 + declared + 1,
        );
        let header = header_declaring(declared);

        let mut reader = BufReader::new(File::open(&path).expect("the sparse fixture is readable"));
        let info = read_metadata(&mut reader, &header);

        assert_eq!(
            info.metadata_over_cap_bytes,
            Some(declared),
            "an over-cap block must be DISCLOSED, not silently defaulted — the fact that it \
             was refused is the whole diagnosis (WR-08's inspect half)"
        );
        assert_eq!(
            info.model_type, None,
            "nothing may be derived from a block that was never read"
        );
    }

    /// The cap is `>`, not `>=`, pinned in the SAME direction as `setfit_tag.rs`'s arm.
    ///
    /// Exactly the cap is admitted, so the read proceeds and then fails to parse a
    /// sparse block as JSON — a `MetadataInfo::default()` whose
    /// `metadata_over_cap_bytes` is `None`. That `None` is the assertion: it is what
    /// distinguishes "the cap did not fire" from "the cap fired".
    #[test]
    fn read_metadata_admits_a_block_of_exactly_the_absolute_cap() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let declared = crate::setfit_tag::MAX_TAG_METADATA_BYTES;
        let path =
            write_sparse_file_of_len(dir.path(), "at-cap.apr", HEADER_SIZE_V2 as u64 + declared);
        let header = header_declaring(declared);

        let mut reader = BufReader::new(File::open(&path).expect("the sparse fixture is readable"));
        let info = read_metadata(&mut reader, &header);

        assert_eq!(
            info.metadata_over_cap_bytes, None,
            "a block of exactly the cap is not OVER it; the boundary must sit where \
             `setfit_tag.rs`'s does, or the two readers disagree by one byte"
        );
    }

    /// NON-VACUITY. The cap must not pass by refusing everything.
    ///
    /// A container written by the production writer still parses and still yields its
    /// `model_type`, and reports no over-cap fact.
    #[test]
    fn read_metadata_still_parses_a_legitimate_artifact_under_the_cap() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = crate::setfit_tag::test_support::write_setfit_shaped_apr(
            dir.path(),
            "honest.apr",
            crate::setfit_tag::SETFIT_MODEL_TYPE,
            Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
        );

        let mut reader = BufReader::new(File::open(&path).expect("the fixture is readable"));
        let header = read_and_parse_header(&mut reader).expect("the fixture is an APR v2");
        let info = read_metadata(&mut reader, &header);

        assert_eq!(
            info.model_type.as_deref(),
            Some(crate::setfit_tag::SETFIT_MODEL_TYPE),
            "non-vacuity: a legitimate artifact must still parse, or the cap proves nothing"
        );
        assert_eq!(
            info.metadata_over_cap_bytes, None,
            "a legitimate artifact must never be disclosed as over-cap"
        );
    }
}
