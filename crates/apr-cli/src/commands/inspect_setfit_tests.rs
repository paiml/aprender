
// ============================================================================
// APR-05 inspection tests
// ============================================================================
//
// The fixture is a REAL APR v2 container whose one `setfit` custom key holds a
// document built by CONSTRUCTING core's `SetFitArtifactDoc` and serializing it —
// not by hand-writing JSON. The difference matters: the compiler then enforces that
// the fixture carries every field the closed schema declares, so a field added to
// the schema breaks this file at compile time rather than quietly going unasserted.
//
// The document is a real document; the TENSORS in the fixture container are not a
// model, and nothing here pretends otherwise. Inspection reads what the artifact
// SAYS, which is exactly the surface under test.

#[cfg(all(test, feature = "setfit"))]
mod setfit_inspection {
    use super::*;
    use crate::setfit_tag::test_support::write_setfit_shaped_apr;
    use aprender::setfit::artifact::SETFIT_ARTIFACT_DOC_FIELDS;
    use aprender::setfit::{
        EncoderArchitecture, SetFitArtifactDoc, SetFitHeadDoc, SetFitPreprocessingDoc,
        SetFitProbeRecord,
    };
    use std::collections::BTreeMap;
    use tempfile::TempDir;

    /// This section's own source, for the source assertions below.
    const INSPECT_SETFIT_SOURCE: &str = include_str!("inspect_setfit.rs");

    const FIXTURE_LABELS: [&str; 3] = ["against", "favor", "neutral"];
    const FIXTURE_DATASET_FP: &str = "aa11bb22cc33dd44ee55ff6600778899aabbccddeeff00112233445566778899";
    const FIXTURE_VALIDATION_FP: &str =
        "11aa22bb33cc44dd55ee66ff778899aabbccddeeff001122334455667788990a";
    const FIXTURE_TABLE_HASH: &str =
        "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa";

    /// A complete `setfit-apr-v1` artifact document.
    fn fixture_doc() -> SetFitArtifactDoc {
        SetFitArtifactDoc {
            schema: "setfit-apr-v1".to_string(),
            schema_version: 1,
            bundle_schema_version: 1,
            format_id: "setfit-apr-v1".to_string(),
            architecture: EncoderArchitecture {
                hidden: 8,
                heads: 2,
                head_dim: 4,
                num_layers: 2,
                intermediate: 16,
                vocab: 48,
                positions: 256,
                type_vocab_size: 2,
                layer_norm_eps: 1e-12,
                pad_token_id: 0,
                hidden_act: "gelu_exact".to_string(),
                source_revision: "pinned-revision-for-inspection".to_string(),
                tokenizer_sha256: "0123456789abcdef".repeat(4),
                vocab_remap: None,
            },
            tokenizer_sha256: "0123456789abcdef".repeat(4),
            preprocessing: SetFitPreprocessingDoc {
                pooling: "mean".to_string(),
                normalization: "l2".to_string(),
                l2_epsilon_hex: "3727c5ac".to_string(),
                truncation_max_sequence_length: 256,
                padding_mode: "max_length".to_string(),
                max_length: 128,
            },
            root_seed: 0x0405_0000_0000_0002,
            head: SetFitHeadDoc {
                n_features: 8,
                num_labels: 3,
            },
            ordered_labels: FIXTURE_LABELS.iter().map(|s| (*s).to_string()).collect(),
            requested_config: serde_json::json!({ "root_seed": 7, "epochs": 1 }),
            resolved_config: serde_json::json!({ "device": "cpu" }),
            evidence: serde_json::json!({
                "schema_version": 1,
                "verdict": "passed",
                "trainable_count": 128,
                "frozen_count": 0,
                "per_class": {},
                "worst_param_name": "attention_key_bias",
                "epsilon_used": 0.001,
                "calibration_regime_id": "minilm-slice-h64-l2-a2-i256-v97@1110a243",
                "contract_version": "setfit-train-lifecycle-v1",
                "table_hash": FIXTURE_TABLE_HASH,
            }),
            provenance: serde_json::json!({
                "dataset_fingerprint": FIXTURE_DATASET_FP,
                "validation_split_fingerprint": FIXTURE_VALIDATION_FP,
                "selection_semantic_hash": "c0ffee".repeat(10) + "abcd",
                "selection_ledger_hash": "decafbad".repeat(8),
                "selection_root_seed": 42_u64,
                "shots_per_class": 8_u32,
            }),
            hf_name_map: BTreeMap::new(),
            probes: vec![SetFitProbeRecord {
                input: "probe_ascii".to_string(),
                embedding_hex: vec!["3f800000".to_string()],
                logits_hex: vec!["3f800000".to_string()],
                probabilities_hex: vec!["3f800000".to_string()],
                label: "favor".to_string(),
            }],
        }
    }

    /// Write the fixture artifact and return `(path, built section)`.
    fn built_section(dir: &TempDir, doc: &SetFitArtifactDoc) -> (std::path::PathBuf, serde_json::Value) {
        let json = serde_json::to_string(doc).expect("the fixture document serializes");
        let path = write_setfit_shaped_apr(
            dir.path(),
            "classifier.apr",
            crate::setfit_tag::SETFIT_MODEL_TYPE,
            Some(&json),
        );
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("the fixture document is JSON");
        let section = build_setfit_inspection(&path, Some(&value))
            .expect("a tagged artifact must produce an APR-05 section");
        (path, section)
    }

    #[test]
    fn inspect_setfit_recovers_every_apr_05_field_by_name() {
        let temp = TempDir::new().expect("tempdir");
        let (_path, section) = built_section(&temp, &fixture_doc());

        // Asserted key BY KEY, deliberately. A `contains`-style check over the
        // rendered JSON would pass for a section that carried the right substrings
        // in the wrong places, and a count check would pass for a renamed field.
        for key in [
            "schema",
            "schema_version",
            "bundle_schema_version",
            "format_id",
            "encoder_revision",
            "encoder_tokenizer_sha256",
            "tokenizer_sha256",
            "hidden_act",
            "architecture",
            "preprocessing",
            "ordered_labels",
            "head",
            "dataset_fingerprint",
            "validation_split_fingerprint",
            "selection_semantic_hash",
            "selection_ledger_hash",
            "root_seed",
            "selection_root_seed",
            "shots_per_class",
            "evidence",
            "artifact_sha256",
            "document_schema_valid",
        ] {
            assert!(
                section.get(key).is_some(),
                "APR-05 field `{key}` is missing from the inspection section"
            );
            assert!(
                !section[key].is_null(),
                "APR-05 field `{key}` rendered as null on a COMPLETE document, which means the \
                 renderer's path into the document is wrong"
            );
        }

        // The nested groups, also key by key.
        for key in [
            "hidden",
            "heads",
            "head_dim",
            "num_layers",
            "intermediate",
            "vocab",
            "positions",
            "type_vocab_size",
        ] {
            assert!(
                !section["architecture"][key].is_null(),
                "architecture.{key} is missing"
            );
        }
        for key in [
            "pooling",
            "normalization",
            "l2_epsilon_hex",
            "truncation_max_sequence_length",
            "padding_mode",
            "max_length",
        ] {
            assert!(
                !section["preprocessing"][key].is_null(),
                "preprocessing.{key} is missing — APR-05 names the pooling/normalization/\
                 truncation policy explicitly"
            );
        }
        for key in ["n_features", "num_labels"] {
            assert!(!section["head"][key].is_null(), "head.{key} is missing");
        }
        for key in ["verdict", "calibration_regime_id", "table_hash"] {
            assert!(
                !section["evidence"][key].is_null(),
                "evidence.{key} is missing — APR-05 names the update-evidence summary AND its \
                 binding table hash"
            );
        }
    }

    #[test]
    fn inspect_setfit_recovers_both_provenance_fingerprints_and_both_seeds() {
        // BOTH, and they must be DIFFERENT values: Phase 2 made the dataset
        // fingerprint and the validation-split fingerprint deliberately distinct, so
        // a renderer that read one path twice would look correct in a test that only
        // checked presence.
        let temp = TempDir::new().expect("tempdir");
        let (_path, section) = built_section(&temp, &fixture_doc());

        assert_eq!(section["dataset_fingerprint"], FIXTURE_DATASET_FP);
        assert_eq!(section["validation_split_fingerprint"], FIXTURE_VALIDATION_FP);
        assert_ne!(
            section["dataset_fingerprint"], section["validation_split_fingerprint"],
            "the two fingerprints digest different things and must not be read off one path"
        );
        assert_eq!(section["root_seed"], 0x0405_0000_0000_0002_u64);
        assert_eq!(
            section["selection_root_seed"], 42,
            "the SELECTION's root seed is a different number from the run's, and APR-05 asks \
             for both"
        );
        assert_eq!(section["evidence"]["table_hash"], FIXTURE_TABLE_HASH);
    }

    #[test]
    fn inspect_setfit_artifact_hash_is_the_file_hash_through_the_bounded_door() {
        let temp = TempDir::new().expect("tempdir");
        let (path, section) = built_section(&temp, &fixture_doc());

        // Computed INDEPENDENTLY here, over the file as it sits on disk. Asking the
        // section for its own value twice would compare a field with itself.
        let bytes = std::fs::read(&path).expect("the fixture is readable");
        let expected = aprender::setfit::artifact_sha256_hex(&bytes);
        assert_eq!(
            section["artifact_sha256"], expected,
            "inspect must report the digest of the FILE, which is the value every other surface \
             reports for the same artifact"
        );
        assert!(
            section.get("artifact_sha256_note").is_none(),
            "a successful hash must not also carry an excuse"
        );
    }

    #[test]
    fn inspect_setfit_document_schema_validity_is_a_signal_and_never_a_gate() {
        let temp = TempDir::new().expect("tempdir");

        // The complete document validates.
        let (_p, good) = built_section(&temp, &fixture_doc());
        assert_eq!(good["document_schema_valid"], true);
        assert!(good.get("document_schema_error").is_none());

        // A document this build cannot parse still RENDERS what it carries. That is
        // the point: an operator inspects a broken artifact precisely because it is
        // broken, and a version of `inspect` that refused would be useless exactly
        // then.
        let partial = serde_json::json!({
            "schema": "setfit-apr-v1",
            "schema_version": 2,
            "ordered_labels": ["against", "favor"],
            "an_unknown_future_field": true,
        });
        let path = write_setfit_shaped_apr(
            temp.path(),
            "future.apr",
            crate::setfit_tag::SETFIT_MODEL_TYPE,
            Some(&partial.to_string()),
        );
        let section = build_setfit_inspection(&path, Some(&partial))
            .expect("a tagged artifact always produces a section");
        assert_eq!(
            section["document_schema_valid"], false,
            "an unknown field must be REPORTED, because deny_unknown_fields is what makes the \
             schema closed"
        );
        assert!(section.get("document_schema_error").is_some());
        assert_eq!(
            section["schema"], "setfit-apr-v1",
            "the fields the document DOES carry must still be recovered"
        );
        assert_eq!(section["ordered_labels"][1], "favor");
        assert!(
            section["dataset_fingerprint"].is_null(),
            "a field the document does not carry renders as null, never as a skipped key: an \
             auditor greps for it, and a missing key is indistinguishable from a field this \
             build has never heard of"
        );
    }

    #[test]
    fn inspect_setfit_top_level_paths_are_the_documents_own_field_names() {
        // Guards the ONE renderer against drifting from the struct it reads. Every
        // top-level path this section walks into must be a field core declares in
        // its normative list, so a renamed field is caught here rather than by a
        // null appearing in production output.
        let walked = [
            "schema",
            "schema_version",
            "bundle_schema_version",
            "format_id",
            "architecture",
            "tokenizer_sha256",
            "preprocessing",
            "root_seed",
            "head",
            "ordered_labels",
            "evidence",
            "provenance",
        ];
        for path in walked {
            assert!(
                SETFIT_ARTIFACT_DOC_FIELDS.contains(&path),
                "`{path}` is not a field of SetFitArtifactDoc; the renderer would read null in \
                 production while this suite stayed green"
            );
        }
        // Non-vacuity: the normative list must be the one that is actually 16 long,
        // so a stubbed-out constant could not make the loop above trivial.
        assert_eq!(SETFIT_ARTIFACT_DOC_FIELDS.len(), 16);
    }

    /// The CODE LINES of `inspect_setfit.rs`.
    ///
    /// Comment lines are filtered because the section header EXPLAINS the APR-04
    /// boundary using the very names the scan forbids — `VerifiedSetFitModel` above
    /// all — and a guard that fails on its own documentation is the F-05 defect. The
    /// explanation is worth more than the names' absence from prose. The filter
    /// carries a non-vacuity assertion, because one that ate the module would make
    /// every claim below trivially true.
    fn production_code_lines() -> String {
        let code: String = INSPECT_SETFIT_SOURCE
            .lines()
            .filter(|line| {
                let t = line.trim_start();
                !t.starts_with("//")
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("fn build_setfit_inspection"),
            "non-vacuity: the comment filter must not have eaten the module's code"
        );
        code
    }

    #[test]
    fn inspect_setfit_reads_the_whole_file_only_through_the_bounded_door() {
        // Review B5, at this file. The needle is assembled at runtime so this scan
        // does not match its own text.
        let code = production_code_lines();
        let banned: String = ["fs::", "read("].concat();
        assert_eq!(
            code.matches(&banned).count(),
            0,
            "the artifact hash must not come from an unbounded whole-file read"
        );
        assert!(
            code.contains("read_setfit_apr_file_bounded"),
            "the ONE bounded artifact door is how this section obtains file bytes"
        );
        // APR-04's boundary: inspection does not verify, so it must not call the
        // loader. A `load_setfit_apr` here would make `apr inspect` unable to look
        // at exactly the artifacts an operator needs to look at.
        for banned in ["load_setfit_apr", "replay_probes", "VerifiedSetFitModel"] {
            assert!(
                !code.contains(banned),
                "inspection is read-only metadata work and must not reach `{banned}`"
            );
        }
    }

    #[test]
    fn inspect_setfit_human_rendering_runs_over_the_same_section() {
        // The human path is a VIEW over the JSON value, so this asserts it consumes
        // the section without panicking on a complete document and on a partial one.
        let temp = TempDir::new().expect("tempdir");
        let (_p, section) = built_section(&temp, &fixture_doc());
        output_setfit_text(Some(&section));
        output_setfit_text(Some(&serde_json::json!({})));
        output_setfit_text(None);
    }
}

#[cfg(test)]
mod setfit_inspection_non_regression {
    use super::*;

    /// The EXACT top-level key set `apr inspect --json` emitted before this branch.
    ///
    /// Spelled out rather than counted: a count would pass for a renamed key, and
    /// downstream tooling greps for these names.
    const PLAIN_APR_TOP_LEVEL_KEYS: [&str; 12] = [
        "file",
        "valid",
        "format",
        "version",
        "tensor_count",
        "size_bytes",
        "checksum_valid",
        "architecture",
        "num_layers",
        "num_heads",
        "hidden_size",
        "vocab_size",
    ];

    #[test]
    fn inspect_json_for_a_plain_apr_gains_no_setfit_key() {
        // The non-regression golden. `setfit` is `skip_serializing_if =
        // "Option::is_none"`, so a plain APR's document is byte-identical to what it
        // was before this plan. A key that appeared on EVERY model — even a null one
        // — would break every downstream `jq` in the fleet.
        let result = InspectResult {
            file: "plain.apr".to_string(),
            valid: true,
            format: "APR v2".to_string(),
            version: "2.0".to_string(),
            tensor_count: 3,
            size_bytes: 1024,
            checksum_valid: true,
            architecture: Some("qwen2".to_string()),
            num_layers: Some(2),
            num_heads: Some(2),
            hidden_size: Some(8),
            vocab_size: Some(48),
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
        let value = serde_json::to_value(&result).expect("the report serializes");
        let object = value.as_object().expect("the report is a JSON object");

        assert!(
            !object.contains_key("setfit"),
            "a plain APR's inspect output must be unchanged; found a `setfit` key"
        );
        for key in PLAIN_APR_TOP_LEVEL_KEYS {
            assert!(object.contains_key(key), "pre-existing key `{key}` disappeared");
        }
        assert!(object.contains_key("flags") && object.contains_key("metadata"));
        assert_eq!(
            object.len(),
            PLAIN_APR_TOP_LEVEL_KEYS.len() + 2,
            "the plain-APR report must have exactly its pre-existing keys plus `flags` and \
             `metadata`; keys observed: {:?}",
            object.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn inspect_builds_no_setfit_section_without_a_document() {
        assert!(
            build_setfit_inspection(std::path::Path::new("/nonexistent.apr"), None).is_none(),
            "no document means no section, and the absent path is never opened"
        );
    }

    #[test]
    fn inspect_metadata_read_bounds_the_declared_block_by_the_file_length() {
        // The Rule-2 half of this task: `metadata_size` is an attacker-controlled
        // u32, and the pre-existing code allocated it before discovering the file was
        // too short. Truncating a real container is the layout-independent way to
        // express "the block does not fit", and the OUTPUT is unchanged — a default
        // MetadataInfo either way, just without paying for the allocation first.
        use crate::setfit_tag::test_support::write_setfit_shaped_apr;
        let temp = tempfile::TempDir::new().expect("tempdir");
        let honest = write_setfit_shaped_apr(temp.path(), "honest.apr", "qwen2", None);
        let bytes = std::fs::read(&honest).expect("the fixture is readable");
        let truncated = temp.path().join("truncated.apr");
        std::fs::write(&truncated, &bytes[..HEADER_SIZE_V2 + 8]).expect("fixture is writable");

        let file = File::open(&truncated).expect("the truncated fixture opens");
        let mut reader = BufReader::new(file);
        let header = read_and_parse_header(&mut reader).expect("the header still parses");
        assert!(
            header.metadata_size > 8,
            "non-vacuity: the declared block must be longer than what survived truncation"
        );
        let info = read_metadata(&mut reader, &header);
        assert!(
            info.model_type.is_none() && info.setfit_doc.is_none(),
            "a block that cannot fit in its own file yields the default record"
        );
    }
}
