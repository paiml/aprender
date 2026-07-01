//! GREEN contract falsifiers for the apr-format extraction (issue #2231).
//!
//! One `#[test]` per proof obligation in
//! `contracts/apr-format-extraction-v1.yaml` + the companion
//! `apr-format-leaf-sovereignty-v1.yaml`, named exactly as those contracts'
//! `falsification_tests[].test` fields cite them so `pv lint` Gate-4 /
//! strict-test-binding resolves the refs with no dangling-ref errors.
//!
//! Stage 1 shipped these as RED `unimplemented!` stubs (the obligation existed
//! before the implementation). Stage 2 discharges them against the real bytes:
//! the golden byte-identity oracle, the `cargo metadata` dependency closure, the
//! CRC known-answer + golden trailer, the metadata round-trip, and the Jidoka
//! quality gate. A falsifier going RED here means its obligation regressed.
//!
//! # f16 scoping note (issue #2231 / PMAT-905 class)
//!
//! Byte-identity is asserted for **F32** payloads only. The golden fixtures use
//! F32 weights, so they are unaffected by the documented f16 write change (the
//! leaf now uses IEEE round-to-nearest-even via the `half` crate instead of the
//! legacy non-RNE `trueno::f32_to_f16`). See `crate::f16`.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::items_after_statements,
    clippy::no_effect_underscore_binding,
    clippy::float_cmp
)]

#[cfg(test)]
mod tests {
    use crate::types::{Compression, Metadata, ModelType, SaveOptions};
    use std::collections::HashMap;
    use std::path::PathBuf;

    use serde::{Deserialize, Serialize};

    /// The exact model that produced `tests/fixtures/golden_v1.apr` (captured
    /// from the pre-extraction in-core save path — the byte-identity oracle).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct GoldenModel {
        name: String,
        weights: Vec<f32>,
        bias: f32,
    }

    fn golden_model() -> GoldenModel {
        GoldenModel {
            name: "golden_v1".to_string(),
            weights: vec![1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0],
            bias: 0.125,
        }
    }

    /// The pinned `SaveOptions` that produced the golden fixture — every field
    /// is fixed (no `chrono_lite_now()` / `CARGO_PKG_VERSION`) so the save is
    /// deterministic and the bytes reproduce exactly.
    fn golden_options() -> SaveOptions {
        let metadata = Metadata {
            created_at: "1700000000".to_string(),
            aprender_version: "0.0.0-golden".to_string(),
            model_name: Some("golden-v1".to_string()),
            description: None,
            training: None,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            custom: HashMap::new(),
            distillation: None,
            distillation_info: None,
            license: None,
            model_card: None,
        };
        SaveOptions {
            compression: Compression::None,
            metadata,
            quality_score: Some(85),
        }
    }

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
    }

    fn golden_v1_bytes() -> Vec<u8> {
        std::fs::read(fixtures().join("golden_v1.apr")).expect("read golden_v1.apr fixture")
    }

    /// FALSIFY-APRF-BYTE-IDENTITY: re-saving the captured golden model with the
    /// pinned `SaveOptions` reproduces `golden_v1.apr` byte-for-byte (full file
    /// incl. the CRC trailer). F32 payload — unaffected by the f16 write change.
    #[test]
    fn test_falsify_aprf_byte_identity_golden_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("aprf_byte_identity_probe.apr");
        crate::save(
            &golden_model(),
            ModelType::LinearRegression,
            &path,
            golden_options(),
        )
        .expect("save golden model with pinned options");

        let produced = std::fs::read(&path).expect("read produced bytes");
        let _ = std::fs::remove_file(&path);

        let golden = golden_v1_bytes();
        assert_eq!(
            produced.len(),
            golden.len(),
            "byte length drifted (extraction changed the on-disk encoding)"
        );
        assert_eq!(
            produced, golden,
            "extracted save() output is NOT byte-identical to golden_v1.apr — \
             the serializer order, padding, header layout, or CRC drifted"
        );

        // And the leaf reads its own/golden bytes back to the captured model.
        let back: GoldenModel = crate::load_from_bytes(&golden, ModelType::LinearRegression)
            .expect("load golden bytes");
        assert_eq!(back, golden_model());
    }

    /// FALSIFY-APRF-SOVEREIGN-DEPS: the leaf's resolved dependency graph (ALL
    /// features) contains zero ML/GPU/tokenizer/framework crates.
    ///
    /// Parses `cargo metadata` (the real resolver output) — this is the same
    /// closure the CI guard `scripts/check_format_sovereignty.sh` checks.
    #[test]
    fn test_falsify_aprf_sovereign_deps_no_ml_gpu() {
        use cargo_metadata::MetadataCommand;

        const FORBIDDEN: &[&str] = &[
            "trueno",
            "aprender-compute",
            "aprender-gpu",
            "aprender-core",
            "wgpu",
            "naga",
            "cudarc",
            "cust",
            "candle-core",
            "candle-nn",
            "tch",
            "torch-sys",
        ];

        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let metadata = MetadataCommand::new()
            .manifest_path(&manifest)
            .features(cargo_metadata::CargoOpt::AllFeatures)
            .exec()
            .expect("cargo metadata for apr-format");

        // Resolve the transitive closure of the apr-format package id.
        let resolve = metadata.resolve.expect("resolve graph present");
        let id2name: HashMap<_, _> = metadata
            .packages
            .iter()
            .map(|p| (p.id.clone(), p.name.clone()))
            .collect();
        let root = metadata
            .packages
            .iter()
            .find(|p| p.name == "apr-format")
            .map(|p| p.id.clone())
            .expect("apr-format package present");
        let nodes: HashMap<_, _> = resolve.nodes.iter().map(|n| (n.id.clone(), n)).collect();

        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            if let Some(node) = nodes.get(&id) {
                for dep in &node.deps {
                    stack.push(dep.pkg.clone());
                }
            }
        }

        let names: std::collections::HashSet<&str> = seen
            .iter()
            .filter_map(|id| id2name.get(id).map(String::as_str))
            .collect();

        let leaked: Vec<&str> = FORBIDDEN
            .iter()
            .copied()
            .filter(|f| names.contains(f))
            .collect();

        assert!(
            leaked.is_empty(),
            "apr-format leaf is NO LONGER sovereign — forbidden ML/GPU/framework \
             crate(s) leaked into its dependency closure: {leaked:?}"
        );
    }

    /// FALSIFY-APRF-CRC-INTEGRITY: the deduplicated `crc32` matches the canonical
    /// check vector AND validates the golden file's stored trailer.
    #[test]
    fn test_falsify_aprf_crc_integrity_matches_legacy() {
        // Canonical IEEE CRC32 check vector "123456789" -> 0xCBF43926.
        assert_eq!(crate::crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crate::crc32(&[]), 0x0000_0000);
        assert_eq!(crate::crc32(&[0x00]), 0xD202_EF8D);

        // The leaf's crc32 must validate the core-written golden trailer.
        let bytes = golden_v1_bytes();
        let stored = u32::from_le_bytes([
            bytes[bytes.len() - 4],
            bytes[bytes.len() - 3],
            bytes[bytes.len() - 2],
            bytes[bytes.len() - 1],
        ]);
        let computed = crate::crc32(&bytes[..bytes.len() - 4]);
        assert_eq!(
            stored, computed,
            "leaf crc32 diverged from the legacy table/fold — existing .apr files \
             would fail integrity"
        );

        // Corrupting one body byte must change the checksum (integrity bite).
        let mut tampered = bytes.clone();
        tampered[crate::HEADER_SIZE + 1] ^= 0xFF;
        let recomputed = crate::crc32(&tampered[..tampered.len() - 4]);
        assert_ne!(recomputed, stored, "crc32 failed to detect a flipped byte");
    }

    /// FALSIFY-APRF-METADATA-FIDELITY: a save->load round-trip preserves every
    /// populated metadata field exactly, and a license sets the LICENSED flag.
    #[test]
    fn test_falsify_aprf_metadata_fidelity_roundtrip() {
        use crate::types::{Header, LicenseInfo, LicenseTier, TrainingInfo, HEADER_SIZE};

        let mut hyper = HashMap::new();
        hyper.insert("lr".to_string(), serde_json::json!(0.001));
        let mut metrics = HashMap::new();
        metrics.insert("acc".to_string(), serde_json::json!(0.97));
        let mut custom = HashMap::new();
        custom.insert("note".to_string(), serde_json::json!("hello"));

        let metadata = Metadata {
            created_at: "1234567890".to_string(),
            aprender_version: "9.9.9-test".to_string(),
            model_name: Some("fidelity".to_string()),
            description: Some("round-trip every field".to_string()),
            training: Some(TrainingInfo {
                samples: Some(42),
                duration_ms: Some(1000),
                source: Some("unit-test".to_string()),
            }),
            hyperparameters: hyper,
            metrics,
            custom,
            distillation: Some("teacher-hash".to_string()),
            distillation_info: None,
            license: Some(LicenseInfo {
                uuid: "uuid-1".to_string(),
                hash: "hash-1".to_string(),
                expiry: None,
                seats: Some(3),
                licensee: Some("ACME".to_string()),
                tier: LicenseTier::Enterprise,
            }),
            model_card: None,
        };

        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        struct M {
            v: Vec<f32>,
        }
        let model = M { v: vec![1.0, 2.0] };

        let dir = std::env::temp_dir();
        let path = dir.join("aprf_metadata_fidelity.apr");
        let options = SaveOptions {
            compression: Compression::None,
            metadata: metadata.clone(),
            quality_score: None,
        };
        crate::save(&model, ModelType::LinearRegression, &path, options).expect("save");

        // License presence must set the LICENSED header flag.
        let raw = std::fs::read(&path).expect("read");
        let header = Header::from_bytes(&raw[..HEADER_SIZE]).expect("hdr");
        assert!(header.flags.is_licensed(), "LICENSED flag not set");

        // Every populated field survives the round-trip via inspect().
        let info = crate::inspect(&path).expect("inspect");
        let _ = std::fs::remove_file(&path);
        let m = info.metadata;
        assert_eq!(m.created_at, metadata.created_at);
        assert_eq!(m.aprender_version, metadata.aprender_version);
        assert_eq!(m.model_name, metadata.model_name);
        assert_eq!(m.description, metadata.description);
        assert_eq!(m.distillation, metadata.distillation);
        assert_eq!(m.hyperparameters, metadata.hyperparameters);
        assert_eq!(m.metrics, metadata.metrics);
        assert_eq!(m.custom, metadata.custom);
        let (got, want) = (m.license.expect("lic"), metadata.license.expect("lic"));
        assert_eq!(got.uuid, want.uuid);
        assert_eq!(got.hash, want.hash);
        assert_eq!(got.seats, want.seats);
        assert_eq!(got.licensee, want.licensee);
        assert_eq!(got.tier, want.tier);
        let tr = m.training.expect("training");
        assert_eq!(tr.samples, Some(42));
    }

    /// FALSIFY-APRF-API-COMPAT: the leaf's public re-export surface resolves
    /// (the container types/functions the framework re-exports are all reachable
    /// from `apr_format::*`). The cross-crate `?`-ergonomics half of this
    /// obligation is proven in `aprender-core`
    /// (`test_from_apr_format_question_mark_ergonomics`).
    #[test]
    fn test_falsify_aprf_api_compat_reexport_resolves() {
        // Touch the public re-export surface so a dropped re-export fails to
        // compile (the framework re-exports exactly these paths to its callers).
        let crc: u32 = crate::crc32(b"abc");
        assert_eq!(crc, crate::crc32(b"abc"));
        let bits: u16 = crate::f32_to_f16(1.0);
        assert_eq!(crate::f16_to_f32(bits), 1.0);

        let hdr = crate::Header::new(ModelType::LinearRegression);
        assert_eq!(hdr.magic, crate::MAGIC);
        assert_eq!(crate::HEADER_SIZE, 32);

        let _info_ty: Option<crate::ModelInfo> = None;
        let _opts = crate::SaveOptions::default();
        let v2 = crate::v2::AprV2Header::new();
        assert_eq!(v2.magic, crate::v2::MAGIC_V2);
        let card = crate::ModelCard::new("m", "1.0.0");
        assert_eq!(card.version, "1.0.0");

        // A full save->load round-trip through the re-exported entry points.
        let dir = std::env::temp_dir();
        let path = dir.join("aprf_api_compat.apr");
        crate::save(
            &vec![1.0_f32, 2.0],
            ModelType::LinearRegression,
            &path,
            crate::SaveOptions::default(),
        )
        .expect("save via re-export");
        let back: Vec<f32> =
            crate::load(&path, ModelType::LinearRegression).expect("load via re-export");
        let _ = std::fs::remove_file(&path);
        assert_eq!(back, vec![1.0, 2.0]);
    }

    /// FALSIFY-APRF-QUALITY-GATE: the Jidoka quality gate is preserved —
    /// `save(quality_score = Some(0))` is REFUSED and a known-good save is `Ok`.
    #[test]
    fn test_falsify_aprf_quality_gate_preserved() {
        #[derive(Serialize)]
        struct M {
            v: Vec<f32>,
        }
        let model = M { v: vec![1.0] };
        let dir = std::env::temp_dir();

        // Some(0): explicit failure — must be refused.
        let bad = SaveOptions {
            quality_score: Some(0),
            ..Default::default()
        };
        let refused = crate::save(
            &model,
            ModelType::LinearRegression,
            dir.join("aprf_qgate_bad.apr"),
            bad,
        );
        assert!(
            matches!(refused, Err(crate::AprFormatError::ValidationError { .. })),
            "Jidoka gate lost: save(Some(0)) was NOT refused"
        );

        // Some(85): passing — must be accepted.
        let good = SaveOptions {
            quality_score: Some(85),
            ..Default::default()
        };
        let path = dir.join("aprf_qgate_good.apr");
        let accepted = crate::save(&model, ModelType::LinearRegression, &path, good);
        assert!(accepted.is_ok(), "known-good save(Some(85)) was refused");
        let _ = std::fs::remove_file(&path);
    }
}
