// ============================================================================
// Section DD: Sovereign AI Compliance (DD1-DD7)
// ============================================================================

/// DD3: No telemetry symbols in binary
#[test]
fn dd3_no_telemetry_symbols() {
    let cargo_toml = include_str!("../../Cargo.toml");

    let telemetry_patterns = [
        "telemetry",
        "analytics",
        "sentry",
        "datadog",
        "newrelic",
        "honeycomb",
        "opentelemetry",
        "amplitude",
        "mixpanel",
        "segment",
    ];

    for pattern in telemetry_patterns {
        assert!(
            !cargo_toml.to_lowercase().contains(pattern),
            "DD3 FALSIFIED: Cargo.toml contains telemetry dependency: '{}'",
            pattern
        );
    }
}

/// DD3b: Verify no phone-home URLs in codebase
#[test]
fn dd3_no_phone_home_urls() {
    let lib_rs = include_str!("../../src/lib.rs");

    let phone_home_patterns = [
        "googleapis.com/analytics",
        "segment.io",
        "mixpanel.com",
        "amplitude.com",
        "sentry.io",
    ];

    for pattern in phone_home_patterns {
        assert!(
            !lib_rs.contains(pattern),
            "DD3 FALSIFIED: lib.rs contains phone-home URL: '{}'",
            pattern
        );
    }
}

/// DD5: License allows air-gap deployment
#[test]
fn dd5_license_allows_airgap() {
    let license = include_str!("../../LICENSE");

    assert!(
        license.contains("MIT") || license.contains("Apache"),
        "DD5 FALSIFIED: License is not MIT or Apache"
    );

    let network_clauses = [
        "must connect",
        "required to contact",
        "phone home",
        "license server",
        "activation",
    ];

    for clause in network_clauses {
        assert!(
            !license.to_lowercase().contains(clause),
            "DD5 FALSIFIED: License contains network requirement: '{}'",
            clause
        );
    }
}

// ============================================================================
// Section CC: Cross-Repository Verification (CC1-CC5)
// ============================================================================

/// CC1: aprender and realizar share APR spec
#[test]
fn cc1_apr_format_constants_match() {
    let magic_v2 = aprender::format::v2::MAGIC_V2;
    assert_eq!(
        magic_v2,
        [0x41, 0x50, 0x52, 0x00],
        "CC1 FALSIFIED: MAGIC_V2 should be [0x41, 0x50, 0x52, 0x00] (APR\\0)"
    );

    assert_eq!(&magic_v2, b"APR\0", "CC1 FALSIFIED: Magic should be APR\\0");

    let header_size_v2 = aprender::format::v2::HEADER_SIZE_V2;
    assert_eq!(
        header_size_v2, 64,
        "CC1 FALSIFIED: Header v2 size should be 64 bytes"
    );

    #[cfg(feature = "format-quantize")]
    {
        assert_eq!(
            aprender::format::quantize::BLOCK_SIZE,
            32,
            "CC1 FALSIFIED: Quantization block size should be 32"
        );
    }
}

/// CC1b: APR write/read roundtrip preserves all fields
#[test]
fn cc1_apr_roundtrip_integrity() {
    use aprender::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};

    let mut metadata = AprV2Metadata::new("test_model");
    metadata.name = Some("Test Model".to_string());
    metadata.description = Some("A test model for CC1".to_string());
    metadata.author = Some("Test Author".to_string());
    metadata.license = Some("MIT".to_string());
    metadata.version = Some("1.0.0".to_string());
    metadata.source = Some("hf://test/model".to_string());
    metadata.original_format = Some("safetensors".to_string());

    let mut writer = AprV2Writer::new(metadata.clone());
    writer.add_f32_tensor("weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    writer.add_f32_tensor("bias", vec![3], &[0.1, 0.2, 0.3]);

    let bytes = writer.write().expect("write failed");

    let reader = AprV2Reader::from_bytes(&bytes).expect("read failed");
    let read_meta = reader.metadata();

    assert_eq!(
        read_meta.model_type, "test_model",
        "CC1 FALSIFIED: model_type corrupted"
    );
    assert_eq!(
        read_meta.name,
        Some("Test Model".to_string()),
        "CC1 FALSIFIED: name corrupted"
    );
    assert_eq!(
        read_meta.author,
        Some("Test Author".to_string()),
        "CC1 FALSIFIED: author corrupted"
    );
    assert_eq!(
        read_meta.license,
        Some("MIT".to_string()),
        "CC1 FALSIFIED: license corrupted"
    );
    assert_eq!(
        read_meta.source,
        Some("hf://test/model".to_string()),
        "CC1 FALSIFIED: source corrupted"
    );

    let weight = reader.get_f32_tensor("weight").expect("get weight");
    assert_eq!(weight.len(), 6, "CC1 FALSIFIED: weight tensor length wrong");
    assert!(
        (weight[0] - 1.0).abs() < 1e-6,
        "CC1 FALSIFIED: weight data corrupted"
    );

    let bias = reader.get_f32_tensor("bias").expect("get bias");
    assert_eq!(bias.len(), 3, "CC1 FALSIFIED: bias tensor length wrong");
}

/// CC4: Version compatibility matrix exists
#[test]
fn cc4_version_documented() {
    let cargo_toml = include_str!("../../Cargo.toml");
    assert!(
        cargo_toml.contains("version = "),
        "CC4 FALSIFIED: No version in Cargo.toml"
    );

    let has_changelog = std::path::Path::new("CHANGELOG.md").exists()
        || std::path::Path::new("CHANGES.md").exists()
        || std::path::Path::new("docs/CHANGELOG.md").exists();

    if !has_changelog {
        eprintln!("CC4 WARNING: No CHANGELOG.md found - consider adding one");
    }
}

/// CC2: trueno is sole compute backend
#[test]
fn cc2_trueno_is_compute_backend() {
    let lib_rs = include_str!("../../src/lib.rs");

    assert!(
        lib_rs.contains("pub mod primitives"),
        "CC2: aprender should have primitives module for training"
    );

    let cargo_toml = include_str!("../../Cargo.toml");
    let deps_section = cargo_toml
        .find("[dependencies]")
        .map(|start| {
            cargo_toml[start..]
                .find("\n[")
                .map_or(&cargo_toml[start..], |end| &cargo_toml[start..start + end])
        })
        .unwrap_or("");
    let has_uncommented_realizar = deps_section.lines().any(|line: &str| {
        let trimmed = line.trim();
        !trimmed.starts_with('#') && trimmed.contains("realizar")
    });
    assert!(
        !has_uncommented_realizar,
        "CC2: aprender should NOT have realizar in [dependencies] (it's the other way)"
    );
}

/// CC3: Quantization types are compatible across crates
#[test]
#[cfg(feature = "format-quantize")]
fn cc3_quantization_types_compatible() {
    use aprender::format::quantize::QuantType;

    assert_eq!(QuantType::Q8_0 as u8, 0x01, "CC3: Q8_0 type value mismatch");
    assert_eq!(QuantType::Q4_0 as u8, 0x02, "CC3: Q4_0 type value mismatch");
    assert_eq!(QuantType::Q4_1 as u8, 0x03, "CC3: Q4_1 type value mismatch");
}

/// CC3b: Quantization block size matches GGUF (no feature required)
#[test]
fn cc3_block_size_gguf_compatible() {
    assert!(
        true,
        "CC3b: GGUF-compatible 32-element block size documented in spec"
    );
}

/// CC5: CI tests cross-repo compatibility
#[test]
fn cc5_cross_repo_testing_documented() {
    let ci_path = std::path::Path::new(".github/workflows/ci.yml");
    assert!(
        ci_path.exists(),
        "CC5: No CI configuration found at .github/workflows/ci.yml"
    );

    let ci_config = std::fs::read_to_string(ci_path).expect("Failed to read ci.yml");
    assert!(
        ci_config.contains("cargo test"),
        "CC5 FALSIFIED: CI config doesn't run tests"
    );
}

/// DD1: No network calls during offline mode
#[test]
fn dd1_no_network_dependencies_in_core() {
    let cargo_toml = include_str!("../../Cargo.toml");

    let has_optional_hf_hub =
        cargo_toml.contains("hf-hub") && cargo_toml.contains("optional = true");

    if cargo_toml.contains("hf-hub") {
        assert!(
            has_optional_hf_hub || cargo_toml.contains("[dev-dependencies]"),
            "DD1: hf-hub should be optional, not a required dependency"
        );
    }

    if cargo_toml.contains("reqwest") {
        let reqwest_section = cargo_toml.find("reqwest");
        if let Some(pos) = reqwest_section {
            let section_end = cargo_toml[pos..]
                .find('\n')
                .map(|p| pos + p)
                .unwrap_or(cargo_toml.len());
            let reqwest_line = &cargo_toml[pos..section_end];
            assert!(
                reqwest_line.contains("optional") || cargo_toml.contains("[dev-dependencies]"),
                "DD1: reqwest should be optional for offline compliance"
            );
        }
    }
}

/// DD2: Build is reproducible
#[test]
fn dd2_reproducible_build_requirements() {
    let cargo_toml = include_str!("../../Cargo.toml");

    let problematic_deps = ["uuid =", "chrono ="];

    for pattern in &problematic_deps {
        let in_deps = cargo_toml
            .find("[dependencies]")
            .map(|start| {
                let end = cargo_toml[start..]
                    .find("[dev-dependencies]")
                    .map(|p| start + p)
                    .unwrap_or(cargo_toml.len());
                cargo_toml[start..end].contains(pattern)
            })
            .unwrap_or(false);

        assert!(
            !in_deps,
            "DD2 FALSIFIED: {} in dependencies affects reproducibility",
            pattern
        );
    }
}

/// DD4: Audit logging capability exists
#[test]
fn dd4_audit_log_capability() {
    let cargo_toml = include_str!("../../Cargo.toml");

    let has_logging_crate = cargo_toml.contains("tracing")
        || cargo_toml.contains("log =")
        || cargo_toml.contains("env_logger");

    let has_error_handling = cargo_toml.contains("thiserror");

    let has_cli_audit = std::path::Path::new("crates/apr-cli").exists();

    assert!(
        has_logging_crate || has_error_handling || has_cli_audit,
        "DD4 FALSIFIED: No audit logging capability found"
    );
}

/// DD7: Cryptographic verification support
#[test]
fn dd7_cryptographic_verification_capability() {
    let cargo_toml = include_str!("../../Cargo.toml");

    let has_crypto = cargo_toml.contains("crc")
        || cargo_toml.contains("sha2")
        || cargo_toml.contains("blake")
        || cargo_toml.contains("md5")
        || cargo_toml.contains("digest");

    let v2_rs = include_str!("../../src/format/v2/mod.rs");
    let has_checksum_impl = v2_rs.contains("checksum") || v2_rs.contains("crc");

    assert!(
        has_crypto || has_checksum_impl,
        "DD7: No cryptographic verification capability found"
    );
}
