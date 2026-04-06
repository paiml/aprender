//! Tests for distribution module

use super::*;

#[test]
fn test_distribution_tier_core_components() {
    let core = DistributionTier::Core;
    let components = core.component_names();

    assert_eq!(components.len(), 3);
    assert!(components.contains(&"entrenar-core"));
    assert!(components.contains(&"trueno"));
    assert!(components.contains(&"aprender"));
}

#[test]
fn test_distribution_tier_standard_components() {
    let standard = DistributionTier::Standard;
    let components = standard.component_names();

    assert_eq!(components.len(), 6);
    assert!(components.contains(&"entrenar-core"));
    assert!(components.contains(&"renacer"));
    assert!(components.contains(&"trueno-db"));
    assert!(components.contains(&"ruchy"));
}

#[test]
fn test_distribution_tier_full_components() {
    let full = DistributionTier::Full;
    let components = full.component_names();

    assert_eq!(components.len(), 11);
    assert!(components.contains(&"entrenar-gpu"));
    assert!(components.contains(&"entrenar-bench"));
    assert!(components.contains(&"entrenar-lora"));
}

#[test]
fn test_distribution_tier_sizes() {
    assert_eq!(DistributionTier::Core.approximate_size_mb(), 50);
    assert_eq!(DistributionTier::Standard.approximate_size_mb(), 200);
    assert_eq!(DistributionTier::Full.approximate_size_mb(), 500);
}

#[test]
fn test_distribution_tier_includes() {
    let core = DistributionTier::Core;
    assert!(core.includes("trueno"));
    assert!(!core.includes("renacer"));

    let standard = DistributionTier::Standard;
    assert!(standard.includes("renacer"));
    assert!(!standard.includes("entrenar-gpu"));

    let full = DistributionTier::Full;
    assert!(full.includes("entrenar-gpu"));
}

#[test]
fn test_distribution_format_extensions() {
    assert_eq!(DistributionFormat::Iso.extension(), "iso");
    assert_eq!(DistributionFormat::Oci.extension(), "tar");
    assert_eq!(DistributionFormat::Nix.extension(), "nix");
    assert_eq!(DistributionFormat::Flatpak.extension(), "flatpak");
    assert_eq!(DistributionFormat::Tarball.extension(), "tar.gz");
}

#[test]
fn test_distribution_format_mime_types() {
    assert_eq!(DistributionFormat::Iso.mime_type(), "application/x-iso9660-image");
    assert_eq!(DistributionFormat::Oci.mime_type(), "application/vnd.oci.image.layer.v1.tar");
}

#[test]
fn test_component_manifest_new() {
    let comp = ComponentManifest::new("test", "1.0.0", "test-crate");

    assert_eq!(comp.name, "test");
    assert_eq!(comp.version, "1.0.0");
    assert_eq!(comp.crate_name, "test-crate");
    assert!(comp.features.is_empty());
}

#[test]
fn test_component_manifest_with_features() {
    let comp = ComponentManifest::new("test", "1.0.0", "test-crate").with_features(["gpu", "cuda"]);

    assert_eq!(comp.features.len(), 2);
    assert!(comp.features.contains(&"gpu".to_string()));
    assert!(comp.features.contains(&"cuda".to_string()));
}

#[test]
fn test_sovereign_distribution_core() {
    let dist = SovereignDistribution::core();

    assert_eq!(dist.tier, DistributionTier::Core);
    assert_eq!(dist.components.len(), 3);
    assert!(dist.has_component("entrenar-core"));
    assert!(dist.has_component("trueno"));
    assert!(dist.has_component("aprender"));
}

#[test]
fn test_sovereign_distribution_standard() {
    let dist = SovereignDistribution::standard();

    assert_eq!(dist.tier, DistributionTier::Standard);
    assert_eq!(dist.components.len(), 6);
    assert!(dist.has_component("renacer"));
}

#[test]
fn test_sovereign_distribution_full() {
    let dist = SovereignDistribution::full();

    assert_eq!(dist.tier, DistributionTier::Full);
    assert_eq!(dist.components.len(), 11);
    assert!(dist.has_component("entrenar-gpu"));
}

#[test]
fn test_sovereign_distribution_checksum() {
    let mut dist = SovereignDistribution::core();
    let data = b"test bundle content";

    dist.compute_checksum(data);
    assert!(!dist.checksum.is_empty());
    assert!(dist.verify_checksum(data));
    assert!(!dist.verify_checksum(b"different content"));
}

#[test]
fn test_sovereign_distribution_empty_checksum_fails() {
    let dist = SovereignDistribution::core();
    assert!(!dist.verify_checksum(b"any content"));
}

#[test]
fn test_sovereign_distribution_to_manifest_json() {
    let dist = SovereignDistribution::core();
    let json = dist.to_manifest_json();

    assert!(json.contains("entrenar-sovereign-core"));
    assert!(json.contains("\"tier\": \"Core\"") || json.contains("\"tier\":\"Core\""));

    // Verify round-trip
    let parsed = SovereignDistribution::from_manifest_json(&json).expect("parsing should succeed");
    assert_eq!(parsed.name, dist.name);
    assert_eq!(parsed.tier, dist.tier);
}

#[test]
fn test_sovereign_distribution_suggested_filename() {
    let dist = SovereignDistribution::core();
    let filename = dist.suggested_filename();

    assert!(filename.starts_with("entrenar-sovereign-core-"));
    assert!(filename.ends_with(".tar.gz"));
}

#[test]
fn test_sovereign_distribution_with_format() {
    let dist = SovereignDistribution::core().with_format(DistributionFormat::Iso);

    assert_eq!(dist.format, DistributionFormat::Iso);
    assert!(dist.suggested_filename().ends_with(".iso"));
}

#[test]
fn test_distribution_tier_display() {
    assert_eq!(format!("{}", DistributionTier::Core), "core");
    assert_eq!(format!("{}", DistributionTier::Standard), "standard");
    assert_eq!(format!("{}", DistributionTier::Full), "full");
}

#[test]
fn test_distribution_format_display() {
    assert_eq!(format!("{}", DistributionFormat::Iso), "ISO");
    assert_eq!(format!("{}", DistributionFormat::Oci), "OCI");
    assert_eq!(format!("{}", DistributionFormat::Nix), "Nix");
}

#[test]
fn test_distribution_serialization() {
    let dist = SovereignDistribution::full();
    let json = serde_json::to_string(&dist).expect("JSON serialization should succeed");
    let parsed: SovereignDistribution =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(dist.name, parsed.name);
    assert_eq!(dist.tier, parsed.tier);
    assert_eq!(dist.components.len(), parsed.components.len());
}
