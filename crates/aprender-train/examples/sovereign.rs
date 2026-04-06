//! Sovereign Deployment Example
//!
//! Demonstrates air-gapped deployment capabilities:
//! - Distribution manifest generation
//! - Offline model registry
//! - Nix flake configuration
//!
//! Run with: cargo run --example sovereign

use entrenar::sovereign::{
    DistributionFormat, DistributionTier, NixFlakeConfig, OfflineModelRegistry,
    SovereignDistribution,
};
use std::path::PathBuf;

fn main() -> entrenar::Result<()> {
    println!("=== Entrenar Sovereign Deployment Demo ===\n");

    // 1. Distribution Manifest (ENT-016)
    println!("--- Distribution Manifest ---");

    let core_dist = SovereignDistribution::core();
    println!("Core Distribution:");
    println!("  Tier: {:?}", core_dist.tier);
    println!("  Format: {:?}", core_dist.format);
    println!("  Components: {} total", core_dist.component_count());
    for comp in &core_dist.components {
        println!("    - {} v{}", comp.name, comp.version);
    }
    println!("  Filename: {}", core_dist.suggested_filename());
    println!("  Size: ~{}MB", core_dist.tier.approximate_size_mb());
    println!();

    let full_dist = SovereignDistribution::full().with_format(DistributionFormat::Iso);
    println!("Full Distribution (ISO):");
    println!("  Tier: {:?}", full_dist.tier);
    println!("  Format: {:?}", full_dist.format);
    println!("  Components: {} total", full_dist.component_count());
    println!("  Filename: {}", full_dist.suggested_filename());
    println!("  Size: ~{}MB", full_dist.tier.approximate_size_mb());
    println!();

    // Generate manifest JSON
    let manifest_json = full_dist.to_manifest_json();
    println!("Manifest JSON (truncated):");
    println!("  {}...", &manifest_json[..manifest_json.len().min(200)]);
    println!();

    // 2. Offline Model Registry (ENT-017)
    println!("--- Offline Model Registry ---");

    let registry_path = PathBuf::from("/tmp/entrenar-models");
    let registry = OfflineModelRegistry::new(registry_path.clone());
    println!("Registry root: {}", registry.root().display());

    // Show default location
    let default_registry = OfflineModelRegistry::default_location();
    println!("Default registry location: {}", default_registry.root().display());
    println!();

    // 3. Nix Flake Configuration (ENT-018)
    println!("--- Nix Flake Configuration ---");

    let nix_config = NixFlakeConfig::sovereign_stack();
    println!("Nix Flake Config:");
    println!("  Description: {}", nix_config.description);
    println!("  Systems: {:?}", nix_config.systems);
    println!("  Crates: {} total", nix_config.crates.len());
    for crate_spec in &nix_config.crates {
        println!("    - {} ({})", crate_spec.name, crate_spec.nix_source());
    }
    println!();

    // Generate flake.nix
    let flake_nix = nix_config.generate_flake_nix();
    println!("Generated flake.nix (first 500 chars):");
    println!("---");
    println!("{}", &flake_nix[..flake_nix.len().min(500)]);
    println!("---");
    println!();

    // Generate Cachix config
    let cachix_config = nix_config.generate_cachix_config();
    println!("Generated Cachix config:");
    println!("{cachix_config}");
    println!();

    // 4. Air-Gapped Workflow Summary
    println!("=== Air-Gapped Deployment Workflow ===");
    println!();
    println!("1. On networked machine:");
    println!("   nix build --out-link result");
    println!("   nix copy --to file://./store result");
    println!();
    println!("2. Transfer ./store to air-gapped machine");
    println!();
    println!("3. On air-gapped machine:");
    println!("   nix copy --from file://./store result");
    println!("   ./result/bin/entrenar train config.yaml");
    println!();

    // 5. Distribution Tiers Summary
    println!("=== Distribution Tiers ===");
    println!();
    for tier in [DistributionTier::Core, DistributionTier::Standard, DistributionTier::Full] {
        let dist = match tier {
            DistributionTier::Core => SovereignDistribution::core(),
            DistributionTier::Standard => SovereignDistribution::standard(),
            DistributionTier::Full => SovereignDistribution::full(),
        };
        println!(
            "{:?}: ~{}MB - {} components",
            tier,
            tier.approximate_size_mb(),
            dist.component_count()
        );
    }
    println!();

    // 6. Distribution Formats Summary
    println!("=== Distribution Formats ===");
    println!();
    for format in [
        DistributionFormat::Tarball,
        DistributionFormat::Iso,
        DistributionFormat::Oci,
        DistributionFormat::Nix,
        DistributionFormat::Flatpak,
    ] {
        let dist = SovereignDistribution::core().with_format(format);
        println!("{:?}: {}", dist.format, dist.suggested_filename());
    }
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
