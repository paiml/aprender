# Sovereign Deployment

The sovereign module enables air-gapped deployment scenarios for
universities, enterprises, and environments without reliable internet
access. Think of it like the old-school Linux .ISO hosting at
university mirror sites.

## Why Sovereign Deployment?

Many organizations require:
- **Air-gapped networks**: Security-sensitive environments without internet
- **Reproducible builds**: Exact same binaries on every deployment
- **Offline model registries**: Pre-downloaded models for training
- **Self-hosted infrastructure**: No dependency on external services

## Components

The sovereign module provides three key components:

### 1. Distribution Manifest (ENT-016)

Package the entire PAIML stack into distributable bundles:

```rust
use entrenar::sovereign::{SovereignDistribution, DistributionTier, DistributionFormat};

// Create a core distribution (~50MB)
let core = SovereignDistribution::core();

// Create a full distribution (~500MB) as ISO
let full = SovereignDistribution::full()
    .with_format(DistributionFormat::Iso);

// Generate manifest JSON
let manifest = full.to_manifest_json();

// Verify bundle checksum
full.verify_checksum(&bundle_data);
```

### 2. Offline Model Registry (ENT-017)

Manage pre-downloaded models for air-gapped training:

```rust
use entrenar::sovereign::{OfflineModelRegistry, ModelSource};
use std::path::PathBuf;

// Create registry at custom location
let mut registry = OfflineModelRegistry::new(PathBuf::from("/data/models"));

// Register a local model file
let entry = registry.register_local("llama-7b", &PathBuf::from("llama-7b.gguf"))?;

// Verify model integrity
assert!(registry.verify(&entry)?);

// Load model for training
let model_path = registry.load("llama-7b")?;
```

### 3. Nix Flake Generation (ENT-018)

Generate reproducible Nix flakes for the entire stack:

```rust
use entrenar::sovereign::{NixFlakeConfig, NixSystem};

// Generate sovereign stack flake
let config = NixFlakeConfig::sovereign_stack();

// Generate flake.nix content
let flake_nix = config.generate_flake_nix();

// Generate Cachix configuration
let cachix_config = config.generate_cachix_config();
```

## Distribution Tiers

| Tier | Size | Components |
|------|------|------------|
| Core | ~50MB | entrenar-core, trueno, aprender |
| Standard | ~200MB | + renacer, trueno-db, ruchy |
| Full | ~500MB | + GPU support, all tooling |

## Distribution Formats

| Format | Use Case |
|--------|----------|
| Tarball | Simple extraction, any platform |
| ISO | Bootable media with NixOS |
| OCI | Container deployments |
| Nix | Nix/NixOS environments |
| Flatpak | Desktop Linux applications |

## Quick Start

```rust
use entrenar::sovereign::*;

// 1. Create distribution bundle
let dist = SovereignDistribution::standard();
println!("Bundle: {}", dist.suggested_filename());

// 2. Set up offline registry
let registry = OfflineModelRegistry::default_location();
println!("Models at: {}", registry.root().display());

// 3. Generate Nix flake
let config = NixFlakeConfig::sovereign_stack();
std::fs::write("flake.nix", config.generate_flake_nix())?;
```

## Air-Gapped Workflow

1. **On networked machine**: Build and cache all dependencies
   ```bash
   nix build --out-link result
   nix copy --to file://./store result
   ```

2. **Transfer**: Copy `./store` to air-gapped machine

3. **On air-gapped machine**: Import and run
   ```bash
   nix copy --from file://./store result
   ./result/bin/entrenar train config.yaml
   ```
