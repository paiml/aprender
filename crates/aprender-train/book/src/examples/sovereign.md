# Sovereign Deployment

This example demonstrates air-gapped deployment with self-contained distributions.

## Running the Example

```bash
cargo run --example sovereign
```

## Code

```rust
{{#include ../../../examples/sovereign.rs}}
```

## Expected Output

```
=== Entrenar Sovereign Deployment Demo ===

--- Distribution Manifest ---
Core Distribution:
  Tier: Core
  Format: Tarball
  Components: 3 total
    - entrenar-core v0.5.6
    - trueno v0.2
    - aprender v0.1
  Filename: entrenar-sovereign-core-0.5.6.tar.gz
  Size: ~50MB

Full Distribution (ISO):
  Tier: Full
  Format: Iso
  Components: 11 total
  Filename: entrenar-sovereign-full-0.5.6.iso
  Size: ~500MB
```

## Distribution Tiers

| Tier | Components | Size | Use Case |
|------|------------|------|----------|
| Core | Essential libs | ~50MB | Minimal deployment |
| Standard | + CLI tools | ~150MB | Development |
| Full | + Models, docs | ~500MB | Air-gapped |

## Offline Installation

```bash
# Create sovereign distribution
entrenar sovereign create --tier full --output sovereign.iso

# Install on air-gapped system
entrenar sovereign install sovereign.iso --prefix /opt/entrenar
```
