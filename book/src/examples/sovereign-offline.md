# Case Study: Sovereign AI Offline Mode

This chapter covers APR's Sovereign AI capabilities, particularly the `--offline` mode that enables air-gapped deployments.

## Overview

**Sovereign AI** refers to AI systems that are fully controlled, operated, and audited by the user, without reliance on centralized APIs or proprietary cloud infrastructure.

Per Section 9.2 of the specification:

> **HARD REQUIREMENT**: The system must be capable of operating continuously in an "Air-Gapped" environment (no internet connection) once necessary artifacts are acquired.

## Compliance Checklist

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Local Execution | All inference runs on localhost via Rust/WASM | ✅ |
| Data Privacy | No telemetry; data never leaves the device | ✅ |
| Auditability | Open Source (Apache 2.0); Reproducible Builds | ✅ |
| Model Provenance | Cryptographic signatures in .apr footer | ✅ |
| Offline First | `apr run --offline` implemented | ✅ |
| Network Isolation | No std::net imports in inference code | ✅ |

## Using Offline Mode

### Basic Usage

```bash
# Run a model in offline mode (production recommended)
apr run --offline model.apr --input data.json

# Offline mode rejects uncached remote sources
apr run --offline hf://org/repo  # ERROR: OFFLINE MODE
```

### Caching Models First

```bash
# Step 1: Import model to cache (requires network)
apr import hf://TinyLlama/TinyLlama-1.1B -o tinyllama.apr

# Step 2: Run in offline mode (no network required)
apr run --offline tinyllama.apr --input prompt.txt
```

## Model Source Types

APR supports three model source types:

| Type | Example | Offline Behavior |
|------|---------|------------------|
| Local | `/path/to/model.apr` | Always allowed |
| HuggingFace | `hf://org/repo` | Requires cached |
| URL | `https://example.com/model.apr` | Requires cached |

## Network Isolation

The inference loop is designed to be **physically incapable** of network IO:

1. No `std::net` imports in inference code
2. No `reqwest` or HTTP client libraries
3. No `hyper` or async networking
4. Type-system enforced isolation

### Verification

Run the V11-V15 tests to verify network isolation:

```bash
cargo test --test spec_checklist_tests v1
```

## Example Code

```rust
//! Sovereign AI: Offline Mode Example
//! Run: cargo run --example sovereign_offline

use std::path::PathBuf;

fn main() {
    println!("=== Sovereign AI: Offline Mode Demo ===\n");

    // Demonstrate source types
    let sources = [
        ("model.apr", "Local"),
        ("hf://org/repo", "HuggingFace"),
        ("https://example.com/model.apr", "URL"),
    ];

    for (source, source_type) in sources {
        println!("{} -> {}", source, source_type);
    }

    // Offline mode behavior
    println!("\nOffline Mode:");
    println!("✅ Local files: Always allowed");
    println!("✅ Cached models: Allowed");
    println!("❌ Uncached HF: REJECTED");
    println!("❌ Uncached URLs: REJECTED");
}
```

Run the example:

```bash
cargo run --example sovereign_offline
```

## Cache Structure

Models are cached in `~/.apr/cache/`:

```
~/.apr/cache/
├── hf/
│   ├── openai/whisper-tiny/
│   └── TinyLlama/TinyLlama-1.1B/
└── urls/
    └── <hash>/  (first 16 chars of URL hash)
```

## Production Deployment

For production deployments:

1. **Pre-cache all models** during deployment
2. **Always use `--offline`** flag
3. **Verify network isolation** with integration tests
4. **Air-gap the inference environment** if required

```bash
# Deployment script
apr import hf://org/model -o /models/model.apr
chmod 444 /models/model.apr  # Read-only

# Runtime
apr run --offline /models/model.apr --input request.json
```

## Popperian Falsification

The offline mode implementation includes Popperian falsification tests:

| Test | Claim | Falsification |
|------|-------|---------------|
| V11 | Offline rejects uncached HF | Allows HF download |
| V12 | Offline rejects uncached URLs | Allows URL download |
| V13 | No network imports | std::net found |
| V14 | Spec mandates isolation | Missing mandate |
| V15 | CLI has --offline flag | Flag missing |

## References

- [Section 9.2: Sovereign AI Compliance](../../../docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md)
- [Local-First Software](https://www.inkandswitch.com/local-first/) (Kleppmann et al., 2019)
- [Example: sovereign_offline.rs](../../../examples/sovereign_offline.rs)
