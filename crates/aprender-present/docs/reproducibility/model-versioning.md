# Model and Artifact Versioning

**Status:** Active
**Last Updated:** 2026-01-12

## Purpose

Track versions of all generated artifacts, models, and computed outputs to ensure reproducibility.

## Versioning Scheme

### Semantic Versioning for Artifacts

All artifacts follow `MAJOR.MINOR.PATCH`:

| Component | Version Format | Example |
|-----------|---------------|---------|
| Specification | `SPEC-NNN vX.Y.Z` | `SPEC-024 v8.0.0` |
| Widget API | `widget-api-X.Y.Z` | `widget-api-1.0.0` |
| Test fixtures | `fixtures-YYYY-MM-DD` | `fixtures-2026-01-12` |
| Benchmark baselines | `bench-X.Y.Z-ARCH` | `bench-1.0.0-x86_64` |

### Git Tags

```bash
# Tag format for releases
git tag -s v0.1.0 -m "Release 0.1.0"

# Tag format for spec versions
git tag -s spec-024-v8.0.0 -m "SPEC-024 version 8.0.0"

# Tag format for benchmark baselines
git tag -s bench-baseline-2026-01-12 -m "Benchmark baseline"
```

### Artifact Checksums

All released artifacts include SHA-256 checksums:

```bash
# Generate checksums
sha256sum target/release/ptop > ptop.sha256

# Verify
sha256sum -c ptop.sha256
```

## Data Version Control

### Test Fixture Management

Test fixtures are versioned in `tests/fixtures/`:

```
tests/fixtures/
├── v1/
│   ├── cpu_48core.json      # 48-core CPU mock data
│   ├── memory_128gb.json    # 128GB memory mock
│   └── manifest.json        # Fixture metadata
├── v2/
│   └── ...
└── current -> v2            # Symlink to current version
```

### Fixture Manifest

```json
{
  "version": "2",
  "created": "2026-01-12T00:00:00Z",
  "hardware_reference": "AMD Threadripper 7960X",
  "checksums": {
    "cpu_48core.json": "sha256:abc123...",
    "memory_128gb.json": "sha256:def456..."
  }
}
```

## Benchmark Baseline Management

### Recording Baselines

```bash
# Run benchmarks and record baseline
cargo criterion --save-baseline current

# Compare against baseline
cargo criterion --baseline current
```

### Baseline Storage

Baselines stored in `benches/baselines/`:

```
benches/baselines/
├── 2026-01-12-x86_64/
│   ├── criterion/
│   │   └── ... (criterion data)
│   └── metadata.json
└── latest -> 2026-01-12-x86_64
```

## Reproducibility Checklist

- [ ] All random seeds documented and controllable
- [ ] Test fixtures versioned with checksums
- [ ] Benchmark baselines stored with hardware metadata
- [ ] Git tags for all releases
- [ ] Dependency versions locked (Cargo.lock committed)
- [ ] Rust toolchain version pinned (rust-toolchain.toml)

## References

- [DVC (Data Version Control)](https://dvc.org/)
- [Reproducible Builds](https://reproducible-builds.org/)
