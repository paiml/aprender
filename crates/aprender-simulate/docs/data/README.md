# Data Documentation

This document describes the data formats, sources, and reproducibility practices used in simular.

## Random Seed Management

All simulations use deterministic random number generation for reproducibility.

### Seed Configuration

```yaml
# experiment.yaml
seed: 42
reproducibility:
  ieee_strict: true
  deterministic_order: true
```

### Implementation

```rust
use rand_pcg::Pcg64;
use rand::SeedableRng;

// Deterministic RNG from seed
let rng = Pcg64::seed_from_u64(seed);
```

### Verification

```bash
# Run same simulation 3 times - must produce identical results
cargo run -- verify experiments/tsp.yaml --runs 3
```

## Data Formats

### Input Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| YAML | `.yaml` | Experiment configuration |
| JSON | `.json` | City coordinates, parameters |
| CSV | `.csv` | Time series data |

### Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| Chrome Trace | `.json` | Performance profiling |
| Flame Graph | `.svg` | Visualization |
| Parquet | `.parquet` | Large datasets |
| CI Metrics | `.json` | Automation |

## TSP Instance Data

### California Cities Dataset

- **Source**: `data/california_cities.yaml`
- **Size**: 25-50 cities
- **Coordinates**: Latitude/longitude (WGS84)
- **Distance**: Haversine formula (great-circle)

### Format

```yaml
cities:
  - name: "San Francisco"
    lat: 37.7749
    lon: -122.4194
  - name: "Los Angeles"
    lat: 34.0522
    lon: -118.2437
```

### Data Integrity

- SHA-256 checksums for all datasets
- Version control for dataset changes
- Immutable test fixtures

## Benchmark Data

### Hardware Reference

All benchmarks run on reference hardware:
- **CPU**: AMD Ryzen 9 5950X (16 cores)
- **RAM**: 64GB DDR4-3600
- **Storage**: NVMe SSD
- **OS**: NixOS 24.05

### Statistical Requirements

| Metric | Requirement |
|--------|-------------|
| Sample size | n ≥ 100 |
| Confidence level | 95% |
| Outlier handling | MAD-based filtering |
| Warm-up iterations | 10 |

### Reporting Format

```json
{
  "benchmark": "tsp_grasp_25",
  "mean_ns": 1234567,
  "std_ns": 12345,
  "ci_lower_ns": 1232000,
  "ci_upper_ns": 1237000,
  "sample_size": 100,
  "confidence": 0.95
}
```

## Reproducibility Checklist

- [x] All seeds explicitly specified
- [x] RNG algorithm documented (PCG64)
- [x] IEEE 754 strict mode enabled
- [x] Deterministic iteration order
- [x] Cargo.lock committed
- [x] Nix flake for environment
- [x] Benchmark hardware documented
- [x] Data checksums verified

## Version Control

### Dataset Versioning

```
data/
├── v1/
│   ├── california_cities.yaml
│   └── checksums.sha256
└── v2/
    ├── california_cities.yaml
    └── checksums.sha256
```

### Model Checkpoints

For ML components (future):
```
models/
├── gp_surrogate_v1.bin
└── manifest.yaml
```

## Data Quality

### Validation Rules

1. **Completeness**: No missing required fields
2. **Consistency**: Cross-reference validation
3. **Accuracy**: Within domain bounds
4. **Timeliness**: Version dated

### Schema Validation

All data files validated against JSON Schema:

```bash
cargo run -- validate data/cities.yaml
```

## References

- [Parquet Format](https://parquet.apache.org/)
- [YAML Specification](https://yaml.org/spec/1.2.2/)
- [IEEE 754](https://ieeexplore.ieee.org/document/8766229)
