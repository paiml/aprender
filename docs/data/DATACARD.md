# Data Card: simular Datasets

## Overview

This data card documents the datasets used in simular for benchmarking and testing.

## Dataset: California Cities TSP

### Basic Information

| Field | Value |
|-------|-------|
| Name | California Cities TSP |
| Version | 1.0 |
| Size | 25-50 cities |
| Format | YAML |
| License | Public Domain |

### Dataset Description

A collection of California city coordinates for Traveling Salesman Problem benchmarks.

**Purpose**: Benchmark GRASP algorithm performance on real-world geographic data.

**Source**: Publicly available geographic coordinates (Wikipedia, US Census).

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| name | string | City name |
| lat | float | Latitude (WGS84) |
| lon | float | Longitude (WGS84) |

### Example Record

```yaml
- name: "San Francisco"
  lat: 37.7749
  lon: -122.4194
```

### Data Quality

- **Completeness**: 100% (no missing values)
- **Accuracy**: GPS coordinates rounded to 4 decimal places (~11m precision)
- **Consistency**: All coordinates verified against Google Maps

### Preprocessing

1. Coordinates converted to decimal degrees
2. Haversine distance computed for routing
3. Normalized to unit square for visualization

### Intended Use

- TSP algorithm benchmarking
- Visualization demonstrations
- Educational examples

### Limitations

- Limited to California cities
- Does not include road distances (uses great-circle)
- Population bias toward larger cities

### Ethical Considerations

- No personal data
- Publicly available coordinates
- No known biases affecting simulation results

## Dataset: Random TSP Instances

### Basic Information

| Field | Value |
|-------|-------|
| Name | Random TSP Instances |
| Version | 1.0 |
| Size | n=10 to n=100 |
| Format | Generated |
| License | N/A (synthetic) |

### Generation Method

```rust
// Uniform random in unit square
let x = rng.gen::<f64>();
let y = rng.gen::<f64>();
```

### Seed Documentation

| Test | Seed | Purpose |
|------|------|---------|
| Reproducibility | 42 | Standard test seed |
| Stress test | 0-99 | 100-trial average |
| Edge cases | 12345 | Specific configurations |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-12 | Initial release |

## Contact

For questions about datasets, open an issue at:
https://github.com/paiml/simular/issues
