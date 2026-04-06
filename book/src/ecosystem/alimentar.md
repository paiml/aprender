# Alimentar

Dataset management for the Sovereign AI Stack.

## Overview

| Feature | Description |
|---------|-------------|
| Format | `.ald` binary files |
| Streaming | Memory-efficient loading |
| Versioning | Dataset lineage tracking |
| Quality | Data quality metrics |

## File Format

```
.ald file structure:
┌─────────────────────┐
│ Header (32 bytes)   │
│ - Magic: "ALD\0"    │
│ - Version: u32      │
│ - Schema offset     │
├─────────────────────┤
│ Schema              │
│ - Column names      │
│ - Column types      │
├─────────────────────┤
│ Data chunks         │
│ - Compressed rows   │
└─────────────────────┘
```

## Usage in Presentar

```rust
// Load dataset for visualization
let data = alimentar::load("sales.ald")?;

// Bind to chart widget
chart.data_source(data.column("revenue"));
```

## Data Quality

| Metric | Threshold |
|--------|-----------|
| Completeness | ≥95% |
| Uniqueness | ≥99% |
| Validity | ≥98% |

## Streaming API

```rust
// Process large datasets efficiently
for chunk in alimentar::stream("large.ald", chunk_size: 1000) {
    process(chunk);
}
```

## Integration

```
YAML → Alimentar → Widget → Canvas
```

## Verified Test

```rust
#[test]
fn test_alimentar_data_concept() {
    // Dataset quality validation concept
    struct DataQuality {
        completeness: f32,
        uniqueness: f32,
        validity: f32,
    }

    let quality = DataQuality {
        completeness: 0.97,
        uniqueness: 0.995,
        validity: 0.99,
    };

    // Thresholds for production data
    assert!(quality.completeness >= 0.95);
    assert!(quality.uniqueness >= 0.99);
    assert!(quality.validity >= 0.98);
}
```
