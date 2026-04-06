# Data Directory

This directory contains versioned datasets for testing and benchmarking.

## Structure

```
data/
├── fixtures/           # Test fixtures (DVC tracked)
│   ├── cpu/           # CPU metric snapshots
│   ├── memory/        # Memory metric snapshots
│   └── network/       # Network metric snapshots
├── benchmarks/        # Benchmark results (DVC tracked)
│   └── baselines/     # Historical baseline measurements
└── experiments/       # Pre-registered experiment data
```

## Data Versioning

All data is tracked using DVC (Data Version Control):

```bash
# Pull latest data
dvc pull

# Check data status
dvc status

# Add new data
dvc add data/fixtures/new_fixture.json
git add data/fixtures/new_fixture.json.dvc
git commit -m "Add new fixture"
dvc push
```

## Random Seeds

All data generation uses fixed seeds from `params.yaml`:

| Parameter | Value | Usage |
|-----------|-------|-------|
| `seed.test` | 42 | Unit test RNG |
| `seed.proptest` | 0xdeadbeef | Property tests |
| `seed.benchmark` | 12345 | Benchmark data |

## Schema Validation

All datasets must pass JSON schema validation:

```bash
# Validate dataset
cargo run --bin validate-schema -- data/fixtures/cpu/48core.json
```

## References

- [DVC Documentation](https://dvc.org/doc)
- [params.yaml](../params.yaml) - Parameter configuration
- [dvc.yaml](../dvc.yaml) - Pipeline definition
