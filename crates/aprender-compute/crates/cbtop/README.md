# cbtop - Compute Block Top

Real-time load testing and hardware monitoring TUI built on the Brick Architecture.

## Installation

```bash
# Build from source
cargo build -p cbtop --release

# Run
./target/release/cbtop
```

## Examples

Explore the PMAT optimization modules with runnable examples:

```bash
# Federated Metrics Aggregation (PMAT-048)
# CRDT-based multi-host metrics with GCounter, LwwRegister, OrSet
cargo run --example federated_metrics_demo -p cbtop

# Adaptive ML Thresholds (PMAT-049)
# Workload-specific threshold learning with anomaly detection
cargo run --example adaptive_ml_demo -p cbtop

# Incremental Profile Snapshots (PMAT-050)
# Delta-compressed profile storage with keyframe intervals
cargo run --example incremental_snapshot_demo -p cbtop

# Predictive Scheduling Optimizer (PMAT-051)
# SLO-aware workload scheduling with cost optimization
cargo run --example predictive_scheduler_demo -p cbtop
```

## Features

- **Real-time Monitoring**: CPU, GPU, memory, network, disk, thermal metrics
- **Load Generation**: SIMD, CUDA, and wgpu compute workloads
- **Compute Scoring**: BrickScore framework (0-100) with letter grades
- **Multi-Backend**: Automatic backend selection (AVX2, CUDA, wgpu)
- **Deterministic Mode**: Reproducible benchmarks for testing

### PMAT Optimization Modules

| Module | Description | Key Types |
|--------|-------------|-----------|
| PMAT-048 | Federated Metrics Aggregation | `GCounter`, `LwwRegister`, `OrSet`, `MetricsFederation` |
| PMAT-049 | Adaptive ML Thresholds | `AdaptiveThresholdMl`, `WorkloadClass`, `TimeSeriesFeatures` |
| PMAT-050 | Incremental Profile Snapshots | `IncrementalSnapshotStore`, `ProfileSnapshot`, `SnapshotQuery` |
| PMAT-051 | Predictive Scheduling Optimizer | `PredictiveScheduler`, `HostProfile`, `SchedulingDecision` |

## Usage

```bash
# Basic monitoring
cbtop

# With load testing
cbtop --load medium --backend simd

# Headless benchmarking (JSON output)
cbtop bench --backend simd --workload gemm --duration 5 --format json
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-r, --refresh <MS>` | Refresh rate in milliseconds | 100 |
| `-b, --backend <TYPE>` | Backend: simd, wgpu, cuda, all | all |
| `-l, --load <LEVEL>` | Load: idle, light, medium, heavy, stress | idle |
| `-w, --workload <TYPE>` | Workload: gemm, conv, attention, bandwidth | gemm |
| `--deterministic` | Enable deterministic mode | false |

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Tab` | Next panel |
| `Space` | Start/Stop load generator |
| `b` | Cycle backend |
| `w` | Cycle workload type |

## Testing

```bash
# Run all tests
cargo test -p cbtop --all-features

# Run falsification tests
cargo test -p cbtop f301
```

## Documentation

See the full specification at `docs/specifications/compute-block-tui-cbtop.md` and the book chapter at `book/src/ecosystem/cbtop.md`.
