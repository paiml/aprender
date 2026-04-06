# Optimization Settings

The `[optimization]` section controls Phase 3 of the pipeline: SIMD vectorization, GPU dispatch, backend selection, and the Trueno compute backend.

## Top-Level Settings

```toml
[optimization]
profile = "balanced"
enable_simd = true
enable_gpu = false
gpu_threshold = 500
use_moe_routing = false
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `profile` | string | `"balanced"` | Optimization profile: `"fast"`, `"balanced"`, or `"aggressive"` |
| `enable_simd` | bool | `true` | Enable SIMD vectorization (AVX2/AVX-512/NEON) |
| `enable_gpu` | bool | `false` | Enable GPU dispatch via wgpu |
| `gpu_threshold` | integer | `500` | Minimum matrix dimension before GPU dispatch is considered |
| `use_moe_routing` | bool | `false` | Enable Mixture-of-Experts backend selection |

### Optimization Profiles

| Profile | Compile Time | Runtime | Use Case |
|---------|-------------|---------|----------|
| `fast` | Fastest | Good | Development iteration |
| `balanced` | Moderate | Better | Default for most projects |
| `aggressive` | Slowest | Best | Production, benchmarking |

## Backend Selection Thresholds

Batuta uses a cost-based backend selector based on the 5x PCIe rule (Gregg and Hazelwood, 2011). The `gpu_threshold` value sets the minimum matrix dimension at which GPU dispatch becomes profitable after accounting for host-to-device transfer overhead.

- Below the threshold: SIMD or scalar execution on CPU.
- Above the threshold: GPU dispatch if `enable_gpu` is true.

When `use_moe_routing` is enabled, a Mixture-of-Experts router learns from prior dispatch decisions and adjusts thresholds adaptively.

## Trueno Backend Configuration

```toml
[optimization.trueno]
backends = ["simd", "cpu"]
adaptive_thresholds = false
cpu_threshold = 500
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backends` | array | `["simd", "cpu"]` | Backend priority order (`"gpu"`, `"simd"`, `"cpu"`) |
| `adaptive_thresholds` | bool | `false` | Learn dispatch thresholds from runtime telemetry |
| `cpu_threshold` | integer | `500` | Element count below which scalar CPU is preferred over SIMD |

### Target Architecture Hints

The `backends` array is ordered by preference. Batuta tries each backend in order and falls back to the next if the preferred one is unavailable or below the dispatch threshold.

```toml
# GPU-first configuration for a machine with a discrete GPU
[optimization.trueno]
backends = ["gpu", "simd", "cpu"]
adaptive_thresholds = true
cpu_threshold = 256
```

```toml
# Conservative CPU-only configuration
[optimization.trueno]
backends = ["cpu"]
adaptive_thresholds = false
cpu_threshold = 0
```

The row-major tensor layout mandate (LAYOUT-002) applies to all backends. See the [Memory Layout](../part2/memory-layout.md) chapter for details.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
