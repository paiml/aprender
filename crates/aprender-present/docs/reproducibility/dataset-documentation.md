# Dataset Documentation

**Status:** Active
**Last Updated:** 2026-01-12

## Purpose

Document all datasets used for testing, benchmarking, and validation of presentar.

## Test Datasets

### System Metrics Mock Data

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| `cpu_48core` | 48-core CPU utilization | 4.8 KB | AMD Threadripper 7960X |
| `cpu_8core` | 8-core CPU utilization | 800 B | Intel i7-12700 |
| `memory_128gb` | Memory stats | 512 B | 128GB DDR5 system |
| `memory_16gb` | Memory stats | 512 B | 16GB DDR4 system |
| `network_10gbe` | Network interface stats | 1.2 KB | 10GbE NIC |
| `disk_nvme` | NVMe disk stats | 2.1 KB | Samsung 990 Pro |
| `gpu_nvidia` | NVIDIA GPU stats | 1.8 KB | RTX 4090 |
| `gpu_amd` | AMD GPU stats | 1.5 KB | RX 7900 XTX |

### Temperature Sensor Layouts

| Layout | Description | Vendor |
|--------|-------------|--------|
| `k10temp_4ccd` | AMD k10temp with 4 CCDs | AMD |
| `k10temp_2ccd` | AMD k10temp with 2 CCDs | AMD |
| `coretemp_8core` | Intel coretemp | Intel |
| `coretemp_24core` | Intel coretemp (Xeon) | Intel |

### Process List Fixtures

| Fixture | Description | Processes |
|---------|-------------|-----------|
| `procs_light` | Minimal system | 50 |
| `procs_desktop` | Typical desktop | 300 |
| `procs_server` | Production server | 1500 |
| `procs_container` | Containerized workload | 800 |

## Data Collection Methodology

### Hardware Specifications

All data collected from real hardware with documented specifications:

```yaml
reference_system:
  cpu: AMD Threadripper 7960X
  cores: 48 (96 threads)
  ram: 128GB DDR5-5200
  storage: 2TB NVMe RAID-0
  gpu: NVIDIA RTX 4090
  os: Ubuntu 24.04 LTS
  kernel: 6.8.0-90-generic
```

### Collection Protocol

1. **Idle baseline**: System at rest, no user processes
2. **Load generation**: Synthetic load using `stress-ng`
3. **Capture**: Record metrics at 100ms intervals for 60s
4. **Validation**: Cross-reference with `top`, `htop`, `btop`

### Data Format

All datasets use JSON with schema validation:

```json
{
  "$schema": "https://presentar.dev/schemas/cpu-metrics-v1.json",
  "version": "1.0.0",
  "collected_at": "2026-01-12T10:00:00Z",
  "hardware": {
    "model": "AMD Threadripper 7960X",
    "cores": 48,
    "threads": 96
  },
  "samples": [
    {
      "timestamp_ms": 0,
      "per_core_percent": [12.5, 8.3, ...],
      "per_core_freq_mhz": [4800, 4750, ...],
      "per_core_temp_c": [65.0, 63.0, ...]
    }
  ]
}
```

## Data Quality Checks

### Validation Rules

| Check | Rule | Action on Failure |
|-------|------|-------------------|
| Schema | Matches JSON schema | Reject dataset |
| Range | Values within physical limits | Flag anomaly |
| Completeness | No missing required fields | Reject dataset |
| Consistency | Related values are coherent | Flag for review |

### Physical Limits

| Metric | Min | Max | Unit |
|--------|-----|-----|------|
| CPU % | 0 | 100 | percent |
| Temperature | -40 | 125 | °C |
| Frequency | 100 | 7000 | MHz |
| Memory | 0 | total | bytes |

## Privacy and Ethics

- No personally identifiable information (PII)
- No real user process names (anonymized)
- No network connection details (localhost only)
- Synthetic data preferred where possible

## References

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Data Documentation Initiative](https://ddialliance.org/)
