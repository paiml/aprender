# ptop Analyzers

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** 13 system analyzers, data sources, Analyzer trait, parity metrics with ttop.

---

## Parity Status

| Component | ttop Lines | ptop Lines | Parity |
|-----------|-----------|-----------|--------|
| Core UI | 7,619 | 8,542 | 100% |
| Analyzers | 12,847 | 13,105 | 100% |
| **Total** | 20,466 | 21,647 | **100%** |

## Analyzer Trait

```rust
pub trait Analyzer: Send + Sync + SelfDescribingBrick {
    fn name(&self) -> &'static str;
    fn collect(&mut self) -> Result<(), AnalyzerError>;
    fn interval(&self) -> Duration;
    fn available(&self) -> bool;
}
```

Analyzers are registered in `AnalyzerRegistry` and auto-detected at startup. Unavailable analyzers return `None`.

## Analyzer Inventory

### ConnectionsAnalyzer (1,200 lines)

| Field | Source |
|-------|--------|
| Local/remote addr + port | `/proc/net/tcp`, `/proc/net/tcp6` |
| TCP state | Parsed from hex state field |
| PID + process name | `/proc/[pid]/fd/` socket mapping |
| Locality (L/R) | RFC 1918 + IPv6 link-local/ULA detection |

### ContainersAnalyzer (420 lines)

| Field | Source |
|-------|--------|
| Container ID, name, image | Docker socket `/var/run/docker.sock` |
| CPU%, memory, net RX/TX | Podman socket `/run/podman/podman.sock` |
| PID count | cgroup stats `/sys/fs/cgroup/` |

### DiskEntropyAnalyzer (665 lines)

| Field | Source |
|-------|--------|
| Shannon entropy (0.0-1.0) | Sample reads from `/dev/[device]` |
| Encryption detection (LUKS) | `/sys/block/[device]/dm/`, `cryptsetup status` |

### DiskIoAnalyzer (930 lines)

| Field | Source |
|-------|--------|
| IOPS, latency, utilization | `/proc/diskstats` |
| Read/write bytes per second | Delta calculation across intervals |

### FileAnalyzer (1,340 lines)

| Field | Source |
|-------|--------|
| Open file descriptors | `/proc/[pid]/fd` |
| Hot files (recently accessed) | Access time tracking |
| Inode stats | `df` output |

### GpuProcsAnalyzer (290 lines)

| Field | Source |
|-------|--------|
| GPU utilization, VRAM | `nvidia-smi` (NVIDIA) |
| Temperature, power | AMDGPU sysfs fallback |
| Process type (G/C) | SM utilization, encoder/decoder |

### NetworkStatsAnalyzer (760 lines)

| Field | Source |
|-------|--------|
| Per-interface packet/error stats | `/proc/net/dev` |
| Protocol statistics (TCP/UDP/ICMP) | `/proc/net/snmp`, `/proc/net/netstat` |

### ProcessExtraAnalyzer (575 lines)

| Field | Source |
|-------|--------|
| cgroup, I/O priority | `/proc/[pid]/cgroup`, `/proc/[pid]/io` |
| OOM score + adjustment | `/proc/[pid]/oom_score`, `/proc/[pid]/oom_score_adj` |
| CPU affinity, NUMA node | `/proc/[pid]/status` (Cpus_allowed) |
| Scheduler, nice value | `sched_getaffinity()` |

### PsiAnalyzer (248 lines)

| Field | Source |
|-------|--------|
| CPU/Memory/IO pressure | `/proc/pressure/cpu`, `/proc/pressure/memory`, `/proc/pressure/io` |
| avg10, avg60, avg300, total | Parsed from PSI format |

### SensorHealthAnalyzer (1,030 lines)

| Field | Source |
|-------|--------|
| Temperature, fan, voltage | `/sys/class/hwmon/hwmon*/` |
| Critical/warning thresholds | `*_crit`, `*_max` sysfs files |
| Sensor status | Comparison against thresholds |

### StorageAnalyzer (800 lines)

| Field | Source |
|-------|--------|
| Mount points, usage | `/proc/mounts`, `df` stats |
| Filesystem type | Mount options |

### SwapAnalyzer (660 lines)

| Field | Source |
|-------|--------|
| Swap devices, usage | `/proc/swaps`, `/proc/meminfo` |
| ZRAM compression ratio | `/sys/block/zram*/` |

### TreemapAnalyzer (1,375 lines)

| Field | Source |
|-------|--------|
| File sizes, directory tree | Filesystem scanning with cache |
| Squarified layout | Bruls et al. (2000) algorithm |

### GeoIpAnalyzer (excluded)

Not planned. Excluded per no-external-databases policy.

## ComputeBlock Architecture

All analyzers follow the trueno ComputeBlock pattern for SIMD optimization where applicable:

```rust
pub trait ComputeBlock {
    type Input;
    type Output;
    fn compute(&mut self, input: &Self::Input) -> Self::Output;
    fn simd_supported(&self) -> bool;
    fn simd_instruction_set(&self) -> &'static str;
}
```

### SIMD-Vectorizable Elements

| ComputeBlock ID | Element | Vectorizable |
|-----------------|---------|--------------|
| CB-CPU-001 | Per-core sparklines | YES (f32x8 history) |
| CB-CPU-007 | Top N consumers | YES (parallel sort) |
| CB-MEM-001 | Per-segment sparklines | YES (4-channel history) |
| CB-CONN-001 | Connection age | YES (batch timestamp diff) |
| CB-CONN-003 | IP locality check | YES (IP range comparison) |
| CB-NET-002 | Protocol stats | YES (counter aggregation) |

## Gap Analysis Summary

All 13 analyzers (excluding GeoIP) are COMPLETE. Key remaining items:
- Process tree view mode (toggle with 't')
- Sparklines per memory/disk row
- PSI footer on relevant panels

## Performance

| Mode | Target |
|------|--------|
| Deterministic (simulated data) | < 100ms |
| Normal (first scan, 2600+ PIDs) | < 5s |
| Normal (cached, incremental) | < 500ms |
| Frame rate | 60fps (16ms poll) |

## References

- Bruls, M. et al. (2000). Squarified Treemaps. *Eurographics/IEEE Viz*.
- Shneiderman, B. (1992). Tree visualization with tree-maps. *ACM Trans. Graphics*.
