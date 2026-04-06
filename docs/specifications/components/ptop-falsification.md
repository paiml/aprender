# ptop Falsification Tests

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** F-series test catalog, pixel comparison framework, headless QA protocol, anti-regression.

---

## Test Summary

```
F500-F517: Analyzer Parity     18/18 PASS
F600-F650: Panel Features       32/32 PASS
F700-F730: Pixel Comparison     21/21 PASS
F800-F820: Data Accuracy        13/13 PASS
F900-F905: Anti-Regression       6/6  PASS
F1000+:    Feature Tests        96/96 PASS
TOTAL: 186 falsification tests
```

## Pixel Comparison Framework

### Methodology

Both ttop and ptop support `--deterministic` flag (frozen timestamps, fixed seed, static synthetic data). Captures are compared using:

1. **Character-Level Diff (CLD):** Threshold < 0.001 (0.1% difference)
2. **CIEDE2000 Color Diff (deltaE00):** Threshold < 1.0 (imperceptible)
3. **Structural Similarity (SSIM):** Threshold > 0.99

### Scoring (0-1000 scale, < 980 = FAIL)

| Deduction | Penalty |
|-----------|---------|
| Misaligned Column | -50 |
| Navigation Lag (>16ms) | -100 |
| "Ghost" Focus State | -200 |
| Clipped Title | -20 |
| Wrong Border Char | -10 |

## F500-F517: Analyzer Parity

| ID | Test | Criterion |
|----|------|-----------|
| F500 | ConnectionsAnalyzer exists | Module present |
| F501-F503 | Connections parsing | IPv4, IPv6, PID mapping |
| F504-F505 | ContainersAnalyzer | Docker socket detection |
| F506-F507 | DiskEntropyAnalyzer | Entropy calculation for encrypted disk |
| F508-F509 | ProcessExtraAnalyzer | OOM score parsing |
| F510-F511 | SensorHealthAnalyzer | hwmon enumeration |
| F512-F513 | GpuProcsAnalyzer | nvidia-smi detection |
| F514-F515 | TreemapAnalyzer | File scanning |
| F516-F517 | PsiAnalyzer | `/proc/pressure/*` parsing |

## F600-F631: Panel Features

| ID | Test | Criterion |
|----|------|-----------|
| F600-F602 | CPU | Governor, per-core freq, turbo indicator |
| F603-F605 | Memory | Huge pages, PSI, ZRAM ratio |
| F606-F608 | Disk | SMART, I/O scheduler, encryption |
| F609-F610 | Network | Packet drops, GeoIP |
| F611-F614 | Process | cgroup, ionice, OOM, affinity |
| F615-F617 | GPU | Per-process VRAM, temperature, power |
| F618-F620 | Containers | Docker, Podman, per-container stats |
| F621-F623 | Sensors | Fans, voltages, thresholds |
| F624-F626 | Treemap | Real files, sizes, inotify |
| F627-F631 | Files/Connections | Hot files, duplicates, all states, process mapping |

## F700-F720: Pixel Comparison

| ID | Test | Threshold |
|----|------|-----------|
| F700-F702 | Full screen CLD/deltaE/SSIM | < 0.001 / < 1.0 / > 0.99 |
| F703-F711 | Per-panel CLD/deltaE | < 0.001 to < 0.005 |
| F712-F715 | Header/Footer/Border/Graph compliance | Match config |
| F716-F720 | Gradient accuracy, alignment, padding, focus | Exact match |

## F800-F812: Data Accuracy

| ID | Test | Reference | Tolerance |
|----|------|-----------|-----------|
| F800 | CPU % | `mpstat` | +/- 5% |
| F801-F802 | Memory/Swap | `free -b` | +/- 1% |
| F803 | Disk usage | `df -B1` | +/- 1% |
| F804-F805 | Network RX/TX | `/proc/net/dev` | +/- 5% |
| F806 | Process count | `ps aux` | Exact |
| F807-F808 | Uptime/Load avg | `/proc/uptime` | +/- 1s / +/- 0.1 |
| F809 | Core count | `nproc` | Exact |
| F810-F812 | Temp/Connections/Containers | `sensors`/`ss`/`docker ps` | +/- 2C / Exact |

## F900-F905: Anti-Regression

| ID | Test | Criterion |
|----|------|-----------|
| F900 | No simulated data | `grep -r "simulate_"` in `src/ptop/` = 0 |
| F901 | CIELAB precision | deltaE < 0.1 for interpolation midpoint |
| F902 | Source attribution | All widgets reference btop/ttop |
| F903 | Symbol integrity | Braille and Block definitions present |
| F904 | Dependency gate | ptop feature is optional (not default) |
| F905 | No magic numbers | Layout constants named |

## Feature-Specific Tests

### Process Tree (F-TREE-001 to F-TREE-003)
- Orphaned child hierarchy, live re-parenting, deep nesting overflow

### Network Protocol Stats (F-NET-005 to F-NET-007)
- UDP rate correlation, TCP retransmission spikes, ICMP counter increments

### Connection Locality (F-CONN-008 to F-CONN-009)
- RFC 1918 compliance, IPv6 link-local handling

### YAML Config (F1000-F1010)
- Config loading, panel enable/disable, position override, hot reload, XDG compliance

### Space Packing (F1020-F1030)
- Grid snap, minimum size, reflow on resize, no overlap/gaps, 60fps reflow

### SIMD Optimization (F1040-F1055)
- SIMD enabled, frame < 16ms, zero alloc in render, ComputeBrick usage, cache hit rate

### Navigation/Explode (F1060-F1075)
- Tab cycling, Enter explode, Esc collapse, focus visible, hjkl, 1-9 toggle

### Dynamic Customization (F1080-F1095)
- Auto-expand, detail levels, GPU G/C badges, sparkline history, min/max enforced

## Headless QA Protocol

### Mandatory Commands

```bash
# CORRECT: Always use cargo run --release
cargo run -p presentar-terminal --bin ptop --features ptop --release -- \
  --render-once --width 120 --height 40

# Deterministic mode (no /proc scan)
cargo run -p presentar-terminal --bin ptop --features ptop --release -- \
  --deterministic --render-once --width 120 --height 40
```

### Automated Checks

```bash
output=$($PTOP --render-once --width 150 --height 40 2>&1)

# F-CPU-001: Meter format validation
# F-CPU-002: No column bleeding (no 4+ digit sequences)
# F-PANEL-001: All panels present (CPU, Memory, Disk, Network, Process)
# F-PERF-001: Deterministic render < 200ms
```

### Performance Targets

| Mode | Target | Acceptable |
|------|--------|------------|
| Deterministic | < 100ms | < 200ms |
| Normal (first) | < 5s | < 8s |
| Normal (cached) | < 500ms | < 1s |

## Falsification Summary by Domain

| Domain | Test Count | Category |
|--------|-----------|----------|
| Analyzer Parity | 18 | F500-F517 |
| Panel Features | 32 | F600-F631 |
| Pixel Comparison | 21 | F700-F720 |
| Data Accuracy | 13 | F800-F812 |
| Anti-Regression | 6 | F900-F905 |
| YAML Config | 11 | F1000-F1010 |
| Space Packing | 11 | F1020-F1030 |
| SIMD | 16 | F1040-F1055 |
| Navigation | 16 | F1060-F1075 |
| Dynamic | 16 | F1080-F1095 |
| Process/Net/Conn | 8 | F-TREE, F-NET, F-CONN |
| Input Handling | 4 | F-INPUT-001 to 004 |
| Architecture | 1 | F-ARCH-001 |
| PMAT Scorer | 20 | F-PMAT-001 to 020 |
