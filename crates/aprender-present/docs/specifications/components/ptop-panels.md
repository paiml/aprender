# ptop Panels

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** 14 panel implementations, layout system, widget inventory, color system, YAML configuration.

---

## Panel Implementation Status

| # | Panel | ttop Lines | ptop Status | Priority |
|---|-------|-----------|-------------|----------|
| 1 | CPU | 61-307 | DONE | P0 |
| 2 | Memory | 310-661 | DONE | P0 |
| 3 | Disk | 663-1003 | DONE | P0 |
| 4 | Network | 1005-1496 | DONE | P0 |
| 5 | Process | 2497-2675 | DONE | P0 |
| 6 | GPU | 1498-1993 | DONE (nvidia-smi/sysfs) | P1 |
| 7 | Battery | 1995-2052 | DONE (/sys/power_supply) | P2 |
| 8 | Sensors | 2055-2258 | DONE | P1 |
| 9 | PSI | 2261-2342 | DONE | P1 |
| 10 | System | 2345-2385 | DONE | P2 |
| 11 | Connections | 2677-2800 | DONE | P1 |
| 12 | Treemap | 2807-2830 | DONE | P3 |
| 13 | Files | 3062-3250 | DONE | P3 |
| 14 | Containers | - | DONE | P1 |

## Layout System

### Grid Layout (matching ttop)

- **Top panels:** 45% height, adaptive 2-column grid
- **Bottom row:** 55% height, 3-column (40/30/30): Process | Connections | Treemap/Files
- Responsive to terminal resize; minimum size handling with graceful degradation
- All panels use btop-style rounded borders (`BorderStyle::Rounded`)

### Adaptive Detail Levels

| Level | Min Height | Components |
|-------|-----------|------------|
| Minimal | 6 | Title + single utilization bar |
| Compact | 9 | + VRAM/secondary bar, basic stats |
| Normal | 15 | + Thermal, Power, Clock |
| Expanded | 20+ | + Process list, history graphs |
| Exploded | Full | Full-screen single panel |

## Panel Specifications

### CPU Panel
- **Title:** ` CPU {pct}% | {cores} cores | {freq}GHz | up {time} | LAV {load} `
- Per-core meters (format: `NN ██████ XXX`), CPU history graph (Block mode)
- Load gauge with trend arrows (up/down/stable), top 3 CPU consumers
- Frequency display with boost icon, percent_color gradient

### Memory Panel
- **Title:** ` Memory | {used}G / {total}G ({pct}%) | ZRAM:{ratio}x `
- Stacked memory bar (Used|Cached|Free), breakdown rows
- ZRAM ratio with compressed/original size and algorithm

### Disk Panel
- **Title:** ` Disk | R: {rate}/s | W: {rate}/s | {used}G / {total}G `
- Per-mount usage bars with percent_color
- I/O rates from `/proc/diskstats`

### Network Panel
- **Title:** ` Network ({iface}) | RX {rx}/s | TX {tx}/s `
- Interface sparklines, RX (cyan) / TX (red) colors
- Primary interface auto-detection (excludes loopback)

### Process Panel
- **Title:** ` Processes ({count}) | Sort: {col} {dir} | Filter: "{filter}" `
- Columns: PID, S, C%, M%, COMMAND. State color coding (R=green, D=orange, Z=red)
- Tree view mode, selection highlighting

### GPU Panel
- **Title:** ` {gpu_name} | {temp}C | {power}W `
- Utilization/VRAM bars, temperature color coding (green<70, yellow<85, red>85)
- Process list with G/C type badges (Graphics=magenta, Compute=cyan)

### Connections Panel
- **Title:** ` Connections | {active} active | {listen} listen `
- Columns: SVC, LOCAL, REMOTE, GEO, ST, AGE, PROC
- Service detection (port-to-name), locality indicator (L/R)

### Other Panels
- **Battery:** Charge meter (inverted color), time remaining, status icon
- **Sensors:** Per-sensor health indicator, temperature color coding, type characters (C/G/D/F/M)
- **PSI:** CPU/Memory/I/O pressure with severity symbols and color escalation
- **System:** Hostname, kernel version, container detection
- **Treemap/Files:** Mount legend, file list with type/IO/entropy/dup markers

## Color System

### Border Colors (matching ttop/theme.rs)

| Panel | Hex |
|-------|-----|
| CPU | #64C8FF |
| Memory | #B478FF |
| Disk | #64B4FF |
| Network | #FF9664 |
| Process | #DCC464 |
| GPU | #64FF96 |
| Battery | #FFDC64 |
| Sensors | #FF6496 |
| PSI | #C85050 |
| Connections | #78B4DC |
| Files | #B48C64 |

### Percent Color Gradient (5-stop)

| Range | Color |
|-------|-------|
| 0-25% | Cyan to Green |
| 25-50% | Green to Yellow |
| 50-75% | Yellow to Orange |
| 75-90% | Orange to Red |
| 90-100% | Bright Red |

## Widget Inventory

**Core:** Border, Text, Layout (flexbox rows/columns/constraints)

**Charts:** Graph, LineChart, Histogram, Heatmap, ScatterPlot, BoxPlot, ViolinPlot, ForceGraph, RocPrCurve, LossCurve, HorizonGraph, Sparkline

**Gauges:** Gauge, Meter, SegmentedMeter, MemoryBar, MultiBar

**Panels:** CpuGrid, ProcessTable, NetworkPanel, ConnectionsPanel, FilesPanel, GpuPanel, SensorsPanel, ContainersPanel

**Interactive:** TextInput, Scrollbar, CollapsiblePanel, Tree, Treemap, ConfusionMatrix

## Navigation & Explode (Feature D)

| Key | Action |
|-----|--------|
| Tab / Shift+Tab | Cycle panel focus |
| h/j/k/l | Vim-style navigation |
| Enter / z | Toggle explode (fullscreen) |
| Esc | Collapse exploded view |
| 1-9 | Toggle panel visibility |
| 0 | Reset all panels |

Focus indicated by thicker/colored border. Explode fills >= 95% of terminal. Navigation < 16ms (input-first loop).

## YAML Configuration

```yaml
layout: { columns: 3, min_panel_width: 30, min_panel_height: 8 }
panels:
  cpu: { enabled: true, position: [0, 0], detail_level: normal }
  # ...
theme:
  cpu_color: "#64C8FF"
refresh: { interval_ms: 1000 }
keybindings:
  toggle_panel: "1-9"
  explode_panel: ["Enter", "z"]
```

Supports `--config`, `--dump-config`, `--dump-default-config` CLI flags. XDG-compliant config paths.

## Defects Resolved (19/19)

All 15 defects from 2026-01-10 live testing plus 4 additional defects resolved. Critical fixes: memory 0.0G (refresh_memory), CPU 0% (two-refresh delta), TRANSPARENT-to-black (Color::Reset), column overflow, tab hang (input-first loop).
