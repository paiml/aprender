# ttop - Terminal Top

**10X Better Than btop** - A pure Rust system monitor with NVIDIA + AMD GPU support, zero-allocation rendering, and sovereign stack architecture.

[![Crates.io](https://img.shields.io/crates/v/ttop.svg)](https://crates.io/crates/ttop)
[![License](https://img.shields.io/crates/l/ttop.svg)](LICENSE)

## Installation

```bash
cargo install ttop
```

## What's New in v2

- **Sovereign Stack**: No ratatui dependency. Built on presentar-terminal with direct crossterm rendering.
- **Zero-Allocation Rendering**: CellBuffer + DiffRenderer with CompactString (24-byte inline). Steady-state heap allocs: 0.
- **NVIDIA + AMD GPU**: Auto-detects via nvidia-smi and sysfs. GPU panel is always visible.
- **Contract Enforcement**: provable-contracts YAML definitions verified at compile time.
- **Brick Architecture**: Tests define the interface. 78 falsification tests with proptest fuzzing.
- **2.5MB Binary**: Stripped release build.

## Features

- **Pure Rust**: Zero C dependencies for rendering, cross-platform (Linux)
- **< 1ms Frame Time**: Zero-allocation rendering at 80x24
- **14 Panels**: CPU, Memory, Disk, Network, Process, GPU, Battery, Sensors, PSI, System, Connections, Treemap, Files, Containers
- **GPU Monitoring**: NVIDIA (nvidia-smi) + AMD (sysfs gpu_busy_percent, hwmon)
- **Deterministic Mode**: Reproducible rendering for testing and CI
- **YAML Configuration**: `~/.config/ttop/config.yaml`

## Panels

| Panel | Description |
|-------|-------------|
| CPU | Per-core utilization with sparklines, frequency, load average |
| Memory | RAM/Swap/Cached with stacked bar, ZRAM ratio |
| Disk | Mount points, I/O rates, usage bars |
| Network | RX/TX throughput per interface with sparklines |
| Process | Sortable process table with tree view |
| GPU | NVIDIA/AMD utilization, VRAM, temperature, power, processes |
| Battery | Charge level and time remaining |
| Sensors | Temperature readings with health status |
| PSI | Pressure Stall Information (CPU, Memory, I/O) |
| Connections | TCP/UDP connections with service detection |
| Files | Open files, hot files, inode stats |
| Containers | Docker container CPU/memory |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `q`, `Esc` | Quit |
| `?` | Toggle help |
| `/` | Filter processes |
| `Tab` | Navigate panels |
| `Enter` | Explode (fullscreen) selected panel |
| `Arrow keys` | Navigate rows/columns |

## Command Line Options

```
ttop [OPTIONS]

Options:
  -r, --refresh <MS>       Refresh interval in milliseconds [default: 1000]
      --deterministic      Enable deterministic mode for testing
      --no-color           Disable colors
      --render-once        Render once to stdout and exit
      --explode <PANEL>    Explode a panel (cpu, memory, disk, network, gpu, etc.)
      --dump-config        Dump default configuration to stdout
  -c, --config <PATH>      Path to custom config file (YAML)
  -h, --help               Print help
  -V, --version            Print version
```

## Examples

```bash
# Run with default settings
ttop

# Fast refresh (500ms)
ttop -r 500

# Render single frame for CI/testing
ttop --deterministic --render-once --width 120 --height 40

# Explode GPU panel fullscreen
ttop --explode gpu

# Dump default config
ttop --dump-config > ~/.config/ttop/config.yaml
```

## GPU Support

| Vendor | Detection | Metrics |
|--------|-----------|---------|
| **NVIDIA** | nvidia-smi | Utilization, VRAM, Temperature, Power, Processes |
| **AMD** | sysfs (amdgpu driver) | Utilization, VRAM, Temperature, Power |

GPU panel is always visible. If no GPU is detected, it shows "No GPU detected".

## Architecture

```
ttop binary (2.5MB)
  └── presentar-terminal::ptop (14 panels, 13 analyzers)
        └── presentar-terminal::direct (CellBuffer, DiffRenderer)
              └── crossterm (terminal I/O)
```

No ratatui. No tui-rs. Zero-allocation steady-state rendering via `CompactString` + `BitVec` dirty tracking.

## Building from Source

```bash
git clone https://github.com/paiml/trueno-viz
cd trueno-viz/crates/ttop
cargo build --release
./target/release/ttop
```

## License

MIT OR Apache-2.0

---

Part of the [Aprender monorepo](https://github.com/paiml/aprender) — 70 workspace crates.
