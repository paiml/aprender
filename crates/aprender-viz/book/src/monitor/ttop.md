# ttop - Terminal Top

**10X Better Than btop** - A pure Rust system monitor with NVIDIA + AMD GPU support, zero-allocation rendering, and sovereign stack architecture.

**Current Version: 2.0.0** ([crates.io](https://crates.io/crates/ttop))

## Installation

```bash
cargo install ttop
```

## What's New in v2

- **Sovereign Stack**: No ratatui. Built on presentar-terminal with direct crossterm rendering.
- **Zero-Allocation Rendering**: CellBuffer + DiffRenderer. Steady-state heap allocs: 0.
- **< 1ms Frame Time**: At 80x24 terminal size.
- **NVIDIA + AMD GPU**: Auto-detects via nvidia-smi and sysfs. GPU panel always visible.
- **Contract Enforcement**: provable-contracts YAML verified at compile time.
- **78 Falsification Tests**: Proptest fuzzing across all terminal sizes and panels.
- **2.5MB Binary**: Stripped release build.

## Architecture

```
ttop binary (2.5MB)
  └── presentar-terminal::ptop (14 panels, 13 analyzers)
        └── presentar-terminal::direct (CellBuffer, DiffRenderer)
              └── crossterm (terminal I/O)
```

## Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  CPU (per-core)  │  Memory (used/cached/free)  │  GPU + Sensors    │
├──────────────────┼─────────────────────────────┼───────────────────┤
│  Disk I/O        │  Network (RX/TX)            │  PSI / Battery    │
├──────────────────┴─────────────────────────────┴───────────────────┤
│  Processes (40%)  │  Connections (30%)  │  Files (30%)             │
└─────────────────────────────────────────────────────────────────────┘
```

## Panels (14)

| Panel | Description |
|-------|-------------|
| CPU | Per-core utilization with sparklines, frequency, load average |
| Memory | RAM/Swap/Cached stacked bar, ZRAM ratio, top consumers |
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
| System | Hostname, uptime, kernel |
| Treemap | Large file visualization |

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

```bash
ttop [OPTIONS]

Options:
  -r, --refresh <MS>       Refresh interval in milliseconds [default: 1000]
      --deterministic      Enable deterministic mode for testing
      --no-color           Disable colors
      --render-once        Render once to stdout and exit
      --explode <PANEL>    Explode a panel (cpu, memory, disk, gpu, etc.)
      --dump-config        Dump default configuration
  -c, --config <PATH>      Path to custom config file (YAML)
  -h, --help               Print help
  -V, --version            Print version
```

## GPU Support

| Vendor | Detection | Metrics |
|--------|-----------|---------|
| **NVIDIA** | nvidia-smi | Utilization, VRAM, Temperature, Power, Processes |
| **AMD** | sysfs (amdgpu driver) | Utilization, VRAM, Temperature, Power |

GPU is detected at startup. Verified on:
- NVIDIA RTX 4090 (sm_89)
- NVIDIA Orin (sm_87)
- NVIDIA Blackwell GB10 (sm_121)
- AMD Radeon Pro W5700X

## Configuration

YAML config at `~/.config/ttop/config.yaml`:

```bash
# Dump default config
ttop --dump-config > ~/.config/ttop/config.yaml
```

## Building from Source

```bash
git clone https://github.com/paiml/trueno-viz
cd trueno-viz/crates/ttop
cargo build --release
./target/release/ttop
```
