# ptop - Pixel-Perfect System Monitor

`ptop` is a pixel-perfect TUI system monitor built with presentar-terminal, designed to replicate and extend ttop/btop functionality with advanced data science visualization capabilities.

> **Note:** ptop is shipped as [`ttop v2.0`](https://crates.io/crates/ttop) — install with `cargo install ttop`. The ptop module in presentar-terminal is the rendering engine; ttop is the published binary.

## Quick Start

```bash
# Install the published binary (recommended)
cargo install ttop

# Or build ptop directly from presentar-terminal
cargo run -p presentar-terminal --features ptop --bin ptop

# Or build in release mode for best performance
cargo build -p presentar-terminal --features ptop --bin ptop --release
./target/release/ptop
```

## Features

### System Panels (13 total)

| Panel | Description | Key Metrics |
|-------|-------------|-------------|
| CPU | Per-core usage with frequency | Load, Frequency, Temperature |
| Memory | RAM/Swap with breakdown | Used, Cached, Free, Swap |
| Disk | Storage and I/O rates | Usage, Read/Write rates |
| Network | Per-interface traffic | RX/TX rates, Total transferred |
| GPU | NVIDIA/AMD metrics | Utilization, VRAM, Temperature |
| Sensors | Temperature monitoring | CPU/GPU/NVMe temps |
| Battery | Power status (laptops) | Charge, State, Health |
| PSI | Pressure Stall Information | CPU/IO/Memory pressure |
| Processes | Process table | CPU%, MEM%, Command |
| Connections | Network connections | TCP states, Ports |
| Containers | Docker/Podman | CPU, Memory, Status |
| Files | Open file descriptors | Path, Type, Size |
| Treemap | Visual disk usage | Hierarchical size |

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle focused panel |
| `Enter` | Explode (fullscreen) focused panel |
| `Esc` | Exit explode mode / Exit filter |
| `j` / `k` | Navigate process list |
| `/` | Enter filter mode |
| `s` | Cycle sort column |
| `S` | Toggle sort direction |
| `k` | Kill selected process (signal dialog) |
| `?` / `h` | Show help overlay |
| `q` | Quit |

### Focus Indicators (WCAG AAA)

- Double-line border for focused panel
- Bright accent color blend
- `►` focus arrow in title
- Unfocused panels dimmed

## Modular Architecture (v0.3.0)

ptop uses a modular UI architecture with 15 TDD-tested modules:

```
src/ptop/ui/
├── mod.rs          # Module orchestration
├── colors.rs       # 42 tests - Color constants, gradients
├── helpers.rs      # 53 tests - format_bytes, format_uptime
├── overlays.rs     # 47 tests - Help, signal dialog, filter
├── core.rs         # Main drawing logic
└── panels/
    ├── mod.rs       # 26 tests - Panel border utilities
    ├── battery.rs   # 34 tests - Battery state, health
    ├── connections.rs # 33 tests - TCP states, ports
    ├── cpu.rs       # 29 tests - CPU title, meter layout
    ├── disk.rs      # 35 tests - I/O rates, segments
    ├── memory.rs    # 27 tests - Memory stats, ZRAM
    ├── network.rs   # 41 tests - Interface types, traffic
    ├── process.rs   # 49 tests - Column widths, state colors
    ├── psi.rs       # 38 tests - Pressure symbols, severity
    └── sensors.rs   # 41 tests - Temp thresholds, fan speed
```

Total: **601 TDD tests** in the UI module alone.

## Configuration

ptop supports YAML configuration:

```yaml
# ~/.config/ptop/config.yaml
version: "1.0"
refresh_ms: 1000

panels:
  cpu:
    enabled: true
    histogram: braille  # braille, block, ascii
    show_temperature: true
    show_frequency: true

  memory:
    enabled: true
    show_swap: true

  disk:
    enabled: true
    show_io_rates: true
```

### Command-Line Flags

```bash
ptop --help
ptop --refresh 500        # 500ms refresh
ptop --deterministic      # Predictable data (for testing)
ptop --config ~/my.yaml   # Custom config file
ptop --render-once        # Single frame then exit (CI/screenshots)
```

## Testing Examples

```bash
# Test connections analyzer
cargo run -p presentar-terminal --features ptop --example test_connections

# Test display rules
cargo run -p presentar-terminal --features ptop --example test_display_rules
```

## Performance

- **<16ms frame time** (60fps capable)
- **Zero allocations** in steady state
- **Differential rendering** (only changed cells)
- **Async data collection** (non-blocking)

## Related Documentation

- [Monitoring Examples](./monitoring-examples.md) - More terminal examples
- [Direct Backend](./direct-backend.md) - Architecture details
- [ptop Specification](../../../docs/specifications/pixel-by-pixel-demo-ptop-ttop.md) - Full specification (SPEC-024)
