# ttop v2 Specification: Sovereign TUI Monitor

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version:** 2.0.0-draft
**Status:** RFC
**Date:** 2026-04-02
**Supersedes:** ttop 0.3.x (ratatui-based)

## Table of Contents

1. [Rationale](#1-rationale)
2. [Architecture](#2-architecture)
3. [Sovereign Dependency Policy](#3-sovereign-dependency-policy)
4. [Contract Enforcement](#4-contract-enforcement)
5. [Testing-First (Brick Architecture)](#5-testing-first-brick-architecture)
6. [Rendering Backend](#6-rendering-backend)
7. [Panel System](#7-panel-system)
8. [Data Collection](#8-data-collection)
9. [Configuration](#9-configuration)
10. [Quality Gates](#10-quality-gates)
11. [Migration Plan](#11-migration-plan)
12. [References](#12-references)

## Component Specifications

| Document | Scope |
|----------|-------|
| [Rendering Backend](components/rendering-backend.md) | CellBuffer, DiffRenderer, DirectCanvas, zero-alloc design |
| [Contract Enforcement](components/contract-enforcement.md) | YAML contracts, build.rs enforcement, Kani proof harnesses |
| [Brick Testing](components/brick-testing.md) | Probar Brick/BrickHouse, falsification tests, pixel coverage |
| [Panel Architecture](components/panel-architecture.md) | Panel trait, widget composition, ComputeBlock SIMD |
| [Migration Plan](components/migration-plan.md) | Phase-by-phase ratatui removal, parity validation |

---

## 1. Rationale

### 1.1 Problem Statement

ttop 0.3.x depends on ratatui (15+ transitive deps, ~450 LOC adapter layer,
2 buffer copies per frame). This violates the Sovereign AI Stack principle:
every layer must be PAIML-owned or formally verified.

### 1.2 Evidence

presentar-terminal already implements `ptop` — a pixel-perfect ttop clone
using direct crossterm with zero-allocation steady-state rendering. The
technology is proven (186 falsification tests, SPEC-024 compliance).

### 1.3 Goals

| Goal | Metric | Target |
|------|--------|--------|
| Zero external UI deps | Transitive dep count | crossterm only |
| Zero-alloc steady state | Heap allocs/frame | 0 |
| Frame budget | Render time (80x24) | < 1ms |
| Contract coverage | Obligations verified | 100% |
| Test coverage | Line coverage | >= 95% |
| Mutation score | cargo-mutants | >= 80% |
| Pixel parity | CLD vs ttop 0.3.x | < 0.1% diff |

### 1.4 Non-Goals

- WASM target (ttop is a native system monitor)
- GPU-accelerated rendering (terminal is CPU-bound)
- Backwards compatibility with ratatui API

### 1.5 Academic Foundations

Popper (1963): falsification testing. Tufte (1983): data-ink ratio.
Nielsen (1994): usability heuristics. Denning (1968): Working Set Model.
Little (1961): L=lambda*W. Meyer (1992): Design by Contract.
Ohno (1988): Toyota Production System (Jidoka, Poka-yoke, Muda).

---

## 2. Architecture

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  ttop v2 Binary                                             │
│  main.rs: event loop, CLI (clap)                            │
├─────────────────────────────────────────────────────────────┤
│  App Layer                                                  │
│  app.rs: state, collectors (sysinfo), key handling          │
├─────────────────────────────────────────────────────────────┤
│  Panel Layer (Brick Architecture)                           │
│  panels/: 14 panels, each impl PanelBrick + ComputeBlock    │
├─────────────────────────────────────────────────────────────┤
│  Widget Layer (presentar-terminal)                          │
│  Border, Gauge, Sparkline, LineChart, ProcessTable, etc.    │
├─────────────────────────────────────────────────────────────┤
│  Rendering Layer (presentar-terminal::direct)               │
│  CellBuffer → DiffRenderer → crossterm → stdout             │
├─────────────────────────────────────────────────────────────┤
│  Contract Layer (provable-contracts)                        │
│  YAML contracts → build.rs → compile-time enforcement       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow (Unidirectional)

```
crossterm::event → App::handle_key() → App state mutation
                                              │
sysinfo/procfs → App::collect_metrics() ──────┤
                                              ▼
                     CellBuffer::clear() → Panel::render(&app, &mut buf)
                                              │
                                              ▼
                     DiffRenderer::render_diff(prev, curr) → stdout
```

### 2.3 Module Structure

```
crates/ttop/
├── contracts/                  # provable-contracts YAML
│   └── panel-render-v1.yaml    # Panel + data contracts
├── src/
│   ├── main.rs                 # Event loop (CellBuffer + DiffRenderer)
│   ├── lib.rs                  # Re-exports + include_str! enforcement
│   └── generated_contracts.rs  # Contract macros (from YAML)
├── tests/
│   ├── brick_interface.rs      # Brick Architecture interface (P-series)
│   └── panel_falsification.rs  # Falsification tests (F-series)
└── build.rs                    # Contract env var emission
```

All panel logic lives in `presentar-terminal::ptop` (14 panels, 13 analyzers,
app state, config, UI layout). ttop is a thin sovereign binary wrapper.

---

## 3. Sovereign Dependency Policy

### 3.1 Allowed Dependencies

| Crate | Owner | Purpose | Justification |
|-------|-------|---------|---------------|
| `presentar-terminal` | PAIML | Rendering | Sovereign stack |
| `presentar-core` | PAIML | Types (Rect, Color) | Sovereign stack |
| `provable-contracts` | PAIML | Contract enforcement | Sovereign stack |
| `jugar-probar` | PAIML | Testing framework | Sovereign stack |
| `batuta-common` | PAIML | Shared formatting | Sovereign stack |
| `crossterm` | External | Terminal I/O | Audited, no alternative |
| `sysinfo` | External | System metrics | Audited, used by ptop |
| `clap` | External | CLI parsing | Audited, industry standard |
| `serde` + `serde_yaml_ng` | External | Configuration | Audited |

### 3.2 Banned Dependencies

| Crate | Reason |
|-------|--------|
| `ratatui` | Replaced by presentar-terminal |
| `tui` | Predecessor of ratatui, same issue |
| `termion` | Replaced by crossterm |

### 3.3 Enforcement

`deny.toml` at workspace root bans ratatui and tui crates:

```toml
[[bans.deny]]
name = "ratatui"
wrappers = []

[[bans.deny]]
name = "tui"
wrappers = []
```

---

## 4. Contract Enforcement

> Full specification: [Contract Enforcement](components/contract-enforcement.md)

### 4.1 Overview

Every public function in ttop v2 has a YAML contract definition in
`contracts/`. The `build.rs` script uses `provable-contracts` to generate
Rust enforcement code at compile time.

### 4.2 Contract Categories

| Category | Scope | Example |
|----------|-------|---------|
| Panel | Render functions | `render() output fits within Rect bounds` |
| Data | Collectors | `cpu_percent in 0.0..=100.0` |
| Layout | Grid math | `sum(panel_heights) <= terminal_height` |
| Theme | Color functions | `RGB components in 0..=255` |

### 4.3 Build Enforcement Pattern

```rust
// build.rs
fn main() {
    provable_contracts::build_helper::generate(
        "contracts/",
        "src/generated_contracts.rs",
    ).expect("contract generation");
    println!("cargo:rerun-if-changed=contracts/");
}
```

### 4.4 Compile-Time Test Enforcement (from presentar)

```rust
// src/panels/mod.rs — tests MUST exist or build fails
#[doc(hidden)]
pub const _ENFORCE_PANEL_TESTS: &str =
    include_str!("../tests/brick_interface.rs");
#[doc(hidden)]
pub const _ENFORCE_FALSIFICATION: &str =
    include_str!("../tests/panel_falsification.rs");
```

---

## 5. Testing-First (Brick Architecture)

> Full specification: [Brick Testing](components/brick-testing.md)

### 5.1 Principle

**Tests define the interface. Implementation follows.**

No panel can be implemented without its Brick assertions existing first.
This is enforced at compile time via `include_str!` (Section 4.4).

### 5.2 Brick Trait

Every panel implements `Brick` from `jugar-probar`:

```rust
pub trait Brick: Send + Sync {
    fn brick_name(&self) -> &'static str;
    fn assertions(&self) -> Vec<BrickAssertion>;
    fn budget(&self) -> BrickBudget;
    fn verify(&self) -> BrickVerification;
}
```

### 5.3 BrickHouse Composition

The full ttop UI is a `BrickHouse` with a 16ms frame budget:

```
BrickHouse("ttop", 16ms)
├── CpuBrick(2ms)
├── MemoryBrick(2ms)
├── DiskBrick(1ms)
├── NetworkBrick(1ms)
├── ProcessBrick(3ms)
├── GpuBrick(1ms)
├── SensorsBrick(1ms)
├── ConnectionsBrick(1ms)
├── FilesBrick(1ms)
└── OverlayBrick(1ms)
Budget: 14ms / 16ms = 87.5% utilization
```

### 5.4 Falsification Test Categories

| Series | Scope | Count |
|--------|-------|-------|
| F500 | Analyzer parity | 18 |
| F600 | Panel features | 32 |
| F700 | Pixel comparison | 21 |
| F800 | Data accuracy | 13 |
| F900 | Anti-regression | 6 |
| F1000+ | Feature tests | 96+ |

---

## 6. Rendering Backend

> Full specification: [Rendering Backend](components/rendering-backend.md)

### 6.1 CellBuffer

Pre-allocated grid of `Cell` structs (symbol + fg + bg + modifiers).
Uses `CompactString` (24-byte inline) for zero-allocation grapheme storage.
`BitVec` dirty tracking for minimal diff rendering.

### 6.2 DiffRenderer

Scans dirty bits, emits minimal ANSI escape sequences, single `flush()`.
80x24 full redraw < 1ms. 10% partial update < 0.1ms.

### 6.3 Performance Comparison

| Metric | ratatui (ttop 0.3.x) | Direct (ttop v2) |
|--------|---------------------|------------------|
| Dependencies | ~15 | ~4 |
| Buffer copies | 2 | 1 |
| Frame overhead | ~0.5ms | ~0.1ms |
| Heap allocs/frame | Many | 0 (steady state) |

---

## 7. Panel System

> Full specification: [Panel Architecture](components/panel-architecture.md)

### 7.1 PanelBrick Trait

```rust
pub trait PanelBrick: Brick {
    fn render(&self, app: &App, buf: &mut CellBuffer, area: Rect);
    fn title(&self, app: &App) -> String;
    fn min_size(&self) -> (u16, u16);
    fn detail_level(&self, height: u16) -> DetailLevel;
}
```

### 7.2 Panel Inventory (14 panels)

| Panel | Widgets Used | SIMD | Priority |
|-------|-------------|------|----------|
| CPU | CpuGrid, Gauge, Sparkline | AVX2 (history) | P0 |
| Memory | MemoryBar, Gauge | - | P0 |
| Disk | Gauge, Sparkline | - | P0 |
| Network | Sparkline, LineChart | - | P0 |
| Process | ProcessTable | - | P0 |
| **GPU** | **GpuPanel, Gauge, Sparkline** | - | **P0 (MUST)** |
| Battery | Gauge | - | P2 |
| Sensors | Heatmap, Gauge | - | P1 |
| PSI | Gauge | - | P1 |
| System | Text | - | P2 |
| Connections | ConnectionsPanel | - | P1 |
| Treemap | Treemap | - | P3 |
| Files | FilesPanel | - | P3 |
| Containers | ContainersPanel | - | P1 |

### 7.3 GPU Panel — MANDATORY (NVIDIA + AMD)

GPU monitoring is a P0 requirement. ttop MUST support both vendors:

| Vendor | Detection | Data Source | Metrics |
|--------|-----------|-------------|---------|
| **NVIDIA** | `nvidia-smi` binary or NVML | `nvidia-smi --query-gpu` / sysfs | Util%, VRAM, Temp, Power, Clock, Procs |
| **AMD** | `/sys/class/drm/card*/device/` | sysfs `gpu_busy_percent`, `mem_info_*` | Util%, VRAM, Temp, Power |

**Detection priority:** NVIDIA first (nvidia-smi), then AMD (sysfs).
If neither detected, panel shows "No GPU detected" (not hidden).

**GPU metrics contract:**
```yaml
- name: gpu_utilization_range
  postconditions:
    - "utilization.map_or(true, |u| u <= 100)"
    - "temperature.map_or(true, |t| t <= 200)"
    - "vram_used.map_or(true, |v| v <= vram_total.unwrap_or(u64::MAX))"
```

**Falsification tests (MUST pass):**
- F026: Exploded GPU renders without panic (any hardware)
- F110: GPU panel shows "No GPU" when neither vendor present
- F111: NVIDIA metrics parse correctly from nvidia-smi output
- F112: AMD metrics parse correctly from sysfs paths

**Hardware test matrix:**
| Machine | GPU | Arch | Test |
|---------|-----|------|------|
| Local | RTX 4090 | sm_89 | `ssh localhost` |
| Jetson | Orin | sm_87 | `ssh jetson` |
| gx10 | Blackwell | sm_121 | `ssh gx10` |
| CI | None | - | Graceful degradation |

### 7.3 Layout Grid

Top panels: 45% height, adaptive 2-column.
Bottom row: 55% height, 3-column (40/30/30).
Exploded mode: single panel fullscreen (Tab key).

---

## 8. Data Collection

### 8.1 Collectors (via sysinfo + procfs)

| Collector | Source | Refresh |
|-----------|--------|---------|
| CPU | sysinfo + `/proc/stat` | 1s |
| Memory | sysinfo + `/proc/meminfo` | 1s |
| Disk | `/proc/diskstats` | 1s |
| Network | sysinfo + `/proc/net/dev` | 1s |
| Process | sysinfo | 2s |
| GPU (NVIDIA) | nvidia-smi / NVML / sysfs | 2s |
| GPU (AMD) | `/sys/class/drm/card*/device/` sysfs | 2s |
| Battery | `/sys/class/power_supply/` | 5s |
| Sensors | `/sys/class/hwmon/` | 2s |
| PSI | `/proc/pressure/*` | 1s |

### 8.2 Analyzers (13 modules)

Connections, Containers, DiskEntropy, DiskIo, FileAnalyzer,
GpuProcs, NetworkStats, ProcessExtra, Psi, SensorHealth,
Storage, Swap, Treemap.

Each analyzer implements the `Analyzer` trait with `update()` and
`metrics()` methods, contract-enforced via YAML.

---

## 9. Configuration

### 9.1 YAML Config (`~/.config/ttop/config.yaml`)

```yaml
refresh_ms: 1000
panels:
  cpu: true
  memory: true
  disk: true
  network: true
  process: true
  gpu: true       # MUST: always shown (NVIDIA + AMD auto-detect)
  battery: auto   # auto-detect power supply
  sensors: true
  psi: true
  connections: true
  files: false    # off by default
theme: default
```

### 9.2 Deterministic Mode

`--deterministic` flag freezes timestamps, uses fixed seed, produces
static synthetic data. Required for pixel comparison tests (F700 series).

---

## 10. Quality Gates

### 10.1 Tier System

| Tier | Trigger | Time | Checks |
|------|---------|------|--------|
| T1 | On save | < 1s | fmt, clippy, check |
| T2 | Pre-commit | < 5s | test --lib, contract lint |
| T3 | Pre-push | < 2min | full tests, probar, coverage |
| T4 | CI/CD | < 10min | clean-room, mutants, parity |

### 10.2 Mandatory Gates

- `provable-contracts lint` — all contracts valid
- `cargo test` — 0 failures
- Probar Brick verification — all assertions pass
- BrickHouse budget — < 16ms total
- Pixel parity — CLD < 0.1% vs reference
- Line coverage >= 95%
- Mutation score >= 80%
- Clean-room build passes

### 10.3 Scoring (0-1000, per SPEC-024)

Score < 980 = FAIL. Deductions: misaligned column (-50), nav lag >16ms (-100),
ghost focus (-200), clipped title (-20), wrong border (-10), contract violation (-500).

---

## 11. Migration Plan

> Full specification: [Migration Plan](components/migration-plan.md)

### 11.1 Phases

| Phase | Scope | Duration | Gate |
|-------|-------|----------|------|
| 0 | Spec + contracts + test stubs | 1 day | Contracts compile |
| 1 | Rendering backend swap | 1 day | CellBuffer renders |
| 2 | Panel migration (P0: CPU, Mem, Disk, Net, Process) | 2 days | F600 pass |
| 3 | Panel migration (P1-P3: remaining 9 panels) | 2 days | F600 full |
| 4 | Pixel parity validation | 1 day | F700 pass, score >= 980 |
| 5 | Clean-room + publish | 1 day | Clean-room green |

### 11.2 Strategy

Port from presentar-terminal's `ptop` module, which already implements
all 14 panels with the sovereign stack. The migration is a structured
extraction, not a rewrite from scratch.

---

## 12. References

1. Popper (1963) *Conjectures and Refutations*; 2. Tufte (1983) *Visual Display*;
3. Nielsen (1994) *Usability Engineering*; 4. Denning (1968) Working Set Model;
5. Little (1961) L=lambda*W; 6. Meyer (1992) Design by Contract, IEEE Computer;
7. Ohno (1988) *Toyota Production System*; 8. Liker (2004) *The Toyota Way*;
9. Fog (2023) SIMD Optimization; 10. Hennessy & Patterson (2017) *Computer Architecture*;
11. Iglewicz & Hoaglin (1993) *Outlier Detection*; 12. Beck (2002) *TDD*;
13. Wilkinson (2005) *Grammar of Graphics*.
