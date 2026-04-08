# Ratatui → Presentar Migration (Improved TUI)

**Version**: 2.0
**Date**: 2026-04-06
**Status**: IN PROGRESS (Phase 1+2 done)
**Contracts**:
- `contracts/ratatui-migration-v1.yaml` — zero-ratatui dep contract
- `contracts/tui-rendering-ux-v1.yaml` — TUI UX/layout contract (provable)
**Priority**: P1 — Sovereign stack compliance + UX improvement

---

## Problem

13 workspace crates depend on `ratatui` (non-sovereign, external TUI library).
The Sovereign AI Stack has `presentar-terminal` as the TUI framework.
ratatui adds unnecessary binary bloat and violates the sovereign-deps principle.

## Mandate

**This is NOT a 1:1 port. The TUI MUST be improved.**

The presentar-terminal migration is an opportunity to deliver a significantly
better TUI experience. Every TUI command gets a provable layout contract
(`contracts/tui-rendering-ux-v1.yaml`) that defines:

1. **Consistent 3-zone layout**: header + body (2-col) + footer across ALL commands
2. **Vim + arrow key navigation**: j/k/Tab/Enter/q/? everywhere
3. **Theme-driven colors**: WCAG AA contrast, graceful degradation (TrueColor→mono)
4. **Responsive layout**: 2-column at 80+, single-column below 80
5. **Widget composition**: presentar Widget/Brick lifecycle (measure→layout→paint)
6. **60 FPS frame budget**: smart diff rendering, zero steady-state allocation
7. **Sortable tables, sparklines, braille charts**: data-dense information display

## Scope

- **65 source files** use `ratatui::` imports
- **125 import statements** to replace
- **13 crates** affected
- presentar-terminal already has **85 imports** in 9 crates (partial migration done)

## Inventory

### ratatui → presentar Type Mapping (Improved)

| ratatui Type | presentar Equivalent | Improvement |
|-------------|---------------------|-------------|
| `ratatui::Frame` | `presentar_terminal::DirectTerminalCanvas` | Zero-alloc, smart diff |
| `ratatui::Terminal` | `presentar_terminal::CrosstermTerminal` | TestableBackend for CI |
| `ratatui::layout::Rect` | `presentar_core::Rect` | f32 coords, sub-cell |
| `ratatui::layout::Constraint` | `presentar_core::Constraints` | Measure-layout-paint |
| `ratatui::layout::Layout` | `presentar_terminal::widgets::Layout` | Rows/Cols/Stack compose |
| `ratatui::style::Color` | `presentar_core::Color` | RGBA f32, WCAG contrast |
| `ratatui::style::Style` | `presentar_terminal::theme::TextStyle` | FontWeight, Theme-driven |
| `ratatui::widgets::Block` | `presentar_terminal::widgets::Border` | BorderStyle + padding |
| `ratatui::widgets::Paragraph` | `presentar_terminal::widgets::Text` | Unicode-aware wrapping |
| `ratatui::widgets::Table` | `presentar_terminal::widgets::DataFrame` | Sortable columns, cell types |
| `ratatui::widgets::List` | `presentar_terminal::widgets::ProcessTable` | Process-aware, filterable |
| `ratatui::widgets::Sparkline` | `presentar_terminal::widgets::Sparkline` | Trend direction, symbols |
| `ratatui::widgets::Tabs` | Layout + selection state | Custom, theme-driven |
| `ratatui::widgets::Gauge` | `presentar_terminal::widgets::Gauge` | Gradient fill |
| — (no equivalent) | `presentar_terminal::widgets::Heatmap` | NEW: tensor vis |
| — (no equivalent) | `presentar_terminal::widgets::LineChart` | NEW: loss curves |
| — (no equivalent) | `presentar_terminal::widgets::BrailleGraph` | NEW: high-density sparklines |
| — (no equivalent) | `presentar_terminal::widgets::Histogram` | NEW: latency distributions |
| `ratatui::backend::CrosstermBackend` | `presentar_terminal::DiffRenderer` | Only changed cells |
| `ratatui::Buffer` | `presentar_terminal::CellBuffer` | CompactString, zero-alloc |

### Crates by Effort

| Crate | Files | Imports | Effort | Strategy |
|-------|-------|---------|--------|----------|
| apr-cli | 7 | 17 | Medium | Replace TUI rendering with presentar |
| aprender-profile | 11 | 45 | Large | Rewrite visualize/ panels |
| aprender-viz | 14 | 71 | Large | Rewrite monitor/ widgets |
| aprender-simulate | 6 | 9 | Small | Replace orbit/tsp TUI |
| aprender-distribute | 3 | 6 | Small | Replace tui/ module |
| aprender-test-lib | 3 | 5 | Small | Replace tui backend |
| aprender-test-showcase | 2 | 4 | Small | Replace keypad/ui |
| aprender-orchestrate | 3 | 3 | Minimal | Replace oracle/stack TUI |
| aprender-serve | 1 | 1 | Minimal | Replace monitor binary |
| aprender-compute | 0 | cfg-only | None | Already gated |
| aprender-explain | 0 | cfg-only | None | Already gated |
| aprender-test | 0 | cfg-only | None | Already gated |
| aprender-viz-ttop | 2 | 2 | Minimal | Already uses presentar |

## Implementation Plan

### Phase 1: Contract + Remove ratatui dep (make optional) — DONE ✓
- Write provable contract ✓
- Make ratatui optional in ALL 13 Cargo.toml files ✓
- `#[cfg(feature = "ratatui")]` gate all 125 imports ✓
- Gate TUI-dependent modules in apr-cli (cbtop TUI, tui, federation/tui, experiment browser) ✓
- Gate visualize module in aprender-profile ✓
- Gate monitor module in aprender-viz ✓
- Stubs return helpful error messages for ungated paths ✓
- Verify: `cargo check --workspace` passes without ratatui ✓

### Phase 2: Replace small crates (effort: Small/Minimal) — DONE ✓
- aprender-serve (1 import) → replaced with presentar-terminal ✓
- aprender-test-lib (5 imports) → replaced with presentar-terminal ✓
- aprender-test-showcase (4 imports) → replaced with presentar-terminal ✓
- aprender-orchestrate — cfg-gated, no presentar replacement needed
- aprender-simulate — cfg-gated, no presentar replacement needed
- aprender-distribute — cfg-gated, no presentar replacement needed

### Phase 3: TUI UX Contract (MUST BE DONE BEFORE PHASE 4)
- **Contract**: `contracts/tui-rendering-ux-v1.yaml` ✓
- Defines 3-zone layout (header/body/footer) for every TUI command
- Defines keyboard navigation (vim+arrows), color theme, responsive breakpoints
- Defines per-command layout: tui model explorer, cbtop pipeline, monitor, experiment browser
- 8 falsification tests proving the layout contract
- **No code written until this contract is reviewed and finalized**

### Phase 4: Implement improved TUI with presentar-terminal — DONE ✓
- apr-cli `tui` command: model explorer with tabs (Overview/Tensors/Stats/Help) ✓
  - Sortable tensor table, histogram in stats tab
  - presentar `DataFrame` + `LineChart` + `BrailleGraph` widgets
- apr-cli `cbtop` command: pipeline monitor with 5 views (Pipeline/Budget/Histogram/GPU/Memory) ✓
  - Live-updating metrics, budget bar charts, gap factor indicators
  - presentar `DirectTerminalCanvas` + `DiffRenderer`
- apr-cli `experiment view`: experiment browser ✓
  - Two-column layout: run table + detail/sparkline
  - Loss curve via braille graph
- apr-cli `federation` TUI: cfg-gated (not actively used)
- aprender-profile: renacer visualize — cfg-gated (Phase 5 cleanup)
- aprender-viz: monitor widgets — cfg-gated (Phase 5 cleanup)

### Phase 5: Delete ratatui
- Remove ratatui from ALL 13 Cargo.toml files
- Remove `#[cfg(feature = "ratatui")]` gates (now dead code)
- Verify: `grep -r "ratatui" crates/` returns 0
- Verify: all falsification tests in `contracts/tui-rendering-ux-v1.yaml` pass

## Falsification

After migration:
- FALSIFY-RATATUI-001: `grep "ratatui" crates/*/Cargo.toml` returns 0
- FALSIFY-RATATUI-002: `grep -r "use ratatui" crates/*/src/` returns 0
- FALSIFY-RATATUI-003: `cargo check --workspace` passes
- FALSIFY-RATATUI-004: `apr tui`, `apr cbtop`, `apr monitor` still work
