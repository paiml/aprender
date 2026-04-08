# Ratatui → Presentar Migration

**Version**: 1.0
**Date**: 2026-04-08
**Status**: PROPOSAL
**Contract**: `contracts/ratatui-migration-v1.yaml`
**Priority**: P1 — Sovereign stack compliance

---

## Problem

13 workspace crates depend on `ratatui` (non-sovereign, external TUI library).
The Sovereign AI Stack has `presentar-terminal` as the TUI framework.
ratatui adds unnecessary binary bloat and violates the sovereign-deps principle.

## Scope

- **65 source files** use `ratatui::` imports
- **125 import statements** to replace
- **13 crates** affected
- presentar-terminal already has **85 imports** in 9 crates (partial migration done)

## Inventory

### ratatui Types Used (must have presentar equivalents)

| ratatui Type | Usage Count | presentar Equivalent |
|-------------|-------------|---------------------|
| `ratatui::Frame` | ~30 | `presentar_terminal::DirectTerminalCanvas` |
| `ratatui::layout::Rect` | ~25 | `presentar_core::Rect` |
| `ratatui::layout::Constraint` | ~15 | `presentar_core::Constraint` |
| `ratatui::style::Color` | ~20 | `presentar_terminal::theme::Color` |
| `ratatui::style::Style` | ~15 | `presentar_terminal::theme::Theme` |
| `ratatui::widgets::*` | ~40 | `presentar_terminal::widgets::*` |
| `ratatui::Terminal` | ~10 | `presentar_terminal::DirectTerminalCanvas` |
| `ratatui::backend::CrosstermBackend` | ~8 | `presentar_terminal::DiffRenderer` |
| `ratatui::Buffer` | ~5 | `presentar_terminal::CellBuffer` |

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

### Phase 1: Contract + Remove ratatui dep (make optional)
- Write provable contract
- Make ratatui optional in ALL 13 Cargo.toml files
- `#[cfg(feature = "ratatui")]` gate all 125 imports
- Verify: `cargo check --workspace` passes without ratatui

### Phase 2: Replace small crates (effort: Small/Minimal)
- aprender-orchestrate (3 imports)
- aprender-serve (1 import)
- aprender-simulate (9 imports)
- aprender-distribute (6 imports)
- aprender-test-lib (5 imports)
- aprender-test-showcase (4 imports)

### Phase 3: Replace medium crates
- apr-cli (17 imports — cbtop, experiment, tui commands)

### Phase 4: Replace large crates
- aprender-profile (45 imports — full visualization rewrite)
- aprender-viz (71 imports — full monitor widget rewrite)

### Phase 5: Delete ratatui
- Remove ratatui from ALL Cargo.toml
- Remove `#[cfg(feature = "ratatui")]` gates (now dead code)
- Verify: `grep -r "ratatui" crates/` returns 0

## Falsification

After migration:
- FALSIFY-RATATUI-001: `grep "ratatui" crates/*/Cargo.toml` returns 0
- FALSIFY-RATATUI-002: `grep -r "use ratatui" crates/*/src/` returns 0
- FALSIFY-RATATUI-003: `cargo check --workspace` passes
- FALSIFY-RATATUI-004: `apr tui`, `apr cbtop`, `apr monitor` still work
