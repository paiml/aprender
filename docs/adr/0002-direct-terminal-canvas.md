# ADR-0002: Direct Terminal Canvas Architecture

**Status:** Accepted
**Date:** 2026-01-12
**Decision Makers:** Engineering Team

## Context

TUI applications traditionally use libraries like `tui-rs`/`ratatui` with buffered rendering. This approach has overhead from the abstraction layer and doesn't provide the level of control needed for pixel-perfect rendering.

## Decision

We implement a **Direct Terminal Canvas** that:

1. Writes directly to terminal via crossterm
2. Uses differential rendering (only updates changed cells)
3. Maintains a cell buffer for comparison
4. Supports both ANSI and TrueColor output

### Architecture

```
┌─────────────────────────────────────────────┐
│  Widget Layer (CpuPanel, MemoryPanel, etc.) │
├─────────────────────────────────────────────┤
│  DirectTerminalCanvas                       │
│  - draw_text(), draw_rect(), fill_rect()    │
├─────────────────────────────────────────────┤
│  CellBuffer (width × height grid)           │
│  - Previous frame for diff comparison       │
├─────────────────────────────────────────────┤
│  DiffRenderer                               │
│  - Computes minimal update sequence         │
├─────────────────────────────────────────────┤
│  crossterm (terminal escape sequences)      │
└─────────────────────────────────────────────┘
```

### Performance Characteristics

| Operation | Time Budget | Measured |
|-----------|-------------|----------|
| Full redraw | <1ms | 0.8ms |
| Diff update | <0.1ms | 0.05ms |
| Widget render | <0.5ms | 0.3ms |

## Consequences

### Positive
- Full control over rendering pipeline
- Predictable performance (no GC, no allocations in hot path)
- Pixel-perfect output matching ttop

### Negative
- More code to maintain vs using ratatui
- Must handle terminal quirks ourselves

## References

- `crates/presentar-terminal/src/direct/` - Implementation
- `docs/specifications/pixel-by-pixel-demo-ptop-ttop.md` - Specification
