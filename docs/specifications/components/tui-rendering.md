# TUI Rendering Backend

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** Direct crossterm backend, CellBuffer, DiffRenderer, zero-allocation design, Brick compliance.

---

## Architecture

Eliminates `ratatui` dependency in favor of direct `crossterm` integration using a "Kernel-Cooperative" architecture (Direct Diffing).

```
Canvas trait --> DirectTerminalCanvas --> crossterm
     |                 |                     |
  presentar         unified               I/O
```

| Metric | ratatui | Direct |
|--------|---------|--------|
| Dependencies | ~15 | ~4 |
| Adapter LOC | ~450 | ~200 |
| Buffer copies | 2 | 1 |
| Frame overhead | ~0.5ms | ~0.1ms |
| Heap Allocs | Many | Near Zero |

## Core Components

### CellBuffer with Optimized Storage

`CompactString` (24-byte inline) eliminates heap allocations for single graphemes (99% of terminal content).

```rust
pub struct Cell {
    pub symbol: CompactString,  // 24 bytes (inlines up to 24 chars)
    pub fg: Color,              // 4 bytes
    pub bg: Color,              // 4 bytes
    pub modifiers: Modifiers,   // 1 byte
}

pub struct CellBuffer {
    pub cells: Vec<Cell>,
    pub width: u16,
    pub height: u16,
    pub dirty: BitVec,          // 1 bit per cell
}
```

Memory: 80x24 terminal = 1920 cells x 40 bytes = 75KB.

### DiffRenderer

Smart Diff algorithm minimizing I/O syscalls and ANSI escape sequences:
1. Iterate dirty cells (via `BitVec::iter_ones`)
2. Move cursor only if not adjacent to previous cell
3. Update style only if changed from previous cell
4. Print content, then clear dirty bits
5. Single `flush()` syscall via `BufWriter`

### Unicode Handling (UAX #11)

- Width 0 (combining char): Append to previous cell
- Width 1: Normal set
- Width 2 (CJK): Set current cell, mark next as CONTINUATION

### Color Mode Detection

```rust
pub fn detect() -> ColorMode {
    match std::env::var("COLORTERM").as_deref() {
        Ok("truecolor" | "24bit") => ColorMode::TrueColor,
        _ => match std::env::var("TERM").as_deref() {
            Ok(t) if t.contains("256color") => ColorMode::Color256,
            Ok(t) if t.contains("xterm") => ColorMode::Color16,
            _ => ColorMode::Mono,
        }
    }
}
```

### Resize Handling

On `SIGWINCH`: allocate new buffer, clear screen, mark ALL dirty. `Drop` restores terminal state (leave alternate screen, show cursor, disable raw mode).

## Dependencies

| Crate | Purpose |
|-------|---------|
| `crossterm` 0.28 | Terminal I/O |
| `unicode-width` 0.2 | Grapheme width |
| `unicode-segmentation` 1.12 | Grapheme iteration |
| `compact_str` 0.8 | Zero-alloc strings |
| `bitvec` 1.0 | Dirty tracking |

## Performance

| Scenario | Target |
|----------|--------|
| 80x24 full redraw | < 1ms |
| 80x24 partial (10%) | < 0.1ms |
| 200x50 full redraw | < 5ms |
| Memory (80x24) | < 100KB |
| Steady-state alloc | **0 bytes** |

## Brick Architecture Compliance (PROBAR-SPEC-009)

```rust
impl Brick for DirectTerminalCanvas<'_> {
    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::MaxLatencyMs(1),
            BrickAssertion::Custom { name: "zero_alloc_steady", .. },
            BrickAssertion::Custom { name: "cell_diff_correct", .. },
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget { measure_ms: 0.1, layout_ms: 0.5, paint_ms: 2.0, total_ms: 3.0 }
    }
}
```

## Falsification Checklist

### Performance (P1-P10)

| ID | Claim | Pass Criteria |
|----|-------|---------------|
| P1 | 80x24 full redraw < 1ms | criterion p95 |
| P2 | 10% diff update < 0.1ms | criterion p95 |
| P3 | Memory < 100KB for 80x24 | heaptrack peak |
| P5 | Zero steady-state alloc | global_allocator counter = 0 after init |
| P8 | Flush batched to single write | strace write_calls = 1 per frame |
| P9 | Cursor moves minimized | moves <= dirty_regions |

### Correctness (C1-C25)

| ID | Claim | Pass Criteria |
|----|-------|---------------|
| C1 | ASCII output matches ratatui | Byte-compare |
| C2 | Wide chars "日本語" = 6 cells | Width correct |
| C5 | TrueColor gradient accurate | RGB match |
| C20 | Resize clears buffer | Fresh buffer after resize |
| C25 | Cleanup on panic | Terminal restored |

### Compatibility (X1-X19)

| ID | Claim | Pass Criteria |
|----|-------|---------------|
| X1 | Works on Linux VT | Renders |
| X14 | Handles SIGWINCH | Resize handled |
| X17 | No terminfo dependency | Fallback works |
| X19 | Non-UTF8 locale | No panic |

### Dependencies (D1-D11)

| ID | Claim | Pass Criteria |
|----|-------|---------------|
| D1 | Compiles without ratatui | Builds |
| D2 | <= 5 direct dependencies | Count |
| D7 | No unsafe in new code | `#![forbid(unsafe_code)]` |
| D11 | CellBuffer compiles for wasm32 | Target build succeeds |

## PMAT Quality Gates

- Line coverage >= 95%
- Mutation score >= 80%
- Quality grade = A (90+)
- Zero SATD markers
- Pre-commit hook enforced

## References

- Pike, R. (1988). A Concurrent Window System. *Computing Systems*.
- `compact_str`: https://docs.rs/compact_str
