# Rendering Backend

> Parent: [ttop-v2-spec.md](../ttop-v2-spec.md) Section 6

**Scope:** CellBuffer, DiffRenderer, DirectCanvas, zero-allocation design.
**Source:** Ported from `presentar-terminal::direct` module.

---

## 1. Architecture

Eliminates ratatui in favor of direct crossterm integration.

```
Panel::render() → CellBuffer → DiffRenderer → crossterm → stdout
                     │              │
              CompactString     BitVec dirty
              (24B inline)      (1 bit/cell)
```

### 1.1 Comparison

| Metric | ratatui (ttop 0.3.x) | Direct (ttop v2) |
|--------|---------------------|------------------|
| Transitive deps | ~15 | ~4 (crossterm, compact_str, bitvec, unicode-width) |
| Adapter LOC | ~450 | ~200 |
| Buffer copies/frame | 2 (app→Buffer→diff) | 1 (app→CellBuffer, diff in-place) |
| Frame overhead (80x24) | ~0.5ms | ~0.1ms |
| Heap allocs (steady state) | O(cells) | 0 |
| Memory (80x24) | ~100KB | ~75KB (1920 cells x 40B) |

---

## 2. CellBuffer

### 2.1 Cell Structure

```rust
pub struct Cell {
    /// Grapheme cluster. CompactString inlines up to 24 chars (covers 99%+ of
    /// terminal content without heap allocation). Reference: UAX #11.
    pub symbol: CompactString,  // 24 bytes
    /// Foreground color (RGB or named).
    pub fg: Color,              // 4 bytes
    /// Background color.
    pub bg: Color,              // 4 bytes
    /// Bold, italic, underline, etc.
    pub modifiers: Modifiers,   // 1 byte
    // Padding: 7 bytes → total 40 bytes/cell
}
```

### 2.2 Buffer Structure

```rust
pub struct CellBuffer {
    pub cells: Vec<Cell>,       // Pre-allocated, never resized during rendering
    pub width: u16,
    pub height: u16,
    pub dirty: BitVec,          // 1 bit per cell, marks changed cells
}
```

Memory footprint: `width * height * 40 bytes + width * height / 8 bytes`.
For 200x50 (large terminal): 400KB + 1.2KB = ~401KB.

### 2.3 Contract: Buffer Bounds

```yaml
# contracts/render.yaml
- name: cell_buffer_bounds
  description: "All cell access is within buffer bounds"
  preconditions:
    - "x < buffer.width"
    - "y < buffer.height"
  postconditions:
    - "index == y * width + x"
    - "index < cells.len()"
  invariants:
    - "cells.len() == width as usize * height as usize"
    - "dirty.len() == cells.len()"
```

### 2.4 Operations

| Method | Complexity | Allocations |
|--------|-----------|-------------|
| `set(x, y, symbol, fg, bg)` | O(1) | 0 (CompactString inline) |
| `clear()` | O(n) | 0 (resets in-place) |
| `resize(w, h)` | O(n) | 1 (Vec realloc) |
| `dirty_count()` | O(n/64) | 0 (BitVec popcount) |

---

## 3. DiffRenderer

### 3.1 Algorithm

```
for each dirty cell (via BitVec::iter_ones()):
    if cursor not adjacent to cell:
        emit MoveTo(x, y)
    if style changed from previous cell:
        emit SetForegroundColor(fg)
        emit SetBackgroundColor(bg)
        emit SetAttribute(modifiers)
    emit Print(symbol)
    clear dirty bit

flush() — single write() syscall via BufWriter
```

### 3.2 Optimizations

1. **Cursor tracking**: Skip MoveTo when cursor is already at correct position
   (adjacent cells in same row). Saves ~40% of escape sequences.

2. **Style deduplication**: Track last-emitted fg/bg/modifiers. Only emit
   changes. Saves ~60% of color escape sequences.

3. **Batched I/O**: All escape sequences written to `BufWriter<Stdout>`.
   Single `flush()` syscall per frame. Eliminates write amplification.

4. **Dirty bit scanning**: `BitVec::iter_ones()` uses hardware POPCNT
   to skip clean 64-cell chunks in O(1).

### 3.3 Contract: Diff Correctness

```yaml
- name: diff_renderer_correctness
  description: "After render_diff, displayed content matches CellBuffer"
  preconditions:
    - "prev_buffer and curr_buffer have same dimensions"
  postconditions:
    - "all dirty bits cleared"
    - "terminal displays curr_buffer content"
  performance:
    budget_ms: 1
    measurement: "wall_clock"
```

---

## 4. Unicode Handling (UAX #11)

| Width | Action | Example |
|-------|--------|---------|
| 0 (combining) | Append to previous cell's symbol | Diacritics |
| 1 (normal) | Set cell symbol | ASCII, Latin |
| 2 (wide/CJK) | Set cell, mark next as CONTINUATION | CJK, emoji |

CONTINUATION cells are skipped during rendering. Wide character at
column `width-1` is replaced with space (prevents wrap artifacts).

---

## 5. Color Mode Detection

```rust
pub enum ColorMode {
    TrueColor,  // COLORTERM=truecolor|24bit → RGB
    Color256,   // TERM contains "256color" → 256 palette
    Color16,    // TERM contains "xterm" → 16 ANSI
    Mono,       // Fallback → no color
}
```

Detection uses `$COLORTERM` and `$TERM` environment variables.
All color functions accept `ColorMode` and downgrade gracefully.

---

## 6. Performance Targets

| Scenario | Target | Measurement |
|----------|--------|-------------|
| Full redraw (80x24) | < 1ms | wall clock |
| Full redraw (200x50) | < 5ms | wall clock |
| Partial update (10%) | < 0.1ms | wall clock |
| Resize event | < 2ms | includes realloc |
| Steady-state allocs | 0 | per frame |

---

## 7. Testing

### 7.1 Brick Assertions

```rust
impl Brick for CellBuffer {
    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::invariant("cells_match_dims",
                self.cells.len() == (self.width as usize * self.height as usize)),
            BrickAssertion::invariant("dirty_match_dims",
                self.dirty.len() == self.cells.len()),
        ]
    }
    fn budget(&self) -> BrickBudget {
        BrickBudget::new_ms(1)  // 1ms for buffer operations
    }
}
```

### 7.2 Property Tests

```rust
proptest! {
    #[test]
    fn set_get_roundtrip(x in 0u16..200, y in 0u16..50, ch in "[a-z]") {
        let mut buf = CellBuffer::new(200, 50);
        buf.set(x, y, &ch, Color::White, Color::Black);
        let cell = buf.get(x, y);
        prop_assert_eq!(cell.symbol.as_str(), ch.as_str());
        prop_assert!(buf.is_dirty(x, y));
    }
}
```

---

## 8. References

- presentar-terminal `direct/` module: reference implementation
- PROBAR-SPEC-009: Brick Architecture compliance
- UAX #11: East Asian Width (Unicode Standard Annex)
- crossterm 0.28: Terminal I/O abstraction
