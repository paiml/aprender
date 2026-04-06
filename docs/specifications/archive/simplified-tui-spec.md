# Simplified TUI Backend Specification

**Version:** 1.1.0
**Status:** DRAFT - Enhanced
**Author:** Claude Code (Edited by Gemini)
**Date:** 2026-01-09

## Abstract

This specification defines a minimal, high-performance terminal rendering backend for `presentar-terminal`. It eliminates the `ratatui` dependency in favor of a direct `crossterm` integration using a "Kernel-Cooperative" architecture (Direct Diffing). The design reduces the dependency tree by ~80%, minimizes heap allocations via small-string optimization, and guarantees sub-millisecond frame rendering for typical terminal sizes, strictly adhering to the **Brick Architecture**.

## 1. Motivation

### 1.1 Current Architecture

```
Canvas trait → TerminalCanvas → ratatui::Buffer → crossterm
     ↑              ↑                ↑              ↑
  presentar      adapter         middleware      I/O
```

### 1.2 Proposed Architecture

```
Canvas trait → DirectTerminalCanvas → crossterm
     ↑              ↑                    ↑
  presentar      unified              I/O
```

### 1.3 Rationale

| Metric | Current (ratatui) | Proposed (direct) |
|--------|-------------------|-------------------|
| Dependencies | ~15 | ~4 (added `compact_str`) |
| Adapter LOC | ~450 | ~200 |
| Buffer copies | 2 | 1 |
| Frame overhead | ~0.5ms | ~0.1ms |
| Heap Allocs | Many (String/Cell) | Near Zero (Inline) |

## 2. Technical Design

### 2.1 Core Components

#### 2.1.1 CellBuffer & Optimized Storage

To achieve the "zero-allocation steady state" target, we must avoid `String` allocations for single graphemes (which constitute 99% of terminal content). We employ `compact_str` (or equivalent) to inline strings ≤ 24 bytes.

```rust
use compact_str::CompactString;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    pub symbol: CompactString,  // 24 bytes (inlines up to 24 chars)
    pub fg: Color,              // 4 bytes
    pub bg: Color,              // 4 bytes
    pub modifiers: Modifiers,   // 1 byte
    // padding: 7 bytes (align to 40)
}

pub struct CellBuffer {
    pub cells: Vec<Cell>,
    pub width: u16,
    pub height: u16,
    pub dirty: BitVec,  // 1 bit per cell, tracked efficiently
}
```

**Memory footprint:** For 80x24 terminal: `1920 cells × 40 bytes ≈ 75KB`.
**Allocation Strategy:** Since `CompactString` inlines small strings, updating a cell with a new character/grapheme (e.g., "a", "界") incurs **zero heap allocation**.

#### 2.1.2 DiffRenderer

The renderer implements a "Smart Diff" algorithm that minimizes I/O syscalls and ANSI escape sequences.

```rust
impl DiffRenderer {
    pub fn flush(&mut self, stdout: &mut impl Write) -> io::Result<usize> {
        // Optimization: buffer writes to avoid small syscalls
        let mut writer = BufWriter::with_capacity(8192, stdout); 
        let mut writes = 0;
        let mut cursor = (u16::MAX, u16::MAX);
        let mut current_style = (Color::Reset, Color::Reset, Modifiers::empty());

        for idx in self.buffer.dirty.iter_ones() {
            let (x, y) = (idx % self.width, idx / self.width);
            let cell = &self.buffer.cells[idx];
            
            // 1. Move Cursor (Optimized: Only if not adjacent)
            if cursor != (x, y) {
                queue!(writer, MoveTo(x, y))?;
            }
            
            // 2. Update Style (Optimized: Only if changed)
            let new_style = (cell.fg, cell.bg, cell.modifiers);
            if new_style != current_style {
                queue!(writer, SetColors(cell.fg, cell.bg), SetAttributes(cell.modifiers))?;
                current_style = new_style;
            }

            // 3. Print Content
            queue!(writer, Print(&cell.symbol))?;
            
            // 4. Update internal state
            cursor = (x + cell.width(), y);
            writes += 1;
        }
        
        self.buffer.dirty.clear();
        writer.flush()?; // Single syscall
        Ok(writes)
    }
}
```

#### 2.1.3 DirectTerminalCanvas

Implements `presentar_core::Canvas` trait directly.

```rust
impl Canvas for DirectTerminalCanvas<'_> {
    fn fill_rect(&mut self, rect: Rect, color: Color) {
        // Optimized bulk set
        for y in rect.y..rect.y + rect.height {
            let start = self.index(rect.x, y);
            let end = start + rect.width as usize;
            for i in start..end {
                self.buffer.cells[i].update(" ", color, color, Modifiers::empty());
                self.buffer.dirty.set(i, true);
            }
        }
    }
    // ...
}
```

### 2.2 Unicode & Wide Character Handling

We strictly adhere to `UAX #11` (East Asian Width).

1.  **Dependency:** `unicode-width` crate.
2.  **Logic:**
    -   If `width == 0` (combining char): Append to previous cell.
    -   If `width == 1`: Normal set.
    -   If `width == 2`: Set current cell, mark next cell as `CONTINUATION` (skip rendering).

```rust
const CONTINUATION_SYMBOL: &str = ""; // or special sentinel

fn set_cell(&mut self, x: u16, y: u16, symbol: &str, fg: Color, bg: Color) {
    let width = UnicodeWidthStr::width(symbol).max(1);
    // ... boundary checks ...
    let idx = self.index(x, y);

    self.buffer.cells[idx].update(symbol, fg, bg, Modifiers::empty());
    self.buffer.dirty.set(idx, true);

    if width > 1 {
        // Handle wide char overlap
        if x + 1 < self.width {
             let next_idx = self.index(x + 1, y);
             self.buffer.cells[next_idx].make_continuation();
             self.buffer.dirty.set(next_idx, true);
        }
    }
}
```

### 2.3 Color Mode Detection

Environment-based detection with `terminfo` fallback if available (but keeping deps low).

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

### 2.4 Resize Handling & Lifecycle

-   **Resize:** On `SIGWINCH` (handled by `crossterm::event`), reallocation occurs.
    -   **Strategy:** Allocate new buffer, clear screen, mark ALL dirty.
-   **Cleanup:** `Drop` implementation MUST restore terminal state (leave alternate screen, show cursor, disable raw mode) to prevent "terminal borking".

## 3. Performance Analysis

### 3.1 Theoretical Bounds

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Full redraw | O(W×H) | O(W×H) |
| Diff flush | O(D) where D = dirty cells | O(1) aux |
| Cell lookup | O(1) | O(1) |
| Resize | O(W×H) | O(W×H) |

### 3.2 Empirical Targets (Validated by P-Checklist)

| Scenario | Target | Rationale |
|----------|--------|-----------|
| 80×24 full redraw | <1ms | 60fps budget = 16.6ms |
| 80×24 partial (10%) | <0.1ms | Typical incremental update |
| 200×50 full redraw | <5ms | Large terminal |
| Memory (80×24) | <100KB | L2 Cache friendly |
| Steady State Alloc | **0 bytes** | Via `CompactString` |

## 4. Dependencies

### 4.1 Required

| Crate | Version | Purpose | Size |
|-------|---------|---------|------|
| `crossterm` | 0.28 | Terminal I/O | ~50KB |
| `unicode-width` | 0.2 | Grapheme width | ~15KB |
| `unicode-segmentation` | 1.12 | Grapheme iteration | ~20KB |
| `compact_str` | 0.8 | **Zero-alloc strings** | ~10KB |
| `bitvec` | 1.0 | Efficient dirty tracking | ~15KB |

### 4.2 Removed

| Crate | Reason |
|-------|--------|
| `ratatui` | Redundant middleware |
| `cassowary` | Not used (presentar has own layout) |
| `itertools` | Transitive, not needed |

## 5. API Surface

### 5.1 Public Types

```rust
// Re-export from presentar_core
pub use presentar_core::{Canvas, Color, Point, Rect, TextStyle};

// New types
pub struct DirectTerminalCanvas<'a> { /* ... */ }
// CellBuffer is internal
// DiffRenderer is internal

// Retained from current implementation
pub enum ColorMode { TrueColor, Color256, Color16, Mono }
pub struct TuiApp<W: Widget> { /* ... */ }
pub struct TuiConfig { /* ... */ }
```

## 5A. Probar Brick Compliance (PROBAR-SPEC-009)

The implementation MUST comply with `probar`'s Brick Architecture.

### 5A.1 Required Trait Bounds

All widgets must satisfy `Brick + Send + Sync`.

### 5A.2 Required Assertions

Per Popper's falsifiability principle:

```rust
impl Brick for DirectTerminalCanvas<'_> {
    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[
            // P1: Frame time < 1ms (Strict)
            BrickAssertion::MaxLatencyMs(1),
            // P5: Zero allocations in steady state
            BrickAssertion::Custom { name: "zero_alloc_steady", validator_id: 0x_MEM_001 },
            // C1: Output correctness
            BrickAssertion::Custom { name: "cell_diff_correct", validator_id: 0x_DIFF_001 },
        ];
        ASSERTIONS
    }
}
```

### 5A.3 Budget Enforcement

```rust
impl Brick for DirectTerminalCanvas<'_> {
    fn budget(&self) -> BrickBudget {
        BrickBudget {
            measure_ms: 0.1, // Near zero
            layout_ms: 0.5,  
            paint_ms: 2.0,   // Flush time
            total_ms: 3.0,   // Strict budget
        }
    }
}
```

## 6. Migration Path

### Phase 1: Verification (TDD)
- Create `tests/direct_canvas_spec.rs`
- Implement Falsification Checklist P1-P25 as empty tests
- Ensure CI fails (Red)

### Phase 2: Implementation (Green)
- Implement `CellBuffer` with `CompactString`
- Implement `DirectTerminalCanvas`
- Pass P-checklist

### Phase 3: Integration (Refactor)
- Switch `presentar-cli` to use `DirectTerminalCanvas`
- Remove `ratatui` dependency

## 7. References

[1] Pike, R. (1988). "A Concurrent Window System." *Computing Systems*.
[2] PROBAR-SPEC-009: "Bug Hunting Probador - Brick Architecture."
[3] `compact_str` Documentation: <https://docs.rs/compact_str>

---

## 8. Popperian Falsification Checklist

### Methodology

**Probar Integration:** Tests execute via `jugar_probar`'s verification protocol.
**Status:** REQUIRED for Release 1.0.

### 8.1 Performance Claims (P1-P25)

| ID | Claim | Falsification Test | Pass Criteria |
|----|-------|-------------------|---------------|
| P1 | Full 80×24 redraw completes in <1ms | `criterion` benchmark | p95 < 1ms |
| P2 | Differential update of 10% cells <0.1ms | Benchmark with random 192-cell update | p95 < 0.1ms |
| P3 | Memory usage <100KB for 80×24 | `heaptrack` / `dhat` | peak < 100KB |
| P4 | 200×50 full redraw <5ms | Benchmark on large terminal | p95 < 5ms |
| P5 | **Zero allocations** in steady-state | `#[global_allocator]` counting hook | alloc_count = 0 (after init) |
| P6 | Dirty bitmap overhead <1% of buffer | sizeof comparison | bitmap < cells/100 |
| P7 | Cell lookup is O(1) | Benchmark varying buffer size | constant time |
| P8 | Flush I/O batched to single write | `strace` or mock writer | write_calls = 1 per frame |
| P9 | Cursor movement minimized | Count MoveTo in output | moves ≤ dirty_regions |
| P10 | Color mode detection <1μs | Benchmark detect() | p95 < 1μs |

### 8.2 Correctness Claims (C1-C25)

| ID | Claim | Falsification Test | Pass Criteria |
|----|-------|-------------------|---------------|
| C1 | Output matches ratatui for ASCII | Byte-compare terminal output | identical content |
| C2 | Wide chars occupy correct cells | Render "日本語" | width = 6 cells |
| C3 | Emoji render correctly | Render "👨‍👩‍👧‍👦" | single logical cell update |
| C4 | Zero-width joiners handled | Render combined chars | correct display |
| C5 | Color accuracy in TrueColor | Render gradient | RGB match |
| C6 | 256-color palette mapping | Compare to xterm | index match |
| C20 | Resize preserves no content | Resize, verify cleared | fresh buffer |
| C25 | Cleanup on panic | Panic handler test | terminal restored (raw mode off) |

### 8.3 Compatibility Claims (X1-X25)

| ID | Claim | Falsification Test | Pass Criteria |
|----|-------|-------------------|---------------|
| X1 | Works on Linux VT | Test in /dev/tty1 | renders |
| X14 | Handles SIGWINCH | Send signal, verify | resize handled |
| X17 | No terminfo dependency | Run without terminfo | fallback works |
| X19 | Non-UTF8 locale | LC_ALL=C | graceful degrade (no panic) |

### 8.4 Dependency Claims (D1-D15)

| ID | Claim | Falsification Test | Pass Criteria |
|----|-------|-------------------|---------------|
| D1 | Compiles without ratatui | Remove from Cargo.toml | builds |
| D2 | Only ~4 direct dependencies | Count Cargo.toml | count ≤ 5 |
| D7 | No unsafe in new code | `#![forbid(unsafe_code)]` | compiles |
| D11 | Compiles for wasm32 (no I/O) | `--target wasm32-unknown-unknown` | CellBuffer compiles |

### 8.5 API Claims (A1-A10)

| ID | Claim | Falsification Test | Pass Criteria |
|----|-------|-------------------|---------------|
| A1 | Public API unchanged | semver-checks | no breaking |
| A2 | Canvas trait fully implemented | Compile existing code | success |

---

## 9. PMAT Quality Gates (Zero Tolerance)

### 9.1 Gate Configuration

Create `.pmat-gates.toml` with **zero tolerance** enforcement:

```toml
# PMAT Quality Gates - Simplified TUI Backend
# ZERO TOLERANCE: All gates MUST pass

[gates]
# Minimum quality grade (A = 90+, B+ = 85+, B = 80+)
min_grade = "A"

# ZERO TOLERANCE: 95% minimum coverage
min_coverage = 95

# Mutation testing threshold
min_mutation_score = 80

# Maximum cyclomatic complexity per function
max_complexity = 10

# Maximum nesting depth (JPL Rule 1)
max_nesting = 3

# Maximum lines per function (JPL Rule 4)
max_function_lines = 40

[thresholds]
# TDG (Test-Driven Grade) score minimum
min_tdg_score = 85

# Repository health minimum
min_repo_score = 90

# Rust project score minimum
min_rust_score = 90

[enforcement]
# CRITICAL: Fail on ANY regression
fail_on_regression = true

# ZERO score drop tolerance
max_score_drop = 0

# ALL gates must pass - no exceptions
require_all_gates = true

# Block merge on failure
block_merge_on_failure = true

[satd]
# Self-Admitted Technical Debt - FORBIDDEN
allow_todo = false
allow_fixme = false
allow_hack = false
allow_xxx = false

[security]
# Audit must pass
require_audit_pass = true

# No known vulnerabilities
max_vulnerabilities = 0
```

### 9.2 Coverage Enforcement

**ZERO TOLERANCE POLICY:** Code coverage MUST be ≥95% at all times.

| Metric | Threshold | Enforcement |
|--------|-----------|-------------|
| Line Coverage | ≥95% | CI blocks merge |
| Region Coverage | ≥95% | CI blocks merge |
| Function Coverage | ≥95% | CI blocks merge |
| Branch Coverage | ≥90% | CI warns, tracked |

```bash
# Verification command (REQUIRED before commit)
cargo llvm-cov --all-features --workspace --fail-under-lines 95

# Or via certeza
cd ../certeza && cargo run -- check ../presentar
```

### 9.3 Per-File Coverage Requirements

**No file may fall below 95% coverage.** Files with <95% coverage MUST be:

1. Immediately addressed (same PR)
2. Or excluded with documented justification in `.coverage-exceptions.toml`:

```toml
# .coverage-exceptions.toml
# REQUIRES: Technical lead approval + issue link

[[exceptions]]
file = "src/app.rs"
reason = "Terminal I/O requires integration testing"
issue = "https://github.com/org/repo/issues/123"
min_coverage = 60  # Reduced threshold with justification
expires = "2026-03-01"  # MUST have expiry
```

### 9.4 Mutation Testing Gate

```bash
# Required mutation score: 80%
cargo mutants --minimum-mutants-tested 80

# Critical paths require 90%
cargo mutants --package presentar-terminal --minimum-mutants-tested 90
```

### 9.5 Pre-Commit Hook

`.git/hooks/pre-commit`:

```bash
#!/bin/bash
set -euo pipefail

echo "=== PMAT Quality Gate Check ==="

# 1. Coverage check (ZERO TOLERANCE)
echo "Checking coverage (min 95%)..."
if ! cargo llvm-cov --quiet --fail-under-lines 95; then
    echo "ERROR: Coverage below 95%. Commit blocked."
    exit 1
fi

# 2. Clippy (deny all warnings)
echo "Running clippy..."
if ! cargo clippy --all-features -- -D warnings; then
    echo "ERROR: Clippy warnings. Commit blocked."
    exit 1
fi

# 3. SATD check (no TODO/FIXME/HACK)
echo "Checking for SATD..."
if grep -rn "TODO\|FIXME\|HACK\|XXX" src/; then
    echo "ERROR: Self-Admitted Technical Debt found. Commit blocked."
    exit 1
fi

echo "=== All gates passed ==="
```

### 9.6 CI Pipeline Integration

```yaml
# .github/workflows/quality-gates.yml
name: PMAT Quality Gates

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install llvm-cov
        run: cargo install cargo-llvm-cov

      - name: Coverage Check (95% REQUIRED)
        run: |
          cargo llvm-cov --all-features --workspace \
            --fail-under-lines 95 \
            --fail-under-regions 95 \
            --fail-under-functions 95

      - name: Upload Coverage
        uses: codecov/codecov-action@v4
        with:
          fail_ci_if_error: true

  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Mutation Testing (80% REQUIRED)
        run: |
          cargo install cargo-mutants
          cargo mutants --minimum-mutants-tested 80

  quality-score:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: PMAT Score Check
        run: |
          # Grade must be A (90+)
          score=$(cargo run -p pmat -- score .)
          if [ "$score" -lt 90 ]; then
            echo "Quality score $score < 90. Blocked."
            exit 1
          fi
```

### 9.7 Regression Prevention

**Score Tracking:** Every commit records quality metrics:

```json
// .pmat-metrics/commit-{hash}-meta.json
{
  "commit": "abc123",
  "timestamp": "2026-01-09T12:00:00Z",
  "metrics": {
    "line_coverage": 96.2,
    "region_coverage": 95.8,
    "mutation_score": 82.5,
    "complexity_avg": 4.2,
    "quality_grade": "A"
  }
}
```

**Regression Detection:**
- Any drop in coverage → CI failure
- Any drop in mutation score → CI failure
- Any increase in complexity → CI warning

### 9.8 Acceptance Criteria (Updated)

For this specification to be considered complete:

- [ ] Line coverage ≥95% (REQUIRED)
- [ ] Region coverage ≥95% (REQUIRED)
- [ ] Mutation score ≥80% (REQUIRED)
- [ ] Quality grade = A (REQUIRED)
- [ ] Zero SATD markers (REQUIRED)
- [ ] All 100 falsification tests pass (REQUIRED)
- [ ] Pre-commit hook installed (REQUIRED)
- [ ] CI pipeline configured (REQUIRED)

---

## Appendix E: Documentation Integration Strategy

To ensure documentation stays in sync with implementation (The "Living Spec" principle), we use `mdbook`'s include feature.

### E.1 Linking Verified Examples

All code examples in this spec must exist as compilable Rust files in `examples/spec_verification/`.

```markdown
<!-- In the spec -->
### Example: Basic Usage
{{#include ../../../examples/spec_verification/tui_basic.rs}}
```

### E.2 Validation

The `validate_docs.sh` script checks:
1.  All `{{#include}}` paths exist.
2.  The included files compile.
3.  The included files pass `clippy`.

---