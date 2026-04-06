# Brick Testing

> Parent: [ttop-v2-spec.md](../ttop-v2-spec.md) Section 5

**Scope:** Probar Brick/BrickHouse architecture, falsification test methodology,
pixel coverage, visual regression, performance budgets.

---

## 1. Principle: Tests Define Interface

The Brick Architecture (PROBAR-SPEC-009) inverts the traditional
test-after-implementation model. In ttop v2:

1. Write Brick assertions (what the panel MUST do)
2. Write falsification tests (how to BREAK it)
3. Implement the panel to satisfy both
4. The build FAILS if assertions or tests are missing

This follows Popper's falsificationism: a theory (panel implementation)
is scientific only if it makes predictions that could be proven wrong
(falsification tests). Tests that always pass are worthless.

---

## 2. Brick Trait

### 2.1 Definition (from jugar-probar)

```rust
pub trait Brick: Send + Sync {
    /// Unique name for identification and diagnostics
    fn brick_name(&self) -> &'static str;

    /// Assertions that MUST hold. Checked before every render.
    fn assertions(&self) -> Vec<BrickAssertion>;

    /// Performance budget. Jidoka stops rendering if exceeded.
    fn budget(&self) -> BrickBudget;

    /// Run all assertions, return pass/fail with diagnostics.
    fn verify(&self) -> BrickVerification;
}
```

### 2.2 BrickAssertion

```rust
pub struct BrickAssertion {
    pub name: &'static str,
    pub kind: AssertionKind,  // Invariant, Precondition, Postcondition
    pub holds: bool,
    pub message: Option<String>,
}
```

### 2.3 BrickBudget

```rust
pub struct BrickBudget {
    pub render_ms: u32,       // Max render time
    pub memory_bytes: usize,  // Max heap allocation
    pub cell_writes: usize,   // Max dirty cells per frame
}
```

---

## 3. BrickHouse Composition

### 3.1 ttop v2 House

```rust
let house = BrickHouseBuilder::new("ttop")
    .budget_ms(16)  // 60fps frame budget
    .brick(Arc::new(CpuBrick::new()), 2)
    .brick(Arc::new(MemoryBrick::new()), 2)
    .brick(Arc::new(DiskBrick::new()), 1)
    .brick(Arc::new(NetworkBrick::new()), 1)
    .brick(Arc::new(ProcessBrick::new()), 3)
    .brick(Arc::new(GpuBrick::new()), 1)
    .brick(Arc::new(SensorsBrick::new()), 1)
    .brick(Arc::new(ConnectionsBrick::new()), 1)
    .brick(Arc::new(FilesBrick::new()), 1)
    .brick(Arc::new(OverlayBrick::new()), 1)
    .build()?;
// Total: 14ms / 16ms = 87.5% budget utilization
```

### 3.2 Jidoka (Stop-the-Line)

If any brick exceeds its budget during rendering:

1. BrickHouse records the violation
2. Current frame is discarded (not flushed to terminal)
3. Next frame reduces detail level for the offending panel
4. After 3 consecutive violations, panel enters "minimal" mode
5. Violation logged to stderr for diagnostics

This prevents a slow panel from causing visible stuttering.

---

## 4. Panel Brick Implementation

### 4.1 PanelBrick Trait

```rust
pub trait PanelBrick: Brick {
    /// Render panel content into CellBuffer at given area.
    fn render(&self, app: &App, buf: &mut CellBuffer, area: Rect);

    /// Generate title string from current app state.
    fn title(&self, app: &App) -> String;

    /// Minimum (width, height) for any rendering.
    fn min_size(&self) -> (u16, u16);

    /// Compute detail level from available height.
    fn detail_level(&self, height: u16) -> DetailLevel;
}
```

### 4.2 Example: CpuBrick

```rust
pub struct CpuBrick;

impl Brick for CpuBrick {
    fn brick_name(&self) -> &'static str { "cpu" }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            // These are checked BEFORE render() is called
            BrickAssertion::invariant("has_cores",
                true), // populated by collector
            BrickAssertion::invariant("history_bounded",
                true), // ring buffer has fixed capacity
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            render_ms: 2,
            memory_bytes: 0,      // zero-alloc rendering
            cell_writes: 10_000,  // max cells for large terminal
        }
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification::from_assertions(self.assertions())
    }
}

impl PanelBrick for CpuBrick {
    fn render(&self, app: &App, buf: &mut CellBuffer, area: Rect) {
        contract_pre_panel_render_bounds!();
        let detail = self.detail_level(area.height);
        // ... render CPU meters, history, top consumers ...
        contract_post_panel_render_bounds!();
    }

    fn title(&self, app: &App) -> String {
        format!(" CPU {}% | {} cores ", app.cpu_total, app.cpu_count)
    }

    fn min_size(&self) -> (u16, u16) { (30, 6) }

    fn detail_level(&self, h: u16) -> DetailLevel {
        match h {
            0..=5 => DetailLevel::Minimal,
            6..=8 => DetailLevel::Compact,
            9..=14 => DetailLevel::Normal,
            15..=19 => DetailLevel::Expanded,
            _ => DetailLevel::Exploded,
        }
    }
}
```

---

## 5. Falsification Test Methodology

### 5.1 Severity Levels (Popperian)

| Level | Meaning | Example |
|-------|---------|---------|
| S5 | Critical | Panel crashes on render |
| S4 | High | Data displayed is wrong |
| S3 | Medium | Visual artifact (misaligned) |
| S2 | Low | Cosmetic (color slightly off) |
| S1 | Info | Performance not optimal |

All tests MUST be S3+ (likely to fail if bug exists).
S1-S2 tests are waste (Muda) — they pass even with bugs.

### 5.2 F-Series Test Catalog

| Series | Scope | Example |
|--------|-------|---------|
| F500 | Analyzer parity | `F501: connections parses IPv6` |
| F600 | Panel features | `F601: CPU shows per-core frequency` |
| F700 | Pixel comparison | `F701: CLD vs ttop 0.3.x < 0.1%` |
| F800 | Data accuracy | `F801: mem_used <= mem_total` |
| F900 | Anti-regression | `F901: no panic on 1x1 terminal` |

### 5.3 Writing a Falsification Test

```rust
/// F601: CPU panel displays per-core frequency when detail >= Normal.
///
/// Falsifiable claim: If we render CPU at height >= 9, the output
/// contains frequency values matching "X.XXGHz" pattern.
///
/// Severity: S3 (visual correctness)
#[test]
fn f601_cpu_shows_per_core_freq() {
    let app = App::new_deterministic();
    let mut buf = CellBuffer::new(80, 20);
    let area = Rect::new(0, 0, 80, 20);

    CpuBrick.render(&app, &mut buf, area);

    let text = buf.as_text();
    assert!(text.contains("GHz"),
        "F601 FALSIFIED: CPU panel at Normal detail must show frequency");
}
```

---

## 6. Pixel Parity Testing

### 6.1 Methodology

Both ttop 0.3.x and ttop v2 support `--deterministic` mode with
identical synthetic data. Captures are compared using:

| Metric | Threshold | Tool |
|--------|-----------|------|
| Character-Level Diff (CLD) | < 0.001 (0.1%) | Custom diff |
| CIEDE2000 Color Diff | deltaE00 < 1.0 | perceptual color |
| Structural Similarity | SSIM > 0.99 | image comparison |

### 6.2 Pixel Parity Test

```rust
#[test]
fn f700_pixel_parity_cpu_panel() {
    // Render ttop 0.3.x reference
    let ref_buf = render_ttop_v1_cpu(80, 24);
    // Render ttop v2
    let new_buf = render_ttop_v2_cpu(80, 24);

    let cld = character_level_diff(&ref_buf, &new_buf);
    assert!(cld < 0.001,
        "F700: CLD {cld} exceeds 0.1% threshold");
}
```

---

## 7. Probar Coverage

### 7.1 TUI Snapshot Testing

```rust
use jugar_probar::tui::{TuiFrame, TuiSnapshot, SoftAssertions};

#[test]
fn probar_cpu_panel_snapshot() {
    let app = App::new_deterministic();
    let mut buf = CellBuffer::new(80, 24);
    CpuBrick.render(&app, &mut buf, Rect::new(0, 0, 80, 24));

    let frame = TuiFrame::from_cell_buffer(&buf);
    let snap = TuiSnapshot::from_frame("cpu_80x24", &frame);

    let mut soft = SoftAssertions::new();
    soft.assert_contains(&snap.text(), "CPU", "title present");
    soft.assert_contains(&snap.text(), "%", "percentage shown");
    soft.assert_true(snap.has_border(), "btop border present");
    soft.verify().expect("cpu panel assertions");
}
```

### 7.2 Coverage Targets

| Scope | Target |
|-------|--------|
| Line coverage | >= 95% |
| Branch coverage | >= 85% |
| Mutation score | >= 80% |
| Brick assertion coverage | 100% |
| F-series test count | >= 186 |

---

## 8. References

- Popper, K. (1963). *Conjectures and Refutations*. Routledge.
- Beck, K. (2002). *Test-Driven Development*. Addison-Wesley.
- PROBAR-SPEC-009: Brick Architecture specification
- jugar-probar 1.0.x: Brick, BrickHouse, TuiFrame, SoftAssertions
- Ohno, T. (1988). Jidoka (autonomation). *Toyota Production System*.
