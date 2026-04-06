# Contract Enforcement

> Parent: [ttop-v2-spec.md](../ttop-v2-spec.md) Section 4

**Scope:** YAML contract definitions, build.rs code generation, compile-time
enforcement, Kani proof harnesses, probar property test generation.

---

## 1. Overview

ttop v2 uses `provable-contracts` (0.2.x) to define formal contracts for
every public interface. Contracts are:

1. **Defined** in YAML (`contracts/*.yaml`)
2. **Generated** into Rust enforcement code by `build.rs`
3. **Verified** at compile time (type-level) and runtime (debug assertions)
4. **Proven** via Kani bounded model checking (optional, CI only)
5. **Tested** via probar property-based test generation

This follows Meyer's Design by Contract (DbC) methodology (IEEE Computer, 1992)
adapted to Rust's ownership model.

---

## 2. Contract YAML Schema

### 2.1 Function Contract

```yaml
# contracts/panel.yaml
contracts:
  - name: panel_render_bounds
    module: panels
    function: "PanelBrick::render"
    description: "Panel rendering stays within allocated Rect"
    preconditions:
      - expr: "area.width >= self.min_size().0"
        description: "Area meets minimum width"
      - expr: "area.height >= self.min_size().1"
        description: "Area meets minimum height"
    postconditions:
      - expr: "buf.dirty_cells_within(area)"
        description: "All dirty cells are within the panel's Rect"
    invariants:
      - expr: "buf.width > 0 && buf.height > 0"
        description: "Buffer has positive dimensions"
    performance:
      budget_ms: 2
      measurement: wall_clock

  - name: cpu_percent_range
    module: app
    function: "App::cpu_percent"
    description: "CPU percentage is in valid range"
    postconditions:
      - expr: "result >= 0.0 && result <= 100.0 * num_cores as f64"
        description: "CPU percent bounded by core count"
```

### 2.2 Data Contract

```yaml
# contracts/data.yaml
contracts:
  - name: memory_values_consistent
    module: app
    function: "App::collect_memory"
    description: "Memory values are internally consistent"
    postconditions:
      - expr: "self.mem_used <= self.mem_total"
        description: "Used memory cannot exceed total"
      - expr: "self.mem_cached <= self.mem_total"
        description: "Cached memory cannot exceed total"
    invariants:
      - expr: "self.mem_total > 0"
        description: "Total memory is always positive"
```

### 2.3 Layout Contract

```yaml
# contracts/render.yaml
contracts:
  - name: layout_grid_fits
    module: ui
    function: "calculate_layout"
    description: "Panel grid fits within terminal dimensions"
    preconditions:
      - expr: "terminal_width >= 40"
        description: "Minimum terminal width"
      - expr: "terminal_height >= 10"
        description: "Minimum terminal height"
    postconditions:
      - expr: "panels.iter().all(|p| p.x + p.width <= terminal_width)"
        description: "No panel exceeds terminal width"
      - expr: "panels.iter().all(|p| p.y + p.height <= terminal_height)"
        description: "No panel exceeds terminal height"
      - expr: "!panels.windows(2).any(|w| rects_overlap(w[0], w[1]))"
        description: "No panels overlap"
```

---

## 3. Build-Time Generation

### 3.1 build.rs

```rust
fn main() {
    // Generate contract enforcement macros from YAML
    provable_contracts::build_helper::generate(
        "contracts/",               // YAML source directory
        "src/generated_contracts.rs" // Generated output
    ).expect("contract generation failed");

    // Rerun if contracts change
    println!("cargo:rerun-if-changed=contracts/");
    println!("cargo:rerun-if-changed=build.rs");
}
```

### 3.2 Generated Code

For each contract, `provable-contracts` generates:

```rust
// src/generated_contracts.rs (auto-generated, do not edit)

/// Panel render bounds enforcement (panel.yaml:panel_render_bounds)
macro_rules! contract_pre_panel_render_bounds {
    () => {
        debug_assert!(area.width >= self.min_size().0,
            "contract violation: panel_render_bounds pre: area meets minimum width");
        debug_assert!(area.height >= self.min_size().1,
            "contract violation: panel_render_bounds pre: area meets minimum height");
    };
}

macro_rules! contract_post_panel_render_bounds {
    () => {
        debug_assert!(buf.dirty_cells_within(area),
            "contract violation: panel_render_bounds post: dirty cells within Rect");
    };
}
```

### 3.3 Usage in Source

```rust
impl PanelBrick for CpuPanel {
    fn render(&self, app: &App, buf: &mut CellBuffer, area: Rect) {
        contract_pre_panel_render_bounds!();

        // ... rendering logic ...

        contract_post_panel_render_bounds!();
    }
}
```

---

## 4. Compile-Time Test Enforcement

### 4.1 include_str! Pattern (from presentar SPEC-024)

```rust
// src/panels/mod.rs
// These constants FAIL compilation if test files don't exist.
// Tests define the interface. Implementation follows.

#[doc(hidden)]
pub const _ENFORCE_BRICK_TESTS: &str =
    include_str!("../tests/brick_interface.rs");

#[doc(hidden)]
pub const _ENFORCE_FALSIFICATION: &str =
    include_str!("../tests/panel_falsification.rs");

#[doc(hidden)]
pub const _ENFORCE_PIXEL_PARITY: &str =
    include_str!("../tests/pixel_parity.rs");
```

### 4.2 Enforcement Guarantee

If any required test file is deleted, renamed, or missing:
```
error[E0433]: file not found: ../tests/brick_interface.rs
```

This is architectural enforcement, not advisory. You cannot compile
ttop v2 without its test suite.

---

## 5. Kani Proof Harnesses

### 5.1 Generated Proofs

`provable-contracts` generates Kani bounded model checking harnesses
for contracts with bounded input domains:

```rust
#[cfg(kani)]
#[kani::proof]
fn verify_cell_buffer_bounds() {
    let w: u16 = kani::any();
    let h: u16 = kani::any();
    kani::assume(w > 0 && w <= 300);
    kani::assume(h > 0 && h <= 100);

    let buf = CellBuffer::new(w, h);
    assert!(buf.cells.len() == w as usize * h as usize);
    assert!(buf.dirty.len() == buf.cells.len());
}
```

### 5.2 CI Integration

Kani proofs run in CI (Tier 4) only — they are too slow for local
development but provide mathematical certainty for critical invariants.

---

## 6. Probar Test Generation

`provable-contracts` also generates probar property-based tests:

```rust
// Auto-generated from contracts/panel.yaml
#[test]
fn probar_panel_render_bounds() {
    proptest!(|(w in 40u16..300, h in 10u16..100)| {
        let mut buf = CellBuffer::new(w, h);
        let area = Rect::new(0, 0, w, h);
        let panel = CpuPanel::new();
        panel.render(&mock_app(), &mut buf, area);
        prop_assert!(buf.dirty_cells_within(area));
    });
}
```

---

## 7. Contract Audit Trail

`provable-contracts audit` traces the full chain:

```
Meyer (1992) → panel_render_bounds.yaml → contract_pre_panel_render_bounds!()
    → tests/brick_interface.rs::test_cpu_render_bounds
    → kani::verify_panel_render_bounds (CI)
```

Every contract links back to its academic foundation, enabling
reproducible verification of the entire system.

---

## 8. References

- Meyer, B. (1992). Applying Design by Contract. *IEEE Computer*, 25(10).
- provable-contracts 0.2.x: YAML contract → Kani/probar verification
- PROBAR-SPEC-009: Brick Architecture enforcement
- Kani: Rust model checker (https://github.com/model-checking/kani)
