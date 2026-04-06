# Migration Plan

> Parent: [ttop-v2-spec.md](../ttop-v2-spec.md) Section 11

**Scope:** Phase-by-phase ratatui removal, extraction from presentar-terminal
ptop, parity validation, release process.

---

## 1. Strategy

ttop v2 is NOT a rewrite from scratch. presentar-terminal already implements
`ptop` — a pixel-perfect ttop clone with all 14 panels, 186 falsification
tests, and SPEC-024 compliance. The migration is a structured extraction.

### 1.1 Source Mapping

| ttop v2 Module | Source |
|---------------|--------|
| `src/app.rs` | `presentar-terminal/src/ptop/app.rs` |
| `src/config.rs` | `presentar-terminal/src/ptop/config.rs` |
| `src/panels/*.rs` | `presentar-terminal/src/ptop/ui/panels/*.rs` |
| `src/ui.rs` | `presentar-terminal/src/ptop/ui/core/*.rs` |
| `src/theme.rs` | `presentar-terminal/src/ptop/ui/colors.rs` + `helpers.rs` |
| `src/analyzers/*.rs` | `presentar-terminal/src/ptop/analyzers/*.rs` |
| `tests/` | `presentar-terminal/tests/ptop_*.rs` + `F*` tests |
| `build.rs` | New (provable-contracts generation) |
| `contracts/` | New (YAML contract definitions) |

### 1.2 Dependency Swap

```toml
# BEFORE (ttop 0.3.x)
[dependencies]
trueno-viz = { version = "0.2.4", features = ["monitor"] }  # brings ratatui

# AFTER (ttop 2.0.0)
[dependencies]
presentar-terminal = { version = "0.3", features = [] }
presentar-core = "0.3"
provable-contracts = "0.2"
crossterm = "0.28"
sysinfo = "0.33"
clap = { version = "4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_yaml_ng = "0.10"
batuta-common = "0.1"

[dev-dependencies]
jugar-probar = { version = "1.0", features = ["tui"] }
proptest = "1.4"
```

---

## 2. Phases

### Phase 0: Contracts + Test Stubs (Day 1, Morning)

**Goal:** All contracts compile, all test stubs exist, build passes.

1. Create `contracts/` directory with YAML definitions:
   - `panel.yaml` — PanelBrick render bounds, title format
   - `render.yaml` — CellBuffer invariants, DiffRenderer correctness
   - `data.yaml` — Collector value ranges, memory consistency

2. Create `build.rs` with provable-contracts generation

3. Create test stub files (empty tests with `#[test]` and `todo!()`):
   - `tests/brick_interface.rs`
   - `tests/panel_falsification.rs`
   - `tests/pixel_parity.rs`
   - `tests/probar_full.rs`

4. Add `include_str!` enforcement in `src/panels/mod.rs`

5. Verify: `cargo check` passes, `cargo test` shows all stubs as `todo!` panics

**Gate:** `cargo check` passes. Contract generation succeeds.

### Phase 1: Rendering Backend Swap (Day 1, Afternoon)

**Goal:** Replace ratatui Terminal/Frame with CellBuffer/DiffRenderer.

1. Remove `trueno-viz` dependency from `Cargo.toml`
2. Add `presentar-terminal` and `presentar-core` dependencies
3. Replace `src/main.rs` event loop:
   - `Terminal<CrosstermBackend>` → `CellBuffer` + `DiffRenderer`
   - `terminal.draw(|f| ui::draw(f, &mut app))` → `ui::draw(&app, &mut buf)` + `renderer.render_diff(&prev, &buf)`
4. Replace `src/ui.rs`:
   - `fn draw(f: &mut Frame, app: &mut App)` → `fn draw(app: &App, buf: &mut CellBuffer)`
   - Layout uses `Rect` from `presentar-core` instead of ratatui
5. Verify: `cargo build` passes. Binary starts and shows empty terminal.

**Gate:** Binary starts. CellBuffer renders at least a border.

### Phase 2: P0 Panel Migration (Days 2-3)

**Goal:** CPU, Memory, Disk, Network, Process panels render correctly.

For each P0 panel:

1. **Write falsification tests first** (F600 series)
2. Copy panel source from `presentar-terminal/src/ptop/ui/panels/`
3. Adapt imports (ptop's `App` → ttop's `App` field mapping)
4. Implement `PanelBrick` and `Brick` traits
5. Add to BrickHouse
6. Run falsification tests — must pass
7. Run probar snapshot — capture reference

Order: CPU → Memory → Disk → Network → Process

**Gate:** 5 panels render. F600-F605 pass. BrickHouse budget < 16ms.

### Phase 3: P1-P3 Panel Migration (Days 3-4)

**Goal:** All 14 panels render correctly.

Same process as Phase 2 for remaining panels:
GPU, Battery, Sensors, PSI, System, Connections, Treemap, Files, Containers.

**Gate:** All 14 panels render. F600-F631 pass. Full BrickHouse.

### Phase 4: Parity Validation (Day 5)

**Goal:** Pixel parity with ttop 0.3.x confirmed.

1. Run both binaries with `--deterministic` flag
2. Capture CellBuffer output from both
3. Compute CLD, deltaE00, SSIM metrics
4. Run F700 series tests
5. Fix any parity deviations
6. Run full probar suite
7. Generate coverage report — must be >= 95%
8. Run cargo-mutants — must be >= 80%
9. Run `provable-contracts lint` — all contracts valid

**Gate:** CLD < 0.1%. Score >= 980. Coverage >= 95%.

### Phase 5: Release (Day 6)

**Goal:** ttop 2.0.0 published to crates.io.

1. Version bump: 0.3.x → 2.0.0 (major version for breaking change)
2. Update README.md
3. Update CHANGELOG.md
4. Run clean-room build (`make clean-room-p1`)
5. Verify all CI green
6. `cargo publish`
7. Verify `cargo install ttop` works
8. Tag release on GitHub

**Gate:** Clean-room passes. `cargo install ttop` works.

---

## 3. Risk Mitigation

### 3.1 Risk: App State Divergence

ptop's `App` struct uses `sysinfo` directly while ttop's `App` uses
trueno-viz collectors. Mitigation: ttop v2 adopts ptop's `App` (sysinfo-based),
which is simpler and has no trueno-viz dependency.

### 3.2 Risk: Widget API Mismatch

ptop widgets render to CellBuffer using internal presentar-terminal types.
If those types aren't public, ttop can't use them. Mitigation: presentar-terminal
already exports `CellBuffer`, `DiffRenderer`, and all widget types publicly.

### 3.3 Risk: Analyzer Divergence

ttop 0.3.x and ptop have separate analyzer implementations that may differ.
Mitigation: ptop's analyzers are tested against ttop's (F500 series parity
tests). Use ptop's implementations which are newer and better tested.

### 3.4 Risk: Clean-Room Failure

presentar-terminal may not be published to crates.io at the required version.
Mitigation: Verify `presentar-terminal` crates.io availability before Phase 1.
If not published, publish it first (it has its own clean-room gate).

---

## 4. Rollback Plan

Each phase produces a working (if incomplete) binary. If any phase fails:

1. `git revert` to previous phase's commit
2. Diagnose failure using Five Whys
3. Fix root cause
4. Retry phase

The old ttop 0.3.x remains published on crates.io. Users are not affected
until ttop 2.0.0 is explicitly published.

---

## 5. Verification Matrix

| Phase | Compile | Tests | Probar | Contracts | Parity | Clean-Room |
|-------|---------|-------|--------|-----------|--------|------------|
| 0 | PASS | STUB | - | PASS | - | - |
| 1 | PASS | PARTIAL | - | PASS | - | - |
| 2 | PASS | PASS (P0) | PASS (P0) | PASS | - | - |
| 3 | PASS | PASS (all) | PASS (all) | PASS | - | - |
| 4 | PASS | PASS | PASS | PASS | PASS | - |
| 5 | PASS | PASS | PASS | PASS | PASS | PASS |

---

## 6. Post-Migration Cleanup

After ttop 2.0.0 ships:

1. Remove `trueno-viz` `monitor` feature (if ttop was the only consumer)
2. Archive ttop 0.3.x test files
3. Update batuta oracle references
4. Update CLAUDE.md stack table
5. Close ttop-related GitHub issues

---

## 7. References

- presentar-terminal `ptop/` module: source for extraction
- SPEC-024: presentar ptop architecture enforcement
- Clean-room policy: `../infra/machines/clean-room/`
- Toyota Way: Jidoka (stop-on-defect), Five Whys (root cause)
