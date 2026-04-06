# 260-Point Popperian Falsification QA Checklist

> **Status:** ACTIVE
> **Target Specification:** PROBAR-SPEC-009 (Brick Architecture)
> **Total Points:** 260
> **Pass Threshold:** 234/260 (90%)
> **Governance:** Zero-Tolerance for Falsification. Any "FALSIFIED" result triggers immediate Jidoka (Stop-the-Line).

---

## Abstract

This checklist operationalizes the scientific method for software quality assurance. Each item is a **falsifiable hypothesis** (Popper, 1959). Testing aims not to "pass," but to aggressively attempt to **falsify** the architecture's claims. If the architecture survives this rigorous falsification, it is deemed valid.

---

## Checklist Categories

| Category | Domain | Points | Focus |
|---|---|---|---|
| **A** | Compile-Time Guarantees | 25 | Type system enforcement, trait bounds |
| **B** | Runtime Assertions | 25 | AssertionValidator, Jidoka gates |
| **C** | Code Generation | 20 | Determinism, validity of artifacts |
| **D** | Presentar Integration | 15 | Widget-Brick unification, rendering |
| **E** | Distributed Tracing | 10 | Dapper compliance, span propagation |
| **F** | Error Handling | 5 | Yuan Gate, zero-swallow policy |
| **G** | Zero Hand-Written | 10 | No manual HTML/CSS/JS |
| **H** | WASM-First | 10 | web-sys, closures, minimal glue |
| **I** | Performance Budget | 15 | BrickHouse, RTF tracking |
| **J** | Performance UX | 10 | ttop-style visualization |
| **K** | Dual-Target Rendering | 15 | TUI/WASM parity, SIMD |
| **L** | Trueno-Viz | 10 | GPU/SIMD integration |
| **M** | Canonical Implementation | 10 | whisper.apr validation |
| **N** | Advanced Phases (7-13) | 80 | Zero-Artifact, Compute, Orchestration, TUI |
| **Total** | | **260** | |

---

## Category A: Compile-Time Guarantees (25 pts)

**Hypothesis:** The Rust type system prevents invalid architectural states from compiling.

- [ ] **A1** (2 pts): `impl Widget for T` fails if `T` does not implement `Brick`.
- [ ] **A2** (2 pts): `Button::default()` fails to compile (must be initialized with a Brick).
- [ ] **A3** (2 pts): Constructing a Widget struct directly (bypassing `new(Brick)`) fails.
- [ ] **A4** (2 pts): `ElementSpec::new("")` (empty selector) fails to compile (or const panic).
- [ ] **A5** (2 pts): `AriaSpec` without label/labelledby fails construction validation.
- [ ] **A6** (2 pts): `TransitionSpec` requires explicit `from` and `to` states.
- [ ] **A7** (2 pts): `VisualSpec` color validation rejects invalid hex strings.
- [ ] **A8** (2 pts): `WorkerSpec` requires a defined message enum.
- [ ] **A9** (2 pts): `TraceSpec` requires a non-empty span name.
- [ ] **A10** (2 pts): `ModelSpec` requires a content hash.
- [ ] **A11** (3 pts): Two `#[brick]` macros with the same ID in a module trigger a compile error.
- [ ] **A12** (2 pts): The `.prs` schema validator rejects manifests missing the `assertions` section.

---

## Category B: Runtime Assertions (25 pts)

**Hypothesis:** Runtime checks aggressively enforce correctness and stop execution on failure.

- [ ] **B1** (3 pts): `verify_assertions()` is called before every render cycle.
- [ ] **B2** (3 pts): If `verify_assertions()` returns `Err`, rendering is blocked.
- [ ] **B3** (3 pts): Every state transition triggers `validate_post_transition`.
- [ ] **B4** (3 pts): A failed transition assertion triggers Jidoka (application halt).
- [ ] **B5** (2 pts): Element assertions correctly verify DOM existence.
- [ ] **B6** (2 pts): Visual assertions correctly verify WCAG AA contrast ratios.
- [ ] **B7** (2 pts): Worker assertions verify the correct "Ready" message type.
- [ ] **B8** (2 pts): Trace assertions verify correct parent-child span hierarchy.
- [ ] **B9** (2 pts): Performance assertions verify RTF meets targets.
- [ ] **B10** (3 pts): The "Yuan Gate" crashes the app on any swallowed error (`_ => {}`).

---

## Category C: Code Generation (20 pts)

**Hypothesis:** Generated code is deterministic, valid, and complete.

- [ ] **C1** (2 pts): `ElementSpec` generates W3C-valid HTML.
- [ ] **C2** (2 pts): `VisualSpec` generates valid CSS.
- [ ] **C3** (2 pts): `InteractionSpec` generates valid, lint-free JavaScript.
- [ ] **C4** (2 pts): `WorkerSpec` generates valid ES Modules (no `importScripts`).
- [ ] **C5** (2 pts): `TraceSpec` generates correct Dapper context propagation code.
- [ ] **C6** (2 pts): Code generation is deterministic (same input = identical output byte-for-byte).
- [ ] **C7** (2 pts): Generated artifacts preserve Brick IDs for traceability.
- [ ] **C8** (2 pts): The generated `.prs` manifest includes all defined assertions.
- [ ] **C9** (2 pts): Generated TypeScript types match the `.prs` schema.
- [ ] **C10** (2 pts): The build process succeeds without ANY hand-written web code.

---

## Category D: Presentar Integration (15 pts)

**Hypothesis:** Presentar acts as a verification engine, not just a renderer.

- [ ] **D1** (3 pts): Presentar enforces `Widget : Brick` trait bounds.
- [ ] **D2** (2 pts): Presentar rejects loading `.prs` v2.0 files without assertions.
- [ ] **D3** (2 pts): Widgets fail to render if the backing Brick is missing.
- [ ] **D4** (2 pts): The `AssertionValidator` runs its pre-render check.
- [ ] **D5** (2 pts): The WebGPU rendering path is gated by assertion success.
- [ ] **D6** (2 pts): Validation overhead does not prevent maintaining 60fps.
- [ ] **D7** (2 pts): `AriaSpec` enforces WCAG AA compliance by construction.

---

## Category E: Distributed Tracing (10 pts)

**Hypothesis:** Causal ordering is preserved across asynchronous boundaries.

- [ ] **E1** (2 pts): `trace_id` is correctly propagated via `postMessage`.
- [ ] **E2** (2 pts): `span_id` correctly links parent and child spans.
- [ ] **E3** (2 pts): Traces reflect the correct causal ordering of events.
- [ ] **E4** (2 pts): Trace context survives `await` points and async boundaries.
- [ ] **E5** (2 pts): Spans include accurate start/end timestamps.

---

## Category F: Error Handling (5 pts)

**Hypothesis:** Errors are never silently ignored.

- [ ] **F1** (2 pts): No wildcard (`_ => {}`) enum matches exist in the codebase.
- [ ] **F2** (1 pt): All `Result` types are propagated or handled (no `.ok()`).
- [ ] **F3** (1 pt): Error messages include source context (file/line).
- [ ] **F4** (1 pt): Panics in Brick code correctly trigger test failures.

---

## Category G: Zero Hand-Written Web Code (10 pts)

**Hypothesis:** The application is built entirely from Rust specifications.

- [ ] **G1** (2 pts): No `.html` files exist in the `src/` directory.
- [ ] **G2** (2 pts): No `.css` files exist in the `src/` directory.
- [ ] **G3** (2 pts): No `.js` files exist in the `src/` directory (except generated).
- [ ] **G4** (2 pts): Total generated JS glue code is ≤ 50 lines.
- [ ] **G5** (2 pts): All web artifacts are located in the `generated/` directory.

---

## Category H: WASM-First Architecture (10 pts)

**Hypothesis:** Logic resides in Rust/WASM, not JavaScript.

- [ ] **H1** (2 pts): DOM manipulation is performed exclusively via `web-sys`.
- [ ] **H2** (2 pts): Event handling uses `wasm-bindgen` closures.
- [ ] **H3** (2 pts): Application state is managed entirely within Rust.
- [ ] **H4** (2 pts): Rendering logic is executed via WASM/WebGPU.
- [ ] **H5** (2 pts): Worker business logic is implemented in WASM.

---

## Category I: Performance Budget (15 pts)

**Hypothesis:** Performance is treated as a hard constraint.

- [ ] **I1** (3 pts): `BrickHouse` validates that the sum of budgets ≤ total budget.
- [ ] **I2** (2 pts): Every Brick definition includes a `budget_ms`.
- [ ] **I3** (3 pts): Runtime verification catches and reports budget violations.
- [ ] **I4** (2 pts): `BudgetReport` accurately lists all violating bricks.
- [ ] **I5** (2 pts): Panels render within their assigned 50ms budget.
- [ ] **I6** (2 pts): Sparklines render within their assigned 10ms budget.
- [ ] **I7** (1 pt): The system maintains 60fps (16ms frame time).

---

## Category J: Performance UX (10 pts)

**Hypothesis:** Performance metrics are visualized effectively.

- [ ] **J1** (2 pts): `PanelSpec` generates correctly bordered panels.
- [ ] **J2** (2 pts): `MeterSpec` generates functional progress bars/gauges.
- [ ] **J3** (2 pts): `SparklineSpec` includes trend indicators.
- [ ] **J4** (2 pts): `GraphSpec` correctly visualizes data sources.
- [ ] **J5** (2 pts): `TuiLayoutSpec` creates valid `ratatui` layouts.

---

## Category K: Dual-Target Rendering (15 pts)

**Hypothesis:** The same Brick renders correctly to both TUI and WASM.

- [ ] **K1** (3 pts): A single Brick renders to both TUI and WASM targets.
- [ ] **K2** (2 pts): TUI output uses native `ratatui` widgets.
- [ ] **K3** (2 pts): WASM output uses `wgpu` (WebGPU) rendering.
- [ ] **K4** (2 pts): Both targets share the same SIMD-optimized ring buffers.
- [ ] **K5** (2 pts): `make_contiguous()` is used for SIMD operations.
- [ ] **K6** (2 pts): WASM uses the WebGPU backend in the browser.
- [ ] **K7** (2 pts): Both targets achieve 60fps performance.

---

## Category L: Trueno-Viz Integration (10 pts)

**Hypothesis:** Scientific computing libraries are integrated correctly.

- [ ] **L1** (2 pts): `RingBuffer` implementation comes from `trueno-viz`.
- [ ] **L2** (2 pts): Color gradients match `ttop`'s perceptual logic.
- [ ] **L3** (2 pts): WASM builds enable 128-bit SIMD.
- [ ] **L4** (2 pts): `wgpu` version aligns with `trueno-viz` dependencies.
- [ ] **L5** (2 pts): Counter wrapping is handled correctly (per `ttop`).

---

## Category M: Canonical Implementation (10 pts)

**Hypothesis:** `whisper.apr` serves as a valid reference implementation.

- [ ] **M1** (2 pts): `whisper.apr` has zero hand-written HTML.
- [ ] **M2** (2 pts): `whisper.apr` TUI and WASM demos share the same bricks.
- [ ] **M3** (2 pts): `whisper.apr` passes all applicable falsification checks.
- [ ] **M4** (2 pts): `whisper.apr` achieves RTF ≤ 2.0x.
- [ ] **M5** (2 pts): `whisper.apr` tests are run via `probar`.

---

## Category N: Advanced Phases 7-13 (80 pts)

**Hypothesis:** Advanced architectural features are functional and integrated.

### Phase 7: Zero-Artifact (10 pts)
- [ ] **N1** (2 pts): `WorkerBrick` generates valid worker JS.
- [ ] **N2** (2 pts): `EventBrick` generates valid event handlers.
- [ ] **N3** (2 pts): `AudioBrick` generates valid AudioWorklet code.
- [ ] **N4** (2 pts): Zero hand-written HTML/CSS/JS/WGSL in the build.
- [ ] **N5** (2 pts): All `web_sys` calls are generated from bricks.

### Phase 8: ComputeBrick (10 pts)
- [ ] **N6** (2 pts): `ComputeBrick` generates valid WGSL shaders.
- [ ] **N7** (2 pts): Tile operations map correctly to GPU instructions.
- [ ] **N8** (2 pts): Numerical output matches CPU reference implementation.
- [ ] **N9** (2 pts): Zero hand-written WGSL shaders exist.
- [ ] **N10** (2 pts): GPU fallback mechanisms are handled gracefully.

### Phase 9: Orchestration (10 pts)
- [ ] **N11** (2 pts): `BrickPipeline` executes stages in the correct order.
- [ ] **N12** (2 pts): Privacy tiers (Sovereign/Private) are enforced.
- [ ] **N13** (2 pts): Checkpointing allows pipeline recovery.
- [ ] **N14** (2 pts): Audit trails are complete and verifiable.
- [ ] **N15** (2 pts): Jidoka validation runs before each pipeline stage.

### Phase 10: Distribution (10 pts)
- [ ] **N16** (2 pts): `DistributedBrick` executes on remote nodes.
- [ ] **N17** (2 pts): Data locality tracking improves scheduling.
- [ ] **N18** (2 pts): Work-stealing successfully balances load.
- [ ] **N19** (2 pts): PUB/SUB coordination propagates updates.
- [ ] **N20** (2 pts): Multi-backend selection chooses the optimal target.

### Phase 11: Determinism (10 pts)
- [ ] **N21** (2 pts): `DeterministicBrick` execution is reproducible.
- [ ] **N22** (2 pts): Persistent data structures enable state snapshots.
- [ ] **N23** (2 pts): Time-travel debugging features function correctly.
- [ ] **N24** (2 pts): Jidoka guards catch invariant violations.
- [ ] **N25** (2 pts): Same input consistently produces identical output.

### Phase 12: Widget Integration (10 pts)
- [ ] **N26** (2 pts): Widget and Brick traits are unified.
- [ ] **N27** (2 pts): Verification blocks invalid rendering attempts.
- [ ] **N28** (2 pts): DrawCommands are correctly batched for the GPU.
- [ ] **N29** (2 pts): Canvas2D fallback functions correctly.
- [ ] **N30** (2 pts): 60fps budget is maintained during integration.

### Phase 13: TUI Integration (10 pts)
- [ ] **N31** (2 pts): `CollectorBrick` gathers accurate system metrics.
- [ ] **N32** (2 pts): `AnalyzerBrick` produces valid insights.
- [ ] **N33** (2 pts): `PanelBrick` renders within 8ms.
- [ ] **N34** (2 pts): `RingBuffer` maintains accurate time-series history.
- [ ] **N35** (2 pts): Panel state machine transitions are correct.

---

## Execution Instructions

1.  **Setup:** Ensure the `probar` CLI and `whisper.apr` repository are checked out.
2.  **Run:** Execute the falsification test suite:
    ```bash
    cargo test --package whisper-apr-demo-tests falsification_tests -- --nocapture
    ```
3.  **Score:** The test output will provide a point breakdown.
4.  **Verify:** Manually verify any non-automated items using this checklist.
5.  **Report:** Commit the completed checklist to `docs/qa/reports/FALSIFICATION-REPORT-YYYY-MM-DD.md`.

## Failure Protocol

If the total score is **< 234 points**:
1.  **HALT** all feature development.
2.  Identify the falsified hypothesis.
3.  Execute **Five-Whys** root cause analysis.
4.  Implement a fix or update the specification if the hypothesis was flawed.
5.  Re-run the full falsification suite.
