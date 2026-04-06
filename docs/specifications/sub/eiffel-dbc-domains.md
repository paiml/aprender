# Eiffel DbC — Domain Applicability (Part 1)

*See also: [eiffel-dbc.md](eiffel-dbc.md) (sections 1-5), [eiffel-dbc-type-invariants.md](eiffel-dbc-type-invariants.md) (sections 6-7)*

## 8. Domain Applicability

### 8.1. Meyer's Universality Argument

Meyer was emphatic in OOSC (1997, Ch. 11) that Design by Contract is
not a systems programming technique — it is a *software correctness*
technique that applies to every domain. His key claims:

1. **Abstract Data Types are universal.** Every domain has them. A
   `Stack` has `push`/`pop` contracts. A `Window` has display
   invariants. A `Transaction` has ACID properties. The mathematical
   structure differs but the contract *form* is identical.

2. **The most valuable contracts encode domain axioms.** `softmax sums
   to 1` is a *domain* contract (from the math). `output buffer not
   corrupted` is a *code* contract. Domain contracts catch deeper bugs
   because they encode what the software *means*, not just what it
   *does*.

3. **Command-Query Separation (CQS)** applies everywhere: functions
   that return values must not have side effects; procedures that change
   state must not return values. This principle is domain-independent.

4. **Seamless development:** contracts flow from *analysis* (the domain
   expert's model) through *design* to *implementation*. The domain
   expert's constraints become the developer's preconditions and
   postconditions directly — no translation layer.

### 8.2. Current Domain Coverage

Our contract system is heavily weighted toward **scientific/numerical
kernels**:

| Project | Domain | Contract Focus |
|---|---|---|
| aprender | ML inference library | Kernel math (softmax, attention, RoPE) |
| entrenar | Training pipeline | Performance budgets (KAIZEN), grad computation |
| trueno | SIMD/CUDA kernels | Low-level numerical equivalence (SIMD = scalar) |
| realizar | Task execution | Pipeline orchestration |
| bashrs | SSC pipeline | Encoder/classifier contracts |
| simular | Simulation engine | Energy conservation, checkpoint roundtrip, gradient stability |
| probar | WASM/TUI testing | (no contracts yet) GUI coverage, accessibility, visual regression |
| forjar | Infrastructure as Code | DAG ordering, atomic writes, recipe determinism, codegen dispatch |
| batuta | Orchestration / transpilation | (no contracts yet) Pipeline stages, semantic equivalence |
| pmat | Code quality / AI context | (no contracts yet) Uniform interface contracts, analysis invariants |
| rmedia | Headless video editor | (no contracts yet) Codec parity, determinism, SVG quality, animation timing |

Simular already consumes provable-contracts (3 YAML contracts +
`provable-contracts-macros` dependency). Forjar is also a direct
consumer (`#[contract]` macros on DAG ordering, atomic state writes,
recipe expansion, codegen dispatch + `build.rs` binding verification
with `WarnOnGaps` policy). Batuta and PMAT do not reference provable-
contracts but both operate in domains with rich contractable properties.
Probar does not yet reference provable-contracts but operates in a
domain — GUI/UX correctness — where Meyer's DbC types are maximally
applicable.

This covers Meyer's "scientific computing" domain thoroughly. But the
stack spans domains where our current property-only obligation types
are insufficient without the Eiffel DbC extensions.

### 8.3. Domain Contract Patterns

#### Presentation / UI (presentar + probar)

Presentar is the stack's WASM-first UI framework — a pure Rust
widget system with constraint-based layout (flexbox + CSS grid),
two-phase measure/layout engine, accessibility tree, real-time
WebSocket streaming, and the Brick Architecture (tests ARE the
interface). Probar tests presentar applications via CDP browser
automation, pixel coverage tracking, and state machine playbooks.

This domain has contracts rooted in *geometric invariants*,
*accessibility standards*, and *event semantics* — not numerical
equations. The concrete architecture maps to DbC as follows.

**Dimensions and Layout Contracts:**

The layout engine runs a two-phase algorithm: measure (bottom-up,
computing intrinsic sizes given `Constraints`) then layout (top-down,
assigning final `Rect` positions). Each phase has contractable
properties:

| Obligation Type | Layout Example |
|---|---|
| `precondition` | `Constraints` are valid: `min ≤ max`, both finite or explicitly unbounded |
| `postcondition` | After measure, returned `Size` satisfies constraints: `constraints.constrain(size) == size` |
| `invariant` | Every visible widget has `width > 0 ∧ height > 0` after layout |
| `frame` | Measure phase reads widget tree only; does not mutate positions or state |
| `conservation` | Flex layout: `Σ child_widths + Σ gaps = parent_content_width` (no pixels lost) |
| `bound` | Total measure + layout time < 16ms (BrickBudget: `measure_ms + layout_ms + paint_ms ≤ 16`) |
| `loop_invariant` | During flex distribution: remaining space ≥ 0 at each child allocation |
| `loop_variant` | Remaining unpositioned children = `total - positioned`, strictly decreasing |
| `old_state` | After resize event, `layout_tree.bounds(widget) ≠ old(layout_tree.bounds(widget))` for affected widgets |
| `determinism` | Same widget tree + same constraints → identical `LayoutTree` |

Concrete types from presentar that carry contracts:

| Type | Contract Surface |
|---|---|
| `Constraints { min_width, max_width, min_height, max_height }` | `tight()` pre: min == max. `loose()` pre: min ≤ max. `constrain()` post: result in [min, max] |
| `Size { width, height }` | Post: `width ≥ 0 ∧ height ≥ 0`. `area()` post: `= width × height` |
| `Rect { x, y, width, height }` | `contains(point)` post: `x ≤ p.x ≤ x+w ∧ y ≤ p.y ≤ y+h` |
| `FlexItem { grow, shrink, basis }` | Pre: `grow ≥ 0`, `shrink ≥ 0`. `collapse_if_empty` post: size = 0 when no content |
| `GridTemplate { columns, rows, gap }` | Pre: all `TrackSize` values ≥ 0. Post: `Σ column_widths + Σ gaps = available_width` |

**Element Contracts (Menu, Footer, Pane, Widget):**

Presentar's `Widget` trait (which extends `Brick`) defines the
verify-measure-layout-paint lifecycle. Each widget type has specific
contracts:

| Widget Element | Key Contracts |
|---|---|
| **Menu** | Pre: items list non-empty. Post: exactly one item has `selected` state. State machine: `closed → open → item_hover → selected → closed`. Frame: opening menu doesn't modify parent layout |
| **Footer** | Invariant: always pinned to viewport bottom (`y + height == viewport.height`). Frame: content updates preserve height unless content overflows |
| **Pane / Panel** | Pre: split ratio ∈ (0, 1). Conservation: `left_width + divider + right_width = parent_width`. Old-state: resizing divider changes ratio but preserves total width |
| **DataTable** | Pre: column count > 0. Post: sorted column satisfies ordering obligation. Idempotency: sorting already-sorted column is no-op. Virtual scroll invariant: only `visible_range + overscan` rows rendered |
| **TextInput** | Pre: `is_focusable() == true`. Post: after `TextInput` event, `value.len() = old(value.len()) + input.len()`. State machine: `unfocused → focused → editing → validated → unfocused` |
| **Button** | Pre: `accessible_name().is_some()`. Post: `click` event emitted only when `enabled`. Subcontract: `IconButton` refines `Button` (same click semantics, adds icon rendering) |
| **Scroll Container** | Loop invariant: `offset ≥ 0 ∧ offset ≤ content_height - viewport_height` at every scroll step. Old-state: `offset_after = old(offset) + delta` clamped to bounds |
| **Border** | Frame: adding border modifies rendered appearance only; child widget's content rect unchanged. Post: `inner_rect = outer_rect.deflate(border_width)` |

**Behavior Contracts (WebSocket, Forms, Events):**

| Behavior | Obligation Types |
|---|---|
| **WebSocket (StreamMessage)** | State machine: `Disconnected → Connecting → Connected → Reconnecting → Failed`. Pre: `ws_url` is valid URI. Post: after `Subscribe`, server sends `Data` messages with matching `id`. Invariant: `seq` numbers strictly increasing per stream. Frame: receiving messages doesn't modify application state until processed by `update()`. Loop invariant: reconnect backoff ≤ `max_backoff` at every retry. Loop variant: retry_count remaining = `max_retries - attempt` |
| **Form Validation** | Pre: all required fields bound via `Binding`. Post: `submit()` only succeeds if all validators pass. State machine: `pristine → dirty → validating → valid \| invalid → submitted`. Old-state: after validation, `errors.len()` either decreases (fixes) or increases (new violations). Frame: validation reads field values only; doesn't modify them |
| **Event Dispatch** | Pre: `event.target` is a valid widget ID in the tree. Post: exactly one widget handles each event (no duplicate dispatch). Frame: event handling modifies only the target widget's state and its ancestors (bubbling). Determinism: same event + same state → same `update()` result |
| **Two-Way Binding** | Invariant: `widget.value == state[binding.source_path]` at all quiescent states. Roundtrip: `set(get(path)) = get(path)` (setting the current value is a no-op). Frame: binding update modifies only the bound property; other state preserved |
| **Virtual Scrolling** | Pre: `estimated_item_height > 0`. Post: only items in `visible_range + overscan` are rendered. Conservation: total scrollable height = `Σ item_heights` (no gaps). Bound: render count ≤ `visible_count + 2 × overscan_count` |
| **Device Emulation** | Pre: `viewport.width > 0 ∧ viewport.height > 0`. Post: `device_scale_factor` applied to all coordinates. Frame: emulation changes viewport only; doesn't modify DOM content. Subcontract: `MobileDevice` refines `Device` (adds touch events, is_mobile = true) |

**Accessibility Contracts:**

| Obligation Type | Accessibility Example |
|---|---|
| `completeness` | Every `is_interactive()` widget has `accessible_name().is_some()` |
| `invariant` | Contrast ratio ≥ 4.5:1 for normal text, ≥ 3.0:1 for large text (WCAG 2.1 AA) |
| `bound` | Flash rate < 3 per second (WCAG 2.3.1 photosensitivity) |
| `postcondition` | After focus change, `AccessibleNode.focused == true` for exactly one node |
| `conservation` | Focus tab order: `Σ focusable_elements` unchanged after re-render |
| `frame` | Accessibility tree update modifies only changed nodes; unchanged nodes preserve all properties |
| `state_machine` | Live region: `Off → Polite → Assertive` transitions; `Assertive` interrupts screen reader immediately |

**Brick Architecture as DbC:**

Probar's Brick Architecture is the purest expression of Meyer's
DbC in the stack. Each `Brick` IS a contract:

```
Brick = {
    assertions: Vec<BrickAssertion>,  ← postconditions
    budget: BrickBudget,               ← bound obligations
    verify() → BrickVerification,      ← contract checker
}

Widget extends Brick:
    VERIFY  → check assertions     ← precondition gate
    MEASURE → compute intrinsic    ← pure function (frame: no mutation)
    LAYOUT  → assign positions     ← postcondition: fits constraints
    PAINT   → generate commands    ← only executes if VERIFY passed
```

The Jidoka principle (stop-the-line if any assertion fails) IS
Meyer's "exception on contract violation" — if a `BrickAssertion`
fails, the widget does not paint. This is the Eiffel `check`
instruction made architectural.

**BrickHouse composition** maps to `subcontract`: each child brick's
budget must sum to ≤ parent's total budget. This is a `conservation`
obligation on performance:

```yaml
proof_obligations:
  - type: conservation
    property: "Sum of child budgets ≤ BrickHouse total budget"
    formal: "Σ brick_budget_ms(child_i) ≤ house_budget_ms"
```

**Key insight:** presentation contracts derive from *design system
axioms* (WCAG, flexbox spec, grid spec) and *accessibility standards*,
just as kernel contracts derive from *paper equations*. The pipeline
phases map directly:

```
WCAG 2.1 / Flexbox Spec / Grid Spec  (≈ arXiv paper)
  → Layout Invariants + A11y Rules    (≈ equations)
    → Contract YAML                    (≈ contract)
      → Widget Trait + Brick           (≈ kernel trait)
        → presentar Implementation     (≈ scalar/SIMD kernel)
          → probar Property Tests      (≈ probar falsification)
            → Kani Model Checking      (≈ Kani harness)
```

#### Data Pipeline / ETL

Data pipelines have contracts rooted in schema conformance, data
quality, and transformation correctness.

| Obligation Type | Data Pipeline Example |
|---|---|
| `precondition` | Input schema matches expected version; no null primary keys |
| `postcondition` | Output row count = input row count (for map transforms) |
| `frame` | Transform modifies target columns only; source columns immutable |
| `invariant` | Foreign key references resolve at every pipeline stage |
| `old_state` | `output.row_count = old(input.row_count) - filtered_count` |
| `conservation` | Sum of monetary values preserved across currency conversion |
| `completeness` | All enum variants in source schema have mapping rules |
| `determinism` | Same input batch produces identical output regardless of parallelism |
| `roundtrip` | `unpivot(pivot(table)) = table` |
| `loop_invariant` | Streaming window: buffer size ≤ `max_window` at every step |

#### API / Service Layer

API contracts derive from protocol specifications, SLAs, and backward
compatibility guarantees.

| Obligation Type | API Example |
|---|---|
| `precondition` | Request body conforms to JSON schema v2.1; auth token valid |
| `postcondition` | Response status 200 implies body conforms to response schema |
| `frame` | GET requests modify no server state (HTTP idempotency) |
| `subcontract` | API v2 is a valid refinement of v1 — accepts all v1 requests, v2 responses are v1-compatible |
| `state_machine` | Order lifecycle: `created → paid → shipped → delivered → closed` |
| `old_state` | After PATCH, `version = old(version) + 1` |
| `bound` | Response latency p99 < 200ms |
| `idempotency` | Repeated PUT with same body produces identical state |
| `determinism` | Same request yields same response (for cacheable endpoints) |

#### Simulation (simular)

Simular is the stack's unified simulation engine covering physics
(orbital mechanics, N-body, rigid body, fluid dynamics), Monte Carlo
methods, optimization (Bayesian, GRASP), and ML training simulations.
It already consumes provable-contracts with 3 YAML contracts
(checkpoint, gradient, loss-functions). Simulation contracts derive
from *physics conservation laws* and *numerical integration theory*.

Meyer's Eiffel DbC maps naturally to simulation because simulations
are inherently stateful, iterative systems where *what changes* and
*what must not change* must be specified precisely.

| Obligation Type | Simulation Example |
|---|---|
| `precondition` | Initial state has finite energy; timestep `dt > 0` |
| `postcondition` | After N-body step, all positions are finite and within bounds |
| `invariant` | Total system energy conserved within drift tolerance |
| `frame` | Integration step modifies positions and velocities only; masses and gravitational constant unchanged |
| `loop_invariant` | Symplectic integrator preserves phase-space volume at every timestep |
| `loop_variant` | Remaining simulation steps = `total_steps - current_step`, strictly decreasing |
| `old_state` | `energy(state_new) - energy(old(state)) < ε_drift` per step |
| `conservation` | Total momentum conserved in closed N-body system |
| `determinism` | Same RNG seed + same initial state → identical trajectory |
| `roundtrip` | `deserialize(serialize(checkpoint)) = checkpoint` (checkpoint fidelity) |
| `bound` | Gradient norm ≤ `max_clip` after clipping |
| `equivalence` | Verlet integrator matches RK4 within tolerance for smooth potentials |
| `monotonicity` | Loss decreases monotonically for convex objectives with valid learning rate |
| `subcontract` | Custom integrator refines base `Integrator` contract |

**Key insight:** Simular's jidoka module (stop-on-error anomaly
detection) is fundamentally a *runtime contract checker*. It monitors
for NaN, Inf, energy drift, and constraint violations — these are
exactly the invariants and postconditions that should be declared in
YAML contracts and verified statically via Kani, not just caught at
runtime. The Eiffel DbC types make this declarative:

```
Runtime jidoka check               →  YAML contract equivalent
─────────────────────────────────────────────────────────────────
NaN/Inf detection                  →  postcondition: output is finite
Energy drift > threshold           →  old_state: |E_new - E_old| < ε
Constraint violation               →  invariant: constraint holds
State corruption                   →  frame: only specified fields change
```

#### Testing / QA (probar)

Probar is the stack's WASM and TUI testing framework — a Playwright-
compatible, zero-JavaScript, pure Rust testing tool for games,
simulations, and terminal UIs. It does not yet consume provable-
contracts, but its domain is rich with contractable properties.

Testing frameworks are a distinctive case for DbC because the
*framework itself* has contracts about what constitutes correct test
behavior. Meyer addressed this in OOSC Ch. 11 under "Who checks the
checker?" — the testing tool's own invariants must be at least as
rigorous as the code it tests.

| Obligation Type | Testing Framework Example |
|---|---|
| `precondition` | Browser/WASM runtime is connected before locator query |
| `postcondition` | After `click(element)`, element's click handler has been invoked |
| `invariant` | Pixel coverage map has dimensions matching viewport at all times |
| `frame` | Visual regression comparison modifies diff buffer only; reference image unchanged |
| `old_state` | After navigation, `history.length = old(history.length) + 1` |
| `state_machine` | Test lifecycle: `init → setup → running → teardown → complete \| failed` |
| `determinism` | Replay with same seed produces identical event sequence |
| `roundtrip` | `deserialize(serialize(playbook)) = playbook` (YAML playbook fidelity) |
| `completeness` | Every interactive element in DOM has at least one locator match |
| `idempotency` | Running same assertion twice yields same pass/fail result |
| `subcontract` | `TuiLocator` refines `Locator` — accepts same selectors, returns TUI-specific elements |
| `bound` | Screenshot capture completes within 500ms |
| `conservation` | Total pixel count in coverage heatmap = viewport width × height |
| `loop_invariant` | During auto-wait polling: timeout budget remaining ≥ 0 |
| `loop_variant` | Retry attempts remaining = `max_retries - attempt`, strictly decreasing |

**Key insight:** Probar's "Brick Architecture" pattern (tests ARE the
interface) is a direct expression of Meyer's seamless development:
the test specification IS the component contract. A brick's type-safe
selectors declare preconditions (element must exist), its assertions
declare postconditions (visual state after interaction), and its
composition rules (`BrickHouse` budgets) declare frame conditions
(total test time must not exceed budget). The gap is that these
contracts live in Rust code, not in declarative YAML contracts that
can be validated, scored, and verified independently.

#### Infrastructure as Code (forjar)

Forjar is the stack's sovereign IaC tool — a single-binary Rust
replacement for Terraform/Ansible that manages bare-metal machines and
containers over SSH using YAML configs, BLAKE3 content-addressed state,
and deterministic DAG execution. It already consumes provable-contracts
with `#[contract]` annotations on its core algorithms (DAG ordering,
atomic writes, recipe determinism, codegen dispatch).

Infrastructure provisioning is a domain where Meyer's DbC is
*especially* natural because the entire paradigm is already
contractual: a desired-state config is a *specification* (contract),
the planner computes a *diff* (obligation), and the executor
*converges* the system (fulfillment). Forjar's existing design
patterns map directly to Eiffel DbC concepts:

| Forjar Concept | Eiffel DbC Equivalent |
|---|---|
| Resource `when:` guards | `precondition` |
| Planner action (Create/Update/NoOp) | `postcondition` (desired state reached) |
| BLAKE3 state snapshot | `old_state` (hash comparison pre/post) |
| Jidoka failure isolation | `frame` (partial failure preserves unchanged resources) |
| DAG wave execution | `loop_invariant` (DAG ordering respected at every wave) |
| Idempotency contract (FJ-210) | `idempotency` + `postcondition` |
| Bashrs transport safety | `precondition` (script passes shell validation) |

| Obligation Type | Infrastructure as Code Example |
|---|---|
| `precondition` | Target machine reachable via SSH; bashrs validates generated script before execution |
| `postcondition` | After apply, resource hash in lock file matches desired-state hash |
| `invariant` | DAG ordering: no resource executes before its dependencies |
| `frame` | Applying resource R modifies only R's state; all other resources' lock entries preserved |
| `old_state` | `lock_hash(resource, after) ≠ lock_hash(resource, old(before))` implies action was Update |
| `idempotency` | Second apply on converged state produces zero changes (FJ-210) |
| `determinism` | Same config YAML + same lock state → identical execution plan |
| `loop_invariant` | During DAG wave execution: all completed resources have converged state |
| `loop_variant` | Remaining unprocessed resources in DAG = `total - completed`, strictly decreasing |
| `roundtrip` | `deserialize(serialize(lock_state)) = lock_state` (BLAKE3 lock fidelity) |
| `conservation` | Total resource count in plan = creates + updates + destroys + no-ops |
| `subcontract` | Pepita transport refines SSH transport — accepts same scripts, adds namespace isolation |
| `state_machine` | Resource lifecycle: `absent → creating → converged → drifted → updating → converged → destroying → absent` |
| `completeness` | Every resource in config has exactly one handler in the resource registry |
| `bound` | SSH retry backoff ≤ max 4 attempts; copia delta block size ≤ 1MB |

**Key insight:** Forjar's planner is already a contract evaluator. It
compares desired state (the "postcondition") against current state (the
"old_state") and computes the minimal set of actions to satisfy the
contract. The Eiffel DbC types make this structure *explicit in the
contract YAML* rather than implicit in Rust code. In particular:

- **`frame` is critical for IaC.** When forjar applies a package
  install, it must not disturb file resources, service resources, or
  other machines. This is the same "only modifies what it claims to
  modify" guarantee that Meyer's `only` clause provides. Forjar's
  jidoka policy (stop on first failure, preserve partial state) is a
  runtime enforcement of the frame condition.

- **`old_state` is the natural language of drift detection.** Forjar's
  tripwire module compares `hash(current)` against `hash(old(lock))`.
  This is exactly `Q(old(state), new(state))` — the DbC old-state
  obligation. Declaring this in the contract makes the drift detection
  contract auditable and testable via Kani.

- **`subcontract` captures transport substitutability.** Forjar's
  transport abstraction (local, SSH, container, pepita) is a textbook
  case of behavioral subtyping: each transport accepts the same script
  input (weakened precondition: pepita adds isolation but doesn't
  require it), and guarantees the same execution semantics (strengthened
  postcondition: pepita adds namespace isolation on top of base
  guarantees).


*Continued in [eiffel-dbc-domains-2.md](eiffel-dbc-domains-2.md)*
