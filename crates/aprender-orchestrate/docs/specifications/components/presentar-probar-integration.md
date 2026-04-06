# Presentar + Probar Integration Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> See also: [agent-and-playbook.md](agent-and-playbook.md), [multi-provider-api.md](multi-provider-api.md)
> Depends on: presentar 0.3.x, presentar-terminal 0.3.x, jugar-probar 1.0.x, probador CLI

---

## 1. Overview

This specification defines how **presentar** (WASM-first visualization and TUI framework) and **probar** (Playwright-compatible testing with pixel coverage) integrate into batuta's agent runtime, multi-provider API, and playbook system. The integration spans three layers:

1. **Rendering**: Agent TUI via presentar-terminal (streaming output, cost dashboard, tool status panels)
2. **Testing**: Agent behavior verification via probar (state machine playbooks, pixel coverage, Brick falsification)
3. **Contracts**: UX invariants enforced via provable-contracts + probar Brick architecture (tests ARE the interface)

### Motivation

The agent runtime (agent-and-playbook.md) and multi-provider API (multi-provider-api.md) define behavior and correctness contracts but lack:
- A concrete rendering layer for agent TUI output
- Visual regression testing for agent UI states
- State machine validation tooling for the agent loop
- Budgeted composition for tool output rendering (Jidoka enforcement)

Presentar and probar fill these gaps with stack-native, zero-external-dependency solutions.

### Non-Goals

- Not replacing batuta's existing CLI (clap-based); extending it with TUI panels
- Not building a web frontend for the agent (WASM target is for demos/dashboards, not the agent itself)
- Not duplicating probar's existing capabilities; wiring them to agent-specific contracts

---

## 2. Architecture

```
batuta agent run
     |
     v
+--------------------------------------------+
|           Agent Runtime (src/agent/)        |
|  perceive -> reason -> act -> remember      |
+-----+-------------------+------------------+
      |                   |
      v                   v
+-------------+    +------------------+
| LLM Driver  |    | Tool Executor    |
| (streaming) |    | (parallel)       |
+------+------+    +--------+---------+
       |                    |
       v                    v
+------+--------------------+---------+
|     presentar-terminal TUI Layer    |
|  +----------+ +----------+ +------+ |
|  | Streaming| | Tool     | | Cost | |
|  | Output   | | Status   | | Dash | |
|  | Panel    | | Panel    | | Panel| |
|  +----------+ +----------+ +------+ |
+-------------------------------------+
       |
       v (verified by)
+------+-----------------------------+
|         probar Test Layer          |
| +--------+ +--------+ +--------+  |
| | State  | | Pixel  | | Brick  |  |
| | Machine| | Cover  | | Budget |  |
| | Playbook| | Heatmap| | Jidoka|  |
| +--------+ +--------+ +--------+  |
+------------------------------------+
```

### Module Layout

```
batuta/
  +-- agent/
  |     +-- tui/                    # (NEW) Agent TUI layer
  |     |     +-- mod.rs            # AgentTui orchestrator
  |     |     +-- streaming.rs      # StreamingOutputPanel (SSE token display)
  |     |     +-- tools.rs          # ToolStatusPanel (parallel tool progress)
  |     |     +-- cost.rs           # CostDashboardPanel (per-turn + cumulative)
  |     |     +-- session.rs        # SessionPanel (history, resume, fork)
  |     |     +-- sandbox.rs        # SandboxPanel (policy display, violation log)
  |     |     +-- layout.rs         # Adaptive layout (responsive to terminal size)
  |     |     +-- theme.rs          # Theme integration (Tokyo Night, Dracula, etc.)
  |     +-- brick/                  # (NEW) Brick-based UX contracts
  |     |     +-- mod.rs            # AgentBrick trait extensions
  |     |     +-- streaming_brick.rs
  |     |     +-- tool_brick.rs
  |     |     +-- cost_brick.rs
  |     +-- test/                   # (NEW) Probar test harnesses
  |           +-- playbook.rs       # Agent loop state machine playbook
  |           +-- pixel.rs          # Pixel coverage for agent TUI
  |           +-- regression.rs     # Visual regression baselines
```

---

## 3. Presentar-Terminal Agent TUI

### 3.1 Panel Architecture

The agent TUI uses presentar-terminal's direct crossterm backend (zero ratatui dependency) with adaptive layout. Six panels compose the agent display:

| Panel | Position | Content | Update Frequency |
|-------|----------|---------|-----------------|
| **StreamingOutput** | Top-left (60%) | Token-by-token LLM output, markdown rendering | Per SSE event (~50ms) |
| **ToolStatus** | Top-right (40%) | Active tools, progress bars, parallel execution status | Per tool event |
| **CostDashboard** | Bottom-left (33%) | Per-turn cost, cumulative session cost, provider breakdown | Per turn |
| **SessionInfo** | Bottom-center (33%) | Session ID, model, provider, context usage, compaction status | Per turn |
| **SandboxLog** | Bottom-right (33%) | Recent sandbox events, blocked actions, policy violations | Per tool event |
| **StatusBar** | Bottom row | Agent state (Idle/Perceive/Reason/Act), iteration count, LoopGuard status | Continuous |

### 3.2 Adaptive Layout

```
Terminal >= 120x40 (Full):
+---------------------------+------------------+
|   Streaming Output (60%)  |  Tool Status     |
|   Token-by-token display  |  Progress bars   |
|   Markdown rendering      |  Parallel status |
+---------------------------+------------------+
| Cost Dashboard | Session  | Sandbox Log      |
| Per-turn + cum | Model/ID | Blocked actions  |
+----------------+----------+------------------+
| [Reason] iter 5/50 | ctx 45% | $0.02 session  |
+----------------------------------------------+

Terminal 80x24 (Compact):
+---------------------------+------------------+
|   Streaming Output (70%)  | Tool Status (30%)|
+---------------------------+------------------+
| Cost: $0.02 | ctx 45% | iter 5/50 | Reason  |
+----------------------------------------------+

Terminal < 80x24 (Minimal):
+----------------------------------------------+
| Streaming Output (100%)                       |
+----------------------------------------------+
| $0.02 | 45% ctx | 5/50 | Reason             |
+----------------------------------------------+
```

Five detail levels from presentar-terminal: Minimal, Compact, Normal, Expanded, Exploded.

### 3.3 Streaming Output Panel

Renders SSE `StreamEvent` tokens in real-time using presentar-terminal's `CellBuffer` with `DiffRenderer` for minimal terminal writes:

```rust
use presentar_terminal::direct::{CellBuffer, DiffRenderer};
use presentar_terminal::widgets::TextBlock;

pub struct StreamingOutputPanel {
    buffer: CellBuffer,
    renderer: DiffRenderer,
    /// Ring buffer of recent tokens for scrollback
    tokens: VecDeque<String>,
    /// Markdown state machine for formatting
    md_state: MarkdownState,
}

impl StreamingOutputPanel {
    /// Append a streaming token and re-render the affected region
    pub fn push_token(&mut self, token: &str) {
        self.tokens.push_back(token.to_string());
        self.md_state.feed(token);
        // Only re-render dirty cells via DiffRenderer
        self.renderer.render_diff(&self.buffer);
    }
}
```

Performance targets (from presentar-terminal benchmarks):
- Full redraw: <1ms (80x24)
- Partial update (token append): <0.1ms
- Zero heap allocations in steady state

### 3.4 Tool Status Panel

Displays parallel tool execution with progress indicators:

```
+-- Tool Status ──────────────+
| [*] file_read src/main.rs   |
|     [========>   ] 80%      |
| [*] grep "error" src/       |
|     [=====>      ] 50%      |
| [v] shell: cargo test       |
|     Completed (2.3s, exit 0)|
| [-] file_write blocked      |
|     Sandbox: /etc/passwd     |
+-----------------------------+
```

Uses presentar-terminal's `Gauge` widget for progress bars and `Border` for framing.

### 3.5 Cost Dashboard Panel

```
+-- Cost ─────────────────────+
| This turn:   $0.003         |
| Session:     $0.021 / $5.00 |
| ████████░░░░░  0.4%         |
| Provider: anthropic (3/3)   |
| Model: claude-sonnet-4      |
| Tokens: 1.2K in / 340 out  |
+-----------------------------+
```

Uses presentar-terminal's `Gauge`, `Sparkline` (cost trend), and CIELAB color interpolation for budget gradient (green → yellow → red).

### 3.6 Theme Integration

The agent TUI inherits presentar-terminal's theme system:

| Theme | Source | Best For |
|-------|--------|----------|
| Tokyo Night | presentar-terminal | Dark terminals, low contrast |
| Dracula | presentar-terminal | High contrast dark |
| Nord | presentar-terminal | Muted blue tones |
| Monokai | presentar-terminal | Familiar to developers |
| Custom | agent.toml `[tui.theme]` | User-defined |

Color mode auto-detection: TrueColor → 256-color → 16-color → Mono.

---

## 4. Probar State Machine Testing

### 4.1 Agent Loop Playbook

The agent's perceive-reason-act loop is a finite state machine. Probar's playbook system validates it via YAML-driven state machine specifications with mutation testing.

```yaml
# tests/playbooks/agent-loop.yaml
version: "1.0"
machine:
  id: "agent_loop"
  initial: "idle"

  states:
    idle:
      invariants:
        - description: "No active LLM call"
          condition: "driver.is_idle()"
        - description: "Context within window"
          condition: "context.token_count() <= context.window()"

    perceive:
      invariants:
        - description: "Memory recall initiated"
          condition: "memory.recall_pending() || memory.recall_complete()"

    reason:
      invariants:
        - description: "LLM call active or complete"
          condition: "driver.is_active() || driver.has_response()"
        - description: "Context not exceeded"
          condition: "context.token_count() <= context.window()"

    act:
      invariants:
        - description: "At least one tool call pending"
          condition: "tools.pending_count() > 0"
        - description: "All tool calls have capability"
          condition: "tools.all_authorized()"

    remember:
      invariants:
        - description: "Agent has final response"
          condition: "response.is_some()"

    done:
      final_state: true
      invariants:
        - description: "All tools completed"
          condition: "tools.pending_count() == 0"

    failed:
      final_state: true
      invariants:
        - description: "Guard triggered with reason"
          condition: "guard.failure_reason().is_some()"

  transitions:
    - from: "idle"
      to: "perceive"
      event: "user_message"
      actions:
        - type: call
          function: "agent.perceive(message)"

    - from: "perceive"
      to: "reason"
      event: "memory_recalled"
      actions:
        - type: call
          function: "agent.reason(context)"

    - from: "reason"
      to: "act"
      event: "tool_use"
      guard: "response.has_tool_calls()"
      actions:
        - type: call
          function: "agent.execute_tools(calls)"

    - from: "reason"
      to: "remember"
      event: "end_turn"
      guard: "!response.has_tool_calls()"

    - from: "act"
      to: "reason"
      event: "tool_result"
      actions:
        - type: call
          function: "agent.reason_with_results(results)"

    - from: "remember"
      to: "done"
      event: "success"

    - from: "*"
      to: "failed"
      event: "guard_triggered"
      guard: "guard.should_stop()"

  forbidden_transitions:
    - from: "idle"
      to: "act"
      reason: "Cannot execute tools without reasoning first"
    - from: "done"
      to: "*"
      reason: "Done is terminal (must create new session)"
    - from: "failed"
      to: "*"
      reason: "Failed is terminal"

  complexity:
    metric: "iterations"
    expected: "O(n)"
    max_n: 50
```

### 4.2 Provider Failover Playbook

```yaml
# tests/playbooks/provider-failover.yaml
version: "1.0"
machine:
  id: "provider_failover"
  initial: "primary"

  states:
    primary:
      invariants:
        - description: "Using highest-priority provider"
          condition: "provider.is_primary()"
    
    failover:
      invariants:
        - description: "Primary failed, using fallback"
          condition: "provider.failures() >= threshold"

    exhausted:
      final_state: true
      invariants:
        - description: "All providers tried"
          condition: "provider.all_exhausted()"

  transitions:
    - from: "primary"
      to: "failover"
      event: "provider_failure"
      guard: "failures >= failure_threshold"
      actions:
        - type: call
          function: "router.next_provider()"

    - from: "failover"
      to: "failover"
      event: "provider_failure"
      guard: "router.has_more_providers()"

    - from: "failover"
      to: "exhausted"
      event: "provider_failure"
      guard: "!router.has_more_providers()"

    - from: "failover"
      to: "primary"
      event: "provider_recovery"
      guard: "primary.health_check_passed()"
```

### 4.3 Mutation Testing (Probar M1-M5)

Probar's playbook mutation testing validates that our state machine tests are meaningful:

| Mutation | What It Does | Expected Result |
|----------|-------------|-----------------|
| **M1: Transition removal** | Remove idle→perceive | Tests must fail (agent can't start) |
| **M2: Invariant modification** | Change `token_count <= window` to `token_count <= window * 2` | Context overflow tests must catch |
| **M3: Guard condition flip** | Flip `has_tool_calls()` to `!has_tool_calls()` | Agent routes to wrong state |
| **M4: Action removal** | Remove `execute_tools()` call | Tool execution tests must fail |
| **M5: Assertion weakening** | Remove `all_authorized()` check | Sandbox tests must catch |

```bash
# Run mutation testing on agent playbook
probador playbook tests/playbooks/agent-loop.yaml --mutate

# Expected output:
# M1: 5/5 killed (100%)
# M2: 4/4 killed (100%)
# M3: 3/3 killed (100%)
# M4: 2/2 killed (100%)
# M5: 2/2 killed (100%)
# Mutation score: 100% (16/16 killed)
```

---

## 5. Probar Pixel Coverage for Agent TUI

### 5.1 Coverage Tracking

Every agent TUI panel is tracked for pixel coverage using probar's `PixelCoverageTracker`. This ensures that test suites exercise all visual states.

```rust
use jugar_probar::pixel_coverage::{PixelCoverageTracker, PixelRegion};
use jugar_probar::pixel_coverage::thresholds;

#[test]
fn agent_tui_pixel_coverage() {
    let mut tracker = PixelCoverageTracker::new(120, 40, 12, 4);  // 120x40 terminal, 12x4 grid

    // Exercise streaming panel
    tracker.record_region(PixelRegion::new(0, 0, 72, 20));  // Top-left 60%

    // Exercise tool status panel
    tracker.record_region(PixelRegion::new(72, 0, 48, 20)); // Top-right 40%

    // Exercise cost dashboard
    tracker.record_region(PixelRegion::new(0, 20, 40, 19)); // Bottom-left 33%

    // Exercise session info
    tracker.record_region(PixelRegion::new(40, 20, 40, 19));

    // Exercise sandbox log
    tracker.record_region(PixelRegion::new(80, 20, 40, 19));

    // Exercise status bar
    tracker.record_region(PixelRegion::new(0, 39, 120, 1));

    let report = tracker.generate_report();
    assert!(report.overall_coverage >= thresholds::STANDARD);  // 80%
}
```

### 5.2 Visual Regression Baselines

Agent TUI states captured as presentar-terminal `TuiSnapshot` baselines:

| Baseline | State | File |
|----------|-------|------|
| `agent_idle.snap` | Agent waiting for input | No active panels |
| `agent_streaming.snap` | Token-by-token output | Streaming panel active |
| `agent_tools.snap` | Parallel tool execution | 3 tools in progress |
| `agent_cost_warn.snap` | Cost approaching budget | Yellow/red cost gauge |
| `agent_sandbox_block.snap` | Sandbox violation | Red border, blocked tool |
| `agent_compact.snap` | Minimal terminal (80x24) | Compressed layout |

```rust
use jugar_probar::visual_regression::VisualRegressionTest;
use presentar_terminal::test::TuiSnapshot;

#[test]
fn agent_streaming_visual_regression() {
    let snapshot = TuiSnapshot::capture(&agent_tui);
    let baseline = TuiSnapshot::load("baselines/agent_streaming.snap");

    let diff = VisualRegressionTest::compare(&baseline, &snapshot);
    assert!(diff.psnr() > 40.0, "Visual regression detected: PSNR={}", diff.psnr());
    assert!(diff.ssim() > 0.99, "Structural change: SSIM={}", diff.ssim());
}
```

### 5.3 Heatmap Reports

```bash
# Generate pixel coverage heatmap for agent TUI tests
probador coverage --pixel-heatmap --output agent_tui_heatmap.png

# Terminal heatmap (inline)
probador coverage --pixel-heatmap --terminal
```

Output shows which TUI regions have low test coverage (cold spots in blue, hot spots in red).

---

## 6. Brick Architecture for Agent UX

### 6.1 Agent Bricks

Each agent TUI panel is a **Brick** — its assertions, performance budget, and verification are defined before implementation.

```rust
use presentar_core::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

pub struct StreamingOutputBrick {
    pub tokens: Vec<String>,
    pub is_streaming: bool,
}

impl Brick for StreamingOutputBrick {
    fn brick_name(&self) -> &'static str { "StreamingOutput" }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),      // WCAG AA
            BrickAssertion::MaxLatencyMs(100),        // <100ms per token render
            BrickAssertion::Custom("token_ordering"),  // Tokens appear in order
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)  // 16ms = 60fps
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        if self.is_streaming && self.tokens.is_empty() {
            v.fail("Streaming active but no tokens rendered");
        }
        v
    }

    fn can_render(&self) -> bool {
        self.verify().passed()  // Jidoka: block rendering if verification fails
    }
}
```

### 6.2 BrickHouse: Agent Panel Composition

All agent panels composed into a budgeted `BrickHouse` with Jidoka enforcement:

```rust
use presentar_core::brick::BrickHouseBuilder;

let house = BrickHouseBuilder::new("agent-tui")
    .budget_ms(16)  // 16ms total frame budget (60fps)
    .brick(streaming_brick, 8)      // 8ms for streaming output
    .brick(tool_status_brick, 3)    // 3ms for tool status
    .brick(cost_brick, 2)           // 2ms for cost dashboard
    .brick(session_brick, 1)        // 1ms for session info
    .brick(sandbox_brick, 1)        // 1ms for sandbox log
    .brick(status_bar_brick, 1)     // 1ms for status bar
    .build()?;

// Jidoka: if any brick exceeds budget, entire frame fails
let report = house.verify_all();
assert!(report.utilization() <= 1.0, "Frame budget exceeded: {}%", report.utilization() * 100.0);
```

### 6.3 Brick Assertion Matrix

| Brick | TextVisible | ContrastRatio | MaxLatencyMs | Custom |
|-------|------------|---------------|-------------|--------|
| StreamingOutput | Yes | 4.5 (AA) | 100 | token_ordering |
| ToolStatus | Yes | 4.5 (AA) | 50 | progress_monotonic |
| CostDashboard | Yes | 4.5 (AA) | 50 | cost_non_negative |
| SessionInfo | Yes | 4.5 (AA) | 50 | context_percentage_valid |
| SandboxLog | Yes | 7.0 (AAA for warnings) | 50 | violation_has_reason |
| StatusBar | Yes | 4.5 (AA) | 16 | state_is_valid_enum |

---

## 7. Probar Falsification for Agent UX

### 7.1 Falsification Test Suite

```rust
use jugar_probar::pixel_coverage::falsification::{FalsificationGate, Hypothesis};

#[test]
fn falsify_agent_streaming_responsiveness() {
    let gate = FalsificationGate::new(Hypothesis {
        id: "H-STREAM-001",
        claim: "First token renders within 2s of user input",
        test: || {
            let start = Instant::now();
            agent.send_message("Hello");
            agent.wait_for_first_token();
            start.elapsed() < Duration::from_secs(2)
        },
        if_fails: "SSE streaming not connected or provider TTFT too high",
    });

    gate.run(100);  // 100 trials
    assert!(gate.rejection_rate() < 0.01, "H-STREAM-001 rejected: {}% failure rate", gate.rejection_rate() * 100.0);
}

#[test]
fn falsify_cost_dashboard_accuracy() {
    let gate = FalsificationGate::new(Hypothesis {
        id: "H-COST-001",
        claim: "Displayed cost matches actual within 5%",
        test: || {
            let displayed = cost_panel.displayed_cost();
            let actual = session.actual_cost();
            (displayed - actual).abs() / actual < 0.05
        },
        if_fails: "Cost estimation diverges from actual — user sees wrong numbers",
    });

    gate.run(1000);
    assert!(gate.rejection_rate() < 0.001);
}
```

### 7.2 UX Falsification Matrix

| Hypothesis | Probar Test | Threshold | What Failure Means |
|-----------|------------|-----------|-------------------|
| H-STREAM-001: TTFT < 2s | FalsificationGate, 100 trials | <1% rejection | Streaming broken or provider too slow |
| H-COST-001: Cost accuracy < 5% error | FalsificationGate, 1000 trials | <0.1% rejection | Users see wrong cost numbers |
| H-TOOL-001: Tool progress monotonic | Brick assertion, per-frame | 0% violation | Progress bar goes backwards |
| H-SANDBOX-001: Violations always shown | Pixel coverage, region check | 100% coverage of sandbox panel | Silent failures — user unaware of blocked actions |
| H-LAYOUT-001: No panel overlap at any terminal size | Probar pixel coverage, 20x10 to 200x60 | 0 overlap pixels | Layout engine bug — panels corrupt each other |
| H-A11Y-001: WCAG AA contrast on all panels | Brick ContrastRatio(4.5) | 100% pass | Accessibility violation |
| H-BUDGET-001: Frame time < 16ms | BrickHouse budget, 10K frames | <1% violation | TUI too slow for 60fps — user sees lag |

---

## 8. Probar Integration with Existing Specs

### 8.1 Agent Loop Testing (from agent-and-playbook.md)

The 9 falsification tests in `agent-loop-v1.yaml` are implemented as probar playbook + pixel coverage tests:

| Contract Test | Probar Implementation |
|--------------|----------------------|
| FALSIFY-AL-001 (loop termination) | `probador playbook agent-loop.yaml` — M4 mutation kills non-terminating loops |
| FALSIFY-AL-002 (state machine) | `probador playbook agent-loop.yaml --validate` — forbidden transition check |
| FALSIFY-AL-003 (compaction safety) | Pixel coverage: verify system prompt panel unchanged after compact |
| FALSIFY-AL-004 (sandbox) | Probar `FalsificationGate` with shell escape attempts |
| FALSIFY-AL-005 (parallel safety) | Probar deterministic replay with concurrent tool calls |
| FALSIFY-AL-006 (crash recovery) | Probar session snapshot + truncate + resume + visual regression |
| FALSIFY-AL-007 (hook blocking) | Probar `FalsificationGate` with destructive command patterns |
| FALSIFY-AL-008 (ping-pong) | Probar playbook complexity check: O(n) not O(n^2) |
| FALSIFY-AL-009 (memory refresh) | Probar state capture before/after compact, verify memory panel updated |

### 8.2 Provider Routing Testing (from multi-provider-api.md)

| Contract Test | Probar Implementation |
|--------------|----------------------|
| FALSIFY-MPA-001 (privacy) | Probar network interception: assert zero external requests under Sovereign |
| FALSIFY-MPA-002 (failover) | Probar playbook `provider-failover.yaml` with mock providers |
| FALSIFY-MPA-003 (cost budget) | Probar `FalsificationGate` on cost panel display vs actual |
| FALSIFY-MPA-005 (translation) | Probar proptest: round-trip arbitrary messages |
| FALSIFY-MPA-006 (SSE) | Probar streaming UX validator: verify stream completeness |
| FALSIFY-MPA-007 (chaos) | Probar chaos injection via mock provider with random 429/500/timeout |

---

## 9. WASM Agent Dashboard (Future)

For browser-based monitoring of remote agents, presentar's WASM target provides a zero-JS dashboard:

```yaml
# agent-dashboard.yaml (presentar manifest)
app:
  name: "Batuta Agent Monitor"
  theme: "tokyo-night"

widgets:
  root:
    type: Column
    children:
      - type: Text
        value: "{{ agent.status }}"
        style: { font_size: 24 }

      - type: Chart
        chart_type: line
        data: "{{ agent.cost_history | limit(50) }}"
        x_label: "Turn"
        y_label: "Cost ($)"

      - type: DataTable
        columns: ["Tool", "Status", "Duration"]
        data: "{{ agent.recent_tools | select('name', 'status', 'duration') }}"

      - type: Gauge
        value: "{{ agent.context_usage | percentage }}"
        label: "Context Window"
        thresholds: [60, 80, 95]
```

Bundle size: <500KB WASM. Expression language executes client-side (zero server round-trips).

---

## 10. CLI Commands

```bash
# Agent TUI mode (presentar-terminal)
batuta agent run --tui --manifest agent.toml
batuta agent run --tui --theme dracula

# Agent TUI testing
probador playbook tests/playbooks/agent-loop.yaml
probador playbook tests/playbooks/agent-loop.yaml --mutate
probador playbook tests/playbooks/agent-loop.yaml --export svg -o agent-fsm.svg
probador playbook tests/playbooks/provider-failover.yaml --validate

# Pixel coverage
probador coverage --pixel-heatmap --output agent_tui_heatmap.png
probador coverage --pixel-heatmap --terminal

# Visual regression
probador test tests/tui/ --update-baselines
probador test tests/tui/ --visual-regression

# Brick budget report
batuta agent brick-report
```

---

## 11. Dependencies

### 11.1 Cargo.toml Additions (batuta)

```toml
[dependencies]
# TUI rendering (agent --tui mode)
presentar-terminal = { version = "0.3", optional = true }
presentar-core = { version = "0.3", optional = true }

[dev-dependencies]
# Agent TUI testing
jugar-probar = { version = "1.0", features = ["tui", "compute-blocks", "proptest"] }

[features]
agent-tui = ["dep:presentar-terminal", "dep:presentar-core"]
```

### 11.2 Feature Flag Matrix

| Feature | What It Enables | Default |
|---------|----------------|---------|
| `agent-tui` | Presentar-terminal TUI for agent runtime | No |
| `native` (existing) | Full CLI including `agent-tui` when combined | Yes |

---

## 12. Design Principles

| Toyota Principle | Presentar/Probar Application |
|-----------------|------------------------------|
| **Jidoka** | BrickHouse stops rendering if any Brick exceeds budget or fails verification |
| **Poka-Yoke** | Brick assertions prevent rendering invalid states (can_render() gate) |
| **Muda** | DiffRenderer only updates changed cells; zero-alloc steady state |
| **Heijunka** | Adaptive layout levels load across terminal sizes |
| **Genchi Genbutsu** | Pixel coverage heatmaps show exactly which UI regions are untested |
| **Kaizen** | Visual regression baselines track UI quality over time |
| **Mieruka** | Heatmap visualization makes coverage gaps immediately visible |

---

## 13. Implementation Phases

| Phase | Scope | Dependencies |
|-------|-------|-------------|
| **1** | StreamingOutputPanel + StatusBar (minimal TUI) | presentar-terminal, agent streaming |
| **2** | ToolStatusPanel + CostDashboardPanel | Phase 1 + cost tracking |
| **3** | Agent loop playbook + mutation testing | probar playbooks |
| **4** | Pixel coverage + visual regression baselines | probar pixel_coverage |
| **5** | Full BrickHouse composition + Jidoka enforcement | presentar-core Brick trait |
| **6** | WASM dashboard (future) | presentar WASM target |

---

## 14. Prior Art

| Project | Relevance |
|---------|-----------|
| **presentar-terminal ptop** (14 panels) | Reference implementation for multi-panel TUI; same CellBuffer/DiffRenderer |
| **ttop 2.0** | Pixel-perfect parity target for system monitoring panels |
| **block/goose TUI** | Agent TUI with streaming + tool status; validates the panel pattern |
| **antinomyhq/forge TUI** | Rust AI agent with rich TUI; proves presentar-terminal approach viable |
| **probar playbook system** | State machine testing with M1-M5 mutations; directly reused |
| **probar pixel coverage** | GUI coverage tracking; directly reused |
| **probar Brick architecture** | Tests-are-interface pattern; directly reused |

---

## 15. Testing Strategy

| Test Type | Tool | Coverage |
|-----------|------|----------|
| Unit: panel rendering | presentar-terminal `TuiTestBackend` | Each panel renders correctly in isolation |
| Unit: Brick assertions | presentar-core `Brick::verify()` | All assertions pass for valid states, fail for invalid |
| Integration: full TUI | probar TUI testing | All panels compose without overlap |
| State machine: agent loop | probar playbook `agent-loop.yaml` | All transitions valid, forbidden transitions blocked |
| State machine: failover | probar playbook `provider-failover.yaml` | Failover cascade correct |
| Mutation: playbooks | probar `--mutate` | 100% mutation kill rate |
| Pixel coverage: TUI | probar `PixelCoverageTracker` | >=80% of terminal cells exercised |
| Visual regression: TUI | probar `VisualRegressionTest` | PSNR >40, SSIM >0.99 vs baselines |
| Falsification: UX | probar `FalsificationGate` | All 7 UX hypotheses survive 100+ trials |
| Performance: frame budget | BrickHouse budget report | <16ms per frame (60fps) |
| Accessibility: WCAG AA | probar `A11yChecker` + Brick `ContrastRatio(4.5)` | All panels pass AA |
