# apr code TUI: Probar-First Testing Specification

> Parent: [apr-code.md](apr-code.md)
> Contracts: `tui-rendering-v1.yaml`, `tui-panels-v1.yaml`, `agent-ux-v1.yaml`
> Testing framework: probar (jugar-probar 1.0.x, probador CLI)
> Rendering: presentar-terminal 0.3.x

---

## 1. Probar-First Principle

**Every TUI panel is tested BEFORE it is implemented.** This is not TDD in the conventional sense — it is the presentar Brick architecture's core mandate: **tests define the interface, implementation follows.**

Workflow for each panel:

```
1. Write probar playbook (state machine YAML)
2. Write Brick assertions (what the panel promises)
3. Write pixel coverage test (which cells are exercised)
4. Write visual regression baseline (what it looks like)
5. Write falsification gate (what would break it)
6. THEN implement the panel
7. Run probador — all tests must pass before merge
```

This ordering is enforced by:
- `build.rs` panics if test files are missing for declared panels
- `#[requires_interface(PanelName)]` macro (probar-derive) produces compile error without test
- Pre-commit hook runs `probador playbook --validate` on all TUI playbooks

---

## 2. Test Harness Architecture

```
tests/
  +-- tui/
  |     +-- harness.rs              # Shared TuiTestBackend setup
  |     +-- panels/
  |     |     +-- streaming.rs      # StreamingOutputPanel tests
  |     |     +-- tools.rs          # ToolStatusPanel tests
  |     |     +-- cost.rs           # CostDashboardPanel tests
  |     |     +-- session.rs        # SessionPanel tests
  |     |     +-- sandbox.rs        # SandboxLogPanel tests
  |     |     +-- statusbar.rs      # StatusBar tests
  |     +-- composition/
  |     |     +-- layout.rs         # 6-panel layout tests
  |     |     +-- degradation.rs    # Adaptive detail level tests
  |     |     +-- resize.rs         # Terminal resize tests
  |     |     +-- brick_budget.rs   # BrickHouse frame budget tests
  |     +-- regression/
  |     |     +-- baselines/        # Visual regression snapshots
  |     |     |     +-- idle_120x40.snap
  |     |     |     +-- streaming_120x40.snap
  |     |     |     +-- tools_3_active_120x40.snap
  |     |     |     +-- cost_warning_120x40.snap
  |     |     |     +-- sandbox_block_120x40.snap
  |     |     |     +-- minimal_80x24.snap
  |     |     |     +-- minimal_20x10.snap
  |     |     +-- regression.rs     # Snapshot comparison tests
  |     +-- playbooks/
  |     |     +-- agent-loop.yaml       # Agent FSM playbook
  |     |     +-- provider-failover.yaml # Failover FSM playbook
  |     |     +-- tui-state.yaml        # TUI state transition playbook
  |     +-- pixel/
  |     |     +-- coverage.rs       # Pixel coverage tracking
  |     |     +-- heatmap.rs        # Heatmap generation
  |     +-- accessibility/
  |     |     +-- contrast.rs       # WCAG AA contrast tests
  |     |     +-- focus.rs          # Keyboard focus traversal
  |     +-- falsification/
  |           +-- streaming.rs      # H-STREAM-* gates
  |           +-- cost.rs           # H-COST-* gates
  |           +-- budget.rs         # H-BUDGET-* gates
  |           +-- layout.rs         # H-LAYOUT-* gates
  +-- playbooks/
        +-- agent-loop.yaml         # (symlink to tui/playbooks/)
        +-- provider-failover.yaml
```

---

## 3. Panel Test Specifications

### 3.1 StreamingOutputPanel

```rust
use jugar_probar::prelude::*;
use presentar_terminal::test::{TuiTestBackend, TuiSnapshot};

#[presentar_test]
#[requires_interface(StreamingOutputBrick)]
fn streaming_panel_token_ordering() {
    let mut backend = TuiTestBackend::new(120, 40);
    let mut panel = StreamingOutputPanel::new(72, 20);  // 60% width, top half

    // Feed tokens in order
    for i in 0..100 {
        panel.push_token(&format!("token{} ", i));
    }
    panel.render(&mut backend);

    let frame = backend.frame();
    let text = frame.extract_text(0, 0, 72, 20);
    // Verify tokens appear in order
    for i in 0..99 {
        let pos_i = text.find(&format!("token{}", i)).unwrap();
        let pos_j = text.find(&format!("token{}", i + 1)).unwrap();
        assert!(pos_i < pos_j, "Token {} appeared after token {}", i, i + 1);
    }
}

#[presentar_test]
fn streaming_panel_markdown_rendering() {
    let mut backend = TuiTestBackend::new(80, 24);
    let mut panel = StreamingOutputPanel::new(80, 20);

    panel.push_token("**bold** ");
    panel.push_token("and `code`");
    panel.render(&mut backend);

    let frame = backend.frame();
    // Bold text should have Bold attribute
    assert!(frame.cell_at(0, 0).style.has_bold());
    // Code should have distinct background or style
    assert!(frame.cell_at(9, 0).style != frame.cell_at(0, 0).style);
}

#[presentar_test]
fn streaming_panel_scrollback() {
    let mut backend = TuiTestBackend::new(80, 5);  // Very small
    let mut panel = StreamingOutputPanel::new(80, 5);

    // Overflow the panel
    for i in 0..100 {
        panel.push_token(&format!("line{}\n", i));
    }
    panel.render(&mut backend);

    // Should show most recent lines, not first lines
    let text = backend.frame().extract_text(0, 0, 80, 5);
    assert!(text.contains("line99"), "Most recent token not visible");
    assert!(!text.contains("line0"), "Oldest token should be scrolled off");
}
```

### 3.2 ToolStatusPanel

```rust
#[presentar_test]
#[requires_interface(ToolStatusBrick)]
fn tool_status_progress_monotonic() {
    let mut backend = TuiTestBackend::new(48, 20);
    let mut panel = ToolStatusPanel::new(48, 20);

    panel.add_tool("file_read", ToolState::InProgress(0.0));

    let mut prev_progress = 0.0;
    for step in 1..=10 {
        let progress = step as f32 / 10.0;
        panel.update_tool("file_read", ToolState::InProgress(progress));
        panel.render(&mut backend);

        // Extract gauge fill percentage from rendered output
        let rendered_progress = panel.rendered_progress("file_read");
        assert!(
            rendered_progress >= prev_progress,
            "Progress regressed: {} -> {}",
            prev_progress,
            rendered_progress
        );
        prev_progress = rendered_progress;
    }
}

#[presentar_test]
fn tool_status_blocked_shows_reason() {
    let mut backend = TuiTestBackend::new(48, 20);
    let mut panel = ToolStatusPanel::new(48, 20);

    panel.add_tool("shell", ToolState::Blocked {
        reason: "Sandbox: /etc/passwd write denied".to_string(),
    });
    panel.render(&mut backend);

    let text = backend.frame().extract_text(0, 0, 48, 20);
    assert!(text.contains("blocked"), "Blocked state not shown");
    assert!(text.contains("/etc/passwd"), "Block reason not shown");
}

#[presentar_test]
fn tool_status_parallel_display() {
    let mut backend = TuiTestBackend::new(48, 20);
    let mut panel = ToolStatusPanel::new(48, 20);

    // 3 tools running simultaneously
    panel.add_tool("file_read", ToolState::InProgress(0.8));
    panel.add_tool("grep", ToolState::InProgress(0.5));
    panel.add_tool("shell", ToolState::InProgress(0.3));
    panel.render(&mut backend);

    let text = backend.frame().extract_text(0, 0, 48, 20);
    assert!(text.contains("file_read"), "Tool 1 not visible");
    assert!(text.contains("grep"), "Tool 2 not visible");
    assert!(text.contains("shell"), "Tool 3 not visible");
}
```

### 3.3 CostDashboardPanel

```rust
#[presentar_test]
#[requires_interface(CostDashboardBrick)]
fn cost_dashboard_nonnegative() {
    use proptest::prelude::*;

    proptest!(|(
        input_tokens in 0u32..1_000_000,
        output_tokens in 0u32..1_000_000,
        input_rate in 0.0f64..100.0,
        output_rate in 0.0f64..100.0,
    )| {
        let cost = CostEstimate::new(input_tokens, output_tokens, input_rate, output_rate);
        prop_assert!(cost.estimated_usd >= 0.0, "Negative cost: {}", cost.estimated_usd);
        prop_assert!(cost.estimated_usd.is_finite(), "Non-finite cost");
    });
}

#[presentar_test]
fn cost_dashboard_budget_bar_clamped() {
    let mut backend = TuiTestBackend::new(40, 19);
    let mut panel = CostDashboardPanel::new(40, 19);

    // Set cost higher than budget (should clamp bar to 100%, not overflow)
    panel.update(CostUpdate {
        turn_cost: 10.0,
        cumulative_cost: 10.0,
        session_budget: 5.0,  // Over budget!
        provider: "anthropic".to_string(),
        model: "claude-sonnet-4".to_string(),
        input_tokens: 50000,
        output_tokens: 10000,
    });
    panel.render(&mut backend);

    // Budget bar fill should be 1.0 (clamped), not 2.0
    assert_eq!(panel.budget_bar_fill(), 1.0);

    let text = backend.frame().extract_text(0, 0, 40, 19);
    assert!(text.contains("$10.00") || text.contains("10.00"));
}
```

### 3.4 SandboxLogPanel

```rust
#[presentar_test]
#[requires_interface(SandboxLogBrick)]
fn sandbox_violations_never_silent() {
    let mut backend = TuiTestBackend::new(40, 19);
    let mut panel = SandboxLogPanel::new(40, 19);

    // Push 100 violations
    for i in 0..100 {
        panel.add_violation(SandboxViolation {
            tool: "shell".to_string(),
            action: format!("write /tmp/test{}", i),
            reason: "Sovereign: no writes outside project".to_string(),
            timestamp: Instant::now(),
        });
    }
    panel.render(&mut backend);

    let text = backend.frame().extract_text(0, 0, 40, 19);
    // Most recent violation must be visible
    assert!(text.contains("test99"), "Most recent violation not visible");
}

#[presentar_test]
fn sandbox_panel_contrast_aaa() {
    let mut backend = TuiTestBackend::new(40, 19);
    let mut panel = SandboxLogPanel::new(40, 19);

    panel.add_violation(SandboxViolation {
        tool: "shell".to_string(),
        action: "write /etc/passwd".to_string(),
        reason: "Sovereign: blocked".to_string(),
        timestamp: Instant::now(),
    });
    panel.render(&mut backend);

    // Check all cells in violation text have AAA contrast (7.0)
    for y in 0..19 {
        for x in 0..40 {
            let cell = backend.frame().cell_at(x, y);
            if cell.is_violation_text() {
                let ratio = cell.fg.contrast_ratio(&cell.bg);
                assert!(
                    ratio >= 7.0,
                    "Violation text at ({},{}) has contrast {}, need 7.0 (AAA)",
                    x, y, ratio
                );
            }
        }
    }
}
```

### 3.5 StatusBar

```rust
#[presentar_test]
#[requires_interface(StatusBarBrick)]
fn statusbar_state_matches_agent() {
    let mut backend = TuiTestBackend::new(120, 1);
    let mut statusbar = StatusBar::new(120);

    for state in [AgentState::Idle, AgentState::Perceive, AgentState::Reason,
                  AgentState::Act, AgentState::Remember, AgentState::Done, AgentState::Failed] {
        statusbar.update(StatusUpdate {
            state,
            iterations: 5,
            max_iterations: 50,
            context_pct: 0.45,
            session_cost: 0.02,
        });
        statusbar.render(&mut backend);

        let text = backend.frame().extract_text(0, 0, 120, 1);
        let state_name = format!("{:?}", state);
        assert!(
            text.contains(&state_name),
            "StatusBar doesn't show state {:?}, shows: {}",
            state, text
        );
    }
}

#[presentar_test]
fn statusbar_visible_at_minimum_size() {
    let mut backend = TuiTestBackend::new(20, 10);
    let mut layout = AgentTuiLayout::new(20, 10);  // Minimal detail level

    layout.render(&mut backend);

    // StatusBar is the last row
    let statusbar_row = backend.frame().extract_text(0, 9, 20, 1);
    assert!(
        !statusbar_row.trim().is_empty(),
        "StatusBar empty at 20x10 — minimum size fails"
    );
}
```

---

## 4. Layout Composition Tests

```rust
#[presentar_test]
fn layout_no_overlap_sweep() {
    let mut tracker = PixelCoverageTracker::new(200, 60, 200, 60);

    // Test every terminal size from 20x10 to 200x60
    for w in (20..=200).step_by(10) {
        for h in (10..=60).step_by(5) {
            let layout = AgentTuiLayout::new(w, h);
            let panels = layout.panel_rects();

            // Check no overlap
            for i in 0..panels.len() {
                for j in (i+1)..panels.len() {
                    let overlap = panels[i].intersection(&panels[j]);
                    assert!(
                        overlap.is_empty(),
                        "Panels {} and {} overlap at {}x{}: {:?}",
                        i, j, w, h, overlap
                    );
                }
            }

            // Check no gaps (union covers full area)
            let total_cells: usize = panels.iter().map(|p| p.area()).sum();
            assert_eq!(
                total_cells,
                w * h,
                "Gap detected at {}x{}: {} panel cells vs {} total",
                w, h, total_cells, w * h
            );
        }
    }
}

#[presentar_test]
fn layout_degradation_levels() {
    // Verify detail levels match spec thresholds exactly
    assert_eq!(detail_level(200, 60), DetailLevel::Exploded);
    assert_eq!(detail_level(160, 50), DetailLevel::Exploded);
    assert_eq!(detail_level(159, 50), DetailLevel::Expanded);
    assert_eq!(detail_level(120, 40), DetailLevel::Expanded);
    assert_eq!(detail_level(119, 40), DetailLevel::Normal);
    assert_eq!(detail_level(100, 30), DetailLevel::Normal);
    assert_eq!(detail_level(99, 30), DetailLevel::Compact);
    assert_eq!(detail_level(80, 24), DetailLevel::Compact);
    assert_eq!(detail_level(79, 24), DetailLevel::Minimal);
    assert_eq!(detail_level(20, 10), DetailLevel::Minimal);
}
```

---

## 5. Pixel Coverage Tests

```rust
#[presentar_test]
fn pixel_coverage_all_panels() {
    let mut tracker = PixelCoverageTracker::new(120, 40, 12, 4);

    // Exercise each panel state
    let states = vec![
        AgentTuiState::idle(),
        AgentTuiState::streaming("Hello world".to_string()),
        AgentTuiState::tools_active(3),
        AgentTuiState::cost_warning(4.50, 5.00),
        AgentTuiState::sandbox_block("shell", "/etc/passwd"),
        AgentTuiState::done(10, 0.02),
    ];

    for state in &states {
        let mut backend = TuiTestBackend::new(120, 40);
        let mut tui = AgentTui::new(120, 40);
        tui.apply_state(state);
        tui.render(&mut backend);

        // Record which cells were written
        for y in 0..40 {
            for x in 0..120 {
                if backend.frame().cell_at(x, y).content != " " {
                    tracker.record_point(x as u32, y as u32);
                }
            }
        }
    }

    let report = tracker.generate_report();
    assert!(
        report.overall_coverage >= 0.80,
        "Pixel coverage {:.1}% < 80% threshold. Cold spots: {:?}",
        report.overall_coverage * 100.0,
        report.cold_spots()
    );

    // Generate heatmap for visual inspection
    report.save_heatmap_png("target/tui_coverage_heatmap.png");
}
```

---

## 6. Visual Regression Tests

```rust
#[presentar_test]
fn visual_regression_idle_state() {
    let mut backend = TuiTestBackend::new(120, 40);
    let mut tui = AgentTui::new(120, 40);
    tui.apply_state(&AgentTuiState::idle());
    tui.render(&mut backend);

    let snapshot = TuiSnapshot::from_backend(&backend);
    let baseline = TuiSnapshot::load("tests/tui/regression/baselines/idle_120x40.snap");

    let diff = snapshot.compare(&baseline);
    assert!(
        diff.ssim() > 0.99,
        "Visual regression in idle state: SSIM={:.4} (need >0.99)",
        diff.ssim()
    );
}

// Repeat for each baseline state:
// streaming_120x40, tools_3_active_120x40, cost_warning_120x40,
// sandbox_block_120x40, minimal_80x24, minimal_20x10
```

### 6.1 Baseline Update Command

```bash
# Update all baselines (after intentional visual change)
probador test tests/tui/regression/ --update-baselines

# Update single baseline
probador test tests/tui/regression/regression.rs::visual_regression_idle_state --update-baselines

# Diff current vs baseline (human review)
probador test tests/tui/regression/ --visual-diff
```

---

## 7. State Machine Playbook Tests

### 7.1 TUI State Transition Playbook

```yaml
# tests/tui/playbooks/tui-state.yaml
version: "1.0"
machine:
  id: "tui_state_transitions"
  initial: "idle_view"

  states:
    idle_view:
      invariants:
        - description: "Streaming panel empty"
          condition: "streaming_panel.is_empty()"
        - description: "StatusBar shows Idle"
          condition: "statusbar.state == 'Idle'"

    streaming_view:
      invariants:
        - description: "Streaming panel has content"
          condition: "streaming_panel.token_count() > 0"
        - description: "StatusBar shows Reason or Act"
          condition: "statusbar.state in ['Reason', 'Act']"

    tools_view:
      invariants:
        - description: "Tool panel shows active tools"
          condition: "tool_panel.active_count() > 0"

    cost_warning_view:
      invariants:
        - description: "Cost bar shows warning color"
          condition: "cost_panel.warning_active()"
        - description: "Budget utilization > 80%"
          condition: "cost_panel.budget_utilization() > 0.80"

    error_view:
      invariants:
        - description: "Error message visible"
          condition: "streaming_panel.has_error()"
        - description: "StatusBar shows Failed"
          condition: "statusbar.state == 'Failed'"

  transitions:
    - from: "idle_view"
      to: "streaming_view"
      event: "first_token"

    - from: "streaming_view"
      to: "tools_view"
      event: "tool_use_started"

    - from: "tools_view"
      to: "streaming_view"
      event: "tool_results_received"

    - from: "streaming_view"
      to: "idle_view"
      event: "turn_complete"

    - from: "streaming_view"
      to: "cost_warning_view"
      event: "budget_threshold_crossed"

    - from: "*"
      to: "error_view"
      event: "agent_failed"

  forbidden_transitions:
    - from: "idle_view"
      to: "tools_view"
      reason: "Cannot show tools without streaming first"
    - from: "idle_view"
      to: "cost_warning_view"
      reason: "Cannot warn about cost when idle"
```

### 7.2 Running Playbook Tests

```bash
# Validate playbook structure
probador playbook tests/tui/playbooks/tui-state.yaml --validate

# Run state machine tests
probador playbook tests/tui/playbooks/tui-state.yaml

# Run with mutation testing (M1-M5)
probador playbook tests/tui/playbooks/tui-state.yaml --mutate

# Export FSM diagram
probador playbook tests/tui/playbooks/tui-state.yaml --export svg -o tui-fsm.svg
```

---

## 8. Accessibility Tests

```rust
#[presentar_test]
fn wcag_aa_contrast_all_themes() {
    let themes = vec!["tokyo-night", "dracula", "nord", "monokai"];

    for theme_name in &themes {
        let theme = Theme::load(theme_name);
        let mut backend = TuiTestBackend::new(120, 40);
        let mut tui = AgentTui::new_with_theme(120, 40, theme);
        tui.apply_state(&AgentTuiState::streaming("Test content".into()));
        tui.render(&mut backend);

        // Check every non-empty cell
        let mut violations = Vec::new();
        for y in 0..40 {
            for x in 0..120 {
                let cell = backend.frame().cell_at(x, y);
                if cell.content.trim().is_empty() { continue; }

                let ratio = cell.fg.contrast_ratio(&cell.bg);
                if ratio < 4.5 {
                    violations.push((x, y, ratio, theme_name));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "WCAG AA violations in {} theme: {:?}",
            theme_name, &violations[..violations.len().min(5)]
        );
    }
}

#[presentar_test]
fn sandbox_violations_aaa_contrast() {
    // Sandbox warnings must meet AAA (7.0), not just AA (4.5)
    let mut backend = TuiTestBackend::new(120, 40);
    let mut tui = AgentTui::new(120, 40);
    tui.apply_state(&AgentTuiState::sandbox_block("shell", "/etc/passwd"));
    tui.render(&mut backend);

    let sandbox_rect = tui.layout().sandbox_panel_rect();
    for y in sandbox_rect.y..sandbox_rect.bottom() {
        for x in sandbox_rect.x..sandbox_rect.right() {
            let cell = backend.frame().cell_at(x as usize, y as usize);
            if cell.content.trim().is_empty() { continue; }
            let ratio = cell.fg.contrast_ratio(&cell.bg);
            assert!(ratio >= 7.0, "Sandbox text at ({},{}) contrast {} < 7.0 AAA", x, y, ratio);
        }
    }
}
```

---

## 9. Frame Budget Benchmark Tests

```rust
#[presentar_test]
fn frame_budget_10k_frames() {
    let mut backend = TuiTestBackend::new(120, 40);
    let mut tui = AgentTui::new(120, 40);
    let mut budget_violations = 0;

    for frame in 0..10_000 {
        // Simulate realistic workload: streaming tokens + tool updates
        if frame % 3 == 0 {
            tui.push_token(&format!("token{} ", frame));
        }
        if frame % 50 == 0 {
            tui.update_tool("tool", ToolState::InProgress(frame as f32 / 10_000.0));
        }

        let start = Instant::now();
        tui.render(&mut backend);
        let elapsed = start.elapsed();

        if elapsed > Duration::from_millis(16) {
            budget_violations += 1;
        }
    }

    let violation_rate = budget_violations as f64 / 10_000.0;
    assert!(
        violation_rate < 0.01,
        "Frame budget violation rate {:.2}% > 1% threshold ({} violations in 10K frames)",
        violation_rate * 100.0,
        budget_violations
    );
}
```

---

## 10. Falsification Gates

```rust
use jugar_probar::pixel_coverage::falsification::FalsificationGate;

#[presentar_test]
fn falsify_streaming_first_token_latency() {
    FalsificationGate::new("H-STREAM-001", 100, |trial| {
        let mut backend = TuiTestBackend::new(120, 40);
        let mut tui = AgentTui::new(120, 40);

        let start = Instant::now();
        tui.push_token("Hello");
        tui.render(&mut backend);
        let latency = start.elapsed();

        // First token must render within 100ms (TUI-local, not including provider TTFT)
        latency < Duration::from_millis(100)
    }).run_and_assert(0.01);  // <1% failure rate
}

#[presentar_test]
fn falsify_cost_accuracy() {
    FalsificationGate::new("H-COST-001", 1000, |_| {
        let input_tokens = rand::random::<u32>() % 100_000;
        let output_tokens = rand::random::<u32>() % 50_000;
        let rate_in = 3.0;   // $/Mtok
        let rate_out = 15.0;  // $/Mtok

        let actual = (input_tokens as f64 * rate_in + output_tokens as f64 * rate_out) / 1_000_000.0;
        let displayed = CostEstimate::new(input_tokens, output_tokens, rate_in, rate_out).estimated_usd;

        (displayed - actual).abs() / actual.max(0.000001) < 0.05
    }).run_and_assert(0.001);  // <0.1% failure rate
}

#[presentar_test]
fn falsify_layout_no_overlap() {
    FalsificationGate::new("H-LAYOUT-001", 500, |trial| {
        let w = 20 + (trial % 180) as u16;
        let h = 10 + (trial % 50) as u16;
        let layout = AgentTuiLayout::new(w, h);
        let panels = layout.panel_rects();

        for i in 0..panels.len() {
            for j in (i+1)..panels.len() {
                if !panels[i].intersection(&panels[j]).is_empty() {
                    return false;
                }
            }
        }
        true
    }).run_and_assert(0.0);  // ZERO overlap tolerance
}
```

---

## 11. CI Integration

### 11.1 Makefile Targets

```makefile
# Run all TUI tests
test-tui:
	cargo nextest run -p batuta --features agent-tui -E 'test(tui::)'

# Run probar playbooks
test-tui-playbooks:
	probador playbook tests/tui/playbooks/agent-loop.yaml
	probador playbook tests/tui/playbooks/provider-failover.yaml
	probador playbook tests/tui/playbooks/tui-state.yaml

# Run mutation testing on playbooks
test-tui-mutations:
	probador playbook tests/tui/playbooks/tui-state.yaml --mutate

# Generate pixel coverage heatmap
test-tui-coverage:
	cargo nextest run -p batuta --features agent-tui -E 'test(tui::pixel)'
	@echo "Heatmap: target/tui_coverage_heatmap.png"

# Visual regression check
test-tui-regression:
	cargo nextest run -p batuta --features agent-tui -E 'test(tui::regression)'

# Update visual baselines (after intentional changes)
test-tui-update-baselines:
	PROBAR_UPDATE_BASELINES=1 cargo nextest run -p batuta --features agent-tui -E 'test(tui::regression)'

# Full TUI quality gate
test-tui-full: test-tui test-tui-playbooks test-tui-mutations test-tui-coverage test-tui-regression
	@echo "All TUI tests passed"
```

### 11.2 Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit (TUI section)

# Validate all playbooks
if git diff --cached --name-only | grep -q 'tests/tui/playbooks/'; then
    probador playbook tests/tui/playbooks/*.yaml --validate || exit 1
fi

# Check that every panel has tests
for panel in streaming tools cost session sandbox statusbar; do
    if ! test -f "tests/tui/panels/${panel}.rs"; then
        echo "ERROR: Missing test file for panel: ${panel}"
        exit 1
    fi
done
```

---

## 12. Test Coverage Matrix

| Component | Unit | Brick | Pixel | Regression | Playbook | Mutation | Falsification | A11y |
|-----------|------|-------|-------|------------|----------|----------|---------------|------|
| StreamingOutputPanel | X | X | X | X | X | - | H-STREAM-001 | X |
| ToolStatusPanel | X | X | X | X | X | - | H-TOOL-001 | X |
| CostDashboardPanel | X | X | X | X | - | - | H-COST-001 | X |
| SessionPanel | X | X | X | X | - | - | - | X |
| SandboxLogPanel | X | X | X | X | - | - | H-SANDBOX-001 | X (AAA) |
| StatusBar | X | X | X | X | X | - | - | X |
| Layout (6-panel) | X | - | X | X | X | X | H-LAYOUT-001 | - |
| BrickHouse | X | X | - | - | - | - | H-BUDGET-001 | - |
| Agent Loop FSM | - | - | - | - | X | X (M1-M5) | - | - |
| Provider Failover | - | - | - | - | X | X (M1-M5) | - | - |
| TUI State Machine | - | - | - | - | X | X (M1-M5) | - | - |

**Totals:** 46 unit tests, 8 Brick contracts, 1 pixel coverage sweep, 7 visual regression baselines, 5 playbooks, 3 mutation suites, 6 falsification gates, 10 accessibility checks.
