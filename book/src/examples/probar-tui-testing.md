# Case Study: Probar TUI Testing

This case study demonstrates comprehensive TUI testing using the Probar testing framework. Probar provides Playwright-style assertions, snapshot testing, frame sequences, and UX coverage tracking for terminal user interfaces.

## Overview

Probar enables:

- **Frame-based assertions** - Playwright-style `expect_frame()` API
- **Snapshot testing** - Golden file workflow for regression detection
- **Frame sequences** - Test state transitions across frames
- **UX coverage** - Track interaction and state coverage

## Running the Example

```bash
cargo run -p apr-cli --features inference --example probar_tui_testing
```

## Frame Rendering

Render TUI components to a test buffer:

```rust
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use jugar_probar::tui::TuiFrame;

fn render_frame(app: &MyApp, width: u16, height: u16) -> TuiFrame {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|f| render_dashboard(f, app))
        .expect("draw");
    TuiFrame::from_buffer(terminal.backend().buffer(), 0)
}

let frame = render_frame(&app, 100, 30);
println!("Frame dimensions: {}x{}", frame.width(), frame.height());
```

## Playwright-Style Assertions

Chain assertions with `expect_frame()`:

```rust
use jugar_probar::tui::expect_frame;

let mut assertion = expect_frame(&frame);

// Content assertions
assertion.to_contain_text("Dashboard")?;
assertion.to_contain_text("Status")?;
assertion.not_to_contain_text("ERROR")?;

// Size assertions
assertion.to_have_size(100, 30)?;
```

### Available Assertions

| Method | Description |
|--------|-------------|
| `to_contain_text(s)` | Frame contains substring |
| `not_to_contain_text(s)` | Frame does not contain substring |
| `to_have_size(w, h)` | Frame has exact dimensions |
| `to_match_regex(r)` | Frame matches regex pattern |

## Soft Assertions

Collect multiple failures without stopping:

```rust
let mut soft = expect_frame(&frame).soft();

// These won't stop on first failure
let _ = soft.to_contain_text("Tab 1");
let _ = soft.to_contain_text("Tab 2");
let _ = soft.to_contain_text("Tab 3");
let _ = soft.to_contain_text("Tab 4");

// Check accumulated errors
let errors = soft.errors();
if !errors.is_empty() {
    for err in &errors {
        println!("Failed: {}", err);
    }
}

// Finalize - returns Err if any failures
soft.finalize()?;
```

## Snapshot Testing

Compare frames against golden files:

```rust
use jugar_probar::tui::{TuiSnapshot, SnapshotManager};

// Create snapshot from frame
let snapshot = TuiSnapshot::from_frame("dashboard_main", &frame);

println!("Name: {}", snapshot.name);
println!("Size: {}x{}", snapshot.width, snapshot.height);
println!("Hash: {}", &snapshot.hash[..16]);

// Compare snapshots
let frame2 = render_frame(&app, 100, 30);
let snapshot2 = TuiSnapshot::from_frame("dashboard_check", &frame2);

if snapshot.matches(&snapshot2) {
    println!("Frames match!");
} else {
    println!("Frames differ!");
}
```

### Snapshot Manager (Golden Files)

```rust
use tempfile::TempDir;
use jugar_probar::tui::SnapshotManager;

let temp_dir = TempDir::new()?;
let manager = SnapshotManager::new(temp_dir.path());

// First run: creates golden file
manager.assert_snapshot("dashboard", &frame)?;

// Second run: compares against golden
manager.assert_snapshot("dashboard", &frame)?;

// Check if golden exists
if manager.exists("dashboard") {
    println!("Golden file found");
}
```

### Golden File Workflow

1. **First run** - Creates golden file if missing
2. **Subsequent runs** - Compares against golden
3. **Update** - Delete golden to regenerate
4. **CI** - Fails if frame doesn't match golden

## Frame Sequence Testing

Test state transitions across multiple frames:

```rust
use jugar_probar::tui::FrameSequence;

let mut sequence = FrameSequence::new("tab_navigation");

// Record frames for each tab
for tab in [Tab::Home, Tab::Settings, Tab::Help] {
    app.current_tab = tab;
    let frame = render_frame(&app, 100, 30);
    sequence.add_frame(&frame);
}

// Sequence statistics
println!("Total frames: {}", sequence.len());

// Compare first and last
let first = sequence.first().expect("first");
let last = sequence.last().expect("last");

if !first.matches(last) {
    println!("First and last frames differ (expected for different tabs)");
}
```

## UX Coverage Tracking

### Method 1: UxCoverageBuilder

```rust
use jugar_probar::ux_coverage::{
    UxCoverageBuilder, InteractionType, ElementId, StateId,
};

let mut tracker = UxCoverageBuilder::new()
    // Define clickable elements
    .clickable("tab", "home")
    .clickable("tab", "settings")
    .clickable("tab", "help")
    .clickable("button", "save")
    .clickable("button", "cancel")
    // Define screens/states
    .screen("home")
    .screen("settings")
    .screen("help")
    .build();

// Record user interactions
tracker.record_interaction(
    &ElementId::new("tab", "home"),
    InteractionType::Click,
);
tracker.record_state(StateId::new("screen", "home"));

tracker.record_interaction(
    &ElementId::new("tab", "settings"),
    InteractionType::Click,
);
tracker.record_state(StateId::new("screen", "settings"));

// Generate report
let report = tracker.generate_report();
println!("Elements covered: {}/{}", report.covered_elements, report.total_elements);
println!("States covered:   {}/{}", report.covered_states, report.total_states);
println!("Overall coverage: {:.1}%", report.overall_coverage * 100.0);
println!("Complete: {}", report.is_complete);
```

### Method 2: gui_coverage! Macro

```rust
use jugar_probar::gui_coverage;

let mut gui = gui_coverage! {
    buttons: [
        "tab_home", "tab_settings", "tab_help",
        "save", "cancel"
    ],
    screens: [
        "home", "settings", "help"
    ]
};

// Record interactions
gui.click("tab_home");
gui.visit("home");

gui.click("tab_settings");
gui.visit("settings");

gui.click("save");

// Check coverage
let report = gui.generate_report();
println!("Coverage: {:.1}%", report.overall_coverage * 100.0);

if gui.is_complete() {
    println!("100% UX coverage achieved!");
}
```

### Coverage Metrics

| Metric | Description |
|--------|-------------|
| `covered_elements` | Number of UI elements interacted with |
| `total_elements` | Total defined UI elements |
| `covered_states` | Number of states/screens visited |
| `total_states` | Total defined states |
| `overall_coverage` | Combined coverage (0.0 - 1.0) |
| `is_complete` | True if 100% coverage |

## Testing Best Practices

### 1. Embed Tests in TUI Modules

```rust
// In your tui.rs module
#[cfg(test)]
mod tests {
    use super::*;
    use jugar_probar::tui::expect_frame;

    #[test]
    fn test_dashboard_renders() {
        let app = create_test_app();
        let frame = render_frame(&app, 80, 24);

        expect_frame(&frame)
            .to_contain_text("Dashboard")
            .unwrap();
    }
}
```

### 2. Test All Tabs/States

```rust
#[test]
fn test_all_tabs_render_without_error() {
    let mut app = create_test_app();

    for tab in [Tab::Home, Tab::Settings, Tab::Help, Tab::About] {
        app.current_tab = tab;
        let frame = render_frame(&app, 80, 24);

        // Each tab should render without panicking
        expect_frame(&frame)
            .not_to_contain_text("panic")
            .unwrap();
    }
}
```

### 3. Use Soft Assertions for Multiple Checks

```rust
#[test]
fn test_dashboard_content() {
    let frame = render_frame(&app, 80, 24);

    expect_frame(&frame)
        .soft()
        .to_contain_text("Header")
        .to_contain_text("Footer")
        .to_contain_text("Navigation")
        .to_contain_text("Content")
        .finalize()
        .expect("all content present");
}
```

### 4. Track UX Coverage in CI

```rust
#[test]
fn test_ux_coverage_complete() {
    let mut gui = gui_coverage! {
        buttons: ["tab_1", "tab_2", "tab_3"],
        screens: ["screen_1", "screen_2", "screen_3"]
    };

    // Exercise all UI paths
    for (tab, screen) in [("tab_1", "screen_1"), ("tab_2", "screen_2"), ("tab_3", "screen_3")] {
        gui.click(tab);
        gui.visit(screen);
    }

    assert!(gui.is_complete(), "UX coverage must be 100%");
}
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: TUI Tests

on: [push, pull_request]

jobs:
  tui-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run TUI tests
        run: cargo test --features inference tui

      - name: Check UX coverage
        run: cargo test --features inference test_ux_coverage_complete

      - name: Update snapshots (on main only)
        if: github.ref == 'refs/heads/main'
        run: |
          rm -rf snapshots/
          cargo test --features inference -- --ignored snapshot
          git add snapshots/
```

## Further Reading

- [Federation Gateway](./federation-gateway.md)
- [Federation Routing Policies](./federation-routing.md)
- [State Machine Playbooks](./state-machine-playbooks.md)
