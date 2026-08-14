# Presentar YAML Testing

Probar provides native support for testing [presentar](https://github.com/anthropics/presentar) TUI dashboard configurations. This enables automated validation of ptop.yaml configurations with a 100-point falsification protocol.

## Overview

The presentar module provides:

- **Schema validation** - Validate ptop.yaml configurations
- **Terminal snapshot testing** - Assert on cell-based terminal output
- **Falsification protocols** - F001-F100 mutation testing
- **Configuration diffing** - Compare configurations

## Quick Start

```rust
use jugar_probar::{
    parse_and_validate_presentar, validate_presentar_config,
    PresentarConfig, TerminalSnapshot, TerminalAssertion,
};

// Parse and validate a configuration
let yaml = r##"
refresh_ms: 500
layout:
  snap_to_grid: true
  grid_size: 8
panels:
  cpu:
    enabled: true
  memory:
    enabled: true
"##;

let (config, result) = parse_and_validate_presentar(yaml).unwrap();
assert!(result.is_ok());
assert_eq!(config.refresh_ms, 500);
```

## Configuration Schema

### PresentarConfig

The root configuration type:

```rust
pub struct PresentarConfig {
    pub refresh_ms: u32,        // Refresh interval (min: 16ms for 60 FPS)
    pub layout: LayoutConfig,   // Layout settings
    pub panels: PanelConfigs,   // Panel configurations
    pub keybindings: KeybindingConfig,
    pub theme: ThemeConfig,
}
```

### Layout Configuration

```rust
pub struct LayoutConfig {
    pub snap_to_grid: bool,     // Snap panels to grid
    pub grid_size: u8,          // Grid size (2-16)
    pub min_panel_width: u16,   // Minimum panel width (default: 30)
    pub min_panel_height: u16,  // Minimum panel height (default: 6)
    pub top_height: f32,        // Top row height ratio (default: 0.45)
    pub bottom_height: f32,     // Bottom row height ratio (default: 0.55)
}
```

### Panel Types

14 panel types are supported:

| Panel | Description |
|-------|-------------|
| `cpu` | CPU usage with sparklines |
| `memory` | Memory usage with ZRAM |
| `disk` | Disk I/O rates |
| `network` | Network RX/TX |
| `process` | Process list |
| `gpu` | GPU utilization |
| `battery` | Battery status |
| `sensors` | Temperature sensors |
| `psi` | Pressure Stall Information |
| `system` | System info |
| `connections` | Network connections |
| `treemap` | Treemap visualization |
| `files` | File browser |

## Validation

### Validation Errors

The validator checks for:

```rust
pub enum PresentarError {
    InvalidRefreshRate(u32),      // < 16ms
    InvalidGridSize(u8),          // Not in 2-16
    InvalidPanelWidth(u16),       // < 10
    InvalidPanelHeight(u16),      // < 3
    InvalidLayoutRatio(f32, f32), // Doesn't sum to 1.0
    DuplicateKeybinding(char, String, String),
    InvalidColorFormat(String),   // Not #RRGGBB
    NoPanelsEnabled,
    InvalidSparklineHistory(u32), // Not in 1-3600
    InvalidProcessColumn(String),
    ParseError(String),
}
```

### Example Validation

```rust
use jugar_probar::{validate_presentar_config, PresentarConfig};

let mut config = PresentarConfig::default();
config.refresh_ms = 5; // Too low!

let result = validate_presentar_config(&config);
assert!(result.is_err());
// Error: "Invalid refresh rate: 5ms (minimum 16ms for 60 FPS)"
```

## Terminal Snapshot Testing

Test TUI output at the cell level:

```rust
use jugar_probar::{TerminalSnapshot, TerminalAssertion, PresentarColor};

// Create a snapshot from text
let snapshot = TerminalSnapshot::from_string(
    "CPU  45% ████████░░░░░░░░ 4 cores\n\
     MEM  60% ██████████░░░░░░ 8GB/16GB",
    80,
    24,
);

// Content assertions
assert!(snapshot.contains("CPU"));
assert!(snapshot.contains_all(&["CPU", "MEM"]));

// Position-based assertions
let assertions = vec![
    TerminalAssertion::Contains("CPU".into()),
    TerminalAssertion::NotContains("GPU".into()),
    TerminalAssertion::CharAt { x: 0, y: 0, expected: 'C' },
];

for assertion in assertions {
    assert!(assertion.check(&snapshot).is_ok());
}
```

### Snapshot Methods

| Method | Description |
|--------|-------------|
| `contains(text)` | Check if text is present |
| `contains_all(&[texts])` | Check all texts are present |
| `contains_any(&[texts])` | Check any text is present |
| `find(text)` | Find first occurrence (x, y) |
| `count_char(ch)` | Count character occurrences |
| `region(x, y, w, h)` | Extract a rectangular region |
| `fg_color_at(x, y)` | Get foreground color |
| `bg_color_at(x, y)` | Get background color |

## Falsification Protocol

The module generates 100 falsification checks (F001-F100) following Popperian testing principles:

### Categories

| Range | Category | Checks |
|-------|----------|--------|
| F001-F014 | Panel Existence | 14 |
| F015-F028 | Panel Content | 14 |
| F029-F042 | Color Consistency | 14 |
| F043-F056 | Layout Consistency | 14 |
| F057-F070 | Keybinding Consistency | 14 |
| F071-F084 | Data Binding | 14 |
| F085-F092 | Performance | 8 |
| F093-F100 | Accessibility | 8 |

### Generating Falsification Playbook

```rust
use jugar_probar::{generate_falsification_playbook, PresentarConfig};

let config = PresentarConfig::default();
let playbook = generate_falsification_playbook(&config);

// Access mutations
if let Some(falsification) = &playbook.falsification {
    for mutation in &falsification.mutations {
        println!("{}: {}", mutation.id, mutation.description);
        println!("  Mutate: {}", mutation.mutate);
        println!("  Expected: {}", mutation.expected_failure);
    }
}
```

### Example Checks

```
F001 - CPU panel exists
  Mutate: panels.cpu.enabled = false
  Expected: CPU panel must be visible

F057 - 'q' quits
  Mutate: keybindings.quit = x
  Expected: 'q' must quit

F085 - 60 FPS render
  Mutate: render_time > 16ms
  Expected: Must render in 16ms
```

## Running the Demo

```bash
cargo run --example presentar_demo -p jugar-probar
```

## References

- Tretmans (2008): Model-Based Testing of Reactive Systems
- Claessen & Hughes (2000): QuickCheck property-based testing
- Jia & Harman (2011): Mutation Testing theory
- Popper (1959): The Logic of Scientific Discovery
