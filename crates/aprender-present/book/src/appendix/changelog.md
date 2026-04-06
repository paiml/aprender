# Changelog

Version history and release notes.

## Version Format

```
MAJOR.MINOR.PATCH

MAJOR - Breaking API changes
MINOR - New features (backward compatible)
PATCH - Bug fixes (backward compatible)
```

## v0.2.0 (Current - In Development)

### Added

| Feature | Description |
|---------|-------------|
| CLI tool | `presentar serve`, `bundle`, `deploy`, `score`, `gate` commands |
| WebGPU rendering | GPU-accelerated primitive rendering via WGSL shaders |
| Browser router | SPA routing with history API integration |
| Canvas2D fallback | Software rendering for non-WebGPU browsers |
| Hot reload | Live reload during development with WebSocket |
| Chart primitives | Interpolation, Bezier curves, arc geometry, histogram binning |
| Chart examples | Scatter/bubble, heatmap, boxplot, area stacked, donut, sparkline, multi-axis |
| Dashboard examples | Performance monitoring, pipeline viz, infrastructure, research, alerts |
| Edge case examples | Unicode/CJK, RTL, numeric edge cases, slow data, high cardinality, theming |
| Data management | Model version history, dataset lineage tracking, batch upload preview |
| Test fixtures | TAR-based fixture loading for integration tests |
| BDD testing | `describe()`, `expect()`, `TestContext` for behavior specs |
| Virtualization | Scroll virtualization for large lists (60fps at 100k items) |
| Undo/Redo | Command-pattern history with merge and branch support |
| Clipboard | Cross-platform clipboard with format negotiation |
| Gestures | Touch gesture recognition (tap, swipe, pinch, pan) |
| Animations | Keyframe animations with easing functions |
| Keyboard shortcuts | Platform-aware shortcut registration |
| Data binding | Two-way reactive bindings with validation |
| Grid layout | CSS Grid-like layout with auto-placement |

### Improved

| Area | Enhancement |
|------|-------------|
| Coverage | 91.18% line coverage, 94.97% function coverage |
| Tests | 3,463+ tests across workspace (194 new example tests) |
| Lint | All clippy warnings resolved with targeted allows |
| YAML | Expression executor with aggregations and transforms |
| Quality | Grade system (F-A) with configurable gates |

### Architecture

- WebGPU instanced rendering pipeline
- Browser event loop integration
- LocalStorage state persistence
- WebSocket real-time communication

## v0.1.0

### Added

| Feature | Description |
|---------|-------------|
| Core widgets | Button, Text, Row, Column, Stack |
| Layout engine | Flexbox-inspired constraint system |
| Test harness | Zero-dependency visual testing |
| YAML config | Declarative app definition |
| A11y checking | WCAG 2.1 AA validation |

### Architecture

- Unidirectional data flow
- Widget trait with measure-layout-paint
- RecordingCanvas for draw commands
- CSS-like selectors for testing

## Versioning Policy

```rust
// Check version at runtime
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn check_compatibility(required: &str) -> bool {
    let current: Vec<u32> = VERSION.split('.')
        .filter_map(|s| s.parse().ok())
        .collect();
    let req: Vec<u32> = required.split('.')
        .filter_map(|s| s.parse().ok())
        .collect();

    // Major version must match
    current.get(0) == req.get(0)
}
```

## Migration Notes

| From | To | Action |
|------|-----|--------|
| 0.0.x | 0.1.x | Update Widget trait |

## Verified Test

```rust
#[test]
fn test_changelog_version_parsing() {
    let version = "0.1.0";
    let parts: Vec<u32> = version.split('.')
        .filter_map(|s| s.parse().ok())
        .collect();

    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], 0);  // Major
    assert_eq!(parts[1], 1);  // Minor
    assert_eq!(parts[2], 0);  // Patch
}
```
