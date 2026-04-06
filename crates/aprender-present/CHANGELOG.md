# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- ProcessTable: PID, USER, COMMAND columns now visible (was black text on dark background)
- NetworkPanel: Interface names (eth0, wlan0) now visible in compact mode

### Added
- cbtop_visibility.rs: Tests validating widget text uses visible colors
- scripts/pixel_diff.sh: Pixel diff testing workflow for visual regression
- SPEC-024: Pixel-by-pixel cbtop/ttop recreation specification with 125-pt falsification checklist
- Popperian falsification test suites (196 tests total across 8 files):
  - f001_f020_symbol_rendering.rs: 25 tests for braille/block symbol arrays
  - f021_f040_color_system.rs: 25 tests for CIELAB gradients, themes, ColorMode
  - f041_f060_widget_layout.rs: 23 tests for widget layout constraints
  - f061_f075_text_rendering.rs: 27 tests for text visibility, truncation, unicode
  - f076_f085_performance.rs: 13 tests for frame budget, memory, large data, paint cost
  - f086_f100_integration.rs: 25 tests for system integration, resize, events
  - f101_f115_edge_cases.rs: 37 tests for NaN/Inf handling, zero dimensions, UTF-8, emoji, RTL, large data, threading
  - f116_f120_accessibility.rs: 21 tests for WCAG contrast, color-independent info, keyboard nav, screen reader labels

## [0.1.3] - 2025-12-15

### Changed
- Updated trueno from v0.7.4 to v0.8.5 (simulation testing framework)

## [0.1.2] - 2025-12-01

### Added
- Integration tests directory with end-to-end tests
- Criterion benchmarks for layout operations
- Enhanced rustdoc with examples
- `.clippy.toml` with unwrap() ban configuration
- Comprehensive README with Installation, Usage, Examples sections

### Changed
- Replaced `unwrap()` calls with `expect()` in production code

## [0.1.0] - 2024-11-29

### Added
- Core types: `Size`, `Point`, `Rect`, `Constraints`, `Color`
- Widget trait with measure-layout-paint cycle
- 20+ widget implementations:
  - Button, Text, Column, Row, Container, Stack
  - Chart, Checkbox, DataCard, DataTable, Image
  - ModelCard, ProgressBar, RadioGroup, Select
  - Slider, Tabs, TextInput, Toggle, Tooltip
- RecordingCanvas for testing
- YAML manifest parsing
- Expression language for data binding
- A11yChecker for WCAG 2.1 AA compliance
- 112 documentation chapters with verified tests
- 1188 unit tests

### Architecture
- Unidirectional data flow (Event → State → Widget → Paint)
- WASM-first design targeting `wasm32-unknown-unknown`
- Flexbox-inspired constraint-based layout
- GPU-ready draw command abstraction

[Unreleased]: https://github.com/paiml/presentar/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/presentar/releases/tag/v0.1.0
