# Changelog

All notable changes to Probar will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-rc.1] - 2026-01-08

### Major Architecture Shift: Brick Architecture (PROBAR-SPEC-009)

This release marks the transition to the **Brick Architecture** - a spec-driven development paradigm where tests ARE the interface. The entire UI is defined by falsifiable assertions and performance budgets.

### Added

#### Feature: Brick Architecture (Spec-Driven Development)
- **180-Point Popperian Falsification Framework**: Complete scientific method for UI testing
- **Zero Hand-Written HTML**: All UI generated from Rust Brick definitions via `gen_index`
- **Cross-Language Contract Tests**: Validate JS postMessage types match Rust WorkerResult variants
- **Wildcard Enum Match Prevention**: `clippy::wildcard_enum_match_arm = "deny"` prevents silent failures
- **Pre-Complete Browser Test Gate**: `.pmat-hooks.toml` requires browser tests before work completion

#### Feature: WCAG 2.1 Contrast & Accessibility Enforcement
- **CIEDE2000 Color Difference**: Lab color space perceptual validation
- **4.5:1 Contrast Ratio**: Automatic enforcement for text elements
- **ARIA Attribute Validation**: Role, label, and live region verification
- **Focus Indicator Testing**: Keyboard navigation accessibility

#### Feature: Dapper-style Distributed Tracing
- **Structured Logging**: `tracing` crate integration with field-based spans
- **RTF Performance Metrics**: Real-Time Factor calculation (inference_time / audio_duration)
- **Performance Span Tracking**: `total_inference_ms`, `processing_start_ms` fields
- **Browser DevTools Integration**: tracing_wasm for Chrome console structured output

#### Feature: Runtime Health Score (probar.toml)
- **15-Point Quality Gate**: module_resolution (5) + app_bootstrap (5) + critical_path (5)
- **Jidoka Automation**: Stop-the-line when runtime health fails
- **Browser Test Requirement**: `browser_required = true` prevents "100/100 with broken app"

### Fixed

#### The 11 Historical Defects (Reference: Appendix B of PROBAR-SPEC-009)
1. **BH-001**: Wildcard match silently ignored new WorkerResult variants
2. **BH-002**: Cross-language type mismatch (JS "ready" vs Rust "Ready")
3. **BH-003**: Console.log statements obscured structured diagnostics
4. **BH-004**: Static HTML diverged from Brick specifications
5. **BH-005**: Missing runtime health validation (passed tests, broken app)
6. **BH-006**: Double-spawn race condition in worker initialization
7. **BH-007**: RTF calculated as 0.0 due to missing timing instrumentation
8. **BH-008**: Pre-complete hook allowed skipping browser tests
9. **BH-009**: Contract tests didn't verify case sensitivity
10. **BH-010**: Performance spans not accumulated across chunks
11. **BH-011**: Health score configuration not enforced

### Technical Details
- **Total Tests**: 335 passing (Brick Architecture validated)
- **Falsification Score**: 180/180 (All hypotheses falsifiable)
- **Coverage**: 95%+ with mutation testing
- **Performance**: 16ms frame budget compliance

## [0.4.1] - 2026-01-05

### Fixed

#### `probar serve` Directory Index Support (Issue #1)
- Fixed directory paths returning HTTP 500 "Is a directory" error
- Now properly serves `index.html` for both `/dir/` and `/dir` paths
- Returns 404 with helpful message for directories without `index.html`

#### `probar serve` Gzip Compression (Issue #2)
- Added automatic gzip compression for all responses
- Reduces WASM transfer sizes by 36-46%
- Uses `tower-http` `CompressionLayer` for efficient streaming compression

### Changed
- Server banner now displays "Gzip: enabled (auto-compression)"
- Normalized path handling (trailing slashes no longer cause errors)

## [0.4.0] - 2025-12-14

### Added

#### PROBAR-SPEC-009: WASM Pixel GUI Demo with GPU-Accelerated Random Fill
- **GpuPixelBuffer**: GPU-accelerated pixel buffer with CPU fallback
- **PCG-XSH-RR RNG**: O'Neill (2014) deterministic random number generation
- **Wilson Score Confidence Intervals**: Statistical rigor for coverage proportions
- **Terminal Heatmap Visualization**: Viridis palette with Unicode block rendering

#### GPU Pixels Module (`gpu_pixels`)
- **PTX Static Analysis**: Detect shared memory u64 addressing bugs
- **Kernel Pixel Tests**: Verify loop branches, barrier sync presence
- **Regression Detection**: Compare PTX patterns across kernel versions
- **Bug Classification**: SharedMemU64Addressing, LoopBranchToEnd, MissingBarrierSync

#### WASM Demo Example
- `wasm_pixel_gui_demo`: 6-phase demonstration of pixel coverage testing
- Convergence to 99% coverage in ~44 frames
- Real-time progress bars with coverage statistics
- Gap detection and Wilson 95% confidence intervals

### Changed
- Enhanced web/validator.rs with comprehensive accessibility and security tests
- Added 32 new tests across gpu_pixels and web modules
- Total tests: 2,914 passing

### Technical Details
- **Coverage**: 95.25% overall (browser module requires integration tests)
- **Clippy**: Clean with `-D warnings`
- **Citations**: O'Neill (2014), Wilson (1927), Nickolls et al. (2008)

## [0.3.0] - 2025-12-13

### Added

#### PIXEL-001 v2.1: Pixel-Perfect Verification Framework
- **Popperian Falsification**: `FalsifiabilityGate` with 15/25 gateway threshold
- **Falsifiable Hypotheses**: `coverage_threshold`, `max_gap_size`, `ssim_threshold` constructors
- **Wilson Score Confidence Intervals**: Statistical rigor for coverage proportions
- **Score Bars**: Visual progress indicators with threshold highlighting

#### Pixel-Perfect Metrics
- **SSIM (Structural Similarity Index)**: Window-based image comparison (8x8 default)
- **PSNR (Peak Signal-to-Noise Ratio)**: dB-scale quality metric with classification
- **CIEDE2000 (ΔE₀₀)**: Lab color space perceptual color difference
- **Perceptual Hashing**: AHash, DHash, PHash algorithms with Hamming distance

#### Rich Terminal Output
- **RichTerminalHeatmap**: Box-drawing characters with ANSI 24-bit color
- **OutputMode**: RichAnsi, NoColorAscii, Json for CI tools
- **Gap Detection**: Actionable recommendations for uncovered regions

#### Configuration Schema
- **PixelCoverageConfig**: TOML/JSON/YAML compatible configuration
- **ThresholdConfig**: Minimum, target, complete thresholds
- **VerificationConfig**: SSIM, PSNR, ΔE, PHash thresholds
- **PerformanceConfig**: Parallel processing, batch size, thread count

#### Parallel Processing Abstractions
- **ParallelContext**: Rayon-ready parallel iteration (sequential fallback)
- **BatchProcessor**: Efficient Delta E and SSIM batch computation
- **Downscaler**: Rapid L1 checks at reduced resolution
- **HashCache**: Perceptual hash caching for reference images

### Changed
- Updated pixel-coverage.md book chapter with PIXEL-001 v2.1 documentation
- Calculator showcase demo now uses full pixel-perfect verification

### Technical Details
- **Total Tests**: 2,669 passing (189 pixel_coverage tests)
- **New Test Categories**: 30 proptest property-based tests
- **Complexity**: Max cyclomatic 1, max cognitive 0 (pixel_coverage module)

## [0.2.0] - 2025-12-12

### Added
- Initial pixel coverage heatmaps
- PNG export with Viridis/Magma/Heat palettes
- Terminal heatmap rendering
- Combined coverage reports

## [0.1.0] - 2025-12-12

### Added

#### Core Testing Framework
- **Locator System**: CSS, XPath, Text, TestId, Entity, Role, Label, Placeholder, AltText selectors
- **Page Object Model**: First-class support for encapsulating UI interactions
- **Fixture System**: Setup/teardown lifecycle with priority ordering
- **Wait Mechanisms**: Custom wait conditions with configurable timeouts

#### Accessibility Testing (WCAG 2.1 AA)
- Color contrast ratio validation (4.5:1 for normal text, 3.0:1 for large text)
- Focus indicator visibility checks
- Semantic structure validation
- Screen reader compatibility helpers

#### Visual Regression Testing
- Golden image comparison with configurable thresholds
- Mask regions for dynamic content exclusion
- Anti-aliasing tolerance options
- Perceptual hash (pHash) for robust frame comparison

#### Device Emulation
- Viewport configuration (mobile, tablet, desktop, ultrawide)
- Device scale factor support
- Touch mode emulation (None, Single, Multi)
- Device descriptor presets (iPhone SE, iPad Mini, Desktop 1080p/4K)

#### UX Coverage Tracking
- Element interaction tracking
- Coverage report generation
- User flow validation

#### Deterministic Replay
- Input recording with frame timestamps
- Reproducible test execution
- Delta encoding for efficient storage

#### Platform Support
- **TUI Testing**: Full ratatui/crossterm integration (default feature)
- **WASM Testing**: `wasm32-unknown-unknown` target support
- **Browser Testing**: Optional CDP/Chromium integration

#### Media Generation
- GIF recording for test artifacts
- PNG snapshot capture
- MP4 video recording (pure Rust)

#### HAR (HTTP Archive) Support
- Network request/response recording
- 50 comprehensive HAR tests

#### Showcase Calculator
- Complete example application demonstrating all Probar features
- 100 H₀ (null hypothesis) tests with EXTREME TDD methodology
- Page Object, Accessibility, Visual Regression, Device Emulation examples

### Documentation
- Comprehensive mdbook documentation
- 100-point QA verification checklist
- Advanced features specification
- Runnable examples (locator_demo, accessibility_demo, coverage_demo, pong_simulation)

### Technical Details
- **Total Tests**: 3,346 passing
- **Test Coverage**: Comprehensive across all modules
- **Rust Version**: 1.75.0+
- **License**: MIT OR Apache-2.0

[1.0.0-rc.1]: https://github.com/paiml/probar/releases/tag/v1.0.0-rc.1
[0.4.1]: https://github.com/paiml/probar/releases/tag/v0.4.1
[0.4.0]: https://github.com/paiml/probar/releases/tag/v0.4.0
[0.3.0]: https://github.com/paiml/probar/releases/tag/v0.3.0
[0.2.0]: https://github.com/paiml/probar/releases/tag/v0.2.0
[0.1.0]: https://github.com/paiml/probar/releases/tag/v0.1.0
