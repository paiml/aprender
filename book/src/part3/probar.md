# Probar: Runtime Testing

> **"Probar (Spanish: 'to test/prove') is a Rust-native testing framework for WASM games and web applications."**

## Overview

**Probar** provides comprehensive runtime testing capabilities:

- **Browser Automation**: Chrome DevTools Protocol (CDP)
- **Visual Regression**: Perceptual image diffing
- **WASM Coverage**: Block-level coverage instrumentation
- **TUI Testing**: Presentar YAML falsification
- **Pixel Coverage**: Heatmap visualization
- **Fault Localization**: Tarantula SBFL (basic)

## Installation

```toml
# Cargo.toml
[dev-dependencies]
jugar-probar = "0.2"
```

```bash
# The crate is published as jugar-probar on crates.io
# (the name "probar" was taken)
```

## Key Features

### **Browser Automation**

Control browsers via CDP:

```rust
use jugar_probar::{Browser, BrowserConfig, Page};

#[tokio::test]
async fn test_login() -> Result<(), Box<dyn std::error::Error>> {
    let browser = Browser::launch(BrowserConfig::default()).await?;
    let page = browser.new_page().await?;

    page.goto("https://example.com/login").await?;
    page.fill("#username", "testuser").await?;
    page.fill("#password", "secret").await?;
    page.click("#submit").await?;

    assert!(page.wait_for_selector(".dashboard").await.is_ok());
    Ok(())
}
```

### **Visual Regression Testing**

Compare screenshots with perceptual diffing:

```rust
use jugar_probar::{VisualRegressionTester, VisualRegressionConfig, MaskRegion};

let tester = VisualRegressionTester::new(
    VisualRegressionConfig::default()
        .with_threshold(0.02)       // 2% pixel difference allowed
        .with_color_threshold(10)   // Per-channel tolerance
);

// Add masks for dynamic content
let comparison = ScreenshotComparison::new()
    .with_mask(MaskRegion::new(0, 0, 100, 50))   // Header
    .with_mask(MaskRegion::new(0, 500, 800, 100)); // Footer

let result = tester.compare_images(&baseline, &current)?;
assert!(result.matches, "Visual regression: {}% diff", result.diff_percentage);
```

### **TUI Testing (Presentar)**

Test terminal UIs with falsification protocol:

```rust
use jugar_probar::{
    TerminalSnapshot, TerminalAssertion,
    PresentarConfig, validate_presentar_config
};

// Load presentar YAML config
let config = PresentarConfig::default();
let result = validate_presentar_config(&config);
assert!(result.is_ok());

// Test terminal output
let snapshot = TerminalSnapshot::from_string(
    "CPU  45% ████████░░░░░░░░ 4 cores\n\
     MEM  60% ██████████░░░░░░ 8GB/16GB",
    80, 24
);

let assertions = [
    TerminalAssertion::Contains("CPU".into()),
    TerminalAssertion::NotContains("ERROR".into()),
    TerminalAssertion::CharAt { x: 0, y: 0, expected: 'C' },
];

for assertion in &assertions {
    assertion.check(&snapshot)?;
}
```

### **Pixel Coverage Heatmaps**

Visualize UI coverage:

```rust
use jugar_probar::pixel_coverage::{PixelCoverageTracker, HeatmapConfig};

let mut tracker = PixelCoverageTracker::new(800, 600);

// Record pixel interactions during tests
tracker.record_click(100, 200);
tracker.record_hover(150, 250);

// Generate heatmap
let heatmap = tracker.generate_heatmap(HeatmapConfig::viridis());
heatmap.save_png("coverage_heatmap.png")?;
```

### **WASM Coverage**

Block-level coverage for WASM modules:

```rust
use jugar_probar::coverage::{CoverageCollector, CoverageConfig, Granularity};

let collector = CoverageCollector::new(
    CoverageConfig::default()
        .with_granularity(Granularity::Block)
);

// Execute WASM with coverage
let report = collector.execute_with_coverage(wasm_module)?;

println!("Coverage: {:.1}%", report.summary().line_coverage * 100.0);
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `browser` | Enable CDP browser control (chromiumoxide, tokio) |
| `runtime` | Enable WASM runtime (wasmtime) |
| `derive` | Enable derive macros for type-safe selectors |

```toml
[dev-dependencies]
jugar-probar = { version = "0.2", features = ["browser", "runtime"] }
```

## Brick Architecture

Probar's unique **Brick Architecture** where tests ARE the interface:

```rust
use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget};

struct StatusBrick {
    message: String,
    is_visible: bool,
}

impl Brick for StatusBrick {
    fn brick_name(&self) -> &'static str {
        "StatusBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),  // WCAG AA
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(50)  // 50ms render budget
    }

    fn verify(&self) -> BrickVerification {
        // Verify assertions...
    }
}
```

## Comparison with Other Tools

| Capability | probar | pmat | oip |
|------------|--------|------|-----|
| Browser Automation | ✅ | ❌ | ❌ |
| Visual Regression | ✅ | ❌ | ❌ |
| WASM Coverage | ✅ | ❌ | ❌ |
| TUI Testing | ✅ | ❌ | ❌ |
| SATD Detection | ❌ | ✅ | ❌ |
| TDG Scoring | ❌ | ✅ | ❌ |
| Defect ML | ❌ | ❌ | ✅ |

**Key insight**: probar executes tests and measures runtime behavior. pmat analyzes static code. oip analyzes test results.

## Toyota Way Principles

Probar applies Toyota Way principles:

| Principle | Implementation |
|-----------|----------------|
| **Poka-Yoke** | Type-safe selectors prevent stringly-typed errors |
| **Muda** | Zero-copy memory views eliminate serialization |
| **Jidoka** | Soft Jidoka (LogAndContinue vs Stop) |
| **Heijunka** | Superblock tiling for amortized scheduling |

## Quality Standards

- **95% minimum test coverage**
- **Zero tolerance for panic paths** (`deny(unwrap_used, expect_used)`)
- **ZERO JavaScript** - pure Rust compiling to `.wasm`

## Version

Current version: **0.2.x** (crates.io: `jugar-probar`)

## Agent Integration: BrowserTool

The `BrowserTool` in the [Agent Runtime](./agent-runtime.md) wraps jugar-probar
as an agent tool. Agents can navigate, screenshot, evaluate JS/WASM, and click
elements via tool calls.

```toml
# Enable in agent manifest
[[capabilities]]
type = "browser"
```

**Privacy enforcement:** Sovereign tier restricts navigation to
`localhost`/`127.0.0.1`/`file://` URLs only. The agent uses BrowserTool
to interact with [wos](../appendix/glossary.md) (WASM OS) for model validation
and visual regression testing.

See [Agent Runtime: BrowserTool](./agent-runtime.md#browsertool-headless-browser-automation)
for full details.

## Next Steps

- **[PMAT: Static Analysis](./pmat.md)**: Pre-test quality checks
- **[OIP: Defect Intelligence](./oip.md)**: Post-test fault analysis
- **[Phase 4: Validation](../part2/phase4-validation.md)**: Batuta's validation workflow
- **[Agent Runtime](./agent-runtime.md)**: BrowserTool integration

---

**Navigate:** [Table of Contents](../SUMMARY.md)
