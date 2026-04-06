# Accessibility Testing

Probar includes WCAG accessibility checking for games.

## Overview

Probar validates accessibility requirements:
- Color contrast ratios (WCAG 2.1)
- Photosensitivity (flashing content)
- Text readability
- Input alternatives

## Color Contrast

```rust
use jugar_probar::accessibility::*;

// Check contrast ratio
let ratio = contrast_ratio(foreground_color, background_color);
assert!(ratio >= 4.5);  // WCAG AA for normal text
assert!(ratio >= 7.0);  // WCAG AAA for normal text
assert!(ratio >= 3.0);  // WCAG AA for large text

// Automatic checking
let result = check_text_contrast(&platform);
assert!(result.passes_aa);
```

## Contrast Levels

| Level | Normal Text | Large Text |
|-------|-------------|------------|
| AA | 4.5:1 | 3:1 |
| AAA | 7:1 | 4.5:1 |

## Photosensitivity

```rust
use jugar_probar::accessibility::*;

// Check for problematic flashing
let mut checker = FlashChecker::new();

for frame in 0..180 {  // 3 seconds at 60fps
    let screenshot = platform.capture_frame();
    checker.add_frame(&screenshot);
}

let result = checker.analyze();
assert!(result.safe_for_photosensitive);

// WCAG 2.3.1: No more than 3 flashes per second
assert!(result.max_flashes_per_second <= 3.0);
```

## Color Blindness Simulation

```rust
use jugar_probar::accessibility::*;

// Simulate different types
let normal = platform.capture_frame();
let protanopia = simulate_color_blindness(&normal, ColorBlindType::Protanopia);
let deuteranopia = simulate_color_blindness(&normal, ColorBlindType::Deuteranopia);
let tritanopia = simulate_color_blindness(&normal, ColorBlindType::Tritanopia);

// Check important elements are still distinguishable
assert!(elements_distinguishable(&protanopia, "player", "enemy"));
```

## Text Accessibility

```rust
use jugar_probar::accessibility::*;

// Check text size
let text_elements = platform.locate_all(Locator::component::<Text>());

for text in text_elements {
    let size = platform.get_font_size(text);
    assert!(size >= 12.0, "Text too small: {}", size);

    // Check contrast
    let fg = platform.get_text_color(text);
    let bg = platform.get_background_color(text);
    let ratio = contrast_ratio(fg, bg);
    assert!(ratio >= 4.5, "Insufficient contrast: {}", ratio);
}
```

## Input Alternatives

```rust
use jugar_probar::accessibility::*;

// Verify all actions have keyboard alternatives
let result = check_keyboard_accessibility(&platform);
assert!(result.all_actions_keyboard_accessible);

// List any mouse-only actions
for action in &result.mouse_only_actions {
    println!("Missing keyboard alternative: {}", action);
}
```

## Running Accessibility Tests

```bash
# Run accessibility demo
cargo run --example accessibility_demo -p jugar-probar

# Run accessibility tests
cargo test -p jugar-web accessibility_
```

## Accessibility Report

```rust
pub struct AccessibilityReport {
    pub passes_wcag_aa: bool,
    pub passes_wcag_aaa: bool,
    pub contrast_issues: Vec<ContrastIssue>,
    pub flash_warnings: Vec<FlashWarning>,
    pub keyboard_issues: Vec<KeyboardIssue>,
    pub overall_score: f32,  // 0.0 - 100.0
}
```

## Example Test

```rust
#[test]
fn test_game_accessibility() {
    let mut platform = WebPlatform::new_for_test(config);

    // Run a few frames
    for _ in 0..60 {
        platform.advance_frame(1.0 / 60.0);
    }

    // Check accessibility
    let report = check_accessibility(&platform);

    // Must pass WCAG AA
    assert!(report.passes_wcag_aa, "WCAG AA failures: {:?}", report.contrast_issues);

    // No flash warnings
    assert!(report.flash_warnings.is_empty(), "Flash warnings: {:?}", report.flash_warnings);

    // Score should be high
    assert!(report.overall_score >= 80.0, "Accessibility score too low: {}", report.overall_score);
}
```

## Continuous Monitoring

```rust
use jugar_probar::accessibility::*;

// Monitor throughout gameplay
let mut monitor = AccessibilityMonitor::new();

for frame in 0..6000 {  // 100 seconds
    platform.advance_frame(1.0 / 60.0);

    // Check each frame
    monitor.check_frame(&platform);
}

let report = monitor.finish();
println!("Accessibility issues found: {}", report.issues.len());
```

## Configuration

```rust
pub struct AccessibilityConfig {
    pub min_contrast_ratio: f32,     // Default: 4.5 (AA)
    pub min_text_size: f32,          // Default: 12.0
    pub max_flashes_per_second: f32, // Default: 3.0
    pub require_keyboard_nav: bool,  // Default: true
}

let config = AccessibilityConfig {
    min_contrast_ratio: 7.0,  // AAA level
    ..Default::default()
};

let report = check_accessibility_with_config(&platform, &config);
```
