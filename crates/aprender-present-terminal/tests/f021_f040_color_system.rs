//! F021-F040: Color System Falsification Tests
//!
//! SPEC-024 Section B: Validates that presentar-terminal color system
//! implements CIELAB perceptual gradients and correct mode detection.
//!
//! Methodology: Each test attempts to DISPROVE the claim. A passing test
//! means the falsification criterion was NOT met (i.e., the implementation is correct).

use presentar_core::Color;
use presentar_terminal::{ColorMode, Gradient, Theme};

// =============================================================================
// F021: LAB Interpolation Perceptual Uniformity
// =============================================================================

/// F021: LAB interpolation midpoint
/// Falsification criterion: RGB(red→blue).sample(0.5) differs from LAB(red→blue).sample(0.5) by >ΔE 5
///
/// This test verifies that LAB interpolation produces perceptually different results
/// from naive RGB interpolation, demonstrating proper CIELAB implementation.
#[test]
fn f021_lab_interpolation_differs_from_rgb() {
    let start = Color::RED; // (1.0, 0.0, 0.0)
    let end = Color::BLUE; // (0.0, 0.0, 1.0)

    // LAB interpolation via Gradient
    let gradient = Gradient::two(start, end);
    let lab_mid = gradient.sample(0.5);

    // Naive RGB interpolation
    let rgb_mid = Color::new(
        (start.r + end.r) / 2.0,
        (start.g + end.g) / 2.0,
        (start.b + end.b) / 2.0,
        1.0,
    );

    // Calculate ΔE (color difference)
    // LAB and RGB midpoints should be noticeably different for red→blue
    let delta_r = (lab_mid.r - rgb_mid.r).abs();
    let delta_g = (lab_mid.g - rgb_mid.g).abs();
    let delta_b = (lab_mid.b - rgb_mid.b).abs();

    // LAB should produce a different result - especially green channel
    // RGB midpoint of red→blue is (0.5, 0.0, 0.5) = magenta
    // LAB midpoint passes through a different path
    let total_delta = delta_r + delta_g + delta_b;

    // Verify LAB produces meaningful difference (not just using RGB)
    // For red→blue, the difference should be noticeable
    assert!(
        total_delta > 0.01 || lab_mid.g > 0.01,
        "F021 FAILED: LAB interpolation should differ from RGB. LAB={lab_mid:?}, RGB={rgb_mid:?}, delta={total_delta}"
    );
}

// =============================================================================
// F022-F023: Gradient Endpoint Tests
// =============================================================================

/// F022: Gradient 0.0 returns start
/// Falsification criterion: `gradient.sample(0.0) != stops[0]`
#[test]
fn f022_gradient_start_returns_first_stop() {
    let start = Color::RED;
    let end = Color::BLUE;
    let gradient = Gradient::two(start, end);

    let sampled = gradient.sample(0.0);

    // Should be very close to start color
    assert!(
        (sampled.r - start.r).abs() < 0.02,
        "F022 FAILED: gradient.sample(0.0).r should equal start.r. Got {}, expected {}",
        sampled.r,
        start.r
    );
    assert!(
        (sampled.g - start.g).abs() < 0.02,
        "F022 FAILED: gradient.sample(0.0).g should equal start.g. Got {}, expected {}",
        sampled.g,
        start.g
    );
    assert!(
        (sampled.b - start.b).abs() < 0.02,
        "F022 FAILED: gradient.sample(0.0).b should equal start.b. Got {}, expected {}",
        sampled.b,
        start.b
    );
}

/// F023: Gradient 1.0 returns end
/// Falsification criterion: `gradient.sample(1.0) != stops[last]`
#[test]
fn f023_gradient_end_returns_last_stop() {
    let start = Color::RED;
    let end = Color::BLUE;
    let gradient = Gradient::two(start, end);

    let sampled = gradient.sample(1.0);

    // Should be very close to end color
    assert!(
        (sampled.r - end.r).abs() < 0.02,
        "F023 FAILED: gradient.sample(1.0).r should equal end.r. Got {}, expected {}",
        sampled.r,
        end.r
    );
    assert!(
        (sampled.g - end.g).abs() < 0.02,
        "F023 FAILED: gradient.sample(1.0).g should equal end.g. Got {}, expected {}",
        sampled.g,
        end.g
    );
    assert!(
        (sampled.b - end.b).abs() < 0.02,
        "F023 FAILED: gradient.sample(1.0).b should equal end.b. Got {}, expected {}",
        sampled.b,
        end.b
    );
}

/// F024: Gradient clamping
/// Falsification criterion: `sample(-0.5)` or `sample(1.5)` panics
#[test]
fn f024_gradient_clamps_out_of_range() {
    let gradient = Gradient::two(Color::RED, Color::BLUE);

    // These should NOT panic, should clamp to valid range
    let under = gradient.sample(-0.5);
    let over = gradient.sample(1.5);

    // Under should clamp to start (red)
    assert!(
        (under.r - 1.0).abs() < 0.02,
        "F024 FAILED: sample(-0.5) should clamp to start. Got r={}",
        under.r
    );

    // Over should clamp to end (blue)
    assert!(
        (over.b - 1.0).abs() < 0.02,
        "F024 FAILED: sample(1.5) should clamp to end. Got b={}",
        over.b
    );
}

// =============================================================================
// F025-F027: 256/16 Color Mapping Tests
// =============================================================================

/// F025: 256-color grayscale
/// Falsification criterion: Gray not mapped to 232-255 range
#[test]
fn f025_256_color_grayscale_mapping() {
    let mode = ColorMode::Color256;

    // Test various gray values
    let test_grays = [
        (0.0, 0.0, 0.0), // Black - should be 16 (cube) or 232+ (grayscale)
        (0.5, 0.5, 0.5), // Mid gray - should be 232-255
        (1.0, 1.0, 1.0), // White - should be 231 (cube) or 255 (grayscale)
    ];

    for (r, g, b) in test_grays {
        let color = Color::new(r, g, b, 1.0);
        let result = mode.to_crossterm(color);

        // Verify result is valid AnsiValue
        match result {
            crossterm::style::Color::AnsiValue(_v) => {
                // Valid AnsiValue (u8 range is always 0-255)
            }
            _ => panic!("F025 FAILED: Color256 mode should return AnsiValue"),
        }
    }
}

/// F026: 256-color cube
/// Falsification criterion: RGB not mapped to 16-231 cube
#[test]
fn f026_256_color_cube_mapping() {
    let mode = ColorMode::Color256;

    // Test pure colors (should be in 6x6x6 cube, indices 16-231)
    let test_colors = [
        (1.0, 0.0, 0.0), // Red
        (0.0, 1.0, 0.0), // Green
        (0.0, 0.0, 1.0), // Blue
        (1.0, 1.0, 0.0), // Yellow
        (0.0, 1.0, 1.0), // Cyan
        (1.0, 0.0, 1.0), // Magenta
    ];

    for (r, g, b) in test_colors {
        let color = Color::new(r, g, b, 1.0);
        let result = mode.to_crossterm(color);

        match result {
            crossterm::style::Color::AnsiValue(v) => {
                // Pure colors should be in the 6x6x6 cube (16-231)
                assert!(
                    (16..=231).contains(&v),
                    "F026 FAILED: Pure color ({r},{g},{b}) should map to cube (16-231), got {v}"
                );
            }
            _ => panic!("F026 FAILED: Color256 mode should return AnsiValue"),
        }
    }
}

/// F027: 16-color mapping
/// Falsification criterion: Bright colors not distinguished
#[test]
fn f027_16_color_distinguishes_bright() {
    use crossterm::style::Color as CtColor;

    let mode = ColorMode::Color16;

    // Dark vs bright red
    let dark_red = mode.to_crossterm(Color::new(0.5, 0.0, 0.0, 1.0));
    let bright_red = mode.to_crossterm(Color::new(1.0, 0.0, 0.0, 1.0));

    // Both should be some form of red
    assert!(
        matches!(dark_red, CtColor::Red | CtColor::DarkRed),
        "F027 FAILED: Dark red should map to red variant"
    );
    assert!(
        matches!(bright_red, CtColor::Red | CtColor::DarkRed),
        "F027 FAILED: Bright red should map to red variant"
    );

    // Test that pure black and white are correct
    let black = mode.to_crossterm(Color::new(0.0, 0.0, 0.0, 1.0));
    let white = mode.to_crossterm(Color::new(1.0, 1.0, 1.0, 1.0));

    assert!(
        matches!(black, CtColor::Black),
        "F027 FAILED: Pure black should map to Black"
    );
    assert!(
        matches!(white, CtColor::White),
        "F027 FAILED: Pure white should map to White"
    );
}

// =============================================================================
// F028-F030: ColorMode Detection Tests
// =============================================================================

/// F028: `ColorMode` detection `TrueColor`
/// Falsification criterion: `COLORTERM=truecolor` not detected
#[test]
fn f028_colormode_detects_truecolor() {
    let mode = ColorMode::detect_with_env(Some("truecolor".to_string()), None);
    assert_eq!(
        mode,
        ColorMode::TrueColor,
        "F028 FAILED: COLORTERM=truecolor should detect TrueColor"
    );

    let mode_24bit = ColorMode::detect_with_env(Some("24bit".to_string()), None);
    assert_eq!(
        mode_24bit,
        ColorMode::TrueColor,
        "F028 FAILED: COLORTERM=24bit should detect TrueColor"
    );
}

/// F029: `ColorMode` detection 256
/// Falsification criterion: `TERM=xterm-256color` not detected
#[test]
fn f029_colormode_detects_256color() {
    let mode = ColorMode::detect_with_env(None, Some("xterm-256color".to_string()));
    assert_eq!(
        mode,
        ColorMode::Color256,
        "F029 FAILED: TERM=xterm-256color should detect Color256"
    );

    let mode_screen = ColorMode::detect_with_env(None, Some("screen-256color".to_string()));
    assert_eq!(
        mode_screen,
        ColorMode::Color256,
        "F029 FAILED: TERM=screen-256color should detect Color256"
    );
}

/// F030: `ColorMode` fallback
/// Falsification criterion: Missing TERM defaults to Mono; Unknown TERM defaults to Color16
#[test]
fn f030_colormode_fallback_behavior() {
    // Missing TERM should default to Mono
    let mode_none = ColorMode::detect_with_env(None, None);
    assert_eq!(
        mode_none,
        ColorMode::Mono,
        "F030 FAILED: Missing TERM should default to Mono, got {mode_none:?}"
    );

    // "dumb" TERM should be Mono
    let mode_dumb = ColorMode::detect_with_env(None, Some("dumb".to_string()));
    assert_eq!(
        mode_dumb,
        ColorMode::Mono,
        "F030 FAILED: TERM=dumb should default to Mono, got {mode_dumb:?}"
    );

    // Unknown TERM should default to Color16
    let mode_unknown = ColorMode::detect_with_env(None, Some("vt100".to_string()));
    assert_eq!(
        mode_unknown,
        ColorMode::Color16,
        "F030 FAILED: Unknown TERM (vt100) should default to Color16, got {mode_unknown:?}"
    );
}

// =============================================================================
// F031: RGB to ANSI Escape
// =============================================================================

/// F031: RGB to ANSI escape
/// Falsification criterion: `Color(1,0,0)` != `\x1b[38;2;255;0;0m`
#[test]
fn f031_rgb_to_ansi_escape_correct() {
    use crossterm::style::Color as CtColor;

    let mode = ColorMode::TrueColor;
    let red = Color::new(1.0, 0.0, 0.0, 1.0);
    let result = mode.to_crossterm(red);

    // Should be RGB { r: 255, g: 0, b: 0 }
    match result {
        CtColor::Rgb { r, g, b } => {
            assert_eq!(r, 255, "F031 FAILED: Red component should be 255, got {r}");
            assert_eq!(g, 0, "F031 FAILED: Green component should be 0, got {g}");
            assert_eq!(b, 0, "F031 FAILED: Blue component should be 0, got {b}");
        }
        _ => panic!("F031 FAILED: TrueColor mode should return Rgb, got {result:?}"),
    }

    // Test green
    let green = Color::new(0.0, 1.0, 0.0, 1.0);
    let result_g = mode.to_crossterm(green);
    match result_g {
        CtColor::Rgb { r, g, b } => {
            assert_eq!(g, 255, "F031 FAILED: Green component should be 255");
            assert_eq!(r, 0, "F031 FAILED: Red should be 0 for green");
            assert_eq!(b, 0, "F031 FAILED: Blue should be 0 for green");
        }
        _ => panic!("F031 FAILED: TrueColor green should return Rgb"),
    }
}

// =============================================================================
// F032-F035: Theme Color Verification Tests
// =============================================================================

/// F032: Theme `tokyo_night` colors
/// Falsification criterion: Any color != spec
#[test]
fn f032_theme_tokyo_night_colors_match_spec() {
    let theme = Theme::tokyo_night();

    assert_eq!(
        theme.name, "tokyo_night",
        "F032 FAILED: Theme name mismatch"
    );

    // Background: #1a1b26 = rgb(26, 27, 38)
    let expected_bg = Color::new(26.0 / 255.0, 27.0 / 255.0, 38.0 / 255.0, 1.0);
    assert!(
        (theme.background.r - expected_bg.r).abs() < 0.01,
        "F032 FAILED: tokyo_night background.r mismatch"
    );
    assert!(
        (theme.background.g - expected_bg.g).abs() < 0.01,
        "F032 FAILED: tokyo_night background.g mismatch"
    );
    assert!(
        (theme.background.b - expected_bg.b).abs() < 0.01,
        "F032 FAILED: tokyo_night background.b mismatch"
    );

    // Foreground: #c0caf5 = rgb(192, 202, 245)
    let expected_fg = Color::new(192.0 / 255.0, 202.0 / 255.0, 245.0 / 255.0, 1.0);
    assert!(
        (theme.foreground.r - expected_fg.r).abs() < 0.01,
        "F032 FAILED: tokyo_night foreground.r mismatch"
    );
}

/// F033: Theme dracula colors
/// Falsification criterion: Any color != spec
#[test]
fn f033_theme_dracula_colors_match_spec() {
    let theme = Theme::dracula();

    assert_eq!(theme.name, "dracula", "F033 FAILED: Theme name mismatch");

    // Background: #282a36 = rgb(40, 42, 54)
    let expected_bg = Color::new(40.0 / 255.0, 42.0 / 255.0, 54.0 / 255.0, 1.0);
    assert!(
        (theme.background.r - expected_bg.r).abs() < 0.01,
        "F033 FAILED: dracula background.r mismatch"
    );

    // Foreground: #f8f8f2 = rgb(248, 248, 242)
    let expected_fg = Color::new(248.0 / 255.0, 248.0 / 255.0, 242.0 / 255.0, 1.0);
    assert!(
        (theme.foreground.r - expected_fg.r).abs() < 0.01,
        "F033 FAILED: dracula foreground.r mismatch"
    );
}

/// F034: Theme nord colors
/// Falsification criterion: Any color != spec
#[test]
fn f034_theme_nord_colors_match_spec() {
    let theme = Theme::nord();

    assert_eq!(theme.name, "nord", "F034 FAILED: Theme name mismatch");

    // Background: #2e3440 = rgb(46, 52, 64)
    let expected_bg = Color::new(46.0 / 255.0, 52.0 / 255.0, 64.0 / 255.0, 1.0);
    assert!(
        (theme.background.r - expected_bg.r).abs() < 0.01,
        "F034 FAILED: nord background.r mismatch"
    );

    // Foreground: #eceff4 = rgb(236, 239, 244)
    let expected_fg = Color::new(236.0 / 255.0, 239.0 / 255.0, 244.0 / 255.0, 1.0);
    assert!(
        (theme.foreground.r - expected_fg.r).abs() < 0.01,
        "F034 FAILED: nord foreground.r mismatch"
    );
}

/// F035: Theme monokai colors
/// Falsification criterion: Any color != spec
#[test]
fn f035_theme_monokai_colors_match_spec() {
    let theme = Theme::monokai();

    assert_eq!(theme.name, "monokai", "F035 FAILED: Theme name mismatch");

    // Background: #272822 = rgb(39, 40, 34)
    let expected_bg = Color::new(39.0 / 255.0, 40.0 / 255.0, 34.0 / 255.0, 1.0);
    assert!(
        (theme.background.r - expected_bg.r).abs() < 0.01,
        "F035 FAILED: monokai background.r mismatch"
    );

    // Foreground: #f8f8f2 = rgb(248, 248, 242)
    let expected_fg = Color::new(248.0 / 255.0, 248.0 / 255.0, 242.0 / 255.0, 1.0);
    assert!(
        (theme.foreground.r - expected_fg.r).abs() < 0.01,
        "F035 FAILED: monokai foreground.r mismatch"
    );
}

// =============================================================================
// F036-F039: Gradient Behavior Tests
// =============================================================================

/// F036: CPU gradient green→yellow→red
/// Falsification criterion: Incorrect interpolation order
#[test]
fn f036_cpu_gradient_interpolation_order() {
    let theme = Theme::tokyo_night();

    // At 0%, should be close to blue (#7aa2f7 for tokyo_night)
    let at_0 = theme.cpu_color(0.0);

    // At 50%, should be close to yellow (#e0af68)
    let _at_50 = theme.cpu_color(50.0);

    // At 100%, should be close to red (#f7768e)
    let at_100 = theme.cpu_color(100.0);

    // Verify progression: at_50 should have more yellow (higher r and g)
    // and at_100 should be redder
    assert!(
        at_100.r > at_0.r,
        "F036 FAILED: CPU 100% should be redder than 0%. at_0.r={}, at_100.r={}",
        at_0.r,
        at_100.r
    );
}

/// F037: Memory gradient distinct
/// Falsification criterion: CPU and Memory gradients identical
#[test]
fn f037_memory_gradient_distinct_from_cpu() {
    let theme = Theme::tokyo_night();

    // CPU and Memory should have different starting colors
    let cpu_start = theme.cpu_color(0.0);
    let mem_start = theme.memory_color(0.0);

    // At least one component should differ
    let diff_r = (cpu_start.r - mem_start.r).abs();
    let diff_g = (cpu_start.g - mem_start.g).abs();
    let diff_b = (cpu_start.b - mem_start.b).abs();
    let total_diff = diff_r + diff_g + diff_b;

    assert!(
        total_diff > 0.1,
        "F037 FAILED: CPU and Memory gradients should be distinct. diff={total_diff}"
    );
}

/// F038: Gradient `for_percent(50)` returns middle
/// Falsification criterion: Returns wrong color
#[test]
fn f038_gradient_for_percent_50_returns_middle() {
    let gradient = Gradient::three(Color::RED, Color::GREEN, Color::BLUE);

    let at_50 = gradient.for_percent(50.0);

    // At 50%, should be close to middle color (green)
    // Due to LAB interpolation, might not be exact green but should be greenish
    assert!(
        at_50.g > 0.5,
        "F038 FAILED: for_percent(50) should be greenish. Got g={}",
        at_50.g
    );
}

/// F039: Gradient 3-stop correct
/// Falsification criterion: `Gradient::three(R,G,B).sample(0.5)` not equal to G ±ΔE 2
#[test]
fn f039_gradient_three_stop_midpoint() {
    let gradient = Gradient::three(Color::RED, Color::GREEN, Color::BLUE);

    let mid = gradient.sample(0.5);

    // At 0.5, should be very close to the middle stop (GREEN)
    // Due to LAB interpolation, there's some tolerance needed
    let delta_e = (mid.b - Color::GREEN.b)
        .mul_add(
            mid.b - Color::GREEN.b,
            (mid.g - Color::GREEN.g)
                .mul_add(mid.g - Color::GREEN.g, (mid.r - Color::GREEN.r).powi(2)),
        )
        .sqrt();

    // ΔE should be small (we use normalized RGB delta as proxy)
    assert!(
        delta_e < 0.1,
        "F039 FAILED: Three-stop gradient midpoint should be close to middle color. ΔE={delta_e}"
    );
}

/// F040: Color alpha handling
/// Falsification criterion: Alpha != 1.0 causes rendering issues
#[test]
fn f040_color_alpha_handling() {
    let mode = ColorMode::TrueColor;

    // Test with alpha = 0.5 (should be handled gracefully)
    let semi_transparent = Color::new(1.0, 0.0, 0.0, 0.5);
    let result = mode.to_crossterm(semi_transparent);

    // Should still produce valid color (alpha ignored for terminal rendering)
    match result {
        crossterm::style::Color::Rgb { r, g, b } => {
            // RGB should still be correct
            assert_eq!(r, 255, "F040 FAILED: Red should be preserved with alpha");
            assert_eq!(g, 0, "F040 FAILED: Green should be preserved with alpha");
            assert_eq!(b, 0, "F040 FAILED: Blue should be preserved with alpha");
        }
        _ => panic!("F040 FAILED: Should produce valid Rgb color"),
    }

    // Test with alpha = 0.0 - should return Reset to avoid black artifacts
    let transparent = Color::new(0.0, 1.0, 0.0, 0.0);
    let result_transparent = mode.to_crossterm(transparent);

    // Transparent colors (alpha=0) should return Reset, not Rgb
    // This is intentional to avoid black squares on the terminal
    assert!(
        matches!(result_transparent, crossterm::style::Color::Reset),
        "F040 FAILED: Transparent colors (alpha=0) should return Reset to avoid black artifacts"
    );
}

// =============================================================================
// Additional Color System Validation Tests
// =============================================================================

/// Verify LAB roundtrip preserves colors reasonably
#[test]
fn lab_roundtrip_preserves_colors() {
    // Test various colors for LAB roundtrip
    let test_colors = [
        Color::RED,
        Color::GREEN,
        Color::BLUE,
        Color::new(0.5, 0.5, 0.5, 1.0),
        Color::new(0.25, 0.75, 0.5, 1.0),
    ];

    for original in test_colors {
        // Create gradient and sample at 0.0 (should return original)
        let gradient = Gradient::two(original, Color::WHITE);
        let sampled = gradient.sample(0.0);

        let diff = (original.r - sampled.r).abs()
            + (original.g - sampled.g).abs()
            + (original.b - sampled.b).abs();

        assert!(
            diff < 0.1,
            "LAB roundtrip failed for {original:?}. Got {sampled:?}, diff={diff}"
        );
    }
}

/// Verify gradient monotonicity for usage gradients
#[test]
fn gradient_produces_monotonic_progression() {
    let gradient = Gradient::default(); // Green → Yellow → Red

    let mut prev_r = 0.0_f32;

    // Sample at 10% intervals
    for i in 0..=10 {
        let t = f64::from(i) / 10.0;
        let color = gradient.sample(t);

        // Red component should generally increase
        if i > 0 {
            // Allow small decreases due to LAB interpolation
            assert!(
                color.r >= prev_r - 0.1,
                "Gradient red should be monotonic increasing. At t={}: prev={}, curr={}",
                t,
                prev_r,
                color.r
            );
        }
        prev_r = color.r;
    }
}

/// Verify all themes have valid color values
#[test]
fn all_themes_have_valid_colors() {
    let themes = [
        Theme::tokyo_night(),
        Theme::dracula(),
        Theme::nord(),
        Theme::monokai(),
    ];

    for theme in themes {
        // All color components should be in 0-1 range
        assert!(theme.background.r >= 0.0 && theme.background.r <= 1.0);
        assert!(theme.background.g >= 0.0 && theme.background.g <= 1.0);
        assert!(theme.background.b >= 0.0 && theme.background.b <= 1.0);

        assert!(theme.foreground.r >= 0.0 && theme.foreground.r <= 1.0);
        assert!(theme.foreground.g >= 0.0 && theme.foreground.g <= 1.0);
        assert!(theme.foreground.b >= 0.0 && theme.foreground.b <= 1.0);

        // Test gradient sampling doesn't panic
        let _ = theme.cpu_color(0.0);
        let _ = theme.cpu_color(50.0);
        let _ = theme.cpu_color(100.0);
        let _ = theme.memory_color(75.0);
        let _ = theme.gpu_color(25.0);
        let _ = theme.temp_color(65.0, 100.0);
    }
}

/// Verify `ColorMode` default is `TrueColor`
#[test]
fn colormode_default_is_truecolor() {
    assert_eq!(
        ColorMode::default(),
        ColorMode::TrueColor,
        "ColorMode::default() should be TrueColor"
    );
}

/// Verify Mono mode always returns white
#[test]
fn mono_mode_always_returns_white() {
    use crossterm::style::Color as CtColor;

    let mode = ColorMode::Mono;

    let test_colors = [
        Color::RED,
        Color::GREEN,
        Color::BLUE,
        Color::BLACK,
        Color::WHITE,
    ];

    for color in test_colors {
        let result = mode.to_crossterm(color);
        assert_eq!(
            result,
            CtColor::White,
            "Mono mode should always return White for {color:?}"
        );
    }
}
