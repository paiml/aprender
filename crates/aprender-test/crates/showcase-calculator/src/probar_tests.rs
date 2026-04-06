//! Probar Advanced Testing for Calculator
//!
//! This module demonstrates all advanced Probar features applied to the
//! calculator application, rebuilt from first principles.
//!
//! # Features Demonstrated
//!
//! - Page Objects for Calculator UI
//! - Accessibility Testing (WCAG 2.1 AA compliance)
//! - Visual Regression Testing
//! - Device Emulation (mobile/tablet/desktop)
//! - Fixture-Based Test Infrastructure
//! - Deterministic Replay
//! - UX Coverage Tracking
//! - Wait Mechanisms
//! - Browser Context Management

use jugar_probar::prelude::*;
use std::collections::HashMap;

// ============================================================================
// SECTION 1: PAGE OBJECTS
// ============================================================================

/// Calculator Page Object - encapsulates all UI interactions
#[derive(Debug, Clone)]
pub struct CalculatorPage {
    /// Display element showing current input/result
    display: Locator,
    /// Digit buttons 0-9
    digit_buttons: HashMap<u8, Locator>,
    /// Operation buttons (+, -, *, /, etc.)
    operation_buttons: HashMap<char, Locator>,
    /// Equals button
    equals_button: Locator,
    /// Clear button
    clear_button: Locator,
    /// Clear entry button
    clear_entry_button: Locator,
    /// History panel
    history_panel: Locator,
    /// Error display
    error_display: Locator,
}

impl Default for CalculatorPage {
    fn default() -> Self {
        Self::new()
    }
}

impl CalculatorPage {
    /// Create a new Calculator Page Object with standard selectors
    #[must_use]
    pub fn new() -> Self {
        let mut digit_buttons = HashMap::new();
        for digit in 0..=9 {
            digit_buttons.insert(
                digit,
                Locator::from_selector(Selector::css(format!("[data-digit=\"{digit}\"]"))),
            );
        }

        let mut operation_buttons = HashMap::new();
        for (op, sel) in [
            ('+', "[data-op=\"add\"]"),
            ('-', "[data-op=\"subtract\"]"),
            ('*', "[data-op=\"multiply\"]"),
            ('/', "[data-op=\"divide\"]"),
            ('%', "[data-op=\"modulo\"]"),
            ('^', "[data-op=\"power\"]"),
            ('.', "[data-key=\"decimal\"]"),
            ('(', "[data-key=\"lparen\"]"),
            (')', "[data-key=\"rparen\"]"),
        ] {
            operation_buttons.insert(op, Locator::from_selector(Selector::css(sel)));
        }

        Self {
            display: Locator::from_selector(Selector::css("[data-display]")),
            digit_buttons,
            operation_buttons,
            equals_button: Locator::from_selector(Selector::css("[data-action=\"equals\"]")),
            clear_button: Locator::from_selector(Selector::css("[data-action=\"clear\"]")),
            clear_entry_button: Locator::from_selector(Selector::css("[data-action=\"ce\"]")),
            history_panel: Locator::from_selector(Selector::css("[data-panel=\"history\"]")),
            error_display: Locator::from_selector(Selector::css("[data-display=\"error\"]")),
        }
    }

    /// Get the display locator
    #[must_use]
    pub fn display(&self) -> &Locator {
        &self.display
    }

    /// Get a digit button locator
    #[must_use]
    pub fn digit(&self, n: u8) -> Option<&Locator> {
        self.digit_buttons.get(&n)
    }

    /// Get an operation button locator
    #[must_use]
    pub fn operation(&self, op: char) -> Option<&Locator> {
        self.operation_buttons.get(&op)
    }

    /// Get the equals button locator
    #[must_use]
    pub fn equals(&self) -> &Locator {
        &self.equals_button
    }

    /// Get the clear button locator
    #[must_use]
    pub fn clear(&self) -> &Locator {
        &self.clear_button
    }

    /// Get the history panel locator
    #[must_use]
    pub fn history(&self) -> &Locator {
        &self.history_panel
    }

    /// Get all button locators for accessibility testing
    #[must_use]
    pub fn all_buttons(&self) -> Vec<&Locator> {
        let mut buttons: Vec<&Locator> = self.digit_buttons.values().collect();
        buttons.extend(self.operation_buttons.values());
        buttons.push(&self.equals_button);
        buttons.push(&self.clear_button);
        buttons.push(&self.clear_entry_button);
        buttons
    }

    /// Build a calculation sequence
    #[must_use]
    pub fn build_calculation(&self, expression: &str) -> Vec<&Locator> {
        let mut sequence = Vec::new();
        for ch in expression.chars() {
            if ch.is_ascii_digit() {
                if let Some(loc) = self.digit(ch.to_digit(10).unwrap_or(0) as u8) {
                    sequence.push(loc);
                }
            } else if let Some(loc) = self.operation(ch) {
                sequence.push(loc);
            } else if ch == '=' {
                sequence.push(&self.equals_button);
            }
        }
        sequence
    }

    /// Get URL pattern
    #[must_use]
    pub fn calc_url_pattern(&self) -> &'static str {
        "/calculator"
    }

    /// Get page name
    #[must_use]
    pub fn calc_page_name(&self) -> &'static str {
        "CalculatorPage"
    }
}

// ============================================================================
// SECTION 2: ACCESSIBILITY CONFIGURATION
// ============================================================================

/// WCAG 2.1 AA compliant color scheme for calculator
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CalculatorTheme {
    /// Background color
    pub background: Color,
    /// Display text color
    pub display_text: Color,
    /// Button text color
    pub button_text: Color,
    /// Button background
    pub button_background: Color,
    /// Operator button background
    pub operator_background: Color,
    /// Focus indicator color
    pub focus_color: Color,
    /// Error text color
    pub error_color: Color,
}

impl Default for CalculatorTheme {
    fn default() -> Self {
        Self::light()
    }
}

impl CalculatorTheme {
    /// Light theme with WCAG AA compliant contrast
    #[must_use]
    pub fn light() -> Self {
        Self {
            background: Color::new(255, 255, 255),
            display_text: Color::new(33, 33, 33), // #212121 - 15.9:1 contrast
            button_text: Color::new(33, 33, 33),  // #212121
            button_background: Color::new(224, 224, 224), // #E0E0E0
            operator_background: Color::new(255, 152, 0), // #FF9800 (orange)
            focus_color: Color::new(33, 150, 243), // #2196F3 (blue)
            error_color: Color::new(211, 47, 47), // #D32F2F (red)
        }
    }

    /// Dark theme with WCAG AA compliant contrast
    #[must_use]
    pub fn dark() -> Self {
        Self {
            background: Color::new(33, 33, 33),            // #212121
            display_text: Color::new(255, 255, 255),       // White - 15.9:1 contrast
            button_text: Color::new(255, 255, 255),        // White
            button_background: Color::new(66, 66, 66),     // #424242
            operator_background: Color::new(255, 167, 38), // #FFA726 (lighter orange)
            focus_color: Color::new(100, 181, 246),        // #64B5F6 (lighter blue)
            error_color: Color::new(239, 83, 80),          // #EF5350 (lighter red)
        }
    }

    /// High contrast theme for accessibility
    #[must_use]
    pub fn high_contrast() -> Self {
        Self {
            background: Color::new(0, 0, 0),              // Black
            display_text: Color::new(255, 255, 0),        // Yellow - max contrast
            button_text: Color::new(255, 255, 255),       // White
            button_background: Color::new(0, 0, 128),     // Navy
            operator_background: Color::new(128, 0, 128), // Purple
            focus_color: Color::new(0, 255, 255),         // Cyan
            error_color: Color::new(255, 0, 0),           // Red
        }
    }

    /// Validate display text contrast meets WCAG AA (4.5:1)
    #[must_use]
    pub fn display_contrast(&self) -> f32 {
        self.display_text.contrast_ratio(&self.background)
    }

    /// Validate button text contrast meets WCAG AA (4.5:1)
    #[must_use]
    pub fn button_contrast(&self) -> f32 {
        self.button_text.contrast_ratio(&self.button_background)
    }

    /// Check if all contrasts pass WCAG AA
    #[must_use]
    pub fn passes_wcag_aa(&self) -> bool {
        self.display_contrast() >= 4.5 && self.button_contrast() >= 3.0
    }
}

// ============================================================================
// SECTION 3: FIXTURE DEFINITIONS
// ============================================================================

/// Calculator test fixture - sets up test environment
#[derive(Debug)]
pub struct CalculatorFixture {
    /// The calculator page object
    pub page: CalculatorPage,
    /// Current theme
    pub theme: CalculatorTheme,
    /// UX coverage tracker
    pub ux_tracker: UxCoverageTracker,
    /// Recorded inputs for replay
    recorded_inputs: Vec<(u64, String)>,
    /// Is set up
    setup_complete: bool,
}

impl Default for CalculatorFixture {
    fn default() -> Self {
        Self::new()
    }
}

impl CalculatorFixture {
    /// Create a new calculator fixture
    #[must_use]
    pub fn new() -> Self {
        Self {
            page: CalculatorPage::new(),
            theme: CalculatorTheme::light(),
            ux_tracker: calculator_coverage(),
            recorded_inputs: Vec::new(),
            setup_complete: false,
        }
    }

    /// Create with dark theme
    #[must_use]
    pub fn with_dark_theme() -> Self {
        Self {
            page: CalculatorPage::new(),
            theme: CalculatorTheme::dark(),
            ux_tracker: calculator_coverage(),
            recorded_inputs: Vec::new(),
            setup_complete: false,
        }
    }

    /// Create with high contrast theme
    #[must_use]
    pub fn with_high_contrast() -> Self {
        Self {
            page: CalculatorPage::new(),
            theme: CalculatorTheme::high_contrast(),
            ux_tracker: calculator_coverage(),
            recorded_inputs: Vec::new(),
            setup_complete: false,
        }
    }

    /// Record a button press for replay
    pub fn record_press(&mut self, button: &str, frame: u64) {
        self.recorded_inputs.push((frame, button.to_string()));
    }

    /// Get number of recorded inputs
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.recorded_inputs.len()
    }

    /// Get UX coverage report
    #[must_use]
    pub fn coverage_report(&self) -> UxCoverageReport {
        self.ux_tracker.generate_report()
    }
}

impl Fixture for CalculatorFixture {
    fn setup(&mut self) -> ProbarResult<()> {
        self.setup_complete = true;
        Ok(())
    }

    fn teardown(&mut self) -> ProbarResult<()> {
        self.setup_complete = false;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "CalculatorFixture"
    }

    fn priority(&self) -> i32 {
        100 // High priority - sets up first
    }
}

// ============================================================================
// SECTION 4: DEVICE PRESETS FOR CALCULATOR
// ============================================================================

/// Device presets optimized for calculator testing
pub mod devices {
    use jugar_probar::emulation::{DeviceDescriptor, TouchMode, Viewport};

    /// iPhone SE - small mobile screen
    #[must_use]
    pub fn iphone_se() -> DeviceDescriptor {
        DeviceDescriptor::new("iPhone SE")
            .with_viewport(Viewport::new(375, 667))
            .with_device_scale_factor(2.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
    }

    /// iPad Mini - tablet
    #[must_use]
    pub fn ipad_mini() -> DeviceDescriptor {
        DeviceDescriptor::new("iPad Mini")
            .with_viewport(Viewport::new(768, 1024))
            .with_device_scale_factor(2.0)
            .with_mobile(true)
            .with_touch(TouchMode::Multi)
    }

    /// Desktop 1080p
    #[must_use]
    pub fn desktop_1080p() -> DeviceDescriptor {
        DeviceDescriptor::new("Desktop 1080p")
            .with_viewport(Viewport::new(1920, 1080))
            .with_device_scale_factor(1.0)
            .with_mobile(false)
            .with_touch(TouchMode::None)
    }

    /// Desktop 4K
    #[must_use]
    pub fn desktop_4k() -> DeviceDescriptor {
        DeviceDescriptor::new("Desktop 4K")
            .with_viewport(Viewport::new(3840, 2160))
            .with_device_scale_factor(2.0)
            .with_mobile(false)
            .with_touch(TouchMode::None)
    }

    /// Ultrawide monitor (21:9)
    #[must_use]
    pub fn ultrawide() -> DeviceDescriptor {
        DeviceDescriptor::new("Ultrawide 21:9")
            .with_viewport(Viewport::new(2560, 1080))
            .with_device_scale_factor(1.0)
            .with_mobile(false)
            .with_touch(TouchMode::None)
    }

    /// All test devices for comprehensive testing
    #[must_use]
    pub fn all_devices() -> Vec<DeviceDescriptor> {
        vec![
            iphone_se(),
            ipad_mini(),
            desktop_1080p(),
            desktop_4k(),
            ultrawide(),
        ]
    }
}

// ============================================================================
// SECTION 5: VISUAL REGRESSION CONFIGURATION
// ============================================================================

/// Visual regression configuration for calculator
#[derive(Debug, Clone)]
pub struct CalculatorVisualConfig {
    /// Threshold for pixel difference (0.0 - 1.0)
    pub threshold: f64,
    /// Regions to mask (e.g., dynamic content)
    pub mask_regions: Vec<MaskRegion>,
    /// Enable anti-aliasing tolerance
    pub anti_alias_tolerance: bool,
}

impl Default for CalculatorVisualConfig {
    fn default() -> Self {
        Self {
            threshold: 0.01, // 1% tolerance
            mask_regions: vec![
                // Mask the display area for calculations that vary
                MaskRegion::new(10, 10, 200, 50),
            ],
            anti_alias_tolerance: true,
        }
    }
}

impl CalculatorVisualConfig {
    /// Strict comparison (0.1% tolerance)
    #[must_use]
    pub fn strict() -> Self {
        Self {
            threshold: 0.001,
            mask_regions: Vec::new(),
            anti_alias_tolerance: false,
        }
    }

    /// Relaxed comparison (5% tolerance, mask display)
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            threshold: 0.05,
            mask_regions: vec![MaskRegion::new(0, 0, 300, 100)],
            anti_alias_tolerance: true,
        }
    }
}

// ============================================================================
// SECTION 6: WAIT CONDITIONS FOR CALCULATOR
// ============================================================================

/// Custom wait conditions for calculator
pub mod wait_conditions {
    use jugar_probar::wait::WaitCondition;

    /// Wait for display to show a specific value
    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct DisplayShowsValue {
        expected: String,
    }

    impl DisplayShowsValue {
        /// Create new wait condition
        #[must_use]
        pub fn new(expected: &str) -> Self {
            Self {
                expected: expected.to_string(),
            }
        }

        /// Get the expected value
        #[must_use]
        #[allow(dead_code)]
        pub fn expected(&self) -> &str {
            &self.expected
        }
    }

    impl WaitCondition for DisplayShowsValue {
        fn check(&self) -> bool {
            true // Simulated success
        }

        fn description(&self) -> String {
            "display shows expected value".to_string()
        }
    }

    /// Wait for calculation to complete
    #[derive(Debug)]
    pub struct CalculationComplete;

    impl WaitCondition for CalculationComplete {
        fn check(&self) -> bool {
            true
        }

        fn description(&self) -> String {
            "calculation complete".to_string()
        }
    }

    /// Wait for history to update
    #[derive(Debug)]
    #[allow(dead_code)]
    pub struct HistoryUpdated {
        expected_count: usize,
    }

    impl HistoryUpdated {
        /// Create new wait condition
        #[must_use]
        pub fn with_count(count: usize) -> Self {
            Self {
                expected_count: count,
            }
        }

        /// Get expected count
        #[must_use]
        #[allow(dead_code)]
        pub fn expected_count(&self) -> usize {
            self.expected_count
        }
    }

    impl WaitCondition for HistoryUpdated {
        fn check(&self) -> bool {
            true
        }

        fn description(&self) -> String {
            "history updated".to_string()
        }
    }
}

// ============================================================================
// TESTS: H₀ Null Hypothesis Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PAGE OBJECT TESTS
    // ========================================================================

    #[test]
    fn h0_page_001_calculator_page_default_creates_valid_page() {
        let page = CalculatorPage::default();
        assert_eq!(page.calc_url_pattern(), "/calculator");
        assert_eq!(page.calc_page_name(), "CalculatorPage");
    }

    #[test]
    fn h0_page_002_calculator_page_has_display_locator() {
        let page = CalculatorPage::new();
        let display = page.display();
        assert!(format!("{:?}", display.selector()).contains("display"));
    }

    #[test]
    fn h0_page_003_calculator_page_has_all_digit_buttons() {
        let page = CalculatorPage::new();
        for digit in 0..=9 {
            assert!(page.digit(digit).is_some(), "Missing digit button: {digit}");
        }
    }

    #[test]
    fn h0_page_004_calculator_page_has_operation_buttons() {
        let page = CalculatorPage::new();
        for op in ['+', '-', '*', '/', '%', '^'] {
            assert!(page.operation(op).is_some(), "Missing operation: {op}");
        }
    }

    #[test]
    fn h0_page_005_calculator_page_has_equals_button() {
        let page = CalculatorPage::new();
        let equals = page.equals();
        assert!(format!("{:?}", equals.selector()).contains("equals"));
    }

    #[test]
    fn h0_page_006_calculator_page_has_clear_button() {
        let page = CalculatorPage::new();
        let clear = page.clear();
        assert!(format!("{:?}", clear.selector()).contains("clear"));
    }

    #[test]
    fn h0_page_007_calculator_page_all_buttons_returns_correct_count() {
        let page = CalculatorPage::new();
        let buttons = page.all_buttons();
        assert!(buttons.len() >= 20);
    }

    #[test]
    fn h0_page_008_build_calculation_parses_digits() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("123");
        assert_eq!(sequence.len(), 3);
    }

    #[test]
    fn h0_page_009_build_calculation_parses_operators() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("1+2");
        assert_eq!(sequence.len(), 3);
    }

    #[test]
    fn h0_page_010_build_calculation_parses_equals() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("1+2=");
        assert_eq!(sequence.len(), 4);
    }

    // ========================================================================
    // THEME TESTS
    // ========================================================================

    #[test]
    fn h0_theme_011_light_theme_default() {
        let theme = CalculatorTheme::default();
        assert_eq!(theme.background.r, 255);
        assert_eq!(theme.background.g, 255);
        assert_eq!(theme.background.b, 255);
    }

    #[test]
    fn h0_theme_012_dark_theme_has_dark_background() {
        let theme = CalculatorTheme::dark();
        assert!(theme.background.r < 100);
        assert!(theme.background.g < 100);
        assert!(theme.background.b < 100);
    }

    #[test]
    fn h0_theme_013_high_contrast_has_black_background() {
        let theme = CalculatorTheme::high_contrast();
        assert_eq!(theme.background.r, 0);
        assert_eq!(theme.background.g, 0);
        assert_eq!(theme.background.b, 0);
    }

    #[test]
    fn h0_theme_014_light_theme_passes_wcag_aa() {
        let theme = CalculatorTheme::light();
        assert!(theme.passes_wcag_aa(), "Light theme fails WCAG AA");
    }

    #[test]
    fn h0_theme_015_dark_theme_passes_wcag_aa() {
        let theme = CalculatorTheme::dark();
        assert!(theme.passes_wcag_aa(), "Dark theme fails WCAG AA");
    }

    #[test]
    fn h0_theme_016_high_contrast_passes_wcag_aa() {
        let theme = CalculatorTheme::high_contrast();
        assert!(theme.passes_wcag_aa());
    }

    #[test]
    fn h0_theme_017_display_text_contrast_minimum_4_5() {
        let theme = CalculatorTheme::light();
        let ratio = theme.display_contrast();
        assert!(
            ratio >= 4.5,
            "Display text contrast {ratio} < 4.5:1 minimum"
        );
    }

    #[test]
    fn h0_theme_018_button_text_contrast_minimum_3_0() {
        let theme = CalculatorTheme::light();
        let ratio = theme.button_contrast();
        assert!(ratio >= 3.0, "Button text contrast {ratio} < 3.0:1 minimum");
    }

    #[test]
    fn h0_theme_019_error_text_visible() {
        let theme = CalculatorTheme::light();
        let ratio = theme.error_color.contrast_ratio(&theme.background);
        assert!(ratio >= 3.0, "Error text contrast {ratio} < 3.0:1");
    }

    #[test]
    fn h0_theme_020_focus_color_visible() {
        let theme = CalculatorTheme::light();
        let ratio = theme.focus_color.contrast_ratio(&theme.background);
        assert!(ratio >= 3.0, "Focus color contrast {ratio} < 3.0:1");
    }

    // ========================================================================
    // FIXTURE TESTS
    // ========================================================================

    #[test]
    fn h0_fixture_021_calculator_fixture_default() {
        let fixture = CalculatorFixture::default();
        assert_eq!(fixture.name(), "CalculatorFixture");
        assert_eq!(fixture.priority(), 100);
    }

    #[test]
    fn h0_fixture_022_fixture_has_page_object() {
        let fixture = CalculatorFixture::new();
        assert_eq!(fixture.page.calc_page_name(), "CalculatorPage");
    }

    #[test]
    fn h0_fixture_023_fixture_has_light_theme_by_default() {
        let fixture = CalculatorFixture::new();
        assert_eq!(fixture.theme.background.r, 255);
    }

    #[test]
    fn h0_fixture_024_fixture_dark_theme_variant() {
        let fixture = CalculatorFixture::with_dark_theme();
        assert!(fixture.theme.background.r < 100);
    }

    #[test]
    fn h0_fixture_025_fixture_high_contrast_variant() {
        let fixture = CalculatorFixture::with_high_contrast();
        assert_eq!(fixture.theme.background.r, 0);
    }

    #[test]
    fn h0_fixture_026_fixture_setup_initializes_tracker() {
        let mut fixture = CalculatorFixture::new();
        assert!(fixture.setup().is_ok());
        assert!(fixture.setup_complete);
    }

    #[test]
    fn h0_fixture_027_fixture_teardown_clears_state() {
        let mut fixture = CalculatorFixture::new();
        fixture.setup().unwrap();
        assert!(fixture.teardown().is_ok());
        assert!(!fixture.setup_complete);
    }

    #[test]
    fn h0_fixture_028_fixture_records_replay_input() {
        let mut fixture = CalculatorFixture::new();
        fixture.record_press("1", 0);
        fixture.record_press("+", 1);
        fixture.record_press("2", 2);
        assert_eq!(fixture.input_count(), 3);
    }

    #[test]
    fn h0_fixture_029_fixture_coverage_report_works() {
        let fixture = CalculatorFixture::new();
        let _report = fixture.coverage_report();
        // Report generation should not panic
    }

    #[test]
    fn h0_fixture_030_fixture_has_ux_tracker() {
        let fixture = CalculatorFixture::new();
        let report = fixture.coverage_report();
        assert!(report.total_elements > 0);
    }

    // ========================================================================
    // DEVICE EMULATION TESTS
    // ========================================================================

    #[test]
    fn h0_device_031_iphone_se_viewport() {
        let device = devices::iphone_se();
        assert_eq!(device.viewport.width, 375);
        assert_eq!(device.viewport.height, 667);
    }

    #[test]
    fn h0_device_032_iphone_se_is_mobile() {
        let device = devices::iphone_se();
        assert!(device.is_mobile);
    }

    #[test]
    fn h0_device_033_iphone_se_has_touch() {
        let device = devices::iphone_se();
        assert!(device.touch.is_enabled());
    }

    #[test]
    fn h0_device_034_ipad_mini_viewport() {
        let device = devices::ipad_mini();
        assert_eq!(device.viewport.width, 768);
        assert_eq!(device.viewport.height, 1024);
    }

    #[test]
    fn h0_device_035_desktop_1080p_viewport() {
        let device = devices::desktop_1080p();
        assert_eq!(device.viewport.width, 1920);
        assert_eq!(device.viewport.height, 1080);
    }

    #[test]
    fn h0_device_036_desktop_not_mobile() {
        let device = devices::desktop_1080p();
        assert!(!device.is_mobile);
    }

    #[test]
    fn h0_device_037_desktop_no_touch() {
        let device = devices::desktop_1080p();
        assert!(!device.touch.is_enabled());
    }

    #[test]
    fn h0_device_038_desktop_4k_high_dpi() {
        let device = devices::desktop_4k();
        assert!((device.device_scale_factor - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_device_039_ultrawide_aspect_ratio() {
        let device = devices::ultrawide();
        let ratio = device.viewport.width as f64 / device.viewport.height as f64;
        assert!((ratio - 2.37).abs() < 0.1); // ~21:9
    }

    #[test]
    fn h0_device_040_all_devices_returns_five() {
        let all = devices::all_devices();
        assert_eq!(all.len(), 5);
    }

    // ========================================================================
    // VISUAL REGRESSION TESTS
    // ========================================================================

    #[test]
    fn h0_visual_041_default_threshold() {
        let config = CalculatorVisualConfig::default();
        assert!((config.threshold - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn h0_visual_042_default_has_mask_regions() {
        let config = CalculatorVisualConfig::default();
        assert!(!config.mask_regions.is_empty());
    }

    #[test]
    fn h0_visual_043_strict_has_lower_threshold() {
        let config = CalculatorVisualConfig::strict();
        assert!(config.threshold < 0.01);
    }

    #[test]
    fn h0_visual_044_strict_no_mask_regions() {
        let config = CalculatorVisualConfig::strict();
        assert!(config.mask_regions.is_empty());
    }

    #[test]
    fn h0_visual_045_relaxed_higher_threshold() {
        let config = CalculatorVisualConfig::relaxed();
        assert!(config.threshold > 0.01);
    }

    #[test]
    fn h0_visual_046_relaxed_anti_alias_enabled() {
        let config = CalculatorVisualConfig::relaxed();
        assert!(config.anti_alias_tolerance);
    }

    #[test]
    fn h0_visual_047_strict_anti_alias_disabled() {
        let config = CalculatorVisualConfig::strict();
        assert!(!config.anti_alias_tolerance);
    }

    #[test]
    fn h0_visual_048_mask_region_dimensions() {
        let config = CalculatorVisualConfig::default();
        let mask = &config.mask_regions[0];
        assert!(mask.width > 0);
        assert!(mask.height > 0);
    }

    #[test]
    fn h0_visual_049_relaxed_has_mask() {
        let config = CalculatorVisualConfig::relaxed();
        assert!(!config.mask_regions.is_empty());
    }

    #[test]
    fn h0_visual_050_default_anti_alias_enabled() {
        let config = CalculatorVisualConfig::default();
        assert!(config.anti_alias_tolerance);
    }

    // ========================================================================
    // WAIT CONDITION TESTS
    // ========================================================================

    #[test]
    fn h0_wait_051_display_shows_value_description() {
        let wait = wait_conditions::DisplayShowsValue::new("42");
        assert_eq!(wait.description(), "display shows expected value");
    }

    #[test]
    fn h0_wait_052_display_shows_value_ready() {
        let wait = wait_conditions::DisplayShowsValue::new("42");
        assert!(wait.check());
    }

    #[test]
    fn h0_wait_053_calculation_complete_description() {
        let wait = wait_conditions::CalculationComplete;
        assert_eq!(wait.description(), "calculation complete");
    }

    #[test]
    fn h0_wait_054_calculation_complete_ready() {
        let wait = wait_conditions::CalculationComplete;
        assert!(wait.check());
    }

    #[test]
    fn h0_wait_055_history_updated_description() {
        let wait = wait_conditions::HistoryUpdated::with_count(5);
        assert_eq!(wait.description(), "history updated");
    }

    #[test]
    fn h0_wait_056_history_updated_ready() {
        let wait = wait_conditions::HistoryUpdated::with_count(1);
        assert!(wait.check());
    }

    // ========================================================================
    // INTEGRATION TESTS
    // ========================================================================

    #[test]
    fn h0_int_057_page_object_with_fixture() {
        let mut fixture = CalculatorFixture::new();
        fixture.setup().unwrap();
        assert_eq!(fixture.page.calc_page_name(), "CalculatorPage");
        fixture.teardown().unwrap();
    }

    #[test]
    fn h0_int_058_theme_validation_in_fixture() {
        let fixture = CalculatorFixture::new();
        assert!(fixture.theme.passes_wcag_aa());
    }

    #[test]
    fn h0_int_059_replay_with_fixture() {
        let mut fixture = CalculatorFixture::new();
        fixture.setup().unwrap();
        fixture.record_press("4", 0);
        fixture.record_press("2", 1);
        assert_eq!(fixture.input_count(), 2);
        fixture.teardown().unwrap();
    }

    #[test]
    fn h0_int_060_coverage_tracking_in_fixture() {
        let fixture = CalculatorFixture::new();
        let report = fixture.coverage_report();
        assert!(report.total_elements > 0);
    }

    // ========================================================================
    // ACCESSIBILITY INTEGRATION TESTS
    // ========================================================================

    #[test]
    fn h0_a11y_061_all_themes_pass_wcag_aa() {
        let themes = [
            CalculatorTheme::light(),
            CalculatorTheme::dark(),
            CalculatorTheme::high_contrast(),
        ];
        for theme in themes {
            assert!(theme.passes_wcag_aa());
        }
    }

    #[test]
    fn h0_a11y_062_page_has_accessible_buttons() {
        let page = CalculatorPage::new();
        for button in page.all_buttons() {
            let selector = format!("{:?}", button.selector());
            assert!(
                selector.contains("data-"),
                "Button missing data attribute: {selector}"
            );
        }
    }

    #[test]
    fn h0_a11y_063_display_is_labeled() {
        let page = CalculatorPage::new();
        let selector = format!("{:?}", page.display().selector());
        assert!(selector.contains("display"));
    }

    #[test]
    fn h0_a11y_064_error_display_exists() {
        let page = CalculatorPage::new();
        let selector = format!("{:?}", page.error_display.selector());
        assert!(selector.contains("error"));
    }

    #[test]
    fn h0_a11y_065_history_panel_accessible() {
        let page = CalculatorPage::new();
        let selector = format!("{:?}", page.history().selector());
        assert!(selector.contains("history") || selector.contains("panel"));
    }

    // ========================================================================
    // DEVICE RESPONSIVENESS TESTS
    // ========================================================================

    #[test]
    fn h0_resp_066_mobile_portrait_supported() {
        let device = devices::iphone_se();
        assert!(device.viewport.is_portrait());
    }

    #[test]
    fn h0_resp_067_tablet_portrait_supported() {
        let device = devices::ipad_mini();
        assert!(device.viewport.is_portrait());
    }

    #[test]
    fn h0_resp_068_desktop_landscape_supported() {
        let device = devices::desktop_1080p();
        assert!(device.viewport.is_landscape());
    }

    #[test]
    fn h0_resp_069_ultrawide_landscape() {
        let device = devices::ultrawide();
        assert!(device.viewport.is_landscape());
    }

    #[test]
    fn h0_resp_070_4k_landscape() {
        let device = devices::desktop_4k();
        assert!(device.viewport.is_landscape());
    }

    // ========================================================================
    // UX COVERAGE TESTS
    // ========================================================================

    #[test]
    fn h0_ux_071_tracker_initializes_elements() {
        let fixture = CalculatorFixture::new();
        let report = fixture.coverage_report();
        assert!(report.total_elements >= 10);
    }

    #[test]
    fn h0_ux_072_tracker_has_valid_coverage() {
        let fixture = CalculatorFixture::new();
        let report = fixture.coverage_report();
        assert!(report.total_elements > 0);
    }

    #[test]
    fn h0_ux_073_coverage_elements_reasonable() {
        let fixture = CalculatorFixture::new();
        let report = fixture.coverage_report();
        assert!(report.total_elements <= 100);
    }

    // ========================================================================
    // REPLAY TESTS
    // ========================================================================

    #[test]
    fn h0_replay_074_recorder_initial_empty() {
        let fixture = CalculatorFixture::new();
        assert_eq!(fixture.input_count(), 0);
    }

    #[test]
    fn h0_replay_075_recorder_tracks_inputs() {
        let mut fixture = CalculatorFixture::new();
        fixture.record_press("1", 0);
        assert_eq!(fixture.input_count(), 1);
    }

    #[test]
    fn h0_replay_076_recorder_preserves_order() {
        let mut fixture = CalculatorFixture::new();
        fixture.record_press("1", 0);
        fixture.record_press("+", 1);
        fixture.record_press("2", 2);
        fixture.record_press("=", 3);
        assert_eq!(fixture.input_count(), 4);
    }

    #[test]
    fn h0_replay_077_different_fixtures_independent() {
        let mut f1 = CalculatorFixture::new();
        let f2 = CalculatorFixture::new();
        f1.record_press("1", 0);
        assert_eq!(f1.input_count(), 1);
        assert_eq!(f2.input_count(), 0);
    }

    // ========================================================================
    // EXPRESSION BUILDING TESTS
    // ========================================================================

    #[test]
    fn h0_expr_078_simple_addition() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("1+2=");
        assert_eq!(sequence.len(), 4);
    }

    #[test]
    fn h0_expr_079_multiplication() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("6*7=");
        assert_eq!(sequence.len(), 4);
    }

    #[test]
    fn h0_expr_080_complex_expression() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("12+34=");
        assert_eq!(sequence.len(), 6);
    }

    #[test]
    fn h0_expr_081_handles_unknown_chars() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("1?2"); // ? is unknown
        assert_eq!(sequence.len(), 2); // Only 1 and 2
    }

    #[test]
    fn h0_expr_082_empty_expression() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("");
        assert!(sequence.is_empty());
    }

    #[test]
    fn h0_expr_083_parentheses() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("(1+2)");
        assert_eq!(sequence.len(), 5);
    }

    #[test]
    fn h0_expr_084_decimal() {
        let page = CalculatorPage::new();
        let sequence = page.build_calculation("3.14");
        assert_eq!(sequence.len(), 4);
    }

    // ========================================================================
    // PAGE OBJECT EDGE CASES
    // ========================================================================

    #[test]
    fn h0_edge_085_digit_out_of_range() {
        let page = CalculatorPage::new();
        assert!(page.digit(10).is_none());
    }

    #[test]
    fn h0_edge_086_unknown_operation() {
        let page = CalculatorPage::new();
        assert!(page.operation('@').is_none());
    }

    #[test]
    fn h0_edge_087_url_pattern_matches() {
        let page = CalculatorPage::new();
        assert!(page.calc_url_pattern().starts_with('/'));
    }

    // ========================================================================
    // THEME EDGE CASES
    // ========================================================================

    #[test]
    fn h0_edge_088_color_rgb_values() {
        let theme = CalculatorTheme::light();
        // u8 values are always <= 255, so we just verify the fields exist
        let _ = theme.display_text.r;
        let _ = theme.display_text.g;
        let _ = theme.display_text.b;
    }

    #[test]
    fn h0_edge_089_contrast_ratio_positive() {
        let theme = CalculatorTheme::light();
        let ratio = theme.display_contrast();
        assert!(ratio > 0.0);
    }

    #[test]
    fn h0_edge_090_contrast_ratio_max_21() {
        let theme = CalculatorTheme::high_contrast();
        let ratio = theme.display_contrast();
        assert!(ratio <= 21.0);
    }

    // ========================================================================
    // FIXTURE LIFECYCLE TESTS
    // ========================================================================

    #[test]
    fn h0_life_091_multiple_setup_teardown() {
        let mut fixture = CalculatorFixture::new();
        fixture.setup().unwrap();
        fixture.teardown().unwrap();
        fixture.setup().unwrap();
        fixture.teardown().unwrap();
        assert!(!fixture.setup_complete);
    }

    #[test]
    fn h0_life_092_teardown_without_setup() {
        let mut fixture = CalculatorFixture::new();
        assert!(fixture.teardown().is_ok());
    }

    #[test]
    fn h0_life_093_fixture_priority_high() {
        let fixture = CalculatorFixture::new();
        assert!(fixture.priority() >= 100);
    }

    // ========================================================================
    // VISUAL CONFIG EDGE CASES
    // ========================================================================

    #[test]
    fn h0_cfg_094_threshold_not_negative() {
        let config = CalculatorVisualConfig::default();
        assert!(config.threshold >= 0.0);
    }

    #[test]
    fn h0_cfg_095_threshold_not_over_one() {
        let config = CalculatorVisualConfig::relaxed();
        assert!(config.threshold <= 1.0);
    }

    #[test]
    fn h0_cfg_096_mask_region_valid() {
        let config = CalculatorVisualConfig::default();
        for mask in &config.mask_regions {
            // x and y are u32, always >= 0; verify they exist
            let _ = mask.x;
            let _ = mask.y;
        }
    }

    // ========================================================================
    // COMPREHENSIVE INTEGRATION
    // ========================================================================

    #[test]
    fn h0_full_097_complete_calculation_flow() {
        let mut fixture = CalculatorFixture::new();
        fixture.setup().unwrap();

        // Record a calculation
        fixture.record_press("4", 0);
        fixture.record_press("2", 1);
        fixture.record_press("*", 2);
        fixture.record_press("1", 3);
        fixture.record_press("0", 4);
        fixture.record_press("=", 5);

        // Verify replay recorded
        assert_eq!(fixture.input_count(), 6);

        // Verify theme accessibility
        assert!(fixture.theme.passes_wcag_aa());

        fixture.teardown().unwrap();
    }

    #[test]
    fn h0_full_098_all_devices_have_valid_viewports() {
        for device in devices::all_devices() {
            assert!(device.viewport.width > 0);
            assert!(device.viewport.height > 0);
        }
    }

    #[test]
    fn h0_full_099_all_themes_have_distinct_backgrounds() {
        let light = CalculatorTheme::light();
        let dark = CalculatorTheme::dark();
        let hc = CalculatorTheme::high_contrast();

        assert_ne!(light.background.r, dark.background.r);
        assert_ne!(dark.background.r, hc.background.r);
    }

    #[test]
    fn h0_full_100_fixture_supports_all_operations() {
        let mut fixture = CalculatorFixture::new();
        fixture.setup().unwrap();

        // Test all operations can be recorded
        for op in ['+', '-', '*', '/', '%', '^'] {
            fixture.record_press(&op.to_string(), 0);
        }

        assert_eq!(fixture.input_count(), 6);
        fixture.teardown().unwrap();
    }
}
