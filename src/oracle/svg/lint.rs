//! SVG Linting and Validation
//!
//! Rules for validating SVG diagrams against Material Design 3 guidelines.

use super::layout::{LayoutEngine, LayoutError, GRID_SIZE};
use super::palette::{Color, MaterialPalette, FORBIDDEN_PAIRINGS};

/// Lint severity level
/// Ordered from least to most severe for comparison purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintSeverity {
    /// Info - suggestion
    Info,
    /// Warning - should fix
    Warning,
    /// Error - must fix
    Error,
}

impl std::fmt::Display for LintSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A lint violation
#[derive(Debug, Clone)]
pub struct LintViolation {
    /// Rule that was violated
    pub rule: LintRule,
    /// Severity
    pub severity: LintSeverity,
    /// Human-readable message
    pub message: String,
    /// Element ID if applicable
    pub element_id: Option<String>,
}

impl LintViolation {
    /// Create a violation with an optional element ID reference.
    fn new(
        rule: LintRule,
        severity: LintSeverity,
        message: String,
        element_id: Option<&str>,
    ) -> Self {
        Self { rule, severity, message, element_id: element_id.map(str::to_string) }
    }
}

impl std::fmt::Display for LintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(id) = &self.element_id {
            write!(f, "[{}] {}: {} (element: {})", self.severity, self.rule, self.message, id)
        } else {
            write!(f, "[{}] {}: {}", self.severity, self.rule, self.message)
        }
    }
}

/// Lint rule identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintRule {
    /// No overlapping elements on same layer
    NoOverlap,
    /// All colors from Material palette
    MaterialColors,
    /// 8px grid alignment
    GridAlignment,
    /// File size under 100KB
    FileSize,
    /// Elements within viewport
    WithinBounds,
    /// Minimum contrast ratio
    ContrastRatio,
    /// Consistent stroke widths
    StrokeConsistency,
    /// Text minimum size
    MinTextSize,
    /// Minimum stroke width (video mode: >= 2px)
    MinStrokeWidth,
    /// Internal padding (video mode: >= 20px from box edge to content)
    InternalPadding,
    /// Block gap (video mode: >= 20px between stroked/filtered boxes)
    BlockGap,
    /// Forbidden color pairings that fail WCAG AA contrast
    ForbiddenPairing,
}

impl std::fmt::Display for LintRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOverlap => write!(f, "NO_OVERLAP"),
            Self::MaterialColors => write!(f, "MATERIAL_COLORS"),
            Self::GridAlignment => write!(f, "GRID_ALIGNMENT"),
            Self::FileSize => write!(f, "FILE_SIZE"),
            Self::WithinBounds => write!(f, "WITHIN_BOUNDS"),
            Self::ContrastRatio => write!(f, "CONTRAST_RATIO"),
            Self::StrokeConsistency => write!(f, "STROKE_CONSISTENCY"),
            Self::MinTextSize => write!(f, "MIN_TEXT_SIZE"),
            Self::MinStrokeWidth => write!(f, "MIN_STROKE_WIDTH"),
            Self::InternalPadding => write!(f, "INTERNAL_PADDING"),
            Self::BlockGap => write!(f, "BLOCK_GAP"),
            Self::ForbiddenPairing => write!(f, "FORBIDDEN_PAIRING"),
        }
    }
}

/// SVG linter configuration
#[derive(Debug, Clone)]
pub struct LintConfig {
    /// Maximum file size in bytes
    pub max_file_size: usize,
    /// Grid size for alignment checks
    pub grid_size: f32,
    /// Minimum text size in pixels
    pub min_text_size: f32,
    /// Minimum contrast ratio (WCAG AA is 4.5:1)
    pub min_contrast_ratio: f32,
    /// Check material colors
    pub check_material_colors: bool,
    /// Check grid alignment
    pub check_grid_alignment: bool,
    /// Minimum stroke width in pixels (video mode)
    pub min_stroke_width: f32,
    /// Minimum internal padding in pixels (video mode)
    pub min_internal_padding: f32,
    /// Minimum gap between blocks in pixels (video mode)
    pub min_block_gap: f32,
    /// Check forbidden color pairings (video mode)
    pub check_forbidden_pairings: bool,
}

impl LintConfig {
    /// Video-mode lint configuration with stricter rules for 1080p.
    pub fn video_mode() -> Self {
        Self {
            max_file_size: 100_000,
            grid_size: GRID_SIZE,
            min_text_size: 18.0,
            min_contrast_ratio: 4.5,
            check_material_colors: false, // Video uses VideoPalette
            check_grid_alignment: false,  // Grid protocol handles alignment
            min_stroke_width: 2.0,
            min_internal_padding: 20.0,
            min_block_gap: 20.0,
            check_forbidden_pairings: true,
        }
    }
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            max_file_size: 100_000, // 100KB
            grid_size: GRID_SIZE,
            min_text_size: 11.0,     // Label small
            min_contrast_ratio: 4.5, // WCAG AA
            check_material_colors: true,
            check_grid_alignment: true,
            min_stroke_width: 1.0,
            min_internal_padding: 0.0,
            min_block_gap: 0.0,
            check_forbidden_pairings: false,
        }
    }
}

/// SVG Linter
#[derive(Debug)]
pub struct SvgLinter {
    config: LintConfig,
    palette: MaterialPalette,
}

impl SvgLinter {
    /// Create a new linter with default config
    pub fn new() -> Self {
        Self { config: LintConfig::default(), palette: MaterialPalette::light() }
    }

    /// Create with custom config
    pub fn with_config(config: LintConfig) -> Self {
        Self { config, palette: MaterialPalette::light() }
    }

    /// Set the palette to validate against
    pub fn with_palette(mut self, palette: MaterialPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Lint the layout engine for overlap and bounds issues
    pub fn lint_layout(&self, layout: &LayoutEngine) -> Vec<LintViolation> {
        let mut violations = Vec::new();

        for error in layout.validate() {
            let violation = match error {
                LayoutError::Overlap { id1, id2 } => LintViolation {
                    rule: LintRule::NoOverlap,
                    severity: LintSeverity::Error,
                    message: format!("Elements '{}' and '{}' overlap", id1, id2),
                    element_id: Some(id1),
                },
                LayoutError::OutOfBounds { id } => LintViolation {
                    rule: LintRule::WithinBounds,
                    severity: LintSeverity::Error,
                    message: "Element is outside viewport bounds".to_string(),
                    element_id: Some(id),
                },
                LayoutError::NotAligned { id } => {
                    if self.config.check_grid_alignment {
                        LintViolation {
                            rule: LintRule::GridAlignment,
                            severity: LintSeverity::Warning,
                            message: format!(
                                "Element is not aligned to {}px grid",
                                self.config.grid_size
                            ),
                            element_id: Some(id),
                        }
                    } else {
                        continue;
                    }
                }
            };
            violations.push(violation);
        }

        violations
    }

    /// Check if a color is valid (in the material palette)
    pub fn lint_color(&self, color: &Color, element_id: Option<&str>) -> Option<LintViolation> {
        if !self.config.check_material_colors {
            return None;
        }

        if !self.palette.is_valid_color(color) {
            Some(LintViolation::new(
                LintRule::MaterialColors,
                LintSeverity::Warning,
                format!("Color {} is not in the Material palette", color.to_css_hex()),
                element_id,
            ))
        } else {
            None
        }
    }

    /// Check file size
    pub fn lint_file_size(&self, svg_content: &str) -> Option<LintViolation> {
        if svg_content.len() > self.config.max_file_size {
            Some(LintViolation {
                rule: LintRule::FileSize,
                severity: LintSeverity::Error,
                message: format!(
                    "File size {} bytes exceeds maximum {} bytes",
                    svg_content.len(),
                    self.config.max_file_size
                ),
                element_id: None,
            })
        } else {
            None
        }
    }

    /// Check text size
    pub fn lint_text_size(&self, size: f32, element_id: Option<&str>) -> Option<LintViolation> {
        if size < self.config.min_text_size {
            Some(LintViolation::new(
                LintRule::MinTextSize,
                LintSeverity::Warning,
                format!("Text size {}px is below minimum {}px", size, self.config.min_text_size),
                element_id,
            ))
        } else {
            None
        }
    }

    /// Calculate luminance for contrast ratio
    fn relative_luminance(color: &Color) -> f64 {
        fn channel_luminance(c: u8) -> f64 {
            let c = c as f64 / 255.0;
            if c <= 0.03928 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }

        let r = channel_luminance(color.r);
        let g = channel_luminance(color.g);
        let b = channel_luminance(color.b);

        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Calculate contrast ratio between two colors
    pub fn contrast_ratio(color1: &Color, color2: &Color) -> f64 {
        let l1 = Self::relative_luminance(color1);
        let l2 = Self::relative_luminance(color2);

        let lighter = l1.max(l2);
        let darker = l1.min(l2);

        (lighter + 0.05) / (darker + 0.05)
    }

    /// Check contrast ratio between foreground and background
    pub fn lint_contrast(
        &self,
        foreground: &Color,
        background: &Color,
        element_id: Option<&str>,
    ) -> Option<LintViolation> {
        let ratio = Self::contrast_ratio(foreground, background);

        if ratio < self.config.min_contrast_ratio as f64 {
            Some(LintViolation::new(
                LintRule::ContrastRatio,
                LintSeverity::Warning,
                format!(
                    "Contrast ratio {:.2}:1 is below minimum {:.1}:1 (WCAG AA)",
                    ratio, self.config.min_contrast_ratio
                ),
                element_id,
            ))
        } else {
            None
        }
    }

    /// Check stroke width (video mode: >= 2px).
    pub fn lint_stroke_width(&self, width: f32, element_id: Option<&str>) -> Option<LintViolation> {
        if width < self.config.min_stroke_width {
            Some(LintViolation::new(
                LintRule::MinStrokeWidth,
                LintSeverity::Warning,
                format!(
                    "Stroke width {}px is below minimum {}px",
                    width, self.config.min_stroke_width
                ),
                element_id,
            ))
        } else {
            None
        }
    }

    /// Check internal padding (video mode: >= 20px).
    pub fn lint_internal_padding(
        &self,
        padding: f32,
        element_id: Option<&str>,
    ) -> Option<LintViolation> {
        if self.config.min_internal_padding > 0.0 && padding < self.config.min_internal_padding {
            Some(LintViolation::new(
                LintRule::InternalPadding,
                LintSeverity::Warning,
                format!(
                    "Internal padding {}px is below minimum {}px",
                    padding, self.config.min_internal_padding
                ),
                element_id,
            ))
        } else {
            None
        }
    }

    /// Check gap between blocks (video mode: >= 20px).
    pub fn lint_block_gap(&self, gap: f32, element_id: Option<&str>) -> Option<LintViolation> {
        if self.config.min_block_gap > 0.0 && gap < self.config.min_block_gap {
            Some(LintViolation::new(
                LintRule::BlockGap,
                LintSeverity::Warning,
                format!("Block gap {}px is below minimum {}px", gap, self.config.min_block_gap),
                element_id,
            ))
        } else {
            None
        }
    }

    /// Check if a text/background color pairing is in the forbidden list.
    pub fn lint_forbidden_pairing(
        &self,
        text: &Color,
        bg: &Color,
        element_id: Option<&str>,
    ) -> Option<LintViolation> {
        if !self.config.check_forbidden_pairings {
            return None;
        }

        let text_hex = text.to_css_hex().to_lowercase();
        let bg_hex = bg.to_css_hex().to_lowercase();

        for (forbidden_text, forbidden_bg) in FORBIDDEN_PAIRINGS {
            if text_hex == *forbidden_text && bg_hex == *forbidden_bg {
                return Some(LintViolation::new(
                    LintRule::ForbiddenPairing,
                    LintSeverity::Error,
                    format!(
                        "Forbidden color pairing: {} on {} fails WCAG AA contrast",
                        text_hex, bg_hex
                    ),
                    element_id,
                ));
            }
        }

        None
    }

    /// Run all lint checks and return violations
    pub fn lint_all(
        &self,
        layout: &LayoutEngine,
        svg_content: &str,
        colors: &[(&str, Color)],
        text_sizes: &[(&str, f32)],
    ) -> LintResult {
        let mut violations = Vec::new();

        // Layout checks
        violations.extend(self.lint_layout(layout));

        // File size check
        if let Some(v) = self.lint_file_size(svg_content) {
            violations.push(v);
        }

        // Color checks
        for (id, color) in colors {
            if let Some(v) = self.lint_color(color, Some(id)) {
                violations.push(v);
            }
        }

        // Text size checks
        for (id, size) in text_sizes {
            if let Some(v) = self.lint_text_size(*size, Some(id)) {
                violations.push(v);
            }
        }

        LintResult::new(violations)
    }
}

impl Default for SvgLinter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of linting
#[derive(Debug)]
pub struct LintResult {
    /// All violations found
    pub violations: Vec<LintViolation>,
}

impl LintResult {
    /// Create a new lint result
    pub fn new(violations: Vec<LintViolation>) -> Self {
        Self { violations }
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.violations.iter().any(|v| v.severity == LintSeverity::Error)
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        self.violations.iter().any(|v| v.severity == LintSeverity::Warning)
    }

    /// Check if lint passed (no errors)
    pub fn passed(&self) -> bool {
        !self.has_errors()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.violations.iter().filter(|v| v.severity == LintSeverity::Error).count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.violations.iter().filter(|v| v.severity == LintSeverity::Warning).count()
    }

    /// Get violations by severity
    pub fn by_severity(&self, severity: LintSeverity) -> Vec<&LintViolation> {
        self.violations.iter().filter(|v| v.severity == severity).collect()
    }

    /// Get violations by rule
    pub fn by_rule(&self, rule: LintRule) -> Vec<&LintViolation> {
        self.violations.iter().filter(|v| v.rule == rule).collect()
    }
}

impl std::fmt::Display for LintResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.violations.is_empty() {
            return writeln!(f, "Lint passed: no violations");
        }

        writeln!(
            f,
            "Lint result: {} error(s), {} warning(s)",
            self.error_count(),
            self.warning_count()
        )?;

        for violation in &self.violations {
            writeln!(f, "  {}", violation)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::svg::layout::Viewport;
    use crate::oracle::svg::shapes::Rect;

    #[test]
    fn test_lint_severity_order() {
        assert!(LintSeverity::Error > LintSeverity::Warning);
        assert!(LintSeverity::Warning > LintSeverity::Info);
    }

    #[test]
    fn test_linter_creation() {
        let linter = SvgLinter::new();
        assert_eq!(linter.config.max_file_size, 100_000);
        assert_eq!(linter.config.grid_size, 8.0);
    }

    #[test]
    fn test_lint_color_valid() {
        let linter = SvgLinter::new();
        let palette = MaterialPalette::light();

        // Valid color
        let violation = linter.lint_color(&palette.primary, Some("test"));
        assert!(violation.is_none());
    }

    #[test]
    fn test_lint_color_invalid() {
        let linter = SvgLinter::new();

        // Invalid color (not in palette)
        let violation = linter.lint_color(&Color::rgb(1, 2, 3), Some("test"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::MaterialColors);
    }

    #[test]
    fn test_lint_file_size_ok() {
        let linter = SvgLinter::new();
        let small_svg = "<svg></svg>";

        let violation = linter.lint_file_size(small_svg);
        assert!(violation.is_none());
    }

    #[test]
    fn test_lint_file_size_too_large() {
        let config = LintConfig { max_file_size: 10, ..Default::default() };
        let linter = SvgLinter::with_config(config);

        let violation = linter.lint_file_size("This is longer than 10 bytes");
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::FileSize);
    }

    #[test]
    fn test_lint_text_size() {
        let linter = SvgLinter::new();

        // Too small
        let violation = linter.lint_text_size(8.0, Some("text1"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::MinTextSize);

        // OK
        let violation = linter.lint_text_size(14.0, Some("text2"));
        assert!(violation.is_none());
    }

    #[test]
    fn test_contrast_ratio_calculation() {
        // Black on white should be ~21:1
        let ratio = SvgLinter::contrast_ratio(&Color::rgb(0, 0, 0), &Color::rgb(255, 255, 255));
        assert!(ratio > 20.0 && ratio < 22.0);

        // Same colors should be 1:1
        let ratio =
            SvgLinter::contrast_ratio(&Color::rgb(128, 128, 128), &Color::rgb(128, 128, 128));
        assert!((ratio - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_lint_contrast() {
        let linter = SvgLinter::new();

        // Good contrast (black on white)
        let violation =
            linter.lint_contrast(&Color::rgb(0, 0, 0), &Color::rgb(255, 255, 255), Some("text"));
        assert!(violation.is_none());

        // Poor contrast (light gray on white)
        let violation = linter.lint_contrast(
            &Color::rgb(200, 200, 200),
            &Color::rgb(255, 255, 255),
            Some("text"),
        );
        assert!(violation.is_some());
    }

    #[test]
    fn test_lint_layout_overlap() {
        let mut layout = LayoutEngine::new(Viewport::new(200.0, 200.0).with_padding(0.0));

        // Add overlapping elements directly to test validation
        layout.elements.insert(
            "r1".to_string(),
            super::super::layout::LayoutRect::new("r1", Rect::new(0.0, 0.0, 50.0, 50.0)),
        );
        layout.elements.insert(
            "r2".to_string(),
            super::super::layout::LayoutRect::new("r2", Rect::new(25.0, 25.0, 50.0, 50.0)),
        );

        let linter = SvgLinter::new();
        let violations = linter.lint_layout(&layout);

        assert!(violations.iter().any(|v| v.rule == LintRule::NoOverlap));
    }

    #[test]
    fn test_lint_result() {
        let violations = vec![
            LintViolation {
                rule: LintRule::NoOverlap,
                severity: LintSeverity::Error,
                message: "Overlap".to_string(),
                element_id: Some("r1".to_string()),
            },
            LintViolation {
                rule: LintRule::MaterialColors,
                severity: LintSeverity::Warning,
                message: "Bad color".to_string(),
                element_id: Some("r2".to_string()),
            },
        ];

        let result = LintResult::new(violations);

        assert!(result.has_errors());
        assert!(result.has_warnings());
        assert!(!result.passed());
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.warning_count(), 1);
    }

    #[test]
    fn test_lint_result_passed() {
        let result = LintResult::new(vec![]);
        assert!(result.passed());
        assert!(!result.has_errors());
    }

    #[test]
    fn test_lint_result_display() {
        let result = LintResult::new(vec![]);
        let output = format!("{}", result);
        assert!(output.contains("no violations"));
    }

    #[test]
    fn test_lint_severity_display() {
        assert_eq!(format!("{}", LintSeverity::Error), "ERROR");
        assert_eq!(format!("{}", LintSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", LintSeverity::Info), "INFO");
    }

    #[test]
    fn test_lint_rule_display() {
        assert_eq!(format!("{}", LintRule::NoOverlap), "NO_OVERLAP");
        assert_eq!(format!("{}", LintRule::MaterialColors), "MATERIAL_COLORS");
        assert_eq!(format!("{}", LintRule::GridAlignment), "GRID_ALIGNMENT");
        assert_eq!(format!("{}", LintRule::FileSize), "FILE_SIZE");
        assert_eq!(format!("{}", LintRule::WithinBounds), "WITHIN_BOUNDS");
        assert_eq!(format!("{}", LintRule::ContrastRatio), "CONTRAST_RATIO");
        assert_eq!(format!("{}", LintRule::StrokeConsistency), "STROKE_CONSISTENCY");
        assert_eq!(format!("{}", LintRule::MinTextSize), "MIN_TEXT_SIZE");
    }

    #[test]
    fn test_lint_violation_display() {
        let violation = LintViolation {
            rule: LintRule::NoOverlap,
            severity: LintSeverity::Error,
            message: "Test message".to_string(),
            element_id: Some("elem1".to_string()),
        };
        let output = format!("{}", violation);
        assert!(output.contains("ERROR"));
        assert!(output.contains("NO_OVERLAP"));
        assert!(output.contains("Test message"));
        assert!(output.contains("elem1"));
    }

    #[test]
    fn test_lint_violation_display_no_element_id() {
        let violation = LintViolation {
            rule: LintRule::FileSize,
            severity: LintSeverity::Warning,
            message: "File too large".to_string(),
            element_id: None,
        };
        let output = format!("{}", violation);
        assert!(output.contains("FILE_SIZE"));
        assert!(!output.contains("element:"));
    }

    #[test]
    fn test_linter_with_palette() {
        let linter = SvgLinter::new().with_palette(MaterialPalette::dark());
        let dark_primary = MaterialPalette::dark().primary;
        let violation = linter.lint_color(&dark_primary, None);
        assert!(violation.is_none());
    }

    #[test]
    fn test_lint_config_default_values() {
        let config = LintConfig::default();
        assert_eq!(config.max_file_size, 100_000);
        assert_eq!(config.grid_size, 8.0);
        assert_eq!(config.min_text_size, 11.0);
        assert_eq!(config.min_contrast_ratio, 4.5);
        assert!(config.check_material_colors);
        assert!(config.check_grid_alignment);
    }

    #[test]
    fn test_lint_result_by_severity() {
        let violations = vec![
            LintViolation {
                rule: LintRule::NoOverlap,
                severity: LintSeverity::Error,
                message: "Error 1".to_string(),
                element_id: None,
            },
            LintViolation {
                rule: LintRule::MaterialColors,
                severity: LintSeverity::Warning,
                message: "Warn 1".to_string(),
                element_id: None,
            },
            LintViolation {
                rule: LintRule::FileSize,
                severity: LintSeverity::Error,
                message: "Error 2".to_string(),
                element_id: None,
            },
        ];
        let result = LintResult::new(violations);
        let errors = result.by_severity(LintSeverity::Error);
        let warnings = result.by_severity(LintSeverity::Warning);
        assert_eq!(errors.len(), 2);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_lint_result_by_rule() {
        let violations = vec![
            LintViolation {
                rule: LintRule::NoOverlap,
                severity: LintSeverity::Error,
                message: "Overlap 1".to_string(),
                element_id: None,
            },
            LintViolation {
                rule: LintRule::NoOverlap,
                severity: LintSeverity::Error,
                message: "Overlap 2".to_string(),
                element_id: None,
            },
            LintViolation {
                rule: LintRule::FileSize,
                severity: LintSeverity::Error,
                message: "Size".to_string(),
                element_id: None,
            },
        ];
        let result = LintResult::new(violations);
        let overlaps = result.by_rule(LintRule::NoOverlap);
        let sizes = result.by_rule(LintRule::FileSize);
        assert_eq!(overlaps.len(), 2);
        assert_eq!(sizes.len(), 1);
    }

    #[test]
    fn test_linter_default() {
        let linter = SvgLinter::default();
        assert_eq!(linter.config.max_file_size, 100_000);
    }

    #[test]
    fn test_lint_color_disabled() {
        let config = LintConfig { check_material_colors: false, ..Default::default() };
        let linter = SvgLinter::with_config(config);
        let violation = linter.lint_color(&Color::rgb(1, 2, 3), Some("test"));
        assert!(violation.is_none());
    }

    #[test]
    fn test_lint_result_display_with_violations() {
        let violations = vec![LintViolation {
            rule: LintRule::NoOverlap,
            severity: LintSeverity::Error,
            message: "Overlap".to_string(),
            element_id: None,
        }];
        let result = LintResult::new(violations);
        let output = format!("{}", result);
        assert!(output.contains("1 error(s)"));
    }

    // =========================================================================
    // Coverage Gap Tests — lint_all
    // =========================================================================

    #[test]
    fn test_lint_all_clean() {
        let linter = SvgLinter::new();
        let layout = LayoutEngine::new(Viewport::new(800.0, 600.0).with_padding(16.0));
        let palette = MaterialPalette::light();
        let colors: Vec<(&str, Color)> = vec![("bg", palette.surface)];
        let text_sizes: Vec<(&str, f32)> = vec![("title", 24.0)];

        let result = linter.lint_all(&layout, "<svg></svg>", &colors, &text_sizes);
        assert!(result.passed());
    }

    #[test]
    fn test_lint_all_with_violations() {
        let config = LintConfig {
            max_file_size: 5, // tiny limit to trigger file size error
            ..Default::default()
        };
        let linter = SvgLinter::with_config(config);
        let layout = LayoutEngine::new(Viewport::new(800.0, 600.0).with_padding(16.0));
        let colors: Vec<(&str, Color)> = vec![("bad", Color::rgb(1, 2, 3))];
        let text_sizes: Vec<(&str, f32)> = vec![("tiny", 5.0)];

        let result = linter.lint_all(&layout, "<svg>large content</svg>", &colors, &text_sizes);

        assert!(!result.passed());
        // Should have file size error + color warning + text size warning
        assert!(result.has_errors(), "Should have file size error");
        assert!(result.has_warnings(), "Should have color + text size warnings");
    }

    #[test]
    fn test_lint_all_empty_inputs() {
        let linter = SvgLinter::new();
        let layout = LayoutEngine::new(Viewport::new(800.0, 600.0).with_padding(16.0));
        let colors: Vec<(&str, Color)> = vec![];
        let text_sizes: Vec<(&str, f32)> = vec![];

        let result = linter.lint_all(&layout, "", &colors, &text_sizes);
        assert!(result.passed());
    }

    // =========================================================================
    // Video-Mode Lint Rule Tests
    // =========================================================================

    #[test]
    fn test_lint_config_video_mode() {
        let config = LintConfig::video_mode();
        assert_eq!(config.min_text_size, 18.0);
        assert_eq!(config.min_stroke_width, 2.0);
        assert_eq!(config.min_contrast_ratio, 4.5);
        assert_eq!(config.min_internal_padding, 20.0);
        assert_eq!(config.min_block_gap, 20.0);
        assert!(config.check_forbidden_pairings);
        assert!(!config.check_material_colors);
        assert!(!config.check_grid_alignment);
    }

    #[test]
    fn test_lint_stroke_width_ok() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        assert!(linter.lint_stroke_width(2.0, Some("rect1")).is_none());
        assert!(linter.lint_stroke_width(3.0, None).is_none());
    }

    #[test]
    fn test_lint_stroke_width_too_thin() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        let violation = linter.lint_stroke_width(1.0, Some("rect1"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::MinStrokeWidth);
    }

    #[test]
    fn test_lint_internal_padding_ok() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        assert!(linter.lint_internal_padding(20.0, Some("box1")).is_none());
        assert!(linter.lint_internal_padding(25.0, None).is_none());
    }

    #[test]
    fn test_lint_internal_padding_too_small() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        let violation = linter.lint_internal_padding(15.0, Some("box1"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::InternalPadding);
    }

    #[test]
    fn test_lint_internal_padding_disabled() {
        let linter = SvgLinter::new(); // default has min_internal_padding = 0
        assert!(linter.lint_internal_padding(5.0, None).is_none());
    }

    #[test]
    fn test_lint_block_gap_ok() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        assert!(linter.lint_block_gap(20.0, Some("gap")).is_none());
        assert!(linter.lint_block_gap(30.0, None).is_none());
    }

    #[test]
    fn test_lint_block_gap_too_small() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        let violation = linter.lint_block_gap(10.0, Some("gap"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::BlockGap);
    }

    #[test]
    fn test_lint_block_gap_disabled() {
        let linter = SvgLinter::new(); // default has min_block_gap = 0
        assert!(linter.lint_block_gap(5.0, None).is_none());
    }

    #[test]
    fn test_lint_forbidden_pairing_detected() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        let text = Color::from_hex("#64748b").expect("unexpected failure");
        let bg = Color::from_hex("#0f172a").expect("unexpected failure");
        let violation = linter.lint_forbidden_pairing(&text, &bg, Some("text1"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::ForbiddenPairing);
    }

    #[test]
    fn test_lint_forbidden_pairing_all_forbidden() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        for (text_hex, bg_hex) in super::super::palette::FORBIDDEN_PAIRINGS {
            let text = Color::from_hex(text_hex).expect("unexpected failure");
            let bg = Color::from_hex(bg_hex).expect("unexpected failure");
            assert!(
                linter.lint_forbidden_pairing(&text, &bg, None).is_some(),
                "Expected forbidden pairing {} on {} to be detected",
                text_hex,
                bg_hex
            );
        }
    }

    #[test]
    fn test_lint_forbidden_pairing_good_combo() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        let text = Color::from_hex("#f1f5f9").expect("unexpected failure");
        let bg = Color::from_hex("#0f172a").expect("unexpected failure");
        assert!(linter.lint_forbidden_pairing(&text, &bg, None).is_none());
    }

    #[test]
    fn test_lint_forbidden_pairing_disabled() {
        let linter = SvgLinter::new(); // default has check_forbidden_pairings = false
        let text = Color::from_hex("#64748b").expect("unexpected failure");
        let bg = Color::from_hex("#0f172a").expect("unexpected failure");
        assert!(linter.lint_forbidden_pairing(&text, &bg, None).is_none());
    }

    #[test]
    fn test_lint_rule_display_new_rules() {
        assert_eq!(format!("{}", LintRule::MinStrokeWidth), "MIN_STROKE_WIDTH");
        assert_eq!(format!("{}", LintRule::InternalPadding), "INTERNAL_PADDING");
        assert_eq!(format!("{}", LintRule::BlockGap), "BLOCK_GAP");
        assert_eq!(format!("{}", LintRule::ForbiddenPairing), "FORBIDDEN_PAIRING");
    }

    #[test]
    fn test_lint_video_mode_text_size_18px() {
        let linter = SvgLinter::with_config(LintConfig::video_mode());
        // 18px should pass
        assert!(linter.lint_text_size(18.0, Some("label")).is_none());
        // 17px should fail
        let violation = linter.lint_text_size(17.0, Some("small"));
        assert!(violation.is_some());
        assert_eq!(violation.expect("unexpected failure").rule, LintRule::MinTextSize);
    }
}
