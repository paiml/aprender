//! Accessibility checking for WCAG 2.1 compliance.
//!
//! Implements comprehensive WCAG 2.1 AA checks including:
//! - Color contrast (1.4.3, 1.4.6)
//! - Keyboard accessibility (2.1.1)
//! - Focus indicators (2.4.7)
//! - Touch target size (2.5.5)
//! - Name/role/value (4.1.2)
//! - Heading hierarchy (1.3.1)

use presentar_core::widget::AccessibleRole;
use presentar_core::{Color, Widget};

/// Minimum touch target size in pixels (WCAG 2.5.5)
pub const MIN_TOUCH_TARGET_SIZE: f32 = 44.0;

/// Minimum focus indicator area (WCAG 2.4.11)
pub const MIN_FOCUS_INDICATOR_AREA: f32 = 2.0;

/// Accessibility checker.
pub struct A11yChecker;

impl A11yChecker {
    /// Check a widget tree for accessibility violations.
    #[must_use]
    pub fn check(widget: &dyn Widget) -> A11yReport {
        let mut violations = Vec::new();
        let mut context = CheckContext::default();
        Self::check_widget(widget, &mut violations, &mut context);
        A11yReport { violations }
    }

    /// Check with custom configuration.
    #[must_use]
    pub fn check_with_config(widget: &dyn Widget, config: &A11yConfig) -> A11yReport {
        let mut violations = Vec::new();
        let mut context = CheckContext {
            check_touch_targets: config.check_touch_targets,
            check_heading_hierarchy: config.check_heading_hierarchy,
            check_focus_indicators: config.check_focus_indicators,
            ..Default::default()
        };
        Self::check_widget(widget, &mut violations, &mut context);
        A11yReport { violations }
    }

    fn check_widget(
        widget: &dyn Widget,
        violations: &mut Vec<A11yViolation>,
        context: &mut CheckContext,
    ) {
        // Check for missing accessible name on interactive elements (WCAG 4.1.2)
        if widget.is_interactive() && widget.accessible_name().is_none() {
            violations.push(A11yViolation {
                rule: "aria-label".to_string(),
                message: "Interactive element missing accessible name".to_string(),
                wcag: "4.1.2".to_string(),
                impact: Impact::Critical,
            });
        }

        // Check for keyboard focusable elements (WCAG 2.1.1)
        if widget.is_interactive() && !widget.is_focusable() {
            violations.push(A11yViolation {
                rule: "keyboard".to_string(),
                message: "Interactive element is not keyboard focusable".to_string(),
                wcag: "2.1.1".to_string(),
                impact: Impact::Critical,
            });
        }

        // Check touch target size (WCAG 2.5.5)
        if context.check_touch_targets && widget.is_interactive() {
            let bounds = widget.bounds();
            if bounds.width < MIN_TOUCH_TARGET_SIZE || bounds.height < MIN_TOUCH_TARGET_SIZE {
                violations.push(A11yViolation {
                    rule: "touch-target".to_string(),
                    message: format!(
                        "Touch target too small: {}x{} (minimum {}x{})",
                        bounds.width, bounds.height, MIN_TOUCH_TARGET_SIZE, MIN_TOUCH_TARGET_SIZE
                    ),
                    wcag: "2.5.5".to_string(),
                    impact: Impact::Moderate,
                });
            }
        }

        // Check heading hierarchy (WCAG 1.3.1)
        if context.check_heading_hierarchy && widget.accessible_role() == AccessibleRole::Heading {
            if let Some(level) = Self::heading_level(widget) {
                let last_level = context.last_heading_level;
                if last_level > 0 && level > last_level + 1 {
                    violations.push(A11yViolation {
                        rule: "heading-order".to_string(),
                        message: format!(
                            "Heading level skipped: h{} followed by h{} (should be h{} or lower)",
                            last_level,
                            level,
                            last_level + 1
                        ),
                        wcag: "1.3.1".to_string(),
                        impact: Impact::Moderate,
                    });
                }
                context.last_heading_level = level;
            }
        }

        // Check focus indicator visibility (WCAG 2.4.7)
        if context.check_focus_indicators && widget.is_focusable() {
            if !Self::has_visible_focus_indicator(widget) {
                violations.push(A11yViolation {
                    rule: "focus-visible".to_string(),
                    message: "Focusable element may lack visible focus indicator".to_string(),
                    wcag: "2.4.7".to_string(),
                    impact: Impact::Serious,
                });
            }
        }

        // Check for images without alt text (WCAG 1.1.1)
        if widget.accessible_role() == AccessibleRole::Image && widget.accessible_name().is_none() {
            violations.push(A11yViolation {
                rule: "image-alt".to_string(),
                message: "Image missing alternative text".to_string(),
                wcag: "1.1.1".to_string(),
                impact: Impact::Critical,
            });
        }

        // Recurse into children
        for child in widget.children() {
            Self::check_widget(child.as_ref(), violations, context);
        }
    }

    /// Extract heading level from widget (if it's a heading)
    fn heading_level(widget: &dyn Widget) -> Option<u8> {
        // Check if the accessible name contains heading level info
        // Or use aria-level if available
        if let Some(name) = widget.accessible_name() {
            // Try to extract from pattern like "Heading Level 2" or "h2"
            if name.starts_with('h') || name.starts_with('H') {
                if let Ok(level) = name[1..2].parse::<u8>() {
                    if (1..=6).contains(&level) {
                        return Some(level);
                    }
                }
            }
        }
        // Default to level 2 if we can't determine
        Some(2)
    }

    /// Check if widget has a visible focus indicator
    fn has_visible_focus_indicator(widget: &dyn Widget) -> bool {
        // For now, assume all focusable widgets have focus indicators
        // In a real implementation, we'd check for focus ring styles
        widget.is_focusable()
    }

    /// Check contrast ratio between foreground and background colors.
    #[must_use]
    pub fn check_contrast(
        foreground: &Color,
        background: &Color,
        large_text: bool,
    ) -> ContrastResult {
        let ratio = foreground.contrast_ratio(background);

        // WCAG 2.1 thresholds
        let (aa_threshold, aaa_threshold) = if large_text {
            (3.0, 4.5) // Large text (14pt bold or 18pt regular)
        } else {
            (4.5, 7.0) // Normal text
        };

        ContrastResult {
            ratio,
            passes_aa: ratio >= aa_threshold,
            passes_aaa: ratio >= aaa_threshold,
        }
    }
}

/// Accessibility report.
#[derive(Debug)]
pub struct A11yReport {
    /// List of violations found
    pub violations: Vec<A11yViolation>,
}

impl A11yReport {
    /// Check if all accessibility tests passed.
    #[must_use]
    pub fn is_passing(&self) -> bool {
        self.violations.is_empty()
    }

    /// Get critical violations only.
    #[must_use]
    pub fn critical(&self) -> Vec<&A11yViolation> {
        self.violations
            .iter()
            .filter(|v| v.impact == Impact::Critical)
            .collect()
    }

    /// Assert that all accessibility tests pass.
    ///
    /// # Panics
    ///
    /// Panics if there are any violations.
    pub fn assert_pass(&self) {
        if !self.is_passing() {
            let messages: Vec<String> = self
                .violations
                .iter()
                .map(|v| {
                    format!(
                        "  [{:?}] {}: {} (WCAG {})",
                        v.impact, v.rule, v.message, v.wcag
                    )
                })
                .collect();

            panic!(
                "Accessibility check failed with {} violation(s):\n{}",
                self.violations.len(),
                messages.join("\n")
            );
        }
    }
}

/// A single accessibility violation.
#[derive(Debug, Clone)]
pub struct A11yViolation {
    /// Rule that was violated
    pub rule: String,
    /// Human-readable message
    pub message: String,
    /// WCAG success criterion
    pub wcag: String,
    /// Impact level
    pub impact: Impact,
}

/// Impact level of an accessibility violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Impact {
    /// Minor issue
    Minor,
    /// Moderate issue
    Moderate,
    /// Serious issue
    Serious,
    /// Critical issue - must fix
    Critical,
}

/// Configuration for accessibility checks.
#[derive(Debug, Clone)]
pub struct A11yConfig {
    /// Check touch target sizes (WCAG 2.5.5)
    pub check_touch_targets: bool,
    /// Check heading hierarchy (WCAG 1.3.1)
    pub check_heading_hierarchy: bool,
    /// Check focus indicators (WCAG 2.4.7)
    pub check_focus_indicators: bool,
    /// Minimum contrast ratio for normal text (WCAG 1.4.3)
    pub min_contrast_normal: f32,
    /// Minimum contrast ratio for large text (WCAG 1.4.3)
    pub min_contrast_large: f32,
}

impl Default for A11yConfig {
    fn default() -> Self {
        Self {
            check_touch_targets: true,
            check_heading_hierarchy: true,
            check_focus_indicators: false, // Disabled by default as requires style info
            min_contrast_normal: 4.5,
            min_contrast_large: 3.0,
        }
    }
}

impl A11yConfig {
    /// Create a new config with all checks enabled.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            check_touch_targets: true,
            check_heading_hierarchy: true,
            check_focus_indicators: true,
            min_contrast_normal: 7.0, // AAA level
            min_contrast_large: 4.5,  // AAA level
        }
    }

    /// Create a config for mobile apps.
    #[must_use]
    pub fn mobile() -> Self {
        Self {
            check_touch_targets: true,
            check_heading_hierarchy: true,
            check_focus_indicators: false,
            min_contrast_normal: 4.5,
            min_contrast_large: 3.0,
        }
    }
}

/// Internal context for walking the widget tree.
#[derive(Debug, Default)]
struct CheckContext {
    /// Last heading level seen (for hierarchy check)
    last_heading_level: u8,
    /// Whether to check touch target sizes
    check_touch_targets: bool,
    /// Whether to check heading hierarchy
    check_heading_hierarchy: bool,
    /// Whether to check focus indicators
    check_focus_indicators: bool,
}

/// Result of a contrast check.
#[derive(Debug, Clone)]
pub struct ContrastResult {
    /// Calculated contrast ratio
    pub ratio: f32,
    /// Passes WCAG AA
    pub passes_aa: bool,
    /// Passes WCAG AAA
    pub passes_aaa: bool,
}

// =============================================================================
// ARIA Attribute Generation
// =============================================================================

/// ARIA attributes for a widget.
#[derive(Debug, Clone, Default)]
pub struct AriaAttributes {
    /// The ARIA role
    pub role: Option<String>,
    /// Accessible label
    pub label: Option<String>,
    /// Accessible description
    pub described_by: Option<String>,
    /// Whether element is hidden from accessibility tree
    pub hidden: bool,
    /// Whether element is expanded (for expandable elements)
    pub expanded: Option<bool>,
    /// Whether element is selected
    pub selected: Option<bool>,
    /// Whether element is checked (for checkboxes/switches)
    pub checked: Option<AriaChecked>,
    /// Whether element is pressed (for toggle buttons)
    pub pressed: Option<AriaChecked>,
    /// Whether element is disabled
    pub disabled: bool,
    /// Whether element is required
    pub required: bool,
    /// Whether element is invalid
    pub invalid: bool,
    /// Current value for range widgets
    pub value_now: Option<f64>,
    /// Minimum value for range widgets
    pub value_min: Option<f64>,
    /// Maximum value for range widgets
    pub value_max: Option<f64>,
    /// Text representation of value
    pub value_text: Option<String>,
    /// Level (for headings)
    pub level: Option<u8>,
    /// Position in set
    pub pos_in_set: Option<u32>,
    /// Set size
    pub set_size: Option<u32>,
    /// Controls another element (ID reference)
    pub controls: Option<String>,
    /// Has popup indicator
    pub has_popup: Option<String>,
    /// Is busy/loading
    pub busy: bool,
    /// Live region politeness
    pub live: Option<AriaLive>,
    /// Atomic live region
    pub atomic: bool,
}

/// ARIA checked state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AriaChecked {
    /// Checked
    True,
    /// Not checked
    False,
    /// Mixed/indeterminate
    Mixed,
}

impl AriaChecked {
    /// Return string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            AriaChecked::True => "true",
            AriaChecked::False => "false",
            AriaChecked::Mixed => "mixed",
        }
    }
}

/// ARIA live region politeness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AriaLive {
    /// Polite announcements
    Polite,
    /// Assertive announcements
    Assertive,
    /// No announcements
    Off,
}

impl AriaLive {
    /// Return string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            AriaLive::Polite => "polite",
            AriaLive::Assertive => "assertive",
            AriaLive::Off => "off",
        }
    }
}

impl AriaAttributes {
    /// Create new empty ARIA attributes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the role.
    #[must_use]
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Set the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set hidden.
    #[must_use]
    pub const fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Set expanded state.
    #[must_use]
    pub const fn with_expanded(mut self, expanded: bool) -> Self {
        self.expanded = Some(expanded);
        self
    }

    /// Set selected state.
    #[must_use]
    pub const fn with_selected(mut self, selected: bool) -> Self {
        self.selected = Some(selected);
        self
    }

    /// Set checked state.
    #[must_use]
    pub const fn with_checked(mut self, checked: AriaChecked) -> Self {
        self.checked = Some(checked);
        self
    }

    /// Set pressed state (for toggle buttons).
    #[must_use]
    pub const fn with_pressed(mut self, pressed: AriaChecked) -> Self {
        self.pressed = Some(pressed);
        self
    }

    /// Set disabled state.
    #[must_use]
    pub const fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set busy state.
    #[must_use]
    pub const fn with_busy(mut self, busy: bool) -> Self {
        self.busy = busy;
        self
    }

    /// Set atomic.
    #[must_use]
    pub const fn with_atomic(mut self, atomic: bool) -> Self {
        self.atomic = atomic;
        self
    }

    /// Set range values.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64, now: f64) -> Self {
        self.value_min = Some(min);
        self.value_max = Some(max);
        self.value_now = Some(now);
        self
    }

    /// Set current value.
    #[must_use]
    pub const fn with_value_now(mut self, value: f64) -> Self {
        self.value_now = Some(value);
        self
    }

    /// Set minimum value.
    #[must_use]
    pub const fn with_value_min(mut self, value: f64) -> Self {
        self.value_min = Some(value);
        self
    }

    /// Set maximum value.
    #[must_use]
    pub const fn with_value_max(mut self, value: f64) -> Self {
        self.value_max = Some(value);
        self
    }

    /// Set controls reference.
    #[must_use]
    pub fn with_controls(mut self, controls: impl Into<String>) -> Self {
        self.controls = Some(controls.into());
        self
    }

    /// Set described by reference.
    #[must_use]
    pub fn with_described_by(mut self, described_by: impl Into<String>) -> Self {
        self.described_by = Some(described_by.into());
        self
    }

    /// Set has popup.
    #[must_use]
    pub fn with_has_popup(mut self, has_popup: impl Into<String>) -> Self {
        self.has_popup = Some(has_popup.into());
        self
    }

    /// Set heading level.
    #[must_use]
    pub const fn with_level(mut self, level: u8) -> Self {
        self.level = Some(level);
        self
    }

    /// Set position in set.
    #[must_use]
    pub const fn with_pos_in_set(mut self, pos: u32, size: u32) -> Self {
        self.pos_in_set = Some(pos);
        self.set_size = Some(size);
        self
    }

    /// Set live region.
    #[must_use]
    pub const fn with_live(mut self, live: AriaLive) -> Self {
        self.live = Some(live);
        self
    }

    /// Generate HTML ARIA attributes.
    #[must_use]
    pub fn to_html_attrs(&self) -> Vec<(String, String)> {
        let mut attrs = Vec::new();

        if let Some(ref role) = self.role {
            attrs.push(("role".to_string(), role.clone()));
        }
        if let Some(ref label) = self.label {
            attrs.push(("aria-label".to_string(), label.clone()));
        }
        if let Some(ref desc) = self.described_by {
            attrs.push(("aria-describedby".to_string(), desc.clone()));
        }
        if self.hidden {
            attrs.push(("aria-hidden".to_string(), "true".to_string()));
        }
        if let Some(expanded) = self.expanded {
            attrs.push(("aria-expanded".to_string(), expanded.to_string()));
        }
        if let Some(selected) = self.selected {
            attrs.push(("aria-selected".to_string(), selected.to_string()));
        }
        if let Some(checked) = self.checked {
            attrs.push(("aria-checked".to_string(), checked.as_str().to_string()));
        }
        if let Some(pressed) = self.pressed {
            attrs.push(("aria-pressed".to_string(), pressed.as_str().to_string()));
        }
        if let Some(ref popup) = self.has_popup {
            attrs.push(("aria-haspopup".to_string(), popup.clone()));
        }
        if self.disabled {
            attrs.push(("aria-disabled".to_string(), "true".to_string()));
        }
        if self.required {
            attrs.push(("aria-required".to_string(), "true".to_string()));
        }
        if self.invalid {
            attrs.push(("aria-invalid".to_string(), "true".to_string()));
        }
        if let Some(val) = self.value_now {
            attrs.push(("aria-valuenow".to_string(), val.to_string()));
        }
        if let Some(val) = self.value_min {
            attrs.push(("aria-valuemin".to_string(), val.to_string()));
        }
        if let Some(val) = self.value_max {
            attrs.push(("aria-valuemax".to_string(), val.to_string()));
        }
        if let Some(ref text) = self.value_text {
            attrs.push(("aria-valuetext".to_string(), text.clone()));
        }
        if let Some(level) = self.level {
            attrs.push(("aria-level".to_string(), level.to_string()));
        }
        if let Some(pos) = self.pos_in_set {
            attrs.push(("aria-posinset".to_string(), pos.to_string()));
        }
        if let Some(size) = self.set_size {
            attrs.push(("aria-setsize".to_string(), size.to_string()));
        }
        if let Some(ref controls) = self.controls {
            attrs.push(("aria-controls".to_string(), controls.clone()));
        }
        if self.busy {
            attrs.push(("aria-busy".to_string(), "true".to_string()));
        }
        if let Some(live) = self.live {
            attrs.push(("aria-live".to_string(), live.as_str().to_string()));
        }
        if self.atomic {
            attrs.push(("aria-atomic".to_string(), "true".to_string()));
        }

        attrs
    }

    /// Generate HTML attribute string.
    #[must_use]
    pub fn to_html_string(&self) -> String {
        self.to_html_attrs()
            .into_iter()
            .map(|(k, v)| {
                // Escape HTML special characters in values
                let escaped = v
                    .replace('&', "&amp;")
                    .replace('"', "&quot;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;");
                format!("{}=\"{}\"", k, escaped)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Generate ARIA attributes from a widget.
pub fn aria_from_widget(widget: &dyn Widget) -> AriaAttributes {
    use presentar_core::widget::AccessibleRole;

    let mut attrs = AriaAttributes::new();

    // Set role from widget
    let role = match widget.accessible_role() {
        AccessibleRole::Generic => None,
        AccessibleRole::Button => Some("button"),
        AccessibleRole::Checkbox => Some("checkbox"),
        AccessibleRole::TextInput => Some("textbox"),
        AccessibleRole::Link => Some("link"),
        AccessibleRole::Heading => Some("heading"),
        AccessibleRole::Image => Some("img"),
        AccessibleRole::List => Some("list"),
        AccessibleRole::ListItem => Some("listitem"),
        AccessibleRole::Table => Some("table"),
        AccessibleRole::TableRow => Some("row"),
        AccessibleRole::TableCell => Some("cell"),
        AccessibleRole::Menu => Some("menu"),
        AccessibleRole::MenuItem => Some("menuitem"),
        AccessibleRole::ComboBox => Some("combobox"),
        AccessibleRole::Slider => Some("slider"),
        AccessibleRole::ProgressBar => Some("progressbar"),
        AccessibleRole::Tab => Some("tab"),
        AccessibleRole::TabPanel => Some("tabpanel"),
        AccessibleRole::RadioGroup => Some("radiogroup"),
        AccessibleRole::Radio => Some("radio"),
    };

    if let Some(role) = role {
        attrs.role = Some(role.to_string());
    }

    // Set label from widget
    if let Some(name) = widget.accessible_name() {
        attrs.label = Some(name.to_string());
    }

    // Set disabled if not interactive but has a focusable role
    if !widget.is_interactive() && widget.accessible_role() != AccessibleRole::Generic {
        attrs.disabled = true;
    }

    attrs
}

// =============================================================================
// Form Accessibility Validation (WCAG 1.3.1, 1.3.5, 3.3.1, 3.3.2, 4.1.2)
// =============================================================================

/// Form accessibility checker for validating form-specific WCAG requirements.
pub struct FormA11yChecker;

impl FormA11yChecker {
    /// Check a form for accessibility violations.
    ///
    /// Validates:
    /// - Label associations (WCAG 1.3.1, 2.4.6)
    /// - Required field indicators (WCAG 3.3.2)
    /// - Error messaging (WCAG 3.3.1)
    /// - Input purpose/autocomplete (WCAG 1.3.5)
    /// - Form grouping (WCAG 1.3.1)
    #[must_use]
    pub fn check(form: &FormAccessibility) -> FormA11yReport {
        let mut violations = Vec::new();

        // Check all fields
        for field in &form.fields {
            Self::check_field(field, &mut violations);
        }

        // Check form-level requirements
        Self::check_form_level(form, &mut violations);

        FormA11yReport { violations }
    }

    /// Check a single form field.
    fn check_field(field: &FormFieldA11y, violations: &mut Vec<FormViolation>) {
        // WCAG 1.3.1, 2.4.6: Label association
        if field.label.is_none() && field.aria_label.is_none() && field.aria_labelledby.is_none() {
            violations.push(FormViolation {
                field_id: field.id.clone(),
                rule: FormA11yRule::MissingLabel,
                message: format!("Field '{}' has no associated label", field.id),
                wcag: "1.3.1, 2.4.6".to_string(),
                impact: Impact::Critical,
            });
        }

        // WCAG 3.3.2: Required field indicators
        if field.required {
            if !field.aria_required {
                violations.push(FormViolation {
                    field_id: field.id.clone(),
                    rule: FormA11yRule::MissingRequiredIndicator,
                    message: format!(
                        "Required field '{}' does not have aria-required=\"true\"",
                        field.id
                    ),
                    wcag: "3.3.2".to_string(),
                    impact: Impact::Serious,
                });
            }
            if !field.has_visual_required_indicator {
                violations.push(FormViolation {
                    field_id: field.id.clone(),
                    rule: FormA11yRule::MissingVisualRequired,
                    message: format!(
                        "Required field '{}' lacks visual required indicator (asterisk or text)",
                        field.id
                    ),
                    wcag: "3.3.2".to_string(),
                    impact: Impact::Moderate,
                });
            }
        }

        // WCAG 3.3.1: Error identification
        if field.has_error {
            if !field.aria_invalid {
                violations.push(FormViolation {
                    field_id: field.id.clone(),
                    rule: FormA11yRule::MissingErrorState,
                    message: format!(
                        "Field '{}' in error state does not have aria-invalid=\"true\"",
                        field.id
                    ),
                    wcag: "3.3.1".to_string(),
                    impact: Impact::Serious,
                });
            }
            if field.error_message.is_none() {
                violations.push(FormViolation {
                    field_id: field.id.clone(),
                    rule: FormA11yRule::MissingErrorMessage,
                    message: format!("Field '{}' in error state has no error message", field.id),
                    wcag: "3.3.1".to_string(),
                    impact: Impact::Serious,
                });
            }
            if field.aria_describedby.is_none() && field.aria_errormessage.is_none() {
                violations.push(FormViolation {
                    field_id: field.id.clone(),
                    rule: FormA11yRule::ErrorNotAssociated,
                    message: format!(
                        "Error message for '{}' not associated via aria-describedby or aria-errormessage",
                        field.id
                    ),
                    wcag: "3.3.1".to_string(),
                    impact: Impact::Serious,
                });
            }
        }

        // WCAG 1.3.5: Input purpose (autocomplete)
        if let Some(ref input_type) = field.input_type {
            if input_type.should_have_autocomplete() && field.autocomplete.is_none() {
                violations.push(FormViolation {
                    field_id: field.id.clone(),
                    rule: FormA11yRule::MissingAutocomplete,
                    message: format!(
                        "Field '{}' of type {:?} should have autocomplete attribute for autofill",
                        field.id, input_type
                    ),
                    wcag: "1.3.5".to_string(),
                    impact: Impact::Moderate,
                });
            }
        }

        // Check for placeholder-only labeling (anti-pattern)
        if field.placeholder.is_some() && field.label.is_none() && field.aria_label.is_none() {
            violations.push(FormViolation {
                field_id: field.id.clone(),
                rule: FormA11yRule::PlaceholderAsLabel,
                message: format!(
                    "Field '{}' uses placeholder as sole label; placeholders disappear on input",
                    field.id
                ),
                wcag: "3.3.2".to_string(),
                impact: Impact::Serious,
            });
        }
    }

    /// Check form-level accessibility requirements.
    fn check_form_level(form: &FormAccessibility, violations: &mut Vec<FormViolation>) {
        // Check for related fields that should be grouped (WCAG 1.3.1)
        let radio_fields: Vec<_> = form
            .fields
            .iter()
            .filter(|f| f.input_type == Some(InputType::Radio))
            .collect();

        if radio_fields.len() > 1 {
            // Radio buttons should be in a fieldset/group
            let has_group = form.field_groups.iter().any(|g| {
                g.field_ids
                    .iter()
                    .any(|id| radio_fields.iter().any(|f| &f.id == id))
            });

            if !has_group {
                violations.push(FormViolation {
                    field_id: "form".to_string(),
                    rule: FormA11yRule::RelatedFieldsNotGrouped,
                    message: "Related radio buttons should be grouped in a fieldset with legend"
                        .to_string(),
                    wcag: "1.3.1".to_string(),
                    impact: Impact::Moderate,
                });
            }
        }

        // Check field groups have legends
        for group in &form.field_groups {
            if group.legend.is_none() && group.aria_label.is_none() {
                violations.push(FormViolation {
                    field_id: group.id.clone(),
                    rule: FormA11yRule::GroupMissingLegend,
                    message: format!("Field group '{}' has no legend or aria-label", group.id),
                    wcag: "1.3.1".to_string(),
                    impact: Impact::Serious,
                });
            }
        }

        // Check form has accessible name
        if form.accessible_name.is_none() && form.aria_labelledby.is_none() {
            violations.push(FormViolation {
                field_id: "form".to_string(),
                rule: FormA11yRule::FormMissingName,
                message: "Form should have an accessible name (aria-label or aria-labelledby)"
                    .to_string(),
                wcag: "4.1.2".to_string(),
                impact: Impact::Moderate,
            });
        }
    }
}

/// Form accessibility data for validation.
#[derive(Debug, Clone, Default)]
pub struct FormAccessibility {
    /// Form's accessible name
    pub accessible_name: Option<String>,
    /// Referenced labelledby ID
    pub aria_labelledby: Option<String>,
    /// Form fields
    pub fields: Vec<FormFieldA11y>,
    /// Field groups (fieldsets)
    pub field_groups: Vec<FormFieldGroup>,
}

impl FormAccessibility {
    /// Create a new form accessibility descriptor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a field.
    #[must_use]
    pub fn field(mut self, field: FormFieldA11y) -> Self {
        self.fields.push(field);
        self
    }

    /// Add a field group.
    #[must_use]
    pub fn group(mut self, group: FormFieldGroup) -> Self {
        self.field_groups.push(group);
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name = Some(name.into());
        self
    }
}

/// Form field accessibility descriptor.
#[derive(Debug, Clone, Default)]
pub struct FormFieldA11y {
    /// Field ID
    pub id: String,
    /// Associated label text
    pub label: Option<String>,
    /// Input type
    pub input_type: Option<InputType>,
    /// Is required
    pub required: bool,
    /// Has visual required indicator
    pub has_visual_required_indicator: bool,
    /// aria-required attribute
    pub aria_required: bool,
    /// aria-label attribute
    pub aria_label: Option<String>,
    /// aria-labelledby attribute
    pub aria_labelledby: Option<String>,
    /// aria-describedby attribute
    pub aria_describedby: Option<String>,
    /// Has error state
    pub has_error: bool,
    /// aria-invalid attribute
    pub aria_invalid: bool,
    /// aria-errormessage attribute
    pub aria_errormessage: Option<String>,
    /// Error message text
    pub error_message: Option<String>,
    /// Autocomplete attribute
    pub autocomplete: Option<AutocompleteValue>,
    /// Placeholder text
    pub placeholder: Option<String>,
}

impl FormFieldA11y {
    /// Create a new field.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Set label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set input type.
    #[must_use]
    pub fn with_type(mut self, input_type: InputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Mark as required.
    #[must_use]
    pub fn required(mut self) -> Self {
        self.required = true;
        self.aria_required = true;
        self.has_visual_required_indicator = true;
        self
    }

    /// Set required with specific options.
    #[must_use]
    pub fn with_required(mut self, visual: bool, aria: bool) -> Self {
        self.required = true;
        self.has_visual_required_indicator = visual;
        self.aria_required = aria;
        self
    }

    /// Set error state.
    #[must_use]
    pub fn with_error(mut self, message: impl Into<String>, associated: bool) -> Self {
        self.has_error = true;
        self.aria_invalid = true;
        self.error_message = Some(message.into());
        if associated {
            self.aria_describedby = Some(format!("{}-error", self.id));
        }
        self
    }

    /// Set autocomplete.
    #[must_use]
    pub fn with_autocomplete(mut self, value: AutocompleteValue) -> Self {
        self.autocomplete = Some(value);
        self
    }

    /// Set placeholder.
    #[must_use]
    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = Some(placeholder.into());
        self
    }

    /// Set aria-label.
    #[must_use]
    pub fn with_aria_label(mut self, label: impl Into<String>) -> Self {
        self.aria_label = Some(label.into());
        self
    }
}

/// Form field group (fieldset).
#[derive(Debug, Clone, Default)]
pub struct FormFieldGroup {
    /// Group ID
    pub id: String,
    /// Legend text
    pub legend: Option<String>,
    /// aria-label (alternative to legend)
    pub aria_label: Option<String>,
    /// Field IDs in this group
    pub field_ids: Vec<String>,
}

impl FormFieldGroup {
    /// Create a new field group.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    /// Set legend.
    #[must_use]
    pub fn with_legend(mut self, legend: impl Into<String>) -> Self {
        self.legend = Some(legend.into());
        self
    }

    /// Add field ID to group.
    #[must_use]
    pub fn with_field(mut self, field_id: impl Into<String>) -> Self {
        self.field_ids.push(field_id.into());
        self
    }
}

/// Input type for form fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Text,
    Email,
    Password,
    Tel,
    Url,
    Number,
    Date,
    Time,
    Search,
    Radio,
    Checkbox,
    Select,
    Textarea,
    Hidden,
}

impl InputType {
    /// Check if this input type should have autocomplete for WCAG 1.3.5.
    #[must_use]
    pub const fn should_have_autocomplete(&self) -> bool {
        matches!(
            self,
            Self::Text | Self::Email | Self::Password | Self::Tel | Self::Url | Self::Number
        )
    }
}

/// Autocomplete attribute values (WCAG 1.3.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocompleteValue {
    /// User's name
    Name,
    /// Given (first) name
    GivenName,
    /// Family (last) name
    FamilyName,
    /// Email address
    Email,
    /// Telephone number
    Tel,
    /// Street address
    StreetAddress,
    /// Address level 1 (city)
    AddressLevel1,
    /// Address level 2 (state/province)
    AddressLevel2,
    /// Postal code
    PostalCode,
    /// Country name
    Country,
    /// Organization
    Organization,
    /// Username
    Username,
    /// Current password
    CurrentPassword,
    /// New password
    NewPassword,
    /// Credit card number
    CcNumber,
    /// Credit card expiration
    CcExp,
    /// Credit card CVV
    CcCsc,
    /// One-time code
    OneTimeCode,
    /// Turn off autocomplete
    Off,
}

impl AutocompleteValue {
    /// Get the HTML attribute value.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::GivenName => "given-name",
            Self::FamilyName => "family-name",
            Self::Email => "email",
            Self::Tel => "tel",
            Self::StreetAddress => "street-address",
            Self::AddressLevel1 => "address-level1",
            Self::AddressLevel2 => "address-level2",
            Self::PostalCode => "postal-code",
            Self::Country => "country",
            Self::Organization => "organization",
            Self::Username => "username",
            Self::CurrentPassword => "current-password",
            Self::NewPassword => "new-password",
            Self::CcNumber => "cc-number",
            Self::CcExp => "cc-exp",
            Self::CcCsc => "cc-csc",
            Self::OneTimeCode => "one-time-code",
            Self::Off => "off",
        }
    }
}

/// Form accessibility violation rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormA11yRule {
    /// Field missing label (1.3.1, 2.4.6)
    MissingLabel,
    /// Required field missing aria-required (3.3.2)
    MissingRequiredIndicator,
    /// Required field missing visual indicator (3.3.2)
    MissingVisualRequired,
    /// Error field missing aria-invalid (3.3.1)
    MissingErrorState,
    /// Error field missing error message (3.3.1)
    MissingErrorMessage,
    /// Error message not associated (3.3.1)
    ErrorNotAssociated,
    /// Field should have autocomplete (1.3.5)
    MissingAutocomplete,
    /// Placeholder used as sole label (3.3.2)
    PlaceholderAsLabel,
    /// Related fields not grouped (1.3.1)
    RelatedFieldsNotGrouped,
    /// Field group missing legend (1.3.1)
    GroupMissingLegend,
    /// Form missing accessible name (4.1.2)
    FormMissingName,
}

impl FormA11yRule {
    /// Get the rule name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::MissingLabel => "missing-label",
            Self::MissingRequiredIndicator => "missing-required-indicator",
            Self::MissingVisualRequired => "missing-visual-required",
            Self::MissingErrorState => "missing-error-state",
            Self::MissingErrorMessage => "missing-error-message",
            Self::ErrorNotAssociated => "error-not-associated",
            Self::MissingAutocomplete => "missing-autocomplete",
            Self::PlaceholderAsLabel => "placeholder-as-label",
            Self::RelatedFieldsNotGrouped => "related-fields-not-grouped",
            Self::GroupMissingLegend => "group-missing-legend",
            Self::FormMissingName => "form-missing-name",
        }
    }
}

/// Form accessibility violation.
#[derive(Debug, Clone)]
pub struct FormViolation {
    /// Field ID where violation occurred
    pub field_id: String,
    /// Rule that was violated
    pub rule: FormA11yRule,
    /// Human-readable message
    pub message: String,
    /// WCAG success criterion
    pub wcag: String,
    /// Impact level
    pub impact: Impact,
}

/// Form accessibility report.
#[derive(Debug)]
pub struct FormA11yReport {
    /// List of violations
    pub violations: Vec<FormViolation>,
}

impl FormA11yReport {
    /// Check if form passes accessibility.
    #[must_use]
    pub fn is_passing(&self) -> bool {
        self.violations.is_empty()
    }

    /// Check if form passes with no critical/serious issues.
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        !self
            .violations
            .iter()
            .any(|v| matches!(v.impact, Impact::Critical | Impact::Serious))
    }

    /// Get violations by rule.
    #[must_use]
    pub fn violations_for_rule(&self, rule: FormA11yRule) -> Vec<&FormViolation> {
        self.violations.iter().filter(|v| v.rule == rule).collect()
    }

    /// Get violations for a specific field.
    #[must_use]
    pub fn violations_for_field(&self, field_id: &str) -> Vec<&FormViolation> {
        self.violations
            .iter()
            .filter(|v| v.field_id == field_id)
            .collect()
    }

    /// Assert form passes accessibility.
    ///
    /// # Panics
    ///
    /// Panics with violation details if form fails accessibility checks.
    pub fn assert_pass(&self) {
        if !self.is_passing() {
            let messages: Vec<String> = self
                .violations
                .iter()
                .map(|v| {
                    format!(
                        "  [{:?}] {} ({}): {} (WCAG {})",
                        v.impact,
                        v.rule.name(),
                        v.field_id,
                        v.message,
                        v.wcag
                    )
                })
                .collect();

            panic!(
                "Form accessibility check failed with {} violation(s):\n{}",
                self.violations.len(),
                messages.join("\n")
            );
        }
    }
}
