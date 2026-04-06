//! Presentar configuration validation.
//!
//! Validates presentar YAML configurations against the schema and business rules.

use super::schema::PresentarConfig;
use std::collections::HashSet;
use thiserror::Error;

/// Presentar validation errors.
#[derive(Debug, Clone, Error)]
pub enum PresentarError {
    /// Refresh rate is too low (below 16ms for 60 FPS).
    #[error("Invalid refresh rate: {0}ms (minimum 16ms for 60 FPS)")]
    InvalidRefreshRate(u32),

    /// Grid size is out of valid range (2-16).
    #[error("Invalid grid size: {0} (must be 2-16)")]
    InvalidGridSize(u8),

    /// Panel width is below minimum (10).
    #[error("Invalid panel width: {0} (minimum 10)")]
    InvalidPanelWidth(u16),

    /// Panel height is below minimum (3).
    #[error("Invalid panel height: {0} (minimum 3)")]
    InvalidPanelHeight(u16),

    /// Layout ratios don't sum to 1.0.
    #[error("Invalid layout ratio: top={0}, bottom={1} (must sum to 1.0)")]
    InvalidLayoutRatio(f32, f32),

    /// Same key is bound to multiple actions.
    #[error("Duplicate keybinding: '{0}' used for both {1} and {2}")]
    DuplicateKeybinding(char, String, String),

    /// Color is not in valid #RRGGBB format.
    #[error("Invalid color format: {0} (expected #RRGGBB)")]
    InvalidColorFormat(String),

    /// At least one panel must be enabled.
    #[error("No panels enabled")]
    NoPanelsEnabled,

    /// Sparkline history is out of valid range (1-3600 seconds).
    #[error("Invalid sparkline history: {0} (must be 1-3600 seconds)")]
    InvalidSparklineHistory(u32),

    /// Process column name is not recognized.
    #[error("Invalid process column: {0}")]
    InvalidProcessColumn(String),

    /// YAML parsing failed.
    #[error("YAML parse error: {0}")]
    ParseError(String),
}

/// Validation result with warnings.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Validation errors (fatal).
    pub errors: Vec<PresentarError>,
    /// Validation warnings (non-fatal).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Check if validation passed (no errors).
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Check if validation failed.
    pub fn is_err(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Add an error.
    pub fn add_error(&mut self, error: PresentarError) {
        self.errors.push(error);
    }

    /// Add a warning.
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
}

/// Validate a presentar configuration.
pub fn validate_config(config: &PresentarConfig) -> ValidationResult {
    let mut result = ValidationResult::default();

    // Validate refresh rate (minimum 16ms for 60 FPS)
    if config.refresh_ms < 16 {
        result.add_error(PresentarError::InvalidRefreshRate(config.refresh_ms));
    } else if config.refresh_ms < 100 {
        result.add_warning(format!(
            "Refresh rate {}ms may cause high CPU usage",
            config.refresh_ms
        ));
    }

    // Validate grid size
    if config.layout.grid_size < 2 || config.layout.grid_size > 16 {
        result.add_error(PresentarError::InvalidGridSize(config.layout.grid_size));
    }

    // Validate panel dimensions
    if config.layout.min_panel_width < 10 {
        result.add_error(PresentarError::InvalidPanelWidth(
            config.layout.min_panel_width,
        ));
    }
    if config.layout.min_panel_height < 3 {
        result.add_error(PresentarError::InvalidPanelHeight(
            config.layout.min_panel_height,
        ));
    }

    // Validate layout ratios
    let ratio_sum = config.layout.top_height + config.layout.bottom_height;
    if (ratio_sum - 1.0).abs() > 0.01 {
        result.add_error(PresentarError::InvalidLayoutRatio(
            config.layout.top_height,
            config.layout.bottom_height,
        ));
    }

    // Validate at least one panel enabled
    let enabled_count = config
        .panels
        .iter_enabled()
        .iter()
        .filter(|(_, e)| *e)
        .count();
    if enabled_count == 0 {
        result.add_error(PresentarError::NoPanelsEnabled);
    }

    // Validate keybindings (no duplicates)
    validate_keybindings(config, &mut result);

    // Validate colors
    validate_colors(config, &mut result);

    // Validate sparkline history
    if config.panels.cpu.sparkline_history == 0 || config.panels.cpu.sparkline_history > 3600 {
        result.add_error(PresentarError::InvalidSparklineHistory(
            config.panels.cpu.sparkline_history,
        ));
    }

    // Validate process columns
    validate_process_columns(config, &mut result);

    result
}

fn validate_keybindings(config: &PresentarConfig, result: &mut ValidationResult) {
    let kb = &config.keybindings;
    let mut seen: HashSet<char> = HashSet::new();
    let bindings = [
        (kb.quit, "quit"),
        (kb.help, "help"),
        (kb.toggle_fps, "toggle_fps"),
        (kb.filter, "filter"),
        (kb.sort_cpu, "sort_cpu"),
        (kb.sort_mem, "sort_mem"),
        (kb.sort_pid, "sort_pid"),
        (kb.kill_process, "kill_process"),
    ];

    for (key, name) in bindings {
        if !seen.insert(key) {
            // Find the first binding with this key
            let first = bindings
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, n)| *n)
                .unwrap_or("unknown");
            result.add_error(PresentarError::DuplicateKeybinding(
                key,
                first.to_string(),
                name.to_string(),
            ));
        }
    }
}

fn validate_colors(config: &PresentarConfig, result: &mut ValidationResult) {
    for (panel, color) in config.theme.iter_panel_colors() {
        if !is_valid_hex_color(color) {
            result.add_error(PresentarError::InvalidColorFormat(format!(
                "{}: {}",
                panel.key(),
                color
            )));
        }
    }
}

fn is_valid_hex_color(color: &str) -> bool {
    if !color.starts_with('#') {
        return false;
    }
    let hex = &color[1..];
    hex.len() == 6 && hex.chars().all(|c| c.is_ascii_hexdigit())
}

fn validate_process_columns(config: &PresentarConfig, result: &mut ValidationResult) {
    let valid_columns = ["pid", "user", "cpu", "mem", "cmd", "state", "time", "name"];
    for col in &config.panels.process.columns {
        if !valid_columns.contains(&col.as_str()) {
            result.add_error(PresentarError::InvalidProcessColumn(col.clone()));
        }
    }
}

/// Parse and validate YAML in one step.
pub fn parse_and_validate(
    yaml: &str,
) -> Result<(PresentarConfig, ValidationResult), PresentarError> {
    let config =
        PresentarConfig::from_yaml(yaml).map_err(|e| PresentarError::ParseError(e.to_string()))?;
    let result = validate_config(&config);
    Ok((config, result))
}

#[cfg(test)]
mod tests {
    use super::super::schema::PanelType;
    use super::*;

    #[test]
    fn test_valid_config() {
        let config = PresentarConfig::default();
        let result = validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_refresh_rate() {
        let mut config = PresentarConfig::default();
        config.refresh_ms = 5;
        let result = validate_config(&config);
        assert!(result.is_err());
        assert!(matches!(
            &result.errors[0],
            PresentarError::InvalidRefreshRate(5)
        ));
    }

    #[test]
    fn test_low_refresh_rate_warning() {
        let mut config = PresentarConfig::default();
        config.refresh_ms = 50;
        let result = validate_config(&config);
        assert!(result.is_ok());
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_invalid_grid_size() {
        let mut config = PresentarConfig::default();
        config.layout.grid_size = 1;
        let result = validate_config(&config);
        assert!(result.is_err());

        config.layout.grid_size = 20;
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_panel_dimensions() {
        let mut config = PresentarConfig::default();
        config.layout.min_panel_width = 5;
        let result = validate_config(&config);
        assert!(result.is_err());

        let mut config = PresentarConfig::default();
        config.layout.min_panel_height = 2;
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_layout_ratio() {
        let mut config = PresentarConfig::default();
        config.layout.top_height = 0.3;
        config.layout.bottom_height = 0.3;
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_panels_enabled() {
        let mut config = PresentarConfig::default();
        for panel in PanelType::all() {
            config.panels.set_enabled(*panel, false);
        }
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_keybinding() {
        let mut config = PresentarConfig::default();
        config.keybindings.quit = 'q';
        config.keybindings.help = 'q'; // Duplicate
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_color_format() {
        let mut config = PresentarConfig::default();
        config
            .theme
            .panel_colors
            .insert("cpu".into(), "invalid".into());
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_hex_color() {
        assert!(is_valid_hex_color("#64C8FF"));
        assert!(is_valid_hex_color("#000000"));
        assert!(is_valid_hex_color("#FFFFFF"));
        assert!(!is_valid_hex_color("64C8FF")); // Missing #
        assert!(!is_valid_hex_color("#64C8F")); // Too short
        assert!(!is_valid_hex_color("#64C8FFF")); // Too long
        assert!(!is_valid_hex_color("#GGGGGG")); // Invalid hex
    }

    #[test]
    fn test_invalid_process_column() {
        let mut config = PresentarConfig::default();
        config.panels.process.columns.push("invalid_col".into());
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_sparkline_history() {
        let mut config = PresentarConfig::default();
        config.panels.cpu.sparkline_history = 0;
        let result = validate_config(&config);
        assert!(result.is_err());

        let mut config = PresentarConfig::default();
        config.panels.cpu.sparkline_history = 5000;
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_and_validate() {
        let yaml = "refresh_ms: 1000";
        let (config, result) = parse_and_validate(yaml).unwrap();
        assert_eq!(config.refresh_ms, 1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_and_validate_invalid_yaml() {
        let yaml = "invalid: yaml: {{{{";
        let result = parse_and_validate(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_result_methods() {
        let mut result = ValidationResult::default();
        assert!(result.is_ok());
        assert!(!result.is_err());

        result.add_error(PresentarError::InvalidRefreshRate(5));
        assert!(result.is_err());
        assert!(!result.is_ok());

        result.add_warning("test warning");
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_error_display() {
        let err = PresentarError::InvalidRefreshRate(5);
        assert!(err.to_string().contains("16ms"));

        let err = PresentarError::InvalidGridSize(1);
        assert!(err.to_string().contains("2-16"));

        let err = PresentarError::DuplicateKeybinding('q', "quit".into(), "help".into());
        assert!(err.to_string().contains("'q'"));
    }
}
