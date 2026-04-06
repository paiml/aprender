//! Configuration panel brick (Layer 3)
//!
//! Displays current configuration, profile selection, and auto-save settings.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{Canvas, Color, Point, TextStyle};
use presentar_terminal::Theme;
use std::any::Any;

/// Configuration profile
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigProfile {
    /// Profile name
    pub name: String,
    /// Short description
    pub description: String,
    /// Is this the active profile
    pub is_active: bool,
}

impl ConfigProfile {
    /// Create a new profile
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into(), is_active: false }
    }

    /// Create a new active profile
    pub fn active(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self { name: name.into(), description: description.into(), is_active: true }
    }
}

/// Configuration panel for viewing and managing settings
pub struct ConfigPanelBrick {
    /// Configuration file path
    pub config_path: String,
    /// Available profiles
    pub profiles: Vec<ConfigProfile>,
    /// Selected profile index
    pub selected_index: usize,
    /// Auto-save on exit
    pub auto_save: bool,
    /// Load last profile on start
    pub load_last: bool,
    /// Theme for rendering
    pub theme: Theme,
}

impl ConfigPanelBrick {
    /// Create a new config panel
    pub fn new() -> Self {
        Self {
            config_path: "~/.config/cbtop/config.toml".to_string(),
            profiles: vec![
                ConfigProfile::active("inference", "LLM Inference"),
                ConfigProfile::new("ml_training", "ML Training"),
                ConfigProfile::new("stress_test", "Stress Testing"),
                ConfigProfile::new("power_saving", "Power Saving"),
            ],
            selected_index: 0,
            auto_save: true,
            load_last: true,
            theme: Theme::tokyo_night(),
        }
    }

    /// Get the active profile
    pub fn active_profile(&self) -> Option<&ConfigProfile> {
        self.profiles.iter().find(|p| p.is_active)
    }

    /// Select next profile
    pub fn next_profile(&mut self) {
        if !self.profiles.is_empty() {
            self.selected_index = (self.selected_index + 1) % self.profiles.len();
        }
    }

    /// Select previous profile
    pub fn prev_profile(&mut self) {
        if !self.profiles.is_empty() {
            self.selected_index =
                (self.selected_index + self.profiles.len() - 1) % self.profiles.len();
        }
    }

    /// Activate selected profile
    pub fn activate_selected(&mut self) {
        for (i, profile) in self.profiles.iter_mut().enumerate() {
            profile.is_active = i == self.selected_index;
        }
    }

    /// Toggle auto-save setting
    pub fn toggle_auto_save(&mut self) {
        self.auto_save = !self.auto_save;
    }

    /// Toggle load-last setting
    pub fn toggle_load_last(&mut self) {
        self.load_last = !self.load_last;
    }

    /// Paint the config panel
    pub fn paint(&self, canvas: &mut dyn Canvas, _width: f32, _height: f32) {
        let label_style = TextStyle { color: self.theme.foreground, ..Default::default() };
        let dim_style = TextStyle { color: self.theme.dim, ..Default::default() };
        let active_style = TextStyle {
            color: Color::new(0.3, 1.0, 0.5, 1.0), // Green for active
            ..Default::default()
        };
        let selected_style = TextStyle {
            color: Color::new(0.3, 0.8, 1.0, 1.0), // Cyan for selected
            ..Default::default()
        };

        canvas.draw_text("Configuration", Point::new(2.0, 2.0), &label_style);

        // Config file path
        canvas.draw_text("Config:", Point::new(2.0, 4.0), &dim_style);
        canvas.draw_text(&self.config_path, Point::new(10.0, 4.0), &label_style);

        // Active profile
        if let Some(profile) = self.active_profile() {
            canvas.draw_text("Profile:", Point::new(2.0, 5.0), &dim_style);
            canvas.draw_text(
                &format!("{} ({})", profile.name, profile.description),
                Point::new(10.0, 5.0),
                &active_style,
            );
        }

        // Settings checkboxes
        let auto_save_check = if self.auto_save { "[x]" } else { "[ ]" };
        let load_last_check = if self.load_last { "[x]" } else { "[ ]" };

        canvas.draw_text(
            &format!("{} Auto-save on exit", auto_save_check),
            Point::new(2.0, 7.0),
            &label_style,
        );
        canvas.draw_text(
            &format!("{} Load last profile on start", load_last_check),
            Point::new(2.0, 8.0),
            &label_style,
        );

        // Profile list
        canvas.draw_text("Profiles:", Point::new(2.0, 10.0), &dim_style);

        for (i, profile) in self.profiles.iter().enumerate() {
            let y = 11.0 + i as f32;
            let prefix = if i == self.selected_index { " > " } else { "   " };
            let suffix = if profile.is_active { " (active)" } else { "" };

            let style = if i == self.selected_index {
                &selected_style
            } else if profile.is_active {
                &active_style
            } else {
                &label_style
            };

            canvas.draw_text(
                &format!("{}{}{}", prefix, profile.name, suffix),
                Point::new(2.0, y),
                style,
            );
        }

        // Help text
        let help_y = 11.0 + self.profiles.len() as f32 + 2.0;
        canvas.draw_text("Press 'P' to activate profile", Point::new(2.0, help_y), &dim_style);
        canvas.draw_text(
            "Press 'S' to save current as new profile",
            Point::new(2.0, help_y + 1.0),
            &dim_style,
        );
    }
}

impl Default for ConfigPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for ConfigPanelBrick {
    fn brick_name(&self) -> &'static str {
        "config_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(45),
            BrickAssertion::MinHeight(18),
            BrickAssertion::max_latency_ms(8),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::FRAME_60FPS
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_config_panel_brick_name() {
        let panel = ConfigPanelBrick::new();
        assert_eq!(panel.brick_name(), "config_panel");
    }

    #[test]
    fn test_config_panel_paint_default() {
        let panel = ConfigPanelBrick::new();
        let mut canvas = RecordingCanvas::new();

        panel.paint(&mut canvas, 80.0, 24.0);

        // Should draw header, config path, profile, settings, and help
        assert!(!canvas.is_empty());
        assert!(canvas.command_count() >= 10);
    }

    #[test]
    fn test_config_panel_paint_profile_list() {
        let panel = ConfigPanelBrick::new();
        let mut canvas = RecordingCanvas::new();

        panel.paint(&mut canvas, 80.0, 24.0);

        // Should draw all 4 profiles + other UI elements
        // At minimum: header, config path, active profile, 2 checkboxes, "Profiles:" label,
        // 4 profile entries, 2 help lines = 12+ commands
        assert!(canvas.command_count() >= 10);
    }

    #[test]
    fn test_config_panel_paint_different_selections() {
        let mut panel = ConfigPanelBrick::new();

        // Test with each profile selected
        for i in 0..panel.profiles.len() {
            panel.selected_index = i;
            let mut canvas = RecordingCanvas::new();
            panel.paint(&mut canvas, 80.0, 24.0);
            assert!(!canvas.is_empty());
        }
    }

    #[test]
    fn test_config_panel_paint_with_settings_toggled() {
        let mut panel = ConfigPanelBrick::new();

        // Toggle settings off
        panel.toggle_auto_save();
        panel.toggle_load_last();

        let mut canvas = RecordingCanvas::new();
        panel.paint(&mut canvas, 80.0, 24.0);

        // Should still render correctly with different checkbox states
        assert!(!canvas.is_empty());
    }

    #[test]
    fn test_config_panel_paint_no_active_profile() {
        let mut panel = ConfigPanelBrick::new();

        // Deactivate all profiles
        for profile in &mut panel.profiles {
            profile.is_active = false;
        }

        let mut canvas = RecordingCanvas::new();
        panel.paint(&mut canvas, 80.0, 24.0);

        // Should still render without crashing
        assert!(!canvas.is_empty());
    }

    #[test]
    fn test_config_panel_has_assertions() {
        let panel = ConfigPanelBrick::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_config_panel_default_profiles() {
        let panel = ConfigPanelBrick::new();
        assert_eq!(panel.profiles.len(), 4);
        assert!(panel.active_profile().is_some());
        assert_eq!(panel.active_profile().unwrap().name, "inference");
    }

    #[test]
    fn test_profile_navigation() {
        let mut panel = ConfigPanelBrick::new();
        assert_eq!(panel.selected_index, 0);

        panel.next_profile();
        assert_eq!(panel.selected_index, 1);

        panel.prev_profile();
        assert_eq!(panel.selected_index, 0);

        panel.prev_profile();
        assert_eq!(panel.selected_index, 3); // Wrap around
    }

    #[test]
    fn test_activate_profile() {
        let mut panel = ConfigPanelBrick::new();

        panel.selected_index = 2;
        panel.activate_selected();

        assert!(!panel.profiles[0].is_active);
        assert!(!panel.profiles[1].is_active);
        assert!(panel.profiles[2].is_active);
        assert!(!panel.profiles[3].is_active);
    }

    #[test]
    fn test_toggle_settings() {
        let mut panel = ConfigPanelBrick::new();
        assert!(panel.auto_save);
        assert!(panel.load_last);

        panel.toggle_auto_save();
        assert!(!panel.auto_save);

        panel.toggle_load_last();
        assert!(!panel.load_last);
    }

    #[test]
    fn test_profile_creation() {
        let profile = ConfigProfile::new("test", "Test Profile");
        assert_eq!(profile.name, "test");
        assert_eq!(profile.description, "Test Profile");
        assert!(!profile.is_active);

        let active = ConfigProfile::active("active", "Active Profile");
        assert!(active.is_active);
    }
}
