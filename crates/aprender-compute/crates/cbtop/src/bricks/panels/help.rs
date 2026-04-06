//! Help panel brick (Layer 3)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{Canvas, Point, TextStyle};
use presentar_terminal::Theme;
use std::any::Any;

pub struct HelpPanelBrick {
    pub theme: Theme,
}

impl HelpPanelBrick {
    pub fn new() -> Self {
        Self { theme: Theme::tokyo_night() }
    }

    pub fn paint(&self, canvas: &mut dyn Canvas, _width: f32, _height: f32) {
        let title_style = TextStyle { color: self.theme.foreground, ..Default::default() };
        let key_style = TextStyle {
            color: self.theme.cpu.sample(0.3), // Use gradient for accent color
            ..Default::default()
        };
        let desc_style = TextStyle { color: self.theme.dim, ..Default::default() };

        canvas.draw_text("Help - Keyboard Controls", Point::new(2.0, 2.0), &title_style);

        let controls = [
            ("1-9", "Switch panels"),
            ("Space", "Start/Stop load generation"),
            ("+/-", "Increase/Decrease intensity"),
            ("b", "Cycle backend (SIMD/wgpu/CUDA/All)"),
            ("w", "Cycle workload type"),
            ("[/]", "Decrease/Increase problem size"),
            ("r", "Reset statistics"),
            ("q/Esc", "Quit"),
        ];

        for (i, (key, desc)) in controls.iter().enumerate() {
            let y = 4.0 + i as f32;
            canvas.draw_text(&format!("{:>8}", key), Point::new(4.0, y), &key_style);
            canvas.draw_text(&format!("  {}", desc), Point::new(12.0, y), &desc_style);
        }
    }
}

impl Brick for HelpPanelBrick {
    fn brick_name(&self) -> &'static str {
        "help_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::MinWidth(30), BrickAssertion::MinHeight(10)]
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

impl Default for HelpPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::RecordingCanvas;

    #[test]
    fn test_help_panel_brick_name() {
        let panel = HelpPanelBrick::new();
        assert_eq!(panel.brick_name(), "help_panel");
    }

    #[test]
    fn test_help_panel_has_assertions() {
        let panel = HelpPanelBrick::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_help_panel_paint() {
        let panel = HelpPanelBrick::new();
        let mut canvas = RecordingCanvas::new();

        panel.paint(&mut canvas, 80.0, 24.0);

        // Should draw title and all keyboard controls
        // 1 title + 8 controls (each has key + description)
        assert!(!canvas.is_empty());
        assert!(canvas.command_count() >= 10);
    }

    #[test]
    fn test_help_panel_default() {
        let panel = HelpPanelBrick::default();
        assert_eq!(panel.brick_name(), "help_panel");
    }

    #[test]
    fn test_help_panel_verify() {
        let panel = HelpPanelBrick::new();
        let verification = panel.verify();
        assert!(verification.is_valid());
    }

    #[test]
    fn test_help_panel_budget() {
        let panel = HelpPanelBrick::new();
        let budget = panel.budget();
        assert_eq!(budget.total_ms(), 16); // FRAME_60FPS
    }
}
