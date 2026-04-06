//! Layout system for the TUI monitor.

use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Layout preset configuration.
#[derive(Debug, Clone)]
pub struct Preset {
    /// Rows in the layout.
    pub rows: Vec<LayoutRow>,
}

/// A row in the layout.
#[derive(Debug, Clone)]
pub struct LayoutRow {
    /// Panels in this row.
    pub panels: Vec<String>,
    /// Height constraint.
    pub height: Constraint,
}

impl Preset {
    /// Default layout preset.
    #[must_use]
    pub fn default_preset() -> Self {
        Self {
            rows: vec![
                LayoutRow { panels: vec!["cpu".to_string()], height: Constraint::Percentage(30) },
                LayoutRow {
                    panels: vec!["memory".to_string()],
                    height: Constraint::Percentage(25),
                },
                LayoutRow {
                    panels: vec!["process".to_string()],
                    height: Constraint::Percentage(45),
                },
            ],
        }
    }

    /// Calculates the layout areas for the given terminal size.
    #[must_use]
    pub fn calculate(&self, area: Rect) -> Vec<Vec<Rect>> {
        let row_constraints: Vec<Constraint> = self.rows.iter().map(|r| r.height).collect();

        let row_areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints(row_constraints)
            .split(area);

        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let panel_count = row.panels.len();
                if panel_count == 0 {
                    return vec![];
                }

                let panel_constraints: Vec<Constraint> =
                    (0..panel_count).map(|_| Constraint::Ratio(1, panel_count as u32)).collect();

                Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(panel_constraints)
                    .split(row_areas[i])
                    .to_vec()
            })
            .collect()
    }
}

impl Default for Preset {
    fn default() -> Self {
        Self::default_preset()
    }
}

/// Layout manager with preset support.
#[derive(Debug, Clone)]
pub struct LayoutManager {
    /// Available presets.
    presets: Vec<Preset>,
    /// Current preset index.
    current: usize,
}

impl LayoutManager {
    /// Creates a new layout manager with default presets.
    #[must_use]
    pub fn new() -> Self {
        Self { presets: vec![Preset::default_preset()], current: 0 }
    }

    /// Switches to a preset by index.
    pub fn switch_to(&mut self, index: usize) {
        if index < self.presets.len() {
            self.current = index;
        }
    }

    /// Returns the current preset.
    #[must_use]
    pub fn current(&self) -> &Preset {
        &self.presets[self.current]
    }

    /// Adds a preset.
    pub fn add_preset(&mut self, preset: Preset) {
        self.presets.push(preset);
    }
}

impl Default for LayoutManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_default() {
        let preset = Preset::default_preset();
        assert_eq!(preset.rows.len(), 3);
    }

    #[test]
    fn test_preset_calculate() {
        let preset = Preset::default_preset();
        let area = Rect::new(0, 0, 100, 50);

        let areas = preset.calculate(area);

        assert_eq!(areas.len(), 3);
        assert!(!areas[0].is_empty());
    }

    #[test]
    fn test_layout_manager_switch() {
        let mut manager = LayoutManager::new();

        // Add another preset
        manager.add_preset(Preset {
            rows: vec![LayoutRow {
                panels: vec!["cpu".to_string(), "memory".to_string()],
                height: Constraint::Percentage(100),
            }],
        });

        manager.switch_to(1);
        assert_eq!(manager.current().rows.len(), 1);

        manager.switch_to(0);
        assert_eq!(manager.current().rows.len(), 3);
    }

    #[test]
    fn test_layout_manager_invalid_switch() {
        let mut manager = LayoutManager::new();
        manager.switch_to(999); // Should not panic
        assert_eq!(manager.current, 0);
    }

    #[test]
    fn test_preset_default_trait() {
        let preset = Preset::default();
        assert_eq!(preset.rows.len(), 3);
    }

    #[test]
    fn test_layout_manager_default_trait() {
        let manager = LayoutManager::default();
        assert_eq!(manager.current().rows.len(), 3);
    }

    #[test]
    fn test_preset_calculate_empty_row() {
        let preset = Preset {
            rows: vec![
                LayoutRow {
                    panels: vec![], // Empty row
                    height: Constraint::Percentage(50),
                },
                LayoutRow { panels: vec!["cpu".to_string()], height: Constraint::Percentage(50) },
            ],
        };
        let area = Rect::new(0, 0, 100, 50);
        let areas = preset.calculate(area);

        assert_eq!(areas.len(), 2);
        assert!(areas[0].is_empty()); // Empty row should produce empty vec
        assert!(!areas[1].is_empty()); // Non-empty row should have areas
    }

    #[test]
    fn test_preset_debug_clone() {
        let preset = Preset::default();
        let cloned = preset.clone();
        let _ = format!("{cloned:?}");
    }

    #[test]
    fn test_layout_row_debug_clone() {
        let row =
            LayoutRow { panels: vec!["test".to_string()], height: Constraint::Percentage(50) };
        let cloned = row.clone();
        let _ = format!("{cloned:?}");
    }

    #[test]
    fn test_layout_manager_debug_clone() {
        let manager = LayoutManager::new();
        let cloned = manager.clone();
        let _ = format!("{cloned:?}");
    }
}
