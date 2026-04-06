//! Layout presets for the TUI monitor.
//!
//! Provides predefined layouts that can be selected with keys 0-9.

use crate::monitor::layout::{LayoutRow, Preset};
use ratatui::layout::Constraint;

/// Creates the default layout preset (system overview).
///
/// Layout:
/// ```text
/// ┌─────────────────────────────────────┐
/// │           CPU (30%)                 │
/// ├─────────────────────────────────────┤
/// │          Memory (25%)               │
/// ├─────────────────────────────────────┤
/// │         Processes (45%)             │
/// └─────────────────────────────────────┘
/// ```
#[must_use]
pub fn preset_default() -> Preset {
    Preset {
        rows: vec![
            LayoutRow { panels: vec!["cpu".to_string()], height: Constraint::Percentage(30) },
            LayoutRow { panels: vec!["memory".to_string()], height: Constraint::Percentage(25) },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(45) },
        ],
    }
}

/// Creates the full system layout preset.
///
/// Layout:
/// ```text
/// ┌──────────────┬──────────────┐
/// │   CPU (25%)  │   GPU (25%)  │
/// ├──────────────┼──────────────┤
/// │  Memory (20%)│ Network (20%)│
/// ├──────────────┴──────────────┤
/// │     Processes (55%)         │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_full_system() -> Preset {
    Preset {
        rows: vec![
            LayoutRow {
                panels: vec!["cpu".to_string(), "gpu".to_string()],
                height: Constraint::Percentage(25),
            },
            LayoutRow {
                panels: vec!["memory".to_string(), "network".to_string()],
                height: Constraint::Percentage(20),
            },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(55) },
        ],
    }
}

/// Creates the ML-focused layout preset.
///
/// Layout:
/// ```text
/// ┌──────────────┬──────────────┐
/// │   LLM (40%)  │ Training(40%)│
/// ├──────────────┼──────────────┤
/// │   GPU (30%)  │   ZRAM (30%) │
/// ├──────────────┴──────────────┤
/// │       Repartir (30%)        │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_ml() -> Preset {
    Preset {
        rows: vec![
            LayoutRow {
                panels: vec!["llm".to_string(), "training".to_string()],
                height: Constraint::Percentage(40),
            },
            LayoutRow {
                panels: vec!["gpu".to_string(), "zram".to_string()],
                height: Constraint::Percentage(30),
            },
            LayoutRow { panels: vec!["repartir".to_string()], height: Constraint::Percentage(30) },
        ],
    }
}

/// Creates the network-focused layout preset.
///
/// Layout:
/// ```text
/// ┌─────────────────────────────┐
/// │        Network (40%)        │
/// ├─────────────────────────────┤
/// │         Disk (30%)          │
/// ├─────────────────────────────┤
/// │       Processes (30%)       │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_network() -> Preset {
    Preset {
        rows: vec![
            LayoutRow { panels: vec!["network".to_string()], height: Constraint::Percentage(40) },
            LayoutRow { panels: vec!["disk".to_string()], height: Constraint::Percentage(30) },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(30) },
        ],
    }
}

/// Creates the process-focused layout preset.
///
/// Layout:
/// ```text
/// ┌──────────────┬──────────────┐
/// │   CPU (20%)  │ Memory (20%) │
/// ├──────────────┴──────────────┤
/// │       Processes (80%)       │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_process() -> Preset {
    Preset {
        rows: vec![
            LayoutRow {
                panels: vec!["cpu".to_string(), "memory".to_string()],
                height: Constraint::Percentage(20),
            },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(80) },
        ],
    }
}

/// Creates the GPU-focused layout preset.
///
/// Layout:
/// ```text
/// ┌─────────────────────────────┐
/// │          GPU (50%)          │
/// ├──────────────┬──────────────┤
/// │   CPU (25%)  │ Memory (25%) │
/// ├──────────────┴──────────────┤
/// │       Processes (25%)       │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_gpu() -> Preset {
    Preset {
        rows: vec![
            LayoutRow { panels: vec!["gpu".to_string()], height: Constraint::Percentage(50) },
            LayoutRow {
                panels: vec!["cpu".to_string(), "memory".to_string()],
                height: Constraint::Percentage(25),
            },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(25) },
        ],
    }
}

/// Creates the sensors layout preset.
///
/// Layout:
/// ```text
/// ┌──────────────┬──────────────┐
/// │   CPU (30%)  │ Sensors (30%)│
/// ├──────────────┼──────────────┤
/// │  Memory (20%)│ Battery (20%)│
/// ├──────────────┴──────────────┤
/// │       Processes (50%)       │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_sensors() -> Preset {
    Preset {
        rows: vec![
            LayoutRow {
                panels: vec!["cpu".to_string(), "sensors".to_string()],
                height: Constraint::Percentage(30),
            },
            LayoutRow {
                panels: vec!["memory".to_string(), "battery".to_string()],
                height: Constraint::Percentage(20),
            },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(50) },
        ],
    }
}

/// Creates the compact layout preset (minimal info).
///
/// Layout:
/// ```text
/// ┌──────────────┬──────────────┐
/// │   CPU (40%)  │ Memory (40%) │
/// ├──────────────┼──────────────┤
/// │Network (40%) │  Disk (40%)  │
/// ├──────────────┴──────────────┤
/// │       Processes (20%)       │
/// └─────────────────────────────┘
/// ```
#[must_use]
pub fn preset_compact() -> Preset {
    Preset {
        rows: vec![
            LayoutRow {
                panels: vec!["cpu".to_string(), "memory".to_string()],
                height: Constraint::Percentage(40),
            },
            LayoutRow {
                panels: vec!["network".to_string(), "disk".to_string()],
                height: Constraint::Percentage(40),
            },
            LayoutRow { panels: vec!["process".to_string()], height: Constraint::Percentage(20) },
        ],
    }
}

/// Returns all predefined presets indexed by hotkey (0-9).
#[must_use]
pub fn all_presets() -> Vec<Preset> {
    vec![
        preset_default(),     // 0 - Default
        preset_full_system(), // 1 - Full system
        preset_ml(),          // 2 - ML focused
        preset_network(),     // 3 - Network focused
        preset_process(),     // 4 - Process focused
        preset_gpu(),         // 5 - GPU focused
        preset_sensors(),     // 6 - Sensors
        preset_compact(),     // 7 - Compact
        preset_default(),     // 8 - Reserved
        preset_default(),     // 9 - Reserved
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_default() {
        let preset = preset_default();
        assert_eq!(preset.rows.len(), 3);
        assert_eq!(preset.rows[0].panels[0], "cpu");
    }

    #[test]
    fn test_preset_full_system() {
        let preset = preset_full_system();
        assert_eq!(preset.rows.len(), 3);
        assert_eq!(preset.rows[0].panels.len(), 2);
    }

    #[test]
    fn test_preset_ml() {
        let preset = preset_ml();
        assert_eq!(preset.rows.len(), 3);
        assert!(preset.rows[0].panels.contains(&"llm".to_string()));
    }

    #[test]
    fn test_preset_network() {
        let preset = preset_network();
        assert_eq!(preset.rows[0].panels[0], "network");
    }

    #[test]
    fn test_preset_process() {
        let preset = preset_process();
        assert_eq!(preset.rows.len(), 2);
        assert_eq!(preset.rows[1].panels[0], "process");
    }

    #[test]
    fn test_preset_gpu() {
        let preset = preset_gpu();
        assert_eq!(preset.rows[0].panels[0], "gpu");
    }

    #[test]
    fn test_preset_sensors() {
        let preset = preset_sensors();
        assert!(preset.rows[0].panels.contains(&"sensors".to_string()));
    }

    #[test]
    fn test_preset_compact() {
        let preset = preset_compact();
        assert_eq!(preset.rows.len(), 3);
    }

    #[test]
    fn test_all_presets() {
        let presets = all_presets();
        assert_eq!(presets.len(), 10);
    }

    #[test]
    fn test_presets_have_valid_constraints() {
        for preset in all_presets() {
            let total: u16 =
                preset
                    .rows
                    .iter()
                    .filter_map(|r| {
                        if let Constraint::Percentage(p) = r.height {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .sum();

            assert_eq!(total, 100, "Preset rows should sum to 100%");
        }
    }
}
