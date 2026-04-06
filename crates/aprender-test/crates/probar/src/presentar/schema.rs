//! Presentar YAML schema types.
//!
//! Defines the schema for ptop configuration files following the presentar spec.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Presentar configuration schema (ptop.yaml).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresentarConfig {
    /// Refresh interval in milliseconds (min: 16 for 60 FPS).
    #[serde(default = "default_refresh_ms")]
    pub refresh_ms: u32,

    /// Layout configuration.
    #[serde(default)]
    pub layout: LayoutConfig,

    /// Panel configurations.
    #[serde(default)]
    pub panels: PanelConfigs,

    /// Keybindings.
    #[serde(default)]
    pub keybindings: KeybindingConfig,

    /// Theme configuration.
    #[serde(default)]
    pub theme: ThemeConfig,
}

impl Default for PresentarConfig {
    fn default() -> Self {
        Self {
            refresh_ms: default_refresh_ms(),
            layout: LayoutConfig::default(),
            panels: PanelConfigs::default(),
            keybindings: KeybindingConfig::default(),
            theme: ThemeConfig::default(),
        }
    }
}

fn default_refresh_ms() -> u32 {
    1000
}

/// Layout configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Snap panels to grid.
    #[serde(default = "default_true")]
    pub snap_to_grid: bool,

    /// Grid size for snapping.
    #[serde(default = "default_grid_size")]
    pub grid_size: u8,

    /// Minimum panel width in columns.
    #[serde(default = "default_min_panel_width")]
    pub min_panel_width: u16,

    /// Minimum panel height in rows.
    #[serde(default = "default_min_panel_height")]
    pub min_panel_height: u16,

    /// Top section height ratio (0.0-1.0).
    #[serde(default = "default_top_height")]
    pub top_height: f32,

    /// Bottom section height ratio (0.0-1.0).
    #[serde(default = "default_bottom_height")]
    pub bottom_height: f32,

    /// Border style.
    #[serde(default)]
    pub border_style: BorderStyle,

    /// Content padding.
    #[serde(default = "default_padding")]
    pub padding: u8,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            snap_to_grid: true,
            grid_size: 4,
            min_panel_width: 30,
            min_panel_height: 6,
            top_height: 0.45,
            bottom_height: 0.55,
            border_style: BorderStyle::Rounded,
            padding: 1,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_grid_size() -> u8 {
    4
}
fn default_min_panel_width() -> u16 {
    30
}
fn default_min_panel_height() -> u16 {
    6
}
fn default_top_height() -> f32 {
    0.45
}
fn default_bottom_height() -> f32 {
    0.55
}
fn default_padding() -> u8 {
    1
}

/// Border style enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BorderStyle {
    /// Rounded corners (btop-style).
    #[default]
    Rounded,
    /// Sharp corners.
    Sharp,
    /// Double-line borders.
    Double,
    /// No borders.
    None,
}

/// Panel configurations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PanelConfigs {
    /// CPU panel configuration.
    #[serde(default)]
    pub cpu: PanelConfig,
    /// Memory panel configuration.
    #[serde(default)]
    pub memory: PanelConfig,
    /// Disk panel configuration.
    #[serde(default)]
    pub disk: PanelConfig,
    /// Network panel configuration.
    #[serde(default)]
    pub network: PanelConfig,
    /// Process panel configuration.
    #[serde(default)]
    pub process: ProcessPanelConfig,
    /// GPU panel configuration.
    #[serde(default)]
    pub gpu: PanelConfig,
    /// Battery panel configuration.
    #[serde(default)]
    pub battery: PanelConfig,
    /// Sensors panel configuration.
    #[serde(default)]
    pub sensors: PanelConfig,
    /// PSI panel configuration.
    #[serde(default)]
    pub psi: PanelConfig,
    /// Connections panel configuration.
    #[serde(default)]
    pub connections: PanelConfig,
    /// Files panel configuration.
    #[serde(default)]
    pub files: PanelConfig,
}

impl PanelConfigs {
    /// Iterate over all panels with their enabled status.
    pub fn iter_enabled(&self) -> Vec<(PanelType, bool)> {
        vec![
            (PanelType::Cpu, self.cpu.enabled),
            (PanelType::Memory, self.memory.enabled),
            (PanelType::Disk, self.disk.enabled),
            (PanelType::Network, self.network.enabled),
            (PanelType::Process, self.process.enabled),
            (PanelType::Gpu, self.gpu.enabled),
            (PanelType::Battery, self.battery.enabled),
            (PanelType::Sensors, self.sensors.enabled),
            (PanelType::Psi, self.psi.enabled),
            (PanelType::Connections, self.connections.enabled),
            (PanelType::Files, self.files.enabled),
        ]
    }

    /// Set enabled status for a panel.
    pub fn set_enabled(&mut self, panel: PanelType, enabled: bool) {
        match panel {
            PanelType::Cpu => self.cpu.enabled = enabled,
            PanelType::Memory => self.memory.enabled = enabled,
            PanelType::Disk => self.disk.enabled = enabled,
            PanelType::Network => self.network.enabled = enabled,
            PanelType::Process => self.process.enabled = enabled,
            PanelType::Gpu => self.gpu.enabled = enabled,
            PanelType::Battery => self.battery.enabled = enabled,
            PanelType::Sensors => self.sensors.enabled = enabled,
            PanelType::Psi => self.psi.enabled = enabled,
            PanelType::Connections => self.connections.enabled = enabled,
            PanelType::Files => self.files.enabled = enabled,
            _ => {}
        }
    }
}

/// Generic panel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelConfig {
    /// Whether the panel is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Histogram style.
    #[serde(default)]
    pub histogram: HistogramStyle,

    /// Show temperature.
    #[serde(default)]
    pub show_temperature: bool,

    /// Show frequency.
    #[serde(default)]
    pub show_frequency: bool,

    /// Sparkline history in seconds.
    #[serde(default = "default_sparkline_history")]
    pub sparkline_history: u32,
}

impl Default for PanelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            histogram: HistogramStyle::Braille,
            show_temperature: true,
            show_frequency: true,
            sparkline_history: 60,
        }
    }
}

fn default_sparkline_history() -> u32 {
    60
}

/// Process panel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessPanelConfig {
    /// Whether the panel is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum processes to display.
    #[serde(default = "default_max_processes")]
    pub max_processes: u32,

    /// Columns to display.
    #[serde(default = "default_columns")]
    pub columns: Vec<String>,
}

impl Default for ProcessPanelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_processes: 20,
            columns: default_columns(),
        }
    }
}

fn default_max_processes() -> u32 {
    20
}

fn default_columns() -> Vec<String> {
    vec![
        "pid".into(),
        "user".into(),
        "cpu".into(),
        "mem".into(),
        "cmd".into(),
    ]
}

/// Histogram style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HistogramStyle {
    /// Braille characters (high resolution).
    #[default]
    Braille,
    /// Block characters.
    Block,
    /// ASCII characters.
    Ascii,
}

/// Keybinding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeybindingConfig {
    /// Key to quit the application.
    #[serde(default = "default_quit")]
    pub quit: char,
    /// Key to show help.
    #[serde(default = "default_help")]
    pub help: char,
    /// Key to toggle FPS display.
    #[serde(default = "default_toggle_fps")]
    pub toggle_fps: char,
    /// Key to filter processes.
    #[serde(default = "default_filter")]
    pub filter: char,
    /// Key to sort by CPU usage.
    #[serde(default = "default_sort_cpu")]
    pub sort_cpu: char,
    /// Key to sort by memory usage.
    #[serde(default = "default_sort_mem")]
    pub sort_mem: char,
    /// Key to sort by PID.
    #[serde(default = "default_sort_pid")]
    pub sort_pid: char,
    /// Key to kill selected process.
    #[serde(default = "default_kill")]
    pub kill_process: char,
    /// Key to explode (expand) a panel.
    #[serde(default = "default_explode")]
    pub explode: String,
    /// Key to collapse a panel.
    #[serde(default = "default_collapse")]
    pub collapse: String,
    /// Key to navigate between panels.
    #[serde(default = "default_navigate")]
    pub navigate: String,
}

impl Default for KeybindingConfig {
    fn default() -> Self {
        Self {
            quit: 'q',
            help: '?',
            toggle_fps: 'f',
            filter: '/',
            sort_cpu: 'c',
            sort_mem: 'm',
            sort_pid: 'p',
            kill_process: 'k',
            explode: "Enter".into(),
            collapse: "Escape".into(),
            navigate: "Tab".into(),
        }
    }
}

fn default_quit() -> char {
    'q'
}
fn default_help() -> char {
    '?'
}
fn default_toggle_fps() -> char {
    'f'
}
fn default_filter() -> char {
    '/'
}
fn default_sort_cpu() -> char {
    'c'
}
fn default_sort_mem() -> char {
    'm'
}
fn default_sort_pid() -> char {
    'p'
}
fn default_kill() -> char {
    'k'
}
fn default_explode() -> String {
    "Enter".into()
}
fn default_collapse() -> String {
    "Escape".into()
}
fn default_navigate() -> String {
    "Tab".into()
}

/// Theme configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Panel border colors (hex).
    #[serde(default)]
    pub panel_colors: HashMap<String, String>,

    /// High contrast mode.
    #[serde(default)]
    pub high_contrast: bool,

    /// Colorblind-safe palette.
    #[serde(default)]
    pub colorblind_safe: bool,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        let mut panel_colors = HashMap::new();
        panel_colors.insert("cpu".into(), "#64C8FF".into());
        panel_colors.insert("memory".into(), "#B478FF".into());
        panel_colors.insert("disk".into(), "#64B4FF".into());
        panel_colors.insert("network".into(), "#FF9664".into());
        panel_colors.insert("process".into(), "#DCC464".into());
        panel_colors.insert("gpu".into(), "#64FF96".into());
        panel_colors.insert("battery".into(), "#FFDC64".into());
        panel_colors.insert("sensors".into(), "#FF6496".into());
        panel_colors.insert("psi".into(), "#C85050".into());
        panel_colors.insert("connections".into(), "#78B4DC".into());
        panel_colors.insert("files".into(), "#B48C64".into());

        Self {
            panel_colors,
            high_contrast: false,
            colorblind_safe: false,
        }
    }
}

impl ThemeConfig {
    /// Get panel colors as iterator.
    pub fn iter_panel_colors(&self) -> impl Iterator<Item = (PanelType, &str)> {
        [
            (PanelType::Cpu, "cpu"),
            (PanelType::Memory, "memory"),
            (PanelType::Disk, "disk"),
            (PanelType::Network, "network"),
            (PanelType::Process, "process"),
            (PanelType::Gpu, "gpu"),
            (PanelType::Battery, "battery"),
            (PanelType::Sensors, "sensors"),
            (PanelType::Psi, "psi"),
            (PanelType::Connections, "connections"),
            (PanelType::Files, "files"),
        ]
        .into_iter()
        .filter_map(|(panel, key)| self.panel_colors.get(key).map(|c| (panel, c.as_str())))
    }
}

/// Panel type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PanelType {
    /// CPU usage panel.
    Cpu,
    /// Memory usage panel.
    Memory,
    /// Disk usage panel.
    Disk,
    /// Network usage panel.
    Network,
    /// Process list panel.
    Process,
    /// GPU usage panel.
    Gpu,
    /// Battery status panel.
    Battery,
    /// Sensors panel.
    Sensors,
    /// Compact sensors panel.
    SensorsCompact,
    /// PSI (Pressure Stall Information) panel.
    Psi,
    /// System info panel.
    System,
    /// Network connections panel.
    Connections,
    /// Treemap visualization panel.
    Treemap,
    /// Files panel.
    Files,
}

impl PanelType {
    /// Get panel index (0-based).
    pub fn index(self) -> usize {
        match self {
            Self::Cpu => 0,
            Self::Memory => 1,
            Self::Disk => 2,
            Self::Network => 3,
            Self::Process => 4,
            Self::Gpu => 5,
            Self::Battery => 6,
            Self::Sensors => 7,
            Self::SensorsCompact => 8,
            Self::Psi => 9,
            Self::System => 10,
            Self::Connections => 11,
            Self::Treemap => 12,
            Self::Files => 13,
        }
    }

    /// Get panel name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Memory => "Memory",
            Self::Disk => "Disk",
            Self::Network => "Network",
            Self::Process => "Process",
            Self::Gpu => "GPU",
            Self::Battery => "Battery",
            Self::Sensors => "Sensors",
            Self::SensorsCompact => "SensorsCompact",
            Self::Psi => "PSI",
            Self::System => "System",
            Self::Connections => "Connections",
            Self::Treemap => "Treemap",
            Self::Files => "Files",
        }
    }

    /// Get panel key for config.
    pub fn key(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Memory => "memory",
            Self::Disk => "disk",
            Self::Network => "network",
            Self::Process => "process",
            Self::Gpu => "gpu",
            Self::Battery => "battery",
            Self::Sensors => "sensors",
            Self::SensorsCompact => "sensors_compact",
            Self::Psi => "psi",
            Self::System => "system",
            Self::Connections => "connections",
            Self::Treemap => "treemap",
            Self::Files => "files",
        }
    }

    /// Get all panel types.
    pub fn all() -> &'static [PanelType] {
        &[
            Self::Cpu,
            Self::Memory,
            Self::Disk,
            Self::Network,
            Self::Process,
            Self::Gpu,
            Self::Battery,
            Self::Sensors,
            Self::SensorsCompact,
            Self::Psi,
            Self::System,
            Self::Connections,
            Self::Treemap,
            Self::Files,
        ]
    }
}

impl PresentarConfig {
    /// Parse configuration from YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml_ng::Error> {
        serde_yaml_ng::from_str(yaml)
    }

    /// Serialize configuration to YAML string.
    pub fn to_yaml(&self) -> Result<String, serde_yaml_ng::Error> {
        serde_yaml_ng::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PresentarConfig::default();
        assert_eq!(config.refresh_ms, 1000);
        assert!(config.layout.snap_to_grid);
        assert_eq!(config.layout.grid_size, 4);
    }

    #[test]
    fn test_parse_minimal_yaml() {
        let yaml = "refresh_ms: 500";
        let config = PresentarConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.refresh_ms, 500);
    }

    #[test]
    fn test_parse_full_yaml() {
        let yaml = r#"
refresh_ms: 1000
layout:
  snap_to_grid: true
  grid_size: 4
panels:
  cpu:
    enabled: true
    histogram: braille
  memory:
    enabled: true
keybindings:
  quit: q
  help: "?"
"#;
        let config = PresentarConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.refresh_ms, 1000);
        assert!(config.panels.cpu.enabled);
        assert_eq!(config.keybindings.quit, 'q');
    }

    #[test]
    fn test_panel_type_index() {
        assert_eq!(PanelType::Cpu.index(), 0);
        assert_eq!(PanelType::Memory.index(), 1);
        assert_eq!(PanelType::Files.index(), 13);
    }

    #[test]
    fn test_panel_type_name() {
        assert_eq!(PanelType::Cpu.name(), "CPU");
        assert_eq!(PanelType::Memory.name(), "Memory");
    }

    #[test]
    fn test_panel_type_key() {
        assert_eq!(PanelType::Cpu.key(), "cpu");
        assert_eq!(PanelType::SensorsCompact.key(), "sensors_compact");
    }

    #[test]
    fn test_iter_enabled() {
        let config = PanelConfigs::default();
        let enabled: Vec<_> = config.iter_enabled();
        assert!(enabled.iter().all(|(_, e)| *e));
    }

    #[test]
    fn test_set_enabled() {
        let mut config = PanelConfigs::default();
        config.set_enabled(PanelType::Cpu, false);
        assert!(!config.cpu.enabled);
    }

    #[test]
    fn test_theme_default_colors() {
        let theme = ThemeConfig::default();
        assert_eq!(theme.panel_colors.get("cpu"), Some(&"#64C8FF".to_string()));
        assert_eq!(
            theme.panel_colors.get("memory"),
            Some(&"#B478FF".to_string())
        );
    }

    #[test]
    fn test_histogram_style_parse() {
        let yaml = r#"
panels:
  cpu:
    histogram: block
"#;
        let config: PresentarConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.panels.cpu.histogram, HistogramStyle::Block);
    }

    #[test]
    fn test_border_style_parse() {
        let yaml = r#"
layout:
  border_style: sharp
"#;
        let config: PresentarConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.layout.border_style, BorderStyle::Sharp);
    }

    #[test]
    fn test_to_yaml_roundtrip() {
        let config = PresentarConfig::default();
        let yaml = config.to_yaml().unwrap();
        let parsed = PresentarConfig::from_yaml(&yaml).unwrap();
        assert_eq!(config.refresh_ms, parsed.refresh_ms);
    }

    #[test]
    fn test_panel_type_all() {
        let all = PanelType::all();
        assert_eq!(all.len(), 14);
        assert_eq!(all[0], PanelType::Cpu);
        assert_eq!(all[13], PanelType::Files);
    }

    #[test]
    fn test_process_panel_columns() {
        let config = ProcessPanelConfig::default();
        assert!(config.columns.contains(&"pid".to_string()));
        assert!(config.columns.contains(&"cpu".to_string()));
        assert!(config.columns.contains(&"mem".to_string()));
    }
}
