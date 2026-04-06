//! Falsification protocol generator for presentar configurations.
//!
//! Generates F001-F100 falsification checks based on presentar configuration.
//!
//! # References
//!
//! - Jia & Harman (2011): Mutation Testing: From Theory to Practice
//! - Popper (1959): The Logic of Scientific Discovery (Falsificationism)

use super::schema::{PanelType, PresentarConfig};
use crate::playbook::schema::{
    FalsificationConfig, MutationDef, PerformanceBudget, Playbook, State, StateMachine,
};
use std::collections::HashMap;

/// Result of a falsification check.
#[derive(Debug, Clone)]
pub struct FalsificationResult {
    /// Check ID (F001-F100).
    pub id: String,
    /// Check description.
    pub description: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

impl FalsificationResult {
    /// Create a passing result.
    pub fn pass(id: &str, description: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            passed: true,
            error: None,
        }
    }

    /// Create a failing result.
    pub fn fail(id: &str, description: &str, error: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            passed: false,
            error: Some(error.to_string()),
        }
    }
}

/// A single falsification check.
#[derive(Debug, Clone)]
pub struct FalsificationCheck {
    /// Check ID (F001-F100).
    pub id: String,
    /// Category (existence, content, color, layout, keybinding, data, performance, accessibility).
    pub category: FalsificationCategory,
    /// Check description.
    pub description: String,
    /// Mutation to apply.
    pub mutation: String,
    /// Expected failure message.
    pub expected_failure: String,
}

/// Falsification check categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalsificationCategory {
    /// F001-F014: Panel existence.
    Existence,
    /// F015-F028: Panel content.
    Content,
    /// F029-F042: Color consistency.
    Color,
    /// F043-F056: Layout consistency.
    Layout,
    /// F057-F070: Keybinding consistency.
    Keybinding,
    /// F071-F084: Data binding.
    DataBinding,
    /// F085-F092: Performance.
    Performance,
    /// F093-F100: Accessibility.
    Accessibility,
}

impl FalsificationCategory {
    /// Get category name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Existence => "Panel Existence",
            Self::Content => "Panel Content",
            Self::Color => "Color Consistency",
            Self::Layout => "Layout Consistency",
            Self::Keybinding => "Keybinding Consistency",
            Self::DataBinding => "Data Binding",
            Self::Performance => "Performance",
            Self::Accessibility => "Accessibility",
        }
    }

    /// Get ID range for category.
    pub fn id_range(self) -> (u32, u32) {
        match self {
            Self::Existence => (1, 14),
            Self::Content => (15, 28),
            Self::Color => (29, 42),
            Self::Layout => (43, 56),
            Self::Keybinding => (57, 70),
            Self::DataBinding => (71, 84),
            Self::Performance => (85, 92),
            Self::Accessibility => (93, 100),
        }
    }
}

/// Generate all 100 falsification checks.
pub fn generate_all_checks() -> Vec<FalsificationCheck> {
    let mut checks = Vec::with_capacity(100);

    // F001-F014: Panel Existence
    checks.extend(generate_existence_checks());

    // F015-F028: Panel Content
    checks.extend(generate_content_checks());

    // F029-F042: Color Consistency
    checks.extend(generate_color_checks());

    // F043-F056: Layout Consistency
    checks.extend(generate_layout_checks());

    // F057-F070: Keybinding Consistency
    checks.extend(generate_keybinding_checks());

    // F071-F084: Data Binding
    checks.extend(generate_data_binding_checks());

    // F085-F092: Performance
    checks.extend(generate_performance_checks());

    // F093-F100: Accessibility
    checks.extend(generate_accessibility_checks());

    checks
}

/// Generate F001-F014: Panel Existence checks.
fn generate_existence_checks() -> Vec<FalsificationCheck> {
    let panels = [
        (1, PanelType::Cpu),
        (2, PanelType::Memory),
        (3, PanelType::Disk),
        (4, PanelType::Network),
        (5, PanelType::Process),
        (6, PanelType::Gpu),
        (7, PanelType::Battery),
        (8, PanelType::Sensors),
        (9, PanelType::SensorsCompact),
        (10, PanelType::Psi),
        (11, PanelType::System),
        (12, PanelType::Connections),
        (13, PanelType::Treemap),
        (14, PanelType::Files),
    ];

    panels
        .iter()
        .map(|(num, panel)| FalsificationCheck {
            id: format!("F{:03}", num),
            category: FalsificationCategory::Existence,
            description: format!("{} panel exists", panel.name()),
            mutation: format!("panels.{}.enabled = false", panel.key()),
            expected_failure: format!("{} panel must be visible", panel.name()),
        })
        .collect()
}

/// Generate F015-F028: Panel Content checks.
fn generate_content_checks() -> Vec<FalsificationCheck> {
    vec![
        FalsificationCheck {
            id: "F015".into(),
            category: FalsificationCategory::Content,
            description: "CPU shows percentage".into(),
            mutation: "panels.cpu.show_percent = false".into(),
            expected_failure: "CPU % must be visible".into(),
        },
        FalsificationCheck {
            id: "F016".into(),
            category: FalsificationCategory::Content,
            description: "CPU shows cores".into(),
            mutation: "panels.cpu.show_cores = false".into(),
            expected_failure: "Core count must be visible".into(),
        },
        FalsificationCheck {
            id: "F017".into(),
            category: FalsificationCategory::Content,
            description: "CPU shows frequency".into(),
            mutation: "panels.cpu.show_frequency = false".into(),
            expected_failure: "Frequency must be visible".into(),
        },
        FalsificationCheck {
            id: "F018".into(),
            category: FalsificationCategory::Content,
            description: "CPU shows temperature".into(),
            mutation: "panels.cpu.show_temperature = false".into(),
            expected_failure: "Temperature must be visible".into(),
        },
        FalsificationCheck {
            id: "F019".into(),
            category: FalsificationCategory::Content,
            description: "Memory shows used/total".into(),
            mutation: "panels.memory.show_usage = false".into(),
            expected_failure: "Memory usage must be visible".into(),
        },
        FalsificationCheck {
            id: "F020".into(),
            category: FalsificationCategory::Content,
            description: "Memory shows ZRAM".into(),
            mutation: "panels.memory.show_zram = false".into(),
            expected_failure: "ZRAM ratio must be visible".into(),
        },
        FalsificationCheck {
            id: "F021".into(),
            category: FalsificationCategory::Content,
            description: "Disk shows R/W rates".into(),
            mutation: "panels.disk.show_io = false".into(),
            expected_failure: "I/O rates must be visible".into(),
        },
        FalsificationCheck {
            id: "F022".into(),
            category: FalsificationCategory::Content,
            description: "Network shows RX/TX".into(),
            mutation: "panels.network.show_rates = false".into(),
            expected_failure: "RX/TX must be visible".into(),
        },
        FalsificationCheck {
            id: "F023".into(),
            category: FalsificationCategory::Content,
            description: "Process shows PID".into(),
            mutation: "panels.process.columns -= pid".into(),
            expected_failure: "PID column must exist".into(),
        },
        FalsificationCheck {
            id: "F024".into(),
            category: FalsificationCategory::Content,
            description: "Process shows CPU%".into(),
            mutation: "panels.process.columns -= cpu".into(),
            expected_failure: "CPU% column must exist".into(),
        },
        FalsificationCheck {
            id: "F025".into(),
            category: FalsificationCategory::Content,
            description: "Process shows MEM%".into(),
            mutation: "panels.process.columns -= mem".into(),
            expected_failure: "MEM% column must exist".into(),
        },
        FalsificationCheck {
            id: "F026".into(),
            category: FalsificationCategory::Content,
            description: "GPU shows utilization".into(),
            mutation: "panels.gpu.show_util = false".into(),
            expected_failure: "GPU util must be visible".into(),
        },
        FalsificationCheck {
            id: "F027".into(),
            category: FalsificationCategory::Content,
            description: "GPU shows VRAM".into(),
            mutation: "panels.gpu.show_vram = false".into(),
            expected_failure: "VRAM must be visible".into(),
        },
        FalsificationCheck {
            id: "F028".into(),
            category: FalsificationCategory::Content,
            description: "Battery shows charge".into(),
            mutation: "panels.battery.show_charge = false".into(),
            expected_failure: "Charge must be visible".into(),
        },
    ]
}

/// Generate F029-F042: Color Consistency checks.
fn generate_color_checks() -> Vec<FalsificationCheck> {
    let panel_colors = [
        (29, "cpu", "#64C8FF"),
        (30, "memory", "#B478FF"),
        (31, "disk", "#64B4FF"),
        (32, "network", "#FF9664"),
        (33, "process", "#DCC464"),
        (34, "gpu", "#64FF96"),
        (35, "battery", "#FFDC64"),
        (36, "sensors", "#FF6496"),
        (37, "psi", "#C85050"),
        (38, "connections", "#78B4DC"),
        (39, "files", "#B48C64"),
    ];

    let mut checks: Vec<FalsificationCheck> = panel_colors
        .iter()
        .map(|(num, panel, color)| FalsificationCheck {
            id: format!("F{:03}", num),
            category: FalsificationCategory::Color,
            description: format!("{} border color", panel),
            mutation: format!("theme.{}_color = #000000", panel),
            expected_failure: format!("{} border must be {}", panel, color),
        })
        .collect();

    // F040-F042: Percent color gradient
    checks.push(FalsificationCheck {
        id: "F040".into(),
        category: FalsificationCategory::Color,
        description: "Percent 0-25% cyan".into(),
        mutation: "percent_color(10) != cyan".into(),
        expected_failure: "0-25% must be cyan".into(),
    });
    checks.push(FalsificationCheck {
        id: "F041".into(),
        category: FalsificationCategory::Color,
        description: "Percent 50-75% yellow".into(),
        mutation: "percent_color(60) != yellow".into(),
        expected_failure: "50-75% must be yellow".into(),
    });
    checks.push(FalsificationCheck {
        id: "F042".into(),
        category: FalsificationCategory::Color,
        description: "Percent 90-100% red".into(),
        mutation: "percent_color(95) != red".into(),
        expected_failure: "90-100% must be red".into(),
    });

    checks
}

/// Generate F043-F056: Layout Consistency checks.
fn generate_layout_checks() -> Vec<FalsificationCheck> {
    vec![
        FalsificationCheck {
            id: "F043".into(),
            category: FalsificationCategory::Layout,
            description: "Top panels 45% height".into(),
            mutation: "layout.top_height = 0.2".into(),
            expected_failure: "Top must be 45% height".into(),
        },
        FalsificationCheck {
            id: "F044".into(),
            category: FalsificationCategory::Layout,
            description: "Bottom row 55% height".into(),
            mutation: "layout.bottom_height = 0.8".into(),
            expected_failure: "Bottom must be 55% height".into(),
        },
        FalsificationCheck {
            id: "F045".into(),
            category: FalsificationCategory::Layout,
            description: "Process 40% width".into(),
            mutation: "layout.process_width = 0.1".into(),
            expected_failure: "Process must be 40% width".into(),
        },
        FalsificationCheck {
            id: "F046".into(),
            category: FalsificationCategory::Layout,
            description: "Connections 30% width".into(),
            mutation: "layout.conn_width = 0.1".into(),
            expected_failure: "Connections must be 30% width".into(),
        },
        FalsificationCheck {
            id: "F047".into(),
            category: FalsificationCategory::Layout,
            description: "Treemap 30% width".into(),
            mutation: "layout.tree_width = 0.1".into(),
            expected_failure: "Treemap must be 30% width".into(),
        },
        FalsificationCheck {
            id: "F048".into(),
            category: FalsificationCategory::Layout,
            description: "Grid snap enabled".into(),
            mutation: "layout.snap_to_grid = false".into(),
            expected_failure: "Grid snap must work".into(),
        },
        FalsificationCheck {
            id: "F049".into(),
            category: FalsificationCategory::Layout,
            description: "Min panel width".into(),
            mutation: "layout.min_panel_width = 5".into(),
            expected_failure: "Min width must be 30".into(),
        },
        FalsificationCheck {
            id: "F050".into(),
            category: FalsificationCategory::Layout,
            description: "Min panel height".into(),
            mutation: "layout.min_panel_height = 2".into(),
            expected_failure: "Min height must be 6".into(),
        },
        FalsificationCheck {
            id: "F051".into(),
            category: FalsificationCategory::Layout,
            description: "Rounded borders".into(),
            mutation: "layout.border_style = sharp".into(),
            expected_failure: "Borders must be rounded".into(),
        },
        FalsificationCheck {
            id: "F052".into(),
            category: FalsificationCategory::Layout,
            description: "Title left-aligned".into(),
            mutation: "layout.title_align = center".into(),
            expected_failure: "Title must be left".into(),
        },
        FalsificationCheck {
            id: "F053".into(),
            category: FalsificationCategory::Layout,
            description: "1-char padding".into(),
            mutation: "layout.padding = 0".into(),
            expected_failure: "Padding must be 1".into(),
        },
        FalsificationCheck {
            id: "F054".into(),
            category: FalsificationCategory::Layout,
            description: "Responsive resize".into(),
            mutation: "resize(80, 24) fails".into(),
            expected_failure: "Must handle resize".into(),
        },
        FalsificationCheck {
            id: "F055".into(),
            category: FalsificationCategory::Layout,
            description: "Graceful degradation".into(),
            mutation: "resize(40, 12) crashes".into(),
            expected_failure: "Must degrade gracefully".into(),
        },
        FalsificationCheck {
            id: "F056".into(),
            category: FalsificationCategory::Layout,
            description: "2-column adaptive".into(),
            mutation: "columns != 2 when width > 100".into(),
            expected_failure: "Must use 2 columns".into(),
        },
    ]
}

/// Generate F057-F070: Keybinding Consistency checks.
fn generate_keybinding_checks() -> Vec<FalsificationCheck> {
    vec![
        FalsificationCheck {
            id: "F057".into(),
            category: FalsificationCategory::Keybinding,
            description: "'q' quits".into(),
            mutation: "keybindings.quit = x".into(),
            expected_failure: "'q' must quit".into(),
        },
        FalsificationCheck {
            id: "F058".into(),
            category: FalsificationCategory::Keybinding,
            description: "'?' shows help".into(),
            mutation: "keybindings.help = x".into(),
            expected_failure: "'?' must show help".into(),
        },
        FalsificationCheck {
            id: "F059".into(),
            category: FalsificationCategory::Keybinding,
            description: "'f' toggles FPS".into(),
            mutation: "keybindings.toggle_fps = x".into(),
            expected_failure: "'f' must toggle FPS".into(),
        },
        FalsificationCheck {
            id: "F060".into(),
            category: FalsificationCategory::Keybinding,
            description: "'/' filters".into(),
            mutation: "keybindings.filter = x".into(),
            expected_failure: "'/' must filter".into(),
        },
        FalsificationCheck {
            id: "F061".into(),
            category: FalsificationCategory::Keybinding,
            description: "'c' sorts by CPU".into(),
            mutation: "keybindings.sort_cpu = x".into(),
            expected_failure: "'c' must sort CPU".into(),
        },
        FalsificationCheck {
            id: "F062".into(),
            category: FalsificationCategory::Keybinding,
            description: "'m' sorts by MEM".into(),
            mutation: "keybindings.sort_mem = x".into(),
            expected_failure: "'m' must sort MEM".into(),
        },
        FalsificationCheck {
            id: "F063".into(),
            category: FalsificationCategory::Keybinding,
            description: "'p' sorts by PID".into(),
            mutation: "keybindings.sort_pid = x".into(),
            expected_failure: "'p' must sort PID".into(),
        },
        FalsificationCheck {
            id: "F064".into(),
            category: FalsificationCategory::Keybinding,
            description: "'k' kills process".into(),
            mutation: "keybindings.kill = x".into(),
            expected_failure: "'k' must kill".into(),
        },
        FalsificationCheck {
            id: "F065".into(),
            category: FalsificationCategory::Keybinding,
            description: "Enter explodes".into(),
            mutation: "keybindings.explode = x".into(),
            expected_failure: "Enter must explode".into(),
        },
        FalsificationCheck {
            id: "F066".into(),
            category: FalsificationCategory::Keybinding,
            description: "Escape collapses".into(),
            mutation: "keybindings.collapse = x".into(),
            expected_failure: "Escape must collapse".into(),
        },
        FalsificationCheck {
            id: "F067".into(),
            category: FalsificationCategory::Keybinding,
            description: "Tab navigates".into(),
            mutation: "keybindings.navigate = x".into(),
            expected_failure: "Tab must navigate".into(),
        },
        FalsificationCheck {
            id: "F068".into(),
            category: FalsificationCategory::Keybinding,
            description: "'1' toggles CPU".into(),
            mutation: "keybindings.toggle_1 = x".into(),
            expected_failure: "'1' must toggle CPU".into(),
        },
        FalsificationCheck {
            id: "F069".into(),
            category: FalsificationCategory::Keybinding,
            description: "'0' resets all".into(),
            mutation: "keybindings.reset = x".into(),
            expected_failure: "'0' must reset".into(),
        },
        FalsificationCheck {
            id: "F070".into(),
            category: FalsificationCategory::Keybinding,
            description: "No key conflicts".into(),
            mutation: "keybindings.quit = keybindings.help".into(),
            expected_failure: "Keys must be unique".into(),
        },
    ]
}

/// Generate F071-F084: Data Binding checks.
fn generate_data_binding_checks() -> Vec<FalsificationCheck> {
    vec![
        FalsificationCheck {
            id: "F071".into(),
            category: FalsificationCategory::DataBinding,
            description: "CPU data updates".into(),
            mutation: "cpu.update_interval = 0".into(),
            expected_failure: "CPU must update".into(),
        },
        FalsificationCheck {
            id: "F072".into(),
            category: FalsificationCategory::DataBinding,
            description: "Memory data updates".into(),
            mutation: "memory.update_interval = 0".into(),
            expected_failure: "Memory must update".into(),
        },
        FalsificationCheck {
            id: "F073".into(),
            category: FalsificationCategory::DataBinding,
            description: "Disk data updates".into(),
            mutation: "disk.update_interval = 0".into(),
            expected_failure: "Disk must update".into(),
        },
        FalsificationCheck {
            id: "F074".into(),
            category: FalsificationCategory::DataBinding,
            description: "Network data updates".into(),
            mutation: "network.update_interval = 0".into(),
            expected_failure: "Network must update".into(),
        },
        FalsificationCheck {
            id: "F075".into(),
            category: FalsificationCategory::DataBinding,
            description: "Process data updates".into(),
            mutation: "process.update_interval = 0".into(),
            expected_failure: "Process must update".into(),
        },
        FalsificationCheck {
            id: "F076".into(),
            category: FalsificationCategory::DataBinding,
            description: "GPU data updates".into(),
            mutation: "gpu.update_interval = 0".into(),
            expected_failure: "GPU must update".into(),
        },
        FalsificationCheck {
            id: "F077".into(),
            category: FalsificationCategory::DataBinding,
            description: "Sparkline history".into(),
            mutation: "sparkline_history = 0".into(),
            expected_failure: "History must exist".into(),
        },
        FalsificationCheck {
            id: "F078".into(),
            category: FalsificationCategory::DataBinding,
            description: "Async data race".into(),
            mutation: "async_delay = 5000ms".into(),
            expected_failure: "Must handle slow data".into(),
        },
        FalsificationCheck {
            id: "F079".into(),
            category: FalsificationCategory::DataBinding,
            description: "Missing data fallback".into(),
            mutation: "data.cpu = null".into(),
            expected_failure: "Must show N/A".into(),
        },
        FalsificationCheck {
            id: "F080".into(),
            category: FalsificationCategory::DataBinding,
            description: "NaN handling".into(),
            mutation: "data.cpu.percent = NaN".into(),
            expected_failure: "Must handle NaN".into(),
        },
        FalsificationCheck {
            id: "F081".into(),
            category: FalsificationCategory::DataBinding,
            description: "Negative values".into(),
            mutation: "data.memory.used = -1".into(),
            expected_failure: "Must clamp to 0".into(),
        },
        FalsificationCheck {
            id: "F082".into(),
            category: FalsificationCategory::DataBinding,
            description: "Overflow values".into(),
            mutation: "data.cpu.percent = 150".into(),
            expected_failure: "Must clamp to 100".into(),
        },
        FalsificationCheck {
            id: "F083".into(),
            category: FalsificationCategory::DataBinding,
            description: "Empty process list".into(),
            mutation: "data.processes = []".into(),
            expected_failure: "Must show empty state".into(),
        },
        FalsificationCheck {
            id: "F084".into(),
            category: FalsificationCategory::DataBinding,
            description: "1000+ processes".into(),
            mutation: "data.processes.len = 5000".into(),
            expected_failure: "Must paginate".into(),
        },
    ]
}

/// Generate F085-F092: Performance checks.
fn generate_performance_checks() -> Vec<FalsificationCheck> {
    vec![
        FalsificationCheck {
            id: "F085".into(),
            category: FalsificationCategory::Performance,
            description: "60 FPS render".into(),
            mutation: "render_time > 16ms".into(),
            expected_failure: "Must render in 16ms".into(),
        },
        FalsificationCheck {
            id: "F086".into(),
            category: FalsificationCategory::Performance,
            description: "Memory stable".into(),
            mutation: "memory_growth > 1MB/min".into(),
            expected_failure: "Must not leak".into(),
        },
        FalsificationCheck {
            id: "F087".into(),
            category: FalsificationCategory::Performance,
            description: "CPU < 5% idle".into(),
            mutation: "cpu_usage > 5%".into(),
            expected_failure: "Must be efficient".into(),
        },
        FalsificationCheck {
            id: "F088".into(),
            category: FalsificationCategory::Performance,
            description: "Startup < 100ms".into(),
            mutation: "startup_time > 100ms".into(),
            expected_failure: "Must start fast".into(),
        },
        FalsificationCheck {
            id: "F089".into(),
            category: FalsificationCategory::Performance,
            description: "Resize < 16ms".into(),
            mutation: "resize_time > 16ms".into(),
            expected_failure: "Must resize fast".into(),
        },
        FalsificationCheck {
            id: "F090".into(),
            category: FalsificationCategory::Performance,
            description: "Filter O(n)".into(),
            mutation: "filter_complexity != O(n)".into(),
            expected_failure: "Filter must be O(n)".into(),
        },
        FalsificationCheck {
            id: "F091".into(),
            category: FalsificationCategory::Performance,
            description: "Sort O(n log n)".into(),
            mutation: "sort_complexity != O(n log n)".into(),
            expected_failure: "Sort must be O(n log n)".into(),
        },
        FalsificationCheck {
            id: "F092".into(),
            category: FalsificationCategory::Performance,
            description: "Render O(panels)".into(),
            mutation: "render_complexity != O(p)".into(),
            expected_failure: "Render must be O(p)".into(),
        },
    ]
}

/// Generate F093-F100: Accessibility checks.
fn generate_accessibility_checks() -> Vec<FalsificationCheck> {
    vec![
        FalsificationCheck {
            id: "F093".into(),
            category: FalsificationCategory::Accessibility,
            description: "High contrast mode".into(),
            mutation: "theme.high_contrast = false".into(),
            expected_failure: "Must support high contrast".into(),
        },
        FalsificationCheck {
            id: "F094".into(),
            category: FalsificationCategory::Accessibility,
            description: "Colorblind safe".into(),
            mutation: "theme.colorblind = false".into(),
            expected_failure: "Must be colorblind safe".into(),
        },
        FalsificationCheck {
            id: "F095".into(),
            category: FalsificationCategory::Accessibility,
            description: "Screen reader text".into(),
            mutation: "aria.labels = null".into(),
            expected_failure: "Must have labels".into(),
        },
        FalsificationCheck {
            id: "F096".into(),
            category: FalsificationCategory::Accessibility,
            description: "Keyboard-only nav".into(),
            mutation: "mouse_only = true".into(),
            expected_failure: "Must work keyboard-only".into(),
        },
        FalsificationCheck {
            id: "F097".into(),
            category: FalsificationCategory::Accessibility,
            description: "Focus visible".into(),
            mutation: "focus.visible = false".into(),
            expected_failure: "Focus must be visible".into(),
        },
        FalsificationCheck {
            id: "F098".into(),
            category: FalsificationCategory::Accessibility,
            description: "No flashing".into(),
            mutation: "animation.flash = true".into(),
            expected_failure: "No flashing content".into(),
        },
        FalsificationCheck {
            id: "F099".into(),
            category: FalsificationCategory::Accessibility,
            description: "Text scalable".into(),
            mutation: "text.scalable = false".into(),
            expected_failure: "Text must scale".into(),
        },
        FalsificationCheck {
            id: "F100".into(),
            category: FalsificationCategory::Accessibility,
            description: "Error messages clear".into(),
            mutation: "error.verbose = false".into(),
            expected_failure: "Errors must be clear".into(),
        },
    ]
}

/// Generate a falsification playbook from presentar configuration.
pub fn generate_falsification_playbook(config: &PresentarConfig) -> Playbook {
    let all_checks = generate_all_checks();

    let mutations: Vec<MutationDef> = all_checks
        .iter()
        .filter(|check| should_include_check(check, config))
        .map(|check| MutationDef {
            id: check.id.clone(),
            description: check.description.clone(),
            mutate: check.mutation.clone(),
            expected_failure: check.expected_failure.clone(),
        })
        .collect();

    // Create minimal state machine for falsification-only playbook
    let mut states = HashMap::new();
    states.insert(
        "initial".into(),
        State {
            id: "initial".into(),
            description: "Initial state for falsification testing".into(),
            on_entry: vec![],
            on_exit: vec![],
            invariants: vec![],
            final_state: false,
        },
    );
    states.insert(
        "final".into(),
        State {
            id: "final".into(),
            description: "Final state - all tests complete".into(),
            on_entry: vec![],
            on_exit: vec![],
            invariants: vec![],
            final_state: true,
        },
    );

    let machine = StateMachine {
        id: "presentar-falsification".into(),
        initial: "initial".into(),
        states,
        transitions: vec![], // No transitions needed for falsification-only
        forbidden: vec![],
        performance: None,
    };

    Playbook {
        version: "1.0".into(),
        name: "presentar-falsification".into(),
        description: format!(
            "Generated falsification playbook with {} mutations",
            mutations.len()
        ),
        machine,
        performance: PerformanceBudget::default(),
        playbook: None,
        assertions: None,
        falsification: Some(FalsificationConfig { mutations }),
        metadata: HashMap::new(),
    }
}

/// Determine if a check should be included based on config.
fn should_include_check(check: &FalsificationCheck, config: &PresentarConfig) -> bool {
    match check.category {
        FalsificationCategory::Existence => {
            // Include existence check if panel is enabled
            match check.id.as_str() {
                "F001" => config.panels.cpu.enabled,
                "F002" => config.panels.memory.enabled,
                "F003" => config.panels.disk.enabled,
                "F004" => config.panels.network.enabled,
                "F005" => config.panels.process.enabled,
                "F006" => config.panels.gpu.enabled,
                "F007" => config.panels.battery.enabled,
                "F008" => config.panels.sensors.enabled,
                "F010" => config.panels.psi.enabled,
                "F012" => config.panels.connections.enabled,
                "F014" => config.panels.files.enabled,
                _ => true,
            }
        }
        _ => true, // Include all other checks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_all_checks() {
        let checks = generate_all_checks();
        assert_eq!(checks.len(), 100);

        // Verify IDs are sequential
        for (i, check) in checks.iter().enumerate() {
            let expected_id = format!("F{:03}", i + 1);
            assert_eq!(check.id, expected_id);
        }
    }

    #[test]
    fn test_existence_checks() {
        let checks = generate_existence_checks();
        assert_eq!(checks.len(), 14);
        assert_eq!(checks[0].id, "F001");
        assert_eq!(checks[13].id, "F014");
    }

    #[test]
    fn test_content_checks() {
        let checks = generate_content_checks();
        assert_eq!(checks.len(), 14);
        assert_eq!(checks[0].id, "F015");
        assert_eq!(checks[13].id, "F028");
    }

    #[test]
    fn test_color_checks() {
        let checks = generate_color_checks();
        assert_eq!(checks.len(), 14);
        assert_eq!(checks[0].id, "F029");
        assert_eq!(checks[13].id, "F042");
    }

    #[test]
    fn test_layout_checks() {
        let checks = generate_layout_checks();
        assert_eq!(checks.len(), 14);
        assert_eq!(checks[0].id, "F043");
        assert_eq!(checks[13].id, "F056");
    }

    #[test]
    fn test_keybinding_checks() {
        let checks = generate_keybinding_checks();
        assert_eq!(checks.len(), 14);
        assert_eq!(checks[0].id, "F057");
        assert_eq!(checks[13].id, "F070");
    }

    #[test]
    fn test_data_binding_checks() {
        let checks = generate_data_binding_checks();
        assert_eq!(checks.len(), 14);
        assert_eq!(checks[0].id, "F071");
        assert_eq!(checks[13].id, "F084");
    }

    #[test]
    fn test_performance_checks() {
        let checks = generate_performance_checks();
        assert_eq!(checks.len(), 8);
        assert_eq!(checks[0].id, "F085");
        assert_eq!(checks[7].id, "F092");
    }

    #[test]
    fn test_accessibility_checks() {
        let checks = generate_accessibility_checks();
        assert_eq!(checks.len(), 8);
        assert_eq!(checks[0].id, "F093");
        assert_eq!(checks[7].id, "F100");
    }

    #[test]
    fn test_generate_playbook() {
        let config = PresentarConfig::default();
        let playbook = generate_falsification_playbook(&config);

        assert_eq!(playbook.version, "1.0");
        assert!(playbook.falsification.is_some());

        let mutations = &playbook.falsification.unwrap().mutations;
        assert!(!mutations.is_empty());
    }

    #[test]
    fn test_category_id_range() {
        assert_eq!(FalsificationCategory::Existence.id_range(), (1, 14));
        assert_eq!(FalsificationCategory::Content.id_range(), (15, 28));
        assert_eq!(FalsificationCategory::Color.id_range(), (29, 42));
        assert_eq!(FalsificationCategory::Layout.id_range(), (43, 56));
        assert_eq!(FalsificationCategory::Keybinding.id_range(), (57, 70));
        assert_eq!(FalsificationCategory::DataBinding.id_range(), (71, 84));
        assert_eq!(FalsificationCategory::Performance.id_range(), (85, 92));
        assert_eq!(FalsificationCategory::Accessibility.id_range(), (93, 100));
    }

    #[test]
    fn test_category_name() {
        assert_eq!(FalsificationCategory::Existence.name(), "Panel Existence");
        assert_eq!(FalsificationCategory::Performance.name(), "Performance");
    }

    #[test]
    fn test_falsification_result() {
        let pass = FalsificationResult::pass("F001", "Test");
        assert!(pass.passed);
        assert!(pass.error.is_none());

        let fail = FalsificationResult::fail("F002", "Test", "Error");
        assert!(!fail.passed);
        assert_eq!(fail.error, Some("Error".into()));
    }

    #[test]
    fn test_should_include_check_disabled_panel() {
        let mut config = PresentarConfig::default();
        config.panels.cpu.enabled = false;

        let check = FalsificationCheck {
            id: "F001".into(),
            category: FalsificationCategory::Existence,
            description: "CPU panel exists".into(),
            mutation: "panels.cpu.enabled = false".into(),
            expected_failure: "CPU panel must be visible".into(),
        };

        assert!(!should_include_check(&check, &config));
    }

    #[test]
    fn test_should_include_check_enabled_panel() {
        let config = PresentarConfig::default();

        let check = FalsificationCheck {
            id: "F001".into(),
            category: FalsificationCategory::Existence,
            description: "CPU panel exists".into(),
            mutation: "panels.cpu.enabled = false".into(),
            expected_failure: "CPU panel must be visible".into(),
        };

        assert!(should_include_check(&check, &config));
    }
}
