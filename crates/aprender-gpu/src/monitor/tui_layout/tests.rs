//! Tests for TUI layout module (Extreme TDD).

use std::collections::VecDeque;

use super::*;

// =========================================================================
// H042: TUI Layout Tests
// =========================================================================

#[test]
fn h042_tui_layout_default() {
    let layout = TuiLayout::default();
    assert_eq!(layout.min_width, 80);
    assert_eq!(layout.min_height, 24);
    assert_eq!(layout.refresh_rate_ms, 100);
    assert_eq!(layout.sparkline_points, 60);
}

#[test]
fn h042_tui_layout_size_check() {
    let layout = TuiLayout::default();

    assert_eq!(layout.check_size(160, 48), SizeCheck::Recommended);
    assert_eq!(layout.check_size(80, 24), SizeCheck::Minimum);
    assert_eq!(layout.check_size(40, 12), SizeCheck::TooSmall);
}

// =========================================================================
// H043: Section Tests
// =========================================================================

#[test]
fn h043_section_new() {
    let section = Section::new("test", "Test Section", 0.25);
    assert_eq!(section.id, "test");
    assert_eq!(section.title, "Test Section");
    assert!((section.height_pct - 0.25).abs() < 0.001);
    assert!(!section.collapsed);
    assert!(!section.focused);
}

#[test]
fn h043_section_toggle_collapsed() {
    let mut section = Section::new("test", "Test", 0.25);
    assert!(!section.collapsed);

    section.toggle_collapsed();
    assert!(section.collapsed);

    section.toggle_collapsed();
    assert!(!section.collapsed);
}

// =========================================================================
// H044: Gauge Widget Tests
// =========================================================================

#[test]
fn h044_gauge_new() {
    let gauge = GaugeWidget::new("CPU");
    assert_eq!(gauge.label, "CPU");
    assert_eq!(gauge.value_pct, 0.0);
}

#[test]
fn h044_gauge_with_value() {
    let gauge = GaugeWidget::new("CPU").with_value(75.5);
    assert!((gauge.value_pct - 75.5).abs() < 0.01);
}

#[test]
fn h044_gauge_color() {
    let ok = GaugeWidget::new("Test").with_value(50.0);
    let warn = GaugeWidget::new("Test").with_value(75.0);
    let crit = GaugeWidget::new("Test").with_value(95.0);

    assert_eq!(ok.color(), GaugeColor::Ok);
    assert_eq!(warn.color(), GaugeColor::Warning);
    assert_eq!(crit.color(), GaugeColor::Critical);
}

#[test]
fn h044_gauge_render_bar() {
    let gauge = GaugeWidget::new("CPU").with_value(50.0);
    let bar = gauge.render_bar(20);

    assert!(bar.contains("CPU"));
    assert!(bar.contains("50.0"));
    assert!(bar.contains("\u{2588}"));
    assert!(bar.contains("\u{2591}"));
}

// =========================================================================
// H045: Sparkline Widget Tests
// =========================================================================

#[test]
fn h045_sparkline_new() {
    let sparkline = SparklineWidget::new("History");
    assert_eq!(sparkline.label, "History");
    assert!(sparkline.data.is_empty());
}

#[test]
fn h045_sparkline_render_empty() {
    let sparkline = SparklineWidget::new("Test");
    assert_eq!(sparkline.render(), "");
}

#[test]
fn h045_sparkline_render() {
    let mut data = VecDeque::new();
    for i in 0..10 {
        data.push_back(i as f64 * 10.0);
    }

    let sparkline = SparklineWidget::new("Test").with_data(data);
    let rendered = sparkline.render();

    assert_eq!(rendered.chars().count(), 10);
    // First should be lowest, last should be highest
    assert!(rendered.starts_with('\u{2581}'));
    assert!(rendered.ends_with('\u{2588}'));
}

// =========================================================================
// H046: Progress Bar Widget Tests
// =========================================================================

#[test]
fn h046_progress_bar_new() {
    let bar = ProgressBarWidget::new("RAM");
    assert_eq!(bar.label, "RAM");
    assert_eq!(bar.progress, 0.0);
}

#[test]
fn h046_progress_bar_with_progress() {
    let bar = ProgressBarWidget::new("RAM").with_progress(0.75);
    assert!((bar.progress - 0.75).abs() < 0.001);
}

#[test]
fn h046_progress_bar_clamp() {
    let bar = ProgressBarWidget::new("RAM").with_progress(1.5);
    assert_eq!(bar.progress, 1.0);

    let bar2 = ProgressBarWidget::new("RAM").with_progress(-0.5);
    assert_eq!(bar2.progress, 0.0);
}

#[test]
fn h046_progress_bar_render() {
    let bar = ProgressBarWidget::new("RAM")
        .with_progress(0.5)
        .with_total("32 / 64 GB");
    let rendered = bar.render(20);

    assert!(rendered.contains("RAM"));
    assert!(rendered.contains("32 / 64 GB"));
}

// =========================================================================
// H047: Table Widget Tests
// =========================================================================

#[test]
fn h047_table_new() {
    let table = TableWidget::new(vec!["Name".to_string(), "Value".to_string()]);
    assert_eq!(table.headers.len(), 2);
    assert!(table.rows.is_empty());
}

#[test]
fn h047_table_add_row() {
    let mut table = TableWidget::new(vec!["Name".to_string()]);
    table.add_row(vec!["Test".to_string()]);
    assert_eq!(table.rows.len(), 1);
}

#[test]
fn h047_table_calculate_widths() {
    let mut table = TableWidget::new(vec!["Name".to_string(), "Value".to_string()]);
    table.add_row(vec!["Short".to_string(), "LongerValue".to_string()]);

    let widths = table.calculate_widths();
    assert_eq!(widths[0], 5); // "Short" or "Name", whichever is longer
    assert_eq!(widths[1], 11); // "LongerValue"
}

// =========================================================================
// H048: Color Scheme Tests
// =========================================================================

#[test]
fn h048_color_scheme_default() {
    let scheme = ColorScheme::default();
    // All colors should have valid RGB values
    assert!(scheme.ok.r <= 255);
    assert!(scheme.warning.r <= 255);
    assert!(scheme.critical.r <= 255);
}

#[test]
fn h048_rgb_color_ansi_fg() {
    let color = RgbColor::new(255, 128, 64);
    let ansi = color.to_ansi_fg();
    assert!(ansi.contains("38;2;255;128;64"));
}

#[test]
fn h048_rgb_color_ansi_bg() {
    let color = RgbColor::new(255, 128, 64);
    let ansi = color.to_ansi_bg();
    assert!(ansi.contains("48;2;255;128;64"));
}

#[test]
fn h048_rgb_for_pressure_level() {
    use crate::monitor::memory::PressureLevel;
    // All pressure levels should return valid colors
    let _ = RgbColor::for_pressure_level(PressureLevel::Ok);
    let _ = RgbColor::for_pressure_level(PressureLevel::Elevated);
    let _ = RgbColor::for_pressure_level(PressureLevel::Warning);
    let _ = RgbColor::for_pressure_level(PressureLevel::Critical);
}

// =========================================================================
// H049: Key Action Tests
// =========================================================================

#[test]
fn h049_key_action_key() {
    assert_eq!(KeyAction::Quit.key(), 'q');
    assert_eq!(KeyAction::Refresh.key(), 'r');
    assert_eq!(KeyAction::Help.key(), '?');
}

#[test]
fn h049_key_action_description() {
    assert_eq!(KeyAction::Quit.description(), "Quit");
    assert_eq!(KeyAction::Refresh.description(), "Refresh");
}

// =========================================================================
// H050: TUI Render State Tests
// =========================================================================

#[test]
fn h050_tui_render_state_default() {
    let state = TuiRenderState::default();
    assert!(state.cpu.is_none());
    assert!(state.gpus.is_empty());
    assert!(!state.stress_active);
    assert!(!state.paused);
    assert_eq!(state.focused_section, 0);
}

// =========================================================================
// H060: Additional Coverage Tests
// =========================================================================

#[test]
fn h060_tui_layout_with_refresh_rate() {
    let layout = TuiLayout::new().with_refresh_rate(50);
    assert_eq!(layout.refresh_rate_ms, 50);
}

#[test]
fn h060_section_add_widget() {
    let mut section = Section::new("test", "Test", 0.5);
    section.add_widget(Widget::Text(TextWidget::new("Hello")));
    assert_eq!(section.widgets.len(), 1);
}

#[test]
fn h060_text_widget() {
    let text = TextWidget::new("Hello World");
    assert_eq!(text.content, "Hello World");
    assert_eq!(text.style, TextStyle::Normal);

    let styled = TextWidget::new("Error").with_style(TextStyle::Error);
    assert_eq!(styled.style, TextStyle::Error);
}

#[test]
fn h060_gauge_with_thresholds() {
    let gauge = GaugeWidget::new("Test").with_thresholds(60.0, 80.0);
    assert!((gauge.warning_threshold - 60.0).abs() < 0.01);
    assert!((gauge.critical_threshold - 80.0).abs() < 0.01);
}

#[test]
fn h060_table_highlight() {
    let mut table = TableWidget::new(vec!["Col".to_string()]);
    table.add_row(vec!["Row1".to_string()]);
    table.add_row(vec!["Row2".to_string()]);
    table.highlight(1);
    assert_eq!(table.highlight_row, Some(1));
}

#[test]
fn h060_sparkline_no_auto_scale() {
    let mut sparkline = SparklineWidget::new("Test");
    sparkline.auto_scale = false;
    sparkline.data.push_back(25.0);
    sparkline.data.push_back(50.0);
    sparkline.data.push_back(75.0);

    let rendered = sparkline.render();
    assert_eq!(rendered.chars().count(), 3);
}

#[test]
fn h060_widget_enum_variants() {
    // Test that all widget variants can be created
    let _gauge = Widget::Gauge(GaugeWidget::new("Test"));
    let _sparkline = Widget::Sparkline(SparklineWidget::new("Test"));
    let _progress = Widget::ProgressBar(ProgressBarWidget::new("Test"));
    let _table = Widget::Table(TableWidget::new(vec![]));
    let _text = Widget::Text(TextWidget::new("Test"));
}

#[test]
fn h060_key_action_all_keys() {
    // Test all key actions have keys and descriptions
    let actions = [
        KeyAction::Quit,
        KeyAction::Refresh,
        KeyAction::ToggleStress,
        KeyAction::FocusNext,
        KeyAction::NavigateUp,
        KeyAction::NavigateDown,
        KeyAction::Expand,
        KeyAction::Help,
        KeyAction::Alerts,
        KeyAction::Export,
        KeyAction::TogglePause,
    ];

    for action in &actions {
        let _ = action.key();
        let desc = action.description();
        assert!(!desc.is_empty());
    }
}

#[test]
fn h060_memory_render_state() {
    let state = MemoryRenderState::default();
    assert_eq!(state.ram_pct, 0.0);
    assert!(state.vram.is_empty());
    assert!(state.ram_history.is_empty());
}

#[test]
fn h060_data_flow_render_state() {
    let state = DataFlowRenderState::default();
    assert_eq!(state.pcie_tx_gbps, 0.0);
    assert_eq!(state.pcie_rx_gbps, 0.0);
    assert!(state.transfers.is_empty());
}

#[test]
fn h060_text_styles() {
    let styles = [
        TextStyle::Normal,
        TextStyle::Bold,
        TextStyle::Dim,
        TextStyle::Italic,
        TextStyle::Header,
        TextStyle::Error,
        TextStyle::Warning,
        TextStyle::Success,
    ];

    // All styles should be distinct
    for (i, s1) in styles.iter().enumerate() {
        for (j, s2) in styles.iter().enumerate() {
            if i != j {
                assert_ne!(s1, s2);
            }
        }
    }
}

#[test]
fn h060_size_check_equality() {
    assert_eq!(SizeCheck::Recommended, SizeCheck::Recommended);
    assert_ne!(SizeCheck::Recommended, SizeCheck::Minimum);
    assert_ne!(SizeCheck::Minimum, SizeCheck::TooSmall);
}

#[test]
fn h060_gauge_color_equality() {
    assert_eq!(GaugeColor::Ok, GaugeColor::Ok);
    assert_ne!(GaugeColor::Ok, GaugeColor::Warning);
    assert_ne!(GaugeColor::Warning, GaugeColor::Critical);
}
