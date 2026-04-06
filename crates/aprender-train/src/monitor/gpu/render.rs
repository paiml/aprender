//! GPU metrics rendering utilities for terminal display.

use super::GpuMetrics;

/// Render a progress bar for terminal display
pub fn render_progress_bar(value: f64, width: usize) -> String {
    let filled = ((value / 100.0) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;

    let bar: String = std::iter::repeat_n('\u{2588}', filled).collect();
    let empty_bar: String = std::iter::repeat_n('\u{2591}', empty).collect();

    format!("{bar}{empty_bar}")
}

/// Render a sparkline from values
pub fn render_sparkline(values: &[u32], max_val: u32) -> String {
    const CHARS: &[char] = &[
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];

    if values.is_empty() || max_val == 0 {
        return String::new();
    }

    values
        .iter()
        .map(|&v| {
            let idx =
                ((f64::from(v) / f64::from(max_val)) * (CHARS.len() - 1) as f64).round() as usize;
            CHARS[idx.min(CHARS.len() - 1)]
        })
        .collect()
}

/// Format GPU metrics for terminal display
pub fn format_gpu_panel(metrics: &GpuMetrics, width: usize) -> Vec<String> {
    let bar_width = width.saturating_sub(25);

    vec![
        format!(
            "───── GPU {}: {} ─────",
            metrics.device_id,
            metrics.name.chars().take(width - 20).collect::<String>()
        ),
        format!(
            "Util: {} {:>3}%  │  Temp: {}°C",
            render_progress_bar(f64::from(metrics.utilization_percent), bar_width),
            metrics.utilization_percent,
            metrics.temperature_celsius
        ),
        format!(
            "VRAM: {} {:.1}/{:.1} GB ({:.0}%)",
            render_progress_bar(metrics.memory_percent(), bar_width),
            metrics.memory_used_mb as f64 / 1024.0,
            metrics.memory_total_mb as f64 / 1024.0,
            metrics.memory_percent()
        ),
        format!(
            "Pow:  {} {:.0}W/{:.0}W",
            render_progress_bar(metrics.power_percent(), bar_width),
            metrics.power_watts,
            metrics.power_limit_watts
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_progress_bar() {
        let bar = render_progress_bar(50.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '\u{2588}').count(), 5);
        assert_eq!(bar.chars().filter(|&c| c == '\u{2591}').count(), 5);
    }

    #[test]
    fn test_render_progress_bar_full() {
        let bar = render_progress_bar(100.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '\u{2588}').count(), 10);
    }

    #[test]
    fn test_render_progress_bar_empty() {
        let bar = render_progress_bar(0.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '\u{2591}').count(), 10);
    }

    #[test]
    fn test_render_sparkline() {
        let sparkline = render_sparkline(&[0, 50, 100], 100);
        assert_eq!(sparkline.chars().count(), 3);
        assert!(sparkline.starts_with('\u{2581}'));
        assert!(sparkline.ends_with('\u{2588}'));
    }

    #[test]
    fn test_render_sparkline_empty() {
        let sparkline = render_sparkline(&[], 100);
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_format_gpu_panel() {
        let metrics = GpuMetrics::mock(0);
        let lines = format_gpu_panel(&metrics, 60);
        assert!(!lines.is_empty());
        assert!(lines[0].contains("GPU 0"));
    }

    #[test]
    fn test_render_sparkline_max_val_zero() {
        let sparkline = render_sparkline(&[1, 2, 3], 0);
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_render_progress_bar_over_100() {
        // Should handle values > 100
        let bar = render_progress_bar(150.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '\u{2588}').count(), 10);
    }

    #[test]
    fn test_render_progress_bar_negative() {
        // Should handle negative values
        let bar = render_progress_bar(-10.0, 10);
        // Negative rounds to 0
        assert!(bar.chars().filter(|&c| c == '\u{2588}').count() == 0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_sparkline_length(values in prop::collection::vec(0u32..100, 0..50)) {
            let sparkline = render_sparkline(&values, 100);
            prop_assert_eq!(sparkline.chars().count(), values.len());
        }

        #[test]
        fn prop_progress_bar_length(value in 0.0f64..100.0, width in 1usize..50) {
            let bar = render_progress_bar(value, width);
            let char_count: usize = bar.chars().count();
            prop_assert_eq!(char_count, width);
        }
    }
}
