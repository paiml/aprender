//! TUI rendering methods for BrickTuner recommendations.
//!
//! Provides console and TUI panel output for tuner recommendations,
//! including full panels, compact status bars, and prediction comparisons.

use super::super::helpers::pad_right;
use super::BrickTuner;
use super::TunerRecommendation;

impl BrickTuner {
    /// Print recommendations to console (TUI-friendly)
    pub fn print_recommendation(&self, rec: &TunerRecommendation) {
        println!("╭─────────────────────────────────────────────────────────────╮");
        println!("│           BrickTuner Recommendations v{}                 │", self.version);
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ Predicted throughput: {:>7.1} tok/s ({:>4.0}% confidence)     │",
            rec.throughput.predicted_tps,
            rec.throughput.confidence * 100.0
        );
        println!(
            "│ Recommended kernel:   {:>15?} ({:>4.0}% conf)       │",
            rec.kernel.top_kernel,
            rec.kernel.confidence * 100.0
        );
        println!(
            "│ Bottleneck class:     {:>15} ({:>4.0}% conf)       │",
            rec.bottleneck.class,
            rec.bottleneck.confidence * 100.0
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Explanation: {}│", pad_right(&rec.bottleneck.explanation, 47));
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Suggested experiments:                                      │");
        for (i, exp) in rec.suggested_experiments.iter().take(3).enumerate() {
            println!("│  {}. {}│", i + 1, pad_right(&exp.to_string(), 56));
        }
        println!("╰─────────────────────────────────────────────────────────────╯");
    }

    // ========================================================================
    // T-TUNER-006: cbtop TUI Integration (GitHub #83)
    // ========================================================================

    /// Render recommendation as TUI panel lines (for cbtop integration)
    ///
    /// Returns a vector of strings that can be rendered in a TUI widget.
    /// Each line is formatted for fixed-width display (width=61 chars).
    pub fn render_panel(&self, rec: &TunerRecommendation) -> Vec<String> {
        let mut lines = Vec::with_capacity(12);

        lines.push(format!(
            "│           BrickTuner Recommendations v{}                 │",
            self.version
        ));
        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push(format!(
            "│ Predicted throughput: {:>7.1} tok/s ({:>4.0}% confidence)     │",
            rec.throughput.predicted_tps,
            rec.throughput.confidence * 100.0
        ));
        lines.push(format!(
            "│ Recommended kernel:   {:>15?} ({:>4.0}% conf)       │",
            rec.kernel.top_kernel,
            rec.kernel.confidence * 100.0
        ));
        lines.push(format!(
            "│ Bottleneck class:     {:>15} ({:>4.0}% conf)       │",
            rec.bottleneck.class,
            rec.bottleneck.confidence * 100.0
        ));
        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push(format!("│ Explanation: {}│", pad_right(&rec.bottleneck.explanation, 47)));
        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push("│ Suggested experiments:                                      │".to_string());

        for (i, exp) in rec.suggested_experiments.iter().take(3).enumerate() {
            lines.push(format!("│  {}. {}│", i + 1, pad_right(&exp.to_string(), 56)));
        }

        // Pad if fewer than 3 suggestions
        for _ in rec.suggested_experiments.len()..3 {
            lines.push(
                "│                                                             │".to_string(),
            );
        }

        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push("│ [Press 'a' to apply] [Press 't' to toggle] [Press 'r' to run]│".to_string());

        lines
    }

    /// Render compact recommendation (single line for status bar)
    pub fn render_compact(&self, rec: &TunerRecommendation) -> String {
        format!(
            "Tuner: {:.0} tok/s | {:?} | {} ({:.0}%)",
            rec.throughput.predicted_tps,
            rec.kernel.top_kernel,
            rec.bottleneck.class,
            rec.confidence_overall * 100.0
        )
    }

    /// Render prediction vs actual comparison
    pub fn render_comparison(&self, rec: &TunerRecommendation, actual_tps: f32) -> Vec<String> {
        let error_pct = if actual_tps > 0.0 {
            ((rec.throughput.predicted_tps - actual_tps) / actual_tps * 100.0).abs()
        } else {
            0.0
        };

        let accuracy_indicator = if error_pct < 5.0 {
            "🎯 Excellent"
        } else if error_pct < 10.0 {
            "✓ Good"
        } else if error_pct < 20.0 {
            "△ Fair"
        } else {
            "✗ Poor"
        };

        vec![
            format!(
                "│ Predicted: {:>7.1} tok/s  Actual: {:>7.1} tok/s           │",
                rec.throughput.predicted_tps, actual_tps
            ),
            format!(
                "│ Error: {:>5.1}%  Accuracy: {:>12}                       │",
                error_pct, accuracy_indicator
            ),
        ]
    }
}
