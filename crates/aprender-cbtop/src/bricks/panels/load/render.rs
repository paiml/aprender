//! Rendering logic for the load control panel.

use crate::brick::{BrickGrade, BrickScore};
use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::{Meter, Widget};

use super::LoadControlPanelBrick;

impl LoadControlPanelBrick {
    /// Render a score bar component
    pub(super) fn render_score_bar(value: u8, max: u8, width: usize) -> String {
        BrickScore::render_bar(value, max, width)
    }

    /// Format problem size for display
    pub(super) fn format_size(size: usize) -> String {
        if size >= 1024 {
            format!("{}K", size / 1024)
        } else {
            format!("{}", size)
        }
    }

    /// Paint the load control panel
    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle { color: self.theme.foreground, ..Default::default() };
        let dim_style = TextStyle { color: self.theme.dim, ..Default::default() };
        let selected_style = TextStyle {
            color: Color::new(0.3, 0.8, 1.0, 1.0), // Cyan for selected
            ..Default::default()
        };
        let running_style = TextStyle {
            color: Color::new(0.3, 1.0, 0.5, 1.0), // Green for running
            ..Default::default()
        };
        let stopped_style = TextStyle {
            color: Color::new(1.0, 0.5, 0.3, 1.0), // Orange for stopped
            ..Default::default()
        };
        let error_style = TextStyle {
            color: Color::new(1.0, 0.2, 0.2, 1.0), // Red for errors
            ..Default::default()
        };

        canvas.draw_text("Load Control", Point::new(2.0, 2.0), &label_style);

        // Status indicator
        let status_text = if self.is_running { "RUNNING" } else { "STOPPED" };
        let status_style = if self.is_running { &running_style } else { &stopped_style };
        canvas.draw_text(status_text, Point::new(width - 12.0, 2.0), status_style);

        // Backend selection
        let backend_style = if self.selected_item == 0 { &selected_style } else { &dim_style };
        canvas.draw_text("Backend:", Point::new(2.0, 4.0), backend_style);
        canvas.draw_text(
            &format!("< {} >", self.backend.name()),
            Point::new(12.0, 4.0),
            &label_style,
        );

        // Workload selection
        let workload_style = if self.selected_item == 1 { &selected_style } else { &dim_style };
        canvas.draw_text("Workload:", Point::new(2.0, 5.0), workload_style);
        canvas.draw_text(
            &format!("< {} >", self.workload.short_name()),
            Point::new(12.0, 5.0),
            &label_style,
        );

        // Intensity slider
        let intensity_style = if self.selected_item == 2 { &selected_style } else { &dim_style };
        canvas.draw_text("Intensity:", Point::new(2.0, 6.0), intensity_style);
        let intensity_color = self.theme.cpu_color(self.intensity);
        let mut intensity_meter = Meter::new(self.intensity, 100.0).with_color(intensity_color);
        intensity_meter.layout(Rect::new(12.0, 6.0, width - 30.0, 1.0));
        intensity_meter.paint(canvas);
        canvas.draw_text(
            &format!("{:.0}%", self.intensity),
            Point::new(width - 16.0, 6.0),
            &label_style,
        );

        // Problem size
        let size_style = if self.selected_item == 3 { &selected_style } else { &dim_style };
        canvas.draw_text("Size:", Point::new(2.0, 7.0), size_style);
        canvas.draw_text(
            &format!("< {} >", Self::format_size(self.problem_size)),
            Point::new(12.0, 7.0),
            &label_style,
        );

        // Start/Stop button indicator
        let button_style = if self.selected_item == 4 { &selected_style } else { &dim_style };
        let button_text = if self.is_running { "[STOP]" } else { "[START]" };
        canvas.draw_text(button_text, Point::new(2.0, 9.0), button_style);

        // Error display
        if let Some(ref err) = self.error {
            canvas.draw_text("Error:", Point::new(2.0, 11.0), &error_style);
            canvas.draw_text(err, Point::new(9.0, 11.0), &error_style);
        }

        // ComputeBrick Score section
        self.paint_score_or_stats(canvas, width, &label_style, &dim_style);

        // Help text
        canvas.draw_text(
            "Use arrow keys to navigate, Enter to toggle",
            Point::new(2.0, 20.0),
            &dim_style,
        );
    }

    /// Paint score section or fallback statistics
    fn paint_score_or_stats(
        &self,
        canvas: &mut dyn Canvas,
        width: f32,
        label_style: &TextStyle,
        dim_style: &TextStyle,
    ) {
        if let Some(ref score) = self.brick_score {
            let grade = score.grade();
            let grade_color = match grade {
                BrickGrade::A | BrickGrade::B => Color::new(0.3, 1.0, 0.5, 1.0), // Green
                BrickGrade::C => Color::new(1.0, 0.8, 0.3, 1.0),                 // Yellow
                BrickGrade::D | BrickGrade::F => Color::new(1.0, 0.3, 0.3, 1.0), // Red
            };
            let grade_style = TextStyle { color: grade_color, ..Default::default() };

            canvas.draw_text("ComputeBrick Score", Point::new(2.0, 13.0), label_style);
            canvas.draw_text(
                &format!("{}/100 ({})", score.total(), grade.letter()),
                Point::new(22.0, 13.0),
                &grade_style,
            );
            canvas.draw_text(
                &format!("{:.2} GFLOP/s", self.gflops),
                Point::new(width - 18.0, 13.0),
                label_style,
            );

            // Score breakdown bars
            let bar_width = 20;
            canvas.draw_text("Performance:", Point::new(2.0, 15.0), dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.performance, 40, bar_width),
                Point::new(14.0, 15.0),
                label_style,
            );
            canvas.draw_text(
                &format!("{}/40", score.performance),
                Point::new(36.0, 15.0),
                dim_style,
            );

            canvas.draw_text("Efficiency:", Point::new(2.0, 16.0), dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.efficiency, 25, bar_width),
                Point::new(14.0, 16.0),
                label_style,
            );
            canvas.draw_text(
                &format!("{}/25", score.efficiency),
                Point::new(36.0, 16.0),
                dim_style,
            );

            canvas.draw_text("Correctness:", Point::new(2.0, 17.0), dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.correctness, 20, bar_width),
                Point::new(14.0, 17.0),
                label_style,
            );
            canvas.draw_text(
                &format!("{}/20", score.correctness),
                Point::new(36.0, 17.0),
                dim_style,
            );

            canvas.draw_text("Stability:", Point::new(2.0, 18.0), dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.stability, 15, bar_width),
                Point::new(14.0, 18.0),
                label_style,
            );
            canvas.draw_text(&format!("{}/15", score.stability), Point::new(36.0, 18.0), dim_style);
        } else {
            // Statistics section (fallback when no score available)
            canvas.draw_text("Statistics", Point::new(2.0, 13.0), label_style);

            canvas.draw_text("Iterations:", Point::new(2.0, 14.0), dim_style);
            canvas.draw_text(
                &format!("{}", self.stats.iterations),
                Point::new(14.0, 14.0),
                label_style,
            );

            canvas.draw_text("Ops/sec:", Point::new(2.0, 15.0), dim_style);
            canvas.draw_text(
                &format!("{:.1}", self.stats.ops_per_sec),
                Point::new(14.0, 15.0),
                label_style,
            );

            canvas.draw_text("Throughput:", Point::new(2.0, 16.0), dim_style);
            canvas.draw_text(
                &format!("{:.2} GB/s", self.stats.throughput_gbs),
                Point::new(14.0, 16.0),
                label_style,
            );

            canvas.draw_text("Avg Latency:", Point::new(2.0, 17.0), dim_style);
            canvas.draw_text(
                &format!("{:.1} us", self.stats.avg_latency_us),
                Point::new(14.0, 17.0),
                label_style,
            );

            canvas.draw_text("P99 Latency:", Point::new(2.0, 18.0), dim_style);
            canvas.draw_text(
                &format!("{:.1} us", self.stats.p99_latency_us),
                Point::new(14.0, 18.0),
                label_style,
            );
        }
    }
}
