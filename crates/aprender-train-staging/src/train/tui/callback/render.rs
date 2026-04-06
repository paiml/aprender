//! Rendering methods for TerminalMonitorCallback.

use std::io::Write;

use crate::train::callback::CallbackContext;
use crate::train::tui::andon::AlertLevel;
use crate::train::tui::capability::DashboardLayout;
use crate::train::tui::progress::format_duration;
use crate::train::tui::sparkline::sparkline;

use super::monitor::TerminalMonitorCallback;

/// Rendering trait for callback display.
pub(crate) trait CallbackRenderer {
    /// Render the current display.
    fn render(&self, ctx: &CallbackContext) -> String;
    /// Print the display to stdout.
    fn print_display(&self, ctx: &CallbackContext);
}

impl CallbackRenderer for TerminalMonitorCallback {
    /// Render the current display.
    fn render(&self, ctx: &CallbackContext) -> String {
        match self.layout {
            DashboardLayout::Minimal => render_minimal(self, ctx),
            DashboardLayout::Compact => render_compact(self, ctx),
            DashboardLayout::Full => render_full(self, ctx),
        }
    }

    /// Print the display to stdout.
    fn print_display(&self, ctx: &CallbackContext) {
        let output = self.render(ctx);
        print!("{output}");
        let _ = std::io::stdout().flush();
    }
}

/// Render minimal single-line display.
fn render_minimal(_callback: &TerminalMonitorCallback, ctx: &CallbackContext) -> String {
    let percent = (ctx.epoch as f32 / ctx.max_epochs as f32) * 100.0;
    format!(
        "\rEpoch {}/{} [{:.1}%] loss={:.4} lr={:.2e}",
        ctx.epoch + 1,
        ctx.max_epochs,
        percent,
        ctx.loss,
        ctx.lr
    )
}

/// Render compact 5-line display.
fn render_compact(callback: &TerminalMonitorCallback, ctx: &CallbackContext) -> String {
    let loss_spark = sparkline(&callback.loss_buffer.values(), callback.sparkline_width);
    let elapsed = callback.start_time.elapsed().as_secs_f64();

    let val_info = ctx.val_loss.map(|v| format!(" val={v:.4}")).unwrap_or_default();

    let best_info = callback.loss_buffer.min().map(|m| format!(" best={m:.4}")).unwrap_or_default();

    format!(
        "\x1b[H\x1b[2J\
         ═══ {} Training ═══\n\
         Epoch {}/{} │ loss={:.4}{}{}\n\
         Loss: {} \n\
         LR: {:.2e} │ {:.1} steps/s\n\
         {}",
        callback.model_name,
        ctx.epoch + 1,
        ctx.max_epochs,
        ctx.loss,
        val_info,
        best_info,
        loss_spark,
        ctx.lr,
        ctx.global_step as f64 / elapsed.max(0.001),
        callback.progress.render()
    )
}

/// Render full dashboard display.
fn render_full(callback: &TerminalMonitorCallback, ctx: &CallbackContext) -> String {
    let loss_spark = sparkline(&callback.loss_buffer.values(), callback.sparkline_width);
    let lr_spark = sparkline(&callback.lr_buffer.values(), callback.sparkline_width);
    let elapsed = callback.start_time.elapsed().as_secs_f64();
    let steps_per_sec = ctx.global_step as f64 / elapsed.max(0.001);

    let val_spark = if callback.val_loss_buffer.is_empty() {
        String::new()
    } else {
        format!(
            "Val Loss: {} {:.4}\n",
            sparkline(&callback.val_loss_buffer.values(), callback.sparkline_width),
            callback.val_loss_buffer.last().unwrap_or(0.0)
        )
    };

    let alerts = render_alerts(callback);

    format!(
        "\x1b[H\x1b[2J\
╔═══════════════════════════════════════════════════════════════════╗
║  ENTRENAR TRAINING MONITOR                              [RUNNING] ║
╠═══════════════════════════════════════════════════════════════════╣
║  Model: {:<20} │ Epoch: {}/{}                  ║
╠═══════════════════════════════════════════════════════════════════╣
║  Loss: {} {:.4}                                 ║
║  {}║  LR:   {} {:.2e}                                 ║
╠═══════════════════════════════════════════════════════════════════╣
║  Steps/s: {:.1}  │  Elapsed: {}                        ║
║  {}║
╚═══════════════════════════════════════════════════════════════════╝
{}",
        callback.model_name,
        ctx.epoch + 1,
        ctx.max_epochs,
        loss_spark,
        ctx.loss,
        val_spark,
        lr_spark,
        ctx.lr,
        steps_per_sec,
        format_duration(elapsed),
        callback.progress.render(),
        alerts
    )
}

/// Render recent alerts.
pub(crate) fn render_alerts(callback: &TerminalMonitorCallback) -> String {
    let alerts = callback.andon.recent_alerts(3);
    if alerts.is_empty() {
        return String::new();
    }

    alerts
        .iter()
        .map(|a| {
            let prefix = match a.level {
                AlertLevel::Info => "ℹ️ ",
                AlertLevel::Warning => "⚠️ ",
                AlertLevel::Critical => "🛑",
            };
            format!("{} {}", prefix, a.message)
        })
        .collect::<Vec<_>>()
        .join("\n")
}
