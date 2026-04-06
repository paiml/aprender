//! Terminal Monitor Callback structure and builder methods.

use std::io::Write;
use std::time::{Duration, Instant};

use super::render::CallbackRenderer;
use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};
use crate::train::tui::andon::AndonSystem;
use crate::train::tui::buffer::MetricsBuffer;
use crate::train::tui::capability::{DashboardLayout, TerminalMode};
use crate::train::tui::progress::ProgressBar;
use crate::train::tui::refresh::RefreshPolicy;

/// Real-time terminal monitoring callback.
///
/// Integrates with training loop to provide live visualization.
#[derive(Debug)]
pub struct TerminalMonitorCallback {
    /// Loss buffer
    pub(crate) loss_buffer: MetricsBuffer,
    /// Validation loss buffer
    pub(crate) val_loss_buffer: MetricsBuffer,
    /// Learning rate buffer
    pub(crate) lr_buffer: MetricsBuffer,
    /// Progress bar
    pub(crate) progress: ProgressBar,
    /// Refresh policy
    pub(crate) refresh_policy: RefreshPolicy,
    /// Andon system
    pub(crate) andon: AndonSystem,
    /// Terminal mode
    pub(crate) mode: TerminalMode,
    /// Dashboard layout
    pub(crate) layout: DashboardLayout,
    /// Sparkline width
    pub(crate) sparkline_width: usize,
    /// Start time
    pub(crate) start_time: Instant,
    /// Model name (for display)
    pub(crate) model_name: String,
}

impl Default for TerminalMonitorCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalMonitorCallback {
    /// Create a new terminal monitor callback.
    pub fn new() -> Self {
        Self {
            loss_buffer: MetricsBuffer::new(100),
            val_loss_buffer: MetricsBuffer::new(100),
            lr_buffer: MetricsBuffer::new(100),
            progress: ProgressBar::new(100, 30),
            refresh_policy: RefreshPolicy::default(),
            andon: AndonSystem::new(),
            mode: TerminalMode::default(),
            layout: DashboardLayout::default(),
            sparkline_width: 20,
            start_time: Instant::now(),
            model_name: "model".to_string(),
        }
    }

    /// Set terminal mode.
    pub fn mode(mut self, mode: TerminalMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set dashboard layout.
    pub fn layout(mut self, layout: DashboardLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set model name.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// Set sparkline width.
    pub fn sparkline_width(mut self, width: usize) -> Self {
        self.sparkline_width = width;
        self
    }

    /// Set refresh interval.
    pub fn refresh_interval_ms(mut self, ms: u64) -> Self {
        self.refresh_policy.min_interval = Duration::from_millis(ms);
        self
    }
}

impl TrainerCallback for TerminalMonitorCallback {
    fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.start_time = Instant::now();
        self.progress = ProgressBar::new(ctx.max_epochs * ctx.steps_per_epoch, 30);

        // Clear screen and hide cursor
        print!("\x1b[?25l\x1b[2J\x1b[H");
        let _ = std::io::stdout().flush();

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, ctx: &CallbackContext) {
        // Final render
        self.print_display(ctx);

        // Show cursor
        println!("\x1b[?25h");
        let _ = std::io::stdout().flush();

        // Print summary
        println!("\nTraining complete!");
        if let Some(best) = self.loss_buffer.min() {
            println!("Best loss: {best:.4}");
        }
        println!(
            "Total time: {}",
            crate::train::tui::progress::format_duration(self.start_time.elapsed().as_secs_f64())
        );
    }

    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Record metrics
        self.loss_buffer.push(ctx.loss);
        self.lr_buffer.push(ctx.lr);
        if let Some(val) = ctx.val_loss {
            self.val_loss_buffer.push(val);
        }

        // Update progress
        self.progress.update(ctx.global_step);

        // Check health
        if self.andon.check_loss(ctx.loss) {
            return CallbackAction::Stop;
        }

        // Rate-limited refresh
        if self.refresh_policy.should_refresh(ctx.global_step) {
            self.print_display(ctx);
        }

        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Force refresh at epoch boundaries
        self.refresh_policy.force_refresh(ctx.global_step);
        self.print_display(ctx);
        CallbackAction::Continue
    }

    fn name(&self) -> &'static str {
        "TerminalMonitorCallback"
    }
}
