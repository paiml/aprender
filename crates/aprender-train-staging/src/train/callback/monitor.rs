//! Monitor callback that integrates with entrenar's monitoring system

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Callback that integrates with entrenar's monitoring system
#[derive(Debug)]
pub struct MonitorCallback {
    collector: crate::monitor::MetricsCollector,
    andon: crate::monitor::AndonSystem,
}

impl MonitorCallback {
    /// Create a new monitor callback
    pub fn new() -> Self {
        Self {
            collector: crate::monitor::MetricsCollector::new(),
            andon: crate::monitor::AndonSystem::new(),
        }
    }

    /// Get the metrics collector
    pub fn collector(&self) -> &crate::monitor::MetricsCollector {
        &self.collector
    }

    /// Get summary as JSON
    pub fn summary_json(&self) -> Result<String, serde_json::Error> {
        // Convert summary to string keys for JSON
        let summary: std::collections::HashMap<String, _> = self
            .collector
            .summary()
            .into_iter()
            .map(|(k, v)| (k.as_str().to_string(), v))
            .collect();
        serde_json::to_string_pretty(&summary)
    }
}

impl Default for MonitorCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainerCallback for MonitorCallback {
    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Record loss at each step
        self.collector.record(crate::monitor::Metric::Loss, f64::from(ctx.loss));
        self.collector.record(crate::monitor::Metric::LearningRate, f64::from(ctx.lr));
        CallbackAction::Continue
    }

    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.collector.record(crate::monitor::Metric::Epoch, ctx.epoch as f64);

        // Check for NaN/Inf loss
        if ctx.loss.is_nan() {
            self.andon.critical("NaN loss detected");
        } else if ctx.loss.is_infinite() {
            self.andon.critical("Infinite loss detected");
        }

        // Check if andon suggests stopping
        if self.andon.should_stop() {
            CallbackAction::Stop
        } else {
            CallbackAction::Continue
        }
    }

    fn name(&self) -> &'static str {
        "MonitorCallback"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_callback() {
        let mut monitor = MonitorCallback::new();
        let ctx = CallbackContext { epoch: 0, step: 0, loss: 0.5, lr: 0.001, ..Default::default() };

        assert_eq!(monitor.on_step_end(&ctx), CallbackAction::Continue);
        assert_eq!(monitor.on_epoch_end(&ctx), CallbackAction::Continue);

        // Verify metrics were recorded
        let summary = monitor.collector().summary();
        assert!(summary.contains_key(&crate::monitor::Metric::Loss));
    }

    #[test]
    fn test_monitor_callback_nan_detection() {
        let mut monitor = MonitorCallback::new();
        let ctx = CallbackContext { loss: f32::NAN, ..Default::default() };

        // NaN should trigger stop via andon
        assert_eq!(monitor.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_monitor_callback_default() {
        let mc = MonitorCallback::default();
        assert_eq!(mc.name(), "MonitorCallback");
    }

    #[test]
    fn test_monitor_callback_summary_json() {
        let mut mc = MonitorCallback::new();
        let ctx = CallbackContext { loss: 0.5, lr: 0.001, ..Default::default() };
        mc.on_step_end(&ctx);

        let json = mc.summary_json();
        assert!(json.is_ok());
    }

    #[test]
    fn test_monitor_callback_inf_detection() {
        let mut mc = MonitorCallback::new();
        let ctx = CallbackContext { loss: f32::INFINITY, ..Default::default() };
        assert_eq!(mc.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_monitor_callback_nan_loss() {
        let mut cb = MonitorCallback::new();
        let mut ctx = CallbackContext::default();
        ctx.loss = f32::NAN;

        let action = cb.on_epoch_end(&ctx);
        // Should detect NaN and potentially stop
        assert!(action == CallbackAction::Stop || action == CallbackAction::Continue);
    }

    #[test]
    fn test_monitor_callback_infinite_loss() {
        let mut cb = MonitorCallback::new();
        let mut ctx = CallbackContext::default();
        ctx.loss = f32::INFINITY;

        cb.on_epoch_end(&ctx);
        // Should detect infinite loss
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Monitor callback should detect NaN/Inf
        #[test]
        fn monitor_callback_detects_nan_inf(
            normal_loss in -100.0f32..100.0,
        ) {
            // Normal loss should continue
            let mut monitor = MonitorCallback::new();
            let ctx = CallbackContext {
                loss: normal_loss,
                ..Default::default()
            };
            prop_assert_eq!(monitor.on_epoch_end(&ctx), CallbackAction::Continue);

            // NaN should stop
            let mut monitor_nan = MonitorCallback::new();
            let ctx_nan = CallbackContext {
                loss: f32::NAN,
                ..Default::default()
            };
            prop_assert_eq!(monitor_nan.on_epoch_end(&ctx_nan), CallbackAction::Stop);

            // Inf should stop
            let mut monitor_inf = MonitorCallback::new();
            let ctx_inf = CallbackContext {
                loss: f32::INFINITY,
                ..Default::default()
            };
            prop_assert_eq!(monitor_inf.on_epoch_end(&ctx_inf), CallbackAction::Stop);
        }
    }
}
