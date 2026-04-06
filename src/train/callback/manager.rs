//! Callback manager for dispatching events to multiple callbacks

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Manages multiple callbacks and dispatches events
pub struct CallbackManager {
    callbacks: Vec<Box<dyn TrainerCallback>>,
}

impl CallbackManager {
    /// Create new callback manager
    pub fn new() -> Self {
        Self { callbacks: Vec::new() }
    }

    /// Add a callback
    pub fn add<C: TrainerCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }

    /// Check if no callbacks are registered
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }

    /// Get number of callbacks
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Fire train begin event
    pub fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_train_begin(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire train end event
    pub fn on_train_end(&mut self, ctx: &CallbackContext) {
        for cb in &mut self.callbacks {
            cb.on_train_end(ctx);
        }
    }

    /// Fire epoch begin event
    pub fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            match cb.on_epoch_begin(ctx) {
                CallbackAction::Stop => return CallbackAction::Stop,
                CallbackAction::SkipEpoch => return CallbackAction::SkipEpoch,
                CallbackAction::Continue => {}
            }
        }
        CallbackAction::Continue
    }

    /// Fire epoch end event
    pub fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_epoch_end(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire step begin event
    pub fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_step_begin(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire step end event
    pub fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_step_end(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::callback::{EarlyStopping, ProgressCallback};

    #[test]
    fn test_callback_manager_dispatch() {
        let mut manager = CallbackManager::new();

        // Add early stopping that triggers after 1 epoch without improvement
        let es = EarlyStopping::new(1, 0.001);
        manager.add(es);

        let mut ctx = CallbackContext::default();
        ctx.loss = 1.0;

        // First epoch
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Second epoch - no improvement, should stop
        ctx.epoch = 1;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_len_and_empty() {
        let mut manager = CallbackManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        manager.add(ProgressCallback::new(10));
        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_callback_manager_on_train_begin_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(manager.on_train_begin(&CallbackContext::default()), CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_on_train_end() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        struct EndCallback {
            called: Arc<AtomicBool>,
        }
        impl TrainerCallback for EndCallback {
            fn on_train_end(&mut self, _: &CallbackContext) {
                self.called.store(true, Ordering::SeqCst);
            }
            fn name(&self) -> &'static str {
                "EndCallback"
            }
        }

        let called = Arc::new(AtomicBool::new(false));
        let mut manager = CallbackManager::new();
        manager.add(EndCallback { called: called.clone() });
        manager.on_train_end(&CallbackContext::default());
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_callback_manager_on_epoch_begin_skip() {
        struct SkipCallback;
        impl TrainerCallback for SkipCallback {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::SkipEpoch
            }
            fn name(&self) -> &'static str {
                "SkipCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(SkipCallback);
        assert_eq!(manager.on_epoch_begin(&CallbackContext::default()), CallbackAction::SkipEpoch);
    }

    #[test]
    fn test_callback_manager_on_epoch_begin_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(manager.on_epoch_begin(&CallbackContext::default()), CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_on_step_begin_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(manager.on_step_begin(&CallbackContext::default()), CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_default() {
        let manager = CallbackManager::default();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_callback_manager_stop_propagation() {
        // Create a callback that always returns Stop
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        manager.add(ProgressCallback::new(10));

        let ctx = CallbackContext::default();
        let action = manager.on_epoch_end(&ctx);
        // Stop should propagate
        assert_eq!(action, CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_on_step_end_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(manager.on_step_end(&CallbackContext::default()), CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_all_continue() {
        // Test that all callbacks continue properly
        struct ContinueCallback;
        impl TrainerCallback for ContinueCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "ContinueCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(ContinueCallback);
        manager.add(ContinueCallback);

        let ctx = CallbackContext::default();
        assert_eq!(manager.on_train_begin(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_epoch_begin(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_step_begin(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_step_end(&ctx), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_multiple_train_end() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountingEndCallback {
            count: Arc<AtomicUsize>,
        }

        impl TrainerCallback for CountingEndCallback {
            fn on_train_end(&mut self, _: &CallbackContext) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            fn name(&self) -> &'static str {
                "CountingEndCallback"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountingEndCallback { count: count.clone() });
        manager.add(CountingEndCallback { count: count.clone() });
        manager.add(CountingEndCallback { count: count.clone() });

        manager.on_train_end(&CallbackContext::default());
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_callback_manager_stop_after_first() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountingStopCallback {
            count: Arc<AtomicUsize>,
        }

        impl TrainerCallback for CountingStopCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "CountingStopCallback"
            }
        }

        struct CountingContinueCallback {
            count: Arc<AtomicUsize>,
        }

        impl TrainerCallback for CountingContinueCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountingContinueCallback"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountingStopCallback { count: count.clone() });
        manager.add(CountingContinueCallback { count: count.clone() });

        // First callback stops, second should not be called
        let action = manager.on_train_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    // ── Additional coverage tests ─────────────────────────────────

    #[test]
    fn test_callback_manager_on_train_begin_continue() {
        let mut manager = CallbackManager::new();
        // No callbacks → should return Continue
        assert_eq!(manager.on_train_begin(&CallbackContext::default()), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_on_epoch_end_continue() {
        let mut manager = CallbackManager::new();
        // No callbacks → should return Continue
        assert_eq!(manager.on_epoch_end(&CallbackContext::default()), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_on_step_begin_continue() {
        let mut manager = CallbackManager::new();
        assert_eq!(manager.on_step_begin(&CallbackContext::default()), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_on_step_end_continue() {
        let mut manager = CallbackManager::new();
        assert_eq!(manager.on_step_end(&CallbackContext::default()), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_on_epoch_begin_continue() {
        let mut manager = CallbackManager::new();
        assert_eq!(manager.on_epoch_begin(&CallbackContext::default()), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_stop_epoch_begin_does_not_call_second() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct StopEpochBegin {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for StopEpochBegin {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopEpochBegin"
            }
        }

        struct CountEpochBegin {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountEpochBegin {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountEpochBegin"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(StopEpochBegin { count: count.clone() });
        manager.add(CountEpochBegin { count: count.clone() });

        let action = manager.on_epoch_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 1); // second never called
    }

    #[test]
    fn test_callback_manager_stop_epoch_end_does_not_call_second() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct StopEpochEnd {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for StopEpochEnd {
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopEpochEnd"
            }
        }

        struct CountEpochEnd {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountEpochEnd {
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountEpochEnd"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(StopEpochEnd { count: count.clone() });
        manager.add(CountEpochEnd { count: count.clone() });

        let action = manager.on_epoch_end(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_manager_stop_step_begin_does_not_call_second() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct StopStepBegin {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for StopStepBegin {
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopStepBegin"
            }
        }

        struct CountStepBegin {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountStepBegin {
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountStepBegin"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(StopStepBegin { count: count.clone() });
        manager.add(CountStepBegin { count: count.clone() });

        let action = manager.on_step_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_manager_stop_step_end_does_not_call_second() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct StopStepEnd {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for StopStepEnd {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopStepEnd"
            }
        }

        struct CountStepEnd {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountStepEnd {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountStepEnd"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(StopStepEnd { count: count.clone() });
        manager.add(CountStepEnd { count: count.clone() });

        let action = manager.on_step_end(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_manager_skip_epoch_does_not_call_second() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct SkipCallback {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for SkipCallback {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::SkipEpoch
            }
            fn name(&self) -> &'static str {
                "SkipCallback"
            }
        }

        struct ContinueCallback {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for ContinueCallback {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "ContinueCallback"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(SkipCallback { count: count.clone() });
        manager.add(ContinueCallback { count: count.clone() });

        let action = manager.on_epoch_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::SkipEpoch);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_callback_manager_with_context_values() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct EpochTracker {
            last_epoch: Arc<AtomicUsize>,
        }
        impl TrainerCallback for EpochTracker {
            fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                self.last_epoch.store(ctx.epoch, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "EpochTracker"
            }
        }

        let last_epoch = Arc::new(AtomicUsize::new(999));
        let mut manager = CallbackManager::new();
        manager.add(EpochTracker { last_epoch: last_epoch.clone() });

        let mut ctx = CallbackContext::default();
        ctx.epoch = 42;
        manager.on_epoch_begin(&ctx);
        assert_eq!(last_epoch.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_callback_manager_train_end_all_called() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountCallback {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountCallback {
            fn on_train_end(&mut self, _: &CallbackContext) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            fn name(&self) -> &'static str {
                "CountCallback"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        for _ in 0..5 {
            manager.add(CountCallback { count: count.clone() });
        }
        assert_eq!(manager.len(), 5);

        manager.on_train_end(&CallbackContext::default());
        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    // ── test_cov4 additional coverage tests ────────────────────────

    #[test]
    fn test_cov4_manager_full_lifecycle() {
        // Exercise complete train begin→step begin→step end→epoch begin→epoch end→train end flow
        use std::sync::Arc;

        struct LifecycleCallback {
            events: Arc<std::sync::Mutex<Vec<String>>>,
        }
        impl TrainerCallback for LifecycleCallback {
            fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                self.events.lock().unwrap().push(format!("train_begin:{}", ctx.epoch));
                CallbackAction::Continue
            }
            fn on_train_end(&mut self, ctx: &CallbackContext) {
                self.events.lock().unwrap().push(format!("train_end:{}", ctx.epoch));
            }
            fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                self.events.lock().unwrap().push(format!("epoch_begin:{}", ctx.epoch));
                CallbackAction::Continue
            }
            fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
                self.events.lock().unwrap().push(format!("epoch_end:{}", ctx.epoch));
                CallbackAction::Continue
            }
            fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                self.events.lock().unwrap().push(format!("step_begin:{}", ctx.step));
                CallbackAction::Continue
            }
            fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
                self.events.lock().unwrap().push(format!("step_end:{}", ctx.step));
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "LifecycleCallback"
            }
        }

        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut manager = CallbackManager::new();
        manager.add(LifecycleCallback { events: events.clone() });

        let mut ctx = CallbackContext::default();
        ctx.max_epochs = 2;
        ctx.steps_per_epoch = 3;

        manager.on_train_begin(&ctx);
        for epoch in 0..2 {
            ctx.epoch = epoch;
            manager.on_epoch_begin(&ctx);
            for step in 0..3 {
                ctx.step = step;
                manager.on_step_begin(&ctx);
                manager.on_step_end(&ctx);
            }
            manager.on_epoch_end(&ctx);
        }
        manager.on_train_end(&ctx);

        let ev = events.lock().unwrap();
        assert_eq!(ev[0], "train_begin:0");
        assert_eq!(ev[1], "epoch_begin:0");
        assert_eq!(ev[2], "step_begin:0");
        assert_eq!(ev[3], "step_end:0");
        assert!(ev.len() >= 16); // 1+2*(1+3*2+1)+1 = 18
        assert_eq!(*ev.last().unwrap(), "train_end:1");
    }

    #[test]
    fn test_cov4_manager_mixed_callbacks_epoch_end() {
        // Mix callbacks with different epoch_end behaviors: first continues, second stops
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct ContinueTracker {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for ContinueTracker {
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "ContinueTracker"
            }
        }

        struct StopTracker {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for StopTracker {
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopTracker"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        // First callback continues, second stops → both called, second triggers stop
        manager.add(ContinueTracker { count: count.clone() });
        manager.add(StopTracker { count: count.clone() });

        let action = manager.on_epoch_end(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 2); // both were called
    }

    #[test]
    fn test_cov4_manager_mixed_callbacks_step_end() {
        // Two continue then stop: all three called
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct ContinueCb {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for ContinueCb {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "ContinueCb"
            }
        }

        struct StopCb {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for StopCb {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCb"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(ContinueCb { count: count.clone() });
        manager.add(ContinueCb { count: count.clone() });
        manager.add(StopCb { count: count.clone() });

        let action = manager.on_step_end(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_cov4_manager_ctx_with_rich_fields() {
        // Use a context with all fields populated
        use std::sync::Arc;

        struct FieldChecker {
            verified: Arc<std::sync::atomic::AtomicBool>,
        }
        impl TrainerCallback for FieldChecker {
            fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
                if ctx.epoch == 3
                    && ctx.max_epochs == 10
                    && ctx.step == 7
                    && ctx.steps_per_epoch == 100
                    && ctx.global_step == 307
                    && (ctx.loss - 0.42).abs() < 1e-5
                    && (ctx.lr - 1e-4).abs() < 1e-8
                    && ctx.best_loss == Some(0.30)
                    && ctx.val_loss == Some(0.50)
                    && (ctx.elapsed_secs - 123.4).abs() < 0.1
                {
                    self.verified.store(true, std::sync::atomic::Ordering::SeqCst);
                }
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "FieldChecker"
            }
        }

        let verified = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut manager = CallbackManager::new();
        manager.add(FieldChecker { verified: verified.clone() });

        let ctx = CallbackContext {
            epoch: 3,
            max_epochs: 10,
            step: 7,
            steps_per_epoch: 100,
            global_step: 307,
            loss: 0.42,
            lr: 1e-4,
            best_loss: Some(0.30),
            val_loss: Some(0.50),
            elapsed_secs: 123.4,
        };

        manager.on_step_end(&ctx);
        assert!(verified.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_cov4_manager_multiple_adds() {
        let mut manager = CallbackManager::new();
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());

        manager.add(ProgressCallback::new(10));
        manager.add(ProgressCallback::new(20));
        manager.add(EarlyStopping::new(5, 0.001));
        assert_eq!(manager.len(), 3);
        assert!(!manager.is_empty());
    }

    #[test]
    fn test_cov4_manager_train_begin_multiple_continue() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountCb {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountCb {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountCb"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountCb { count: count.clone() });
        manager.add(CountCb { count: count.clone() });
        manager.add(CountCb { count: count.clone() });

        let action = manager.on_train_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_cov4_manager_step_begin_multiple_continue() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountCb {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountCb {
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountCb"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountCb { count: count.clone() });
        manager.add(CountCb { count: count.clone() });

        let action = manager.on_step_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_cov4_manager_epoch_begin_multiple_continue() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountCb {
            count: Arc<AtomicUsize>,
        }
        impl TrainerCallback for CountCb {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountCb"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountCb { count: count.clone() });
        manager.add(CountCb { count: count.clone() });
        manager.add(CountCb { count: count.clone() });

        let action = manager.on_epoch_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_cov4_manager_train_end_empty() {
        let mut manager = CallbackManager::new();
        // Should not panic with no callbacks
        manager.on_train_end(&CallbackContext::default());
    }

    #[test]
    fn test_cov4_manager_early_stopping_with_improvement() {
        let mut manager = CallbackManager::new();
        manager.add(EarlyStopping::new(3, 0.001));

        let mut ctx = CallbackContext::default();

        // Epoch 0: loss=1.0
        ctx.epoch = 0;
        ctx.loss = 1.0;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Epoch 1: loss improves to 0.5
        ctx.epoch = 1;
        ctx.loss = 0.5;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Epoch 2: loss worsens to 0.6 (1 epoch no improvement)
        ctx.epoch = 2;
        ctx.loss = 0.6;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Epoch 3: loss improves again to 0.3 — resets patience
        ctx.epoch = 3;
        ctx.loss = 0.3;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Epoch 4-6: no improvement
        for i in 4..7 {
            ctx.epoch = i;
            ctx.loss = 0.35;
            let action = manager.on_epoch_end(&ctx);
            if i == 6 {
                assert_eq!(action, CallbackAction::Stop);
            } else {
                assert_eq!(action, CallbackAction::Continue);
            }
        }
    }

    #[test]
    fn test_cov4_manager_default_new_equivalent() {
        let m1 = CallbackManager::new();
        let m2 = CallbackManager::default();
        assert_eq!(m1.len(), m2.len());
        assert_eq!(m1.is_empty(), m2.is_empty());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::train::callback::EarlyStopping;
    use proptest::prelude::*;

    proptest! {
        /// Callback manager should propagate stop action
        #[test]
        fn callback_manager_propagates_stop(
            patience in 1usize..5,
        ) {
            let mut manager = CallbackManager::new();
            manager.add(EarlyStopping::new(patience, 0.001));

            let mut ctx = CallbackContext::default();
            ctx.loss = 1.0;

            // Should continue until patience exhausted
            for epoch in 0..patience {
                ctx.epoch = epoch;
                let action = manager.on_epoch_end(&ctx);
                if epoch < patience - 1 {
                    prop_assert_eq!(action, CallbackAction::Continue);
                }
            }

            // Final epoch should stop
            ctx.epoch = patience;
            prop_assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Stop);
        }

        /// Multiple callbacks should all fire
        #[test]
        fn multiple_callbacks_fire(
            num_callbacks in 1usize..5,
        ) {
            use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

            struct CounterCallback {
                counter: Arc<AtomicUsize>,
            }

            impl TrainerCallback for CounterCallback {
                fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                    self.counter.fetch_add(1, Ordering::SeqCst);
                    CallbackAction::Continue
                }
                fn on_train_end(&mut self, _: &CallbackContext) {}
                fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn name(&self) -> &'static str { "CounterCallback" }
            }

            let counter = Arc::new(AtomicUsize::new(0));
            let mut manager = CallbackManager::new();

            for _ in 0..num_callbacks {
                manager.add(CounterCallback { counter: counter.clone() });
            }

            let ctx = CallbackContext::default();
            manager.on_train_begin(&ctx);

            prop_assert_eq!(counter.load(Ordering::SeqCst), num_callbacks);
        }
    }
}
