//! Tests for training loops

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::optim::Adam;
    use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};
    use crate::train::{Batch, EarlyStopping, MSELoss, TrainConfig, Trainer};
    use crate::Tensor;

    #[test]
    fn test_train_with_callbacks() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(EarlyStopping::new(2, 0.0001));

        // Batches that produce constant loss (will trigger early stopping)
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
        ];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);

        // Should stop early due to no improvement
        assert!(result.stopped_early);
        assert!(result.final_epoch < 10);
        assert!(result.elapsed_secs > 0.0);
        assert!(result.best_loss > 0.0);
    }

    #[test]
    fn test_train_runs_all_epochs() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        // No callbacks - should run all epochs

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![2.0, 3.0], false),
        )];

        let result = trainer.train(3, || batches.clone(), std::clone::Clone::clone);

        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 3);
    }

    #[test]
    fn test_train_result_fields() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(2, || batches.clone(), std::clone::Clone::clone);

        // Verify all fields are populated
        assert!(result.final_loss.is_finite());
        assert!(result.best_loss.is_finite());
        assert!(
            result.best_loss <= result.final_loss
                || (result.best_loss - result.final_loss).abs() < 0.001
        );
        assert!(result.elapsed_secs >= 0.0);
    }

    #[test]
    fn test_gradient_accumulation() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100).with_gradient_accumulation(2);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create 4 batches - with accum_steps=2, we get 2 optimizer steps per epoch
        let batches = vec![
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
            Batch::new(
                Tensor::from_vec(vec![1.0, 2.0], false),
                Tensor::from_vec(vec![2.0, 3.0], false),
            ),
        ];

        let result = trainer.train(1, || batches.clone(), std::clone::Clone::clone);

        // Should complete successfully
        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 1);
        assert!(result.final_loss.is_finite());
        // 4 batches = 4 steps
        assert_eq!(trainer.metrics.steps, 4);
    }

    #[test]
    fn test_gradient_accumulation_partial_window() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100).with_gradient_accumulation(3);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        // Create 5 batches with accum_steps=3
        // Optimizer steps at: batch 2 (0,1,2), batch 4 (3,4 - partial)
        let batches = vec![
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
        ];

        let result = trainer.train(1, || batches.clone(), std::clone::Clone::clone);

        assert!(!result.stopped_early);
        assert_eq!(trainer.metrics.steps, 5);
        assert!(result.final_loss.is_finite());
    }

    #[test]
    fn test_early_stopping_monitor_validation() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(EarlyStopping::new(2, 0.0001).monitor_validation());

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);

        // Should run and eventually stop (training loss is used as fallback)
        assert!(result.final_loss.is_finite());
    }

    #[test]
    fn test_train_with_val() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let train_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![2.0, 3.0], false),
        )];

        let val_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![2.5, 3.5], false),
        )];

        let result = trainer.train_with_val(
            3,
            || train_batches.clone(),
            || val_batches.clone(),
            std::clone::Clone::clone,
        );

        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 3);
        assert!(result.final_loss.is_finite());
        assert!(result.best_loss.is_finite());
        // Val losses should be tracked
        assert_eq!(trainer.metrics.val_losses.len(), 3);
    }

    #[test]
    fn test_train_with_val_early_stopping() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(EarlyStopping::new(2, 0.0001).monitor_validation());

        let train_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        // Validation with same targets - constant val loss triggers early stopping
        let val_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train_with_val(
            100,
            || train_batches.clone(),
            || val_batches.clone(),
            std::clone::Clone::clone,
        );

        // Should stop early due to no val improvement
        assert!(result.stopped_early);
        assert!(result.final_epoch < 100);
    }

    #[test]
    fn test_train_with_val_empty_validation() {
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let train_batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        // Empty validation - should still work
        let val_batches: Vec<Batch> = vec![];

        let result = trainer.train_with_val(
            2,
            || train_batches.clone(),
            || val_batches.clone(),
            std::clone::Clone::clone,
        );

        assert!(!result.stopped_early);
        assert_eq!(result.final_epoch, 2);
        // No val losses recorded
        assert_eq!(trainer.metrics.val_losses.len(), 0);
    }

    #[test]
    fn test_train_stops_at_train_begin() {
        struct StopAtBeginCallback;
        impl TrainerCallback for StopAtBeginCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopAtBegin"
            }
        }

        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::default();

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(StopAtBeginCallback);

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);
        assert!(result.stopped_early);
        assert_eq!(result.final_epoch, 0);
    }

    #[test]
    fn test_train_with_epoch_skip() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct SkipFirstEpochCallback {
            skipped: Arc<AtomicUsize>,
        }
        impl TrainerCallback for SkipFirstEpochCallback {
            fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                if ctx.epoch == 0 {
                    self.skipped.fetch_add(1, Ordering::SeqCst);
                    CallbackAction::SkipEpoch
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &'static str {
                "SkipFirstEpoch"
            }
        }

        let skipped = Arc::new(AtomicUsize::new(0));
        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(SkipFirstEpochCallback { skipped: skipped.clone() });

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(3, || batches.clone(), std::clone::Clone::clone);
        assert!(!result.stopped_early);
        assert_eq!(skipped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_train_with_step_begin_stop() {
        struct StopAtStepBeginCallback;
        impl TrainerCallback for StopAtStepBeginCallback {
            fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
                if ctx.step >= 1 {
                    CallbackAction::Stop
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &'static str {
                "StopAtStepBegin"
            }
        }

        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(StopAtStepBeginCallback);

        let batches = vec![
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
            Batch::new(Tensor::from_vec(vec![1.0], false), Tensor::from_vec(vec![2.0], false)),
        ];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);
        assert!(result.stopped_early);
    }

    #[test]
    fn test_train_with_step_end_stop() {
        struct StopAtStepEndCallback;
        impl TrainerCallback for StopAtStepEndCallback {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopAtStepEnd"
            }
        }

        let params = vec![Tensor::from_vec(vec![1.0], true)];
        let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));
        trainer.add_callback(StopAtStepEndCallback);

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0], false),
            Tensor::from_vec(vec![2.0], false),
        )];

        let result = trainer.train(10, || batches.clone(), std::clone::Clone::clone);
        assert!(result.stopped_early);
        // Only one step executed
        assert_eq!(trainer.metrics.steps, 1);
    }

    #[test]
    fn test_train_with_grad_clipping() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], true)];
        let optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
        let config = TrainConfig::new().with_log_interval(100).with_grad_clip(1.0);

        let mut trainer = Trainer::new(params, Box::new(optimizer), config);
        trainer.set_loss(Box::new(MSELoss));

        let batches = vec![Batch::new(
            Tensor::from_vec(vec![1.0, 2.0], false),
            Tensor::from_vec(vec![100.0, 200.0], false), // Large targets for big gradients
        )];

        let result = trainer.train(2, || batches.clone(), std::clone::Clone::clone);
        assert!(!result.stopped_early);
        assert!(result.final_loss.is_finite());
    }
}
