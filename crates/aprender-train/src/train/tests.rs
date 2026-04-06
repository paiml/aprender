//! Integration tests for training module

#![allow(clippy::field_reassign_with_default)]

use super::*;
use crate::optim::Adam;
use crate::Tensor;

#[test]
fn test_end_to_end_training() {
    // Setup a simple model
    let params = vec![Tensor::zeros(5, true)];
    let optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8);
    let config = TrainConfig::new().with_log_interval(100);

    let mut trainer = Trainer::new(params, Box::new(optimizer), config);
    trainer.set_loss(Box::new(MSELoss));

    // Create training data
    let batches = vec![
        Batch::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false),
            Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], false),
        ),
        Batch::new(
            Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], false),
            Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0], false),
        ),
    ];

    // Train for multiple epochs (simple identity function)
    let initial_loss = trainer.train_epoch(batches.clone(), std::clone::Clone::clone);

    // Train a few more epochs
    for _ in 0..3 {
        trainer.train_epoch(batches.clone(), std::clone::Clone::clone);
    }

    let final_loss =
        trainer.metrics.losses.last().copied().expect("collection should not be empty");

    // Loss should be finite
    assert!(initial_loss.is_finite());
    assert!(final_loss.is_finite());
    assert_eq!(trainer.metrics.epoch, 4);
}

#[test]
fn test_metrics_tracking() {
    let params = vec![Tensor::zeros(5, true)];
    let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let config = TrainConfig::default();

    let mut trainer = Trainer::new(params, Box::new(optimizer), config);
    trainer.set_loss(Box::new(MSELoss));

    let batch = Batch::new(
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], false),
        Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0], false),
    );

    // Train for 3 epochs
    for _ in 0..3 {
        trainer.train_epoch(vec![batch.clone()], std::clone::Clone::clone);
    }

    assert_eq!(trainer.metrics.epoch, 3);
    assert_eq!(trainer.metrics.losses.len(), 3);
    assert!(trainer.metrics.best_loss().is_some());
}

#[test]
fn test_gradient_clipping() {
    let params = vec![Tensor::from_vec(vec![100.0, 200.0], true)];
    let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Config with gradient clipping
    let config_with_clip = TrainConfig::new().with_grad_clip(1.0);
    let mut trainer_clip = Trainer::new(params.clone(), Box::new(optimizer), config_with_clip);
    trainer_clip.set_loss(Box::new(MSELoss));

    // Config without gradient clipping
    let optimizer2 = Adam::new(0.01, 0.9, 0.999, 1e-8);
    let config_no_clip = TrainConfig::new().without_grad_clip();
    let mut trainer_no_clip = Trainer::new(params.clone(), Box::new(optimizer2), config_no_clip);
    trainer_no_clip.set_loss(Box::new(MSELoss));

    let batch = Batch::new(
        Tensor::from_vec(vec![10.0, 20.0], false),
        Tensor::from_vec(vec![0.0, 0.0], false),
    );

    let loss_clip = trainer_clip.train_step(&batch, std::clone::Clone::clone);
    let loss_no_clip = trainer_no_clip.train_step(&batch, std::clone::Clone::clone);

    // Both should produce valid losses
    assert!(loss_clip.is_finite());
    assert!(loss_no_clip.is_finite());
}

#[test]
fn test_learning_rate_update() {
    let params = vec![Tensor::zeros(5, true)];
    let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
    let config = TrainConfig::default();

    let mut trainer = Trainer::new(params, Box::new(optimizer), config);

    assert_eq!(trainer.lr(), 0.001);

    trainer.set_lr(0.01);
    assert_eq!(trainer.lr(), 0.01);
}
