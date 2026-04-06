//! Tests for the trainer module.

use super::*;
use crate::hf_pipeline::loader::TeacherModel;
use crate::hf_pipeline::SafeTensorsTeacher;
use std::time::Duration;

// =========================================================================
// TrainerConfig Tests
// =========================================================================

#[test]
fn test_trainer_config_default() {
    let config = TrainerConfig::default();
    assert!(config.teacher_model.is_empty());
    assert!(config.student_model.is_empty());
    assert_eq!(config.epochs, 3);
    assert_eq!(config.log_every_n_steps, 10);
}

#[test]
fn test_trainer_config_builder() {
    let config = TrainerConfig::new("teacher/model", "student/model")
        .temperature(6.0)
        .alpha(0.8)
        .epochs(5)
        .output_dir("/tmp/distill");

    assert_eq!(config.teacher_model, "teacher/model");
    assert_eq!(config.student_model, "student/model");
    assert_eq!(config.distillation_loss.temperature, 6.0);
    assert_eq!(config.distillation_loss.alpha, 0.8);
    assert_eq!(config.epochs, 5);
}

#[test]
fn test_trainer_config_progressive() {
    let config = TrainerConfig::new("t", "s").with_progressive(vec![(0, 2), (1, 5)]);

    assert!(config.progressive.is_some());
    let prog = config.progressive.expect("config should be valid");
    assert_eq!(prog.layer_mapping.len(), 2);
}

#[test]
fn test_trainer_config_attention_transfer() {
    let config = TrainerConfig::new("t", "s").with_attention_transfer(0.2);

    assert!(config.attention_transfer.is_some());
    let at = config.attention_transfer.expect("config should be valid");
    assert_eq!(at.weight, 0.2);
}

// =========================================================================
// TrainingState Tests
// =========================================================================

#[test]
fn test_training_state_new() {
    let state = TrainingState::new();
    assert_eq!(state.epoch, 0);
    assert_eq!(state.global_step, 0);
    assert_eq!(state.best_val_loss, f32::INFINITY);
}

#[test]
fn test_training_state_step() {
    let mut state = TrainingState::new();
    state.step();
    assert_eq!(state.global_step, 1);
    assert_eq!(state.epoch_step, 1);

    state.step();
    assert_eq!(state.global_step, 2);
}

#[test]
fn test_training_state_new_epoch() {
    let mut state = TrainingState::new();
    state.step();
    state.step();
    state.new_epoch();

    assert_eq!(state.epoch, 1);
    assert_eq!(state.epoch_step, 0);
    assert_eq!(state.global_step, 2); // Global step unchanged
}

#[test]
fn test_training_state_record_loss() {
    let mut state = TrainingState::new();
    state.record_loss(1.5);
    state.step();
    state.record_loss(1.2);
    state.step();

    assert_eq!(state.loss_history.len(), 2);
    assert_eq!(state.loss_history[0], (0, 1.5));
    assert_eq!(state.loss_history[1], (1, 1.2));
}

#[test]
fn test_training_state_record_val_loss_improvement() {
    let mut state = TrainingState::new();
    let improved = state.record_val_loss(1.0);
    assert!(improved);
    assert_eq!(state.best_val_loss, 1.0);

    let improved = state.record_val_loss(0.8);
    assert!(improved);
    assert_eq!(state.best_val_loss, 0.8);

    let improved = state.record_val_loss(0.9);
    assert!(!improved);
    assert_eq!(state.best_val_loss, 0.8);
}

#[test]
fn test_training_state_avg_loss() {
    let mut state = TrainingState::new();
    state.record_loss(1.0);
    state.step();
    state.record_loss(2.0);
    state.step();
    state.record_loss(3.0);

    let avg = state.avg_loss(2).expect("operation should succeed");
    assert!((avg - 2.5).abs() < 0.01);

    let avg_all = state.avg_loss(10).expect("operation should succeed");
    assert!((avg_all - 2.0).abs() < 0.01);
}

#[test]
fn test_training_state_avg_loss_empty() {
    let state = TrainingState::new();
    assert!(state.avg_loss(10).is_none());
}

// =========================================================================
// DistillationTrainer Tests
// =========================================================================

#[test]
fn test_trainer_creation() {
    let config = TrainerConfig::new("teacher", "student");
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let trainer = DistillationTrainer::new(config, teacher);

    assert_eq!(trainer.config.teacher_model, "teacher");
    assert_eq!(trainer.state().global_step, 0);
}

#[test]
fn test_trainer_is_parameter_efficient() {
    let mut config = TrainerConfig::new("t", "s");
    let teacher = SafeTensorsTeacher::mock(12, 768);

    // Default uses LoRA
    let trainer = DistillationTrainer::new(config.clone(), SafeTensorsTeacher::mock(12, 768));
    assert!(trainer.is_parameter_efficient());

    // Full fine-tuning is not parameter efficient
    config.fine_tune = config.fine_tune.full_fine_tune();
    let trainer = DistillationTrainer::new(config, teacher);
    assert!(!trainer.is_parameter_efficient());
}

#[test]
fn test_trainer_compute_loss() {
    use ndarray::Array2;

    let config = TrainerConfig::new("t", "s");
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let trainer = DistillationTrainer::new(config, teacher);

    let student_logits =
        Array2::from_shape_vec((2, 10), vec![1.0; 20]).expect("operation should succeed");
    let teacher_logits =
        Array2::from_shape_vec((2, 10), vec![1.1; 20]).expect("operation should succeed");
    let targets = vec![5, 3];

    let loss =
        trainer.compute_loss(&student_logits, &teacher_logits, &targets, None, None, None, None);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_trainer_compute_loss_with_progressive() {
    use ndarray::Array2;

    let config = TrainerConfig::new("t", "s").with_progressive(vec![(0, 0)]);
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let trainer = DistillationTrainer::new(config, teacher);

    let student_logits =
        Array2::from_shape_vec((2, 10), vec![1.0; 20]).expect("operation should succeed");
    let teacher_logits =
        Array2::from_shape_vec((2, 10), vec![1.1; 20]).expect("operation should succeed");
    let targets = vec![5, 3];

    let sh = vec![Array2::<f32>::zeros((2, 768))];
    let th = vec![Array2::<f32>::ones((2, 768))];

    let loss = trainer.compute_loss(
        &student_logits,
        &teacher_logits,
        &targets,
        Some(&sh),
        Some(&th),
        None,
        None,
    );

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_trainer_simulate_step() {
    let config = TrainerConfig::new("t", "s");
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let mut trainer = DistillationTrainer::new(config, teacher);

    trainer.simulate_step(1.5);
    assert_eq!(trainer.state().global_step, 1);
    assert_eq!(trainer.state().loss_history.len(), 1);

    trainer.simulate_step(1.2);
    assert_eq!(trainer.state().global_step, 2);
}

#[test]
fn test_trainer_simulate_epoch() {
    let config = TrainerConfig::new("t", "s");
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let mut trainer = DistillationTrainer::new(config, teacher);

    trainer.simulate_step(1.0);
    trainer.simulate_step(1.0);
    trainer.simulate_epoch();

    assert_eq!(trainer.state().epoch, 1);
    assert_eq!(trainer.state().epoch_step, 0);
}

#[test]
fn test_trainer_estimate_memory() {
    let config = TrainerConfig::new("t", "s");
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let trainer = DistillationTrainer::new(config, teacher);

    let mem = trainer.estimate_total_memory();
    assert!(mem > 0);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_training_state_default() {
    let state = TrainingState::default();
    assert_eq!(state.epoch, 0);
    assert_eq!(state.global_step, 0);
}

#[test]
fn test_training_state_elapsed() {
    let state = TrainingState::new();
    std::thread::sleep(Duration::from_millis(5));
    assert!(state.elapsed().as_millis() >= 4);
}

#[test]
fn test_training_state_steps_per_second() {
    let mut state = TrainingState::new();
    state.step();
    state.step();
    std::thread::sleep(Duration::from_millis(10));
    // Should be positive if steps completed
    let sps = state.steps_per_second();
    assert!(sps > 0.0);
}

#[test]
fn test_training_state_steps_per_second_zero_time() {
    // Fresh state with no elapsed time should return 0 or positive
    let state = TrainingState::new();
    let sps = state.steps_per_second();
    assert!(sps >= 0.0);
}

#[test]
fn test_training_state_eta() {
    let mut state = TrainingState::new();
    state.step();
    std::thread::sleep(Duration::from_millis(10));
    let eta = state.eta(100);
    // ETA should be some duration (could be very large or zero)
    assert!(eta.as_secs_f32() >= 0.0);
}

#[test]
fn test_training_state_eta_zero_steps() {
    let state = TrainingState::new();
    let eta = state.eta(100);
    assert_eq!(eta, Duration::ZERO);
}

#[test]
fn test_trainer_teacher_ref() {
    let config = TrainerConfig::new("t", "s");
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let trainer = DistillationTrainer::new(config, teacher);

    // Access teacher reference
    let teacher_ref = trainer.teacher();
    assert!(teacher_ref.param_count() > 0);
}

#[test]
fn test_trainer_compute_loss_with_attention_transfer() {
    use ndarray::Array2;

    let config = TrainerConfig::new("t", "s").with_attention_transfer(0.5);
    let teacher = SafeTensorsTeacher::mock(12, 768);
    let trainer = DistillationTrainer::new(config, teacher);

    let student_logits =
        Array2::from_shape_vec((2, 10), vec![1.0; 20]).expect("operation should succeed");
    let teacher_logits =
        Array2::from_shape_vec((2, 10), vec![1.1; 20]).expect("operation should succeed");
    let targets = vec![5, 3];

    // Attention maps (simulating attention scores)
    let sa = vec![Array2::<f32>::zeros((2, 12))];
    let ta = vec![Array2::<f32>::ones((2, 12))];

    let loss = trainer.compute_loss(
        &student_logits,
        &teacher_logits,
        &targets,
        None,
        None,
        Some(&sa),
        Some(&ta),
    );

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_trainer_state_clone() {
    let mut state = TrainingState::new();
    state.step();
    state.record_loss(1.0);

    let cloned = state.clone();
    assert_eq!(state.global_step, cloned.global_step);
    assert_eq!(state.loss_history.len(), cloned.loss_history.len());
}

#[test]
fn test_trainer_config_clone() {
    let config = TrainerConfig::new("t", "s").temperature(4.0).epochs(10);

    let cloned = config.clone();
    assert_eq!(config.epochs, cloned.epochs);
    assert_eq!(config.distillation_loss.temperature, cloned.distillation_loss.temperature);
}
