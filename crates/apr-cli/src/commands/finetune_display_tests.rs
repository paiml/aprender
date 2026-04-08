// ═══════════════════════════════════════════════════════════════════════════════
// Tests for display_train_result (finetune.rs:1585)
// ═══════════════════════════════════════════════════════════════════════════════

use super::*;
use entrenar::finetune::{EpochMetrics, TrainResult};
use tempfile::tempdir;

fn make_epoch(epoch: usize, train_loss: f32, val_loss: f32) -> EpochMetrics {
    EpochMetrics {
        epoch,
        train_loss,
        train_accuracy: 0.8 + (epoch as f32) * 0.02,
        val_loss,
        val_accuracy: 0.75 + (epoch as f32) * 0.03,
        learning_rate: 2e-4 * 0.9_f32.powi(epoch as i32),
        epoch_time_ms: 1000 + epoch as u64 * 100,
        samples_per_sec: 50.0 + epoch as f32,
    }
}

fn make_train_result(epochs: usize) -> TrainResult {
    let epoch_metrics: Vec<EpochMetrics> = (0..epochs)
        .map(|i| make_epoch(i, 2.0 - (i as f32) * 0.3, 2.5 - (i as f32) * 0.2))
        .collect();
    let best_epoch = epochs.saturating_sub(1);
    let best_val_loss = epoch_metrics.last().map_or(0.0, |m| m.val_loss);
    TrainResult {
        epoch_metrics,
        best_epoch,
        best_val_loss,
        stopped_early: false,
        total_time_ms: epochs as u64 * 1100,
    }
}

// ── JSON output tests ──────────────────────────────────────────────────

#[test]
fn test_display_train_result_json_basic() {
    let result = make_train_result(3);
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr,safetensors", true);
}

#[test]
fn test_display_train_result_json_empty_epochs() {
    let result = TrainResult {
        epoch_metrics: vec![],
        best_epoch: 0,
        best_val_loss: 0.0,
        stopped_early: false,
        total_time_ms: 0,
    };
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr", true);
}

#[test]
fn test_display_train_result_json_single_epoch() {
    let result = make_train_result(1);
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "safetensors", true);
}

#[test]
fn test_display_train_result_json_early_stopping() {
    let mut result = make_train_result(5);
    result.stopped_early = true;
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr", true);
}

// ── Text output tests ──────────────────────────────────────────────────

#[test]
fn test_display_train_result_text_basic() {
    let result = make_train_result(3);
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr,safetensors", false);
}

#[test]
fn test_display_train_result_text_empty_epochs() {
    let result = TrainResult {
        epoch_metrics: vec![],
        best_epoch: 0,
        best_val_loss: 0.0,
        stopped_early: false,
        total_time_ms: 0,
    };
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr", false);
}

#[test]
fn test_display_train_result_text_early_stopping() {
    let mut result = make_train_result(5);
    result.stopped_early = true;
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "safetensors", false);
}

#[test]
fn test_display_train_result_text_single_epoch() {
    let result = make_train_result(1);
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr", false);
}

#[test]
fn test_display_train_result_text_many_epochs() {
    let result = make_train_result(20);
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr,safetensors", false);
}

// ── Edge cases ─────────────────────────────────────────────────────────

#[test]
fn test_display_train_result_zero_time() {
    let result = TrainResult {
        epoch_metrics: vec![make_epoch(0, 1.0, 1.5)],
        best_epoch: 0,
        best_val_loss: 1.5,
        stopped_early: false,
        total_time_ms: 0,
    };
    let dir = tempdir().expect("tempdir");
    // Should handle zero time without panic (division by zero protection)
    display_train_result(&result, dir.path(), "apr", true);
    display_train_result(&result, dir.path(), "apr", false);
}

#[test]
fn test_display_train_result_best_epoch_not_last() {
    let mut result = make_train_result(5);
    result.best_epoch = 2; // Not the last epoch
    result.best_val_loss = 1.5;
    let dir = tempdir().expect("tempdir");
    display_train_result(&result, dir.path(), "apr", false);
}
