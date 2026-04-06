//! Tests for TerminalMonitorCallback.

use std::time::Instant;

use crate::train::callback::{CallbackAction, CallbackContext, TrainerCallback};
use crate::train::tui::capability::{DashboardLayout, TerminalMode};
use crate::train::tui::progress::ProgressBar;

use super::render::{render_alerts, CallbackRenderer};
use super::TerminalMonitorCallback;

fn make_test_context() -> CallbackContext {
    CallbackContext {
        epoch: 0,
        max_epochs: 10,
        step: 0,
        steps_per_epoch: 100,
        global_step: 0,
        loss: 1.0,
        lr: 0.001,
        best_loss: None,
        val_loss: None,
        elapsed_secs: 0.0,
    }
}

#[test]
fn test_terminal_monitor_callback_new() {
    let callback = TerminalMonitorCallback::new();
    assert_eq!(callback.sparkline_width, 20);
    assert_eq!(callback.model_name, "model");
}

#[test]
fn test_terminal_monitor_callback_builders() {
    let callback = TerminalMonitorCallback::new()
        .mode(TerminalMode::Ascii)
        .layout(DashboardLayout::Full)
        .model_name("test_model")
        .sparkline_width(30)
        .refresh_interval_ms(200);

    assert_eq!(callback.mode, TerminalMode::Ascii);
    assert_eq!(callback.layout, DashboardLayout::Full);
    assert_eq!(callback.model_name, "test_model");
    assert_eq!(callback.sparkline_width, 30);
}

#[test]
fn test_terminal_monitor_callback_render_minimal() {
    let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Minimal);
    let ctx = make_test_context();
    let output = callback.render(&ctx);
    assert!(output.contains("Epoch"));
    assert!(output.contains("loss="));
}

#[test]
fn test_terminal_monitor_callback_render_compact() {
    let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
    let ctx = make_test_context();
    let output = callback.render(&ctx);
    assert!(output.contains("Training"));
}

#[test]
fn test_terminal_monitor_callback_render_full() {
    let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
    let ctx = make_test_context();
    let output = callback.render(&ctx);
    assert!(output.contains("ENTRENAR"));
}

#[test]
fn test_terminal_monitor_callback_name() {
    let callback = TerminalMonitorCallback::new();
    assert_eq!(callback.name(), "TerminalMonitorCallback");
}

#[test]
fn test_terminal_monitor_callback_on_step_end() {
    let mut callback = TerminalMonitorCallback::new();
    let ctx = make_test_context();

    // Initialize
    callback.start_time = Instant::now();
    callback.progress = ProgressBar::new(1000, 30);

    let action = callback.on_step_end(&ctx);
    assert_eq!(action, CallbackAction::Continue);
    assert_eq!(callback.loss_buffer.len(), 1);
}

#[test]
fn test_terminal_monitor_callback_on_step_end_nan() {
    let mut callback = TerminalMonitorCallback::new();
    let mut ctx = make_test_context();
    ctx.loss = f32::NAN;

    callback.start_time = Instant::now();
    callback.progress = ProgressBar::new(1000, 30);

    let action = callback.on_step_end(&ctx);
    assert_eq!(action, CallbackAction::Stop);
}

#[test]
fn test_terminal_monitor_callback_with_val_loss() {
    let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
    let mut ctx = make_test_context();
    ctx.val_loss = Some(0.8);
    let output = callback.render(&ctx);
    assert!(output.contains("val="));
}

#[test]
fn test_terminal_monitor_callback_default() {
    let callback = TerminalMonitorCallback::default();
    assert_eq!(callback.sparkline_width, 20);
}

#[test]
fn test_terminal_monitor_callback_on_train_begin() {
    let mut callback = TerminalMonitorCallback::new();
    let ctx = make_test_context();
    let action = callback.on_train_begin(&ctx);
    assert_eq!(action, CallbackAction::Continue);
}

#[test]
fn test_terminal_monitor_callback_on_epoch_end() {
    let mut callback = TerminalMonitorCallback::new();
    let ctx = make_test_context();
    callback.on_train_begin(&ctx);
    let action = callback.on_epoch_end(&ctx);
    assert_eq!(action, CallbackAction::Continue);
}

#[test]
fn test_terminal_monitor_callback_on_train_end() {
    let mut callback = TerminalMonitorCallback::new();
    let ctx = make_test_context();
    callback.on_train_begin(&ctx);
    callback.loss_buffer.push(0.5);
    callback.on_train_end(&ctx);
    // No assertion - just verify it doesn't panic
}

#[test]
fn test_terminal_monitor_callback_render_full_with_val() {
    let mut callback = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
    // Push some validation losses
    callback.val_loss_buffer.push(0.9);
    callback.val_loss_buffer.push(0.8);
    let ctx = make_test_context();
    let output = callback.render(&ctx);
    assert!(output.contains("ENTRENAR"));
}

#[test]
fn test_terminal_monitor_callback_render_alerts_empty() {
    let callback = TerminalMonitorCallback::new();
    let alerts = render_alerts(&callback);
    assert!(alerts.is_empty());
}

#[test]
fn test_terminal_monitor_callback_render_alerts_with_alerts() {
    let mut callback = TerminalMonitorCallback::new();
    callback.andon.warning("Test warning");
    callback.andon.critical("Test critical");
    callback.andon.info("Test info");
    let alerts = render_alerts(&callback);
    assert!(!alerts.is_empty());
}

#[test]
fn test_terminal_monitor_callback_step_with_val_loss() {
    let mut callback = TerminalMonitorCallback::new();
    callback.on_train_begin(&make_test_context());

    let mut ctx = make_test_context();
    ctx.val_loss = Some(0.75);

    let action = callback.on_step_end(&ctx);
    assert_eq!(action, CallbackAction::Continue);
    assert_eq!(callback.val_loss_buffer.len(), 1);
}

#[test]
fn test_terminal_monitor_callback_render_compact_with_best() {
    let mut callback = TerminalMonitorCallback::new().layout(DashboardLayout::Compact);
    // Push losses to establish a minimum
    callback.loss_buffer.push(1.0);
    callback.loss_buffer.push(0.5);
    callback.loss_buffer.push(0.8);
    let ctx = make_test_context();
    let output = callback.render(&ctx);
    assert!(output.contains("best="));
}

#[test]
fn test_terminal_monitor_callback_render_full_empty_val() {
    let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Full);
    let ctx = make_test_context();
    let output = callback.render(&ctx);
    // Should render without validation spark since buffer is empty
    assert!(output.contains("ENTRENAR"));
}

#[test]
fn test_terminal_monitor_callback_print_display() {
    let callback = TerminalMonitorCallback::new().layout(DashboardLayout::Minimal);
    let ctx = make_test_context();
    // This will print to stdout, but shouldn't panic
    callback.print_display(&ctx);
}
