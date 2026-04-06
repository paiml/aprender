//! Real-Time Terminal Monitoring Example
//!
//! Demonstrates entrenar's TUI monitoring capabilities:
//! - MetricsBuffer for streaming metrics
//! - Sparklines for inline visualization
//! - Progress bar with Kalman-filtered ETA
//! - LossCurveDisplay with trueno-viz
//! - AndonSystem for health monitoring

use entrenar::train::{
    sparkline, sparkline_range, AndonSystem, LossCurveDisplay, MetricsBuffer, ProgressBar,
    ReferenceCurve, TerminalCapabilities, TerminalMode, SPARK_CHARS,
};
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== Real-Time Terminal Monitoring Demo ===\n");

    // Detect terminal capabilities
    demo_terminal_detection();

    // MetricsBuffer demo
    demo_metrics_buffer();

    // Sparkline demo
    demo_sparklines();

    // Progress bar demo
    demo_progress_bar();

    // Loss curve display demo
    demo_loss_curve();

    // Andon system demo
    demo_andon_system();

    // Reference curve demo
    demo_reference_curve();

    println!("\n=== Demo Complete ===");
}

fn demo_terminal_detection() {
    println!("--- Terminal Capabilities ---\n");

    let caps = TerminalCapabilities::detect();
    println!("  Unicode support: {}", caps.unicode);
    println!("  ANSI color: {}", caps.ansi_color);
    println!("  True color (24-bit): {}", caps.true_color);
    println!("  Terminal width: {} columns", caps.width);
    println!("  Recommended mode: {:?}", caps.recommended_mode());
    println!();
}

fn demo_metrics_buffer() {
    println!("--- MetricsBuffer Demo ---\n");

    let mut buffer = MetricsBuffer::new(10);

    // Simulate loss values
    let losses = [1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.31, 0.30, 0.29];

    for loss in &losses {
        buffer.push(*loss);
    }

    println!("  Capacity: {}", buffer.capacity());
    println!("  Length: {}", buffer.len());
    println!("  Last value: {:?}", buffer.last());
    println!("  Min: {:?}", buffer.min());
    println!("  Max: {:?}", buffer.max());
    println!("  Mean: {:?}", buffer.mean());
    println!("  Last 5: {:?}", buffer.last_n(5));
    println!();
}

fn demo_sparklines() {
    println!("--- Sparkline Demo ---\n");

    // Decreasing loss curve
    let losses: Vec<f32> = (0..20).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    println!("  Loss trend: {}", sparkline(&losses, 20));

    // Oscillating accuracy
    let acc: Vec<f32> =
        (0..20).map(|i| 0.5 + 0.3 * (i as f32 * 0.5).sin() + 0.15 * (i as f32 / 20.0)).collect();
    println!("  Accuracy:   {}", sparkline(&acc, 20));

    // Fixed range sparkline
    let normalized = sparkline_range(&losses, 20, 0.0, 1.0);
    println!("  Normalized: {normalized}");

    // Show sparkline characters
    println!("\n  Sparkline chars: {SPARK_CHARS:?}");
    println!();
}

fn demo_progress_bar() {
    println!("--- Progress Bar Demo ---\n");

    let total_steps = 20;
    let mut progress = ProgressBar::new(total_steps, 40);

    println!("  Simulating {total_steps} steps...\n");

    for step in 0..=total_steps {
        progress.update(step);

        print!("\r  {}", progress.render());
        std::io::Write::flush(&mut std::io::stdout()).expect("operation should succeed");

        // Simulate varying step durations
        let delay = 50 + (step % 3) * 10;
        thread::sleep(Duration::from_millis(delay as u64));
    }
    println!("\n");
}

fn demo_loss_curve() {
    println!("--- LossCurve Display Demo ---\n");

    let mut display = LossCurveDisplay::new(60, 15).terminal_mode(TerminalMode::Unicode);

    // Simulate training with decreasing loss
    for epoch in 0..30 {
        let t = epoch as f32 / 30.0;
        let train_loss = 1.0 * (-2.0 * t).exp() + 0.1 * (epoch as f32 * 0.5).sin().abs();
        let val_loss = train_loss * 1.15 + 0.05 * (epoch as f32 * 0.3).cos().abs();

        display.push_losses(train_loss, val_loss);
    }

    println!("  Epochs recorded: {}", display.epochs());
    println!("\n  Series Summary:");
    for (name, min, last, best_epoch) in display.summary() {
        println!(
            "    {}: min={:.4}, last={:.4}, best_epoch={:?}",
            name,
            min.unwrap_or(0.0),
            last.unwrap_or(0.0),
            best_epoch
        );
    }

    println!("\n  Terminal Render:");
    println!("{}", display.render_terminal());
    println!();
}

fn demo_andon_system() {
    println!("--- Andon System Demo ---\n");

    let mut andon = AndonSystem::new()
        .with_sigma_threshold(10.0)
        .with_stall_threshold(5)
        .with_stop_on_critical(false);

    // Normal losses
    let normal_losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3];
    for loss in &normal_losses {
        andon.check_loss(*loss);
    }
    println!("  After normal training: has_critical={}", andon.has_critical());

    // Stall (no improvement)
    for _ in 0..6 {
        andon.check_loss(0.3);
    }

    // Check for NaN detection
    let mut nan_andon = AndonSystem::new().with_stop_on_critical(false);
    nan_andon.check_loss(f32::NAN);
    println!("  After NaN: has_critical={}", nan_andon.has_critical());

    // Check for Inf detection
    let mut inf_andon = AndonSystem::new().with_stop_on_critical(false);
    inf_andon.check_loss(f32::INFINITY);
    println!("  After Inf: has_critical={}", inf_andon.has_critical());

    println!();
}

fn demo_reference_curve() {
    println!("--- Reference Curve Demo ---\n");

    // Create a reference curve from a "golden" training run (JSON array format)
    let reference =
        ReferenceCurve::from_json("[1.0, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.13, 0.12]")
            .expect("Failed to parse reference");

    // Current training progress (slightly worse)
    let current = vec![1.0, 0.65, 0.45, 0.35, 0.28];

    println!("  Reference: 10 values (baseline run)");
    println!("  Current: {} epochs", current.len());

    // Check deviation at each epoch
    println!("\n  Per-epoch deviation from reference:");
    for (epoch, &value) in current.iter().enumerate() {
        if let Some(dev) = reference.check_deviation(epoch, value) {
            println!("    Epoch {}: +{:.1}% (exceeds 10% tolerance)", epoch, dev * 100.0);
        } else {
            println!("    Epoch {epoch}: within tolerance");
        }
    }

    // Visual comparison using sparklines
    let comparison = reference.comparison_sparkline(&current, 20);
    println!("\n  Deviation sparkline:");
    println!("  {comparison} (neg=better, pos=worse)");
    println!();
}
