#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! ML Training Metrics Example
//!
//! Demonstrates real-time training loss/accuracy visualization.
//! Similar to `TensorBoard` but in the terminal.
//!
//! Run with: cargo run -p presentar-terminal --example `training_metrics`

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{BrailleGraph, ColorMode, GraphMode};

fn main() {
    println!("=== ML Training Metrics Dashboard ===\n");

    // Simulate training history
    let epochs = 100;
    let (train_loss, val_loss) = simulate_training_loss(epochs);
    let (train_acc, val_acc) = simulate_accuracy(epochs);
    let learning_rates = simulate_lr_schedule(epochs);

    // Create buffer
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);

    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Background
        canvas.fill_rect(
            Rect::new(0.0, 0.0, 80.0, 24.0),
            Color::new(0.05, 0.05, 0.1, 1.0),
        );

        // Title
        let title_style = TextStyle {
            color: Color::new(1.0, 0.8, 0.3, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "ML Training Dashboard - Epoch 100/100",
            Point::new(2.0, 1.0),
            &title_style,
        );

        // Loss curves (left side)
        draw_loss_graph(
            &mut canvas,
            &train_loss,
            &val_loss,
            Rect::new(2.0, 3.0, 36.0, 7.0),
        );

        // Accuracy curves (right side)
        draw_accuracy_graph(
            &mut canvas,
            &train_acc,
            &val_acc,
            Rect::new(42.0, 3.0, 36.0, 7.0),
        );

        // Learning rate schedule
        draw_lr_graph(
            &mut canvas,
            &learning_rates,
            Rect::new(2.0, 11.0, 36.0, 5.0),
        );

        // Training statistics
        draw_training_stats(
            &mut canvas,
            &train_loss,
            &val_loss,
            &train_acc,
            &val_acc,
            42.0,
            11.0,
        );

        // Hyperparameters
        draw_hyperparams(&mut canvas, 2.0, 17.0);

        // Current epoch metrics
        draw_current_metrics(
            &mut canvas,
            &train_loss,
            &val_loss,
            &train_acc,
            &val_acc,
            42.0,
            17.0,
        );
    }

    // Render
    let mut output = Vec::with_capacity(8192);
    let cells_written = renderer.flush(&mut buffer, &mut output).unwrap();

    println!("Buffer: {}x{}", buffer.width(), buffer.height());
    println!("Cells written: {cells_written}");
    println!("Output bytes: {}\n", output.len());

    println!("Rendered output:");
    println!("{}", "─".repeat(82));
    std::io::Write::write_all(&mut std::io::stdout(), &output).unwrap();
    println!();
    println!("{}", "─".repeat(82));
}

fn draw_loss_graph(
    canvas: &mut DirectTerminalCanvas<'_>,
    train: &[f64],
    val: &[f64],
    bounds: Rect,
) {
    let label_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    canvas.draw_text("Loss Curves", Point::new(bounds.x, bounds.y), &label_style);

    // Legend
    let train_style = TextStyle {
        color: Color::new(0.3, 0.7, 1.0, 1.0),
        ..Default::default()
    };
    let val_style = TextStyle {
        color: Color::new(1.0, 0.5, 0.3, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "━Train",
        Point::new(bounds.x + 13.0, bounds.y),
        &train_style,
    );
    canvas.draw_text("━Val", Point::new(bounds.x + 22.0, bounds.y), &val_style);

    // Find max for scale
    let max_loss = train
        .iter()
        .chain(val.iter())
        .fold(0.0_f64, |a, &b| a.max(b));

    // Draw training loss graph
    let mut graph = BrailleGraph::new(train.to_vec())
        .with_color(Color::new(0.3, 0.7, 1.0, 1.0))
        .with_range(0.0, max_loss * 1.1)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_accuracy_graph(
    canvas: &mut DirectTerminalCanvas<'_>,
    train: &[f64],
    _val: &[f64],
    bounds: Rect,
) {
    let label_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Accuracy Curves",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    // Legend
    let train_style = TextStyle {
        color: Color::new(0.3, 1.0, 0.5, 1.0),
        ..Default::default()
    };
    let val_style = TextStyle {
        color: Color::new(0.9, 0.3, 0.9, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "━Train",
        Point::new(bounds.x + 17.0, bounds.y),
        &train_style,
    );
    canvas.draw_text("━Val", Point::new(bounds.x + 26.0, bounds.y), &val_style);

    // Draw accuracy graph
    let mut graph = BrailleGraph::new(train.to_vec())
        .with_color(Color::new(0.3, 1.0, 0.5, 1.0))
        .with_range(0.0, 100.0)
        .with_mode(GraphMode::Braille);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_lr_graph(canvas: &mut DirectTerminalCanvas<'_>, lr: &[f64], bounds: Rect) {
    let label_style = TextStyle {
        color: Color::new(0.7, 0.7, 0.7, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "Learning Rate Schedule",
        Point::new(bounds.x, bounds.y),
        &label_style,
    );

    let current_lr = lr.last().copied().unwrap_or(0.0);
    let lr_style = TextStyle {
        color: Color::new(0.9, 0.7, 0.3, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        &format!("lr={current_lr:.2e}"),
        Point::new(bounds.x + 24.0, bounds.y),
        &lr_style,
    );

    let max_lr = lr.iter().fold(0.0_f64, |a, &b| a.max(b));
    let mut graph = BrailleGraph::new(lr.to_vec())
        .with_color(Color::new(0.9, 0.7, 0.3, 1.0))
        .with_range(0.0, max_lr * 1.1)
        .with_mode(GraphMode::Block);

    graph.layout(Rect::new(
        bounds.x,
        bounds.y + 1.0,
        bounds.width,
        bounds.height - 1.0,
    ));
    graph.paint(canvas);
}

fn draw_training_stats(
    canvas: &mut DirectTerminalCanvas<'_>,
    train_loss: &[f64],
    val_loss: &[f64],
    train_acc: &[f64],
    val_acc: &[f64],
    x: f32,
    y: f32,
) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color: Color::new(0.9, 0.9, 0.9, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Training Statistics:", Point::new(x, y), &label_style);

    // Best metrics
    let best_train_loss = train_loss.iter().fold(f64::MAX, |a, &b| a.min(b));
    let best_val_loss = val_loss.iter().fold(f64::MAX, |a, &b| a.min(b));
    let best_train_acc = train_acc.iter().fold(0.0_f64, |a, &b| a.max(b));
    let best_val_acc = val_acc.iter().fold(0.0_f64, |a, &b| a.max(b));

    canvas.draw_text(
        &format!("Best Train Loss: {best_train_loss:.4}"),
        Point::new(x, y + 1.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Best Val Loss:   {best_val_loss:.4}"),
        Point::new(x, y + 2.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Best Train Acc:  {best_train_acc:.2}%"),
        Point::new(x, y + 3.0),
        &value_style,
    );
    canvas.draw_text(
        &format!("Best Val Acc:    {best_val_acc:.2}%"),
        Point::new(x, y + 4.0),
        &value_style,
    );
}

fn draw_hyperparams(canvas: &mut DirectTerminalCanvas<'_>, x: f32, y: f32) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };
    let value_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Hyperparameters:", Point::new(x, y), &label_style);
    canvas.draw_text(
        "Model: ResNet-50  Optimizer: AdamW",
        Point::new(x, y + 1.0),
        &value_style,
    );
    canvas.draw_text(
        "Batch: 32  LR: 1e-4  WD: 0.01",
        Point::new(x, y + 2.0),
        &value_style,
    );
    canvas.draw_text(
        "Scheduler: CosineAnnealing  T_max: 100",
        Point::new(x, y + 3.0),
        &value_style,
    );
}

fn draw_current_metrics(
    canvas: &mut DirectTerminalCanvas<'_>,
    train_loss: &[f64],
    val_loss: &[f64],
    train_acc: &[f64],
    val_acc: &[f64],
    x: f32,
    y: f32,
) {
    let label_style = TextStyle {
        color: Color::new(0.6, 0.6, 0.6, 1.0),
        ..Default::default()
    };

    canvas.draw_text("Current Epoch Metrics:", Point::new(x, y), &label_style);

    let curr_train_loss = train_loss.last().copied().unwrap_or(0.0);
    let curr_val_loss = val_loss.last().copied().unwrap_or(0.0);
    let curr_train_acc = train_acc.last().copied().unwrap_or(0.0);
    let curr_val_acc = val_acc.last().copied().unwrap_or(0.0);

    let good_style = TextStyle {
        color: Color::new(0.3, 1.0, 0.5, 1.0),
        ..Default::default()
    };

    canvas.draw_text(
        &format!("Loss:     {curr_train_loss:.4} / {curr_val_loss:.4}"),
        Point::new(x, y + 1.0),
        &good_style,
    );
    canvas.draw_text(
        &format!("Accuracy: {curr_train_acc:.2}% / {curr_val_acc:.2}%"),
        Point::new(x, y + 2.0),
        &good_style,
    );

    // ETA
    let eta_style = TextStyle {
        color: Color::new(0.8, 0.8, 0.8, 1.0),
        ..Default::default()
    };
    canvas.draw_text(
        "ETA: 00:00:00 (Complete)",
        Point::new(x, y + 3.0),
        &eta_style,
    );
}

fn simulate_training_loss(epochs: usize) -> (Vec<f64>, Vec<f64>) {
    let train: Vec<f64> = (0..epochs)
        .map(|e| {
            let t = e as f64 / epochs as f64;
            let base = 2.5f64.mul_add((-3.0 * t).exp(), 0.05);
            let noise = ((e * 7919) % 100) as f64 / 500.0;
            (base + noise).max(0.05)
        })
        .collect();

    let val: Vec<f64> = (0..epochs)
        .map(|e| {
            let t = e as f64 / epochs as f64;
            let base = 2.5f64.mul_add((-2.8 * t).exp(), 0.08);
            let overfit = if t > 0.7 { (t - 0.7) * 0.3 } else { 0.0 };
            let noise = ((e * 6971) % 100) as f64 / 400.0;
            (base + noise + overfit).max(0.08)
        })
        .collect();

    (train, val)
}

fn simulate_accuracy(epochs: usize) -> (Vec<f64>, Vec<f64>) {
    let train: Vec<f64> = (0..epochs)
        .map(|e| {
            let t = e as f64 / epochs as f64;
            let base = 48.0f64.mul_add(1.0 - (-4.0 * t).exp(), 50.0);
            let noise = ((e * 7919) % 50) as f64 / 25.0;
            (base + noise).clamp(50.0, 99.5)
        })
        .collect();

    let val: Vec<f64> = (0..epochs)
        .map(|e| {
            let t = e as f64 / epochs as f64;
            let base = 45.0f64.mul_add(1.0 - (-3.5 * t).exp(), 50.0);
            let noise = ((e * 6971) % 60) as f64 / 30.0;
            (base + noise).clamp(50.0, 97.0)
        })
        .collect();

    (train, val)
}

fn simulate_lr_schedule(epochs: usize) -> Vec<f64> {
    let initial_lr = 1e-3;
    (0..epochs)
        .map(|e| {
            let t = e as f64 / epochs as f64;
            // Cosine annealing
            initial_lr * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
        })
        .collect()
}
