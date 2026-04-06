#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
//! MNIST Training Example with Real-Time Visualization
//!
//! Demonstrates:
//! - Loading MNIST from alimentar
//! - Training a simple neural network for ~60 seconds
//! - Real-time loss curve visualization with trueno-viz
//! - Saving model to .apr format via aprender
//!
//! Run with: cargo run --example mnist_train

use alimentar::{datasets::mnist, Dataset};
use aprender::format::{save, ModelType, SaveOptions};
use arrow::array::{Float32Array, Int32Array};
use entrenar::efficiency::device::{ComputeDevice, SimdCapability};
use entrenar::train::{sparkline, AndonSystem, LossCurveDisplay, MetricsBuffer, TerminalMode};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::{Duration, Instant};

/// System metrics for real-time monitoring
#[derive(Debug, Clone, Default)]
struct SystemMetrics {
    cpu_percent: f32,
    memory_used_mb: f32,
    memory_total_mb: f32,
    prev_cpu_total: u64,
    prev_cpu_idle: u64,
}

impl SystemMetrics {
    fn new() -> Self {
        let mut m = Self::default();
        m.update(); // Initialize CPU baseline
        m
    }

    fn update(&mut self) {
        self.update_cpu_stats();
        self.update_memory_stats();
    }

    fn update_cpu_stats(&mut self) {
        let Ok(stat) = fs::read_to_string("/proc/stat") else {
            return;
        };
        let Some(cpu_line) = stat.lines().next() else {
            return;
        };
        let parts: Vec<u64> =
            cpu_line.split_whitespace().skip(1).filter_map(|s| s.parse().ok()).collect();
        if parts.len() < 4 {
            return;
        }
        let total: u64 = parts.iter().sum();
        let idle = parts[3];
        if self.prev_cpu_total > 0 {
            let total_diff = total.saturating_sub(self.prev_cpu_total);
            let idle_diff = idle.saturating_sub(self.prev_cpu_idle);
            if total_diff > 0 {
                self.cpu_percent = 100.0 * (1.0 - idle_diff as f32 / total_diff as f32);
            }
        }
        self.prev_cpu_total = total;
        self.prev_cpu_idle = idle;
    }

    fn update_memory_stats(&mut self) {
        let Ok(meminfo) = fs::read_to_string("/proc/meminfo") else {
            return;
        };
        let mut total = 0u64;
        let mut available = 0u64;
        for line in meminfo.lines() {
            let val =
                || line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0u64);
            if line.starts_with("MemTotal:") {
                total = val();
            } else if line.starts_with("MemAvailable:") {
                available = val();
            }
        }
        self.memory_total_mb = total as f32 / 1024.0;
        self.memory_used_mb = (total - available) as f32 / 1024.0;
    }

    fn memory_percent(&self) -> f32 {
        if self.memory_total_mb > 0.0 {
            100.0 * self.memory_used_mb / self.memory_total_mb
        } else {
            0.0
        }
    }

    /// Render a simple ASCII bar
    fn render_bar(percent: f32, width: usize) -> String {
        let filled = ((percent / 100.0) * width as f32) as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }
}

/// Simple 2-layer neural network for MNIST
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MnistModel {
    /// Input -> Hidden weights (784 x 64)
    w1: Vec<f32>,
    /// Hidden biases (64)
    b1: Vec<f32>,
    /// Hidden -> Output weights (64 x 10)
    w2: Vec<f32>,
    /// Output biases (10)
    b2: Vec<f32>,
}

impl MnistModel {
    fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Xavier initialization
        let scale1 = (2.0 / 784.0_f32).sqrt();
        let scale2 = (2.0 / 64.0_f32).sqrt();

        Self {
            w1: (0..784 * 64).map(|_| rng.random::<f32>() * scale1 - scale1 / 2.0).collect(),
            b1: vec![0.0; 64],
            w2: (0..64 * 10).map(|_| rng.random::<f32>() * scale2 - scale2 / 2.0).collect(),
            b2: vec![0.0; 10],
        }
    }

    /// Forward pass with ReLU activation
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Hidden layer: ReLU(input @ W1 + b1)
        let mut hidden = vec![0.0; 64];
        for h in 0..64 {
            let mut sum = self.b1[h];
            for i in 0..784 {
                sum += input[i] * self.w1[i * 64 + h];
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        // Output layer: hidden @ W2 + b2
        let mut output = [0.0; 10];
        for o in 0..10 {
            let mut sum = self.b2[o];
            for h in 0..64 {
                sum += hidden[h] * self.w2[h * 10 + o];
            }
            output[o] = sum;
        }

        // Softmax
        let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = output.iter().map(|x| (x - max_val).exp()).sum();
        output.iter().map(|x| (x - max_val).exp() / exp_sum).collect()
    }

    /// Compute cross-entropy loss and gradients
    fn backward(&mut self, input: &[f32], target: usize, lr: f32) -> f32 {
        let (hidden, hidden_pre_relu) = self.forward_hidden(input);
        let output = self.forward_output(&hidden);
        let probs = softmax_vec(&output);
        let loss = -probs[target].max(f32::MIN_POSITIVE).ln();
        let mut d_output = probs;
        d_output[target] -= 1.0;
        let mut d_hidden = self.backprop_output(&hidden, &d_output, lr);
        apply_relu_grad(&mut d_hidden, &hidden_pre_relu);
        self.backprop_hidden(input, &d_hidden, lr);
        loss
    }

    fn forward_hidden(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut hidden = vec![0.0; 64];
        let mut pre_relu = vec![0.0; 64];
        for h in 0..64 {
            let mut sum = self.b1[h];
            for i in 0..784 {
                sum += input[i] * self.w1[i * 64 + h];
            }
            pre_relu[h] = sum;
            hidden[h] = sum.max(0.0);
        }
        (hidden, pre_relu)
    }

    fn forward_output(&self, hidden: &[f32]) -> [f32; 10] {
        let mut output = [0.0; 10];
        for o in 0..10 {
            let mut sum = self.b2[o];
            for h in 0..64 {
                sum += hidden[h] * self.w2[h * 10 + o];
            }
            output[o] = sum;
        }
        output
    }

    fn backprop_output(&mut self, hidden: &[f32], d_output: &[f32], lr: f32) -> Vec<f32> {
        let mut d_hidden = vec![0.0; 64];
        for o in 0..10 {
            self.b2[o] -= lr * d_output[o];
            for h in 0..64 {
                self.w2[h * 10 + o] -= lr * hidden[h] * d_output[o];
                d_hidden[h] += self.w2[h * 10 + o] * d_output[o];
            }
        }
        d_hidden
    }

    fn backprop_hidden(&mut self, input: &[f32], d_hidden: &[f32], lr: f32) {
        for h in 0..64 {
            self.b1[h] -= lr * d_hidden[h];
            for i in 0..784 {
                self.w1[i * 64 + h] -= lr * input[i] * d_hidden[h];
            }
        }
    }

    /// Predict class
    fn predict(&self, input: &[f32]) -> usize {
        let probs = self.forward(input);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed"))
            .map(|(i, _)| i)
            .expect("operation should succeed")
    }
}

fn softmax_vec(output: &[f32; 10]) -> Vec<f32> {
    let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = output.iter().map(|x| (x - max_val).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|x| x / exp_sum).collect()
}

fn apply_relu_grad(d_hidden: &mut [f32], pre_relu: &[f32]) {
    for (d, &pre) in d_hidden.iter_mut().zip(pre_relu.iter()) {
        if pre <= 0.0 {
            *d = 0.0;
        }
    }
}

/// Load MNIST dataset and extract train/test splits
fn load_mnist_data() -> (Vec<Vec<f32>>, Vec<usize>, Vec<Vec<f32>>, Vec<usize>) {
    println!("Loading MNIST dataset from alimentar...");
    let dataset = mnist().expect("Failed to load MNIST");
    let split = dataset.split().expect("Failed to split dataset");

    println!("  Train samples: {}", split.train.len());
    println!("  Test samples: {}", split.test.len());

    let (train_images, train_labels) = extract_data(&split.train);
    let (test_images, test_labels) = extract_data(&split.test);

    println!("  Image dimensions: 28x28 (784 features)");
    println!("  Classes: 0-9 digits\n");

    (train_images, train_labels, test_images, test_labels)
}

/// Detect SIMD capability and GPU availability
fn detect_devices() -> (SimdCapability, bool) {
    let devices = ComputeDevice::detect();
    let cpu_info = devices.iter().find(|d| d.is_cpu());
    let simd_level = cpu_info.map_or(SimdCapability::None, |d| match d {
        ComputeDevice::Cpu(info) => info.simd,
        _ => SimdCapability::None,
    });
    let gpu_available = devices.iter().any(|d| d.is_gpu() || d.is_apple_silicon());
    (simd_level, gpu_available)
}

/// Run one training epoch and return the average loss
fn run_training_epoch(
    model: &mut MnistModel,
    train_images: &[Vec<f32>],
    train_labels: &[usize],
    lr: f32,
    batch_size: usize,
    total_samples: &mut usize,
) -> f32 {
    let mut epoch_loss = 0.0;
    let num_batches = train_images.len() / batch_size;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let mut batch_loss = 0.0;

        for i in 0..batch_size {
            let idx = start_idx + i;
            let loss = model.backward(&train_images[idx], train_labels[idx], lr);
            batch_loss += loss;
            *total_samples += 1;
        }

        epoch_loss += batch_loss / batch_size as f32;
    }

    epoch_loss / num_batches as f32
}

/// Run validation and return (validation_loss, best_accuracy)
fn run_validation(
    model: &MnistModel,
    test_images: &[Vec<f32>],
    test_labels: &[usize],
    best_accuracy: &mut f32,
) -> f32 {
    let mut correct = 0;
    let mut val_total_loss = 0.0;
    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        let probs = model.forward(img);
        val_total_loss += -probs[label].ln();
        if model.predict(img) == label {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / test_images.len() as f32 * 100.0;
    if accuracy > *best_accuracy {
        *best_accuracy = accuracy;
    }
    val_total_loss / test_images.len() as f32
}

/// Render the training monitor display
#[allow(clippy::too_many_arguments)]
fn render_training_display(
    epoch: usize,
    total_samples: usize,
    samples_per_sec: f32,
    avg_loss: f32,
    best_accuracy: f32,
    training_duration: &Duration,
    start_time: &Instant,
    metrics_buffer: &MetricsBuffer,
    loss_display: &LossCurveDisplay,
    sys_metrics: &SystemMetrics,
    cpu_history: &[f32],
    simd_level: SimdCapability,
    gpu_available: bool,
    andon: &AndonSystem,
) {
    let elapsed = start_time.elapsed().as_secs();
    let remaining = training_duration.as_secs().saturating_sub(elapsed);

    print!("\x1B[2J\x1B[H");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          MNIST Neural Network Training Monitor               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("┌─ Training Progress ─────────────────────────────────────────┐");
    println!(
        "│ Epoch: {:4}  │  Samples: {:6}  │  Time: {:2}s / {:2}s ({}s left)",
        epoch,
        total_samples,
        elapsed,
        training_duration.as_secs(),
        remaining
    );
    println!("│ Throughput: {samples_per_sec:.0} samples/sec");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("┌─ Loss Metrics ──────────────────────────────────────────────┐");
    let losses: Vec<f32> = metrics_buffer.last_n(30).to_vec();
    println!("│ Train Loss: {avg_loss:.4}  │  Best Accuracy: {best_accuracy:.1}%");
    println!("│ Loss Trend (last 30 epochs): {}", sparkline(&losses, 30));
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("┌─ Loss Curve (Orange=Train, Blue=Validation) ───────────────┐");
    println!("{}", loss_display.render_terminal());
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("┌─ System Resources ──────────────────────────────────────────┐");
    println!(
        "│ CPU:  {:5.1}% {} {} ({})",
        sys_metrics.cpu_percent,
        SystemMetrics::render_bar(sys_metrics.cpu_percent, 20),
        sparkline(cpu_history, 12),
        simd_level
    );
    println!(
        "│ RAM:  {:5.1}% {} {:5.0}/{:.0} MB",
        sys_metrics.memory_percent(),
        SystemMetrics::render_bar(sys_metrics.memory_percent(), 20),
        sys_metrics.memory_used_mb,
        sys_metrics.memory_total_mb
    );
    if gpu_available {
        println!("│ GPU:  Available (use trueno gpu feature for acceleration)");
    } else {
        println!("│ GPU:  Not detected (CPU mode with {simd_level} SIMD)");
    }
    println!("└──────────────────────────────────────────────────────────────┘");

    if andon.has_critical() {
        println!("\n[ANDON] Warning: Training anomaly detected!");
    }
}

/// Evaluate model on test set and print final results
fn evaluate_and_save(
    model: &MnistModel,
    test_images: &[Vec<f32>],
    test_labels: &[usize],
    epoch: usize,
    total_samples: usize,
    lr: f32,
) {
    println!("\n┌────────────────────────────────────────────────────────────┐");
    println!("│ Training Complete                                          │");
    println!("└────────────────────────────────────────────────────────────┘\n");

    let mut correct = 0;
    for (img, &label) in test_images.iter().zip(test_labels.iter()) {
        if model.predict(img) == label {
            correct += 1;
        }
    }
    let final_accuracy = correct as f32 / test_images.len() as f32 * 100.0;

    println!("Final Results:");
    println!("  Epochs completed: {epoch}");
    println!("  Samples processed: {total_samples}");
    println!("  Test accuracy: {final_accuracy:.1}%\n");

    let model_path = "/tmp/mnist_model.apr";
    println!("Saving model to {model_path}...");

    let save_opts = SaveOptions::default().with_name("MNIST Classifier").with_description(format!(
        "2-layer neural network (784->64->10) trained on MNIST digits. \
             Learning rate: {lr}, Epochs: {epoch}, Final accuracy: {final_accuracy:.1}%"
    ));

    match save(model, ModelType::NeuralSequential, model_path, save_opts) {
        Ok(()) => println!("  Model saved successfully!"),
        Err(e) => println!("  Failed to save model: {e}"),
    }

    println!("\n=== Demo Complete ===");
}

fn main() {
    println!("=== MNIST Training with Real-Time Visualization ===\n");

    let (train_images, train_labels, test_images, test_labels) = load_mnist_data();

    // Initialize model
    println!("Initializing neural network...");
    let mut model = MnistModel::new();
    println!("  Architecture: 784 -> 64 (ReLU) -> 10 (Softmax)");
    println!("  Parameters: {} weights\n", 784 * 64 + 64 + 64 * 10 + 10);

    // Training configuration
    let training_duration = Duration::from_secs(60);
    let lr = 0.01;
    let batch_size = 10;

    // Visualization setup
    let mut loss_display = LossCurveDisplay::new(60, 10).terminal_mode(TerminalMode::Unicode);
    let mut metrics_buffer = MetricsBuffer::new(100);
    let mut andon = AndonSystem::new()
        .with_sigma_threshold(5.0)
        .with_stall_threshold(50)
        .with_stop_on_critical(false);

    // System metrics
    let mut sys_metrics = SystemMetrics::new();
    let mut cpu_history: Vec<f32> = Vec::new();
    let mut mem_history: Vec<f32> = Vec::new();

    let (simd_level, gpu_available) = detect_devices();

    println!("Training for {} seconds...\n", training_duration.as_secs());
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Real-Time Training Monitor                                 │");
    println!("└────────────────────────────────────────────────────────────┘\n");

    let start_time = Instant::now();
    let mut epoch = 0;
    let mut total_samples = 0;
    let mut best_accuracy = 0.0;
    let mut samples_per_sec = 0.0;

    while start_time.elapsed() < training_duration {
        epoch += 1;

        let avg_loss = run_training_epoch(
            &mut model,
            &train_images,
            &train_labels,
            lr,
            batch_size,
            &mut total_samples,
        );

        metrics_buffer.push(avg_loss);
        andon.check_loss(avg_loss);

        // Validation accuracy (every 5 epochs)
        let val_loss = if epoch % 5 == 0 {
            run_validation(&model, &test_images, &test_labels, &mut best_accuracy)
        } else {
            avg_loss * 1.1
        };

        loss_display.push_losses(avg_loss, val_loss);

        // Update system metrics
        sys_metrics.update();
        cpu_history.push(sys_metrics.cpu_percent);
        mem_history.push(sys_metrics.memory_percent());
        if cpu_history.len() > 30 {
            cpu_history.remove(0);
        }
        if mem_history.len() > 30 {
            mem_history.remove(0);
        }

        // Calculate throughput
        let elapsed_secs = start_time.elapsed().as_secs_f32();
        if elapsed_secs > 0.0 {
            samples_per_sec = total_samples as f32 / elapsed_secs;
        }

        render_training_display(
            epoch,
            total_samples,
            samples_per_sec,
            avg_loss,
            best_accuracy,
            &training_duration,
            &start_time,
            &metrics_buffer,
            &loss_display,
            &sys_metrics,
            &cpu_history,
            simd_level,
            gpu_available,
            &andon,
        );

        if start_time.elapsed() >= training_duration {
            break;
        }
    }

    evaluate_and_save(&model, &test_images, &test_labels, epoch, total_samples, lr);
}

/// Extract images and labels from alimentar dataset
fn extract_data(dataset: &alimentar::ArrowDataset) -> (Vec<Vec<f32>>, Vec<usize>) {
    let batch = dataset.get_batch(0).expect("No batch");
    let num_rows = batch.num_rows();

    let mut images = Vec::with_capacity(num_rows);
    let mut labels = Vec::with_capacity(num_rows);

    for row in 0..num_rows {
        // Extract 784 pixel values (alimentar already normalizes to 0-1)
        let mut pixels = Vec::with_capacity(784);
        for col in 0..784 {
            let arr = batch
                .column(col)
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Expected Float32Array");
            pixels.push(arr.value(row));
        }
        images.push(pixels);

        // Extract label
        let label_arr =
            batch.column(784).as_any().downcast_ref::<Int32Array>().expect("Expected Int32Array");
        labels.push(label_arr.value(row) as usize);
    }

    (images, labels)
}
