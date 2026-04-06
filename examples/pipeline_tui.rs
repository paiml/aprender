//! Realizar Pipeline TUI - Visual performance and correctness simulation
//!
//! Simulates the entire inference pipeline and identifies theoretical issues:
//! - Model creation and configuration
//! - Forward pass through transformer layers
//! - Attention computation
//! - KV cache operations
//! - Text generation with sampling
//! - Quantized operations (Q4_K, Q8_0)
//!
//! Run with: cargo run --release --example pipeline_tui

use std::time::Instant;

const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

fn color_for_time(time_ms: f64, good: f64, warn: f64) -> &'static str {
    if time_ms <= good {
        GREEN
    } else if time_ms <= warn {
        YELLOW
    } else {
        RED
    }
}

fn bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max) * width as f64).min(width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!(
        "{}{}{}",
        "\u{2588}".repeat(filled),
        DIM,
        "\u{2591}".repeat(empty)
    )
}

fn check_pass(passed: bool) -> &'static str {
    if passed {
        "\u{2713}"
    } else {
        "\u{2717}"
    }
}

fn main() {
    println!(
        "\n{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}"
    );
    println!("{BOLD}{CYAN}          REALIZAR PIPELINE TUI - INFERENCE SIMULATION          {RESET}");
    println!(
        "{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}\n"
    );

    run_pipeline_tests();

    println!(
        "\n{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}"
    );
    println!("{DIM}Legend: {GREEN}FAST{RESET} {YELLOW}OK{RESET} {RED}SLOW{RESET}");
    println!(
        "{BOLD}{CYAN}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}{RESET}\n"
    );
}

fn bench_model_configs() {
    use realizar::layers::{Model, ModelConfig};

    println!("{BOLD}{CYAN}\u{25b6} MODEL CONFIGURATIONS{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    let configs = [
        (
            "Tiny",
            ModelConfig {
                vocab_size: 1000,
                hidden_dim: 64,
                num_heads: 2,
                num_layers: 2,
                intermediate_dim: 256,
                eps: 1e-5,
            },
        ),
        (
            "Small",
            ModelConfig {
                vocab_size: 10000,
                hidden_dim: 256,
                num_heads: 4,
                num_layers: 4,
                intermediate_dim: 1024,
                eps: 1e-5,
            },
        ),
        (
            "Medium",
            ModelConfig {
                vocab_size: 32000,
                hidden_dim: 512,
                num_heads: 8,
                num_layers: 6,
                intermediate_dim: 2048,
                eps: 1e-5,
            },
        ),
    ];

    for (name, config) in &configs {
        let start = Instant::now();
        let model = Model::new(config.clone()).expect("model");
        let init_time = start.elapsed().as_secs_f64() * 1000.0;

        let params = model.num_parameters();
        let params_str = if params > 1_000_000 {
            format!("{:.1}M", params as f64 / 1_000_000.0)
        } else if params > 1_000 {
            format!("{:.1}K", params as f64 / 1_000.0)
        } else {
            format!("{params}")
        };

        let color = color_for_time(init_time, 10.0, 50.0);
        println!(
            "  {:8} model ({:>6} params)  {:>6.1}ms   {} {}{RESET}",
            name,
            params_str,
            init_time,
            bar(init_time, 50.0, 15),
            color
        );
    }
}

fn bench_forward_pass(model: &realizar::layers::Model, _config: &realizar::layers::ModelConfig) {
    println!("\n{BOLD}{CYAN}\u{25b6} FORWARD PASS{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    let seq_lengths = [1, 5, 10, 20, 50];

    for seq_len in seq_lengths {
        let tokens: Vec<usize> = (0..seq_len).map(|i| i % 1000).collect();

        // Warmup
        for _ in 0..3 {
            let _ = model.forward(&tokens);
        }

        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&tokens).expect("forward");
        }
        let time_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let expected = 1.0 + seq_len as f64 * 0.5;
        let color = color_for_time(time_ms, expected, expected * 2.0);
        println!(
            "  Forward (seq={:>2})         {:>6.1}ms   {} {}{RESET}",
            seq_len,
            time_ms,
            bar(time_ms, expected * 2.0, 20),
            color
        );
    }
}

fn bench_text_generation(model: &realizar::layers::Model, _config: &realizar::layers::ModelConfig) {
    use realizar::generate::{GenerationConfig, SamplingStrategy};

    println!("\n{BOLD}{CYAN}\u{25b6} TEXT GENERATION{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    let gen_configs = [
        ("Greedy", GenerationConfig::default()),
        (
            "Top-k (k=5)",
            GenerationConfig {
                max_tokens: 20,
                temperature: 1.0,
                strategy: SamplingStrategy::TopK { k: 5 },
                eos_token_id: None,
                seed: None,
            },
        ),
        (
            "Top-p (p=0.9)",
            GenerationConfig {
                max_tokens: 20,
                temperature: 1.0,
                strategy: SamplingStrategy::TopP { p: 0.9 },
                eos_token_id: None,
                seed: None,
            },
        ),
    ];

    let prompt_tokens: Vec<usize> = vec![1, 5, 10];

    for (name, gen_config) in &gen_configs {
        // Warmup
        let _ = model.generate(&prompt_tokens, gen_config);

        let start = Instant::now();
        let generated = model
            .generate(&prompt_tokens, gen_config)
            .expect("generate");
        let time_ms = start.elapsed().as_secs_f64() * 1000.0;

        let num_generated = generated.len() - prompt_tokens.len();
        let tokens_per_sec = num_generated as f64 / (time_ms / 1000.0);

        let color = color_for_time(time_ms, 50.0, 100.0);
        println!(
            "  {:15} {:>3} tokens  {:>6.1}ms   {:>5.0} tok/s {} {}{RESET}",
            name,
            num_generated,
            time_ms,
            tokens_per_sec,
            bar(tokens_per_sec, 500.0, 10),
            color
        );
    }
}

fn bench_matrix_ops() {
    use trueno::{Matrix, Vector};

    println!("\n{BOLD}{CYAN}\u{25b6} TRUENO MATRIX OPERATIONS{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    // Test vector operations
    let sizes = [1024, 4096, 16384, 65536];
    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.002).cos()).collect();
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        // Warmup
        for _ in 0..3 {
            let _ = va.dot(&vb);
        }

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = va.dot(&vb);
        }
        let time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;
        let gflops = (2.0 * size as f64) / (time_us / 1_000_000.0) / 1e9;

        let color = if gflops > 20.0 {
            GREEN
        } else if gflops > 10.0 {
            YELLOW
        } else {
            RED
        };
        println!(
            "  dot({:>6})          {:>6.1}\u{03bc}s   {:>5.1} GFLOPS {} {}{RESET}",
            size,
            time_us,
            gflops,
            bar(gflops, 50.0, 10),
            color
        );
    }

    // Test matrix multiply (vocab projection pattern)
    let matmul_sizes = [
        (1, 128, 1000, "1x128x1000 (vocab proj)"),
        (1, 256, 10000, "1x256x10000 (vocab proj)"),
        (1, 512, 32000, "1x512x32000 (vocab proj)"),
        (32, 128, 128, "32x128x128 (batch)"),
    ];

    for (m, k, n, desc) in matmul_sizes {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.0001).cos()).collect();
        let ma = Matrix::from_vec(m, k, a).expect("test");
        let mb = Matrix::from_vec(k, n, b).expect("test");

        // Warmup
        for _ in 0..3 {
            let _ = ma.matmul(&mb);
        }

        let iterations = if m * k * n > 1_000_000 { 5 } else { 20 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ma.matmul(&mb).expect("test");
        }
        let time_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        let ops = (2 * m * k * n) as f64;
        let gflops = ops / (time_ms / 1000.0) / 1e9;

        let expected = if n > 10000 { 10.0 } else { 2.0 };
        let color = color_for_time(time_ms, expected, expected * 3.0);
        println!(
            "  {:25} {:>6.1}ms {:>5.1} GFLOPS {} {}{RESET}",
            desc,
            time_ms,
            gflops,
            bar(gflops, 15.0, 10),
            color
        );
    }
}

fn bench_activations() {
    use trueno::Vector;

    println!("\n{BOLD}{CYAN}\u{25b6} ACTIVATION FUNCTIONS{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    let size = 65536;
    let data: Vec<f32> = (0..size).map(|i| ((i as f32) * 0.01).sin()).collect();
    let v = Vector::from_slice(&data);

    #[allow(clippy::type_complexity)]
    let activations: Vec<(&str, Box<dyn Fn(&Vector<f32>) -> Vector<f32>>)> = vec![
        ("relu", Box::new(|v: &Vector<f32>| v.relu().expect("test"))),
        (
            "sigmoid",
            Box::new(|v: &Vector<f32>| v.sigmoid().expect("test")),
        ),
        ("tanh", Box::new(|v: &Vector<f32>| v.tanh().expect("test"))),
        (
            "softmax",
            Box::new(|v: &Vector<f32>| v.softmax().expect("test")),
        ),
        ("gelu", Box::new(|v: &Vector<f32>| v.gelu().expect("test"))),
    ];

    for (name, func) in &activations {
        // Warmup
        for _ in 0..3 {
            let _ = func(&v);
        }

        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = func(&v);
        }
        let time_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64;
        let throughput = size as f64 / time_us; // M elem/s

        let color = if throughput > 100.0 {
            GREEN
        } else if throughput > 50.0 {
            YELLOW
        } else {
            RED
        };
        println!(
            "  {:>10} (n={size})    {:>6.1}\u{03bc}s   {:>5.1}M elem/s {} {}{RESET}",
            name,
            time_us,
            throughput,
            bar(throughput, 200.0, 10),
            color
        );
    }
}

fn bench_correctness(model: &realizar::layers::Model, config: &realizar::layers::ModelConfig) {
    use realizar::generate::GenerationConfig;
    use trueno::{Matrix, Vector};

    println!("\n{BOLD}{CYAN}\u{25b6} CORRECTNESS CHECKS{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    // Check forward output is finite
    let tokens: Vec<usize> = vec![1, 5, 10];
    let logits = model.forward(&tokens).expect("forward");
    let logits_data = logits.data();
    let logits_finite = logits_data.iter().all(|x| x.is_finite());
    println!(
        "  Forward output finite   {} {}{RESET}",
        check_pass(logits_finite),
        if logits_finite { GREEN } else { RED }
    );

    // Check softmax sums to 1
    let shape = logits.shape();
    let vocab_size = shape[shape.len() - 1];
    let last_logits_start = logits_data.len() - vocab_size;
    let last_logits = &logits_data[last_logits_start..];
    let softmax_v = Vector::from_slice(last_logits);
    let softmax_out = softmax_v.softmax().expect("test");
    let softmax_sum: f32 = softmax_out.as_slice().iter().sum();
    let softmax_ok = (softmax_sum - 1.0).abs() < 0.001;
    println!(
        "  Softmax sums to 1       {} (sum={:.6}) {}{RESET}",
        check_pass(softmax_ok),
        softmax_sum,
        if softmax_ok { GREEN } else { RED }
    );

    // Check no NaN in logits
    let no_nan = logits_data.iter().all(|x| !x.is_nan());
    println!(
        "  No NaN in logits        {} {}{RESET}",
        check_pass(no_nan),
        if no_nan { GREEN } else { RED }
    );

    // Check generation produces valid tokens
    let gen_config = GenerationConfig::default();
    let generated = model.generate(&tokens, &gen_config).expect("generate");
    let tokens_valid = generated.iter().all(|&t| t < config.vocab_size);
    println!(
        "  Generated tokens valid  {} (len={}) {}{RESET}",
        check_pass(tokens_valid),
        generated.len(),
        if tokens_valid { GREEN } else { RED }
    );

    // Check dot product is commutative
    let a: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32 * 0.2).cos()).collect();
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let dot_ab = va.dot(&vb).expect("test");
    let dot_ba = vb.dot(&va).expect("test");
    let dot_commutative = (dot_ab - dot_ba).abs() < 1e-5;
    println!(
        "  Dot commutative         {} {}{RESET}",
        check_pass(dot_commutative),
        if dot_commutative { GREEN } else { RED }
    );

    // Check matmul dimensions
    let m1 = Matrix::from_vec(3, 4, vec![1.0; 12]).expect("test");
    let m2 = Matrix::from_vec(4, 5, vec![1.0; 20]).expect("test");
    let m3 = m1.matmul(&m2).expect("test");
    let dims_ok = m3.rows() == 3 && m3.cols() == 5;
    println!(
        "  Matmul dimensions       {} (3x4 @ 4x5 = {}x{}) {}{RESET}",
        check_pass(dims_ok),
        m3.rows(),
        m3.cols(),
        if dims_ok { GREEN } else { RED }
    );
}

fn bench_throughput_summary(
    model: &realizar::layers::Model,
    _config: &realizar::layers::ModelConfig,
) {
    use realizar::generate::GenerationConfig;

    println!("\n{BOLD}{CYAN}\u{25b6} THROUGHPUT SUMMARY{RESET}");
    println!(
        "{DIM}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}{RESET}"
    );

    let tokens: Vec<usize> = vec![1, 5, 10];

    // Measure sustained generation throughput
    let gen_config = GenerationConfig {
        max_tokens: 50,
        ..GenerationConfig::default()
    };

    let start = Instant::now();
    let generated = model.generate(&tokens, &gen_config).expect("generate");
    let total_time = start.elapsed().as_secs_f64() * 1000.0;
    let num_tokens = generated.len() - tokens.len();
    let tokens_per_sec = num_tokens as f64 / (total_time / 1000.0);

    let color = if tokens_per_sec > 200.0 {
        GREEN
    } else if tokens_per_sec > 100.0 {
        YELLOW
    } else {
        RED
    };
    println!(
        "  Generate {num_tokens} tokens       {:>6.1}ms   {:>5.0} tok/s {} {}{RESET}",
        total_time,
        tokens_per_sec,
        bar(tokens_per_sec, 500.0, 15),
        color
    );

    // Report model size and theoretical peak
    let params = model.num_parameters();
    let params_mb = (params * 4) as f64 / 1e6; // 4 bytes per f32
    println!(
        "  Model size              {:>6.1}MB   ({} params)",
        params_mb, params
    );
}

fn run_pipeline_tests() {
    use realizar::layers::{Model, ModelConfig};

    bench_model_configs();

    // Use small model for remaining tests
    let config = ModelConfig {
        vocab_size: 1000,
        hidden_dim: 128,
        num_heads: 4,
        num_layers: 4,
        intermediate_dim: 512,
        eps: 1e-5,
    };
    let model = Model::new(config.clone()).expect("model");

    bench_forward_pass(&model, &config);
    bench_text_generation(&model, &config);
    bench_matrix_ops();
    bench_activations();
    bench_correctness(&model, &config);
    bench_throughput_summary(&model, &config);
}
