#![allow(clippy::disallowed_methods)]
//! TUI Performance Test - Visual performance regression detection
//!
//! Run with: cargo run --release --example perf_tui
//!
//! Shows performance across multiple operation types and sizes to identify bottlenecks.

use std::time::Instant;
use trueno::{Matrix, Vector};

type ActivationFn = Box<dyn Fn(&Vector<f32>) -> Vector<f32>>;

const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";

fn color_for_gflops(gflops: f64, expected_min: f64) -> &'static str {
    if gflops >= expected_min * 1.5 {
        GREEN
    } else if gflops >= expected_min {
        YELLOW
    } else {
        RED
    }
}

fn bar(value: f64, max: f64, width: usize) -> String {
    let filled = ((value / max) * width as f64).min(width as f64) as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}{}", "█".repeat(filled), DIM, "░".repeat(empty))
}

fn benchmark<F>(_name: &str, ops: f64, iterations: usize, mut f: F) -> (f64, f64)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let gflops = (2.0 * ops) / (time_ms / 1000.0) / 1e9;
    (time_ms, gflops)
}

fn print_section_header(title: &str) {
    println!("\n{}{}▶ {}{}", BOLD, CYAN, title, RESET);
    println!("{}─────────────────────────────────────────────────────────────────{}", DIM, RESET);
}

fn bench_vector_ops() {
    print_section_header("VECTOR OPERATIONS");

    let sizes = [1024, 4096, 16384, 65536, 262144];
    let expected_gflops = 15.0;

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.002).collect();
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let (time, gflops) = benchmark("dot", size as f64, 100, || {
            let _ = va.dot(&vb);
        });

        let color = color_for_gflops(gflops, expected_gflops);
        println!(
            "  dot({:>6}) {:>7.2}ms {:>6.1} GFLOPS {} {}{}",
            size,
            time,
            gflops,
            bar(gflops, 30.0, 20),
            color,
            RESET
        );
    }
}

fn bench_vecmat_multiply() {
    print_section_header("VECTOR-MATRIX MULTIPLY (ML inference pattern)");

    let patterns = [
        (1, 384, 51865, "Whisper vocab projection"),
        (1, 768, 50257, "GPT-2 vocab projection"),
        (1, 512, 32000, "LLaMA vocab projection"),
        (1, 384, 1500, "Whisper cross-attention"),
        (1, 384, 384, "Small square"),
    ];

    for (m, k, n, desc) in patterns {
        let ops = (m * k * n) as f64;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0001).collect();
        let ma = Matrix::from_vec(m, k, a).unwrap();
        let mb = Matrix::from_vec(k, n, b).unwrap();

        let (time, gflops) = benchmark(desc, ops, 10, || {
            let _ = ma.matmul(&mb);
        });

        let expected = if n > 10000 { 8.0 } else { 5.0 };
        let color = color_for_gflops(gflops, expected);
        println!(
            "  {}x{}x{} {:>20} {:>6.1}ms {:>5.1} GFLOPS {} {}{}",
            m,
            k,
            n,
            desc,
            time,
            gflops,
            bar(gflops, 15.0, 15),
            color,
            RESET
        );
    }
}

fn bench_general_matmul() {
    print_section_header("GENERAL MATRIX MULTIPLY");

    let sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)];

    for (m, k, n) in sizes {
        let ops = (m * k * n) as f64;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.001).collect();
        let ma = Matrix::from_vec(m, k, a).unwrap();
        let mb = Matrix::from_vec(k, n, b).unwrap();

        let iters = if m >= 512 { 3 } else { 10 };
        let (time, gflops) = benchmark("matmul", ops, iters, || {
            let _ = ma.matmul(&mb);
        });

        let expected = 2.0;
        let color = color_for_gflops(gflops, expected);
        println!(
            "  {}x{}x{} {:>8.1}ms {:>5.1} GFLOPS {} {}{}",
            m,
            k,
            n,
            time,
            gflops,
            bar(gflops, 10.0, 20),
            color,
            RESET
        );
    }
}

fn bench_transpose() {
    print_section_header("TRANSPOSE OPERATIONS");

    let sizes = [(384, 51865), (768, 50257), (1024, 1024), (2048, 2048)];

    for (rows, cols) in sizes {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let m = Matrix::from_vec(rows, cols, data).unwrap();

        let start = Instant::now();
        for _ in 0..5 {
            let _ = m.transpose();
        }
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0 / 5.0;
        let gb_per_sec = (rows * cols * 4 * 2) as f64 / 1e9 / (time_ms / 1000.0);

        let color = if gb_per_sec > 10.0 {
            GREEN
        } else if gb_per_sec > 5.0 {
            YELLOW
        } else {
            RED
        };
        println!(
            "  {}x{} {:>8.1}ms {:>5.1} GB/s {} {}{}",
            rows,
            cols,
            time_ms,
            gb_per_sec,
            bar(gb_per_sec, 20.0, 20),
            color,
            RESET
        );
    }
}

fn bench_activations() {
    print_section_header("ACTIVATION FUNCTIONS (vector size: 65536)");

    let size = 65536;
    let data: Vec<f32> = (0..size).map(|i| ((i as f32) * 0.01).sin()).collect();

    let activations: Vec<(&str, ActivationFn)> = vec![
        ("relu", Box::new(|v: &Vector<f32>| v.relu().unwrap())),
        ("sigmoid", Box::new(|v: &Vector<f32>| v.sigmoid().unwrap())),
        ("tanh", Box::new(|v: &Vector<f32>| v.tanh().unwrap())),
        ("softmax", Box::new(|v: &Vector<f32>| v.softmax().unwrap())),
        ("gelu", Box::new(|v: &Vector<f32>| v.gelu().unwrap())),
    ];

    for (name, func) in &activations {
        let v = Vector::from_slice(&data);

        // Warmup
        for _ in 0..3 {
            let _ = func(&v);
        }

        let start = Instant::now();
        for _ in 0..50 {
            let _ = func(&v);
        }
        let elapsed = start.elapsed();
        let time_us = elapsed.as_secs_f64() * 1_000_000.0 / 50.0;
        let throughput = size as f64 / time_us;

        let color = if throughput > 100.0 {
            GREEN
        } else if throughput > 50.0 {
            YELLOW
        } else {
            RED
        };
        println!(
            "  {:>10} {:>8.1}us {:>6.1}M elem/s {} {}{}",
            name,
            time_us,
            throughput,
            bar(throughput, 200.0, 15),
            color,
            RESET
        );
    }
}

fn bench_memory_alloc() {
    print_section_header("MEMORY ALLOCATION OVERHEAD");

    let sizes_mb = [1, 10, 50, 100];

    for size_mb in sizes_mb {
        let elements = size_mb * 1024 * 1024 / 4; // f32 = 4 bytes

        let start = Instant::now();
        for _ in 0..10 {
            let m = Matrix::<f32>::zeros(1, elements);
            std::hint::black_box(&m);
        }
        let elapsed = start.elapsed();
        let time_ms = elapsed.as_secs_f64() * 1000.0 / 10.0;
        let gb_per_sec = (size_mb as f64) / 1000.0 / (time_ms / 1000.0);

        let color = if time_ms < 5.0 {
            GREEN
        } else if time_ms < 20.0 {
            YELLOW
        } else {
            RED
        };
        println!(
            "  {:>3}MB alloc {:>8.1}ms {:>5.1} GB/s {} {}{}",
            size_mb,
            time_ms,
            gb_per_sec,
            bar(gb_per_sec, 50.0, 15),
            color,
            RESET
        );
    }
}

fn main() {
    println!(
        "\n{}{}═══════════════════════════════════════════════════════════════{}",
        BOLD, CYAN, RESET
    );
    println!(
        "{}{}                    TRUENO PERFORMANCE DASHBOARD                {}",
        BOLD, CYAN, RESET
    );
    println!(
        "{}{}═══════════════════════════════════════════════════════════════{}\n",
        BOLD, CYAN, RESET
    );

    bench_vector_ops();
    bench_vecmat_multiply();
    bench_general_matmul();
    bench_transpose();
    bench_activations();
    bench_memory_alloc();

    println!(
        "\n{}{}═══════════════════════════════════════════════════════════════{}",
        BOLD, CYAN, RESET
    );
    println!("{}Legend: {}FAST{} {}OK{} {}SLOW{}", DIM, GREEN, RESET, YELLOW, RESET, RED, RESET);
    println!(
        "{}{}═══════════════════════════════════════════════════════════════{}\n",
        BOLD, CYAN, RESET
    );
}
