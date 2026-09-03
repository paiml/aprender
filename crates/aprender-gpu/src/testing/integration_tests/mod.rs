//! Integration tests using sovereign stack for visual regression
//!
//! Per spec E2E-VISUAL-PROBAR-001: Uses trueno-viz + simular (NO external crates)

use super::*;
use simular::engine::rng::SimRng;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Shared GPU pixel renderer for fast test execution (initialized once)
static SHARED_RENDERER: OnceLock<GpuPixelRenderer> = OnceLock::new();

/// Get shared renderer (fast) - always succeeds since GpuPixelRenderer is CPU-based
fn get_shared_renderer() -> &'static GpuPixelRenderer {
    SHARED_RENDERER.get_or_init(GpuPixelRenderer::new)
}

fn test_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("trueno_sovereign_{}_{}", name, std::process::id()))
}

fn cleanup(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Simulate correct GEMM output
fn simulate_gemm(size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let mut acc = 0.0f32;
            for k in 0..size {
                acc += (i * size + k) as f32 * (k * size + j) as f32;
            }
            output.push(acc);
        }
    }
    output
}

/// Simulate buggy GEMM with uninitialized accumulator
fn simulate_gemm_buggy(size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let garbage = if i % 2 == 0 { 1000.0 } else { 0.0 };
            let mut acc = garbage; // Injected fault: accumulator not initialized to 0
            for k in 0..size {
                acc += (i * size + k) as f32 * (k * size + j) as f32;
            }
            output.push(acc);
        }
    }
    output
}

// ============================================================================
// Tests using sovereign stack (trueno-viz, simular)
// ============================================================================

#[test]
fn test_sovereign_determinism() {
    let renderer = get_shared_renderer();
    let size = 8;
    let output = simulate_gemm(size);

    let png1 = renderer.render_to_png(&output, size as u32, size as u32);
    let png2 = renderer.render_to_png(&output, size as u32, size as u32);

    let result = compare_png_bytes(&png1, &png2, 0);
    assert!(
        result.matches(0.0),
        "Same input should produce identical output"
    );
    assert_eq!(result.different_pixels, 0, "Should be pixel-perfect match");
}

#[test]
fn test_sovereign_detects_bug() {
    let renderer = get_shared_renderer();
    let size = 8;

    let correct = simulate_gemm(size);
    let buggy = simulate_gemm_buggy(size);

    let png_correct = renderer.render_to_png(&correct, size as u32, size as u32);
    let png_buggy = renderer.render_to_png(&buggy, size as u32, size as u32);

    let result = compare_png_bytes(&png_correct, &png_buggy, 0);
    assert!(!result.matches(0.0), "Should detect difference from bug");
    assert!(result.different_pixels > 0, "Should have pixel diffs");

    println!(
        "Bug detected: {} pixels differ ({:.2}%)",
        result.different_pixels,
        result.diff_percentage()
    );
}

#[test]
fn test_sovereign_special_values() {
    let renderer = get_shared_renderer();
    let buffer = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];

    let png = renderer.render_to_png(&buffer, 2, 2);

    // Verify PNG is valid by comparing to itself
    let result = compare_png_bytes(&png, &png, 0);
    assert_eq!(result.different_pixels, 0);
}

#[test]
fn test_sovereign_threshold() {
    let renderer = get_shared_renderer();
    let size = 8;

    let output1 = simulate_gemm(size);
    let mut output2 = output1.clone();
    output2[0] += 0.001; // Tiny change

    let png1 = renderer.render_to_png(&output1, size as u32, size as u32);
    let png2 = renderer.render_to_png(&output2, size as u32, size as u32);

    // Strict threshold (0 tolerance)
    let result_strict = compare_png_bytes(&png1, &png2, 0);

    // Relaxed threshold (allow 1 byte diff)
    let result_relaxed = compare_png_bytes(&png1, &png2, 1);

    println!(
        "Strict: {} diffs, Relaxed: {} diffs",
        result_strict.different_pixels, result_relaxed.different_pixels
    );
}

#[test]
fn test_sovereign_deterministic_rng() {
    // Use simular for deterministic RNG
    let mut rng = SimRng::new(42);

    let input1: Vec<f32> = (0..64)
        .map(|_| rng.gen_range_f64(0.0, 1.0) as f32)
        .collect();

    // Reset RNG with same seed
    let mut rng2 = SimRng::new(42);
    let input2: Vec<f32> = (0..64)
        .map(|_| rng2.gen_range_f64(0.0, 1.0) as f32)
        .collect();

    assert_eq!(input1, input2, "Same seed should produce same sequence");

    let renderer = get_shared_renderer();
    let png1 = renderer.render_to_png(&input1, 8, 8);
    let png2 = renderer.render_to_png(&input2, 8, 8);

    let result = compare_png_bytes(&png1, &png2, 0);
    assert_eq!(result.different_pixels, 0);
}

mod reports;
mod stress_tests;
