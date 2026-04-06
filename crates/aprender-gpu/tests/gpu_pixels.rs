//! GPU Pixel Testing Integration with Probar TUI Visualization
//!
//! Run with: cargo test --test gpu_pixels --features gpu-pixels
//! TUI mode: cargo test --test gpu_pixels --features gpu-pixels -- --nocapture
//!
//! NOTE: This test requires jugar-probar v0.4+ with gpu_pixels module.

#![cfg(feature = "gpu-pixels")]

use jugar_probar::gpu_pixels::{
    run_kernel_pixels, validate_ptx, GpuRegressionSuite, KernelPixelConfig, PtxBugClass,
    RegressionConfig,
};
use trueno_gpu::kernels::{AttentionKernel, GemmKernel, Kernel, LayerNormKernel, SoftmaxKernel};

/// TUI Reporter for GPU Pixel Tests
#[cfg(feature = "gpu-pixels")]
mod tui_report {
    use crossterm::{
        event::{self, Event, KeyCode},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use jugar_probar::gpu_pixels::GpuPixelTestSuite;
    use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
    use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
    use presentar_terminal::{ColorMode, Theme};
    use std::io;

    pub struct GpuPixelTuiReport {
        pub suites: Vec<GpuPixelTestSuite>,
    }

    impl GpuPixelTuiReport {
        pub fn new() -> Self {
            Self { suites: Vec::new() }
        }

        pub fn add_suite(&mut self, suite: GpuPixelTestSuite) {
            self.suites.push(suite);
        }

        pub fn total_passed(&self) -> usize {
            self.suites.iter().map(|s| s.passed_count()).sum()
        }

        pub fn total_tests(&self) -> usize {
            self.suites.iter().map(|s| s.results.len()).sum()
        }

        pub fn all_passed(&self) -> bool {
            self.suites.iter().all(|s| s.all_passed())
        }

        pub fn render_to_terminal(&self) -> io::Result<()> {
            enable_raw_mode()?;
            let mut stdout = io::stdout();
            execute!(stdout, EnterAlternateScreen)?;

            let (width, height) = crossterm::terminal::size()?;
            let mut buffer = CellBuffer::new(width, height);
            let renderer = DiffRenderer::with_color_mode(ColorMode::TrueColor);
            let theme = Theme::tokyo_night();

            {
                let mut canvas = DirectTerminalCanvas::new(&mut buffer);
                self.render_ui(&mut canvas, width, height, &theme);
            }

            let mut output = Vec::with_capacity(8192);
            renderer.flush(&mut buffer, &mut output).ok();
            std::io::Write::write_all(&mut stdout, &output)?;

            // Wait for key press
            loop {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                        break;
                    }
                }
            }

            disable_raw_mode()?;
            execute!(stdout, LeaveAlternateScreen)?;
            Ok(())
        }

        fn render_ui(
            &self,
            canvas: &mut DirectTerminalCanvas<'_>,
            width: u16,
            height: u16,
            _theme: &Theme,
        ) {
            let green = Color::new(0.3, 1.0, 0.3, 1.0);
            let red = Color::new(1.0, 0.3, 0.3, 1.0);
            let yellow = Color::new(1.0, 1.0, 0.3, 1.0);
            let cyan = Color::new(0.3, 1.0, 1.0, 1.0);
            let gray = Color::new(0.5, 0.5, 0.5, 1.0);

            let _w = width as f32;
            let mut y: f32 = 1.0;

            // Header: Summary
            let total_passed: usize = self.suites.iter().map(|s| s.passed_count()).sum();
            let total_tests: usize = self.suites.iter().map(|s| s.results.len()).sum();
            let pass_rate = if total_tests > 0 {
                (total_passed as f64 / total_tests as f64) * 100.0
            } else {
                0.0
            };

            let header_color = if pass_rate == 100.0 { green } else { yellow };
            let header_style = TextStyle::default().with_color(header_color);

            canvas.draw_text(
                Point::new(1.0, y),
                &format!(
                    "--- Summary ---  GPU Pixel Tests: {}/{} passed ({:.1}%)",
                    total_passed, total_tests, pass_rate
                ),
                header_style,
            );
            y += 2.0;

            // Kernel results
            canvas.draw_text(
                Point::new(1.0, y),
                "--- Kernels ---",
                TextStyle::default().with_color(gray),
            );
            y += 1.0;

            for suite in &self.suites {
                if y >= height as f32 - 6.0 {
                    break;
                }
                let status = if suite.all_passed() { "✓" } else { "✗" };
                let color = if suite.all_passed() { green } else { red };
                let style = TextStyle::default().with_color(color);

                let line = format!(
                    "[{}] {} ({}/{})",
                    status,
                    suite.kernel_name,
                    suite.passed_count(),
                    suite.results.len()
                );
                canvas.draw_text(Point::new(2.0, y), &line, style);
                y += 1.0;
            }

            y += 1.0;

            // Bug class legend
            let legend_style = TextStyle::default().with_color(cyan);
            canvas.draw_text(
                Point::new(1.0, y),
                "--- Legend (press 'q' to exit) ---",
                legend_style,
            );
            y += 1.0;
            canvas.draw_text(Point::new(1.0, y), "Bug Classes Detected:", legend_style);
            y += 1.0;
            canvas.draw_text(
                Point::new(1.0, y),
                "  shared_mem_u64 - Shared memory uses 64-bit addressing (should be 32-bit)",
                legend_style,
            );
            y += 1.0;
            canvas.draw_text(
                Point::new(1.0, y),
                "  loop_branch_end - Loop branches to END instead of START",
                legend_style,
            );
            y += 1.0;
            canvas.draw_text(
                Point::new(1.0, y),
                "  missing_barrier - No bar.sync with shared memory",
                legend_style,
            );
        }

        pub fn print_summary(&self) {
            let total_passed: usize = self.suites.iter().map(|s| s.passed_count()).sum();
            let total_tests: usize = self.suites.iter().map(|s| s.results.len()).sum();
            let pass_rate = if total_tests > 0 {
                (total_passed as f64 / total_tests as f64) * 100.0
            } else {
                0.0
            };
            let all_pass = total_passed == total_tests;

            // Header
            println!();
            println!("┌─────────────────────────────────────────────────────────────────────┐");
            println!("│                     GPU PIXEL TEST REPORT                           │");
            println!("│                     trueno-gpu + probar                             │");
            println!("├─────────────────────────────────────────────────────────────────────┤");

            // Summary bar
            let bar_width = 40;
            let filled = (pass_rate / 100.0 * bar_width as f64) as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
            let status_color = if all_pass { "\x1b[32m" } else { "\x1b[33m" };
            println!(
                "│  Tests: {}{:>3}/{:<3}\x1b[0m ({:>5.1}%)  [{}]  │",
                status_color, total_passed, total_tests, pass_rate, bar
            );
            println!("├─────────────────────────────────────────────────────────────────────┤");
            println!("│  KERNELS                                                            │");
            println!("├─────────────────────────────────────────────────────────────────────┤");

            // Kernel results
            for suite in &self.suites {
                let passed = suite.passed_count();
                let total = suite.results.len();
                let suite_pass = passed == total;
                let status = if suite_pass { "✓" } else { "✗" };
                let color = if suite_pass { "\x1b[32m" } else { "\x1b[31m" };

                // Mini progress bar
                let mini_bar_width = 10;
                let mini_filled = (passed as f64 / total as f64 * mini_bar_width as f64) as usize;
                let mini_bar: String =
                    "▓".repeat(mini_filled) + &"░".repeat(mini_bar_width - mini_filled);

                println!(
                    "│  {}[{}]\x1b[0m {:<32} {:>2}/{:<2} [{}] │",
                    color, status, suite.kernel_name, passed, total, mini_bar
                );

                // Show failures with details
                for result in suite.failures() {
                    println!("│      └─ \x1b[31m{:<55}\x1b[0m │", result.name);
                    if let Some(err) = &result.error {
                        let truncated = if err.len() > 50 { &err[..50] } else { err.as_str() };
                        println!("│         └─ {:52} │", truncated);
                    }
                }
            }

            // Bug classes legend
            println!("├─────────────────────────────────────────────────────────────────────┤");
            println!("│  BUG CLASSES DETECTED                                               │");
            println!("├─────────────────────────────────────────────────────────────────────┤");
            println!("│  \x1b[36mshared_mem_u64\x1b[0m  - Shared memory 64-bit addressing (should be 32)  │");
            println!("│  \x1b[36mloop_branch_end\x1b[0m - Loop branches to END instead of START          │");
            println!("│  \x1b[36mmissing_barrier\x1b[0m - No bar.sync with shared memory                 │");
            println!("│  \x1b[36mmissing_entry\x1b[0m   - Kernel entry point missing                     │");

            // Statistics
            println!("├─────────────────────────────────────────────────────────────────────┤");
            println!("│  STATISTICS                                                         │");
            println!("├─────────────────────────────────────────────────────────────────────┤");
            println!(
                "│  Kernels Tested:  {:>3}                                               │",
                self.suites.len()
            );
            println!(
                "│  Pixel Tests:     {:>3}                                               │",
                total_tests
            );
            println!(
                "│  Bugs Found:      {:>3}                                               │",
                total_tests - total_passed
            );

            // Footer
            println!("├─────────────────────────────────────────────────────────────────────┤");
            if all_pass {
                println!("│  \x1b[32m✓ ALL TESTS PASSED\x1b[0m                                                 │");
            } else {
                println!("│  \x1b[31m✗ {} TESTS FAILED\x1b[0m                                                  │", total_tests - total_passed);
            }
            println!("└─────────────────────────────────────────────────────────────────────┘");
            println!();
        }
    }
}

#[cfg(feature = "gpu-pixels")]
use tui_report::GpuPixelTuiReport;

// ============================================================================
// PIXEL TESTS: Each test validates one atomic kernel property
// ============================================================================

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_gemm_tiled_shared_mem_addressing() {
    let kernel = GemmKernel::tiled(32, 32, 128, 32);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "GEMM kernel uses u64 for shared memory (should be u32)"
    );
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_gemm_tensor_core_shared_mem_addressing() {
    let kernel = GemmKernel::tensor_core(32, 32, 128);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "Tensor core GEMM uses u64 for shared memory"
    );
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_attention_shared_mem_addressing() {
    let kernel = AttentionKernel::new(64, 64);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "Attention kernel uses u64 for shared memory"
    );
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_attention_causal_kernel_name() {
    let kernel = AttentionKernel::new(64, 64).with_causal();
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("flash_attention_causal"),
        "Causal attention should have _causal suffix in kernel name"
    );
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_attention_barrier_sync() {
    let kernel = AttentionKernel::new(64, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("bar.sync"), "Attention kernel must have barrier synchronization");
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_gemm_barrier_sync() {
    let kernel = GemmKernel::tiled(32, 32, 64, 32);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("bar.sync"), "Tiled GEMM must have barrier synchronization");
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_softmax_kernel_entry() {
    let kernel = SoftmaxKernel::new(128);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        "Softmax kernel must have entry point"
    );
}

#[test]
#[cfg(feature = "gpu-pixels")]
fn pixel_layernorm_kernel_entry() {
    let kernel = LayerNormKernel::new(256);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        "LayerNorm kernel must have entry point"
    );
}

// ============================================================================
// FULL SUITE TEST: Runs all pixel tests with TUI report
// ============================================================================

#[test]
#[cfg(feature = "gpu-pixels")]
fn gpu_pixel_suite_all_kernels() {
    let config = KernelPixelConfig::default();
    let mut report = GpuPixelTuiReport::new();

    // Test all kernel types
    let kernels: Vec<(&str, String)> = vec![
        ("gemm_tiled_32x32x64", GemmKernel::tiled(32, 32, 64, 32).emit_ptx()),
        ("gemm_tiled_64x64x128", GemmKernel::tiled(64, 64, 128, 32).emit_ptx()),
        ("gemm_tensor_core", GemmKernel::tensor_core(32, 32, 64).emit_ptx()),
        ("attention_64x64", AttentionKernel::new(64, 64).emit_ptx()),
        ("attention_causal", AttentionKernel::new(64, 64).with_causal().emit_ptx()),
        ("softmax_128", SoftmaxKernel::new(128).emit_ptx()),
        ("layernorm_256", LayerNormKernel::new(256).emit_ptx()),
    ];

    for (name, ptx) in &kernels {
        let suite = run_kernel_pixels(name, ptx, &config);
        report.add_suite(suite);
    }

    // Print summary (always)
    report.print_summary();

    // TUI mode if TTY and --nocapture
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        if let Err(e) = report.render_to_terminal() {
            eprintln!("TUI render error: {e}");
        }
    }

    // Assert all passed
    assert!(
        report.all_passed(),
        "Not all GPU pixel tests passed: {}/{}",
        report.total_passed(),
        report.total_tests()
    );
}

// ============================================================================
// REGRESSION SUITE: Detect PTX regressions between versions
// ============================================================================

#[test]
#[cfg(feature = "gpu-pixels")]
fn gpu_pixel_regression_detection() {
    let config = RegressionConfig::default();
    let mut suite = GpuRegressionSuite::new(config);

    // Add current PTX as baseline (would normally load from file)
    let gemm_ptx = GemmKernel::tiled(32, 32, 64, 32).emit_ptx();
    suite.add_baseline("gemm_tiled", &gemm_ptx);

    // Test current version against baseline
    let result = suite.test_kernel("gemm_tiled", &gemm_ptx);

    assert!(!result.is_regression, "Unexpected regression detected");
    assert!(result.pixel_results.all_passed(), "Pixel tests failed");
}
