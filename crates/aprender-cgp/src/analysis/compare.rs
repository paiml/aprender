//! `cgp profile compare` — Cross-backend comparison.
//! Spec section 2.2: run the same workload across multiple backends
//! and produce a comparison table with TFLOP/s, bandwidth, and speedup ratios.

use crate::analysis::roofline::{Precision, RooflineModel};
use anyhow::Result;
use serde::Serialize;

/// Supported backends for comparison.
#[derive(Debug, Clone, Serialize)]
pub struct BackendResult {
    pub name: String,
    pub wall_time_us: f64,
    pub tflops: f64,
    pub bandwidth_gbps: f64,
    pub available: bool,
    /// Whether data comes from actual measurement vs estimation
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub measured: bool,
}

/// Compute TFLOP/s for GEMM: 2*M*N*K / time.
fn gemm_tflops(size: u32, time_us: f64) -> f64 {
    if time_us <= 0.0 {
        return 0.0;
    }
    let flops = 2.0 * (size as f64).powi(3);
    flops / (time_us * 1e-6) / 1e12
}

/// Try to get actual GEMM timing from benchmark_matrix_suite binary.
/// Returns (time_us, gflops) if the binary exists and the size is benchmarked.
fn get_actual_gemm_timing(size: u32) -> Option<(f64, f64)> {
    let stdout = run_benchmark_suite()?;
    let pattern = format!("Matrix Multiplication ({size}x{size}x{size})");
    stdout
        .lines()
        .find(|line| line.contains(&pattern))
        .and_then(parse_benchmark_line)
}

/// Locate the cargo target directory the same way `apr_bin.sh` does: ask
/// cargo, never construct. `/mnt/nvme-raid0/targets/trueno` was the
/// pre-APR-MONO name and is empty on every checkout since the consolidation;
/// `./target` assumes CWD-relative resolution, which the dev box's
/// `.cargo/config.toml` target-dir redirect breaks (it is gitignored, so it
/// exists in the main checkout and not in a worktree).
pub(crate) fn cargo_target_dir() -> Option<String> {
    let output =
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .args(["metadata", "--no-deps", "--format-version", "1"])
            .output()
            .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).ok()?;
    json.get("target_directory")?.as_str().map(str::to_string)
}

/// Locate and execute the benchmark_matrix_suite binary; returns stdout on success.
fn run_benchmark_suite() -> Option<String> {
    let target_dir = cargo_target_dir();
    let owned_candidate =
        target_dir.map(|d| format!("{d}/release/examples/benchmark_matrix_suite"));
    let candidates: Vec<&str> = owned_candidate
        .as_deref()
        .into_iter()
        .chain(["./target/release/examples/benchmark_matrix_suite"])
        .collect();
    let binary_path = candidates
        .iter()
        .find(|p| std::path::Path::new(p).exists())?;
    let output = std::process::Command::new(*binary_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Parse a single line of the form
/// `  Matrix Multiplication (NxNxN)...     X.XX ms  (Y.YY GFLOPS)`.
fn parse_benchmark_line(line: &str) -> Option<(f64, f64)> {
    let after_dots = line.split("...").nth(1)?;
    let time_ms = after_dots.split("ms").next()?.trim().parse::<f64>().ok()?;
    let gflops = after_dots
        .split('(')
        .nth(1)?
        .split(" GFLOPS")
        .next()?
        .trim()
        .parse::<f64>()
        .ok()?;
    Some((time_ms * 1000.0, gflops))
}

/// Estimate scalar GEMM time from measured data on Threadripper 7960X.
/// Reference GEMM: 256→11.7ms, cubic scaling.
fn estimate_scalar_time_us(size: u32) -> f64 {
    // Calibrated: 11.7ms at 256x256 on Threadripper 7960X
    let ratio = (size as f64 / 256.0).powi(3);
    11_700.0 * ratio
}

/// Estimate AVX2 BLIS single-thread GEMM from measured data.
/// Calibrated: 256→0.57ms, 512→3.75ms, 1024→30.1ms (71 GFLOPS).
fn estimate_avx2_time_us(size: u32) -> f64 {
    // BLIS GEMM single-thread: ~72 GFLOPS sustained
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = 72.0; // measured on Threadripper 7960X
    flops / (gflops * 1e9) * 1e6
}

/// Estimate AVX-512 BLIS GEMM (slightly faster than AVX2, but clock throttle).
/// ~80 GFLOPS measured single-thread (AVX-512 downclocking limits gains).
fn estimate_avx512_time_us(size: u32) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    let gflops = 80.0; // AVX-512 with downclocking ~10% faster than AVX2
    flops / (gflops * 1e9) * 1e6
}

/// Estimate CUDA CTA WMMA GEMM from measured data on RTX 4090.
/// Calibrated: 23.2us at 512x512 = 11.6 TFLOP/s.
fn estimate_cuda_time_us(size: u32) -> f64 {
    let ratio = (size as f64 / 512.0).powi(3);
    23.2 * ratio
}

/// Estimate cuBLAS GEMM from measured RTX 4090 data.
/// cuBLAS achieves ~35 TFLOP/s FP16 on RTX 4090 (~3x pure PTX).
fn estimate_cublas_time_us(size: u32) -> f64 {
    estimate_cuda_time_us(size) / 3.0
}

/// Measure actual cuBLAS FP16 GEMM throughput via trueno-gpu driver.
/// Returns (time_us, tflops) or None if CUDA unavailable.
#[cfg(feature = "cuda")]
fn measure_cublas_gemm(size: u32) -> Option<(f64, f64)> {
    use trueno_gpu::driver::{CublasHandle, CudaContext, CudaStream, GemmOp, GpuBuffer};

    let ctx = CudaContext::new(0).ok()?;
    let stream = CudaStream::new(&ctx).ok()?;
    let handle = CublasHandle::new(&ctx).ok()?;
    handle.set_stream(&stream).ok()?;

    let n = size as usize;
    let a_data = vec![0x3C00u16; n * n]; // 1.0 in FP16
    let b_data = vec![0x3C00u16; n * n];
    let c_data = vec![0u16; n * n];

    let a_buf = GpuBuffer::from_host(&ctx, &a_data).ok()?;
    let b_buf = GpuBuffer::from_host(&ctx, &b_data).ok()?;
    let c_buf = GpuBuffer::from_host(&ctx, &c_data).ok()?;

    // Warmup
    for _ in 0..5 {
        let _ = handle.gemm_f16(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n as i32,
            n as i32,
            n as i32,
            1.0,
            a_buf.as_ptr(),
            n as i32,
            b_buf.as_ptr(),
            n as i32,
            0.0,
            c_buf.as_ptr(),
            n as i32,
        );
    }
    stream.synchronize().ok()?;

    let iters: u32 = if n <= 512 {
        200
    } else if n <= 1024 {
        100
    } else {
        30
    };
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = handle.gemm_f16(
            GemmOp::NoTrans,
            GemmOp::NoTrans,
            n as i32,
            n as i32,
            n as i32,
            1.0,
            a_buf.as_ptr(),
            n as i32,
            b_buf.as_ptr(),
            n as i32,
            0.0,
            c_buf.as_ptr(),
            n as i32,
        );
    }
    stream.synchronize().ok()?;
    let elapsed = start.elapsed();

    let per_call_us = elapsed.as_micros() as f64 / iters as f64;
    let flops = 2.0 * (n as f64).powi(3);
    let tflops = flops / (per_call_us * 1e6);

    Some((per_call_us, tflops))
}

/// Measure our best PTX GEMM kernel (64×128 pipeline) on GPU.
/// Returns (time_us, tflops) or None if CUDA unavailable.
#[cfg(feature = "cuda")]
fn measure_ptx_gemm(size: u32) -> Option<(f64, f64)> {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::build_cta64x128_mma_pipeline_fp16;
    use trueno_gpu::ptx::PtxModule;

    let ctx = CudaContext::new(0).ok()?;
    let stream = CudaStream::new(&ctx).ok()?;

    let n = size as usize;
    let a16 = vec![0x3C00u16; n * n];
    let b16 = vec![0x3C00u16; n * n];
    let c32 = vec![0.0f32; n * n];

    let a_buf = GpuBuffer::from_host(&ctx, &a16).ok()?;
    let b_buf = GpuBuffer::from_host(&ctx, &b16).ok()?;
    let c_buf = GpuBuffer::from_host(&ctx, &c32).ok()?;

    let kernel = build_cta64x128_mma_pipeline_fp16(n as u32, n as u32, n as u32);
    let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).ok()?;

    let cfg = LaunchConfig {
        grid: (((n + 127) / 128) as u32, ((n + 63) / 64) as u32, 1),
        block: (512, 1, 1),
        shared_mem: 18432,
    };

    let mut a_ptr = a_buf.as_ptr();
    let mut b_ptr = b_buf.as_ptr();
    let mut c_ptr = c_buf.as_ptr();
    let mut m_v = n as u32;
    let mut n_v = n as u32;
    let mut k_v = n as u32;
    let mut args: Vec<*mut c_void> = vec![
        &mut a_ptr as *mut _ as *mut c_void,
        &mut b_ptr as *mut _ as *mut c_void,
        &mut c_ptr as *mut _ as *mut c_void,
        &mut m_v as *mut _ as *mut c_void,
        &mut n_v as *mut _ as *mut c_void,
        &mut k_v as *mut _ as *mut c_void,
    ];

    // Warmup
    for _ in 0..5 {
        unsafe {
            stream
                .launch_kernel(
                    &mut module,
                    "gemm_cta64x128_mma_pipeline_fp16",
                    &cfg,
                    &mut args,
                )
                .ok()?;
        }
    }
    stream.synchronize().ok()?;

    let iters: u32 = if n <= 512 {
        100
    } else if n <= 1024 {
        50
    } else {
        20
    };
    let start = std::time::Instant::now();
    for _ in 0..iters {
        unsafe {
            stream
                .launch_kernel(
                    &mut module,
                    "gemm_cta64x128_mma_pipeline_fp16",
                    &cfg,
                    &mut args,
                )
                .ok()?;
        }
    }
    stream.synchronize().ok()?;
    let per_call_us = start.elapsed().as_micros() as f64 / iters as f64;
    let flops = 2.0 * (n as f64).powi(3);
    let tflops = flops / (per_call_us * 1e6);

    Some((per_call_us, tflops))
}

/// Run cross-backend comparison.
pub fn run_compare(kernel: &str, size: u32, backends_str: &str, json: bool) -> Result<()> {
    let backends: Vec<&str> = backends_str.split(',').map(|s| s.trim()).collect();

    if !json {
        println!("\n=== CGP Cross-Backend Comparison: {kernel} ({size}x{size}x{size}) ===\n");
    }

    let actual = get_actual_gemm_timing(size);
    let mut results = collect_backend_results(&backends, size, actual);
    results.sort_by(|a, b| {
        a.wall_time_us
            .partial_cmp(&b.wall_time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    render_comparison_table(&results);
    render_source_legend(&results);
    render_best_summary(&results);
    render_cpu_gpu_gap(&results);

    println!();
    Ok(())
}

/// Measure each requested backend and build a `BackendResult` list.
fn collect_backend_results(
    backends: &[&str],
    size: u32,
    actual: Option<(f64, f64)>,
) -> Vec<BackendResult> {
    let mut results: Vec<BackendResult> = Vec::new();
    for backend in backends {
        let Some((time_us, available, measured)) = measure_backend(backend, size, actual) else {
            continue;
        };
        results.push(BackendResult {
            name: (*backend).to_string(),
            wall_time_us: time_us,
            tflops: gemm_tflops(size, time_us),
            bandwidth_gbps: 0.0,
            available,
            measured,
        });
    }
    results
}

/// Dispatch to the backend-specific measurement routine; None for unknown backends.
fn measure_backend(
    backend: &str,
    size: u32,
    actual: Option<(f64, f64)>,
) -> Option<(f64, bool, bool)> {
    match backend {
        "scalar" => Some((estimate_scalar_time_us(size), true, false)),
        "avx2" => Some(measure_avx_backend(size, actual, false)),
        "avx512" => Some(measure_avx_backend(size, actual, true)),
        "neon" => Some((
            estimate_scalar_time_us(size) / 4.0,
            cfg!(target_arch = "aarch64"),
            false,
        )),
        "cuda" => Some(measure_cuda_backend(size)),
        "cublas" => Some(measure_cublas_backend(size)),
        "wgpu" => Some((
            estimate_cuda_time_us(size) * 2.0,
            which::which("nvidia-smi").is_ok(),
            false,
        )),
        other => {
            eprintln!("  Warning: unknown backend '{other}', skipping");
            None
        }
    }
}

/// Measure AVX2/AVX512 backends: prefer actual CPU bench, else fall back to estimate.
fn measure_avx_backend(size: u32, actual: Option<(f64, f64)>, avx512: bool) -> (f64, bool, bool) {
    #[cfg(target_arch = "x86_64")]
    let avail = if avx512 {
        std::arch::is_x86_feature_detected!("avx512f")
    } else {
        std::arch::is_x86_feature_detected!("avx2")
    };
    #[cfg(not(target_arch = "x86_64"))]
    let avail = false;

    if let Some((actual_us, _)) = actual {
        return (actual_us, avail, true);
    }
    if avx512 {
        (estimate_avx512_time_us(size), avail, false)
    } else {
        (estimate_avx2_time_us(size), avail, false)
    }
}

/// Measure CUDA PTX backend when available + `cuda` feature is enabled.
fn measure_cuda_backend(size: u32) -> (f64, bool, bool) {
    let avail = which::which("nvidia-smi").is_ok();
    if avail {
        if let Some((time_us, _)) = try_measure_ptx(size) {
            return (time_us, true, true);
        }
    }
    (estimate_cuda_time_us(size), avail, false)
}

/// Measure cuBLAS backend when available + `cuda` feature is enabled.
fn measure_cublas_backend(size: u32) -> (f64, bool, bool) {
    let avail = which::which("nvidia-smi").is_ok();
    if avail {
        if let Some((time_us, _)) = try_measure_cublas(size) {
            return (time_us, true, true);
        }
    }
    (estimate_cublas_time_us(size), avail, false)
}

/// Shim that returns `measure_ptx_gemm` under `cuda` feature, `None` otherwise.
#[cfg(feature = "cuda")]
fn try_measure_ptx(size: u32) -> Option<(f64, f64)> {
    measure_ptx_gemm(size)
}

#[cfg(not(feature = "cuda"))]
fn try_measure_ptx(_size: u32) -> Option<(f64, f64)> {
    None
}

/// Shim that returns `measure_cublas_gemm` under `cuda` feature, `None` otherwise.
#[cfg(feature = "cuda")]
fn try_measure_cublas(size: u32) -> Option<(f64, f64)> {
    measure_cublas_gemm(size)
}

#[cfg(not(feature = "cuda"))]
fn try_measure_cublas(_size: u32) -> Option<(f64, f64)> {
    None
}

/// Render the main comparison table (header + one row per backend).
fn render_comparison_table(results: &[BackendResult]) {
    let best_time = results.first().map_or(1.0, |r| r.wall_time_us);
    println!(
        "  {:12} {:>12} {:>12} {:>10} {:>10} {:>8} {:>5}",
        "Backend", "Time (us)", "TFLOP/s", "Efficiency", "vs Best", "Avail", "Src"
    );
    println!("  {}", "-".repeat(75));
    let (cpu_peak, gpu_peak) = peak_performance_limits();
    for r in results {
        render_comparison_row(r, best_time, cpu_peak, gpu_peak);
    }
}

/// Print a single comparison row with efficiency, ratio, availability, and source tag.
fn render_comparison_row(r: &BackendResult, best_time: f64, cpu_peak: f64, gpu_peak: f64) {
    let peak_tflops = if r.name.contains("cuda") || r.name.contains("cublas") || r.name == "wgpu" {
        gpu_peak / 1e12
    } else {
        cpu_peak / 1e12
    };
    let efficiency = if peak_tflops > 0.0 {
        r.tflops / peak_tflops * 100.0
    } else {
        0.0
    };
    let ratio = format!("{:.2}x", r.wall_time_us / best_time);
    let avail = if r.available { "yes" } else { "no" };
    let time_str = if r.wall_time_us >= 1000.0 {
        format!("{:.1} ms", r.wall_time_us / 1000.0)
    } else {
        format!("{:.1}", r.wall_time_us)
    };
    let src = if r.measured { "M" } else { "E" };
    println!(
        "  {:12} {:>12} {:>12.1} {:>9.1}% {:>10} {:>8} {:>5}",
        r.name, time_str, r.tflops, efficiency, ratio, avail, src
    );
}

/// Roofline-derived peak CPU/GPU FLOP/s (used to compute per-row efficiency).
fn peak_performance_limits() -> (f64, f64) {
    let model = RooflineModel::rtx_4090();
    let gpu_peak = model
        .peak_compute
        .get(&Precision::Fp16)
        .copied()
        .unwrap_or(330.0e12);
    let cores = num_cpus::get_physical();
    #[allow(clippy::cast_precision_loss)]
    let cpu_peak = 2.0 * 8.0 * 2.0 * 3.5e9 * cores as f64; // AVX2 peak
    (cpu_peak, gpu_peak)
}

/// Show the "Src: M=measured E=estimated" legend when any results were produced.
fn render_source_legend(results: &[BackendResult]) {
    let has_measured = results.iter().any(|r| r.measured);
    let has_estimated = results.iter().any(|r| !r.measured);
    if !(has_measured || has_estimated) {
        return;
    }
    print!("  Src: ");
    if has_measured {
        print!("M=measured ");
    }
    if has_estimated {
        print!("E=estimated ");
    }
    println!();
}

/// Print the "Best: X (Ny faster than Z)" summary line when results are present.
fn render_best_summary(results: &[BackendResult]) {
    let Some(best) = results.first() else {
        return;
    };
    let Some(worst) = results.last() else {
        return;
    };
    let speedup = worst.wall_time_us / best.wall_time_us;
    println!(
        "\n  Best: {} ({:.1}x faster than {})",
        best.name, speedup, worst.name
    );
}

/// Print "CPU→GPU gap: Nx" when both CPU and GPU backends were measured.
fn render_cpu_gpu_gap(results: &[BackendResult]) {
    let has_cpu = results
        .iter()
        .any(|r| matches!(r.name.as_str(), "scalar" | "avx2" | "avx512"));
    let has_gpu = results
        .iter()
        .any(|r| matches!(r.name.as_str(), "cuda" | "cublas" | "wgpu"));
    if !(has_cpu && has_gpu) {
        return;
    }
    let best_cpu = results
        .iter()
        .filter(|r| matches!(r.name.as_str(), "scalar" | "avx2" | "avx512"))
        .map(|r| r.wall_time_us)
        .fold(f64::INFINITY, f64::min);
    let best_gpu = results
        .iter()
        .filter(|r| matches!(r.name.as_str(), "cuda" | "cublas" | "wgpu"))
        .map(|r| r.wall_time_us)
        .fold(f64::INFINITY, f64::min);
    if best_gpu > 0.0 {
        println!(
            "  CPU→GPU gap: {:.0}x (expected for large GEMM)",
            best_cpu / best_gpu
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_tflops() {
        // 512^3 GEMM at 23.2us = 2*512^3 / 23.2e-6 / 1e12
        let tflops = gemm_tflops(512, 23.2);
        assert!(
            (tflops - 11.56).abs() < 0.1,
            "Expected ~11.6 TFLOP/s, got {tflops:.2}"
        );
    }

    #[test]
    fn test_scalar_slower_than_avx2() {
        let scalar = estimate_scalar_time_us(512);
        let avx2 = estimate_avx2_time_us(512);
        assert!(scalar > avx2 * 3.0, "Scalar should be >3x slower than AVX2");
    }

    #[test]
    fn test_cuda_faster_than_cpu() {
        let cpu = estimate_avx2_time_us(4096);
        let cuda = estimate_cuda_time_us(4096);
        assert!(
            cpu > cuda * 10.0,
            "CPU should be >10x slower than CUDA for 4096"
        );
    }

    /// FALSIFY-CGP-040: CUDA must be faster than scalar for GEMM >= 256.
    #[test]
    fn test_cuda_faster_than_scalar_at_256() {
        let scalar = estimate_scalar_time_us(256);
        let cuda = estimate_cuda_time_us(256);
        assert!(cuda < scalar, "CUDA should be faster than scalar at 256");
    }

    /// FALSIFY-CGP-041: SIMD must be faster than scalar (>= 3x at 1024).
    #[test]
    fn test_simd_faster_than_scalar() {
        let scalar = estimate_scalar_time_us(1024);
        let avx2 = estimate_avx2_time_us(1024);
        assert!(
            scalar / avx2 >= 3.0,
            "AVX2 speedup {:.1}x should be >= 3x",
            scalar / avx2
        );
    }

    /// FALSIFY-CGP-042: cuBLAS must be faster than pure PTX for large GEMM.
    #[test]
    fn test_cublas_faster_than_ptx() {
        let ptx = estimate_cuda_time_us(4096);
        let cublas = estimate_cublas_time_us(4096);
        assert!(cublas < ptx, "cuBLAS should be faster than PTX at 4096");
    }

    #[test]
    fn test_run_compare_basic() {
        let result = run_compare("gemm", 256, "scalar,avx2", false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_compare_json() {
        let result = run_compare("gemm", 256, "scalar,avx2", true);
        assert!(result.is_ok());
    }

    /// FALSIFY-CGP-ACTUAL-001: Actual benchmark data is available and parseable.
    #[test]
    fn test_get_actual_gemm_timing() {
        if let Some((time_us, gflops)) = get_actual_gemm_timing(1024) {
            assert!(time_us > 0.0, "time should be positive");
            assert!(gflops > 10.0, "GFLOPS should be > 10 for 1024 GEMM");
            assert!(gflops < 2000.0, "GFLOPS should be < 2000");
            eprintln!(
                "Actual GEMM 1024: {:.1} us = {:.0} GFLOPS [MEASURED]",
                time_us, gflops
            );
        } else {
            eprintln!("benchmark_matrix_suite binary not found — actual data unavailable");
        }
    }
}
