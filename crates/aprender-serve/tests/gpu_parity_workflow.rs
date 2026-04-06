//! PARITY-114: CUDA vs wgpu Matmul Parity Workflow Tests
//!
//! This test suite verifies that CUDA and wgpu schedulers produce identical
//! results for matrix multiplication operations. It "watches the flow" of
//! data through both paths to catch parity bugs.
//!
//! # Running
//! ```bash
//! # Run all parity tests
//! cargo test --test gpu_parity_workflow --features cuda -- --nocapture
//!
//! # Run specific test
//! cargo test --test gpu_parity_workflow test_parity_114a --features cuda -- --nocapture
//! ```
//!
//! # Bug History
//! - PARITY-114: CudaScheduler wiring bug caused 8x smaller results than wgpu
//!   - Root cause: [identified by this test suite]
//!   - Fixed in: [commit hash TBD]
//!
//! # Toyota Way Alignment
//! - **Genchi Genbutsu**: Test actual GPU execution, not mocks
//! - **Jidoka**: Stop when numerical parity fails
//! - **Poka-Yoke**: Catch scheduler bugs before inference runs

#[allow(unused_imports)]
use std::time::Instant;

/// Tolerance for floating-point comparison
const EPSILON: f32 = 1e-4;

/// Test configuration for workflow tests
#[derive(Debug, Clone)]
pub struct WorkflowConfig {
    /// Matrix M dimension (rows of A, rows of C)
    pub m: usize,
    /// Matrix K dimension (cols of A, rows of B)
    pub k: usize,
    /// Matrix N dimension (cols of B, cols of C)
    pub n: usize,
    /// Show verbose flow output
    pub verbose: bool,
}

impl WorkflowConfig {
    #[allow(dead_code)]
    fn small() -> Self {
        Self {
            m: 4,
            k: 64,
            n: 192,
            verbose: true,
        }
    }

    #[allow(dead_code)]
    fn single_tile() -> Self {
        Self {
            m: 4,
            k: 32,
            n: 64,
            verbose: true,
        }
    }
}

/// Result of a scheduler matmul operation
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MatmulResult {
    /// Output matrix
    pub output: Vec<f32>,
    /// Execution time
    pub duration: std::time::Duration,
    /// Scheduler name
    pub scheduler: String,
    /// Sum of all output elements
    pub output_sum: f32,
    /// Min value in output
    pub output_min: f32,
    /// Max value in output
    pub output_max: f32,
}

impl MatmulResult {
    #[allow(dead_code)]
    fn new(output: Vec<f32>, duration: std::time::Duration, scheduler: &str) -> Self {
        let output_sum: f32 = output.iter().sum();
        let output_min = output.iter().copied().fold(f32::INFINITY, f32::min);
        let output_max = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Self {
            output,
            duration,
            scheduler: scheduler.to_string(),
            output_sum,
            output_min,
            output_max,
        }
    }

    #[allow(dead_code)]
    fn print_summary(&self) {
        println!("  {} Results:", self.scheduler);
        println!("    Duration: {:?}", self.duration);
        println!("    Output sum: {:.6}", self.output_sum);
        println!("    Output min: {:.6}", self.output_min);
        println!("    Output max: {:.6}", self.output_max);
        println!(
            "    First 5: {:?}",
            &self.output[..self.output.len().min(5)]
        );
    }
}

/// Compare two matmul results for parity
#[allow(dead_code)]
fn check_parity(result_a: &MatmulResult, result_b: &MatmulResult) -> ParityResult {
    let max_diff = result_a
        .output
        .iter()
        .zip(result_b.output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let avg_diff = result_a
        .output
        .iter()
        .zip(result_b.output.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / result_a.output.len() as f32;

    let sum_ratio = if result_b.output_sum.abs() > 1e-10 {
        result_a.output_sum / result_b.output_sum
    } else {
        f32::INFINITY
    };

    let passed = max_diff < EPSILON;

    ParityResult {
        passed,
        max_diff,
        avg_diff,
        sum_ratio,
        scheduler_a: result_a.scheduler.clone(),
        scheduler_b: result_b.scheduler.clone(),
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct ParityResult {
    passed: bool,
    max_diff: f32,
    avg_diff: f32,
    sum_ratio: f32,
    scheduler_a: String,
    scheduler_b: String,
}

impl ParityResult {
    #[allow(dead_code)]
    fn print_summary(&self) {
        let status = if self.passed { "PASS" } else { "FAIL" };
        println!("\n  Parity Check [{status}]:");
        println!("    {} vs {}", self.scheduler_a, self.scheduler_b);
        println!(
            "    Max diff: {:.6} (tolerance: {})",
            self.max_diff, EPSILON
        );
        println!("    Avg diff: {:.6}", self.avg_diff);
        println!("    Sum ratio: {:.6} (should be ~1.0)", self.sum_ratio);
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

/// CPU reference matmul for ground truth comparison
#[allow(dead_code, clippy::many_single_char_names)]
fn cpu_matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ============================================================================
// PARITY-114 Test Suite
// ============================================================================

/// PARITY-114a: Basic CUDA vs wgpu vs CPU parity (small matrix)
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114a_basic_cuda_wgpu_cpu_parity() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!("PARITY-114a: Basic CUDA vs wgpu vs CPU Parity");
    println!("================================================");

    let config = WorkflowConfig::small();
    let WorkflowConfig {
        m,
        k,
        n,
        verbose: _,
    } = config;

    println!(
        "\n  Matrix dimensions: {}x{} * {}x{} = {}x{}",
        m, k, k, n, m, n
    );

    // Generate test data: uniform 1.0 for easy verification
    // Expected result: each output element = k (sum of k 1.0*1.0 products)
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let expected_sum = (m * n * k) as f32;

    println!("  Input A: {} uniform 1.0 values", m * k);
    println!("  Input B: {} uniform 1.0 values", k * n);
    println!("  Expected output element value: {}", k);
    println!("  Expected output sum: {}", expected_sum);

    // CPU reference (ground truth)
    println!("\n--- CPU Reference ---");
    let start = Instant::now();
    let cpu_output = cpu_matmul_reference(&a, &b, m, k, n);
    let cpu_result = MatmulResult::new(cpu_output, start.elapsed(), "CPU");
    cpu_result.print_summary();

    // wgpu via HybridScheduler (force GPU threshold to 0 to use GPU)
    println!("\n--- wgpu (HybridScheduler) ---");
    let mut wgpu_scheduler =
        HybridScheduler::with_threshold(0).expect("Failed to create HybridScheduler");
    println!("  Has GPU: {}", wgpu_scheduler.has_gpu());
    println!(
        "  Should use GPU: {}",
        wgpu_scheduler.should_use_gpu(m, k, n)
    );

    let start = Instant::now();
    let wgpu_output = wgpu_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("wgpu matmul failed");
    let wgpu_result = MatmulResult::new(wgpu_output, start.elapsed(), "wgpu");
    wgpu_result.print_summary();

    // CUDA via CudaScheduler
    println!("\n--- CUDA (CudaScheduler) ---");
    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    println!("  Has CUDA: {}", cuda_scheduler.has_cuda());
    println!(
        "  Device: {}",
        cuda_scheduler
            .device_name()
            .unwrap_or_else(|_| "Unknown".to_string())
    );

    let start = Instant::now();
    let cuda_output = cuda_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    let cuda_result = MatmulResult::new(cuda_output, start.elapsed(), "CUDA");
    cuda_result.print_summary();

    // Parity checks
    println!("\n=== PARITY CHECKS ===");

    let cpu_wgpu_parity = check_parity(&cpu_result, &wgpu_result);
    cpu_wgpu_parity.print_summary();

    let cpu_cuda_parity = check_parity(&cpu_result, &cuda_result);
    cpu_cuda_parity.print_summary();

    let wgpu_cuda_parity = check_parity(&wgpu_result, &cuda_result);
    wgpu_cuda_parity.print_summary();

    // Assertions
    assert!(
        cpu_wgpu_parity.passed,
        "CPU vs wgpu parity failed! Max diff: {}",
        cpu_wgpu_parity.max_diff
    );
    assert!(
        cpu_cuda_parity.passed,
        "CPU vs CUDA parity failed! Max diff: {} (ratio: {:.1}x)",
        cpu_cuda_parity.max_diff, cpu_cuda_parity.sum_ratio
    );
    assert!(
        wgpu_cuda_parity.passed,
        "wgpu vs CUDA parity failed! Max diff: {} (ratio: {:.1}x)",
        wgpu_cuda_parity.max_diff, wgpu_cuda_parity.sum_ratio
    );

    // Verify expected values
    let expected_element = k as f32;
    assert!(
        (cpu_result.output[0] - expected_element).abs() < 0.001,
        "CPU output[0] = {} but expected {}",
        cpu_result.output[0],
        expected_element
    );
    assert!(
        (wgpu_result.output[0] - expected_element).abs() < 0.001,
        "wgpu output[0] = {} but expected {}",
        wgpu_result.output[0],
        expected_element
    );
    assert!(
        (cuda_result.output[0] - expected_element).abs() < 0.001,
        "CUDA output[0] = {} but expected {}",
        cuda_result.output[0],
        expected_element
    );

    println!("\n  PARITY-114a: ALL CHECKS PASSED");
}

/// PARITY-114b: Single tile CUDA parity (k=32, one tile)
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114b_single_tile_cuda_parity() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!("PARITY-114b: Single Tile CUDA Parity (k=32)");
    println!("============================================");

    let config = WorkflowConfig::single_tile();
    let WorkflowConfig { m, k, n, .. } = config;

    println!(
        "\n  Matrix dimensions: {}x{} * {}x{} = {}x{}",
        m, k, k, n, m, n
    );
    println!("  Tile size: 32, Expected tiles: {}", k / 32);

    // Uniform 1.0 data
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    // CPU reference
    let cpu_output = cpu_matmul_reference(&a, &b, m, k, n);
    let cpu_result = MatmulResult::new(cpu_output, std::time::Duration::ZERO, "CPU");

    // wgpu
    let mut wgpu_scheduler =
        HybridScheduler::with_threshold(0).expect("Failed to create HybridScheduler");
    let wgpu_output = wgpu_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("wgpu matmul failed");
    let wgpu_result = MatmulResult::new(wgpu_output, std::time::Duration::ZERO, "wgpu");

    // CUDA
    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let cuda_output = cuda_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    let cuda_result = MatmulResult::new(cuda_output, std::time::Duration::ZERO, "CUDA");

    println!("\n  CPU output[0]: {}", cpu_result.output[0]);
    println!("  wgpu output[0]: {}", wgpu_result.output[0]);
    println!("  CUDA output[0]: {}", cuda_result.output[0]);
    println!("  Expected: {} (k={})", k, k);

    // Check parity
    let parity = check_parity(&cpu_result, &cuda_result);
    parity.print_summary();

    assert!(
        parity.passed,
        "PARITY-114b: Single tile CUDA failed! expected={}, got={}, ratio={:.1}x",
        cpu_result.output[0], cuda_result.output[0], parity.sum_ratio
    );

    println!("\n  PARITY-114b: PASSED");
}

/// PARITY-114c: Multi-tile CUDA parity (k=128, 4 tiles)
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA runtime library access"]
fn test_parity_114c_multi_tile_cuda_parity() {
    use realizar::gpu::CudaScheduler;

    println!("PARITY-114c: Multi-Tile CUDA Parity (k=128)");
    println!("===========================================");

    let m = 4;
    let k = 128; // 4 tiles of 32
    let n = 64;

    println!(
        "\n  Matrix dimensions: {}x{} * {}x{} = {}x{}",
        m, k, k, n, m, n
    );
    println!("  Tile size: 32, Expected tiles: {}", k / 32);

    // Uniform 1.0 data
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    // CPU reference
    let cpu_output = cpu_matmul_reference(&a, &b, m, k, n);
    let cpu_result = MatmulResult::new(cpu_output, std::time::Duration::ZERO, "CPU");

    // CUDA
    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let cuda_output = cuda_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    let cuda_result = MatmulResult::new(cuda_output, std::time::Duration::ZERO, "CUDA");

    println!("\n  CPU output[0]: {}", cpu_result.output[0]);
    println!("  CUDA output[0]: {}", cuda_result.output[0]);
    println!("  Expected: {} (k={})", k, k);

    // Detailed flow analysis
    println!("\n--- Flow Analysis ---");
    println!(
        "  If n_tiles computed as k/tile_size = {}/{} = {}",
        k,
        32,
        k / 32
    );
    println!(
        "  Accumulation should be: {} iterations * 32 products = {}",
        k / 32,
        k
    );

    let ratio = cpu_result.output[0] / cuda_result.output[0];
    if ratio.abs() > 1.5 {
        println!("  WARNING: Ratio {:.1}x suggests tiling bug!", ratio);
        if (ratio - 4.0).abs() < 0.1 {
            println!("  4x ratio suggests only 1 of 4 tiles accumulated");
        } else if (ratio - 8.0).abs() < 0.1 {
            println!("  8x ratio suggests tile loop or accumulator bug");
        }
    }

    // Check parity
    let parity = check_parity(&cpu_result, &cuda_result);
    parity.print_summary();

    assert!(
        parity.passed,
        "PARITY-114c: Multi-tile CUDA failed! expected={}, got={}, ratio={:.1}x",
        cpu_result.output[0], cuda_result.output[0], ratio
    );

    println!("\n  PARITY-114c: PASSED");
}

/// PARITY-114d: Varied data pattern parity test
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114d_varied_data_pattern() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!("PARITY-114d: Varied Data Pattern Parity");
    println!("=======================================");

    let m = 4;
    let k = 64;
    let n = 128;

    println!(
        "\n  Matrix dimensions: {}x{} * {}x{} = {}x{}",
        m, k, k, n, m, n
    );

    // Generate patterned data
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 10) as f32) / 10.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) / 7.0).collect();

    println!("  A pattern: cyclic 0-9 scaled to [0, 0.9]");
    println!("  B pattern: cyclic 0-6 scaled to [0, 0.86]");

    // CPU reference
    let cpu_output = cpu_matmul_reference(&a, &b, m, k, n);
    let cpu_result = MatmulResult::new(cpu_output, std::time::Duration::ZERO, "CPU");

    // wgpu
    let mut wgpu_scheduler =
        HybridScheduler::with_threshold(0).expect("Failed to create HybridScheduler");
    let wgpu_output = wgpu_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("wgpu matmul failed");
    let wgpu_result = MatmulResult::new(wgpu_output, std::time::Duration::ZERO, "wgpu");

    // CUDA
    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let cuda_output = cuda_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    let cuda_result = MatmulResult::new(cuda_output, std::time::Duration::ZERO, "CUDA");

    println!("\n  Results:");
    cpu_result.print_summary();
    wgpu_result.print_summary();
    cuda_result.print_summary();

    // Check all parities
    let cpu_cuda = check_parity(&cpu_result, &cuda_result);
    cpu_cuda.print_summary();

    assert!(
        cpu_cuda.passed,
        "PARITY-114d: Varied data pattern failed! Max diff: {}",
        cpu_cuda.max_diff
    );

    println!("\n  PARITY-114d: PASSED");
}

/// PARITY-114e: Large matrix stress test
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114e_large_matrix_stress() {
    use realizar::gpu::CudaScheduler;

    println!("PARITY-114e: Large Matrix Stress Test");
    println!("=====================================");

    let m = 64;
    let k = 512;
    let n = 1024;

    println!(
        "\n  Matrix dimensions: {}x{} * {}x{} = {}x{}",
        m, k, k, n, m, n
    );
    println!("  Total elements: A={}, B={}, C={}", m * k, k * n, m * n);

    // Random-ish data via simple pattern
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 11 + 17) % 100) as f32 / 100.0)
        .collect();

    // CPU reference
    let start = Instant::now();
    let cpu_output = cpu_matmul_reference(&a, &b, m, k, n);
    let cpu_result = MatmulResult::new(cpu_output, start.elapsed(), "CPU");

    // CUDA
    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let start = Instant::now();
    let cuda_output = cuda_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    let cuda_result = MatmulResult::new(cuda_output, start.elapsed(), "CUDA");

    println!("\n  Performance:");
    println!("    CPU:  {:?}", cpu_result.duration);
    println!("    CUDA: {:?}", cuda_result.duration);

    let parity = check_parity(&cpu_result, &cuda_result);
    parity.print_summary();

    assert!(
        parity.passed,
        "PARITY-114e: Large matrix stress test failed! Max diff: {}",
        parity.max_diff
    );

    println!("\n  PARITY-114e: PASSED");
}

/// PARITY-114f: M=1 edge case (single row)
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA runtime library access"]
fn test_parity_114f_single_row_m1() {
    use realizar::gpu::CudaScheduler;

    println!("PARITY-114f: Single Row (M=1) Edge Case");
    println!("=======================================");

    let m = 1;
    let k = 64;
    let n = 128;

    println!(
        "\n  Matrix dimensions: {}x{} * {}x{} = {}x{}",
        m, k, k, n, m, n
    );
    println!("  Note: This is vector-matrix multiply (common in token generation)");

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    // CPU reference
    let cpu_output = cpu_matmul_reference(&a, &b, m, k, n);
    let cpu_result = MatmulResult::new(cpu_output, std::time::Duration::ZERO, "CPU");

    // CUDA (CudaScheduler always uses CUDA, even for m=1)
    let mut cuda_scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");
    let cuda_output = cuda_scheduler
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    let cuda_result = MatmulResult::new(cuda_output, std::time::Duration::ZERO, "CUDA");

    println!(
        "\n  CPU output[0]: {} (expected: {})",
        cpu_result.output[0], k
    );
    println!("  CUDA output[0]: {}", cuda_result.output[0]);

    let parity = check_parity(&cpu_result, &cuda_result);
    parity.print_summary();

    assert!(
        parity.passed,
        "PARITY-114f: M=1 edge case failed! CPU={}, CUDA={}",
        cpu_result.output[0], cuda_result.output[0]
    );

    println!("\n  PARITY-114f: PASSED");
}

// ============================================================================
// Workflow Flow Visualization
// ============================================================================

/// PARITY-114-flow: Detailed flow visualization for debugging
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114_flow_visualization() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!("PARITY-114-flow: Data Flow Visualization");
    println!("========================================");
    println!();
    println!("This test visualizes the data flow through schedulers");
    println!("to help diagnose parity issues.");
    println!();

    let m = 2;
    let k = 64;
    let n = 4;

    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];

    println!("Input Flow:");
    println!("  A: {}x{} matrix (all 1.0)", m, k);
    println!("  B: {}x{} matrix (all 1.0)", k, n);
    println!("  Expected C[i,j] = sum(A[i,:] * B[:,j]) = {}", k);
    println!();

    // CPU
    let cpu_out = cpu_matmul_reference(&a, &b, m, k, n);
    println!("CPU Flow:");
    println!("  Output shape: {}x{}", m, n);
    println!("  C = {:?}", cpu_out);
    println!();

    // wgpu
    let mut wgpu = HybridScheduler::with_threshold(0).expect("test");
    let wgpu_out = wgpu.matmul(&a, &b, m, k, n).expect("test");
    println!("wgpu Flow:");
    println!("  Backend: {}", if wgpu.has_gpu() { "GPU" } else { "CPU" });
    println!("  C = {:?}", wgpu_out);
    println!();

    // CUDA
    let mut cuda = CudaScheduler::new().expect("test");
    let cuda_out = cuda.matmul(&a, &b, m, k, n).expect("test");
    println!("CUDA Flow:");
    println!(
        "  Device: {}",
        cuda.device_name().unwrap_or_else(|_| "Unknown".to_string())
    );
    println!("  C = {:?}", cuda_out);
    println!();

    // Comparison
    println!("Comparison:");
    for i in 0..m * n {
        let cpu_v = cpu_out[i];
        let wgpu_v = wgpu_out[i];
        let cuda_v = cuda_out[i];
        let status = if (cpu_v - cuda_v).abs() < EPSILON {
            "OK"
        } else {
            "MISMATCH"
        };
        println!(
            "  C[{}]: CPU={:.1}, wgpu={:.1}, CUDA={:.1} [{}]",
            i, cpu_v, wgpu_v, cuda_v, status
        );
    }
    println!();

    // Assertions
    assert!(
        (cpu_out[0] - cuda_out[0]).abs() < EPSILON,
        "Flow visualization found parity error: CPU={}, CUDA={}",
        cpu_out[0],
        cuda_out[0]
    );

    println!("PARITY-114-flow: All flows match!");
}

// ============================================================================
// Summary Test
// ============================================================================

/// PARITY-114: Test suite summary
#[test]
fn test_parity_114_summary() {
    println!("=== PARITY-114: CUDA vs wgpu Parity Test Suite ===");
    println!();
    println!("Tests:");
    println!("  114a: Basic CUDA/wgpu/CPU parity (small matrix)");
    println!("  114b: Single tile parity (k=32)");
    println!("  114c: Multi-tile parity (k=128, 4 tiles)");
    println!("  114d: Varied data patterns");
    println!("  114e: Large matrix stress test");
    println!("  114f: M=1 edge case (vector-matrix)");
    println!("  114-flow: Flow visualization for debugging");
    println!("  114-tui-sim: TUI simulation (probar-style watch flow)");
    println!();
    println!("Run with:");
    println!("  cargo test --test gpu_parity_workflow --features cuda -- --nocapture");
    println!();
    println!("This test suite catches bugs like:");
    println!("  - Tile loop accumulation errors");
    println!("  - Grid/block dimension misconfiguration");
    println!("  - Shared memory addressing issues");
    println!("  - n_tiles calculation errors");
}

// ============================================================================
// TUI Simulation Module (Probar-style Watch Flow)
// ============================================================================

/// TUI frame representation for workflow visualization
#[derive(Debug, Clone)]
pub struct TuiFrame {
    /// Frame lines (each line is a row of the TUI)
    lines: Vec<String>,
    /// Frame width
    width: usize,
    /// Frame timestamp
    timestamp_ms: u64,
}

impl TuiFrame {
    fn new(width: usize, height: usize) -> Self {
        Self {
            lines: vec![String::new(); height],
            width,
            timestamp_ms: 0,
        }
    }

    fn set_line(&mut self, row: usize, content: &str) {
        if row < self.lines.len() {
            self.lines[row] = content.to_string();
        }
    }

    fn render(&self) -> String {
        use std::fmt::Write;
        let border = "═".repeat(self.width);
        let mut output = format!("╔{}╗\n", border);
        for line in &self.lines {
            let padded = format!("{:width$}", line, width = self.width);
            let _ = writeln!(output, "║{}║", padded);
        }
        let _ = write!(output, "╚{}╝", border);
        output
    }
}

/// Simulation step for watching matmul flow
#[derive(Debug, Clone)]
pub struct SimulationStep {
    /// Step name
    pub name: String,
    /// Step description
    pub description: String,
    /// Values at this step
    pub values: Vec<f32>,
    /// Expected values (for comparison)
    pub expected: Option<Vec<f32>>,
    /// Status: pass/fail/pending
    pub status: StepStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepStatus {
    Pending,
    Running,
    Pass,
    Fail,
}

impl StepStatus {
    fn symbol(&self) -> &str {
        match self {
            StepStatus::Pending => "○",
            StepStatus::Running => "◐",
            StepStatus::Pass => "●",
            StepStatus::Fail => "✗",
        }
    }
}

/// Matmul workflow simulator with TUI visualization
pub struct MatmulSimulator {
    /// Simulation steps
    steps: Vec<SimulationStep>,
    /// Current step index
    current_step: usize,
    /// TUI width
    width: usize,
    /// Frame history
    frames: Vec<TuiFrame>,
}

impl Default for MatmulSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl MatmulSimulator {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            current_step: 0,
            width: 70,
            frames: Vec::new(),
        }
    }

    pub fn add_step(&mut self, name: &str, description: &str) -> usize {
        let step = SimulationStep {
            name: name.to_string(),
            description: description.to_string(),
            values: Vec::new(),
            expected: None,
            status: StepStatus::Pending,
        };
        self.steps.push(step);
        self.steps.len() - 1
    }

    pub fn start_step(&mut self, idx: usize) {
        if idx < self.steps.len() {
            self.steps[idx].status = StepStatus::Running;
            self.current_step = idx;
            self.capture_frame();
        }
    }

    pub fn complete_step(&mut self, idx: usize, values: Vec<f32>, expected: Option<Vec<f32>>) {
        if idx < self.steps.len() {
            let status = if let Some(ref exp) = expected {
                let max_diff = values
                    .iter()
                    .zip(exp.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                if max_diff < EPSILON {
                    StepStatus::Pass
                } else {
                    StepStatus::Fail
                }
            } else {
                StepStatus::Pass
            };
            self.steps[idx].values = values;
            self.steps[idx].expected = expected;
            self.steps[idx].status = status;
            self.capture_frame();
        }
    }

    fn capture_frame(&mut self) {
        let mut frame = TuiFrame::new(self.width, self.steps.len() + 6);

        // Header
        frame.set_line(0, "  CUDA vs wgpu Matmul Parity Simulation");
        frame.set_line(1, &"─".repeat(self.width));

        // Steps
        for (i, step) in self.steps.iter().enumerate() {
            let status_str = format!(
                "  {} {} - {}",
                step.status.symbol(),
                step.name,
                step.description
            );
            frame.set_line(i + 2, &status_str);

            // Show values for completed steps
            if step.status == StepStatus::Pass || step.status == StepStatus::Fail {
                let _val_str = if step.values.len() <= 4 {
                    format!("      Values: {:?}", step.values)
                } else {
                    format!(
                        "      Values: [{:.2}, {:.2}, ... {} total]",
                        step.values[0],
                        step.values[1],
                        step.values.len()
                    )
                };
                // We'd need more lines for this - simplified for now
            }
        }

        // Footer
        let footer_line = self.steps.len() + 3;
        frame.set_line(footer_line, &"─".repeat(self.width));

        let pass_count = self
            .steps
            .iter()
            .filter(|s| s.status == StepStatus::Pass)
            .count();
        let fail_count = self
            .steps
            .iter()
            .filter(|s| s.status == StepStatus::Fail)
            .count();
        frame.set_line(
            footer_line + 1,
            &format!(
                "  Progress: {} pass, {} fail, {} pending",
                pass_count,
                fail_count,
                self.steps.len() - pass_count - fail_count
            ),
        );

        frame.timestamp_ms = self.frames.len() as u64 * 100; // 100ms per frame
        self.frames.push(frame);
    }

    pub fn render_final(&self) -> String {
        if let Some(frame) = self.frames.last() {
            frame.render()
        } else {
            String::new()
        }
    }

    pub fn all_passed(&self) -> bool {
        self.steps.iter().all(|s| s.status == StepStatus::Pass)
    }

    pub fn get_failures(&self) -> Vec<&SimulationStep> {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Fail)
            .collect()
    }
}

/// PARITY-114-tui-sim: TUI simulation for watching matmul flow
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114_tui_simulation() {
    use realizar::gpu::{CudaScheduler, HybridScheduler};

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  PARITY-114 TUI SIMULATION: Watch Matmul Flow Through Schedulers     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut sim = MatmulSimulator::new();

    // Define simulation steps
    let step_init = sim.add_step("INIT", "Initialize test matrices");
    let step_cpu = sim.add_step("CPU", "Compute reference (CPU matmul)");
    let step_wgpu = sim.add_step("WGPU", "Execute via HybridScheduler (wgpu)");
    let step_cuda = sim.add_step("CUDA", "Execute via CudaScheduler (CUDA)");
    let step_parity_wgpu = sim.add_step("CHECK-WGPU", "Verify wgpu matches CPU");
    let step_parity_cuda = sim.add_step("CHECK-CUDA", "Verify CUDA matches CPU");

    // Test configuration
    let m = 4;
    let k = 64;
    let n = 8;

    // Step 1: Initialize
    sim.start_step(step_init);
    println!(
        "  ◐ Initializing matrices: A[{}x{}] × B[{}x{}] = C[{}x{}]",
        m, k, k, n, m, n
    );
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    sim.complete_step(step_init, vec![m as f32, k as f32, n as f32], None);
    println!("  ● Matrices initialized (all 1.0 values)");
    println!();

    // Step 2: CPU reference
    sim.start_step(step_cpu);
    println!("  ◐ Computing CPU reference...");
    let cpu_out = cpu_matmul_reference(&a, &b, m, k, n);
    let expected_val = k as f32; // Each output should be k (sum of k 1.0*1.0 products)
    sim.complete_step(step_cpu, cpu_out.clone(), None);
    println!(
        "  ● CPU result: C[0,0] = {} (expected: {})",
        cpu_out[0], expected_val
    );
    println!("    Full output: {:?}", &cpu_out[..cpu_out.len().min(8)]);
    println!();

    // Step 3: wgpu execution
    sim.start_step(step_wgpu);
    println!("  ◐ Executing via HybridScheduler (wgpu backend)...");
    let mut wgpu_sched =
        HybridScheduler::with_threshold(0).expect("Failed to create HybridScheduler");
    let wgpu_out = wgpu_sched
        .matmul(&a, &b, m, k, n)
        .expect("wgpu matmul failed");
    sim.complete_step(step_wgpu, wgpu_out.clone(), Some(cpu_out.clone()));
    println!("  ● wgpu result: C[0,0] = {}", wgpu_out[0]);
    println!("    Full output: {:?}", &wgpu_out[..wgpu_out.len().min(8)]);
    println!();

    // Step 4: CUDA execution
    sim.start_step(step_cuda);
    println!("  ◐ Executing via CudaScheduler (CUDA backend)...");
    let mut cuda_sched = CudaScheduler::new().expect("Failed to create CudaScheduler");
    println!(
        "    Device: {}",
        cuda_sched
            .device_name()
            .unwrap_or_else(|_| "Unknown".to_string())
    );
    let cuda_out = cuda_sched
        .matmul(&a, &b, m, k, n)
        .expect("CUDA matmul failed");
    sim.complete_step(step_cuda, cuda_out.clone(), Some(cpu_out.clone()));
    println!(
        "  {} CUDA result: C[0,0] = {}",
        if (cuda_out[0] - expected_val).abs() < EPSILON {
            "●"
        } else {
            "✗"
        },
        cuda_out[0]
    );
    println!("    Full output: {:?}", &cuda_out[..cuda_out.len().min(8)]);
    println!();

    // Step 5: wgpu parity check
    sim.start_step(step_parity_wgpu);
    let wgpu_diff: f32 = cpu_out
        .iter()
        .zip(wgpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let wgpu_pass = wgpu_diff < EPSILON;
    sim.complete_step(step_parity_wgpu, vec![wgpu_diff], Some(vec![0.0]));
    println!(
        "  {} wgpu parity: max_diff = {} (tolerance: {})",
        if wgpu_pass { "●" } else { "✗" },
        wgpu_diff,
        EPSILON
    );

    // Step 6: CUDA parity check
    sim.start_step(step_parity_cuda);
    let cuda_diff: f32 = cpu_out
        .iter()
        .zip(cuda_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cuda_pass = cuda_diff < EPSILON;
    sim.complete_step(step_parity_cuda, vec![cuda_diff], Some(vec![0.0]));
    println!(
        "  {} CUDA parity: max_diff = {} (tolerance: {})",
        if cuda_pass { "●" } else { "✗" },
        cuda_diff,
        EPSILON
    );

    // Render final TUI frame
    println!();
    println!("{}", sim.render_final());
    println!();

    // Analysis section (if CUDA failed)
    if !cuda_pass {
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║  PARITY FAILURE ANALYSIS                                             ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");
        println!();

        let ratio = if cuda_out[0].abs() > 1e-10 {
            cpu_out[0] / cuda_out[0]
        } else {
            f32::INFINITY
        };

        println!("  Expected: {} (k = {})", expected_val, k);
        println!("  Got:      {} (CUDA)", cuda_out[0]);
        println!("  Ratio:    {:.1}x (CPU/CUDA)", ratio);
        println!();

        // Diagnose based on ratio
        if (ratio - 8.0).abs() < 0.5 {
            println!("  DIAGNOSIS: 8x ratio suggests accumulator or tile loop bug");
            println!("    - Check if tile accumulation is being overwritten instead of summed");
            println!("    - Check if inner loop iterations are correct");
            println!("    - Check shared memory addressing");
        } else if (ratio - 4.0).abs() < 0.5 {
            println!("  DIAGNOSIS: 4x ratio suggests only 1 of 4 tiles accumulated");
        } else if (ratio - 2.0).abs() < 0.5 {
            println!("  DIAGNOSIS: 2x ratio suggests only half tiles accumulated");
        } else {
            println!("  DIAGNOSIS: Unusual ratio - needs deeper investigation");
        }
        println!();

        // Show tile analysis
        let tile_size = 32;
        let n_tiles = k.div_ceil(tile_size);
        println!("  Tile Analysis:");
        println!(
            "    k = {}, tile_size = {}, n_tiles = {}",
            k, tile_size, n_tiles
        );
        println!(
            "    Expected: {} tiles × {} iterations = {} accumulations",
            n_tiles,
            tile_size,
            n_tiles * tile_size
        );
        println!(
            "    Actual result suggests ~{} accumulations",
            cuda_out[0] as usize
        );
    }

    // Final assertion
    println!();
    if sim.all_passed() {
        println!("═══════════════════════════════════════════════════════════════════════");
        println!("  TUI SIMULATION: ALL STEPS PASSED ●");
        println!("═══════════════════════════════════════════════════════════════════════");
    } else {
        println!("═══════════════════════════════════════════════════════════════════════");
        println!("  TUI SIMULATION: FAILURES DETECTED ✗");
        for failure in sim.get_failures() {
            println!("    - {} ({})", failure.name, failure.description);
        }
        println!("═══════════════════════════════════════════════════════════════════════");
    }

    // Assert for CI
    assert!(wgpu_pass, "wgpu parity check failed");
    assert!(
        cuda_pass,
        "CUDA parity check failed: expected {}, got {} (ratio: {:.1}x)",
        expected_val,
        cuda_out[0],
        expected_val / cuda_out[0]
    );
}

/// PARITY-114-multi-step: Multi-step simulation showing data flow through CudaScheduler
#[test]
#[cfg(feature = "cuda")]
fn test_parity_114_cuda_scheduler_flow() {
    use realizar::gpu::CudaScheduler;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  CudaScheduler Data Flow Simulation                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create scheduler
    let mut scheduler = CudaScheduler::new().expect("Failed to create CudaScheduler");

    println!("Step 1: CudaScheduler Initialization");
    println!(
        "  ├─ Device: {}",
        scheduler
            .device_name()
            .unwrap_or_else(|_| "Unknown".to_string())
    );
    println!("  ├─ Has CUDA: {}", scheduler.has_cuda());
    println!(
        "  └─ Uses CUDA for m=4, k=64, n=8: {}",
        scheduler.uses_cuda_for(4, 64, 8)
    );
    println!();

    // Simulate multiple matmul operations to check for state issues
    println!("Step 2: Sequential Matmul Operations (checking for state bugs)");
    println!("  ┌─────────────────────────────────────────────────────────────────┐");

    let test_cases = [
        (4, 64, 8, "small"),
        (8, 128, 16, "medium"),
        (4, 64, 8, "small again (state check)"),
    ];

    let mut all_passed = true;
    for (i, (m, k, n, label)) in test_cases.iter().enumerate() {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let expected = *k as f32;

        let result = scheduler.matmul(&a, &b, *m, *k, *n).expect("matmul failed");
        let actual = result[0];
        let passed = (actual - expected).abs() < EPSILON;

        let status = if passed { "●" } else { "✗" };
        println!(
            "  │ Op {}: {} - {}x{}x{} ({})",
            i + 1,
            status,
            m,
            k,
            n,
            label
        );
        println!("  │      Expected: {}, Got: {}", expected, actual);

        if !passed {
            all_passed = false;
            println!("  │      ERROR: Ratio = {:.1}x", expected / actual);
        }
    }

    println!("  └─────────────────────────────────────────────────────────────────┘");
    println!();

    // Check for state persistence issues
    println!("Step 3: State Persistence Check");
    let a1 = vec![2.0f32; 4 * 64];
    let b1 = vec![1.0f32; 64 * 8];
    let r1 = scheduler.matmul(&a1, &b1, 4, 64, 8).expect("matmul failed");

    let a2 = vec![1.0f32; 4 * 64];
    let b2 = vec![1.0f32; 64 * 8];
    let r2 = scheduler.matmul(&a2, &b2, 4, 64, 8).expect("matmul failed");

    println!("  ├─ Input A=2.0, expected C[0]=128, got {}", r1[0]);
    println!("  ├─ Input A=1.0, expected C[0]=64, got {}", r2[0]);

    let state_ok = (r1[0] - 128.0).abs() < EPSILON && (r2[0] - 64.0).abs() < EPSILON;
    println!(
        "  └─ State isolation: {}",
        if state_ok {
            "● OK"
        } else {
            "✗ FAIL (state leak)"
        }
    );

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    if all_passed && state_ok {
        println!("  CudaScheduler Flow: ALL CHECKS PASSED ●");
    } else {
        println!("  CudaScheduler Flow: FAILURES DETECTED ✗");
        if !all_passed {
            println!("    - Matmul parity failures");
        }
        if !state_ok {
            println!("    - State isolation failure (possible buffer reuse bug)");
        }
    }
    println!("═══════════════════════════════════════════════════════════════════════");

    assert!(all_passed && state_ok, "CudaScheduler flow check failed");
}
