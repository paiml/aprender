//! Roofline model implementation per Williams, Waterman & Patterson (2009) [4].
//! Supports hierarchical GPU roofline per Yang et al. (2020) [13].
//! Uses Empirical Roofline Toolkit (ERT) methodology [6].

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Floating-point precision levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Tf32,
    Int8,
    Bf16,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::Fp32 => write!(f, "FP32"),
            Precision::Fp16 => write!(f, "FP16 Tensor"),
            Precision::Tf32 => write!(f, "TF32 Tensor"),
            Precision::Int8 => write!(f, "INT8 Tensor"),
            Precision::Bf16 => write!(f, "BF16"),
        }
    }
}

/// Memory hierarchy levels for hierarchical roofline [13].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLevel {
    L1Cache,
    L2Cache,
    Dram,
    Pcie,
}

impl std::fmt::Display for MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLevel::L1Cache => write!(f, "L1 Cache"),
            MemoryLevel::L2Cache => write!(f, "L2 Cache"),
            MemoryLevel::Dram => write!(f, "DRAM"),
            MemoryLevel::Pcie => write!(f, "PCIe"),
        }
    }
}

/// Whether a kernel is compute-bound or memory-bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Bound {
    /// Below ridge point: memory bandwidth is the bottleneck.
    Memory { bandwidth_utilization: f64 },
    /// Above ridge point: compute throughput is the bottleneck.
    Compute { compute_utilization: f64 },
}

/// Roofline model for a specific hardware target.
/// Implements the Empirical Roofline Toolkit (ERT) methodology [6].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineModel {
    /// Hardware target name (e.g., "RTX 4090", "AMD EPYC AVX2")
    pub target: String,
    /// Peak compute throughput (FLOP/s) per precision
    pub peak_compute: HashMap<Precision, f64>,
    /// Peak memory bandwidth (bytes/s) per memory level
    pub peak_bandwidth: HashMap<MemoryLevel, f64>,
}

impl RooflineModel {
    /// Compute the ridge point for a given precision and memory level.
    /// Ridge = peak_compute / peak_bandwidth (FLOP/byte).
    /// This is the arithmetic intensity where the kernel transitions
    /// from memory-bound to compute-bound.
    pub fn ridge_point(&self, precision: Precision, mem_level: MemoryLevel) -> Option<f64> {
        let compute = self.peak_compute.get(&precision)?;
        let bandwidth = self.peak_bandwidth.get(&mem_level)?;
        if *bandwidth <= 0.0 {
            return None;
        }
        Some(compute / bandwidth)
    }

    /// Compute the theoretical peak throughput at a given arithmetic intensity.
    /// throughput = min(peak_compute, AI * peak_bandwidth)
    pub fn theoretical_peak(
        &self,
        arithmetic_intensity: f64,
        precision: Precision,
        mem_level: MemoryLevel,
    ) -> Option<f64> {
        let compute = self.peak_compute.get(&precision)?;
        let bandwidth = self.peak_bandwidth.get(&mem_level)?;
        Some(compute.min(arithmetic_intensity * bandwidth))
    }

    /// Classify a kernel as compute-bound or memory-bound.
    pub fn classify(
        &self,
        arithmetic_intensity: f64,
        achieved_throughput: f64,
        precision: Precision,
        mem_level: MemoryLevel,
    ) -> Option<KernelRooflinePoint> {
        let ridge = self.ridge_point(precision, mem_level)?;
        let peak = self.theoretical_peak(arithmetic_intensity, precision, mem_level)?;
        let peak_compute = *self.peak_compute.get(&precision)?;

        let bound = if arithmetic_intensity < ridge {
            Bound::Memory { bandwidth_utilization: achieved_throughput / peak * 100.0 }
        } else {
            Bound::Compute { compute_utilization: achieved_throughput / peak_compute * 100.0 }
        };

        let efficiency = if peak > 0.0 { achieved_throughput / peak * 100.0 } else { 0.0 };

        let distance_to_ridge =
            if arithmetic_intensity > 0.0 { ridge / arithmetic_intensity } else { f64::INFINITY };

        Some(KernelRooflinePoint {
            arithmetic_intensity,
            achieved_throughput,
            peak_throughput: peak,
            efficiency,
            bound,
            distance_to_ridge,
        })
    }

    /// Create the RTX 4090 roofline model with spec values.
    pub fn rtx_4090() -> Self {
        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, 82.6e12); // 82.6 TFLOP/s
        peak_compute.insert(Precision::Fp16, 330.0e12); // 330 TFLOP/s (Tensor)
        peak_compute.insert(Precision::Tf32, 165.0e12); // 165 TFLOP/s (Tensor)
        peak_compute.insert(Precision::Int8, 660.0e12); // 660 TOP/s (Tensor)

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::L1Cache, 19.0e12); // ~19 TB/s
        peak_bandwidth.insert(MemoryLevel::L2Cache, 5.3e12); // ~5.3 TB/s
        peak_bandwidth.insert(MemoryLevel::Dram, 1008.0e9); // 1008 GB/s
        peak_bandwidth.insert(MemoryLevel::Pcie, 32.0e9); // 32 GB/s PCIe 4.0 x16

        RooflineModel {
            target: "NVIDIA GeForce RTX 4090 (SM 8.9)".to_string(),
            peak_compute,
            peak_bandwidth,
        }
    }

    /// Create a CPU AVX2+FMA roofline model.
    /// Assumes dual 256-bit FMA units (e.g., AMD EPYC / Intel Skylake).
    pub fn cpu_avx2(freq_ghz: f64, cores: usize, mem_bandwidth_gbps: f64) -> Self {
        // FP32: 2 FMA units * 8 floats * 2 (FMA = mul + add) * freq * cores
        let fp32_peak = 2.0 * 8.0 * 2.0 * freq_ghz * 1e9 * cores as f64;

        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, fp32_peak);

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::Dram, mem_bandwidth_gbps * 1e9);

        RooflineModel {
            target: format!("CPU AVX2+FMA ({cores} cores @ {freq_ghz} GHz)"),
            peak_compute,
            peak_bandwidth,
        }
    }

    /// Create a CPU AVX-512 roofline model.
    /// AVX-512: 2 FMA units * 16 floats * 2 (FMA) * freq * cores.
    pub fn cpu_avx512(freq_ghz: f64, cores: usize, mem_bandwidth_gbps: f64) -> Self {
        let fp32_peak = 2.0 * 16.0 * 2.0 * freq_ghz * 1e9 * cores as f64;

        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, fp32_peak);

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::Dram, mem_bandwidth_gbps * 1e9);

        RooflineModel {
            target: format!("CPU AVX-512+FMA ({cores} cores @ {freq_ghz} GHz)"),
            peak_compute,
            peak_bandwidth,
        }
    }

    /// Create an ARM NEON roofline model.
    /// NEON: 2 FMA units * 4 floats * 2 (FMA) * freq * cores (typical A76/A78).
    pub fn cpu_neon(freq_ghz: f64, cores: usize, mem_bandwidth_gbps: f64) -> Self {
        let fp32_peak = 2.0 * 4.0 * 2.0 * freq_ghz * 1e9 * cores as f64;

        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, fp32_peak);

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::Dram, mem_bandwidth_gbps * 1e9);

        RooflineModel {
            target: format!("CPU NEON ({cores} cores @ {freq_ghz} GHz)"),
            peak_compute,
            peak_bandwidth,
        }
    }
}

/// A kernel's position on the roofline chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRooflinePoint {
    /// FLOPs per byte transferred
    pub arithmetic_intensity: f64,
    /// Achieved throughput (FLOP/s)
    pub achieved_throughput: f64,
    /// Roofline ceiling throughput (FLOP/s)
    pub peak_throughput: f64,
    /// Achieved / peak percentage
    pub efficiency: f64,
    /// Compute or memory bound classification
    pub bound: Bound,
    /// Ridge point / arithmetic_intensity (>1 = memory-bound)
    pub distance_to_ridge: f64,
}

/// Empirical roofline measurement results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalResult {
    /// Measured DRAM bandwidth (bytes/s) via STREAM-like test
    pub measured_bandwidth_bps: f64,
    /// Measured peak FLOPS via tight FMA loop
    pub measured_peak_flops: f64,
    /// Empirical ridge point (FLOP/byte)
    pub measured_ridge_point: f64,
    /// Theoretical vs empirical bandwidth ratio
    pub bandwidth_efficiency: f64,
    /// Theoretical vs empirical compute ratio
    pub compute_efficiency: f64,
}

/// Measure actual DRAM bandwidth via STREAM-like copy test.
/// Allocates 64 MB arrays, performs timed copy, returns bytes/s.
fn measure_bandwidth() -> f64 {
    const N: usize = 16 * 1024 * 1024; // 16M f32 = 64 MB
    const ITERS: usize = 10;

    let a: Vec<f32> = vec![1.0f32; N];
    let mut b: Vec<f32> = vec![0.0f32; N];

    // Warmup
    b.copy_from_slice(&a);

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        b.copy_from_slice(&a);
        // Prevent dead-code elimination
        std::hint::black_box(&b);
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Each iteration reads N f32 and writes N f32 = 2 * N * 4 bytes
    let bytes = 2.0 * N as f64 * 4.0 * ITERS as f64;
    bytes / elapsed
}

/// Measure actual DRAM bandwidth via STREAM-like triad: a[i] = b[i] + s * c[i].
/// More representative than copy — exercises FMA pipeline and memory subsystem together.
fn measure_bandwidth_triad() -> f64 {
    const N: usize = 16 * 1024 * 1024; // 16M f32 = 64 MB per array
    const ITERS: usize = 10;

    let b: Vec<f32> = vec![1.0f32; N];
    let c: Vec<f32> = vec![2.0f32; N];
    let mut a: Vec<f32> = vec![0.0f32; N];
    let s = 3.0f32;

    // Warmup
    for i in 0..N {
        a[i] = b[i] + s * c[i];
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        for i in 0..N {
            a[i] = b[i] + s * c[i];
        }
        std::hint::black_box(&a);
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Triad: reads 2 arrays, writes 1 = 3 * N * 4 bytes per iteration
    let bytes = 3.0 * N as f64 * 4.0 * ITERS as f64;
    bytes / elapsed
}

/// Measure peak single-core FP32 FLOPS using AVX-512/AVX2 FMA intrinsics.
/// Falls back to scalar if SIMD not available.
fn measure_peak_flops_single_core() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f detected above
            return unsafe { measure_peak_flops_avx512() };
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: avx2+fma detected above
            return unsafe { measure_peak_flops_avx2() };
        }
    }
    measure_peak_flops_scalar()
}

/// Scalar fallback for peak FLOPS measurement.
fn measure_peak_flops_scalar() -> f64 {
    const ITERS: u64 = 500_000_000;
    let mut a0 = 1.0f32;
    let mut a1 = 1.1f32;
    let mut a2 = 1.2f32;
    let mut a3 = 1.3f32;
    let m = 1.0000001f32;
    let add = 0.0000001f32;

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        a0 = a0.mul_add(m, add);
        a1 = a1.mul_add(m, add);
        a2 = a2.mul_add(m, add);
        a3 = a3.mul_add(m, add);
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(a0 + a1 + a2 + a3);
    // 4 FMA = 8 FLOP per iteration
    ITERS as f64 * 8.0 / elapsed
}

/// AVX2 FMA peak: 2 FMA units * 8 FP32/vec * 2 ops/FMA = 32 FLOP/cycle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn measure_peak_flops_avx2() -> f64 {
    use std::arch::x86_64::*;

    const ITERS: u64 = 100_000_000;

    // 10 independent accumulators to saturate both FMA ports
    let mut v0 = _mm256_set1_ps(1.0);
    let mut v1 = _mm256_set1_ps(1.1);
    let mut v2 = _mm256_set1_ps(1.2);
    let mut v3 = _mm256_set1_ps(1.3);
    let mut v4 = _mm256_set1_ps(1.4);
    let mut v5 = _mm256_set1_ps(1.5);
    let mut v6 = _mm256_set1_ps(1.6);
    let mut v7 = _mm256_set1_ps(1.7);
    let mut v8 = _mm256_set1_ps(1.8);
    let mut v9 = _mm256_set1_ps(1.9);
    let mul = _mm256_set1_ps(1.0000001);
    let add = _mm256_set1_ps(0.0000001);

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        // 10 vfmadd231ps: each = 8 FMA = 16 FLOP → 160 FLOP/iter
        v0 = _mm256_fmadd_ps(v0, mul, add);
        v1 = _mm256_fmadd_ps(v1, mul, add);
        v2 = _mm256_fmadd_ps(v2, mul, add);
        v3 = _mm256_fmadd_ps(v3, mul, add);
        v4 = _mm256_fmadd_ps(v4, mul, add);
        v5 = _mm256_fmadd_ps(v5, mul, add);
        v6 = _mm256_fmadd_ps(v6, mul, add);
        v7 = _mm256_fmadd_ps(v7, mul, add);
        v8 = _mm256_fmadd_ps(v8, mul, add);
        v9 = _mm256_fmadd_ps(v9, mul, add);
    }
    let elapsed = start.elapsed().as_secs_f64();
    // Prevent dead-code elimination
    let sum = _mm256_add_ps(v0, v1);
    let sum = _mm256_add_ps(sum, v2);
    let sum = _mm256_add_ps(sum, v3);
    let sum = _mm256_add_ps(sum, v4);
    let sum = _mm256_add_ps(sum, v5);
    let sum = _mm256_add_ps(sum, v6);
    let sum = _mm256_add_ps(sum, v7);
    let sum = _mm256_add_ps(sum, v8);
    let sum = _mm256_add_ps(sum, v9);
    std::hint::black_box(sum);

    // 10 FMAs * 8 elements * 2 ops(mul+add) = 160 FLOP per iteration
    ITERS as f64 * 160.0 / elapsed
}

/// AVX-512 FMA peak: 2 FMA units * 16 FP32/vec * 2 ops/FMA = 64 FLOP/cycle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn measure_peak_flops_avx512() -> f64 {
    use std::arch::x86_64::*;

    const ITERS: u64 = 100_000_000;

    // 10 independent accumulators to saturate both FMA512 ports
    let mut v0 = _mm512_set1_ps(1.0);
    let mut v1 = _mm512_set1_ps(1.1);
    let mut v2 = _mm512_set1_ps(1.2);
    let mut v3 = _mm512_set1_ps(1.3);
    let mut v4 = _mm512_set1_ps(1.4);
    let mut v5 = _mm512_set1_ps(1.5);
    let mut v6 = _mm512_set1_ps(1.6);
    let mut v7 = _mm512_set1_ps(1.7);
    let mut v8 = _mm512_set1_ps(1.8);
    let mut v9 = _mm512_set1_ps(1.9);
    let mul = _mm512_set1_ps(1.0000001);
    let add = _mm512_set1_ps(0.0000001);

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        // 10 vfmadd231ps zmm: each = 16 FMA = 32 FLOP → 320 FLOP/iter
        v0 = _mm512_fmadd_ps(v0, mul, add);
        v1 = _mm512_fmadd_ps(v1, mul, add);
        v2 = _mm512_fmadd_ps(v2, mul, add);
        v3 = _mm512_fmadd_ps(v3, mul, add);
        v4 = _mm512_fmadd_ps(v4, mul, add);
        v5 = _mm512_fmadd_ps(v5, mul, add);
        v6 = _mm512_fmadd_ps(v6, mul, add);
        v7 = _mm512_fmadd_ps(v7, mul, add);
        v8 = _mm512_fmadd_ps(v8, mul, add);
        v9 = _mm512_fmadd_ps(v9, mul, add);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let sum = _mm512_add_ps(v0, v1);
    let sum = _mm512_add_ps(sum, v2);
    let sum = _mm512_add_ps(sum, v3);
    let sum = _mm512_add_ps(sum, v4);
    let sum = _mm512_add_ps(sum, v5);
    let sum = _mm512_add_ps(sum, v6);
    let sum = _mm512_add_ps(sum, v7);
    let sum = _mm512_add_ps(sum, v8);
    let sum = _mm512_add_ps(sum, v9);
    std::hint::black_box(sum);

    // 10 FMAs * 16 elements * 2 ops = 320 FLOP per iteration
    ITERS as f64 * 320.0 / elapsed
}

/// Run empirical roofline measurement and return enhanced model.
pub fn measure_empirical(theoretical: &RooflineModel) -> EmpiricalResult {
    let bw_copy = measure_bandwidth();
    let bw_triad = measure_bandwidth_triad();
    // Use the better of copy and triad as the bandwidth number
    let measured_bw = bw_copy.max(bw_triad);
    let measured_flops = measure_peak_flops_single_core();

    let theoretical_bw = theoretical.peak_bandwidth.get(&MemoryLevel::Dram).copied().unwrap_or(1.0);
    let theoretical_flops = theoretical.peak_compute.get(&Precision::Fp32).copied().unwrap_or(1.0);

    // For single-core measurement, divide theoretical by core count
    // The theoretical model includes all cores, so single-core peak = theoretical / cores
    let cores = num_cpus::get_physical() as f64;
    let single_core_theoretical = theoretical_flops / cores;

    EmpiricalResult {
        measured_bandwidth_bps: measured_bw,
        measured_peak_flops: measured_flops,
        measured_ridge_point: measured_flops / measured_bw,
        bandwidth_efficiency: measured_bw / theoretical_bw * 100.0,
        compute_efficiency: measured_flops / single_core_theoretical * 100.0,
    }
}

/// Print roofline model to stdout in human-readable format.
fn print_roofline(model: &RooflineModel) {
    println!("\n=== cgp Roofline: {} ===\n", model.target);

    println!("  Peak Compute:");
    let mut precisions: Vec<_> = model.peak_compute.iter().collect();
    precisions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (prec, peak) in &precisions {
        println!("    {prec:15}: {:8.1} TFLOP/s", *peak / 1e12);
    }

    println!("\n  Peak Bandwidth:");
    let mut levels: Vec<_> = model.peak_bandwidth.iter().collect();
    levels.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (level, bw) in &levels {
        if **bw >= 1e12 {
            println!("    {level:15}: {:8.1} TB/s", *bw / 1e12);
        } else {
            println!("    {level:15}: {:8.1} GB/s", *bw / 1e9);
        }
    }

    println!("\n  Ridge Points (vs DRAM):");
    for (prec, _) in &precisions {
        if let Some(ridge) = model.ridge_point(**prec, MemoryLevel::Dram) {
            println!("    {prec:15}: {:8.1} FLOP/byte", ridge);
        }
    }
}

/// Run the `cgp roofline` command.
pub fn run_roofline(
    target: &str,
    _kernels: Option<&str>,
    export: Option<&str>,
    empirical: bool,
    json: bool,
) -> Result<()> {
    let model = match target {
        "cuda" => RooflineModel::rtx_4090(),
        "avx2" => {
            let cores = num_cpus::get_physical();
            RooflineModel::cpu_avx2(3.5, cores, 204.8)
        }
        "avx512" => {
            let cores = num_cpus::get_physical();
            RooflineModel::cpu_avx512(3.5, cores, 204.8)
        }
        "neon" => {
            let cores = num_cpus::get_physical();
            RooflineModel::cpu_neon(3.0, cores, 51.2)
        }
        "wgpu" => RooflineModel::rtx_4090(),
        other => anyhow::bail!(
            "Unknown roofline target: {other}. Supported: cuda, avx2, avx512, neon, wgpu"
        ),
    };

    // JSON + empirical: output combined JSON only
    if json && empirical && !target.starts_with("cuda") && target != "wgpu" {
        let emp = measure_empirical(&model);
        #[derive(Serialize)]
        struct EmpiricalJson<'a> {
            theoretical: &'a RooflineModel,
            empirical: &'a EmpiricalResult,
        }
        let combined = EmpiricalJson { theoretical: &model, empirical: &emp };
        println!("{}", serde_json::to_string_pretty(&combined)?);
        return Ok(());
    }

    if json {
        let json_str = serde_json::to_string_pretty(&model)?;
        println!("{json_str}");
        return Ok(());
    }

    print_roofline(&model);

    if empirical && !target.starts_with("cuda") && target != "wgpu" {
        println!("\n  --- Empirical Measurement (single-core) ---\n");
        let emp = measure_empirical(&model);

        println!(
            "    DRAM Bandwidth:  {:8.1} GB/s  ({:.0}% of theoretical)",
            emp.measured_bandwidth_bps / 1e9,
            emp.bandwidth_efficiency
        );
        println!(
            "    Peak FP32 FLOPS: {:8.1} GFLOP/s (single-core, {:.0}% of theoretical)",
            emp.measured_peak_flops / 1e9,
            emp.compute_efficiency
        );
        println!("    Empirical Ridge: {:8.1} FLOP/byte", emp.measured_ridge_point);
    } else if empirical {
        println!("\n  (Empirical measurement for GPU targets requires CUDA — use cgp roofline --target avx2 --empirical for CPU)");
    }

    if let Some(path) = export {
        let json_str = serde_json::to_string_pretty(&model)?;
        std::fs::write(path, json_str)?;
        println!("\n  Exported to: {path}");
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CGP-021: Ridge point must be correctly computed.
    /// Given: peak_compute = 330 TFLOP/s, peak_bandwidth = 1008 GB/s
    /// Then: ridge_point = 330000 / 1008 = 327.4 FLOP/byte (within 1%)
    #[test]
    fn test_ridge_point_rtx4090_fp16() {
        let model = RooflineModel::rtx_4090();
        let ridge = model.ridge_point(Precision::Fp16, MemoryLevel::Dram).unwrap();
        let expected = 330_000.0 / 1008.0; // 327.38...
        assert!(
            (ridge - expected).abs() < 0.5,
            "Ridge point {ridge:.1} not within 0.5 of expected {expected:.1}"
        );
    }

    /// FALSIFY-CGP-021: All precision ridge points match manual calculation.
    #[test]
    fn test_ridge_points_all_precisions() {
        let model = RooflineModel::rtx_4090();
        let dram_bw = 1008.0e9;

        let cases = [
            (Precision::Fp32, 82.6e12),
            (Precision::Fp16, 330.0e12),
            (Precision::Tf32, 165.0e12),
            (Precision::Int8, 660.0e12),
        ];

        for (prec, peak) in cases {
            let ridge = model.ridge_point(prec, MemoryLevel::Dram).unwrap();
            let expected = peak / dram_bw;
            assert!(
                (ridge - expected).abs() / expected < 0.001,
                "{prec}: ridge {ridge:.2} != expected {expected:.2}"
            );
        }
    }

    /// FALSIFY-ROOF-002: Memory-bound kernel classified correctly.
    #[test]
    fn test_memory_bound_classification() {
        let model = RooflineModel::rtx_4090();
        // AI = 8.0 FLOP/byte, well below ridge of 327.4
        let point = model.classify(8.0, 5e12, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!(matches!(point.bound, Bound::Memory { .. }));
        assert!(point.distance_to_ridge > 1.0);
    }

    /// FALSIFY-ROOF-003: Compute-bound kernel classified correctly.
    #[test]
    fn test_compute_bound_classification() {
        let model = RooflineModel::rtx_4090();
        // AI = 500.0 FLOP/byte, above ridge of 327.4
        let point = model.classify(500.0, 300e12, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!(matches!(point.bound, Bound::Compute { .. }));
        assert!(point.distance_to_ridge < 1.0);
    }

    /// Theoretical peak follows min(compute, AI*bandwidth).
    #[test]
    fn test_theoretical_peak() {
        let model = RooflineModel::rtx_4090();
        // Memory-bound region: peak = AI * bandwidth
        let low_ai = model.theoretical_peak(8.0, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!((low_ai - 8.0 * 1008.0e9).abs() / low_ai < 0.001);

        // Compute-bound region: peak = compute peak
        let high_ai = model.theoretical_peak(500.0, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!((high_ai - 330.0e12).abs() / high_ai < 0.001);
    }

    /// CPU AVX2 model: peak = 2 FMA units * 8 floats * 2 ops * freq * cores.
    #[test]
    fn test_cpu_avx2_peak() {
        let model = RooflineModel::cpu_avx2(3.5, 8, 51.2);
        let fp32_peak = *model.peak_compute.get(&Precision::Fp32).unwrap();
        let expected = 2.0 * 8.0 * 2.0 * 3.5e9 * 8.0; // 896 GFLOP/s
        assert!(
            (fp32_peak - expected).abs() / expected < 0.001,
            "FP32 peak {:.1} GFLOP/s != expected {:.1} GFLOP/s",
            fp32_peak / 1e9,
            expected / 1e9
        );
    }

    /// RTX 4090 bandwidth spec: 384-bit * 21 Gbps = 1008 GB/s.
    #[test]
    fn test_rtx4090_bandwidth_spec() {
        let model = RooflineModel::rtx_4090();
        let dram = *model.peak_bandwidth.get(&MemoryLevel::Dram).unwrap();
        assert!(
            (dram - 1008.0e9).abs() < 1e6,
            "DRAM bandwidth {:.1} GB/s != 1008.0 GB/s",
            dram / 1e9
        );
    }

    /// FALSIFY-CGP-EMPIRICAL-001: Empirical bandwidth must be > 0 and < theoretical.
    #[test]
    fn test_empirical_bandwidth_positive() {
        let bw = measure_bandwidth();
        assert!(bw > 0.0, "Measured bandwidth must be positive, got {bw}");
        // Single-core bandwidth should be less than full system theoretical
        assert!(bw < 500.0e9, "Single-core bandwidth {:.1} GB/s suspiciously high", bw / 1e9);
    }

    /// FALSIFY-CGP-EMPIRICAL-002: Empirical FLOPS must be > 0.
    #[test]
    fn test_empirical_flops_positive() {
        let flops = measure_peak_flops_single_core();
        assert!(flops > 0.0, "Measured FLOPS must be positive, got {flops}");
        // Single-core should be at least 1 GFLOP/s on any modern CPU
        assert!(flops > 1.0e9, "Single-core FLOPS {:.1} GFLOP/s suspiciously low", flops / 1e9);
    }

    /// FALSIFY-CGP-EMPIRICAL-003: Empirical ridge point must be plausible.
    #[test]
    fn test_empirical_ridge_plausible() {
        let model = RooflineModel::cpu_avx512(3.5, 24, 204.8);
        let emp = measure_empirical(&model);
        // Ridge point should be > 0 and < 1000 FLOP/byte for any CPU
        assert!(
            emp.measured_ridge_point > 0.0 && emp.measured_ridge_point < 1000.0,
            "Empirical ridge {:.1} FLOP/byte implausible",
            emp.measured_ridge_point
        );
    }

    /// FALSIFY-CGP-EMPIRICAL-004: Triad bandwidth should be > 0.
    #[test]
    fn test_triad_bandwidth_positive() {
        let bw = measure_bandwidth_triad();
        assert!(bw > 0.0, "Triad bandwidth must be positive, got {bw}");
    }
}
