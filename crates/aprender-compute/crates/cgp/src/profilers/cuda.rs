//! CUDA profiler: wraps ncu, nsys, and CUPTI.
//! See spec sections 4.1.1 (ncu), 4.1.2 (nsys), 4.1.3 (CUPTI).

use crate::analysis::roofline::{Bound, MemoryLevel, Precision, RooflineModel};
use crate::metrics::catalog::*;
use crate::profilers::system;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

/// ncu metric sections — lazily collect only what's requested.
#[derive(Debug, Clone, Copy)]
pub enum NcuSection {
    LaunchStats,
    ComputeThroughput,
    MemoryThroughput,
    Occupancy,
    Roofline,
    WarpState,
}

impl NcuSection {
    fn as_ncu_arg(&self) -> &str {
        match self {
            NcuSection::LaunchStats => "LaunchStats",
            NcuSection::ComputeThroughput => "ComputeWorkloadAnalysis",
            NcuSection::MemoryThroughput => "MemoryWorkloadAnalysis",
            NcuSection::Occupancy => "Occupancy",
            NcuSection::Roofline => "SpeedOfLight",
            NcuSection::WarpState => "WarpStateStats",
        }
    }
}

/// Wraps `ncu` CLI for kernel-level profiling.
pub struct NcuProfiler {
    pub ncu_path: PathBuf,
    pub sections: Vec<NcuSection>,
}

impl NcuProfiler {
    pub fn detect() -> Option<Self> {
        which::which("ncu").ok().map(|path| Self {
            ncu_path: path,
            sections: vec![
                NcuSection::LaunchStats,
                NcuSection::Roofline,
                NcuSection::ComputeThroughput,
                NcuSection::MemoryThroughput,
                NcuSection::Occupancy,
            ],
        })
    }

    /// Run ncu and collect metrics for a kernel.
    pub fn profile(
        &self,
        binary: &str,
        binary_args: &[&str],
        kernel_regex: &str,
    ) -> Result<HashMap<String, String>> {
        let mut cmd = Command::new(&self.ncu_path);
        cmd.arg("--target-processes").arg("all");
        cmd.arg("--kernel-name-base").arg("demangled");

        if !kernel_regex.is_empty() {
            cmd.arg("--kernel-id").arg(format!("::regex:{kernel_regex}:"));
        }

        for section in &self.sections {
            cmd.arg("--section").arg(section.as_ncu_arg());
        }

        cmd.arg("--csv");
        cmd.arg("--log-file").arg("/dev/null");
        cmd.arg(binary);
        cmd.args(binary_args);

        let output = cmd
            .output()
            .with_context(|| format!("Failed to run ncu: {}", self.ncu_path.display()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // ncu often needs root — provide helpful message
            if stderr.contains("permission") || stderr.contains("ERR_NVGPUCTRPERM") {
                anyhow::bail!(
                    "ncu requires elevated permissions. Run with:\n  \
                     sudo cgp profile kernel ...\n  \
                     or set: sudo sysctl kernel.perf_event_paranoid=2"
                );
            }
            anyhow::bail!("ncu failed (exit {}): {}", output.status, stderr.trim());
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_ncu_csv(&stdout)
    }
}

/// Parse ncu CSV output into a metric name → value map.
/// ncu CSV format: "ID","Metric Name","Metric Unit","Metric Value"
pub fn parse_ncu_csv(csv: &str) -> Result<HashMap<String, String>> {
    let mut metrics = HashMap::new();
    for line in csv.lines() {
        // Skip header and non-data lines
        if line.starts_with('"') && !line.starts_with("\"ID\"") {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 4 {
                let name = fields[1].trim_matches('"').to_string();
                let value = fields[3].trim_matches('"').to_string();
                metrics.insert(name, value);
            }
        }
    }
    Ok(metrics)
}

/// Extract a float metric, returning 0.0 if not found.
fn get_f64(metrics: &HashMap<String, String>, key: &str) -> f64 {
    metrics.get(key).and_then(|v| v.replace(',', "").parse::<f64>().ok()).unwrap_or(0.0)
}

/// Extract a u64 metric.
#[allow(dead_code)]
fn get_u64(metrics: &HashMap<String, String>, key: &str) -> u64 {
    metrics.get(key).and_then(|v| v.replace(',', "").parse::<u64>().ok()).unwrap_or(0)
}

/// Extract a u32 metric.
fn get_u32(metrics: &HashMap<String, String>, key: &str) -> u32 {
    metrics.get(key).and_then(|v| v.replace(',', "").parse::<u32>().ok()).unwrap_or(0)
}

/// Build a FullProfile from ncu metrics.
#[allow(clippy::implicit_hasher)]
pub fn ncu_metrics_to_profile(
    metrics: &HashMap<String, String>,
    kernel_name: &str,
    size: u32,
) -> FullProfile {
    let duration_us = get_f64(metrics, "gpu__time_duration.sum") / 1000.0; // ns → us
    let flops = 2.0 * (size as f64).powi(3); // GEMM: 2*M*N*K
    let tflops = if duration_us > 0.0 { flops / (duration_us * 1e-6) / 1e12 } else { 0.0 };

    let sm_pct = get_f64(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed");
    let dram_pct = get_f64(metrics, "dram__throughput.avg.pct_of_peak_sustained_elapsed");
    let occupancy_pct = get_f64(metrics, "sm__warps_active.avg.pct_of_peak_sustained_elapsed");
    let warp_eff = get_f64(metrics, "smsp__thread_inst_executed_per_inst_executed.pct");
    let tc_pct =
        get_f64(metrics, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed");
    let regs = get_u32(metrics, "launch__registers_per_thread");
    let smem = get_u32(metrics, "launch__shared_mem_per_block_driver");
    let l2_hit = get_f64(metrics, "lts__t_sector_hit_rate.pct");
    let global_load_eff =
        get_f64(metrics, "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct");

    // Roofline analysis
    let model = RooflineModel::rtx_4090();
    let ai = if duration_us > 0.0 {
        // Approximate: use dram throughput pct to estimate bytes
        let dram_bw = 1008.0e9; // GB/s
        let actual_bw = dram_bw * dram_pct / 100.0;
        let bytes = actual_bw * duration_us * 1e-6;
        if bytes > 0.0 {
            flops / bytes
        } else {
            0.0
        }
    } else {
        0.0
    };

    let roofline =
        model.classify(ai, tflops * 1e12, Precision::Fp16, MemoryLevel::Dram).map(|point| {
            let bound_str = match &point.bound {
                Bound::Memory { .. } => "memory".to_string(),
                Bound::Compute { .. } => "compute".to_string(),
            };
            RooflineMetrics {
                peak_compute_tflops: 330.0,
                peak_bandwidth_gbps: 1008.0,
                ridge_point: 327.4,
                bound: bound_str,
                efficiency_pct: point.efficiency,
                distance_to_ridge: point.distance_to_ridge,
            }
        });

    let timestamp = chrono::Utc::now().to_rfc3339();

    FullProfile {
        version: "2.0".to_string(),
        timestamp,
        hardware: HardwareInfo {
            gpu: Some("NVIDIA GeForce RTX 4090".to_string()),
            gpu_sm: Some("8.9".to_string()),
            gpu_memory_gb: Some(24.0),
            gpu_bandwidth_gbps: Some(1008.0),
            ..Default::default()
        },
        kernel: Some(KernelInfo {
            name: kernel_name.to_string(),
            dimensions: vec![size, size, size],
            shared_memory_bytes: Some(smem),
            registers_per_thread: Some(regs),
            ..Default::default()
        }),
        timing: TimingMetrics { wall_clock_time_us: duration_us, samples: 1, ..Default::default() },
        throughput: ThroughputMetrics {
            tflops,
            bandwidth_gbps: 1008.0 * dram_pct / 100.0,
            arithmetic_intensity: ai,
            ..Default::default()
        },
        roofline,
        gpu_compute: Some(GpuComputeMetrics {
            sm_utilization_pct: sm_pct,
            achieved_occupancy_pct: occupancy_pct,
            warp_execution_efficiency_pct: warp_eff,
            tensor_core_utilization_pct: tc_pct,
            register_usage_per_thread: regs,
            shared_memory_per_block: smem,
            ..Default::default()
        }),
        gpu_memory: Some(GpuMemoryMetrics {
            dram_throughput_pct: dram_pct,
            l2_hit_rate_pct: l2_hit,
            global_load_efficiency_pct: global_load_eff,
            ..Default::default()
        }),
        system_health: system::collect_system_health(),
        vram: system::collect_vram(),
        energy: system::collect_system_health()
            .and_then(|h| system::compute_energy(h.gpu_power_watts, tflops, duration_us)),
        ..Default::default()
    }
}

/// Render the profile to stdout in the spec-mandated format.
fn render_profile(profile: &FullProfile) {
    let kernel = profile.kernel.as_ref().map_or("unknown", |k| &k.name);
    let dims = profile
        .kernel
        .as_ref()
        .map(|k| k.dimensions.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x"))
        .unwrap_or_default();

    let gpu_name = profile.hardware.gpu.as_deref().unwrap_or("Unknown GPU");
    let sm = profile.hardware.gpu_sm.as_deref().unwrap_or("?");

    println!("\n=== CGP Kernel Profile: {kernel} ({dims}) ===\n");
    println!(
        "Backend: CUDA ({gpu_name}, SM {sm}, Driver {})",
        detect_driver_version().unwrap_or_else(|| "?".to_string())
    );

    let t = &profile.timing;
    let tp = &profile.throughput;
    let peak_pct = if tp.tflops > 0.0 { tp.tflops / 330.0 * 100.0 } else { 0.0 };
    println!(
        "Execution: {:.1} us  |  {:.1} TFLOP/s  |  {:.1}% of peak",
        t.wall_clock_time_us, tp.tflops, peak_pct
    );

    // Roofline
    if let Some(roof) = &profile.roofline {
        println!("\n  Roofline Position:");
        println!("    Arithmetic Intensity: {:.1} FLOP/byte", tp.arithmetic_intensity);
        println!("    Ridge Point: {:.1} FLOP/byte", roof.ridge_point);
        let status = if roof.bound == "memory" {
            format!("MEMORY-BOUND ({:.1}x below ridge)", roof.distance_to_ridge)
        } else {
            format!("COMPUTE-BOUND ({:.1}% efficiency)", roof.efficiency_pct)
        };
        println!("    Status: {status}");
    }

    // Compute metrics
    if let Some(gc) = &profile.gpu_compute {
        println!("\n  Compute:");
        if gc.tensor_core_utilization_pct > 0.0 {
            println!(
                "    Tensor core utilization: {:5.1}%   {}",
                gc.tensor_core_utilization_pct,
                quality_badge(gc.tensor_core_utilization_pct, 50.0)
            );
        }
        println!(
            "    Warp execution eff:     {:5.1}%   {}",
            gc.warp_execution_efficiency_pct,
            quality_badge(gc.warp_execution_efficiency_pct, 95.0)
        );
        println!("    SM utilization:         {:5.1}%", gc.sm_utilization_pct);
        println!("    Achieved occupancy:     {:5.1}%", gc.achieved_occupancy_pct);
        println!("    Register usage:          {:3}/255", gc.register_usage_per_thread);
        if gc.shared_memory_per_block > 0 {
            println!("    Shared memory/block:   {:5} bytes", gc.shared_memory_per_block);
        }
    }

    // Memory metrics
    if let Some(gm) = &profile.gpu_memory {
        println!("\n  Memory:");
        println!(
            "    DRAM throughput:        {:5.1}% of peak ({:.1} GB/s)",
            gm.dram_throughput_pct,
            1008.0 * gm.dram_throughput_pct / 100.0
        );
        println!(
            "    Global load coalescing: {:5.1}%   {}",
            gm.global_load_efficiency_pct,
            quality_badge(gm.global_load_efficiency_pct, 60.0)
        );
        println!(
            "    L2 hit rate:            {:5.1}%   {}",
            gm.l2_hit_rate_pct,
            quality_badge(gm.l2_hit_rate_pct, 50.0)
        );
    }

    // VRAM metrics
    if let Some(vram) = &profile.vram {
        println!("\n  VRAM:");
        println!(
            "    Used: {:.0} / {:.0} MB ({:.1}%)",
            vram.vram_used_mb, vram.vram_total_mb, vram.vram_utilization_pct
        );
    }

    // System health
    if let Some(health) = &profile.system_health {
        println!("\n  System Health:");
        println!("    GPU temp:  {:.0}°C", health.gpu_temperature_celsius);
        println!("    GPU power: {:.0} W", health.gpu_power_watts);
        println!(
            "    GPU clock: {:.0} MHz (mem: {:.0} MHz)",
            health.gpu_clock_mhz, health.gpu_memory_clock_mhz
        );
        if health.cpu_frequency_mhz > 0.0 {
            println!("    CPU freq:  {:.0} MHz", health.cpu_frequency_mhz);
        }
    }

    // Energy efficiency
    if let Some(energy) = &profile.energy {
        println!("\n  Energy:");
        println!("    Efficiency: {:.4} TFLOP/s per watt", energy.tflops_per_watt);
        println!("    Energy:     {:.6} J per inference", energy.joules_per_inference);
    }

    println!();
}

fn quality_badge(value: f64, threshold: f64) -> &'static str {
    if value >= threshold {
        "[OK]"
    } else {
        "[WARN]"
    }
}

/// Get driver version from nvidia-smi.
fn detect_driver_version() -> Option<String> {
    Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// Detect GPU name from nvidia-smi.
fn detect_gpu_name() -> Option<String> {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

// ── Public API: profile commands ──

/// Profile a CUDA PTX kernel via ncu.
pub fn profile_kernel(name: &str, size: u32, roofline: bool, _metrics: Option<&str>) -> Result<()> {
    let profiler = match NcuProfiler::detect() {
        Some(mut p) => {
            if roofline {
                // Ensure we have all sections needed for roofline
                p.sections = vec![
                    NcuSection::LaunchStats,
                    NcuSection::Roofline,
                    NcuSection::ComputeThroughput,
                    NcuSection::MemoryThroughput,
                    NcuSection::Occupancy,
                ];
            }
            p
        }
        None => {
            // Fallback: show static roofline data
            println!("\n=== CGP Kernel Profile: {name} ({size}x{size}x{size}) ===\n");
            println!("  ncu not found. Showing static analysis only.\n");
            let model = RooflineModel::rtx_4090();
            let flops = 2.0 * (size as f64).powi(3);
            println!("  Expected FLOPs: {:.2e}", flops);
            if let Some(ridge) = model.ridge_point(Precision::Fp16, MemoryLevel::Dram) {
                println!("  FP16 ridge point: {:.1} FLOP/byte", ridge);
            }
            println!("  Install NVIDIA Nsight Compute for runtime profiling.");
            return Ok(());
        }
    };

    // Try to find a trueno benchmark binary that exercises this kernel
    let binary = find_kernel_binary(name);

    match binary {
        Some((bin_path, bin_args)) => {
            eprintln!("Profiling {name} via ncu (this may take a moment)...");
            let metrics = profiler.profile(
                &bin_path,
                &bin_args.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                name,
            )?;
            let profile = ncu_metrics_to_profile(&metrics, name, size);
            render_profile(&profile);

            // Export JSON alongside
            let json_path = format!("/tmp/cgp-{name}-{size}.json");
            let json = serde_json::to_string_pretty(&profile)?;
            std::fs::write(&json_path, &json)?;
            println!("  Profile exported: {json_path}");
        }
        None => {
            // No binary found — run ncu with a placeholder message
            println!("\n=== CGP Kernel Profile: {name} ({size}x{size}x{size}) ===\n");
            println!("  Backend: CUDA (ncu at {})", profiler.ncu_path.display());
            println!("  GPU: {}", detect_gpu_name().unwrap_or_else(|| "Unknown".to_string()));
            println!(
                "  Driver: {}",
                detect_driver_version().unwrap_or_else(|| "Unknown".to_string())
            );
            println!("\n  No binary found for kernel '{name}'.");
            println!("  To profile a specific binary, use:");
            println!("    cgp profile binary ./your_binary --kernel-filter {name}");
            println!();

            // Still show roofline analysis
            if roofline {
                let model = RooflineModel::rtx_4090();
                let flops = 2.0 * (size as f64).powi(3);
                println!("  Roofline analysis (static):");
                println!("    Problem FLOPs: {:.2e}", flops);
                if let Some(ridge) = model.ridge_point(Precision::Fp16, MemoryLevel::Dram) {
                    println!("    FP16 ridge: {:.1} FLOP/byte", ridge);
                }
                println!();
            }
        }
    }

    Ok(())
}

/// Try to find a built CUDA binary that exercises a given kernel.
fn find_kernel_binary(_kernel_name: &str) -> Option<(String, Vec<String>)> {
    // Look for GPU-specific examples/benches
    let candidates = [
        "/mnt/nvme-raid0/targets/trueno/release/examples/gpu_batch_demo",
        "./target/release/examples/gpu_batch_demo",
        "/mnt/nvme-raid0/targets/trueno/release/examples/wgpu_backward_demo",
        "./target/release/examples/wgpu_backward_demo",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some((path.to_string(), vec![]));
        }
    }
    None
}

/// Profile cuBLAS operations.
pub fn profile_cublas(op: &str, size: u32) -> Result<()> {
    println!("\n=== CGP cuBLAS Profile: {op} ({size}x{size}) ===\n");

    let gpu_name = detect_gpu_name().unwrap_or_else(|| "Unknown GPU".to_string());
    let driver = detect_driver_version().unwrap_or_else(|| "?".to_string());
    println!("  GPU: {gpu_name} (Driver {driver})");

    // Calculate expected performance from roofline
    let flops = 2.0 * (size as f64).powi(3);
    println!("  Operation: {op}");
    println!("  Problem size: {size}x{size}x{size}");
    println!("  FLOPs: {flops:.2e}");

    // cuBLAS peak estimates for RTX 4090
    let (peak_tflops, precision) = match op {
        "gemm_f16" | "hgemm" => (330.0, "FP16 Tensor"),
        "gemm_f32" | "sgemm" => (82.6, "FP32"),
        "gemm_tf32" => (165.0, "TF32 Tensor"),
        _ => (82.6, "FP32 (default)"),
    };
    println!("  Precision: {precision} (peak: {peak_tflops:.1} TFLOP/s)");

    // Try to find a cuBLAS benchmark binary
    let cublas_bin = find_cublas_binary();
    match cublas_bin {
        Some(bin) => {
            println!("  Binary: {bin}");
            if let Some(nsys) = NsysProfiler::detect() {
                println!("  Profiling via nsys...");
                match nsys.profile_binary(&bin, &[]) {
                    Ok(stats) => {
                        // Find cuBLAS kernels
                        let cublas_kernels: Vec<_> = stats
                            .iter()
                            .filter(|s| {
                                s.name.contains("gemm")
                                    || s.name.contains("cublas")
                                    || s.name.contains("Gemm")
                            })
                            .collect();
                        if cublas_kernels.is_empty() {
                            println!("  No cuBLAS kernels found in trace.");
                        } else {
                            println!("\n  cuBLAS Kernels:");
                            for k in &cublas_kernels {
                                let tflops = flops / (k.avg_us * 1e-6) / 1e12;
                                let eff = tflops / peak_tflops * 100.0;
                                println!(
                                    "    {:50} avg={:.1}us  {:.1} TFLOP/s  ({:.1}% eff)",
                                    k.name, k.avg_us, tflops, eff
                                );
                            }
                        }
                    }
                    Err(e) => println!("  nsys profiling failed: {e}"),
                }
            }
        }
        None => {
            // Estimate from known measurements
            let est_time_us = flops / (peak_tflops * 0.9 * 1e12) * 1e6; // 90% of peak
            let est_tflops = flops / (est_time_us * 1e-6) / 1e12;
            println!("\n  Estimated (no binary found):");
            println!("    Time: ~{est_time_us:.1} us");
            println!(
                "    Throughput: ~{est_tflops:.1} TFLOP/s ({:.1}% of peak)",
                est_tflops / peak_tflops * 100.0
            );
            println!("\n  To profile real cuBLAS:");
            println!("    cgp profile binary ./your_cublas_bench --kernel-filter gemm");
        }
    }

    println!();
    Ok(())
}

/// Find a cuBLAS benchmark binary.
fn find_cublas_binary() -> Option<String> {
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_default();
    let mut candidates: Vec<String> = Vec::new();
    if !target_dir.is_empty() {
        candidates.push(format!("{target_dir}/release/examples/bench_cublas_vs_ptx"));
        candidates.push(format!("{target_dir}/release/examples/gpu_batch_demo"));
    }
    candidates.extend_from_slice(&[
        "/mnt/nvme-raid0/targets/trueno/release/examples/bench_cublas_vs_ptx".to_string(),
        "/mnt/nvme-raid0/targets/trueno/release/examples/gpu_batch_demo".to_string(),
    ]);
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.clone());
        }
    }
    None
}

// ── nsys integration ──

/// Wraps `nsys` CLI for system-wide timeline profiling.
pub struct NsysProfiler {
    pub nsys_path: PathBuf,
}

/// Parsed nsys stats output — one entry per kernel.
#[derive(Debug, Clone)]
pub struct NsysKernelStat {
    pub name: String,
    pub calls: u64,
    pub total_us: f64,
    pub avg_us: f64,
    pub min_us: f64,
    pub max_us: f64,
}

impl NsysProfiler {
    pub fn detect() -> Option<Self> {
        which::which("nsys").ok().map(|path| Self { nsys_path: path })
    }

    /// Run nsys profile and capture stats.
    pub fn profile_binary(
        &self,
        binary: &str,
        binary_args: &[&str],
    ) -> Result<Vec<NsysKernelStat>> {
        let report_path = format!("/tmp/cgp-nsys-{}", std::process::id());

        let mut cmd = Command::new(&self.nsys_path);
        cmd.arg("profile")
            .arg("--stats=true")
            .arg("--force-overwrite=true")
            .arg("-o")
            .arg(&report_path)
            .arg(binary)
            .args(binary_args);

        let output = cmd.output().with_context(|| "Failed to run nsys")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}\n{stderr}");

        let stats = parse_nsys_stats(&combined);

        // Clean up report files
        let _ = std::fs::remove_file(format!("{report_path}.nsys-rep"));
        let _ = std::fs::remove_file(format!("{report_path}.sqlite"));

        Ok(stats)
    }
}

/// Parse nsys stats output for CUDA kernel summary.
fn parse_nsys_stats(output: &str) -> Vec<NsysKernelStat> {
    let mut stats = Vec::new();
    let mut in_kernel_section = false;

    for line in output.lines() {
        if line.contains("CUDA Kernel Statistics") || line.contains("cuda_gpu_kern_sum") {
            in_kernel_section = true;
            continue;
        }
        if in_kernel_section && line.trim().is_empty() {
            in_kernel_section = false;
            continue;
        }
        if in_kernel_section {
            // nsys stats format varies; try to parse common patterns
            // Typical: Time(%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  Name
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 8 {
                if let (Ok(total_ns), Ok(instances)) = (
                    parts[1].replace(',', "").parse::<f64>(),
                    parts[2].replace(',', "").parse::<u64>(),
                ) {
                    let avg_ns = parts[3].replace(',', "").parse::<f64>().unwrap_or(0.0);
                    let min_ns = parts[5].replace(',', "").parse::<f64>().unwrap_or(0.0);
                    let max_ns = parts[6].replace(',', "").parse::<f64>().unwrap_or(0.0);
                    let name = parts[7..].join(" ");
                    stats.push(NsysKernelStat {
                        name,
                        calls: instances,
                        total_us: total_ns / 1000.0,
                        avg_us: avg_ns / 1000.0,
                        min_us: min_ns / 1000.0,
                        max_us: max_ns / 1000.0,
                    });
                }
            }
        }
    }
    stats
}

/// Profile an arbitrary binary via nsys.
pub fn profile_binary(
    path: &str,
    kernel_filter: Option<&str>,
    _trace: bool,
    _duration: Option<&str>,
) -> Result<()> {
    println!("\n=== CGP Binary Profile: {path} ===\n");

    let profiler = match NsysProfiler::detect() {
        Some(p) => p,
        None => {
            println!("  nsys not found. Install NVIDIA Nsight Systems.");
            println!("  Falling back to wall-clock timing...");

            // Fallback: just run and time it
            let start = std::time::Instant::now();
            let status = Command::new(path).status()?;
            let elapsed = start.elapsed();
            println!(
                "  Wall time: {:.2}ms (exit code: {})",
                elapsed.as_secs_f64() * 1000.0,
                status.code().unwrap_or(-1)
            );
            return Ok(());
        }
    };

    eprintln!("Running nsys profile (this may take a moment)...");
    let stats = profiler.profile_binary(path, &[])?;

    if stats.is_empty() {
        println!("  No CUDA kernels detected. Binary may be CPU-only.");
        println!("  For CPU profiling, use: cgp profile simd or cgp profile scalar");
    } else {
        println!(
            "  {:40} {:>8} {:>12} {:>12} {:>12}",
            "Kernel", "Calls", "Avg (us)", "Min (us)", "Max (us)"
        );
        println!("  {}", "-".repeat(88));

        for stat in &stats {
            let name = if let Some(filter) = kernel_filter {
                if !stat.name.contains(filter) {
                    continue;
                }
                &stat.name
            } else {
                &stat.name
            };

            let display_name =
                if name.len() > 40 { format!("{}...", &name[..37]) } else { name.to_string() };

            println!(
                "  {:40} {:>8} {:>12.1} {:>12.1} {:>12.1}",
                display_name, stat.calls, stat.avg_us, stat.min_us, stat.max_us
            );
        }
        println!();
        println!(
            "  Total kernels: {}, Total CUDA time: {:.1} us",
            stats.len(),
            stats.iter().map(|s| s.total_us).sum::<f64>()
        );
    }

    println!();
    Ok(())
}

/// Profile a Python script via nsys + perf stat.
pub fn profile_python(args: &[String]) -> Result<()> {
    let cmd_str = args.join(" ");
    println!("\n=== CGP Python Profile ===\n");
    println!("  Command: {cmd_str}");

    if let Some(profiler) = NsysProfiler::detect() {
        eprintln!("Running nsys profile on Python script...");
        // Build the full command: uv run python <args>
        let python_cmd = if args.is_empty() {
            anyhow::bail!(
                "No Python script specified. Usage: cgp profile python -- uv run python script.py"
            );
        } else {
            args[0].clone()
        };
        let python_args: Vec<&str> = args[1..].iter().map(|s| s.as_str()).collect();
        let stats = profiler.profile_binary(&python_cmd, &python_args)?;

        if stats.is_empty() {
            println!("  No CUDA kernels detected (CPU-only Python workload).");
            println!("  Use perf stat for CPU profiling.");
        } else {
            println!("  CUDA kernels captured via nsys:");
            for stat in &stats {
                println!("    {} — {} calls, avg {:.1}us", stat.name, stat.calls, stat.avg_us);
            }
        }
    } else {
        println!("  nsys not found — cannot capture CUDA kernels.");
        println!("  Falling back to wall-clock timing.");

        if !args.is_empty() {
            let start = std::time::Instant::now();
            let status = Command::new(&args[0]).args(&args[1..]).status()?;
            let elapsed = start.elapsed();
            println!(
                "  Wall time: {:.2}ms (exit code: {})",
                elapsed.as_secs_f64() * 1000.0,
                status.code().unwrap_or(-1)
            );
        }
    }

    println!();
    Ok(())
}

/// Run `cgp trace` — system-wide timeline via nsys.
pub fn run_trace(binary: &str, duration: Option<&str>) -> Result<()> {
    println!("\n=== CGP System Trace: {binary} ===\n");

    let profiler = match NsysProfiler::detect() {
        Some(p) => p,
        None => {
            anyhow::bail!("nsys not found. Install NVIDIA Nsight Systems for timeline tracing.");
        }
    };

    let report_path = format!("/tmp/cgp-trace-{}", std::process::id());

    let mut cmd = Command::new(&profiler.nsys_path);
    cmd.arg("profile")
        .arg("--stats=true")
        .arg("--force-overwrite=true")
        .arg("--trace=cuda,nvtx,osrt")
        .arg("-o")
        .arg(&report_path);

    if let Some(dur) = duration {
        cmd.arg("--duration").arg(dur);
    }

    cmd.arg(binary);

    eprintln!("Running nsys trace (this may take a while)...");
    let output = cmd.output().with_context(|| "Failed to run nsys trace")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Print stats summary
    let combined = format!("{stdout}\n{stderr}");
    let mut in_summary = false;
    for line in combined.lines() {
        if line.contains("CUDA API Statistics")
            || line.contains("CUDA Kernel Statistics")
            || line.contains("OS Runtime Statistics")
        {
            in_summary = true;
            println!("  {line}");
            continue;
        }
        if in_summary {
            if line.trim().is_empty() {
                in_summary = false;
                println!();
            } else {
                println!("  {line}");
            }
        }
    }

    let report_file = format!("{report_path}.nsys-rep");
    if std::path::Path::new(&report_file).exists() {
        println!("  Report: {report_file}");
        println!("  View: nsys-ui {report_file}");
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncu_section_args() {
        assert_eq!(NcuSection::LaunchStats.as_ncu_arg(), "LaunchStats");
        assert_eq!(NcuSection::Roofline.as_ncu_arg(), "SpeedOfLight");
        assert_eq!(NcuSection::WarpState.as_ncu_arg(), "WarpStateStats");
    }

    #[test]
    fn test_ncu_command_build() {
        if let Some(profiler) = NcuProfiler::detect() {
            assert!(profiler.ncu_path.exists());
        }
    }

    #[test]
    fn test_parse_ncu_csv_empty() {
        let metrics = parse_ncu_csv("").unwrap();
        assert!(metrics.is_empty());
    }

    #[test]
    fn test_parse_ncu_csv_sample() {
        let csv = r#""ID","Metric Name","Metric Unit","Metric Value"
"0","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42.3"
"0","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","7.8"
"0","launch__registers_per_thread","register/thread","48"
"0","gpu__time_duration.sum","nsecond","23200"
"#;
        let metrics = parse_ncu_csv(csv).unwrap();
        assert_eq!(get_f64(&metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed"), 42.3);
        assert_eq!(get_u32(&metrics, "launch__registers_per_thread"), 48);
        assert_eq!(get_f64(&metrics, "gpu__time_duration.sum"), 23200.0);
    }

    #[test]
    fn test_ncu_metrics_to_profile() {
        let mut metrics = HashMap::new();
        metrics.insert(
            "gpu__time_duration.sum".to_string(),
            "23200".to_string(), // 23.2 us
        );
        metrics.insert(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed".to_string(),
            "42.3".to_string(),
        );
        metrics.insert(
            "dram__throughput.avg.pct_of_peak_sustained_elapsed".to_string(),
            "7.8".to_string(),
        );
        metrics.insert("launch__registers_per_thread".to_string(), "48".to_string());
        metrics.insert(
            "smsp__thread_inst_executed_per_inst_executed.pct".to_string(),
            "100.0".to_string(),
        );

        let profile = ncu_metrics_to_profile(&metrics, "gemm_cta_wmma_fp16", 512);
        assert_eq!(profile.kernel.as_ref().unwrap().name, "gemm_cta_wmma_fp16");
        assert!(profile.timing.wall_clock_time_us > 0.0);
        assert!(profile.throughput.tflops > 0.0);
        assert!(profile.gpu_compute.is_some());
    }

    #[test]
    fn test_parse_nsys_stats_empty() {
        let stats = parse_nsys_stats("");
        assert!(stats.is_empty());
    }

    #[test]
    fn test_render_profile_no_panic() {
        let profile = FullProfile {
            version: "2.0".to_string(),
            hardware: HardwareInfo {
                gpu: Some("Test GPU".to_string()),
                gpu_sm: Some("8.9".to_string()),
                ..Default::default()
            },
            kernel: Some(KernelInfo {
                name: "test_kernel".to_string(),
                dimensions: vec![256, 256, 256],
                ..Default::default()
            }),
            timing: TimingMetrics { wall_clock_time_us: 10.0, samples: 1, ..Default::default() },
            throughput: ThroughputMetrics { tflops: 5.0, ..Default::default() },
            gpu_compute: Some(GpuComputeMetrics {
                sm_utilization_pct: 40.0,
                warp_execution_efficiency_pct: 100.0,
                achieved_occupancy_pct: 33.0,
                register_usage_per_thread: 48,
                ..Default::default()
            }),
            gpu_memory: Some(GpuMemoryMetrics {
                dram_throughput_pct: 7.8,
                l2_hit_rate_pct: 87.0,
                global_load_efficiency_pct: 72.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        // Should not panic
        render_profile(&profile);
    }
}
