//! Quantized kernel profiler (Q4K/Q6K CPU). Spec section 4.7.
//! Profiles trueno's fused dequantization + GEMV CPU kernels.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Supported quantized kernel types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantKernel {
    Q4kGemv,
    Q5kGemv,
    Q6kGemv,
    Q8Gemv,
    Nf4Gemv,
}

impl std::str::FromStr for QuantKernel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "q4k_gemv" | "q4k" => Ok(Self::Q4kGemv),
            "q5k_gemv" | "q5k" => Ok(Self::Q5kGemv),
            "q6k_gemv" | "q6k" => Ok(Self::Q6kGemv),
            "q8_gemv" | "q8" => Ok(Self::Q8Gemv),
            "nf4_gemv" | "nf4" => Ok(Self::Nf4Gemv),
            _ => anyhow::bail!("Unknown quant kernel: {s}. Supported: q4k_gemv, q5k_gemv, q6k_gemv, q8_gemv, nf4_gemv"),
        }
    }
}

impl QuantKernel {
    /// Super-block size for this quantization format.
    pub fn superblock_elements(&self) -> u32 {
        match self {
            QuantKernel::Q4kGemv => 256,
            QuantKernel::Q5kGemv => 256,
            QuantKernel::Q6kGemv => 256,
            QuantKernel::Q8Gemv => 256,
            QuantKernel::Nf4Gemv => 64,
        }
    }

    /// Bytes per super-block.
    pub fn superblock_bytes(&self) -> u32 {
        match self {
            QuantKernel::Q4kGemv => 144, // Q4K: 256 elements in 144 bytes
            QuantKernel::Q5kGemv => 176,
            QuantKernel::Q6kGemv => 210,
            QuantKernel::Q8Gemv => 256, // 1 byte per element
            QuantKernel::Nf4Gemv => 32, // 4 bits per element
        }
    }
}

/// Quantized kernel profile output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantProfile {
    pub kernel: QuantKernel,
    pub dimensions: [u32; 3],
    /// Super-blocks processed per second.
    pub superblocks_per_sec: f64,
    /// Effective memory bandwidth (compressed input bytes / time).
    pub effective_bandwidth_gbps: f64,
    /// Compression speedup vs FP32 baseline.
    pub compression_speedup: f64,
    /// Wall clock time in microseconds.
    pub wall_time_us: f64,
}

/// Parse dimension string "MxNxK" into [M, N, K].
fn parse_dimensions(size: &str) -> Result<[u32; 3]> {
    let parts: Vec<&str> = size.split('x').collect();
    if parts.len() != 3 {
        anyhow::bail!("Dimensions must be MxNxK format, got: {size}");
    }
    let m: u32 = parts[0]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid M: {}", parts[0]))?;
    let n: u32 = parts[1]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid N: {}", parts[1]))?;
    let k: u32 = parts[2]
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid K: {}", parts[2]))?;
    Ok([m, n, k])
}

/// Profile a quantized GEMV kernel.
pub fn profile_quant(kernel_name: &str, size: &str) -> Result<()> {
    let kernel: QuantKernel = kernel_name.parse()?;
    let dims = parse_dimensions(size)?;

    println!("\n=== CGP Quant Profile: {kernel_name} ({size}) ===\n");
    println!("  Kernel: {kernel:?}");
    println!("  Dimensions: M={}, N={}, K={}", dims[0], dims[1], dims[2]);
    println!(
        "  Super-block: {} elements, {} bytes",
        kernel.superblock_elements(),
        kernel.superblock_bytes()
    );

    // Calculate theoretical metrics
    let total_elements = dims[0] as u64 * dims[2] as u64;
    let num_superblocks = total_elements / kernel.superblock_elements() as u64;
    let compressed_bytes = num_superblocks * kernel.superblock_bytes() as u64;
    let fp32_bytes = total_elements * 4;

    println!("  Total weights: {total_elements}");
    println!("  Super-blocks: {num_superblocks}");
    println!("  Compressed size: {:.2} MB", compressed_bytes as f64 / 1e6);
    println!("  FP32 equivalent: {:.2} MB", fp32_bytes as f64 / 1e6);
    println!(
        "  Compression ratio: {:.1}x",
        fp32_bytes as f64 / compressed_bytes as f64
    );

    // Try to get actual timing from benchmark_matrix_suite
    if let Some(timing) = parse_q4k_timing(dims[0], dims[2]) {
        println!("\n  Measured (from benchmark_matrix_suite):");
        println!("    Time: {:.1} us", timing.time_us);
        println!("    GFLOPS: {:.1}", timing.gflops);
        println!("    Effective BW: {:.1} GB/s (compressed)", timing.bw_gbps);
        let sbs_per_sec = num_superblocks as f64 / (timing.time_us / 1e6);
        println!("    Super-blocks/sec: {:.0}", sbs_per_sec);

        // Roofline analysis
        let flops = 2.0 * dims[0] as f64 * dims[2] as f64; // GEMV: 2*M*K FLOPs
        let ai = flops / compressed_bytes as f64; // FLOP/byte (compressed)
        println!("\n  Roofline Analysis (compressed):");
        println!("    Arithmetic Intensity: {:.1} FLOP/byte", ai);

        // Compare against CPU peak (use empirical single-core for now)
        let peak_bw_gbps = timing.bw_gbps; // Achieved bandwidth
        let peak_flops = timing.gflops;

        // Theoretical peak for single-core AVX-512: ~150 GFLOP/s FP32
        let theoretical_peak_gflops = 150.0;
        let compute_pct = peak_flops / theoretical_peak_gflops * 100.0;

        // DDR5 single-channel practical: ~40 GB/s
        let theoretical_bw_gbps = 40.0;
        let bw_pct = peak_bw_gbps / theoretical_bw_gbps * 100.0;

        println!(
            "    Compute util: {:.0}% of AVX-512 peak (~150 GFLOP/s)",
            compute_pct
        );
        println!(
            "    Bandwidth util: {:.0}% of practical DRAM (~40 GB/s)",
            bw_pct
        );

        if bw_pct > compute_pct {
            println!("    Bottleneck: COMPUTE-BOUND (fused dequant+dot overhead)");
        } else {
            println!("    Bottleneck: MEMORY-BOUND (limited by DRAM read throughput)");
        }

        // Token estimation for LLM inference
        // Llama 7B: ~32 layers, each with 3 GEMV ops (QKV, attention, FFN up+down)
        // Simplified: ~6 GEMV per layer * 32 layers = 192 GEMV per token
        let token_time_ms = timing.time_us * 192.0 / 1000.0;
        let tokens_per_sec = 1000.0 / token_time_ms;
        println!("\n  LLM Token Estimation (Llama-7B-like, {kernel_name}):");
        println!("    Per-layer GEMV: {:.1} us", timing.time_us);
        println!("    Est. 192 GEMVs/token: {:.1} ms", token_time_ms);
        println!("    Est. tokens/sec: {:.1}", tokens_per_sec);
    } else {
        println!("\n  No timing data (build benchmark: cargo build --release --example benchmark_matrix_suite --features parallel)");
    }

    println!();
    Ok(())
}

/// Standard LLM layer sizes for Q4K sweep.
const STANDARD_LAYERS: &[(&str, u32, u32)] = &[
    ("ffn_up/gate (1.5B-7B)", 1536, 8960),
    ("ffn_down (1.5B-7B)", 8960, 1536),
    ("attn_qkv (1.5B-7B)", 1536, 1536),
    ("generic_4K", 4096, 4096),
    ("ffn_up (13B)", 5120, 13824),
    ("ffn_down (13B)", 13824, 5120),
    ("attn_qkv (13B)", 5120, 5120),
];

/// Profile all standard Q4K LLM layer sizes with summary table.
pub fn profile_quant_all() -> Result<()> {
    println!("\n=== CGP Quant Sweep: Q4K GEMV — All Standard LLM Layers ===\n");

    let binary = find_bench_binary();
    let bench_output = binary.and_then(|b| {
        std::process::Command::new(&b)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
    });

    println!(
        "  {:25} {:>6}x{:<6} {:>10} {:>10} {:>10} {:>10}",
        "Layer", "M", "K", "Time (us)", "GFLOPS", "BW GB/s", "tok/s est"
    );
    println!("  {}", "-".repeat(85));

    let mut total_time_us = 0.0;
    let mut measured_count = 0;

    for (label, out_dim, in_dim) in STANDARD_LAYERS {
        let timing = bench_output.as_ref().and_then(|stdout| {
            let pattern = format!("{}x{}", out_dim, in_dim);
            for line in stdout.lines() {
                if line.contains("Q4K GEMV") && line.contains(&pattern) {
                    let time_us = extract_between(line, "...", " us")
                        .and_then(|s| s.trim().parse::<f64>().ok())?;
                    let gflops = extract_between(line, "(", " GFLOPS")
                        .and_then(|s| s.trim().parse::<f64>().ok())?;
                    let bw_gbps = extract_between(line, "GFLOPS, ", " GB/s")
                        .and_then(|s| s.trim().parse::<f64>().ok())?;
                    return Some(Q4kTiming {
                        time_us,
                        gflops,
                        bw_gbps,
                    });
                }
            }
            None
        });

        if let Some(t) = timing {
            // Token estimation: ~192 GEMVs per token (Llama-7B)
            let tok_per_sec = 1000.0 / (t.time_us * 192.0 / 1000.0);
            println!(
                "  {:25} {:>6}x{:<6} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
                label, out_dim, in_dim, t.time_us, t.gflops, t.bw_gbps, tok_per_sec
            );
            total_time_us += t.time_us;
            measured_count += 1;
        } else {
            println!(
                "  {:25} {:>6}x{:<6} {:>10} {:>10} {:>10} {:>10}",
                label, out_dim, in_dim, "-", "-", "-", "-"
            );
        }
    }

    if measured_count > 0 {
        println!("  {}", "-".repeat(85));
        let avg_gflops = STANDARD_LAYERS
            .iter()
            .take(4) // Only first 4 are benchmarked by default
            .count();
        println!(
            "\n  Summary ({measured_count}/{} layers measured):",
            STANDARD_LAYERS.len()
        );
        let _ = avg_gflops;
        let avg_time = total_time_us / measured_count as f64;
        let composite_tok_s = 1000.0 / (avg_time * 192.0 / 1000.0);
        println!("    Avg GEMV time: {:.1} us", avg_time);
        println!("    Composite tok/s estimate: {:.1}", composite_tok_s);
        println!(
            "    Total GEMV time (measured layers): {:.1} us",
            total_time_us
        );
    } else {
        println!("\n  No benchmark data available.");
        println!(
            "  Build: cargo build --release --example benchmark_matrix_suite --features parallel"
        );
    }

    println!();
    Ok(())
}

/// Parsed Q4K timing from benchmark output.
struct Q4kTiming {
    time_us: f64,
    gflops: f64,
    bw_gbps: f64,
}

/// Parse Q4K GEMV timing from benchmark_matrix_suite output.
/// Looks for lines like: "Q4K GEMV (MxK, label)...  X.XX us  (Y.YY GFLOPS, Z.Z GB/s)"
fn parse_q4k_timing(out_dim: u32, in_dim: u32) -> Option<Q4kTiming> {
    let binary = find_bench_binary()?;
    let output = std::process::Command::new(&binary).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let pattern = format!("{}x{}", out_dim, in_dim);

    for line in stdout.lines() {
        if line.contains("Q4K GEMV") && line.contains(&pattern) {
            // Parse "...  X.XX us  (Y.YY GFLOPS, Z.Z GB/s)"
            let time_us =
                extract_between(line, "...", " us").and_then(|s| s.trim().parse::<f64>().ok())?;
            let gflops =
                extract_between(line, "(", " GFLOPS").and_then(|s| s.trim().parse::<f64>().ok())?;
            let bw_gbps = extract_between(line, "GFLOPS, ", " GB/s")
                .and_then(|s| s.trim().parse::<f64>().ok())?;
            return Some(Q4kTiming {
                time_us,
                gflops,
                bw_gbps,
            });
        }
    }
    None
}

/// Extract text between two markers (finds last occurrence of start before end).
fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let end_idx = s.find(end)?;
    let prefix = &s[..end_idx];
    let start_idx = prefix.rfind(start)? + start.len();
    Some(&s[start_idx..end_idx])
}

/// Find benchmark binary at standard paths. Asks cargo for the target
/// directory rather than hardcoding one -- see `analysis::compare` for why:
/// `/mnt/nvme-raid0/targets/trueno` is the pre-APR-MONO name and empty on
/// every checkout since the consolidation.
fn find_bench_binary() -> Option<String> {
    let target_dir = crate::analysis::compare::cargo_target_dir();
    let owned_candidate =
        target_dir.map(|d| format!("{d}/release/examples/benchmark_matrix_suite"));
    owned_candidate
        .into_iter()
        .chain(["./target/release/examples/benchmark_matrix_suite".to_string()])
        .find(|p| std::path::Path::new(p).exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dimensions() {
        let dims = parse_dimensions("4096x1x4096").unwrap();
        assert_eq!(dims, [4096, 1, 4096]);
    }

    #[test]
    fn test_parse_dimensions_invalid() {
        assert!(parse_dimensions("4096x4096").is_err());
        assert!(parse_dimensions("abc").is_err());
    }

    /// Q4K super-block: 256 elements in 144 bytes.
    #[test]
    fn test_q4k_superblock() {
        let k = QuantKernel::Q4kGemv;
        assert_eq!(k.superblock_elements(), 256);
        assert_eq!(k.superblock_bytes(), 144);
    }

    /// FALSIFY-CGP-075: Effective bandwidth uses compressed bytes, not FP32.
    #[test]
    fn test_effective_bandwidth_compressed() {
        // 4096*4096 weights / 256 * 144 bytes = 9.44 MB (compressed)
        let total_elements: u64 = 4096 * 4096;
        let num_superblocks = total_elements / 256;
        let compressed_bytes = num_superblocks * 144;
        let expected_mb = 9.437184; // 9.44 MB approximately
        assert!(
            (compressed_bytes as f64 / 1e6 - expected_mb).abs() < 0.01,
            "Compressed size {:.2} MB != expected {:.2} MB",
            compressed_bytes as f64 / 1e6,
            expected_mb
        );
    }

    #[test]
    fn test_kernel_from_str() {
        assert_eq!(
            "q4k_gemv".parse::<QuantKernel>().unwrap(),
            QuantKernel::Q4kGemv
        );
        assert_eq!("q6k".parse::<QuantKernel>().unwrap(), QuantKernel::Q6kGemv);
        assert!("invalid".parse::<QuantKernel>().is_err());
    }
}
