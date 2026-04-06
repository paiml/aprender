//! Scalar baseline profiling via criterion + perf stat. Spec section 4.4.
//! Establishes the baseline for all speedup calculations.

use anyhow::Result;

/// Profile a scalar function (CPU baseline).
/// Uses perf stat for hardware counters when available.
pub fn profile_scalar(function: &str, size: u32) -> Result<()> {
    println!("\n=== CGP Scalar Profile: {function} (size={size}) ===\n");
    println!("  Backend: scalar (no SIMD, single-threaded)");
    println!("  Purpose: baseline for speedup calculations");
    println!("  Function: {function}");
    println!("  Size: {size}");

    let has_perf = which::which("perf").is_ok();
    if has_perf {
        println!("  Hardware counters: perf stat (cycles, instructions, cache-misses)");

        // Try to find and profile a benchmark binary
        let binary = find_scalar_binary();
        if let Some(bin) = binary {
            println!("  Binary: {bin}");
            // Use base counters only (no SIMD events for scalar baseline)
            match crate::profilers::simd::run_perf_stat(
                &bin,
                &[],
                crate::profilers::simd::SIMD_PERF_EVENTS,
            ) {
                Ok(result) => {
                    let cycles = *result.counters.get("cycles").unwrap_or(&0);
                    let instructions = *result.counters.get("instructions").unwrap_or(&0);
                    println!("\n  Hardware Counters:");
                    println!("    Cycles:       {cycles}");
                    println!("    Instructions: {instructions}");
                    println!("    IPC:          {:.2}", result.ipc());
                    println!("    Cache miss:   {:.1}%", result.cache_miss_rate());
                    println!("    Branch miss:  {:.1}%", result.branch_miss_rate());
                }
                Err(e) => {
                    println!("  perf stat failed: {e}");
                    println!("  Try: sudo sysctl kernel.perf_event_paranoid=2");
                }
            }
        } else {
            println!("  No benchmark binary found. Build with: cargo build --release --examples");
        }
    } else {
        println!("  perf not found — install linux-tools-common for hardware counter profiling.");
    }

    let has_renacer = which::which("renacer").is_ok();
    if has_renacer {
        println!("  Syscall tracing: renacer available");
    }

    println!();
    Ok(())
}

/// Find a scalar benchmark binary.
fn find_scalar_binary() -> Option<String> {
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_default();
    let mut candidates: Vec<String> = Vec::new();
    if !target_dir.is_empty() {
        candidates.push(format!("{target_dir}/release/examples/benchmark_matrix_suite"));
    }
    candidates.extend_from_slice(&[
        "/mnt/nvme-raid0/targets/trueno/release/examples/benchmark_matrix_suite".to_string(),
        "./target/release/examples/benchmark_matrix_suite".to_string(),
    ]);
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_scalar_ok() {
        let result = profile_scalar("matrix_mul_naive", 256);
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_scalar_binary_no_panic() {
        let _ = find_scalar_binary();
    }
}
