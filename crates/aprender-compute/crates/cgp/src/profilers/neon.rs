//! ARM NEON SIMD profiling. Spec section 4.5.
//! Uses perf stat with ARM PMU counters on aarch64 hosts.
//! On x86 hosts, reports graceful error per FALSIFY-CGP-071.

use anyhow::Result;

/// ARM PMU counters for NEON profiling.
pub const ARM_PMU_EVENTS: &[&str] = &["INST_RETIRED", "CPU_CYCLES", "ASE_SPEC"];

/// Whether we're running natively on ARM.
pub fn is_native_arm() -> bool {
    cfg!(target_arch = "aarch64")
}

/// Profile a NEON function.
/// On x86: returns helpful message per FALSIFY-CGP-071.
/// On ARM: uses perf stat with ARM PMU counters.
pub fn profile_neon(function: &str, size: u32) -> Result<()> {
    println!("\n=== CGP NEON Profile: {function} (size={size}) ===\n");

    if !is_native_arm() {
        println!("  NEON not available -- use --cross-profile for QEMU-based analysis");
        println!("  This host is {arch}. NEON requires aarch64.", arch = std::env::consts::ARCH);
        println!("  Alternatives:");
        println!("    - Run on ARM host (Apple Silicon, Graviton, Ampere)");
        println!("    - Use QEMU user-mode: qemu-aarch64 -cpu max ./binary");
        println!("    - Use cgp profile simd --arch avx2 for x86 SIMD profiling");
        return Ok(());
    }

    // Native ARM path
    let has_perf = which::which("perf").is_ok();
    if !has_perf {
        println!("  perf not found. Install linux-tools for ARM PMU counters.");
        println!("  Function: {function}");
        println!("  ARM PMU events: {}", ARM_PMU_EVENTS.join(", "));
        return Ok(());
    }

    println!("  Backend: perf stat (ARM PMU)");
    println!("  Function: {function}");
    println!("  Size: {size}");
    println!("  Events: {}", ARM_PMU_EVENTS.join(", "));

    // On ARM, we'd run perf stat with ARM events. Same pattern as simd.rs.
    #[cfg(target_arch = "aarch64")]
    {
        use crate::profilers::simd;
        if let Some(binary) = find_arm_binary() {
            match simd::run_perf_stat(&binary, &[], ARM_PMU_EVENTS) {
                Ok(result) => {
                    let cycles = *result.counters.get("CPU_CYCLES").unwrap_or(&0);
                    let insts = *result.counters.get("INST_RETIRED").unwrap_or(&0);
                    let ase = *result.counters.get("ASE_SPEC").unwrap_or(&0);
                    println!("\n  ARM PMU Counters:");
                    println!("    CPU_CYCLES:   {cycles}");
                    println!("    INST_RETIRED: {insts}");
                    println!("    ASE_SPEC:     {ase} (SIMD/FP instructions)");
                    if insts > 0 {
                        let neon_pct = ase as f64 / insts as f64 * 100.0;
                        println!("    NEON util:    {neon_pct:.1}%");
                    }
                }
                Err(e) => println!("  perf stat failed: {e}"),
            }
        } else {
            println!("  No benchmark binary found for ARM target.");
        }
    }

    println!();
    Ok(())
}

/// Find ARM benchmark binary (aarch64 only).
#[cfg(target_arch = "aarch64")]
fn find_arm_binary() -> Option<String> {
    let candidates = [
        "./target/release/examples/benchmark_matrix_suite",
        "./target/aarch64-unknown-linux-gnu/release/examples/benchmark_matrix_suite",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CGP-071: On x86 host, NEON should not be marked as native.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_neon_not_native_on_x86() {
        assert!(!is_native_arm());
    }

    #[test]
    fn test_arm_events_defined() {
        assert_eq!(ARM_PMU_EVENTS.len(), 3);
    }

    /// FALSIFY-CGP-071: profile_neon must not crash on x86.
    #[test]
    fn test_profile_neon_graceful_on_any_arch() {
        let result = profile_neon("vector_add_neon", 1024);
        assert!(result.is_ok());
    }
}
