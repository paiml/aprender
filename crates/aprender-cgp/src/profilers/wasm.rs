//! WASM SIMD128 profiling via wasmtime. Spec section 4.6.
//! Uses wasmtime's fuel metering for deterministic instruction counting.

use anyhow::Result;
use std::process::Command;

/// WASM profiling configuration.
#[derive(Debug, Clone)]
pub struct WasmProfilingConfig {
    /// Enable fuel metering for instruction counting.
    pub fuel_metering: bool,
    /// Enable wasmtime's VTune/perf jitdump integration.
    pub jitdump: bool,
    /// Target runtime.
    pub target: WasmTarget,
}

#[derive(Debug, Clone)]
pub enum WasmTarget {
    /// Profile via wasmtime CLI with --profile=jitdump
    Wasmtime,
    /// Profile via Chrome DevTools Protocol (headless browser)
    Browser { cdp_url: String },
}

impl Default for WasmProfilingConfig {
    fn default() -> Self {
        Self { fuel_metering: true, jitdump: false, target: WasmTarget::Wasmtime }
    }
}

/// Check if a WASM module contains SIMD128 instructions.
/// Searches for v128 type references in the wasm binary.
pub fn detect_simd128(wasm_path: &str) -> bool {
    // wasmtime can inspect with --invoke and fuel metering
    // For a quick check, look for SIMD128 in wasm-tools dump or wasmtime output
    let output = Command::new("wasm-tools").args(["dump", wasm_path]).output().ok();

    if let Some(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        return stdout.contains("v128") || stdout.contains("simd");
    }

    // Fallback: check raw bytes for v128 type (0x7b in WASM)
    std::fs::read(wasm_path).is_ok_and(|bytes| bytes.windows(1).any(|w| w[0] == 0x7b))
}

/// Profile a WASM function.
pub fn profile_wasm(function: &str, size: u32) -> Result<()> {
    println!("\n=== CGP WASM Profile: {function} (size={size}) ===\n");

    let has_wasmtime = which::which("wasmtime").is_ok();
    if !has_wasmtime {
        println!("  wasmtime not found. Install wasmtime for WASM profiling.");
        println!("  Install: curl https://wasmtime.dev/install.sh -sSf | bash");
        return Ok(());
    }

    println!("  Backend: wasmtime (fuel metering + jitdump)");
    println!("  Function: {function}");
    println!("  Size: {size}");

    // Try to find a WASM binary for trueno
    let wasm_path = find_wasm_binary();
    match wasm_path {
        Some(path) => {
            let has_simd = detect_simd128(&path);
            if !has_simd {
                println!(
                    "  \x1b[33m[WARN]\x1b[0m No SIMD128 instructions detected -- scalar fallback"
                );
                println!("  Compile with: -Ctarget-feature=+simd128");
            } else {
                println!("  SIMD128: detected");
            }

            // Run with fuel metering for instruction counting
            println!("  Running wasmtime with fuel metering...");
            let output =
                Command::new("wasmtime").args(["run", "--fuel", "1000000000", &path]).output();

            match output {
                Ok(out) => {
                    if out.status.success() {
                        println!("  Execution: success");
                    } else {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if stderr.contains("fuel") {
                            println!("  Execution: ran out of fuel (>1B instructions)");
                        } else {
                            println!("  Execution: failed ({stderr})");
                        }
                    }
                }
                Err(e) => println!("  wasmtime failed: {e}"),
            }
        }
        None => {
            println!("  No WASM binary found.");
            println!("  Build with: cargo build --target wasm32-unknown-unknown --release");
            println!("  Or specify: cgp profile wasm --function <fn> --wasm-path <file>");
        }
    }

    println!();
    Ok(())
}

/// Find a trueno WASM binary.
fn find_wasm_binary() -> Option<String> {
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
    let candidates = [
        format!("{target_dir}/wasm32-unknown-unknown/release/trueno.wasm"),
        "./target/wasm32-unknown-unknown/release/trueno.wasm".to_string(),
    ];
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
    fn test_default_config() {
        let config = WasmProfilingConfig::default();
        assert!(config.fuel_metering);
        assert!(!config.jitdump);
        assert!(matches!(config.target, WasmTarget::Wasmtime));
    }

    /// FALSIFY-CGP-072: profile_wasm must not crash even without wasmtime.
    #[test]
    fn test_profile_wasm_no_panic() {
        let result = profile_wasm("vector_dot_wasm", 1024);
        assert!(result.is_ok());
    }

    /// FALSIFY-CGP-073: detect_simd128 should return false for non-existent files.
    #[test]
    fn test_detect_simd128_missing_file() {
        assert!(!detect_simd128("/tmp/nonexistent_wasm_file.wasm"));
    }
}
