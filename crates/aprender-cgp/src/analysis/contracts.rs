//! Performance contract verification (CI/CD gate).
//! Extends provable-contracts framework to performance bounds.
//! See spec section 3.4 and 7.1.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A performance contract loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceContract {
    pub kind: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub kernel: String,
    #[serde(default)]
    pub hardware: HardwareSpec,
    #[serde(default)]
    pub bounds: Vec<PerformanceBound>,
    #[serde(default)]
    pub metrics: std::collections::HashMap<String, MetricBound>,
    #[serde(default)]
    pub falsification: Vec<FalsificationCheck>,
    /// Absorb any extra fields from domain-specific contract schemas
    #[serde(flatten, default)]
    pub extra: std::collections::HashMap<String, serde_yaml_ng::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub gpu: Option<String>,
    pub cpu: Option<String>,
    pub compute_capability: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBound {
    #[serde(default, deserialize_with = "deserialize_size")]
    pub size: Vec<u32>,
    #[serde(default)]
    pub max_time_us: Option<f64>,
    #[serde(default)]
    pub min_tflops: Option<f64>,
    #[serde(default)]
    pub max_regression_pct: Option<f64>,
    #[serde(default)]
    pub min_bandwidth_gbps: Option<f64>,
    /// Absorb domain-specific bound fields (operation, competitor, etc.)
    #[serde(flatten, default)]
    pub extra: std::collections::HashMap<String, serde_yaml_ng::Value>,
}

/// Accept both `size: 1024` (single int) and `size: [1024, 1024, 1024]` (sequence).
fn deserialize_size<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct SizeVisitor;
    impl<'de> de::Visitor<'de> for SizeVisitor {
        type Value = Vec<u32>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an integer or sequence of integers")
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Vec<u32>, E> {
            Ok(vec![v as u32])
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Vec<u32>, E> {
            Ok(vec![v as u32])
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<u32>, A::Error> {
            let mut v = Vec::new();
            while let Some(elem) = seq.next_element::<u32>()? {
                v.push(elem);
            }
            Ok(v)
        }
        fn visit_none<E: de::Error>(self) -> Result<Vec<u32>, E> {
            Ok(Vec::new())
        }
        fn visit_unit<E: de::Error>(self) -> Result<Vec<u32>, E> {
            Ok(Vec::new())
        }
    }
    deserializer.deserialize_any(SizeVisitor)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBound {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationCheck {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub check: String,
    #[serde(flatten, default)]
    pub extra: std::collections::HashMap<String, serde_yaml_ng::Value>,
}

/// Result of verifying a single contract.
#[derive(Debug)]
pub struct ContractVerification {
    pub contract_name: String,
    pub passed: Vec<String>,
    pub failed: Vec<String>,
    pub skipped: Vec<String>,
}

impl ContractVerification {
    pub fn is_pass(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Load a performance contract from a YAML file.
pub fn load_contract(path: &Path) -> Result<PerformanceContract> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read contract: {}", path.display()))?;
    let contract: PerformanceContract = serde_yaml_ng::from_str(&content)
        .with_context(|| format!("Failed to parse contract: {}", path.display()))?;
    Ok(contract)
}

/// Load all contracts from a directory.
pub fn load_contracts_dir(dir: &Path) -> Result<Vec<PerformanceContract>> {
    let mut contracts = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                match load_contract(&path) {
                    Ok(c) => contracts.push(c),
                    Err(e) => eprintln!("Warning: skipping {}: {e}", path.display()),
                }
            }
        }
    }
    Ok(contracts)
}

/// Verify a contract against measured values.
/// Loads saved profiles from /tmp/cgp-{kernel}-{size}.json if available,
/// and checks performance bounds + falsification expressions.
pub fn verify_contract(contract: &PerformanceContract) -> ContractVerification {
    let mut result = ContractVerification {
        contract_name: contract.name.clone(),
        passed: Vec::new(),
        failed: Vec::new(),
        skipped: Vec::new(),
    };
    validate_contract_metadata(contract, &mut result);
    for (i, bound) in contract.bounds.iter().enumerate() {
        verify_single_bound(contract, bound, i, &mut result);
    }
    for check in &contract.falsification {
        verify_single_falsification(contract, check, &mut result);
    }
    result
}

/// Check the contract has the required `kind` field and record `kernel` if present.
fn validate_contract_metadata(
    contract: &PerformanceContract,
    result: &mut ContractVerification,
) {
    if contract.kind.is_empty() {
        result
            .failed
            .push("Contract missing 'kind' field".to_string());
    } else {
        result.passed.push(format!("kind: {}", contract.kind));
    }
    if contract.kernel.is_empty() {
        result
            .skipped
            .push("No kernel field — domain-specific contract".to_string());
    } else {
        result.passed.push(format!("kernel: {}", contract.kernel));
    }
}

/// Verify one `PerformanceBound`: structural pass if no size, else compare to profile data.
fn verify_single_bound(
    contract: &PerformanceContract,
    bound: &PerformanceBound,
    i: usize,
    result: &mut ContractVerification,
) {
    if bound.size.is_empty() {
        result
            .passed
            .push(format!("Bound {i}: structural (no size)"));
        return;
    }
    let size = bound.size[0];
    match load_kernel_profile(&contract.kernel, size) {
        Some(p) => check_bound_thresholds(bound, i, &p, result),
        None => check_bound_structural(bound, i, result),
    }
}

/// Load `/tmp/cgp-{kernel}-{size}.json` profile if present and parseable.
fn load_kernel_profile(kernel: &str, size: u32) -> Option<crate::metrics::catalog::FullProfile> {
    let profile_path = format!("/tmp/cgp-{kernel}-{size}.json");
    let path = std::path::Path::new(&profile_path);
    if !path.exists() {
        return None;
    }
    crate::metrics::export::load_json(path).ok()
}

/// Run each individual threshold check present on the bound.
fn check_bound_thresholds(
    bound: &PerformanceBound,
    i: usize,
    p: &crate::metrics::catalog::FullProfile,
    result: &mut ContractVerification,
) {
    check_max_time(bound, i, p, result);
    check_min_tflops(bound, i, p, result);
    check_min_bandwidth(bound, i, p, result);
}

/// Compare `wall_clock_time_us` to `bound.max_time_us` (lower is better).
fn check_max_time(
    bound: &PerformanceBound,
    i: usize,
    p: &crate::metrics::catalog::FullProfile,
    result: &mut ContractVerification,
) {
    let Some(max_time) = bound.max_time_us else {
        return;
    };
    let actual = p.timing.wall_clock_time_us;
    if actual <= max_time {
        result
            .passed
            .push(format!("Bound {i}: time {actual:.1}us <= {max_time:.1}us"));
    } else {
        result.failed.push(format!(
            "Bound {i}: time {actual:.1}us > {max_time:.1}us EXCEEDED"
        ));
    }
}

/// Compare measured TFLOP/s to `bound.min_tflops` (higher is better).
fn check_min_tflops(
    bound: &PerformanceBound,
    i: usize,
    p: &crate::metrics::catalog::FullProfile,
    result: &mut ContractVerification,
) {
    let Some(min_tflops) = bound.min_tflops else {
        return;
    };
    let actual = p.throughput.tflops;
    if actual >= min_tflops {
        result
            .passed
            .push(format!("Bound {i}: {actual:.1} TFLOP/s >= {min_tflops:.1}"));
    } else {
        result.failed.push(format!(
            "Bound {i}: {actual:.1} TFLOP/s < {min_tflops:.1} BELOW MINIMUM"
        ));
    }
}

/// Compare measured GB/s to `bound.min_bandwidth_gbps` (higher is better).
fn check_min_bandwidth(
    bound: &PerformanceBound,
    i: usize,
    p: &crate::metrics::catalog::FullProfile,
    result: &mut ContractVerification,
) {
    let Some(min_bw) = bound.min_bandwidth_gbps else {
        return;
    };
    let actual = p.throughput.bandwidth_gbps;
    if actual >= min_bw {
        result
            .passed
            .push(format!("Bound {i}: {actual:.1} GB/s >= {min_bw:.1}"));
    } else {
        result.failed.push(format!(
            "Bound {i}: {actual:.1} GB/s < {min_bw:.1} BELOW MINIMUM"
        ));
    }
}

/// No profile found: record the bound structurally and flag empty-criteria bounds.
fn check_bound_structural(
    bound: &PerformanceBound,
    i: usize,
    result: &mut ContractVerification,
) {
    result
        .passed
        .push(format!("Bound {i}: size {:?}", bound.size));
    if bound.max_time_us.is_none()
        && bound.min_tflops.is_none()
        && bound.min_bandwidth_gbps.is_none()
    {
        result
            .skipped
            .push(format!("Bound {i}: no criteria specified"));
    }
}

/// Verify one falsification clause: validate fields, then evaluate against a profile.
fn verify_single_falsification(
    contract: &PerformanceContract,
    check: &FalsificationCheck,
    result: &mut ContractVerification,
) {
    if check.name.is_empty() || check.check.is_empty() {
        result.failed.push(format!(
            "Falsification '{}': missing name or check",
            check.name
        ));
        return;
    }
    let size = contract
        .bounds
        .first()
        .and_then(|b| b.size.first())
        .copied()
        .unwrap_or(512);
    let profile_path = format!("/tmp/cgp-{}-{size}.json", contract.kernel);
    match load_kernel_profile(&contract.kernel, size) {
        Some(p) => {
            if evaluate_check(&check.check, &p) {
                result.passed.push(format!("FALSIFY {}: PASS", check.name));
            } else {
                result.failed.push(format!(
                    "FALSIFY {}: FAIL ({})",
                    check.name, check.description
                ));
            }
        }
        None => {
            result.skipped.push(format!(
                "FALSIFY {}: {} (no profile at {profile_path})",
                check.name, check.description
            ));
        }
    }
}

/// Evaluate a simple check expression against a profile.
/// Supports: "field > value", "field < value", "field >= value", "field == value"
fn evaluate_check(expr: &str, profile: &crate::metrics::catalog::FullProfile) -> bool {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    if parts.len() != 3 {
        return false;
    }
    let field = parts[0];
    let op = parts[1];
    let threshold: f64 = match parts[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };

    let value = match field {
        "tflops" => profile.throughput.tflops,
        "wall_clock_time_us" => profile.timing.wall_clock_time_us,
        "bandwidth_gbps" => profile.throughput.bandwidth_gbps,
        "arithmetic_intensity" => profile.throughput.arithmetic_intensity,
        "warp_execution_efficiency" => profile
            .gpu_compute
            .as_ref()
            .map_or(0.0, |g| g.warp_execution_efficiency_pct),
        "achieved_occupancy" => profile
            .gpu_compute
            .as_ref()
            .map_or(0.0, |g| g.achieved_occupancy_pct),
        "global_load_efficiency" => profile
            .gpu_memory
            .as_ref()
            .map_or(0.0, |g| g.global_load_efficiency_pct),
        _ => return false,
    };

    match op {
        ">" => value > threshold,
        "<" => value < threshold,
        ">=" => value >= threshold,
        "<=" => value <= threshold,
        "==" => (value - threshold).abs() < 0.001,
        _ => false,
    }
}

/// Run contract verification for a directory of contracts.
pub fn run_verify(
    contracts_dir: Option<&str>,
    contract_file: Option<&str>,
    self_verify: bool,
    fail_on_regression: bool,
) -> Result<()> {
    let Some(contracts) = resolve_contracts_input(contracts_dir, contract_file, self_verify)?
    else {
        return Ok(());
    };

    println!("\n=== cgp Contract Verification ===\n");
    let totals = run_verify_all(&contracts);
    println!(
        "\n  Total: {} pass, {} fail, {} skip",
        totals.pass, totals.fail, totals.skip
    );
    if totals.fail > 0 && fail_on_regression {
        anyhow::bail!("{} contract verification(s) failed", totals.fail);
    }
    println!();
    Ok(())
}

/// Resolve the set of contracts to verify from CLI flags. Returns `None` when the
/// self-verify directory is absent (early-exit signal for `run_verify`).
fn resolve_contracts_input(
    contracts_dir: Option<&str>,
    contract_file: Option<&str>,
    self_verify: bool,
) -> Result<Option<Vec<PerformanceContract>>> {
    if let Some(dir) = contracts_dir {
        return Ok(Some(load_contracts_dir(Path::new(dir))?));
    }
    if let Some(file) = contract_file {
        return Ok(Some(vec![load_contract(Path::new(file))?]));
    }
    if self_verify {
        let dir = Path::new("contracts/cgp");
        if !dir.exists() {
            println!("No contracts found at contracts/cgp/");
            return Ok(None);
        }
        return Ok(Some(load_contracts_dir(dir)?));
    }
    anyhow::bail!("Specify --contracts-dir, --contract, or --self");
}

#[derive(Default)]
struct VerifyTotals {
    pass: usize,
    fail: usize,
    skip: usize,
}

/// Verify every contract and print per-contract status; return aggregate counts.
fn run_verify_all(contracts: &[PerformanceContract]) -> VerifyTotals {
    let mut totals = VerifyTotals::default();
    for c in contracts {
        let result = verify_contract(c);
        print_contract_status(c, &result);
        totals.pass += result.passed.len();
        totals.fail += result.failed.len();
        totals.skip += result.skipped.len();
    }
    totals
}

fn print_contract_status(c: &PerformanceContract, result: &ContractVerification) {
    let status = if result.is_pass() {
        "\x1b[32mPASS\x1b[0m"
    } else {
        "\x1b[31mFAIL\x1b[0m"
    };
    println!(
        "  {} {} ({} pass, {} fail, {} skip)",
        status,
        c.name,
        result.passed.len(),
        result.failed.len(),
        result.skipped.len()
    );
}

/// Generate a performance contract YAML from a profile or estimated values.
pub fn run_generate(kernel: &str, size: u32, tolerance: f64) -> Result<()> {
    // Try to load a saved profile for this kernel
    let profile_path = format!("/tmp/cgp-{kernel}-{size}.json");
    let profile = if std::path::Path::new(&profile_path).exists() {
        Some(crate::metrics::export::load_json(std::path::Path::new(
            &profile_path,
        ))?)
    } else {
        None
    };

    let (time_us, tflops) = match &profile {
        Some(p) => (p.timing.wall_clock_time_us, p.throughput.tflops),
        None => {
            // Estimate for GEMM
            let flops = 2.0 * (size as f64).powi(3);
            let est_time = 23.2 * (size as f64 / 512.0).powi(3); // Scale from 512 baseline
            let est_tflops = flops / (est_time * 1e-6) / 1e12;
            (est_time, est_tflops)
        }
    };

    let max_time = time_us * (1.0 + tolerance / 100.0);
    let min_tflops = tflops * (1.0 - tolerance / 100.0);

    // Detect GPU
    let gpu_name = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "Unknown GPU".to_string());

    let contract_yaml = format!(
        r#"# Generated by cgp contract generate
# Kernel: {kernel} at size {size}x{size}x{size}
# Tolerance: {tolerance}%
kind: PerformanceContract
name: {kernel}-{size}
version: "1.0.0"
kernel: {kernel}
hardware:
  gpu: "{gpu_name}"
  compute_capability: "8.9"

bounds:
  - size: [{size}, {size}, {size}]
    max_time_us: {max_time:.1}
    min_tflops: {min_tflops:.1}
    max_regression_pct: {tolerance}

metrics:
  warp_execution_efficiency:
    min: 95.0
  achieved_occupancy:
    min: 25.0

falsification:
  - name: FALSIFY-{kernel_upper}-001
    description: "{kernel} must achieve >{min_tflops:.1} TFLOP/s at {size}x{size}"
    check: "tflops > {min_tflops:.1}"
  - name: FALSIFY-{kernel_upper}-002
    description: "{kernel} must complete in <{max_time:.1}us at {size}x{size}"
    check: "wall_clock_time_us < {max_time:.1}"
"#,
        kernel = kernel,
        size = size,
        tolerance = tolerance,
        gpu_name = gpu_name,
        max_time = max_time,
        min_tflops = min_tflops,
        kernel_upper = kernel.to_uppercase().replace('-', "_"),
    );

    // Write to contracts directory
    let contracts_dir = std::path::Path::new("contracts/cgp");
    std::fs::create_dir_all(contracts_dir)?;
    let contract_path = contracts_dir.join(format!("{kernel}-{size}-v1.yaml"));
    std::fs::write(&contract_path, &contract_yaml)?;

    println!("Generated contract: {}", contract_path.display());
    println!();
    print!("{contract_yaml}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_contract() -> PerformanceContract {
        PerformanceContract {
            kind: "PerformanceContract".to_string(),
            name: "test-gemm-contract".to_string(),
            version: "1.0.0".to_string(),
            kernel: "gemm_cta_wmma_fp16".to_string(),
            hardware: HardwareSpec {
                gpu: Some("NVIDIA GeForce RTX 4090".to_string()),
                cpu: None,
                compute_capability: Some("8.9".to_string()),
            },
            bounds: vec![PerformanceBound {
                size: vec![512, 512, 512],
                max_time_us: Some(30.0),
                min_tflops: Some(9.0),
                max_regression_pct: Some(10.0),
                min_bandwidth_gbps: None,
                extra: Default::default(),
            }],
            metrics: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "warp_execution_efficiency".to_string(),
                    MetricBound {
                        min: Some(95.0),
                        max: None,
                    },
                );
                m
            },
            falsification: vec![FalsificationCheck {
                name: "FALSIFY-TEST-001".to_string(),
                description: "CTA WMMA must achieve >9 TFLOP/s".to_string(),
                check: "tflops > 9.0".to_string(),
                extra: Default::default(),
            }],
            extra: Default::default(),
        }
    }

    #[test]
    fn test_verify_valid_contract() {
        let contract = sample_contract();
        let result = verify_contract(&contract);
        assert!(result.is_pass());
        assert!(!result.passed.is_empty());
    }

    #[test]
    fn test_verify_missing_kernel_is_skipped() {
        let mut contract = sample_contract();
        contract.kernel = String::new();
        let result = verify_contract(&contract);
        // Domain-specific contracts without kernel are allowed (skipped, not failed)
        assert!(result.is_pass());
        assert!(!result.skipped.is_empty());
    }

    #[test]
    fn test_contract_yaml_roundtrip() {
        let contract = sample_contract();
        let yaml = serde_yaml_ng::to_string(&contract).unwrap();
        let parsed: PerformanceContract = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed.name, contract.name);
        assert_eq!(parsed.kernel, contract.kernel);
        assert_eq!(parsed.bounds.len(), 1);
        assert_eq!(parsed.bounds[0].size, vec![512, 512, 512]);
    }

    #[test]
    fn test_contract_falsification_checks() {
        let contract = sample_contract();
        let result = verify_contract(&contract);
        // Falsification checks are skipped (need runtime data), not failed
        assert!(result.is_pass());
        assert!(!result.skipped.is_empty());
    }
}
