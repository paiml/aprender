//! Runtime roofline model derived from contract YAML.
//!
//! Reads `roofline-model-v1.yaml` and provides performance ceiling
//! calculations. Consumer crates use this instead of hardcoded formulas.
//! Section 24: Deep Stack Integration.

use std::path::Path;

/// Performance ceilings derived from the roofline contract.
#[derive(Debug, Clone)]
pub struct RooflineCeiling {
    /// Model size in bytes
    pub model_bytes: f64,
    /// Bandwidth-limited ceiling (tokens/sec)
    pub bw_ceiling: f64,
    /// Compute-limited ceiling (tokens/sec)
    pub compute_ceiling: f64,
    /// Effective ceiling: min(bw, compute)
    pub throughput_ceiling: f64,
    /// Whether bandwidth-bound or compute-bound
    pub bottleneck: Bottleneck,
    /// Source contract ID
    pub contract_id: String,
}

/// Whether the workload is memory-bandwidth or compute bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Bandwidth,
    Compute,
}

/// Hardware profile for roofline calculation.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Effective memory bandwidth in GB/s
    pub bandwidth_gb_s: f64,
    /// Effective compute throughput in GFLOPS
    pub compute_gflops: f64,
    /// Operations per token (depends on model architecture)
    pub ops_per_token: f64,
}

impl HardwareProfile {
    /// Create a profile for Apple M-series (conservative estimates).
    pub fn apple_m_series() -> Self {
        Self {
            bandwidth_gb_s: 100.0,  // M1 Pro ~200, M2 Ultra ~800, use conservative
            compute_gflops: 1000.0, // GPU TFLOPS varies; conservative
            ops_per_token: 2.0,     // 2 FLOPs per parameter per token (forward pass)
        }
    }

    /// Create a profile for NVIDIA A100.
    pub fn nvidia_a100() -> Self {
        Self {
            bandwidth_gb_s: 2039.0,  // HBM2e bandwidth
            compute_gflops: 19500.0, // FP16 tensor core
            ops_per_token: 2.0,
        }
    }
}

/// Compute roofline ceilings from contract equations + hardware profile.
///
/// Implements the 4 equations from `roofline-model-v1.yaml`:
/// 1. `model_bytes = total_params × bits_per_weight / 8`
/// 2. `bw_ceiling = effective_bandwidth_GB_s / (model_bytes / 1e9)`
/// 3. `compute_ceiling = effective_GFLOPS / ops_per_token`
/// 4. `throughput <= min(bw_ceiling, compute_ceiling)`
pub fn compute_roofline(
    total_params: u64,
    bits_per_weight: u32,
    hw: &HardwareProfile,
) -> RooflineCeiling {
    // Equation 1: model_bytes
    #[allow(clippy::cast_precision_loss)]
    // model params fit within f64 mantissa for all real models
    let total_params_f = total_params as f64;
    let model_bytes = total_params_f * f64::from(bits_per_weight) / 8.0;

    // Equation 2: bw_ceiling (tokens/sec)
    let model_gb = model_bytes / 1e9;
    let bw_ceiling = if model_gb > 0.0 {
        hw.bandwidth_gb_s / model_gb
    } else {
        f64::INFINITY
    };

    // Equation 3: compute_ceiling (tokens/sec)
    let compute_ceiling = if hw.ops_per_token > 0.0 {
        hw.compute_gflops * 1e9 / (total_params_f * hw.ops_per_token)
    } else {
        f64::INFINITY
    };

    // Equation 4: throughput <= min(bw_ceiling, compute_ceiling)
    let throughput_ceiling = bw_ceiling.min(compute_ceiling);
    let bottleneck = if bw_ceiling < compute_ceiling {
        Bottleneck::Bandwidth
    } else {
        Bottleneck::Compute
    };

    RooflineCeiling {
        model_bytes,
        bw_ceiling,
        compute_ceiling,
        throughput_ceiling,
        bottleneck,
        contract_id: "roofline-model-v1".to_string(),
    }
}

/// Try to load roofline contract from a standard path.
/// Returns the contract description if found, None if not.
pub fn load_roofline_contract(contracts_dir: &Path) -> Option<String> {
    let path = contracts_dir.join("roofline-model-v1.yaml");
    if path.exists() {
        let content = std::fs::read_to_string(&path).ok()?;
        // Extract description
        for line in content.lines() {
            if let Some(desc) = line.trim().strip_prefix("description:") {
                return Some(desc.trim().trim_matches('"').to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roofline_7b_q4() {
        // 7B params, 4-bit quantized, on hypothetical hardware
        let hw = HardwareProfile {
            bandwidth_gb_s: 200.0,
            compute_gflops: 5000.0,
            ops_per_token: 2.0,
        };
        let r = compute_roofline(7_000_000_000, 4, &hw);
        assert!((r.model_bytes - 3_500_000_000.0).abs() < 1.0); // 7B * 4 / 8 = 3.5GB
        assert!(r.bw_ceiling > 0.0);
        assert!(r.compute_ceiling > 0.0);
        assert!((r.throughput_ceiling - r.bw_ceiling.min(r.compute_ceiling)).abs() < f64::EPSILON);
        assert_eq!(r.bottleneck, Bottleneck::Bandwidth); // Memory-bound at 4-bit
    }

    #[test]
    fn roofline_contract_id() {
        let hw = HardwareProfile::apple_m_series();
        let r = compute_roofline(1_000_000, 16, &hw);
        assert_eq!(r.contract_id, "roofline-model-v1");
    }

    #[test]
    fn bottleneck_classification() {
        // Tiny model on fast hardware = compute bound
        let hw = HardwareProfile {
            bandwidth_gb_s: 1000.0,
            compute_gflops: 1.0, // Very slow compute
            ops_per_token: 2.0,
        };
        let r = compute_roofline(1000, 32, &hw);
        assert_eq!(r.bottleneck, Bottleneck::Compute);
    }
}
