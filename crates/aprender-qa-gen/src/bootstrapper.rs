//! Playbook Bootstrapper
//!
//! Generates architecture-aware playbook YAML from family contract constraints
//! and kernel profiles. Bootstrapped playbooks include targeted prompts that
//! stress-test the specific kernels each model architecture exercises.

use crate::kernel_profile::{
    profile_from_constraints, ArchConstraints, ArchSizeVariant, KernelProfile,
};
use serde::{Deserialize, Serialize};

/// Configuration for bootstrapping a playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Model family name (e.g., "qwen2", "llama")
    pub family: String,
    /// Size variant key (e.g., "1.5b", "7b")
    pub size_variant: String,
    /// HuggingFace repository ID
    pub hf_repo: String,
    /// Certification tier (e.g., "mvp", "smoke", "quick")
    pub tier: String,
    /// Optional kernel profile override (auto-derived if not provided)
    pub kernel_profile: Option<KernelProfile>,
}

/// A bootstrapped playbook representation ready for YAML serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedPlaybook {
    /// Playbook name
    pub name: String,
    /// Version
    pub version: String,
    /// Model configuration section
    pub model: BootstrappedModel,
    /// Test matrix section
    pub test_matrix: BootstrappedTestMatrix,
    /// Kernel profile metadata (documents which kernels are under test)
    pub kernel_profile: BootstrappedKernelProfile,
    /// Falsification gates
    pub falsification_gates: Vec<BootstrappedGate>,
    /// Differential test configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub differential_tests: Option<BootstrappedDifferential>,
    /// Profile CI assertions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_ci: Option<BootstrappedProfileCi>,
    /// Reference to kernel proof model (for dim-smoke tier)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_proof_ref: Option<String>,
}

/// Model section of bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedModel {
    /// HuggingFace repo
    pub hf_repo: String,
    /// Formats to test
    pub formats: Vec<String>,
    /// Quantizations
    pub quantizations: Vec<String>,
    /// Size category
    pub size_category: String,
    /// Expected hidden dim
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_hidden_dim: Option<u32>,
    /// Expected number of layers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_num_layers: Option<u32>,
    /// Expected number of attention heads
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_num_heads: Option<u32>,
    /// Expected number of KV heads
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_num_kv_heads: Option<u32>,
    /// Expected vocab size
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_vocab_size: Option<u32>,
    /// Expected intermediate dim
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_intermediate_dim: Option<u32>,
    /// Family identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Size variant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_variant: Option<String>,
}

/// Test matrix section of bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedTestMatrix {
    /// Modalities
    pub modalities: Vec<String>,
    /// Backends
    pub backends: Vec<String>,
    /// Scenario count per combination
    pub scenario_count: usize,
    /// Architecture-specific prompts
    pub prompts: Vec<String>,
}

/// Kernel profile metadata in the playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedKernelProfile {
    /// Family name
    pub family: String,
    /// List of kernel operation names
    pub kernel_ops: Vec<String>,
    /// Total prompt count
    pub prompt_count: usize,
    /// Whether long context is supported
    pub long_context: bool,
}

/// Falsification gate in bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedGate {
    /// Gate ID
    pub id: String,
    /// Description
    pub description: String,
    /// Condition
    pub condition: String,
    /// Severity
    pub severity: String,
}

/// Differential test config in bootstrapped playbook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedDifferential {
    /// Format validation
    pub format_validation: BootstrappedFormatValidation,
}

/// Format validation section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedFormatValidation {
    /// Enabled flag
    pub enabled: bool,
    /// Checks to run
    pub checks: Vec<String>,
}

/// Profile CI section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrappedProfileCi {
    /// Enabled flag
    pub enabled: bool,
    /// Minimum throughput (tokens/sec)
    pub min_throughput: f64,
    /// Max p99 latency (ms)
    pub max_p99_ms: f64,
}

/// Return the number of test scenarios appropriate for the given tier
fn scenario_count_for_tier(tier: &str) -> usize {
    match tier {
        "dim-smoke" | "smoke" => 1,
        "quick" => 5,
        "standard" => 10,
        "deep" => 50,
        // mvp and unknown tiers default to 3
        _ => 3,
    }
}

/// Return size-aware performance thresholds (min throughput tok/s, max p99 ms)
fn performance_thresholds(size_category: &str) -> (f64, f64) {
    match size_category {
        "tiny" => (50.0, 200.0),
        "small" => (30.0, 500.0),
        "medium" => (15.0, 1000.0),
        "large" => (5.0, 3000.0),
        "xlarge" => (2.0, 5000.0),
        _ => (1.0, 10000.0),
    }
}

/// Build the standard falsification gates G0 through G4
fn standard_gates() -> Vec<BootstrappedGate> {
    vec![
        BootstrappedGate {
            id: "G0".to_string(),
            description: "Model integrity (config.json matches tensor metadata)".to_string(),
            condition: "config_matches_tensors".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G1".to_string(),
            description: "Model loads successfully".to_string(),
            condition: "exit_code == 0".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G2".to_string(),
            description: "Basic inference produces output".to_string(),
            condition: "output.len() > 0".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G3".to_string(),
            description: "No crashes or panics".to_string(),
            condition: "!stderr.contains('panic')".to_string(),
            severity: "P0".to_string(),
        },
        BootstrappedGate {
            id: "G4".to_string(),
            description: "Output is not garbage (LAYOUT-002)".to_string(),
            condition: "!garbage_oracle.is_garbage(output)".to_string(),
            severity: "P0".to_string(),
        },
    ]
}

/// Bootstrap a playbook from architecture constraints.
///
/// Generates a complete playbook representation with architecture-specific
/// prompts and kernel profile metadata.
#[must_use]
pub fn bootstrap_playbook(
    config: &BootstrapConfig,
    constraints: &ArchConstraints,
    size_variant: &ArchSizeVariant,
    size_category: &str,
) -> BootstrappedPlaybook {
    let profile = config.kernel_profile.clone().unwrap_or_else(|| {
        profile_from_constraints(
            &config.family,
            constraints,
            size_variant.max_position_embeddings,
        )
    });

    let is_dim_smoke = config.tier == "dim-smoke";
    let is_smoke = config.tier == "smoke" || is_dim_smoke;
    let (min_throughput, max_p99_ms) = performance_thresholds(size_category);

    let modalities = if is_smoke {
        vec!["run".to_string()]
    } else {
        vec!["run".to_string(), "chat".to_string()]
    };

    let backends = if is_smoke {
        vec!["cpu".to_string()]
    } else {
        vec!["cpu".to_string(), "gpu".to_string()]
    };

    let formats = if is_dim_smoke {
        vec!["safetensors".to_string()]
    } else {
        vec![
            "gguf".to_string(),
            "safetensors".to_string(),
            "apr".to_string(),
        ]
    };

    let kernel_proof_ref = if is_dim_smoke {
        use crate::kernel_class::KernelClass;
        KernelClass::from_family(&config.family).map(|kc| kc.representative_model().to_string())
    } else {
        None
    };

    let model = BootstrappedModel {
        hf_repo: config.hf_repo.clone(),
        formats,
        quantizations: vec!["q4_k_m".to_string()],
        size_category: size_category.to_string(),
        expected_hidden_dim: Some(size_variant.hidden_dim),
        expected_num_layers: Some(size_variant.num_layers),
        expected_num_heads: size_variant.num_heads,
        expected_num_kv_heads: size_variant.num_kv_heads,
        expected_vocab_size: size_variant.vocab_size,
        expected_intermediate_dim: size_variant.intermediate_dim,
        family: Some(config.family.clone()),
        size_variant: Some(config.size_variant.clone()),
    };

    let kernel_profile_meta = BootstrappedKernelProfile {
        family: profile.family.clone(),
        kernel_ops: profile
            .kernel_ops
            .iter()
            .map(|op| op.serde_name().to_string())
            .collect(),
        prompt_count: profile.prompt_count(),
        long_context: profile.long_context,
    };

    let differential_tests = if is_smoke {
        None
    } else {
        Some(BootstrappedDifferential {
            format_validation: BootstrappedFormatValidation {
                enabled: true,
                checks: vec![
                    "dtype_mapping".to_string(),
                    "tensor_alignment".to_string(),
                    "header_integrity".to_string(),
                ],
            },
        })
    };

    let profile_ci = if is_smoke {
        None
    } else {
        Some(BootstrappedProfileCi {
            enabled: true,
            min_throughput,
            max_p99_ms,
        })
    };

    BootstrappedPlaybook {
        name: format!("{}-{}-{}", config.family, config.size_variant, config.tier),
        version: "1.0.0".to_string(),
        model,
        test_matrix: BootstrappedTestMatrix {
            modalities,
            backends,
            scenario_count: scenario_count_for_tier(&config.tier),
            prompts: profile.all_prompts(),
        },
        kernel_profile: kernel_profile_meta,
        falsification_gates: standard_gates(),
        differential_tests,
        profile_ci,
        kernel_proof_ref,
    }
}

/// Serialize a bootstrapped playbook to YAML.
///
/// # Errors
///
/// Returns an error string if YAML serialization fails.
pub fn to_yaml(playbook: &BootstrappedPlaybook) -> Result<String, String> {
    use std::fmt::Write;

    let mut yaml = String::new();
    yaml.push_str("# Auto-generated playbook - bootstrapped from family contract\n");
    let _ = writeln!(yaml, "# Family: {}", playbook.kernel_profile.family);
    let _ = writeln!(
        yaml,
        "# Kernel ops: {}",
        playbook.kernel_profile.kernel_ops.join(", ")
    );
    let _ = writeln!(
        yaml,
        "# Prompts: {} architecture-targeted prompts",
        playbook.kernel_profile.prompt_count
    );
    yaml.push('\n');

    let body =
        serde_yaml::to_string(playbook).map_err(|e| format!("YAML serialization error: {e}"))?;
    yaml.push_str(&body);

    Ok(yaml)
}

#[cfg(test)]
#[path = "bootstrapper_tests.rs"]
mod bootstrapper_tests;
