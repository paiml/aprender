//! Mixture of Experts (MoE) Construction from Dense Models (GH-445)
//!
//! Constructs MoE architectures by combining multiple dense models into
//! a single sparse model with learned routing. Each dense model contributes
//! expert FFN weights, and a gating network learns to route tokens to the
//! most relevant experts.
//!
//! # Key Design Decisions
//!
//! - **Round-robin assignment**: Experts are assigned to layers in a balanced
//!   round-robin pattern across source models to ensure diversity
//! - **Load balancing**: Measured via coefficient of variation to detect
//!   routing collapse (all tokens routed to same expert)
//! - **Router initialization**: Supports random, uniform, and balanced
//!   strategies to prevent early routing bias
//!
//! # References
//!
//! - Shazeer et al. 2017: "Outrageously Large Neural Networks: The
//!   Sparsely-Gated Mixture-of-Experts Layer"
//! - Fedus et al. 2022: "Switch Transformers: Scaling to Trillion
//!   Parameter Models with Simple and Efficient Sparsity"
//! - Zhou et al. 2022: "Mixture-of-Experts with Expert Choice Routing"
//!
//! # Toyota Way Principles
//!
//! - **Heijunka**: Load-balanced expert assignment prevents hotspots
//! - **Jidoka**: Validation stops construction on invalid configurations
//! - **Muda Elimination**: Only activate top-k experts per token

use crate::error::{AprenderError, Result};

/// Routing method for dispatching tokens to experts.
///
/// Each method trades off between load balance and quality:
/// - `TopK`: Best quality, potential load imbalance
/// - `SwitchTransformer`: Good balance with auxiliary loss
/// - `ExpertChoice`: Perfect balance, experts choose their tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMethod {
    /// Standard top-k gating (Shazeer et al. 2017).
    /// Each token selects its top-k experts by gate score.
    TopK,
    /// Switch Transformer routing (Fedus et al. 2022).
    /// Each token routed to exactly one expert with capacity factor.
    SwitchTransformer,
    /// Expert Choice routing (Zhou et al. 2022).
    /// Each expert selects its top-k tokens, guaranteeing perfect balance.
    ExpertChoice,
}

impl Default for RoutingMethod {
    fn default() -> Self {
        Self::TopK
    }
}

/// Router weight initialization strategy.
///
/// Controls how the gating network weights are initialized before
/// training. Proper initialization prevents early routing collapse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterInit {
    /// Random initialization from uniform distribution.
    /// Each weight sampled from U(-scale, scale) where scale = 1/sqrt(hidden_dim).
    Random,
    /// Uniform initialization (all weights equal).
    /// Ensures equal probability for all experts at start.
    Uniform,
    /// Balanced initialization with small perturbation.
    /// Base uniform value with small noise to break symmetry.
    Balanced,
}

impl Default for RouterInit {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Configuration for MoE construction from dense models.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total number of experts in the MoE layer.
    pub num_experts: usize,
    /// Number of experts activated per token (default: 2).
    pub num_experts_per_tok: usize,
    /// Routing method for token-to-expert dispatch.
    pub routing_method: RoutingMethod,
    /// Hidden dimension for the gating network.
    /// If `None`, uses the model's hidden dimension directly.
    pub gate_hidden_dim: Option<usize>,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            num_experts_per_tok: 2,
            routing_method: RoutingMethod::default(),
            gate_hidden_dim: None,
        }
    }
}

impl MoeConfig {
    /// Validate configuration constraints.
    ///
    /// # Errors
    ///
    /// Returns `AprenderError::FormatError` if:
    /// - `num_experts` is zero
    /// - `num_experts_per_tok` is zero or exceeds `num_experts`
    /// - `gate_hidden_dim` is `Some(0)`
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(AprenderError::FormatError {
                message: "num_experts must be > 0".to_string(),
            });
        }
        if self.num_experts_per_tok == 0 {
            return Err(AprenderError::FormatError {
                message: "num_experts_per_tok must be > 0".to_string(),
            });
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(AprenderError::FormatError {
                message: format!(
                    "num_experts_per_tok ({}) must not exceed num_experts ({})",
                    self.num_experts_per_tok, self.num_experts
                ),
            });
        }
        if self.gate_hidden_dim == Some(0) {
            return Err(AprenderError::FormatError {
                message: "gate_hidden_dim must be > 0 when specified".to_string(),
            });
        }
        Ok(())
    }
}

/// Assignment of a single expert within a layer.
///
/// Maps an expert slot to its source dense model and layer,
/// enabling reconstruction of the MoE from original weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertAssignment {
    /// Index of this expert within the MoE layer (0..num_experts).
    pub expert_index: usize,
    /// Index of the source dense model providing this expert's weights.
    pub source_model: usize,
    /// Layer index within the source model to extract weights from.
    pub source_layer: usize,
}

/// Complete construction plan for building an MoE from dense models.
///
/// Contains per-layer expert assignments and router initialization
/// strategy. Used as a blueprint before actual weight extraction.
#[derive(Debug, Clone)]
pub struct MoeConstructionPlan {
    /// Per-layer expert assignments. `assignments[layer][expert]`.
    pub assignments: Vec<Vec<ExpertAssignment>>,
    /// Number of transformer layers in the MoE model.
    pub num_layers: usize,
    /// Router weight initialization strategy.
    pub router_init: RouterInit,
}

/// Summary report of an MoE construction plan.
#[derive(Debug, Clone)]
pub struct MoeReport {
    /// Total number of experts per layer.
    pub num_experts: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Load balance score (0.0 = perfectly balanced, higher = worse).
    pub load_balance: f64,
    /// Estimated total parameter count across all experts and routers.
    pub total_params_estimate: u64,
}

/// Plan MoE construction by assigning experts from source models.
///
/// Creates a round-robin assignment where experts are distributed
/// evenly across source models. For each layer, expert `i` is assigned
/// to model `i % num_models`, using the corresponding layer from that
/// source model.
///
/// # Arguments
///
/// * `num_models` - Number of source dense models
/// * `num_layers` - Number of transformer layers in the output MoE
/// * `config` - MoE configuration (validated internally)
///
/// # Errors
///
/// Returns error if `num_models` is zero, `num_layers` is zero, or
/// `config` fails validation.
///
/// # Example
///
/// ```rust,ignore
/// use aprender::online::moe_construction::{MoeConfig, plan_moe_construction};
///
/// let config = MoeConfig { num_experts: 8, ..Default::default() };
/// let plan = plan_moe_construction(4, 32, &config)?;
/// assert_eq!(plan.assignments.len(), 32);
/// assert_eq!(plan.assignments[0].len(), 8);
/// ```
pub fn plan_moe_construction(
    num_models: usize,
    num_layers: usize,
    config: &MoeConfig,
) -> Result<MoeConstructionPlan> {
    if num_models == 0 {
        return Err(AprenderError::FormatError {
            message: "num_models must be > 0".to_string(),
        });
    }
    if num_layers == 0 {
        return Err(AprenderError::FormatError {
            message: "num_layers must be > 0".to_string(),
        });
    }
    config.validate()?;

    let mut assignments = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let mut layer_assignments = Vec::with_capacity(config.num_experts);

        for expert_idx in 0..config.num_experts {
            // Round-robin: distribute experts across source models
            let source_model = expert_idx % num_models;
            let source_layer = layer_idx;

            layer_assignments.push(ExpertAssignment {
                expert_index: expert_idx,
                source_model,
                source_layer,
            });
        }

        assignments.push(layer_assignments);
    }

    Ok(MoeConstructionPlan {
        assignments,
        num_layers,
        router_init: RouterInit::default(),
    })
}

/// Compute initial gate weights for the router network.
///
/// The gate projects from `hidden_dim` to `num_experts`, producing
/// logits that are softmaxed to get routing probabilities.
///
/// # Arguments
///
/// * `hidden_dim` - Input hidden dimension of the transformer
/// * `num_experts` - Number of experts to route to
/// * `init` - Initialization strategy
///
/// # Returns
///
/// Flattened weight matrix of shape `[hidden_dim, num_experts]` in
/// row-major order, consistent with LAYOUT-002.
#[must_use]
pub fn compute_gate_weights(hidden_dim: usize, num_experts: usize, init: RouterInit) -> Vec<f64> {
    let total = hidden_dim * num_experts;
    if total == 0 {
        return vec![];
    }

    match init {
        RouterInit::Random => {
            // Xavier/Glorot-style scale: 1/sqrt(hidden_dim)
            let scale = 1.0 / (hidden_dim as f64).sqrt();
            // Deterministic pseudo-random using a simple LCG seeded from indices.
            // This avoids pulling in rand as a dependency.
            let mut weights = Vec::with_capacity(total);
            let mut state: u64 = 0x5DEE_CE66_D1A4_F681;
            for _ in 0..total {
                // LCG step (Knuth MMIX parameters)
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                // Map to [-scale, scale]
                let frac = (state >> 33) as f64 / (u32::MAX as f64);
                weights.push((frac * 2.0 - 1.0) * scale);
            }
            weights
        }
        RouterInit::Uniform => {
            // Equal weight for all experts: 1/num_experts per output
            let val = 1.0 / num_experts as f64;
            vec![val; total]
        }
        RouterInit::Balanced => {
            // Uniform base with small symmetry-breaking perturbation.
            // Perturbation magnitude: 0.01 / sqrt(hidden_dim)
            let base = 1.0 / num_experts as f64;
            let perturbation_scale = 0.01 / (hidden_dim as f64).sqrt();
            let mut weights = Vec::with_capacity(total);
            let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
            for _ in 0..total {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let frac = (state >> 33) as f64 / (u32::MAX as f64);
                let noise = (frac * 2.0 - 1.0) * perturbation_scale;
                weights.push(base + noise);
            }
            weights
        }
    }
}

/// Compute load balance across expert assignments.
///
/// Measures how evenly source models are utilized across all layers.
/// Uses coefficient of variation (std_dev / mean) of per-model
/// assignment counts. Returns 0.0 for perfectly balanced plans.
///
/// # Arguments
///
/// * `assignments` - Per-layer expert assignments
///
/// # Returns
///
/// Load balance score where 0.0 = perfectly balanced and higher
/// values indicate worse imbalance.
#[must_use]
pub fn compute_expert_load_balance(assignments: &[Vec<ExpertAssignment>]) -> f64 {
    if assignments.is_empty() {
        return 0.0;
    }

    // Count how many times each source model is used across all layers
    let max_model = assignments
        .iter()
        .flat_map(|layer| layer.iter())
        .map(|a| a.source_model)
        .max()
        .unwrap_or(0);

    let num_models = max_model + 1;
    let mut counts = vec![0u64; num_models];

    for layer in assignments {
        for assignment in layer {
            counts[assignment.source_model] += 1;
        }
    }

    // Coefficient of variation: std_dev / mean
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let mean = total as f64 / num_models as f64;
    if mean == 0.0 {
        return 0.0;
    }

    let variance = counts
        .iter()
        .map(|&c| {
            let diff = c as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / num_models as f64;

    variance.sqrt() / mean
}

impl MoeConstructionPlan {
    /// Generate a summary report of this construction plan.
    ///
    /// Estimates total parameters assuming each expert has
    /// `3 * hidden_dim * intermediate_dim` parameters (gate + up + down
    /// projections) plus router parameters.
    ///
    /// # Arguments
    ///
    /// * `hidden_dim` - Hidden dimension of the transformer
    /// * `intermediate_dim` - Intermediate (FFN) dimension
    /// * `num_experts` - Number of experts per layer
    #[must_use]
    pub fn report(
        &self,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_experts: usize,
    ) -> MoeReport {
        let load_balance = compute_expert_load_balance(&self.assignments);

        // Expert FFN params: gate_proj + up_proj + down_proj per expert per layer
        // gate_proj: hidden_dim * intermediate_dim
        // up_proj:   hidden_dim * intermediate_dim
        // down_proj: intermediate_dim * hidden_dim
        let expert_params_per_layer =
            num_experts as u64 * 3 * hidden_dim as u64 * intermediate_dim as u64;

        // Router params per layer: hidden_dim * num_experts
        let router_params_per_layer = hidden_dim as u64 * num_experts as u64;

        let total_params_estimate =
            (expert_params_per_layer + router_params_per_layer) * self.num_layers as u64;

        MoeReport {
            num_experts,
            num_layers: self.num_layers,
            load_balance,
            total_params_estimate,
        }
    }
}

#[cfg(test)]
#[path = "moe_construction_tests.rs"]
mod tests;
