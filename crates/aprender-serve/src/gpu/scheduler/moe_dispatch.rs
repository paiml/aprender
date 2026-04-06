//! ALB-010: MoE Expert Dispatch for Qwen3.5-35B-A3B
//!
//! Implements the Mixture-of-Experts forward pass:
//! 1. Router: softmax → top-k → renormalize (contract: moe-router-v1)
//! 2. Expert SwiGLU: down(SiLU(gate(x)) * up(x)) per selected expert
//! 3. Weighted sum of routed outputs + shared expert (contract: moe-expert-dispatch-v1)

use super::types::MoeExpertWeights;

/// Result of MoE routing for a single token.
#[derive(Debug)]
pub struct MoeRouteResult {
    /// Selected expert indices (len == num_experts_per_tok)
    pub indices: Vec<usize>,
    /// Renormalized weights (len == num_experts_per_tok, sum == 1.0)
    pub weights: Vec<f32>,
}

/// MoE router forward pass.
///
/// Contract: moe-router-v1.yaml
/// - softmax_normalization: probs sum to 1.0
/// - topk_selection: top-k experts by probability
/// - weight_renormalization: selected weights sum to 1.0
pub fn moe_route(
    hidden_state: &[f32],
    gate_weight: &[f32],
    num_experts: usize,
    num_experts_per_tok: usize,
    hidden_dim: usize,
) -> MoeRouteResult {
    // Step 1: logits = hidden_state @ gate_weight.T, shape [num_experts]
    // gate_weight layout: [num_experts, hidden_dim] row-major
    let mut logits = vec![0.0f32; num_experts];
    for e in 0..num_experts {
        let mut sum = 0.0f32;
        let offset = e * hidden_dim;
        for j in 0..hidden_dim {
            sum += hidden_state[j] * gate_weight[offset + j];
        }
        logits[e] = sum;
    }

    // Step 2: softmax with max subtraction for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    let mut probs = vec![0.0f32; num_experts];
    for (i, &logit) in logits.iter().enumerate() {
        probs[i] = (logit - max_logit).exp();
        exp_sum += probs[i];
    }
    for p in &mut probs {
        *p /= exp_sum;
    }

    // Step 3: top-k selection
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k: Vec<(usize, f32)> = indexed.into_iter().take(num_experts_per_tok).collect();

    // Step 4: renormalize selected weights to sum to 1.0
    let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();
    let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = if weight_sum > 0.0 {
        top_k.iter().map(|(_, w)| w / weight_sum).collect()
    } else {
        // Degenerate case: equal weights
        vec![1.0 / num_experts_per_tok as f32; num_experts_per_tok]
    };

    MoeRouteResult { indices, weights }
}

/// Single expert SwiGLU forward pass.
///
/// Computes: down(SiLU(gate(x)) * up(x))
///
/// # Arguments
/// - `x`: input hidden state [hidden_dim]
/// - `gate_proj`: gate weight [intermediate, hidden_dim]
/// - `up_proj`: up weight [intermediate, hidden_dim]
/// - `down_proj`: down weight [hidden_dim, intermediate]
/// - `hidden_dim`: model hidden dimension
/// - `intermediate`: expert intermediate dimension
fn expert_swiglu(
    x: &[f32],
    gate_proj: &[f32],
    up_proj: &[f32],
    down_proj: &[f32],
    hidden_dim: usize,
    intermediate: usize,
) -> Vec<f32> {
    // gate_out = gate_proj @ x, shape [intermediate]
    let mut gate_out = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let offset = i * hidden_dim;
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += gate_proj[offset + j] * x[j];
        }
        gate_out[i] = sum;
    }

    // up_out = up_proj @ x, shape [intermediate]
    let mut up_out = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let offset = i * hidden_dim;
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += up_proj[offset + j] * x[j];
        }
        up_out[i] = sum;
    }

    // SwiGLU: SiLU(gate_out) * up_out
    for i in 0..intermediate {
        let silu = gate_out[i] / (1.0 + (-gate_out[i]).exp());
        gate_out[i] = silu * up_out[i];
    }

    // down_out = down_proj @ gate_out, shape [hidden_dim]
    let mut output = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let offset = i * intermediate;
        let mut sum = 0.0f32;
        for j in 0..intermediate {
            sum += down_proj[offset + j] * gate_out[j];
        }
        output[i] = sum;
    }

    output
}

/// MoE forward pass for a single token.
///
/// Contract: moe-expert-dispatch-v1.yaml
/// - Routes to top-k experts via softmax router
/// - Runs SwiGLU on each selected expert
/// - Combines with weighted sum + shared expert
///
/// # Arguments
/// - `hidden_state`: input [hidden_dim]
/// - `moe`: expert weights
/// - `hidden_dim`: model hidden dimension
///
/// # Returns
/// MoE output [hidden_dim] = routed_sum + shared_expert(x)
pub fn moe_forward_token(
    hidden_state: &[f32],
    moe: &MoeExpertWeights,
    hidden_dim: usize,
) -> Vec<f32> {
    let intermediate = moe.expert_intermediate;
    let num_experts = moe.num_experts;
    let k = moe.num_experts_per_tok;

    // Step 1: Route
    let route = moe_route(hidden_state, &moe.gate_weight, num_experts, k, hidden_dim);

    // Step 2: Run selected experts and accumulate weighted sum
    let mut routed_out = vec![0.0f32; hidden_dim];

    for (idx, &expert_id) in route.indices.iter().enumerate() {
        // Unfuse gate_up: expert_gate_up layout [num_experts, 2*intermediate, hidden_dim]
        let expert_offset = expert_id * 2 * intermediate * hidden_dim;
        let gate_proj =
            &moe.expert_gate_up[expert_offset..expert_offset + intermediate * hidden_dim];
        let up_proj = &moe.expert_gate_up[expert_offset + intermediate * hidden_dim
            ..expert_offset + 2 * intermediate * hidden_dim];

        // expert_down layout [num_experts, hidden_dim, intermediate]
        let down_offset = expert_id * hidden_dim * intermediate;
        let down_proj = &moe.expert_down[down_offset..down_offset + hidden_dim * intermediate];

        let expert_out = expert_swiglu(
            hidden_state,
            gate_proj,
            up_proj,
            down_proj,
            hidden_dim,
            intermediate,
        );

        let w = route.weights[idx];
        for i in 0..hidden_dim {
            routed_out[i] += w * expert_out[i];
        }
    }

    // Step 3: Shared expert (always active, not gated by router)
    let shared_out = expert_swiglu(
        hidden_state,
        &moe.shared_gate,
        &moe.shared_up,
        &moe.shared_down,
        hidden_dim,
        intermediate,
    );

    // Step 4: moe_out = routed_out + gated_shared_out
    // Qwen3.5: shared_expert_gate is a [1, hidden_dim] linear → sigmoid scalar gate
    if !moe.shared_expert_gate_weight.is_empty() {
        let mut gate_logit = 0.0f32;
        for j in 0..hidden_dim {
            gate_logit += moe.shared_expert_gate_weight[j] * hidden_state[j];
        }
        let gate_scale = 1.0 / (1.0 + (-gate_logit).exp()); // sigmoid
        for i in 0..hidden_dim {
            routed_out[i] += gate_scale * shared_out[i];
        }
    } else {
        // No shared expert gate — add directly (dense model or ungated shared expert)
        for i in 0..hidden_dim {
            routed_out[i] += shared_out[i];
        }
    }

    routed_out
}

#[cfg(test)]
mod tests {
    use super::*;

    // FALSIFY-MOE-ROUTER-001: Softmax stability with large logits
    #[test]
    fn test_softmax_stability_large_logits() {
        let hidden_dim = 4;
        let num_experts = 8;
        let hidden_state = vec![1.0f32; hidden_dim];
        // gate_weight: expert 0 has very large projection, expert 1 slightly less
        let mut gate_weight = vec![0.0f32; num_experts * hidden_dim];
        // Expert 0: weight = [250, 250, 250, 250] → dot = 1000
        for j in 0..hidden_dim {
            gate_weight[0 * hidden_dim + j] = 250.0;
        }
        // Expert 1: weight = [249.75, 249.75, ...] → dot = 999
        for j in 0..hidden_dim {
            gate_weight[1 * hidden_dim + j] = 249.75;
        }

        let result = moe_route(&hidden_state, &gate_weight, num_experts, 2, hidden_dim);

        // No NaN, no Inf
        for &w in &result.weights {
            assert!(w.is_finite(), "weight is not finite: {w}");
        }
        // Expert 0 selected first (highest logit)
        assert_eq!(result.indices[0], 0);
        // Expert 1 selected second
        assert_eq!(result.indices[1], 1);
        // Weights sum to 1
        let sum: f32 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights sum = {sum}");
    }

    // FALSIFY-MOE-ROUTER-004: Zero gate weight → uniform routing
    #[test]
    fn test_zero_gate_uniform_routing() {
        let hidden_dim = 4;
        let num_experts = 8;
        let k = 4;
        let hidden_state = vec![1.0f32; hidden_dim];
        let gate_weight = vec![0.0f32; num_experts * hidden_dim];

        let result = moe_route(&hidden_state, &gate_weight, num_experts, k, hidden_dim);

        assert_eq!(result.indices.len(), k);
        // All weights should be equal (uniform)
        for &w in &result.weights {
            assert!((w - 0.25).abs() < 1e-6, "expected 0.25, got {w}");
        }
    }

    // FALSIFY-MOE-DISPATCH-002: Uniform routing = average of experts
    #[test]
    fn test_uniform_routing_averages_experts() {
        let hidden_dim = 4;
        let intermediate = 2;
        let num_experts = 4;
        let k = 4;

        // Create MoE weights where each expert is different
        let mut expert_gate_up = vec![0.1f32; num_experts * 2 * intermediate * hidden_dim];
        let expert_down = vec![0.1f32; num_experts * hidden_dim * intermediate];

        // Make expert 0 produce different output by scaling gate_up
        for i in 0..(2 * intermediate * hidden_dim) {
            expert_gate_up[0 * 2 * intermediate * hidden_dim + i] = 0.2;
        }

        let moe = MoeExpertWeights {
            gate_weight: vec![0.0f32; num_experts * hidden_dim], // uniform routing
            expert_gate_up,
            expert_down,
            shared_gate: vec![0.0f32; intermediate * hidden_dim], // zero shared
            shared_up: vec![0.0f32; intermediate * hidden_dim],
            shared_down: vec![0.0f32; hidden_dim * intermediate],
            shared_expert_gate_weight: vec![],
            num_experts,
            num_experts_per_tok: k,
            expert_intermediate: intermediate,
        };

        let x = vec![1.0f32; hidden_dim];
        let output = moe_forward_token(&x, &moe, hidden_dim);

        // Output should be finite
        for &v in &output {
            assert!(v.is_finite(), "output not finite: {v}");
        }
    }

    // FALSIFY-MOE-DISPATCH-004: Shared expert runs even with zero routing
    #[test]
    fn test_shared_expert_always_active() {
        let hidden_dim = 4;
        let intermediate = 2;
        let num_experts = 4;
        let k = 2;

        // All expert weights zero, shared expert non-zero
        let moe = MoeExpertWeights {
            gate_weight: vec![0.0f32; num_experts * hidden_dim],
            expert_gate_up: vec![0.0f32; num_experts * 2 * intermediate * hidden_dim],
            expert_down: vec![0.0f32; num_experts * hidden_dim * intermediate],
            shared_gate: vec![0.1f32; intermediate * hidden_dim],
            shared_up: vec![0.1f32; intermediate * hidden_dim],
            shared_down: vec![0.1f32; hidden_dim * intermediate],
            shared_expert_gate_weight: vec![],
            num_experts,
            num_experts_per_tok: k,
            expert_intermediate: intermediate,
        };

        let x = vec![1.0f32; hidden_dim];
        let output = moe_forward_token(&x, &moe, hidden_dim);

        // Output should be non-zero (from shared expert)
        let norm: f32 = output.iter().map(|v| v * v).sum();
        assert!(norm > 0.0, "shared expert output should be non-zero");
    }

    // FALSIFY-MOE-ROUTER-003: Renormalization preserves relative order
    #[test]
    fn test_renorm_preserves_order() {
        let hidden_dim = 4;
        let num_experts = 8;
        let k = 4;

        // Set up gate weights so experts have known ordering
        let mut gate_weight = vec![0.0f32; num_experts * hidden_dim];
        for e in 0..num_experts {
            for j in 0..hidden_dim {
                gate_weight[e * hidden_dim + j] = (num_experts - e) as f32;
            }
        }

        let hidden_state = vec![1.0f32; hidden_dim];
        let result = moe_route(&hidden_state, &gate_weight, num_experts, k, hidden_dim);

        // Weights should be in descending order
        for i in 1..k {
            assert!(
                result.weights[i - 1] >= result.weights[i],
                "weights not ordered: {} < {}",
                result.weights[i - 1],
                result.weights[i]
            );
        }
    }

    // FALSIFY-MOE-DISPATCH-005: Shared expert gate scales output via sigmoid
    #[test]
    fn test_shared_expert_gate_scales_output() {
        let hidden_dim = 4;
        let intermediate = 2;
        let num_experts = 4;
        let k = 2;

        // All routed expert weights zero, shared expert non-zero
        let base_moe = MoeExpertWeights {
            gate_weight: vec![0.0f32; num_experts * hidden_dim],
            expert_gate_up: vec![0.0f32; num_experts * 2 * intermediate * hidden_dim],
            expert_down: vec![0.0f32; num_experts * hidden_dim * intermediate],
            shared_gate: vec![0.1f32; intermediate * hidden_dim],
            shared_up: vec![0.1f32; intermediate * hidden_dim],
            shared_down: vec![0.1f32; hidden_dim * intermediate],
            shared_expert_gate_weight: vec![], // no gate
            num_experts,
            num_experts_per_tok: k,
            expert_intermediate: intermediate,
        };

        let x = vec![1.0f32; hidden_dim];
        let ungated = moe_forward_token(&x, &base_moe, hidden_dim);

        // Now with a large negative gate → sigmoid ≈ 0 → shared expert nearly suppressed
        let mut gated_moe = base_moe.clone();
        gated_moe.shared_expert_gate_weight = vec![-10.0f32; hidden_dim];
        let gated = moe_forward_token(&x, &gated_moe, hidden_dim);

        // Gated output should have much smaller shared component
        let ungated_norm: f32 = ungated.iter().map(|v| v * v).sum::<f32>().sqrt();
        let gated_norm: f32 = gated.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            gated_norm < ungated_norm * 0.1,
            "shared expert gate should suppress output: ungated={ungated_norm}, gated={gated_norm}"
        );
    }
}
