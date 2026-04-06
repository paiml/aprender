//! Mixture of Experts Construction from Dense Models (GH-445)
//!
//! Demonstrates MoE construction: planning expert assignments, computing
//! gate weights, and measuring load balance.
//!
//! Run: cargo run --example moe_construction

use aprender::online::moe_construction::{
    compute_expert_load_balance, compute_gate_weights, plan_moe_construction, MoeConfig,
    RouterInit, RoutingMethod,
};

fn main() {
    println!("=== MoE Construction from Dense Models (GH-445) ===\n");

    // ── 1. Configuration ──
    println!("── 1. MoE Configuration ──");
    let cfg = MoeConfig {
        num_experts: 8,
        num_experts_per_tok: 2,
        routing_method: RoutingMethod::TopK,
        gate_hidden_dim: None,
    };
    println!("  Experts:     {}", cfg.num_experts);
    println!("  Per-token:   {}", cfg.num_experts_per_tok);
    println!("  Routing:     {:?}", cfg.routing_method);
    assert!(cfg.validate().is_ok());

    // ── 2. Plan Construction ──
    println!("\n── 2. Construction Plan (4 models, 8 layers) ──");
    let plan = plan_moe_construction(4, 8, &cfg).expect("plan_moe_construction");
    println!("  Layers: {}", plan.num_layers);
    println!("  Router init: {:?}", plan.router_init);
    for (i, layer) in plan.assignments.iter().enumerate() {
        let models: Vec<usize> = layer.iter().map(|a| a.source_model).collect();
        println!("  Layer {}: models {:?}", i, models);
    }

    // ── 3. Load Balance ──
    println!("\n── 3. Load Balance ──");
    let balance = compute_expert_load_balance(&plan.assignments);
    println!("  Balance score: {:.4} (0.0 = perfect)", balance);

    // ── 4. Gate Weights ──
    println!("\n── 4. Gate Weight Initialization ──");
    for init in [
        RouterInit::Uniform,
        RouterInit::Balanced,
        RouterInit::Random,
    ] {
        let weights = compute_gate_weights(64, 8, init);
        let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {:?}: {} weights, range [{:.6}, {:.6}]",
            init,
            weights.len(),
            min,
            max
        );
    }

    // ── 5. Report ──
    println!("\n── 5. MoE Report ──");
    let report = plan.report(4096, 11008, 8);
    println!("  Experts/layer:  {}", report.num_experts);
    println!("  Layers:         {}", report.num_layers);
    println!("  Load balance:   {:.4}", report.load_balance);
    println!(
        "  Est. params:    {:.1}B",
        report.total_params_estimate as f64 / 1e9
    );

    // ── 6. Single Model Reuse ──
    println!("\n── 6. Single Model -> 8 Experts ──");
    let single_plan = plan_moe_construction(1, 4, &cfg).expect("plan_moe_construction");
    let single_balance = compute_expert_load_balance(&single_plan.assignments);
    println!("  Balance: {:.4} (trivially balanced)", single_balance);

    println!("\n=== MoE construction verified ===");
}
