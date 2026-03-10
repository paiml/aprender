//! Per-Layer Merge Granularity (GH-452)
//!
//! Demonstrates fine-grained per-layer merge configuration: YAML parsing,
//! pattern matching, and validation.
//!
//! Run: cargo run --example per_layer_merge

use aprender::online::per_layer_merge::{
    match_layer_rule, parse_merge_yaml, validate_merge_config, LayerMergeConfig, LayerRule,
};

fn main() {
    println!("=== Per-Layer Merge Granularity (GH-452) ===\n");

    // ── 1. Layer Rule Matching ──
    println!("── 1. Layer Rule Matching ──");
    let rules = vec![
        LayerRule {
            layer_pattern: "self_attn".to_string(),
            strategy: "slerp".to_string(),
            weights: Some(vec![0.7, 0.3]),
            scale: None,
        },
        LayerRule {
            layer_pattern: "mlp".to_string(),
            strategy: "average".to_string(),
            weights: None,
            scale: None,
        },
        LayerRule {
            layer_pattern: "embed".to_string(),
            strategy: "passthrough".to_string(),
            weights: None,
            scale: Some(1.0),
        },
    ];

    let tensors = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.embed_tokens.weight",
        "model.layers.5.self_attn.v_proj.weight",
        "model.norm.weight",
    ];

    for tensor in &tensors {
        match match_layer_rule(tensor, &rules) {
            Some(rule) => println!("  {} -> {} ({})", tensor, rule.strategy, rule.layer_pattern),
            None => println!("  {} -> default", tensor),
        }
    }

    // ── 2. YAML Config Parsing ──
    println!("\n── 2. YAML Parsing ──");
    let yaml = r#"
models:
  - path: llama-7b-base.apr
    weight: 0.6
  - path: llama-7b-chat.apr
    weight: 0.4
output: merged-llama.apr
default_strategy: average
layers:
  - layer_pattern: self_attn
    strategy: slerp
    weights: [0.7, 0.3]
  - layer_pattern: mlp
    strategy: ties
    weights: [0.5, 0.5]
    scale: 0.8
"#;

    let config = parse_merge_yaml(yaml).expect("parse_merge_yaml");
    println!("  Models: {}", config.models.len());
    for m in &config.models {
        println!("    {} (weight: {:?})", m.path, m.weight);
    }
    println!("  Output: {}", config.output);
    println!("  Default strategy: {}", config.default_strategy);
    if let Some(ref layers) = config.layers {
        println!("  Layer rules: {}", layers.len());
        for r in layers {
            println!(
                "    pattern='{}' strategy='{}'",
                r.layer_pattern, r.strategy
            );
        }
    }

    // ── 3. Validation ──
    println!("\n── 3. Config Validation ──");
    match validate_merge_config(&config) {
        Ok(()) => println!("  Config is valid"),
        Err(e) => println!("  Validation error: {}", e),
    }

    // ── 4. Invalid Config ──
    println!("\n── 4. Invalid Config Rejection ──");
    let bad_yaml = "models:\n  - path: only-one.apr\noutput: out.apr\ndefault_strategy: average\n";
    let bad_config = parse_merge_yaml(bad_yaml).expect("parse_merge_yaml");
    match validate_merge_config(&bad_config) {
        Ok(()) => println!("  Unexpected: accepted"),
        Err(e) => println!("  Correctly rejected: {}", e),
    }

    // ── 5. Layer Merge Config ──
    println!("\n── 5. LayerMergeConfig ──");
    let layer_cfg = LayerMergeConfig {
        layer_rules: rules,
        default_strategy: "average".to_string(),
        default_weights: vec![0.5, 0.5],
    };
    println!("  Rules: {}", layer_cfg.layer_rules.len());
    println!("  Default: {}", layer_cfg.default_strategy);
    println!("  Default weights: {:?}", layer_cfg.default_weights);

    println!("\n=== Per-layer merge verified ===");
}
