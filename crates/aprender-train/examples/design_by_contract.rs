//! Design by Contract examples for entrenar
//!
//! Demonstrates the DbC contracts enforced by entrenar's configuration
//! and evaluation modules. See docs/design-by-contract.md for the full spec.
//!
//! Run with: cargo run --example design_by_contract

use entrenar::eval::{EvalResult, Leaderboard, Metric};
use entrenar::transformer::TransformerConfig;

fn main() {
    println!("=== Design by Contract: TransformerConfig ===\n");
    demonstrate_transformer_config();

    println!("\n=== Design by Contract: Leaderboard N-06 ===\n");
    demonstrate_leaderboard_n06();
}

/// Factory methods produce internally consistent configurations.
/// The postcondition `hidden_size % num_attention_heads == 0` holds
/// for every factory, guaranteeing a valid `head_dim()`.
fn demonstrate_transformer_config() {
    let configs = [
        ("tiny (test)", TransformerConfig::tiny()),
        ("LLaMA-2 7B", TransformerConfig::llama2_7b()),
        ("LLaMA-2 13B", TransformerConfig::llama2_13b()),
        ("Mistral 7B", TransformerConfig::mistral_7b()),
        ("Qwen2 0.5B", TransformerConfig::qwen2_0_5b()),
    ];

    for (name, cfg) in &configs {
        let head_dim: usize = cfg.head_dim();
        // Postcondition: hidden_size must be evenly divisible by num_attention_heads
        assert_eq!(
            cfg.hidden_size % cfg.num_attention_heads,
            0,
            "DbC violation: hidden_size not divisible by num_attention_heads for {name}"
        );
        // Postcondition: GQA ratio must be integral
        assert_eq!(
            cfg.num_attention_heads % cfg.num_kv_heads,
            0,
            "DbC violation: num_attention_heads not divisible by num_kv_heads for {name}"
        );
        println!(
            "{name:>14}: hidden={}, heads={}, kv_heads={}, head_dim={head_dim}, vocab={}",
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_kv_heads, cfg.vocab_size,
        );
    }
}

/// N-06: Models with missing scores sort LAST, not to an arbitrary position.
/// This prevents a model that was never evaluated on a metric from appearing
/// above models that were actually measured.
fn demonstrate_leaderboard_n06() {
    // Higher-is-better metric: Accuracy
    let metric = Metric::Accuracy;
    let mut lb = Leaderboard::new(metric);

    let mut r1 = EvalResult::new("model-A");
    r1.add_score(metric, 0.92);
    lb.add(r1);

    let mut r2 = EvalResult::new("model-B");
    r2.add_score(metric, 0.87);
    lb.add(r2);

    // model-C has NO accuracy score -- N-06 says it must sort last
    let r3 = EvalResult::new("model-C (no score)");
    lb.add(r3);

    println!("Ranked by Accuracy (higher is better):");
    for (i, result) in lb.results.iter().enumerate() {
        let score_str = match result.get_score(metric) {
            Some(s) => format!("{s:.4}"),
            None => "-- missing --".to_string(),
        };
        println!("  #{}: {} = {}", i + 1, result.model_name, score_str);
    }

    // Verify N-06 postcondition
    assert_eq!(
        lb.results.last().map(|r| r.model_name.as_str()),
        Some("model-C (no score)"),
        "N-06 violation: model with missing score did not sort last"
    );
    println!("\nN-06 contract holds: missing-score model sorted last.");
}
