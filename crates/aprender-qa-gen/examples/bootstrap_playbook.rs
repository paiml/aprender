//! Example: Bootstrapping Architecture-Aware Playbooks
//!
//! This example demonstrates how to use kernel profiles and the bootstrapper
//! to generate playbook YAML that targets the specific kernel operations
//! exercised by a model architecture.
//!
//! Run with: `cargo run --example bootstrap_playbook -p apr-qa-gen`

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::expect_used)]

use aprender_qa_gen::{
    bootstrap_playbook, profile_from_constraints, to_yaml, ArchConstraints, ArchSizeVariant,
    BootstrapConfig, BootstrappedPlaybook, KernelProfile,
};

fn main() {
    println!("=== Kernel Profile-Driven Playbook Bootstrap ===\n");

    // =========================================================================
    // Step 1: Define architecture constraints (from a family contract)
    // =========================================================================
    println!("--- Step 1: Architecture Constraints ---\n");

    let constraints = ArchConstraints {
        attention_type: Some("gqa".to_string()),
        activation: Some("silu".to_string()),
        norm_type: Some("rmsnorm".to_string()),
        has_bias: Some(true),
        tied_embeddings: Some(false),
        positional_encoding: Some("rope".to_string()),
        mlp_type: Some("swiglu".to_string()),
    };

    println!("Family:     qwen2");
    println!("Attention:  {:?}", constraints.attention_type);
    println!("Activation: {:?}", constraints.activation);
    println!("Norm:       {:?}", constraints.norm_type);
    println!("Positional: {:?}", constraints.positional_encoding);
    println!("MLP:        {:?}", constraints.mlp_type);
    println!();

    // =========================================================================
    // Step 2: Build a kernel profile from constraints
    // =========================================================================
    println!("--- Step 2: Kernel Profile ---\n");

    let profile: KernelProfile = profile_from_constraints("qwen2", &constraints, Some(32_768));

    println!("Kernel operations ({}):", profile.kernel_ops.len());
    for op in &profile.kernel_ops {
        println!("  - {op}");
    }
    println!();

    println!("Prompt categories ({}):", profile.prompt_categories.len());
    for cat in &profile.prompt_categories {
        println!(
            "  {} ({} prompts, oracle: {})",
            cat.name,
            cat.prompts.len(),
            cat.oracle_type
        );
        println!("    Rationale: {}", cat.rationale);
        if let Some(first) = cat.prompts.first() {
            println!("    Example:   {first}");
        }
    }
    println!();

    println!(
        "Long context: {} (max_position_embeddings: 32768)",
        profile.long_context
    );
    println!("Total prompts: {}", profile.prompt_count());
    println!();

    // =========================================================================
    // Step 3: Define size variant
    // =========================================================================
    println!("--- Step 3: Size Variant ---\n");

    let size_variant = ArchSizeVariant {
        parameters: "1.5B".to_string(),
        hidden_dim: 1536,
        num_layers: 28,
        num_heads: Some(12),
        num_kv_heads: Some(2),
        intermediate_dim: Some(8960),
        vocab_size: Some(151_936),
        max_position_embeddings: Some(32_768),
    };

    println!("Parameters: {}", size_variant.parameters);
    println!("Hidden dim: {}", size_variant.hidden_dim);
    println!("Layers:     {}", size_variant.num_layers);
    println!("Heads:      {:?}", size_variant.num_heads);
    println!("KV heads:   {:?}", size_variant.num_kv_heads);
    println!();

    // =========================================================================
    // Step 4: Bootstrap the playbook
    // =========================================================================
    println!("--- Step 4: Bootstrap Playbook ---\n");

    let config = BootstrapConfig {
        family: "qwen2".to_string(),
        size_variant: "1.5b".to_string(),
        hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
        tier: "mvp".to_string(),
        kernel_profile: None, // auto-derived from constraints
    };

    let playbook: BootstrappedPlaybook =
        bootstrap_playbook(&config, &constraints, &size_variant, "small");

    println!("Playbook:    {}", playbook.name);
    println!("Version:     {}", playbook.version);
    println!("HF repo:     {}", playbook.model.hf_repo);
    println!("Size:        {}", playbook.model.size_category);
    println!("Formats:     {:?}", playbook.model.formats);
    println!("Modalities:  {:?}", playbook.test_matrix.modalities);
    println!("Backends:    {:?}", playbook.test_matrix.backends);
    println!("Scenarios:   {}", playbook.test_matrix.scenario_count);
    println!(
        "Prompts:     {} (architecture-targeted)",
        playbook.test_matrix.prompts.len()
    );
    println!("Gates:       {}", playbook.falsification_gates.len());
    println!(
        "Kernel ops:  {}",
        playbook.kernel_profile.kernel_ops.join(", ")
    );
    println!();

    // =========================================================================
    // Step 5: Serialize to YAML
    // =========================================================================
    println!("--- Step 5: Generated YAML ---\n");

    let yaml = to_yaml(&playbook).expect("YAML serialization");
    println!("{yaml}");

    println!("=== Bootstrap Complete ===");
}
