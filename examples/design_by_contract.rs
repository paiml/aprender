//! Design by Contract in Aprender
//!
//! Demonstrates the three pillars of aprender's contract system:
//! 1. Tensor layout contract — shape validation for GGUF→APR conversion
//! 2. Block size constants — quantization format invariants
//! 3. Contract error handling — typed errors for violations
//!
//! Run: `cargo run --example design_by_contract`

use aprender::format::layout_contract::{block_sizes, ContractError, LayoutContract};
use aprender::format::model_family_loader::load_family_yaml;
use std::path::Path;

fn main() {
    println!("=== Aprender Design by Contract ===\n");

    let contract = LayoutContract::new();

    // --- 1. Tensor transpose contract ---
    println!("1. Tensor transpose contract (GGUF→APR):");
    println!(
        "   output.weight should transpose: {}",
        contract.should_transpose_gguf("output.weight")
    );
    println!(
        "   token_embd.weight should transpose: {}",
        contract.should_transpose_gguf("token_embd.weight")
    );
    println!(
        "   blk.0.attn_q.weight should transpose: {}",
        contract.should_transpose_gguf("blk.0.attn_q.weight")
    );

    // --- 2. Critical tensor identification ---
    println!("\n2. Critical tensor identification:");
    println!(
        "   output.weight is critical: {}",
        contract.is_critical_tensor("output.weight")
    );
    println!(
        "   blk.0.attn_norm.weight is critical: {}",
        contract.is_critical_tensor("blk.0.attn_norm.weight")
    );

    // --- 3. APR shape validation (pass) ---
    println!("\n3. APR shape validation:");
    let vocab = 151_936;
    let hidden = 896;
    match contract.validate_apr_shape("lm_head.weight", &[vocab, hidden], vocab, hidden) {
        Ok(()) => println!("   PASS: lm_head.weight [{vocab}, {hidden}]"),
        Err(e) => println!("   FAIL: {e}"),
    }

    // --- 4. APR shape validation (contract violation) ---
    println!("\n4. Shape contract violation:");
    match contract.validate_apr_shape("lm_head.weight", &[hidden, vocab], vocab, hidden) {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(ContractError::ShapeMismatch {
            tensor,
            expected,
            actual,
        }) => {
            println!("   REJECTED: '{tensor}'");
            println!("   Expected: {expected}");
            println!("   Actual:   {actual:?}");
        }
        Err(e) => println!("   Error: {e}"),
    }

    // --- 5. Block size constants ---
    println!("\n5. Quantization block size contracts:");
    println!("   Q4_K: {} bytes/block ({} elements/block)", block_sizes::Q4_K, block_sizes::QK_K);
    println!("   Q5_K: {} bytes/block ({} elements/block)", block_sizes::Q5_K, block_sizes::QK_K);
    println!("   Q6_K: {} bytes/block ({} elements/block)", block_sizes::Q6_K, block_sizes::QK_K);

    // --- 6. Expected byte calculation ---
    println!("\n6. Expected byte calculation for quantized tensors:");
    let out_dim = 4096;
    let in_dim = 4096;
    let q4k_bytes = LayoutContract::calculate_q4k_bytes(out_dim, in_dim);
    let q6k_bytes = LayoutContract::calculate_q6k_bytes(out_dim, in_dim);
    println!("   Q4_K [{out_dim}x{in_dim}]: {q4k_bytes} bytes");
    println!("   Q6_K [{out_dim}x{in_dim}]: {q6k_bytes} bytes");

    // Verify the math: ceil(4096/256) = 16 superblocks per row
    let superblocks = (in_dim + block_sizes::QK_K - 1) / block_sizes::QK_K;
    assert_eq!(superblocks, 16);
    assert_eq!(q4k_bytes, out_dim * superblocks * block_sizes::Q4_K);
    assert_eq!(q6k_bytes, out_dim * superblocks * block_sizes::Q6_K);
    println!("   Verified: {superblocks} superblocks/row, math checks out");

    // --- 7. Model Family Contract: Qwen3.5 Hybrid Attention ---
    println!("\n7. Model Family Contract: Qwen3.5 (GH-278 Hybrid Attention):");
    let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts/model-families");
    let qwen35_path = contracts_dir.join("qwen3_5.yaml");
    if qwen35_path.exists() {
        let family = load_family_yaml(&qwen35_path).expect("load qwen3_5.yaml");
        println!("   Family: {} (vendor: {})", family.display_name, family.vendor);
        println!("   Architectures: {:?}", family.architectures);

        if let Some(variant) = family.size_variants.get("9b") {
            println!("   9B variant:");
            println!("     hidden_dim={}, head_dim={} (explicit, not hidden/heads)",
                variant.hidden_dim, variant.head_dim);
            println!("     num_heads={}, num_kv_heads={} (GQA ratio: {}:1)",
                variant.num_heads, variant.num_kv_heads,
                variant.num_heads / variant.num_kv_heads);
            println!("     vocab_size={}, max_pos_embed={}",
                variant.vocab_size, variant.max_position_embeddings);

            // Contract invariant: hidden_dim == num_heads * head_dim
            let computed = variant.num_heads * variant.head_dim;
            assert_eq!(computed, variant.hidden_dim,
                "Invariant violated: num_heads * head_dim != hidden_dim");
            println!("     Invariant: {}*{} = {} == hidden_dim (PASS)",
                variant.num_heads, variant.head_dim, computed);
        }

        println!("   Constraints: has_bias={}, activation={:?}, mlp={:?}",
            family.constraints.has_bias,
            family.constraints.activation,
            family.constraints.mlp_type);
    } else {
        println!("   SKIP: qwen3_5.yaml not found");
    }

    // --- 8. Fine-Tuning Config Validation (Qwen3.5) ---
    println!("\n8. Fine-Tuning Config Validation (Qwen3.5-9B):");
    let ft_config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
    println!("   hidden_size={}, num_layers={}", ft_config.hidden_size, ft_config.num_hidden_layers);
    println!("   num_heads={}, num_kv_heads={} (GQA {}:1)",
        ft_config.num_attention_heads, ft_config.num_kv_heads,
        ft_config.num_attention_heads / ft_config.num_kv_heads);
    println!("   head_dim={} (4096/16)", ft_config.head_dim());
    println!("   vocab_size={}", ft_config.vocab_size);
    println!("   use_bias={} (KEY: no attention bias)", ft_config.use_bias);

    // Validate against contract
    if qwen35_path.exists() {
        let family = load_family_yaml(&qwen35_path).expect("load qwen3_5.yaml");
        if let Some(variant) = family.size_variants.get("9b") {
            assert_eq!(ft_config.hidden_size, variant.hidden_dim,
                "hidden_size mismatch: config vs contract");
            assert_eq!(ft_config.vocab_size, variant.vocab_size,
                "vocab_size mismatch: config vs contract");
            assert_eq!(ft_config.head_dim(), variant.head_dim,
                "head_dim mismatch: config vs contract");
            assert_eq!(ft_config.num_hidden_layers, variant.num_layers,
                "num_layers mismatch: config vs contract");
            println!("   Contract sync: ALL dimensions match qwen3_5.yaml (PASS)");
        }
    }

    // LoRA target dimensions
    let q_proj_dim = ft_config.num_attention_heads * ft_config.head_dim(); // 16*256 = 4096
    let kv_proj_dim = ft_config.num_kv_heads * ft_config.head_dim();       // 4*256 = 1024
    println!("   LoRA targets: q_proj [{q_proj_dim}, {}], v_proj [{kv_proj_dim}, {}]",
        ft_config.hidden_size, ft_config.hidden_size);

    println!("\n=== All contract demonstrations complete ===");
}
